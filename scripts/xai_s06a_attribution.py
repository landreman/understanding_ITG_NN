#!/usr/bin/env python3
"""Benchmark input-attribution methods for S06a."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from itg_nn.data import InferenceData, load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.attribution import (
    AttributionMap,
    absolute_rank_correlation,
    attribution_equivariance_error,
    attribution_sensitivity,
    attribution_sparsity,
    completeness_residual,
    curve_area,
    cyclic_grouped_occlusion,
    deletion_insertion_curves,
    expected_gradients,
    grouped_bootstrap_mean,
    integrated_gradients,
    native_scaled_gradients,
    periodic_extremal_mask,
    perturbation_infidelity,
    temporal_saliency_rescale,
    toy_recovery,
    vargrad,
)
from itg_nn.xai.perturbations import (
    ReferenceBackgrounds,
    ScaledPCASupport,
    ValidityTag,
    low_pass,
)
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import CANONICAL_FUNCTION, InvariantMember


FUNCTION_NAMES = ("original_f", "invariant_tilde_f")
METHOD_NAMES = (
    "scaled_gradient",
    "ig_robust_constant",
    "ig_matched_observed",
    "ig_medoid",
    "ig_low_pass",
    "expected_gradients",
    "vargrad",
    "cyclic_occlusion",
    "periodic_mask",
    "tsr_scaled_gradient",
    "tsr_ig_robust_constant",
)
METHOD_BASELINES = {
    "scaled_gradient": "none_local",
    "ig_robust_constant": "robust_constant",
    "ig_matched_observed": "matched_observed",
    "ig_medoid": "medoid",
    "ig_low_pass": "low_pass",
    "expected_gradients": "observed_background_distribution",
    "vargrad": "none_local_noise",
    "cyclic_occlusion": "matched_observed",
    "periodic_mask": "matched_observed",
    "tsr_scaled_gradient": "none_local",
    "tsr_ig_robust_constant": "robust_constant",
}
BASELINE_VALIDITY = {
    "none_local": ValidityTag.PLAUSIBLY_LOCAL.value,
    "none_local_noise": ValidityTag.PLAUSIBLY_LOCAL.value,
    "robust_constant": ValidityTag.OFF_MANIFOLD.value,
    "matched_observed": ValidityTag.OBSERVED_COMPARISON.value,
    "medoid": ValidityTag.OBSERVED_COMPARISON.value,
    "low_pass": ValidityTag.OFF_MANIFOLD.value,
    "observed_background_distribution": ValidityTag.OBSERVED_COMPARISON.value,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xai/S06a_attribution.json")
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--channel-scales", type=Path)
    parser.add_argument("--baseline-registry", type=Path)
    parser.add_argument("--published-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--members", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    return parser


def _resolve(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if args.pilot:
        resolved.update(config["pilot"])
        # Preserve the historical development gate. Production was corrected
        # after review to avoid pooled denominator cancellation; rewriting the
        # pilot selection would erase the instability that it exposed.
        resolved["selection_rule"]["faithfulness_strata"] = list(
            config["pilot"]["faithfulness_strata"]
        )
    resolved["mode"] = "pilot" if args.pilot else "production"
    for name in ("device", "seed", "members"):
        value = getattr(args, name)
        if value is not None:
            resolved[name] = value
    if args.rows is not None:
        resolved["panel_varied_rows"] = args.rows
    for argument, key in (
        (args.dataset, "dataset"),
        (args.checkpoint, "checkpoint"),
        (args.cohorts, "cohorts"),
        (args.channel_scales, "channel_scales"),
        (args.baseline_registry, "baseline_registry"),
        (args.published_dir, "published_dir"),
    ):
        if argument is not None:
            resolved[key] = str(argument)
    return resolved


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def _h5_take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return dataset[unique][inverse]


def _strings(values: list[str] | tuple[str, ...] | np.ndarray, width: int = 96) -> np.ndarray:
    return np.asarray([str(value).encode() for value in values], dtype=f"S{width}")


def _stratified_panel_rows(
    dataset: Path,
    registered: np.ndarray,
    count: int,
    stable_threshold: float,
) -> np.ndarray:
    if not 1 <= count <= len(registered):
        raise ValueError("panel row cap is outside the frozen S01 panel")
    if count == len(registered):
        return registered.copy()
    data = load_hdf5_rows(dataset, registered, gradient_set="varied", include_targets=True)
    if data.actual_log_heat_flux is None:
        raise RuntimeError("registered varied targets were not loaded")
    with h5py.File(dataset, "r") as h5_file:
        classes = _h5_take(h5_file["equilibrium_class"], registered).astype(np.int16)
    stable = (data.actual_log_heat_flux.numpy() <= stable_threshold).astype(np.int8)
    labels = np.asarray([f"{class_value}:{floor}" for class_value, floor in zip(classes, stable)])
    unique, sizes = np.unique(labels, return_counts=True)
    if count < len(unique):
        raise ValueError("pilot count cannot cover all class-by-stability strata")
    allocation = np.ones(len(unique), dtype=np.int64)
    remaining = count - len(unique)
    capacity = sizes - 1
    if remaining:
        ideal = remaining * capacity / capacity.sum()
        allocation += np.floor(ideal).astype(np.int64)
        leftover = count - int(allocation.sum())
        order = np.argsort(-(ideal - np.floor(ideal)), kind="stable")
        allocation[order[:leftover]] += 1
    selected: list[int] = []
    for label, quota in zip(unique, allocation):
        candidates = np.flatnonzero(labels == label)
        positions = np.floor(
            (np.arange(quota, dtype=np.float64) + 0.5) * len(candidates) / quota
        ).astype(np.int64)
        selected.extend(candidates[positions].tolist())
    rows = np.sort(registered[np.asarray(selected, dtype=np.int64)])
    if len(rows) != count:
        raise RuntimeError("stratified panel selection returned the wrong count")
    return rows


def _load_panel(
    dataset: Path, cohorts: dict[str, Any], config: dict[str, Any]
) -> tuple[InferenceData, dict[str, np.ndarray]]:
    registered = np.asarray(
        cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64
    )
    rows = _stratified_panel_rows(
        dataset,
        registered,
        int(config["panel_varied_rows"]),
        float(config["stable_threshold_log_Q"]),
    )
    panel = load_hdf5_rows(dataset, rows, gradient_set="varied", include_targets=True)
    if panel.actual_log_heat_flux is None:
        raise RuntimeError("panel targets were not loaded")
    with h5py.File(dataset, "r") as h5_file:
        equilibrium_class = _h5_take(h5_file["equilibrium_class"], rows).astype(np.int16)
        equilibrium_file = _decode(_h5_take(h5_file["equilibrium_files"], rows))
    stable = panel.actual_log_heat_flux.numpy() <= float(
        config["stable_threshold_log_Q"]
    )
    if not stable.any() or stable.all():
        raise RuntimeError("selected panel must retain both floor and unstable strata")
    return panel, {
        "equilibrium_class": equilibrium_class,
        "equilibrium_file": equilibrium_file,
        "stable_or_near_floor": stable,
    }


def _load_support(
    dataset: Path,
    cohorts: dict[str, Any],
    panel_rows: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[InferenceData, dict[str, np.ndarray]]:
    registered = np.asarray(cohorts["reference_varied"]["row_ids"], dtype=np.int64)
    with h5py.File(dataset, "r") as h5_file:
        groups = _decode(_h5_take(h5_file["equilibrium_files"], registered))
        classes = _h5_take(h5_file["equilibrium_class"], registered).astype(np.int16)
        panel_groups = set(
            _decode(_h5_take(h5_file["equilibrium_files"], panel_rows)).tolist()
        )
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    seen: set[str] = set()
    for index in rng.permutation(len(registered)):
        group = str(groups[index])
        if group not in panel_groups and group not in seen:
            selected.append(int(index))
            seen.add(group)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("not enough equilibrium-unique off-panel support rows")
    selected_array = np.asarray(selected, dtype=np.int64)
    rows = registered[selected_array]
    data = load_hdf5_rows(dataset, rows, gradient_set="varied", include_targets=False)
    return data, {
        "equilibrium_class": classes[selected_array],
        "equilibrium_file": groups[selected_array],
    }


def _channel_scales(path: Path) -> torch.Tensor:
    rows: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["channel"]), float(row["robust_sigma_iqr"])))
    rows.sort()
    if [index for index, _ in rows] != list(range(7)):
        raise ValueError("S01 robust-scale artifact must contain channels 0 through 6")
    values = torch.as_tensor([value for _, value in rows], dtype=torch.float32)
    if not torch.isfinite(values).all() or torch.any(values <= 0):
        raise ValueError("S01 robust scales must be finite and positive")
    return values


def _baselines(
    panel: InferenceData,
    metadata: dict[str, np.ndarray],
    support: InferenceData,
    support_metadata: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    gradients = np.column_stack((support.a_over_lt.numpy(), support.a_over_ln.numpy()))
    backgrounds = ReferenceBackgrounds(
        support.geometry,
        gradients,
        support_metadata["equilibrium_class"],
        support.row_indices,
    )
    query_gradients = np.column_stack((panel.a_over_lt.numpy(), panel.a_over_ln.numpy()))
    constant = backgrounds.constant().expand_as(panel.geometry).clone()
    matched = backgrounds.matched_observed(
        query_gradients, metadata["equilibrium_class"], source_row_ids=panel.row_indices
    )
    medoid = backgrounds.medoid().expand_as(panel.geometry).clone()
    smoothed = low_pass(panel.geometry, int(config["low_pass_frequency"]))
    expected_count = int(config["expected_backgrounds"])
    if not 1 <= expected_count <= len(support.geometry):
        raise ValueError("expected-background count is outside the support cohort")
    return {
        "robust_constant": constant,
        "matched_observed": matched,
        "medoid": medoid,
        "low_pass": smoothed,
        "expected_pool": support.geometry[:expected_count].clone(),
    }


def _subset_baselines(
    baselines: dict[str, torch.Tensor], count: int, *, shift: int = 0
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, values in baselines.items():
        selected = values if name == "expected_pool" else values[:count]
        output[name] = torch.roll(selected, shifts=shift, dims=1) if shift else selected
    return output


def _forward(
    member: InvariantMember,
    function_name: str,
    a_over_lt: torch.Tensor,
    a_over_ln: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def drives(sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_count = len(a_over_lt)
        if sample_count % base_count:
            raise ValueError("attribution backend batch is not a multiple of panel rows")
        repetitions = sample_count // base_count
        return a_over_lt.repeat(repetitions), a_over_ln.repeat(repetitions)

    if function_name == "original_f":
        return lambda geometry: member.original(geometry, *drives(len(geometry)))
    if function_name == "invariant_tilde_f":
        return lambda geometry: member.invariant(geometry, *drives(len(geometry)))
    raise ValueError(f"unknown function {function_name}")


def _run_method(
    name: str,
    forward: Callable[[torch.Tensor], torch.Tensor],
    geometry: torch.Tensor,
    baselines: dict[str, torch.Tensor],
    scales: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
) -> AttributionMap:
    if name == "scaled_gradient":
        return native_scaled_gradients(forward, geometry, scales)
    if name.startswith("ig_"):
        baseline_name = name.removeprefix("ig_")
        return integrated_gradients(
            forward,
            geometry,
            baselines[baseline_name],
            steps=int(config["ig_steps"]),
            backend="auto",
        )
    if name == "expected_gradients":
        return expected_gradients(
            forward,
            geometry,
            baselines["expected_pool"],
            samples=int(config["expected_samples"]),
            seed=seed,
            backend="auto",
        )
    if name == "vargrad":
        return vargrad(
            forward,
            geometry,
            robust_scales=scales,
            samples=int(config["vargrad_samples"]),
            noise_fraction=float(config["vargrad_noise_fraction"]),
            seed=seed,
        )
    if name == "cyclic_occlusion":
        return cyclic_grouped_occlusion(
            forward,
            geometry,
            baselines["matched_observed"],
            window=int(config["occlusion_window"]),
            stride=int(config["occlusion_stride"]),
        )
    if name == "periodic_mask":
        return periodic_extremal_mask(
            forward,
            geometry,
            baselines["matched_observed"],
            area_fraction=float(config["mask_area_fraction"]),
            steps=int(config["mask_steps"]),
            learning_rate=float(config["mask_learning_rate"]),
            seed=seed,
        )
    if name == "tsr_scaled_gradient":
        return temporal_saliency_rescale(
            native_scaled_gradients(forward, geometry, scales)
        )
    if name == "tsr_ig_robust_constant":
        return temporal_saliency_rescale(
            integrated_gradients(
                forward,
                geometry,
                baselines["robust_constant"],
                steps=int(config["ig_steps"]),
                backend="auto",
            )
        )
    raise ValueError(f"unknown attribution method {name}")


def _method_baseline(name: str, baselines: dict[str, torch.Tensor]) -> torch.Tensor:
    baseline_name = METHOD_BASELINES[name]
    if baseline_name in ("none_local", "none_local_noise"):
        # Local estimators have no path baseline. Their faithfulness diagnostic
        # still needs a replacement endpoint, and S03 forbids silently using
        # the all-zero geometry, so use the registered matched observation.
        return baselines["matched_observed"]
    if baseline_name == "observed_background_distribution":
        return baselines["expected_pool"].mean(dim=0, keepdim=True).expand_as(
            baselines["matched_observed"]
        )
    return baselines[baseline_name]


def _toy_benchmark(config: dict[str, Any], scales: torch.Tensor) -> dict[str, dict[str, float]]:
    row_count = int(config["toy_rows"])
    generator = torch.Generator().manual_seed(int(config["seed"]) + 701)
    geometry = 0.05 * torch.randn((row_count, 96, 7), generator=generator)
    positions = (94, 95, 0, 1)
    geometry[:, positions, 2] += torch.linspace(1.0, 2.0, row_count).reshape(-1, 1)
    baseline = torch.zeros_like(geometry)

    def toy_forward(values: torch.Tensor) -> torch.Tensor:
        return values[:, positions, 2].square().mean(dim=1)

    toy_baselines = {
        "robust_constant": baseline,
        "matched_observed": baseline,
        "medoid": baseline,
        "low_pass": baseline,
        "expected_pool": torch.zeros((2, 96, 7)),
    }
    results: dict[str, dict[str, float]] = {}
    for index, method in enumerate(METHOD_NAMES):
        attribution = _run_method(
            method,
            toy_forward,
            geometry,
            toy_baselines,
            torch.ones_like(scales),
            config,
            seed=int(config["seed"]) + 719 + index,
        )
        recovery = toy_recovery(
            attribution.values,
            relevant_channels=(2,),
            relevant_positions=positions,
        )
        curves = deletion_insertion_curves(
            toy_forward,
            geometry,
            baseline,
            attribution.values,
            fractions=config["deletion_fractions"],
            robust_scales=torch.ones_like(scales),
            seed=int(config["seed"]) + 733,
        )
        results[method] = {
            **recovery,
            "deletion_auc": curve_area(curves, "deletion_output"),
            "random_deletion_auc": curve_area(curves, "random_deletion_output"),
            "insertion_auc": curve_area(curves, "insertion_output"),
            "random_insertion_auc": curve_area(curves, "random_insertion_output"),
        }
    random_map = torch.rand(
        geometry.shape, generator=torch.Generator().manual_seed(int(config["seed"]) + 739)
    )
    recovery = toy_recovery(
        random_map, relevant_channels=(2,), relevant_positions=positions
    )
    random_curves = deletion_insertion_curves(
        toy_forward,
        geometry,
        baseline,
        random_map,
        fractions=config["deletion_fractions"],
        robust_scales=torch.ones_like(scales),
        seed=int(config["seed"]) + 743,
    )
    results["random_map_control"] = {
        **recovery,
        "deletion_auc": curve_area(random_curves, "deletion_output"),
        "random_deletion_auc": curve_area(
            random_curves, "random_deletion_output"
        ),
        "insertion_auc": curve_area(random_curves, "insertion_output"),
        "random_insertion_auc": curve_area(
            random_curves, "random_insertion_output"
        ),
    }
    return results


def _randomized_member(member: InvariantMember, seed: int) -> InvariantMember:
    randomized = copy.deepcopy(member)
    devices = [next(randomized.parameters()).device.index] if next(randomized.parameters()).is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for module in randomized.model.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
    randomized.eval()
    return randomized


def _select_methods(
    rows: list[dict[str, Any]],
    toy: dict[str, dict[str, float]],
    rule: dict[str, Any],
) -> dict[str, Any]:
    by_method = {
        row["method"]: row
        for row in rows
        if row["function"] == "invariant_tilde_f" and row["stratum"] == "all"
    }
    by_method_stratum = {
        (row["method"], row["stratum"]): row
        for row in rows
        if row["function"] == "invariant_tilde_f"
    }
    faithfulness_strata = tuple(rule["faithfulness_strata"])

    def eligible(method: str) -> bool:
        metrics = toy[method]
        row = by_method[method]
        return bool(
            metrics["channel_top1"] >= float(rule["minimum_toy_channel_top1"])
            and metrics["position_average_precision"]
            >= float(rule["minimum_toy_position_average_precision"])
            and all(
                float(by_method_stratum[(method, stratum)]["deletion_margin_vs_random"])
                > float(rule["minimum_deletion_margin"])
                and float(
                    by_method_stratum[(method, stratum)]["insertion_margin_vs_random"]
                )
                > float(rule["minimum_insertion_margin"])
                for stratum in faithfulness_strata
            )
            and float(row["parameter_randomization_correlation"])
            < float(rule["maximum_parameter_randomization_correlation"])
        )

    def choose(candidates: list[str]) -> str | None:
        kept = [method for method in candidates if eligible(method)]
        if not kept:
            return None
        return min(
            kept,
            key=lambda method: (
                float(by_method[method]["normalized_infidelity"])
                if np.isfinite(float(by_method[method]["normalized_infidelity"]))
                else float("inf"),
                float(by_method[method]["runtime_seconds"]),
                method,
            ),
        )

    primary_path = choose(list(rule["path_candidates"]))
    primary_perturbation = choose(list(rule["perturbation_candidates"]))
    return {
        "primary_path_gradient": primary_path,
        "primary_perturbation": primary_perturbation,
        "rule": rule,
        "eligible": {method: eligible(method) for method in METHOD_NAMES},
        "passed": primary_path is not None and primary_perturbation is not None,
    }


def _curve_margin_components(
    curves: list[dict[str, Any]], sample_rows: np.ndarray, direction: str
) -> dict[str, float]:
    aggregated: list[dict[str, float]] = []
    output_keys = (
        "deletion_output",
        "insertion_output",
        "random_deletion_output",
        "random_insertion_output",
    )
    for curve in curves:
        aggregated.append(
            {
                "fraction": float(curve["fraction"]),
                "original_output": float(
                    np.mean(curve["original_output_per_sample"][sample_rows])
                ),
                "baseline_output": float(
                    np.mean(curve["baseline_output_per_sample"][sample_rows])
                ),
                **{
                    key: float(np.mean(curve[f"{key}_per_sample"][sample_rows]))
                    for key in output_keys
                },
            }
        )
    if direction == "deletion":
        positive_key = "random_deletion_output"
        negative_key = "deletion_output"
    elif direction == "insertion":
        positive_key = "insertion_output"
        negative_key = "random_insertion_output"
    else:
        raise ValueError(f"unknown faithfulness direction {direction}")
    fractions = np.asarray([row["fraction"] for row in aggregated], dtype=np.float64)
    native_difference = np.asarray(
        [row[positive_key] - row[negative_key] for row in aggregated],
        dtype=np.float64,
    )
    native_gap = float(
        np.sum(
            0.5
            * (native_difference[:-1] + native_difference[1:])
            * np.diff(fractions)
        )
    )
    denominator = aggregated[0]["original_output"] - aggregated[0]["baseline_output"]
    per_sample_difference = np.stack(
        [
            np.asarray(curve[f"{positive_key}_per_sample"])[sample_rows]
            - np.asarray(curve[f"{negative_key}_per_sample"])[sample_rows]
            for curve in curves
        ]
    )
    per_sample_native_gap = np.sum(
        0.5
        * (per_sample_difference[:-1] + per_sample_difference[1:])
        * np.diff(fractions).reshape(-1, 1),
        axis=0,
    )
    per_sample_denominator = (
        np.asarray(curves[0]["original_output_per_sample"])[sample_rows]
        - np.asarray(curves[0]["baseline_output_per_sample"])[sample_rows]
    )
    per_sample_orientation = np.where(per_sample_denominator >= 0, 1.0, -1.0)
    per_sample_oriented_gap = per_sample_orientation * per_sample_native_gap
    normalized_margin = (
        native_gap / denominator
        if abs(denominator) > np.finfo(np.float64).eps
        else float("nan")
    )
    return {
        "normalized_margin": float(normalized_margin),
        "native_gap": native_gap,
        "denominator": float(denominator),
        "per_row_oriented_native_gap": float(np.mean(per_sample_oriented_gap)),
        "row_favouring_fraction": float(np.mean(per_sample_oriented_gap > 0)),
    }


def _curve_margin(
    curves: list[dict[str, Any]], sample_rows: np.ndarray, direction: str
) -> float:
    return _curve_margin_components(curves, sample_rows, direction)["normalized_margin"]


def _grouped_curve_margin_interval(
    curves: list[dict[str, Any]],
    equilibrium_files: np.ndarray,
    sample_rows: np.ndarray,
    *,
    direction: str,
    replicates: int,
    seed: int,
    control_curves: list[dict[str, Any]] | None = None,
) -> dict[str, float | int | str]:
    groups = np.asarray(equilibrium_files)
    selected_groups = groups[sample_rows]
    unique_groups = np.unique(selected_groups)
    group_rows = [
        sample_rows[np.flatnonzero(selected_groups == group)] for group in unique_groups
    ]
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    native_gaps = np.empty(int(replicates), dtype=np.float64)
    denominators = np.empty(int(replicates), dtype=np.float64)
    per_row_oriented_gaps = np.empty(int(replicates), dtype=np.float64)
    control_gaps = np.empty(int(replicates), dtype=np.float64)
    method_minus_control = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        chosen = rng.integers(0, len(group_rows), size=len(group_rows))
        resampled_rows = np.concatenate([group_rows[index] for index in chosen])
        components = _curve_margin_components(curves, resampled_rows, direction)
        samples[replicate] = components["normalized_margin"]
        native_gaps[replicate] = components["native_gap"]
        denominators[replicate] = components["denominator"]
        per_row_oriented_gaps[replicate] = components[
            "per_row_oriented_native_gap"
        ]
        if control_curves is not None:
            control = _curve_margin_components(
                control_curves, resampled_rows, direction
            )
            control_gaps[replicate] = control["per_row_oriented_native_gap"]
            method_minus_control[replicate] = (
                components["per_row_oriented_native_gap"]
                - control["per_row_oriented_native_gap"]
            )
    lower, upper = np.quantile(samples, (0.025, 0.975))
    native_lower, native_upper = np.quantile(native_gaps, (0.025, 0.975))
    denominator_lower, denominator_upper = np.quantile(denominators, (0.025, 0.975))
    estimate = _curve_margin_components(curves, sample_rows, direction)
    orientation = 1.0 if estimate["denominator"] >= 0 else -1.0
    oriented_native_gaps = orientation * native_gaps
    oriented_native_lower, oriented_native_upper = np.quantile(
        oriented_native_gaps, (0.025, 0.975)
    )
    per_row_oriented_lower, per_row_oriented_upper = np.quantile(
        per_row_oriented_gaps, (0.025, 0.975)
    )
    result: dict[str, float | int | str] = {
        "estimate": estimate["normalized_margin"],
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "native_gap_estimate": estimate["native_gap"],
        "native_gap_ci_lower": float(native_lower),
        "native_gap_ci_upper": float(native_upper),
        "oriented_native_gap_estimate": orientation * estimate["native_gap"],
        "oriented_native_gap_ci_lower": float(oriented_native_lower),
        "oriented_native_gap_ci_upper": float(oriented_native_upper),
        "per_row_oriented_native_gap_estimate": estimate[
            "per_row_oriented_native_gap"
        ],
        "per_row_oriented_native_gap_ci_lower": float(per_row_oriented_lower),
        "per_row_oriented_native_gap_ci_upper": float(per_row_oriented_upper),
        "row_favouring_fraction_estimate": estimate["row_favouring_fraction"],
        "denominator_estimate": estimate["denominator"],
        "denominator_ci_lower": float(denominator_lower),
        "denominator_ci_upper": float(denominator_upper),
        "denominator_negative_fraction": float(np.mean(denominators < 0)),
        "denominator_abs_below_0_005_fraction": float(
            np.mean(np.abs(denominators) < 0.005)
        ),
        "replicates": int(replicates),
        "resampling_unit": "equilibrium_files",
    }
    if control_curves is not None:
        control_estimate = _curve_margin_components(
            control_curves, sample_rows, direction
        )
        control_lower, control_upper = np.quantile(control_gaps, (0.025, 0.975))
        paired_lower, paired_upper = np.quantile(
            method_minus_control, (0.025, 0.975)
        )
        result.update(
            {
                "control_map": "absolute_input_minus_baseline",
                "control_map_normalized_margin_estimate": control_estimate[
                    "normalized_margin"
                ],
                "control_map_per_row_oriented_native_gap_estimate": control_estimate[
                    "per_row_oriented_native_gap"
                ],
                "control_map_per_row_oriented_native_gap_ci_lower": float(
                    control_lower
                ),
                "control_map_per_row_oriented_native_gap_ci_upper": float(
                    control_upper
                ),
                "control_map_row_favouring_fraction_estimate": control_estimate[
                    "row_favouring_fraction"
                ],
                "method_minus_control_map_gap_estimate": estimate[
                    "per_row_oriented_native_gap"
                ]
                - control_estimate["per_row_oriented_native_gap"],
                "method_minus_control_map_gap_ci_lower": float(paired_lower),
                "method_minus_control_map_gap_ci_upper": float(paired_upper),
            }
        )
    return result


def _attribution_equivariance_pair(
    fixed_attributor: Any,
    co_shifted_attributor: Any,
    inputs: torch.Tensor,
    *,
    shift: int,
) -> tuple[float, float]:
    co_shifted = attribution_equivariance_error(
        fixed_attributor,
        inputs,
        shift=shift,
        shifted_attributor=co_shifted_attributor,
    )
    fixed = attribution_equivariance_error(
        fixed_attributor,
        inputs,
        shift=shift,
    )
    return co_shifted, fixed


def _plot_benchmark(rows: list[dict[str, Any]], selection: dict[str, Any], path: Path) -> None:
    selected_rows = [
        row
        for row in rows
        if row["function"] == "invariant_tilde_f" and row["stratum"] == "all"
    ]
    labels = [str(row["method"]).replace("_", "\n") for row in selected_rows]
    margins = [
        min(
            float(candidate["deletion_margin_vs_random"])
            for candidate in rows
            if candidate["function"] == "invariant_tilde_f"
            and candidate["method"] == row["method"]
            and candidate["stratum"] in selection["rule"]["faithfulness_strata"]
        )
        for row in selected_rows
    ]
    randomization = [1 - float(row["parameter_randomization_correlation"]) for row in selected_rows]
    colors = [
        "#2474b5"
        if row["method"]
        in (selection["primary_path_gradient"], selection["primary_perturbation"])
        else "#9ebbd1"
        for row in selected_rows
    ]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(np.arange(len(labels)), margins, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("minimum stratum-specific deletion margin")
    axes[0].set_title("S06a canonical stratum-aware attribution benchmark")
    axes[1].bar(np.arange(len(labels)), randomization, color=colors)
    axes[1].set_ylabel("1 − randomized-map rank correlation")
    axes[1].set_xticks(np.arange(len(labels)), labels, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def run(config: dict[str, Any], args: argparse.Namespace) -> Path:
    set_deterministic_seed(int(config["seed"]))
    repository = Path(__file__).resolve().parents[1]
    dataset = Path(config["dataset"]).resolve()
    checkpoint = Path(config["checkpoint"]).resolve()
    cohorts_path = Path(config["cohorts"]).resolve()
    channel_scales_path = Path(config["channel_scales"]).resolve()
    baseline_registry_path = Path(config["baseline_registry"]).resolve()
    output_dir = (
        args.output_dir or Path("output/xai/S06a") / str(config["run_id"])
    ).resolve()
    manifest_path = output_dir / "manifest.json"
    if args.resume and manifest_path.is_file():
        return manifest_path
    artifacts = RunArtifacts(output_dir)
    cohorts = json.loads(cohorts_path.read_text(encoding="utf-8"))
    baseline_registry = json.loads(baseline_registry_path.read_text(encoding="utf-8"))
    if baseline_registry.get("all_zero_default_forbidden") is not True:
        raise RuntimeError("S03 baseline registry no longer forbids the zero default")
    panel, metadata = _load_panel(dataset, cohorts, config)
    support, support_metadata = _load_support(
        dataset,
        cohorts,
        panel.row_indices,
        count=int(config["support_reference_rows"]),
        seed=int(config["seed"]) + 41,
    )
    if set(metadata["equilibrium_file"]).intersection(support_metadata["equilibrium_file"]):
        raise RuntimeError("support equilibria overlap the analysis panel")
    split = int(len(support.geometry) * float(config["support_fit_fraction"]))
    support_model = ScaledPCASupport.fit(
        support.geometry[:split].numpy(),
        support.geometry[split:].numpy(),
        components=int(config["support_components"]),
    )
    baselines = _baselines(panel, metadata, support, support_metadata, config)
    scales = _channel_scales(channel_scales_path)
    device = torch.device(str(config["device"]))
    geometry = panel.geometry.to(device)
    drives_lt = panel.a_over_lt.to(device)
    drives_ln = panel.a_over_ln.to(device)
    baselines = {name: values.to(device) for name, values in baselines.items()}
    scales = scales.to(device)

    ensemble = load_ensemble(checkpoint, device=device)
    top_ids = tuple(cohorts["member_cohorts"]["stored_validation_top_10"])
    member_count = int(config["members"])
    if member_count != 1:
        raise ValueError("S06a benchmarks exactly the registered top member")
    member_id = top_ids[0]
    index_by_id = {value: index for index, value in enumerate(ensemble.member_ids)}
    member = InvariantMember(ensemble.models[index_by_id[member_id]])
    randomized_member = _randomized_member(member, int(config["seed"]) + 811)

    toy = _toy_benchmark(config, scales.cpu())
    if any(toy[method]["channel_top1"] < 1 for method in METHOD_NAMES):
        raise RuntimeError(f"analytic toy channel gate failed: {toy}")

    maps: dict[tuple[str, str], AttributionMap] = {}
    for function_index, function_name in enumerate(FUNCTION_NAMES):
        forward = _forward(member, function_name, drives_lt, drives_ln)
        for method_index, method in enumerate(METHOD_NAMES):
            result = _run_method(
                method,
                forward,
                geometry,
                baselines,
                scales,
                config,
                seed=int(config["seed"]) + 1009 * function_index + method_index,
            )
            maps[(function_name, method)] = result
            print(
                f"map {function_name} {method} {result.runtime_seconds:.2f}s",
                flush=True,
            )

    map_array = np.stack(
        [
            np.stack([maps[(function, method)].values.cpu().numpy() for method in METHOD_NAMES])
            for function in FUNCTION_NAMES
        ]
    ).astype(np.float32)
    # Stored axes put channel before z, matching the repository artifact contract.
    map_array = np.transpose(map_array, (0, 1, 2, 4, 3))[:, :, None]
    map_difference = map_array[1] - map_array[0]
    map_path = artifacts.write_hdf5(
        "attribution_maps.h5",
        {
            "attribution": map_array,
            "canonical_minus_original": map_difference,
            "function_name": _strings(FUNCTION_NAMES),
            "method_name": _strings(METHOD_NAMES),
            "member_id": _strings((member_id,)),
            "row_id": panel.row_indices,
            "signed": np.asarray([maps[(FUNCTION_NAMES[0], method)].signed for method in METHOD_NAMES]),
            "validity_tag": _strings(
                tuple(maps[(FUNCTION_NAMES[0], method)].validity.value for method in METHOD_NAMES),
                width=64,
            ),
        },
        axes={
            "attribution": ("function", "method", "member", "sample", "channel", "z"),
            "canonical_minus_original": ("method", "member", "sample", "channel", "z"),
            "function_name": ("function",),
            "method_name": ("method",),
            "member_id": ("member",),
            "row_id": ("sample",),
            "signed": ("method",),
            "validity_tag": ("method",),
        },
        attributes={
            "estimand": "native max(log Q, -2)",
            "canonical_function": CANONICAL_FUNCTION,
            "gradient_set": "varied",
            "member_level_primary": True,
            "absolute_summaries_derived_only": True,
        },
        compression="gzip",
    )

    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for function_index, function_name in enumerate(FUNCTION_NAMES):
        forward = _forward(member, function_name, drives_lt, drives_ln)
        robust_ig = maps[(function_name, "ig_robust_constant")].values
        for method_index, method in enumerate(METHOD_NAMES):
            result = maps[(function_name, method)]
            baseline = _method_baseline(method, baselines)
            all_curves = deletion_insertion_curves(
                forward,
                geometry,
                baseline,
                result.values,
                fractions=config["deletion_fractions"],
                robust_scales=scales,
                seed=int(config["seed"]) + 1201 + method_index,
                support_scorer=support_model.score,
            )
            for stratum, mask in (
                ("all", np.ones(len(geometry), dtype=bool)),
                ("stable_or_near_floor", metadata["stable_or_near_floor"]),
                ("unstable", ~metadata["stable_or_near_floor"]),
            ):
                indices = np.flatnonzero(mask)
                sub_geometry = geometry[indices]
                sub_baseline = baseline[indices]
                sub_map = result.values[indices]
                sub_forward = _forward(
                    member,
                    function_name,
                    drives_lt[indices],
                    drives_ln[indices],
                )
                curves = all_curves if stratum == "all" else deletion_insertion_curves(
                    sub_forward,
                    sub_geometry,
                    sub_baseline,
                    sub_map,
                    fractions=config["deletion_fractions"],
                    robust_scales=scales,
                    seed=int(config["seed"]) + 1201 + method_index,
                    support_scorer=support_model.score,
                )
                for row in curves:
                    curve_rows.append(
                        {
                            "function": function_name,
                            "method": method,
                            "stratum": stratum,
                            "row_count": len(indices),
                            **row,
                        }
                    )
                deletion_auc = curve_area(curves, "deletion_output")
                random_deletion_auc = curve_area(curves, "random_deletion_output")
                insertion_auc = curve_area(curves, "insertion_output")
                random_insertion_auc = curve_area(curves, "random_insertion_output")
                row: dict[str, Any] = {
                    "function": function_name,
                    "method": method,
                    "artifact_method": result.method,
                    "batch_layout_adapter": result.metadata.get(
                        "batch_layout_adapter", "none"
                    ),
                    "deterministic_optimizer": result.metadata.get(
                        "deterministic_optimizer", "not_applicable"
                    ),
                    "stratum": stratum,
                    "row_count": len(indices),
                    "signed": result.signed,
                    "contribution_valued": bool(
                        result.metadata.get("contribution_valued", False)
                    ),
                    "validity_tag": result.validity.value,
                    "baseline_family": METHOD_BASELINES[method],
                    "baseline_validity_tag": BASELINE_VALIDITY[METHOD_BASELINES[method]],
                    "registered_baseline_convention": (
                        "co_shifted_input_derived"
                        if method == "ig_low_pass"
                        else "fixed_matched_observed"
                        if method == "periodic_mask"
                        else "fixed"
                        if METHOD_BASELINES[method]
                        not in ("none_local", "none_local_noise")
                        else "not_applicable"
                    ),
                    "faithfulness_replacement": "matched_observed"
                    if METHOD_BASELINES[method] in ("none_local", "none_local_noise")
                    else METHOD_BASELINES[method],
                    "runtime_seconds": result.runtime_seconds,
                    "sparsity_fraction_for_90pct_mass": attribution_sparsity(sub_map),
                    "deletion_auc": deletion_auc,
                    "random_deletion_auc": random_deletion_auc,
                    "deletion_margin_vs_random": random_deletion_auc - deletion_auc,
                    "insertion_auc": insertion_auc,
                    "random_insertion_auc": random_insertion_auc,
                    "insertion_margin_vs_random": insertion_auc - random_insertion_auc,
                    "toy_channel_top1": toy[method]["channel_top1"],
                    "toy_position_average_precision": toy[method]["position_average_precision"],
                    "toy_deletion_margin_vs_random": toy[method]["random_deletion_auc"]
                    - toy[method]["deletion_auc"],
                }
                if stratum == "all":
                    infidelity_count = min(int(config["infidelity_rows"]), len(geometry))
                    row["normalized_infidelity"] = (
                        perturbation_infidelity(
                            _forward(
                                member,
                                function_name,
                                drives_lt[:infidelity_count],
                                drives_ln[:infidelity_count],
                            ),
                            geometry[:infidelity_count],
                            baseline[:infidelity_count],
                            result.values[:infidelity_count],
                            trials=int(config["infidelity_trials"]),
                            removal_fraction=float(
                                config["infidelity_removal_fraction"]
                            ),
                            seed=int(config["seed"]) + 1301 + method_index,
                        )
                        if result.metadata.get("contribution_valued", False)
                        else float("nan")
                    )
                    symmetry_count = min(int(config["symmetry_rows"]), len(geometry))
                    symmetry_baselines = _subset_baselines(baselines, symmetry_count)
                    shifted_baselines = _subset_baselines(
                        baselines,
                        symmetry_count,
                        shift=int(config["symmetry_shift"]),
                    )
                    symmetry_geometry = geometry[:symmetry_count]
                    symmetry_forward = _forward(
                        member,
                        function_name,
                        drives_lt[:symmetry_count],
                        drives_ln[:symmetry_count],
                    )

                    def fixed_baseline_attributor(values: torch.Tensor) -> AttributionMap:
                        return _run_method(
                            method,
                            symmetry_forward,
                            values,
                            symmetry_baselines,
                            scales,
                            config,
                            seed=int(config["seed"]) + 1409 + method_index,
                        )

                    def co_shifted_baseline_attributor(
                        values: torch.Tensor,
                    ) -> AttributionMap:
                        return _run_method(
                            method,
                            symmetry_forward,
                            values,
                            shifted_baselines,
                            scales,
                            config,
                            seed=int(config["seed"]) + 1409 + method_index,
                        )

                    co_shifted_error, fixed_error = _attribution_equivariance_pair(
                        fixed_baseline_attributor,
                        co_shifted_baseline_attributor,
                        symmetry_geometry,
                        shift=int(config["symmetry_shift"]),
                    )
                    row["cyclic_equivariance_relative_rms"] = co_shifted_error
                    row["cyclic_equivariance_baseline_convention"] = "co_shifted"
                    row[
                        "cyclic_equivariance_fixed_baseline_relative_rms"
                    ] = fixed_error
                    random_count = min(int(config["randomization_rows"]), len(geometry))
                    random_baselines = _subset_baselines(baselines, random_count)
                    random_map = _run_method(
                        method,
                        _forward(
                            randomized_member,
                            function_name,
                            drives_lt[:random_count],
                            drives_ln[:random_count],
                        ),
                        geometry[:random_count],
                        random_baselines,
                        scales,
                        config,
                        seed=int(config["seed"]) + 1511 + method_index,
                    ).values
                    row["parameter_randomization_correlation"] = absolute_rank_correlation(
                        result.values[:random_count], random_map
                    )
                    row["trained_map_input_baseline_abs_rank_correlation"] = float(
                        "nan"
                    )
                    row[
                        "randomized_map_input_baseline_abs_rank_correlation"
                    ] = float("nan")
                    if method == "ig_low_pass":
                        input_baseline_factor = (
                            geometry[:random_count] - baseline[:random_count]
                        )
                        row[
                            "trained_map_input_baseline_abs_rank_correlation"
                        ] = absolute_rank_correlation(
                            input_baseline_factor, result.values[:random_count]
                        )
                        row[
                            "randomized_map_input_baseline_abs_rank_correlation"
                        ] = absolute_rank_correlation(
                            input_baseline_factor, random_map
                        )
                    sensitivity_count = min(int(config["sensitivity_rows"]), len(geometry))
                    sensitivity_geometry = geometry[:sensitivity_count]
                    sensitivity_baselines = _subset_baselines(
                        baselines, sensitivity_count
                    )
                    sensitivity_forward = _forward(
                        member,
                        function_name,
                        drives_lt[:sensitivity_count],
                        drives_ln[:sensitivity_count],
                    )
                    row["relative_local_sensitivity"] = attribution_sensitivity(
                        lambda values: _run_method(
                            method,
                            sensitivity_forward,
                            values,
                            sensitivity_baselines,
                            scales,
                            config,
                            seed=int(config["seed"]) + 1601 + method_index,
                        ),
                        sensitivity_geometry,
                        robust_scales=scales,
                        trials=int(config["sensitivity_trials"]),
                        noise_fraction=float(config["sensitivity_noise_fraction"]),
                        seed=int(config["seed"]) + 1613 + method_index,
                    )
                    if method.startswith("ig_") or method == "tsr_ig_robust_constant":
                        residual = completeness_residual(
                            forward, geometry, baseline, result.values
                        ).abs()
                        row["completeness_median_abs"] = float(
                            residual.median().cpu()
                        )
                        row["completeness_q90_abs"] = float(
                            torch.quantile(residual, 0.9).cpu()
                        )
                        row["completeness_max_abs"] = float(residual.max().cpu())
                    else:
                        row["completeness_median_abs"] = float("nan")
                        row["completeness_q90_abs"] = float("nan")
                        row["completeness_max_abs"] = float("nan")
                    row["baseline_rank_correlation_vs_robust_constant"] = (
                        absolute_rank_correlation(result.values, robust_ig)
                        if method.startswith("ig_")
                        else float("nan")
                    )
                    row["canonical_original_rank_correlation"] = absolute_rank_correlation(
                        result.values,
                        maps[(FUNCTION_NAMES[1 - function_index], method)].values,
                    )
                metric_rows.append(row)

                sample_statistic = sub_map.abs().mean(dim=(1, 2)).cpu().numpy()
                grouped = grouped_bootstrap_mean(
                    sample_statistic,
                    metadata["equilibrium_file"][indices],
                    replicates=int(config["bootstrap_replicates"]),
                    seed=int(config["seed"]) + 1709 + method_index,
                )
                bootstrap_rows.append(
                    {
                        "function": function_name,
                        "method": method,
                        "stratum": stratum,
                        "statistic": "sample_mean_absolute_attribution",
                        "estimate": grouped.estimate,
                        "ci_lower": grouped.lower,
                        "ci_upper": grouped.upper,
                        "replicates": len(grouped.samples),
                        "resampling_unit": grouped.resampling_unit,
                    }
                )

    convergence_rows: list[dict[str, Any]] = []
    convergence_count = min(int(config["convergence_rows"]), len(geometry))
    for function_name in FUNCTION_NAMES:
        forward = _forward(
            member,
            function_name,
            drives_lt[:convergence_count],
            drives_ln[:convergence_count],
        )
        subset = _subset_baselines(baselines, convergence_count)
        for method in ("ig_robust_constant", "ig_matched_observed", "ig_medoid", "ig_low_pass"):
            full = _run_method(
                method,
                forward,
                geometry[:convergence_count],
                subset,
                scales,
                config,
                seed=int(config["seed"]) + 1801,
            ).values
            half_config = dict(config)
            half_config["ig_steps"] = max(4, int(config["ig_steps"]) // 2)
            half = _run_method(
                method,
                forward,
                geometry[:convergence_count],
                subset,
                scales,
                half_config,
                seed=int(config["seed"]) + 1801,
            ).values
            convergence_rows.append(
                {
                    "function": function_name,
                    "method": method,
                    "full_steps": int(config["ig_steps"]),
                    "half_steps": int(half_config["ig_steps"]),
                    "absolute_rank_correlation": absolute_rank_correlation(full, half),
                    "mean_absolute_difference": float((full - half).abs().mean().cpu()),
                }
            )

    selection = _select_methods(metric_rows, toy, config["selection_rule"])
    if not selection["passed"]:
        raise RuntimeError(f"no primary method pair passed the registered rule: {selection}")

    selected_names = (
        str(selection["primary_path_gradient"]),
        str(selection["primary_perturbation"]),
    )
    for selected_index, method in enumerate(selected_names):
        method_index = METHOD_NAMES.index(method)
        result = maps[(CANONICAL_FUNCTION, method)]
        baseline = _method_baseline(method, baselines)
        for stratum_index, (stratum, mask) in enumerate(
            (
                ("all", np.ones(len(geometry), dtype=bool)),
                ("stable_or_near_floor", metadata["stable_or_near_floor"]),
                ("unstable", ~metadata["stable_or_near_floor"]),
            )
        ):
            selected_rows = np.flatnonzero(mask)
            sample_rows = np.arange(len(selected_rows))
            sample_curves = deletion_insertion_curves(
                _forward(
                    member,
                    CANONICAL_FUNCTION,
                    drives_lt[selected_rows],
                    drives_ln[selected_rows],
                ),
                geometry[selected_rows],
                baseline[selected_rows],
                result.values[selected_rows],
                fractions=config["deletion_fractions"],
                robust_scales=scales,
                seed=int(config["seed"]) + 1201 + method_index,
                include_per_sample=True,
            )
            control_curves = deletion_insertion_curves(
                _forward(
                    member,
                    CANONICAL_FUNCTION,
                    drives_lt[selected_rows],
                    drives_ln[selected_rows],
                ),
                geometry[selected_rows],
                baseline[selected_rows],
                torch.abs(geometry[selected_rows] - baseline[selected_rows]),
                fractions=config["deletion_fractions"],
                robust_scales=scales,
                seed=int(config["seed"]) + 1201 + method_index,
                include_per_sample=True,
            )
            for direction_index, direction in enumerate(("deletion", "insertion")):
                interval = _grouped_curve_margin_interval(
                    sample_curves,
                    metadata["equilibrium_file"][selected_rows],
                    sample_rows,
                    direction=direction,
                    replicates=int(config["bootstrap_replicates"]),
                    seed=(
                        int(config["seed"])
                        + 1901
                        + 100 * selected_index
                        + 10 * stratum_index
                        + direction_index
                    ),
                    control_curves=control_curves,
                )
                metric_row = next(
                    row
                    for row in metric_rows
                    if row["function"] == CANONICAL_FUNCTION
                    and row["method"] == method
                    and row["stratum"] == stratum
                )
                expected = metric_row[f"{direction}_margin_vs_random"]
                if not np.isclose(float(interval["estimate"]), float(expected), atol=1e-9):
                    raise RuntimeError(
                        f"bootstrap point estimate disagrees for {method} {stratum} {direction}"
                    )
                metric_row["control_map"] = interval["control_map"]
                metric_row[
                    f"{direction}_control_map_margin_vs_random"
                ] = interval["control_map_normalized_margin_estimate"]
                metric_row[f"{direction}_margin_vs_control_map"] = float(
                    expected
                ) - float(interval["control_map_normalized_margin_estimate"])
                bootstrap_rows.append(
                    {
                        "function": CANONICAL_FUNCTION,
                        "method": method,
                        "stratum": stratum,
                        "statistic": f"{direction}_margin_vs_random",
                        **interval,
                    }
                )

    metrics_path = artifacts.write_text("benchmark_metrics.csv", _csv_text(metric_rows))
    curves_path = artifacts.write_text("faithfulness_curves.csv", _csv_text(curve_rows))
    bootstrap_path = artifacts.write_text(
        "grouped_uncertainty.csv", _csv_text(bootstrap_rows)
    )
    convergence_path = artifacts.write_text(
        "ig_convergence.csv", _csv_text(convergence_rows)
    )
    toy_path = artifacts.write_json("toy_controls.json", toy)
    selection_path = artifacts.write_json("selected_methods.json", selection)
    summary = {
        "step": "S06a",
        "mode": config["mode"],
        "run_id": config["run_id"],
        "estimand": "native max(log Q, -2)",
        "canonical_function": CANONICAL_FUNCTION,
        "member_id": member_id,
        "row_count": len(panel.row_indices),
        "stable_or_near_floor_rows": int(metadata["stable_or_near_floor"].sum()),
        "unstable_rows": int((~metadata["stable_or_near_floor"]).sum()),
        "selection": selection,
        "method_count": len(METHOD_NAMES),
        "function_count": len(FUNCTION_NAMES),
        "signed_methods": [method for method in METHOD_NAMES if maps[(FUNCTION_NAMES[0], method)].signed],
        "magnitude_only_methods": [method for method in METHOD_NAMES if not maps[(FUNCTION_NAMES[0], method)].signed],
        "baseline_families": METHOD_BASELINES,
        "baseline_validity_tags": BASELINE_VALIDITY,
        "support_fit_rows": split,
        "support_calibration_rows": len(support.geometry) - split,
        "panel_support_equilibrium_overlap": 0,
    }
    summary_path = artifacts.write_json("summary.json", summary)
    plot_file = output_dir / "benchmark.png"
    _plot_benchmark(metric_rows, selection, plot_file)
    plot_path = artifacts.register_existing("benchmark.png")

    publish_count = min(int(config["review_publish_rows"]), len(panel.row_indices))
    selected_indices = [METHOD_NAMES.index(name) for name in selected_names]
    review_map = map_array[:, selected_indices, :, :publish_count]
    review_difference = map_difference[selected_indices, :, :publish_count]
    review_path = artifacts.write_hdf5(
        "selected_review_maps.h5",
        {
            "attribution": review_map,
            "canonical_minus_original": review_difference,
            "function_name": _strings(FUNCTION_NAMES),
            "method_name": _strings(selected_names),
            "member_id": _strings((member_id,)),
            "row_id": panel.row_indices[:publish_count],
        },
        axes={
            "attribution": ("function", "method", "member", "sample", "channel", "z"),
            "canonical_minus_original": ("method", "member", "sample", "channel", "z"),
            "function_name": ("function",),
            "method_name": ("method",),
            "member_id": ("member",),
            "row_id": ("sample",),
        },
        attributes={
            "estimand": "native max(log Q, -2)",
            "review_slice_mapping": "use load_review_slice_index().slice_rows()",
            "research_source": "canonical external HDF5; review slice was not used",
        },
        compression="gzip",
    )

    published_dir = None
    if config["mode"] == "production" and not args.no_publish:
        published_dir = Path(config["published_dir"])
        published_dir.mkdir(parents=True, exist_ok=True)
        for source in (
            metrics_path,
            curves_path,
            bootstrap_path,
            convergence_path,
            toy_path,
            selection_path,
            summary_path,
            plot_path,
            review_path,
        ):
            target = published_dir / source.name
            target.write_bytes(source.read_bytes())

    return artifacts.finalize(
        config={
            **config,
            "cohorts_sha256": sha256_file(cohorts_path),
            "channel_scales_sha256": sha256_file(channel_scales_path),
            "baseline_registry_sha256": sha256_file(baseline_registry_path),
            "script_sha256": sha256_file(__file__),
            "attribution_module_sha256": sha256_file(
                repository / "itg_nn/xai/attribution.py"
            ),
        },
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=(member_id,),
        row_ids=panel.row_indices.tolist(),
        gradient_set="varied",
        device=device,
        repository=repository,
        command=sys.argv,
        published_dir=published_dir,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = run(_resolve(config, args), args)
    print(manifest)


if __name__ == "__main__":
    main()
