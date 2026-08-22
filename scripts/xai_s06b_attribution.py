#!/usr/bin/env python3
"""Scale the S06a-selected attribution methods across members and S01 strata."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import shutil
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
    integrated_gradients,
    periodic_extremal_mask,
)
from itg_nn.xai.attribution_scaled import (
    build_stratification_masks,
    hierarchical_group_bootstrap,
    independent_sign_agreement_null,
    native_scalar_sensitivities,
    signed_consensus,
    validation_stability_correlation,
)
from itg_nn.xai.perturbations import ReferenceBackgrounds, ValidityTag, low_pass
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import CANONICAL_FUNCTION, InvariantMember


FUNCTIONS = ("original_f", CANONICAL_FUNCTION)
METHODS = ("ig_low_pass", "periodic_mask")
GRADIENT_SETS = ("varied", "fixed")
CHANNEL_NAMES = (
    "bmag",
    "gbdrift",
    "cvdrift",
    "gbdrift0_over_shat",
    "gds2",
    "gds21_over_shat",
    "gds22_over_shat_squared",
)
METHOD_SIGNED = {"ig_low_pass": True, "periodic_mask": False}
METHOD_VALIDITY = {
    "ig_low_pass": ValidityTag.OFF_MANIFOLD.value,
    "periodic_mask": ValidityTag.OFF_MANIFOLD.value,
}
METHOD_BASELINE = {
    "ig_low_pass": "input-derived low-pass; deliberately off-manifold diagnostic",
    "periodic_mask": "fixed matched observation; observed endpoint, off-manifold path",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/xai/S06b_attribution.json"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--selected-methods", type=Path)
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
    resolved["mode"] = "pilot" if args.pilot else "production"
    for name in ("device", "seed", "members"):
        value = getattr(args, name)
        if value is not None:
            resolved[name] = value
    if args.rows is not None:
        resolved["panel_varied_rows"] = args.rows
    for value, key in (
        (args.dataset, "dataset"),
        (args.checkpoint, "checkpoint"),
        (args.cohorts, "cohorts"),
        (args.selected_methods, "selected_methods"),
        (args.published_dir, "published_dir"),
    ):
        if value is not None:
            resolved[key] = str(value)
    return resolved


def _csv_text(rows: list[dict[str, Any]]) -> str:
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


def _strings(values: Any, width: int = 96) -> np.ndarray:
    return np.asarray([str(value).encode() for value in values], dtype=f"S{width}")


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def _h5_take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return dataset[unique][inverse]


def _stratified_rows(
    dataset: Path, registered: np.ndarray, count: int, stable_threshold: float
) -> np.ndarray:
    if not 1 <= count <= len(registered):
        raise ValueError("row count is outside the S01 registered panel")
    if count == len(registered):
        return registered.copy()
    data = load_hdf5_rows(dataset, registered, gradient_set="varied", include_targets=True)
    assert data.actual_log_heat_flux is not None
    with h5py.File(dataset, "r") as h5_file:
        classes = _h5_take(h5_file["equilibrium_class"], registered).astype(np.int16)
    stable = (data.actual_log_heat_flux.numpy() <= stable_threshold).astype(np.int8)
    labels = np.asarray([f"{class_value}:{floor}" for class_value, floor in zip(classes, stable)])
    unique, sizes = np.unique(labels, return_counts=True)
    if count < len(unique):
        raise ValueError("row count cannot cover every class-by-stability stratum")
    allocation = np.ones(len(unique), dtype=np.int64)
    remaining = count - len(unique)
    capacity = sizes - 1
    if remaining:
        ideal = remaining * capacity / capacity.sum()
        allocation += np.floor(ideal).astype(np.int64)
        order = np.argsort(-(ideal - np.floor(ideal)), kind="stable")
        allocation[order[: count - int(allocation.sum())]] += 1
    selected: list[int] = []
    for label, quota in zip(unique, allocation):
        candidates = np.flatnonzero(labels == label)
        positions = np.floor((np.arange(quota) + 0.5) * len(candidates) / quota).astype(int)
        selected.extend(candidates[positions].tolist())
    result = np.sort(registered[np.asarray(selected, dtype=np.int64)])
    if len(result) != count:
        raise RuntimeError("stratified selector returned the wrong row count")
    return result


def _metadata(dataset: Path, rows: np.ndarray) -> dict[str, np.ndarray]:
    with h5py.File(dataset, "r") as h5_file:
        return {
            "equilibrium_file": _decode(_h5_take(h5_file["equilibrium_files"], rows)),
            "equilibrium_class": _h5_take(h5_file["equilibrium_class"], rows).astype(np.int16),
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
        panel_groups = set(_decode(_h5_take(h5_file["equilibrium_files"], panel_rows)))
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
        raise RuntimeError("insufficient off-panel equilibrium-unique support rows")
    selected_array = np.asarray(selected, dtype=np.int64)
    data = load_hdf5_rows(dataset, registered[selected_array], gradient_set="varied")
    return data, {
        "equilibrium_class": classes[selected_array],
        "equilibrium_file": groups[selected_array],
    }


def _member_forward(
    member: InvariantMember,
    function_name: str,
    a_over_lt: torch.Tensor,
    a_over_ln: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def drives(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        if count % len(a_over_lt):
            raise ValueError("attribution batch is not a multiple of source rows")
        repeat = count // len(a_over_lt)
        return a_over_lt.repeat(repeat), a_over_ln.repeat(repeat)

    if function_name == "original_f":
        return lambda geometry: member.original(geometry, *drives(len(geometry)))
    if function_name == CANONICAL_FUNCTION:
        return lambda geometry: member.invariant(geometry, *drives(len(geometry)))
    raise ValueError(function_name)


def _full_forward(
    member: InvariantMember, function_name: str
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if function_name == "original_f":
        return member.original
    if function_name == CANONICAL_FUNCTION:
        return member.invariant
    raise ValueError(function_name)


def _estimate(
    forward: Callable[[torch.Tensor], torch.Tensor],
    method: str,
    geometry: torch.Tensor,
    baseline: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
) -> AttributionMap:
    if method == "ig_low_pass":
        return integrated_gradients(
            forward,
            geometry,
            baseline,
            steps=int(config["ig_steps"]),
            backend="auto",
        )
    if method == "periodic_mask":
        return periodic_extremal_mask(
            forward,
            geometry,
            baseline,
            area_fraction=float(config["mask_area_fraction"]),
            steps=int(config["mask_steps"]),
            learning_rate=float(config["mask_learning_rate"]),
            seed=seed,
        )
    raise ValueError(method)


def _run_member_maps(
    member: InvariantMember,
    data: InferenceData,
    matched_baseline: torch.Tensor,
    config: dict[str, Any],
    *,
    functions: tuple[str, ...] = FUNCTIONS,
    methods: tuple[str, ...] = METHODS,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(member.parameters()).device
    count = len(data.row_indices)
    maps = np.empty((len(functions), len(methods), count, 7, 96), dtype=np.float32)
    predictions = np.empty((len(functions), count), dtype=np.float32)
    scalar = np.empty((len(functions), count, 2), dtype=np.float32)
    batch_size = int(config["batch_size"])
    for function_index, function_name in enumerate(functions):
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            geometry = data.geometry[start:stop].to(device)
            drive_lt = data.a_over_lt[start:stop].to(device)
            drive_ln = data.a_over_ln[start:stop].to(device)
            with torch.no_grad():
                predictions[function_index, start:stop] = (
                    _full_forward(member, function_name)(geometry, drive_lt, drive_ln)
                    .detach()
                    .cpu()
                    .numpy()
                )
            scalar[function_index, start:stop] = native_scalar_sensitivities(
                _full_forward(member, function_name),
                geometry,
                drive_lt,
                drive_ln,
                robust_scales=np.asarray(config["scalar_robust_scales"]),
            ).values.astype(np.float32)
            forward = _member_forward(member, function_name, drive_lt, drive_ln)
            for method_index, method in enumerate(methods):
                baseline = (
                    low_pass(geometry, int(config["low_pass_frequency"]))
                    if method == "ig_low_pass"
                    else matched_baseline[start:stop].to(device)
                )
                estimate = _estimate(
                    forward,
                    method,
                    geometry,
                    baseline,
                    config,
                    seed=seed + 1000 * function_index + 100 * method_index + start,
                )
                if estimate.method != config["estimator_backends"][method]:
                    raise RuntimeError("estimator backend changed within one registered run")
                maps[function_index, method_index, start:stop] = (
                    estimate.values.detach().cpu().numpy().transpose(0, 2, 1)
                )
    return maps, predictions, scalar


def _run_predictions_and_scalar(
    member: InvariantMember,
    data: InferenceData,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute cheap native outputs when a verified map cache is resumed."""

    device = next(member.parameters()).device
    count = len(data.row_indices)
    predictions = np.empty((len(FUNCTIONS), count), dtype=np.float32)
    scalar = np.empty((len(FUNCTIONS), count, 2), dtype=np.float32)
    for function_index, function_name in enumerate(FUNCTIONS):
        for start in range(0, count, int(config["batch_size"])):
            stop = min(start + int(config["batch_size"]), count)
            geometry = data.geometry[start:stop].to(device)
            drive_lt = data.a_over_lt[start:stop].to(device)
            drive_ln = data.a_over_ln[start:stop].to(device)
            with torch.no_grad():
                predictions[function_index, start:stop] = (
                    _full_forward(member, function_name)(geometry, drive_lt, drive_ln)
                    .detach()
                    .cpu()
                    .numpy()
                )
            scalar[function_index, start:stop] = native_scalar_sensitivities(
                _full_forward(member, function_name),
                geometry,
                drive_lt,
                drive_ln,
                robust_scales=np.asarray(config["scalar_robust_scales"]),
            ).values.astype(np.float32)
    return predictions, scalar


def _robust_scalar_scales(dataset: Path, reference_rows: np.ndarray) -> list[float]:
    with h5py.File(dataset, "r") as h5_file:
        group = h5_file["varied_gradient_simulations"]
        values = np.column_stack(
            (_h5_take(group["a_over_LT"], reference_rows), _h5_take(group["a_over_Ln"], reference_rows))
        ).astype(np.float64)
    scales = np.subtract(*np.quantile(values, (0.75, 0.25), axis=0)) / 1.349
    if np.any(scales <= 0):
        raise RuntimeError("reference scalar robust scale is nonpositive")
    return scales.tolist()


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_order = np.argsort(np.argsort(np.asarray(left), kind="stable"), kind="stable").astype(float)
    right_order = np.argsort(np.argsort(np.asarray(right), kind="stable"), kind="stable").astype(float)
    left_order -= left_order.mean()
    right_order -= right_order.mean()
    denominator = np.linalg.norm(left_order) * np.linalg.norm(right_order)
    return float(left_order @ right_order / denominator) if denominator else 1.0


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    scale = torch.sqrt(torch.mean(left.square())).clamp_min(torch.finfo(left.dtype).eps)
    return float((torch.sqrt(torch.mean((right - left).square())) / scale).detach().cpu())


def _symmetry_rows(
    members: list[tuple[str, InvariantMember]],
    data: InferenceData,
    matched: torch.Tensor,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    count = min(int(config["symmetry_rows"]), len(data.row_indices))
    shift = int(config["symmetry_shift"])
    device = next(members[0][1].parameters()).device
    geometry = data.geometry[:count].to(device)
    drive_lt = data.a_over_lt[:count].to(device)
    drive_ln = data.a_over_ln[:count].to(device)
    shifted_geometry = torch.roll(geometry, shifts=shift, dims=1)
    rows: list[dict[str, Any]] = []
    for member_index, (member_id, member) in enumerate(members):
        for function_name in FUNCTIONS:
            forward = _member_forward(member, function_name, drive_lt, drive_ln)
            with torch.no_grad():
                prediction = forward(geometry)
                shifted_prediction = forward(shifted_geometry)
            prediction_error = _relative_error(prediction, shifted_prediction)
            for method_index, method in enumerate(METHODS):
                baseline = (
                    low_pass(geometry, int(config["low_pass_frequency"]))
                    if method == "ig_low_pass"
                    else matched[:count].to(device)
                )
                shifted_baseline = torch.roll(baseline, shifts=shift, dims=1)
                reference = _estimate(
                    forward, method, geometry, baseline, config, seed=int(config["seed"]) + member_index
                )
                co_shifted = _estimate(
                    forward,
                    method,
                    shifted_geometry,
                    shifted_baseline,
                    config,
                    seed=int(config["seed"]) + member_index,
                )
                fixed = _estimate(
                    forward,
                    method,
                    shifted_geometry,
                    baseline,
                    config,
                    seed=int(config["seed"]) + member_index,
                )
                if not (reference.method == co_shifted.method == fixed.method):
                    raise RuntimeError("estimator backend changed during symmetry checks")
                expected = torch.roll(reference.values, shifts=shift, dims=1)
                rows.append(
                    {
                        "member_id": member_id,
                        "function": function_name,
                        "method": method,
                        "rows": count,
                        "shift": shift,
                        "prediction_invariance_relative_rms": prediction_error,
                        "co_shifted_equivariance_relative_rms": _relative_error(expected, co_shifted.values),
                        "fixed_baseline_equivariance_relative_rms": _relative_error(expected, fixed.values),
                        "baseline_convention": METHOD_BASELINE[method],
                        "validity_tag": METHOD_VALIDITY[method],
                        "estimator_backend": reference.method,
                    }
                )
    return rows


def _publish(artifacts: list[Path], published_dir: Path) -> None:
    published_dir.mkdir(parents=True, exist_ok=True)
    for source in artifacts:
        shutil.copy2(source, published_dir / source.name)


def run(config: dict[str, Any], args: argparse.Namespace) -> Path:
    set_deterministic_seed(int(config["seed"]))
    probe = torch.ones(1, 2, 1)
    probe_result = integrated_gradients(
        lambda values: values.sum(dim=(1, 2)),
        probe,
        torch.zeros_like(probe),
        steps=2,
        backend="auto",
    )
    config["estimator_backends"] = {
        "ig_low_pass": probe_result.method,
        "periodic_mask": "periodic_extremal_mask",
    }
    repository = Path(__file__).resolve().parents[1]
    dataset = Path(config["dataset"]).resolve()
    checkpoint = Path(config["checkpoint"]).resolve()
    cohorts_path = Path(config["cohorts"]).resolve()
    selected_path = Path(config["selected_methods"]).resolve()
    output_dir = (args.output_dir or Path("output/xai/S06b") / str(config["run_id"])).resolve()
    previous_manifest_path = output_dir / "manifest.json"
    previous_manifest: dict[str, Any] | None = None
    previous_manifest_hash: str | None = None
    if args.resume and previous_manifest_path.is_file():
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        previous_manifest_hash = sha256_file(previous_manifest_path)
        if previous_manifest.get("config", {}).get("step") != "S06b":
            raise RuntimeError("resume manifest is not an S06b run")
        cached_map = output_dir / "attribution_maps.h5"
        expected_hash = previous_manifest.get("output_hashes", {}).get("attribution_maps.h5")
        if not cached_map.is_file() or sha256_file(cached_map) != expected_hash:
            raise RuntimeError("resume map is absent or disagrees with the prior manifest")
        if previous_manifest.get("dataset", {}).get("sha256") != sha256_file(dataset):
            raise RuntimeError("dataset fingerprint changed before resume")
        if previous_manifest.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint):
            raise RuntimeError("checkpoint fingerprint changed before resume")
    artifacts = RunArtifacts(output_dir)
    cohorts = json.loads(cohorts_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected["primary_path_gradient"] != "ig_low_pass" or selected["primary_perturbation"] != "periodic_mask":
        raise RuntimeError("S06a selected-method registry changed; S06b config must be reviewed")
    if selected["perturbation_fallback_used"] is not True:
        raise RuntimeError("S06b expects the documented secondary perturbation fallback")

    registered = np.asarray(cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64)
    headline_rows = _stratified_rows(
        dataset, registered, int(config["panel_varied_rows"]), float(config["stable_threshold_log_Q"])
    )
    sensitivity_rows = _stratified_rows(
        dataset,
        registered,
        min(int(config["sensitivity_rows_per_gradient_set"]), len(headline_rows)),
        float(config["stable_threshold_log_Q"]),
    )
    if not set(sensitivity_rows).issubset(set(headline_rows)):
        # Full production contains every registered row; this can only matter for
        # an overridden pilot count.  Use deterministic positions from headline.
        positions = np.floor(
            (np.arange(len(sensitivity_rows)) + 0.5) * len(headline_rows) / len(sensitivity_rows)
        ).astype(int)
        sensitivity_rows = headline_rows[positions]

    panel_data = {
        gradient_set: load_hdf5_rows(
            dataset, headline_rows, gradient_set=gradient_set, include_targets=True
        )
        for gradient_set in GRADIENT_SETS
    }
    panel_metadata = _metadata(dataset, headline_rows)
    support, support_metadata = _load_support(
        dataset,
        cohorts,
        headline_rows,
        count=int(config["support_reference_rows"]),
        seed=int(config["seed"]) + 41,
    )
    if set(panel_metadata["equilibrium_file"]).intersection(support_metadata["equilibrium_file"]):
        raise RuntimeError("support equilibria overlap the analysis panel")
    backgrounds = ReferenceBackgrounds(
        support.geometry,
        np.column_stack((support.a_over_lt.numpy(), support.a_over_ln.numpy())),
        support_metadata["equilibrium_class"],
        support.row_indices,
    )
    matched = {}
    for gradient_set, data in panel_data.items():
        matched[gradient_set] = backgrounds.matched_observed(
            np.column_stack((data.a_over_lt.numpy(), data.a_over_ln.numpy())),
            panel_metadata["equilibrium_class"],
            source_row_ids=data.row_indices,
        )

    config["scalar_robust_scales"] = _robust_scalar_scales(
        dataset, np.asarray(cohorts["reference_varied"]["row_ids"], dtype=np.int64)
    )
    ensemble = load_ensemble(checkpoint, device=str(config["device"]))
    index_by_id = {member_id: index for index, member_id in enumerate(ensemble.member_ids)}
    top_registry = list(cohorts["member_cohorts"]["stored_validation_top_10"])
    headline_ids = top_registry[: int(config["members"])]
    per_band = int(config["sensitivity_members_per_band"])

    def spread_select(values: list[str], count: int) -> list[str]:
        positions = np.floor((np.arange(count) + 0.5) * len(values) / count).astype(int)
        return [values[position] for position in positions]

    band_11_50 = spread_select(list(cohorts["member_cohorts"]["stored_validation_ranks_11_50"]), per_band)
    band_51_100 = spread_select(list(cohorts["member_cohorts"]["stored_validation_ranks_51_100"]), per_band)
    sensitivity_ids = band_11_50 + band_51_100
    config["headline_member_ids"] = headline_ids
    config["sensitivity_member_ids"] = sensitivity_ids
    members = {
        member_id: InvariantMember(ensemble.models[index_by_id[member_id]])
        for member_id in headline_ids + sensitivity_ids
    }

    headline_maps = np.empty(
        (len(FUNCTIONS), len(METHODS), len(headline_ids), len(GRADIENT_SETS), len(headline_rows), 7, 96),
        dtype=np.float32,
    )
    predictions = np.empty(
        (len(FUNCTIONS), len(headline_ids), len(GRADIENT_SETS), len(headline_rows)), dtype=np.float32
    )
    scalar_values = np.empty(
        (len(FUNCTIONS), len(headline_ids), len(GRADIENT_SETS), len(headline_rows), 2), dtype=np.float32
    )
    if previous_manifest is None:
        for member_index, member_id in enumerate(headline_ids):
            for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
                member_maps, member_predictions, member_scalar = _run_member_maps(
                    members[member_id],
                    panel_data[gradient_set],
                    matched[gradient_set],
                    config,
                    seed=int(config["seed"]) + 10000 * member_index + 1000 * gradient_index,
                )
                headline_maps[:, :, member_index, gradient_index] = member_maps
                predictions[:, member_index, gradient_index] = member_predictions
                scalar_values[:, member_index, gradient_index] = member_scalar
                print(f"headline {member_index + 1}/{len(headline_ids)} {member_id} {gradient_set}", flush=True)

        full_map_path = artifacts.write_hdf5(
            "attribution_maps.h5",
            {
                "attribution": headline_maps,
                "canonical_minus_original": headline_maps[1] - headline_maps[0],
                "function_name": _strings(FUNCTIONS),
                "method_name": _strings(METHODS),
                "estimator_backend": _strings(
                    [config["estimator_backends"][method] for method in METHODS]
                ),
                "member_id": _strings(headline_ids),
                "gradient_set": _strings(GRADIENT_SETS),
                "row_id": headline_rows,
                "signed": np.asarray([METHOD_SIGNED[name] for name in METHODS]),
            },
            axes={
                "attribution": ("function", "method", "member", "gradient_set", "sample", "channel", "z"),
                "canonical_minus_original": ("method", "member", "gradient_set", "sample", "channel", "z"),
                "function_name": ("function",),
                "method_name": ("method",),
                "estimator_backend": ("method",),
                "member_id": ("member",),
                "gradient_set": ("gradient_set",),
                "row_id": ("sample",),
                "signed": ("method",),
            },
            attributes={
                "estimand": "native max(log Q, -2)",
                "member_level_signed_before_aggregation": True,
                "stable_feature_claims_permitted": False,
            },
            compression="gzip",
        )
    else:
        with h5py.File(output_dir / "attribution_maps.h5", "r+") as h5_file:
            np.testing.assert_array_equal(h5_file["row_id"][:], headline_rows)
            if [value.decode() for value in h5_file["member_id"][:]] != headline_ids:
                raise RuntimeError("cached map member IDs disagree with the registered cohort")
            headline_maps[...] = h5_file["attribution"][:]
            if "estimator_backend" not in h5_file:
                backend = h5_file.create_dataset(
                    "estimator_backend",
                    data=_strings(
                        [config["estimator_backends"][method] for method in METHODS]
                    ),
                )
                backend.attrs["axes"] = json.dumps(["method"])
        full_map_path = artifacts.register_existing("attribution_maps.h5")
        for member_index, member_id in enumerate(headline_ids):
            for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
                member_predictions, member_scalar = _run_predictions_and_scalar(
                    members[member_id], panel_data[gradient_set], config
                )
                predictions[:, member_index, gradient_index] = member_predictions
                scalar_values[:, member_index, gradient_index] = member_scalar
        print("resumed verified headline map cache", flush=True)

    stable_threshold = float(config["stable_threshold_log_Q"])
    channel_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    for function_index, function_name in enumerate(FUNCTIONS):
        for method_index, method in enumerate(METHODS):
            for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
                target = panel_data[gradient_set].actual_log_heat_flux
                assert target is not None
                target_values = target.numpy()
                strata = (
                    ("all", np.ones(len(target_values), dtype=bool)),
                    ("stable_or_near_floor", target_values <= stable_threshold),
                    ("unstable", target_values > stable_threshold),
                )
                all_maps = headline_maps[function_index, method_index, :, gradient_index]
                consensus_all = signed_consensus(all_maps)
                other_function = headline_maps[1 - function_index, method_index, :, gradient_index]
                other_importance = np.abs(other_function).mean(axis=(1, 3))
                canonical_original = np.median(
                    [
                        _rank_correlation(consensus_all.member_channel_importance[index], other_importance[index])
                        for index in range(len(headline_ids))
                    ]
                )
                agreement_rows.append(
                    {
                        "function": function_name,
                        "method": method,
                        "gradient_set": gradient_set,
                        "members": len(headline_ids),
                        "samples": len(headline_rows),
                        "median_pairwise_channel_rank_agreement": consensus_all.median_rank_agreement,
                        "mean_cell_sign_agreement": float(consensus_all.sign_agreement.mean()),
                        "median_member_canonical_original_channel_rank_correlation": canonical_original,
                        "signed": METHOD_SIGNED[method],
                        "estimator_backend": config["estimator_backends"][method],
                        "independent_sign_null": independent_sign_agreement_null(len(headline_ids)),
                    }
                )
                for stratum_index, (stratum, mask) in enumerate(strata):
                    maps = all_maps[:, mask]
                    consensus = signed_consensus(maps)
                    for channel, channel_name in enumerate(CHANNEL_NAMES):
                        channel_rows.append(
                            {
                                "function": function_name,
                                "method": method,
                                "gradient_set": gradient_set,
                                "stratum": stratum,
                                "sample_count": int(mask.sum()),
                                "channel": channel,
                                "channel_name": channel_name,
                                "median_signed": float(consensus.median_signed[channel].mean()),
                                "q25_signed": float(consensus.q25_signed[channel].mean()),
                                "q75_signed": float(consensus.q75_signed[channel].mean()),
                                "median_absolute": float(consensus.median_absolute[channel].mean()),
                                "sign_agreement": float(consensus.sign_agreement[channel].mean()),
                                "signed": METHOD_SIGNED[method],
                                "contribution_valued": method == "ig_low_pass",
                                "estimand": "native max(log Q, -2)",
                                "validity_tag": METHOD_VALIDITY[method],
                                "baseline_convention": METHOD_BASELINE[method],
                                "feature_claims_permitted": stratum != "stable_or_near_floor",
                                "estimator_backend": config["estimator_backends"][method],
                            }
                        )
                        values = np.abs(maps[:, :, channel]).mean(axis=2)
                        interval = hierarchical_group_bootstrap(
                            values,
                            panel_metadata["equilibrium_file"][mask],
                            replicates=int(config["bootstrap_replicates"]),
                            seed=int(config["seed"]) + 100000 * function_index + 10000 * method_index + 1000 * gradient_index + 100 * stratum_index + channel,
                        )
                        uncertainty_rows.append(
                            {
                                "function": function_name,
                                "method": method,
                                "gradient_set": gradient_set,
                                "stratum": stratum,
                                "channel": channel,
                                "channel_name": channel_name,
                                "statistic": "mean_absolute_attribution",
                                "estimate": interval.estimate,
                                "ci_lower": interval.lower,
                                "ci_upper": interval.upper,
                                "replicates": len(interval.samples),
                                "member_resampling_unit": interval.resampling_units[0],
                                "sample_resampling_unit": interval.resampling_units[1],
                                "estimator_backend": config["estimator_backends"][method],
                            }
                        )

    # Covariate strata use the canonical primary path.  Error and disagreement
    # are computed per row from member-level native outputs, never pooled first.
    stratified_rows: list[dict[str, Any]] = []
    for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
        data = panel_data[gradient_set]
        target = data.actual_log_heat_flux
        assert target is not None
        absolute_error = np.abs(predictions[1, :, gradient_index] - target.numpy()[None]).mean(axis=0)
        spread = predictions[1, :, gradient_index].std(axis=0)
        masks = build_stratification_masks(
            gradient_set=np.repeat(gradient_set, len(headline_rows)),
            target=target.numpy(),
            a_over_lt=data.a_over_lt.numpy(),
            a_over_ln=data.a_over_ln.numpy(),
            equilibrium_class=panel_metadata["equilibrium_class"],
            stable_threshold=stable_threshold,
            member_absolute_error=absolute_error,
            ensemble_spread=spread,
        )
        for key, mask in masks.items():
            if "|flux=" in key:
                detail = key.split("|", 1)[1]
            elif "|a_over_" in key:
                detail = key.split("|", 1)[1]
            elif "|equilibrium_class=" in key:
                detail = key.split("|", 1)[1]
            elif "|member_absolute_error=" in key or "|ensemble_spread=" in key:
                detail = key.split("|", 1)[1]
            else:
                continue
            if not mask.any():
                continue
            stratifier, value = detail.split("=", 1)
            stable_sample_count = int(
                np.count_nonzero(mask & (target.numpy() <= stable_threshold))
            )
            consensus = signed_consensus(
                headline_maps[1, 0, :, gradient_index][:, mask]
            )
            for channel, channel_name in enumerate(CHANNEL_NAMES):
                stratified_rows.append(
                    {
                        "function": CANONICAL_FUNCTION,
                        "method": "ig_low_pass",
                        "gradient_set": gradient_set,
                        "stratifier": stratifier,
                        "stratum_value": value,
                        "sample_count": int(mask.sum()),
                        "channel": channel,
                        "channel_name": channel_name,
                        "median_signed": float(consensus.median_signed[channel].mean()),
                        "median_absolute": float(consensus.median_absolute[channel].mean()),
                        "sign_agreement": float(consensus.sign_agreement[channel].mean()),
                        "sample_count_stable": stable_sample_count,
                        "feature_claims_permitted": stable_sample_count == 0,
                        "estimand": "native max(log Q, -2)",
                        "validity_tag": METHOD_VALIDITY["ig_low_pass"],
                        "signed": METHOD_SIGNED["ig_low_pass"],
                        "baseline_convention": METHOD_BASELINE["ig_low_pass"],
                        "estimator_backend": config["estimator_backends"]["ig_low_pass"],
                    }
                )

    scalar_rows: list[dict[str, Any]] = []
    for function_index, function_name in enumerate(FUNCTIONS):
        for member_index, member_id in enumerate(headline_ids):
            for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
                for drive_index, drive in enumerate(("a_over_lt", "a_over_ln")):
                    values = scalar_values[function_index, member_index, gradient_index, :, drive_index]
                    scalar_rows.append(
                        {
                            "function": function_name,
                            "member_id": member_id,
                            "gradient_set": gradient_set,
                            "drive": drive,
                            "median_signed": float(np.median(values)),
                            "q25_signed": float(np.quantile(values, 0.25)),
                            "q75_signed": float(np.quantile(values, 0.75)),
                            "median_absolute": float(np.median(np.abs(values))),
                            "signed": True,
                            "scale": "robust_per_scalar_drive",
                            "robust_scale": float(config["scalar_robust_scales"][drive_index]),
                            "estimand": "native max(log Q, -2)",
                        }
                    )

    # Wider-rank sensitivity: the same canonical low-pass map and same rows for
    # every member, so the validation-R2 correlation is not confounded by panel.
    sensitivity_positions = np.asarray([np.flatnonzero(headline_rows == row)[0] for row in sensitivity_rows])
    top_sensitivity_maps = np.take(
        headline_maps[1, 0], sensitivity_positions, axis=2
    )
    top_importance = np.abs(top_sensitivity_maps).mean(axis=(1, 2, 4))
    band_importance = []
    for band_index, member_id in enumerate(sensitivity_ids):
        gradient_maps = []
        for gradient_set in GRADIENT_SETS:
            data = panel_data[gradient_set]
            subset = InferenceData(
                geometry=data.geometry[sensitivity_positions],
                a_over_lt=data.a_over_lt[sensitivity_positions],
                a_over_ln=data.a_over_ln[sensitivity_positions],
                row_indices=sensitivity_rows,
                actual_log_heat_flux=(
                    None if data.actual_log_heat_flux is None else data.actual_log_heat_flux[sensitivity_positions]
                ),
            )
            maps, _, _ = _run_member_maps(
                members[member_id],
                subset,
                matched[gradient_set][sensitivity_positions],
                config,
                functions=(CANONICAL_FUNCTION,),
                methods=("ig_low_pass",),
                seed=int(config["seed"]) + 500000 + 10000 * band_index,
            )
            gradient_maps.append(maps[0, 0])
        band_importance.append(np.abs(np.stack(gradient_maps)).mean(axis=(0, 1, 3)))
        print(f"sensitivity {band_index + 1}/{len(sensitivity_ids)} {member_id}", flush=True)
    all_ids = headline_ids + sensitivity_ids
    all_importance = np.concatenate((top_importance, np.asarray(band_importance)), axis=0)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validation_by_id = {str(member["id"]): float(member["validation_r2"]) for member in bundle["members"]}
    validation = np.asarray([validation_by_id[member_id] for member_id in all_ids])
    stability = validation_stability_correlation(all_importance, validation)
    rank_rows = []
    for index, member_id in enumerate(all_ids):
        cohort = (
            "stored_validation_top_10"
            if member_id in headline_ids
            else "stored_validation_ranks_11_50"
            if member_id in band_11_50
            else "stored_validation_ranks_51_100"
        )
        rank_rows.append(
            {
                "member_id": member_id,
                "member_cohort": cohort,
                "stored_validation_r2": validation[index],
                "canonical_map_rank_agreement": stability.stability[index],
                "correlation_metric": stability.metric,
                "overall_spearman_rho": stability.spearman_rho,
                "method": "ig_low_pass",
                "estimator_backend": config["estimator_backends"]["ig_low_pass"],
                "rows_per_gradient_set": len(sensitivity_rows),
            }
        )

    symmetry_rows = _symmetry_rows(
        [(member_id, members[member_id]) for member_id in headline_ids],
        panel_data["varied"],
        matched["varied"],
        config,
    )

    channel_path = artifacts.write_text("channel_consensus.csv", _csv_text(channel_rows))
    uncertainty_path = artifacts.write_text("hierarchical_uncertainty.csv", _csv_text(uncertainty_rows))
    agreement_path = artifacts.write_text("member_agreement.csv", _csv_text(agreement_rows))
    rank_path = artifacts.write_text("rank_sensitivity.csv", _csv_text(rank_rows))
    scalar_path = artifacts.write_text("scalar_sensitivities.csv", _csv_text(scalar_rows))
    stratified_path = artifacts.write_text("stratified_consensus.csv", _csv_text(stratified_rows))
    symmetry_path = artifacts.write_text("symmetry_checks.csv", _csv_text(symmetry_rows))

    review_count = min(int(config["review_publish_rows_per_gradient_set"]), len(headline_rows))
    review_path = artifacts.write_hdf5(
        "selected_review_maps.h5",
        {
            "attribution": headline_maps[:, :, :, :, :review_count],
            "canonical_minus_original": headline_maps[1, :, :, :, :review_count] - headline_maps[0, :, :, :, :review_count],
            "function_name": _strings(FUNCTIONS),
            "method_name": _strings(METHODS),
            "estimator_backend": _strings(
                [config["estimator_backends"][method] for method in METHODS]
            ),
            "member_id": _strings(headline_ids),
            "gradient_set": _strings(GRADIENT_SETS),
            "row_id": headline_rows[:review_count],
            "signed": np.asarray([METHOD_SIGNED[name] for name in METHODS]),
        },
        axes={
            "attribution": ("function", "method", "member", "gradient_set", "sample", "channel", "z"),
            "canonical_minus_original": ("method", "member", "gradient_set", "sample", "channel", "z"),
            "function_name": ("function",),
            "method_name": ("method",),
            "estimator_backend": ("method",),
            "member_id": ("member",),
            "gradient_set": ("gradient_set",),
            "row_id": ("sample",),
            "signed": ("method",),
        },
        attributes={
            "estimand": "native max(log Q, -2)",
            "stable_feature_claims_permitted": "false",
            "review_slice_mapping": "use load_review_slice_index().slice_rows(parent_row_ids)",
            "research_source": "canonical external HDF5; review slice was not used",
        },
        compression="gzip",
    )

    summary = {
        "step": "S06b",
        "run_id": config["run_id"],
        "mode": config["mode"],
        "estimand": "native max(log Q, -2)",
        "canonical_function": CANONICAL_FUNCTION,
        "original_function": "original_f",
        "selected_primary_path": "ig_low_pass",
        "selected_secondary_perturbation": "periodic_mask",
        "estimator_backends": config["estimator_backends"],
        "perturbation_is_fallback": True,
        "headline_members": len(headline_ids),
        "headline_rows_per_gradient_set": len(headline_rows),
        "gradient_sets_reported_separately": list(GRADIENT_SETS),
        "stable_rows": {
            gradient_set: int(
                (panel_data[gradient_set].actual_log_heat_flux.numpy() <= stable_threshold).sum()
            )
            for gradient_set in GRADIENT_SETS
        },
        "stable_feature_claims_permitted": False,
        "hierarchical_resampling_units": ["members", "equilibrium_files"],
        "validation_r2_vs_map_stability_spearman": stability.spearman_rho,
        "initial_production_wall_time_seconds": config[
            "initial_production_wall_time_seconds"
        ],
        "scalar_robust_scales": config["scalar_robust_scales"],
        "negative_results_carried_from_s06a": [
            "stable/near-floor maps do not support feature-level claims",
            "periodic mask is a secondary fallback and fails fixed-background equivariance",
            "low-pass IG has qualified parameter-randomization response and baseline sensitivity",
        ],
    }
    summary_path = artifacts.write_json("summary.json", summary)

    varied_unstable = panel_data["varied"].actual_log_heat_flux.numpy() > stable_threshold
    atlas = signed_consensus(headline_maps[1, 0, :, 0][:, varied_unstable])
    figure, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    signed_scale = np.max(np.abs(atlas.median_signed))
    image0 = axes[0].imshow(atlas.median_signed, aspect="auto", cmap="coolwarm", vmin=-signed_scale, vmax=signed_scale)
    axes[0].set_title("Canonical low-pass IG: median signed map, varied unstable rows")
    axes[0].set_yticks(range(7), CHANNEL_NAMES)
    figure.colorbar(image0, ax=axes[0], label="native clipped-log contribution")
    image1 = axes[1].imshow(atlas.sign_agreement, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    axes[1].set_title("Cross-member sign agreement")
    axes[1].set_yticks(range(7), CHANNEL_NAMES)
    axes[1].set_xlabel("periodic z index")
    figure.colorbar(image1, ax=axes[1], label="agreement fraction")
    figure.tight_layout()
    plot_file = output_dir / "consensus_atlas.png"
    figure.savefig(plot_file, dpi=170)
    plt.close(figure)
    plot_path = artifacts.register_existing("consensus_atlas.png")

    publish_paths = [
        channel_path,
        uncertainty_path,
        agreement_path,
        rank_path,
        scalar_path,
        review_path,
        stratified_path,
        symmetry_path,
        summary_path,
        plot_path,
    ]
    published_dir = None
    if config["mode"] == "production" and not args.no_publish:
        published_dir = Path(config["published_dir"])
        _publish(publish_paths, published_dir)

    return artifacts.finalize(
        config={
            **config,
            "cohorts_sha256": sha256_file(cohorts_path),
            "selected_methods_sha256": sha256_file(selected_path),
            "script_sha256": sha256_file(__file__),
            "scaled_module_sha256": sha256_file(repository / "itg_nn/xai/attribution_scaled.py"),
            "attribution_module_sha256": sha256_file(repository / "itg_nn/xai/attribution.py"),
            "resume_source_manifest_sha256": previous_manifest_hash,
            "source_production_wall_time_seconds": (
                None
                if previous_manifest is None
                else config["initial_production_wall_time_seconds"]
            ),
        },
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=all_ids,
        row_ids=np.concatenate((headline_rows, headline_rows)).tolist(),
        gradient_set="fixed_and_varied_separate",
        device=config["device"],
        repository=repository,
        command=sys.argv,
        published_dir=published_dir,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(run(_resolve(config, args), args))


if __name__ == "__main__":
    main()
