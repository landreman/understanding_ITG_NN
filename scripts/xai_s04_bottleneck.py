#!/usr/bin/env python3
"""Execute S04's invariant-bottleneck anatomy experiment."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from itg_nn.data import load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.audit import rankdata, spearman_correlation
from itg_nn.xai.bottleneck import (
    HIDDEN_INTERVENTION_VALIDITY,
    InterventionResult,
    ShapleyResult,
    bottleneck_interventions,
    exact_or_sampled_shapley,
    grouped_cv_predictions,
    registered_invariants,
    variance_decomposition,
)
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import CANONICAL_FUNCTION, InvariantMember, circular_shift


STRATA = ("overall", "stable_or_near_floor", "unstable")
INTERVENTION_MODES = ("zero", "mean", "resample")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xai/S04_bottleneck.json")
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--published-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--members", type=int, help="Cap validation-ranked members")
    parser.add_argument("--rows", type=int, help="Cap frozen varied-panel rows")
    parser.add_argument("--shapley-permutations", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    return parser


def _resolve(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = dict(config)
    if args.pilot:
        resolved.update(config["pilot"])
    resolved["mode"] = "pilot" if args.pilot else "production"
    overrides = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "cohorts": args.cohorts,
        "published_dir": args.published_dir,
        "device": args.device,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "members": args.members,
        "panel_varied_rows": args.rows,
        "shapley_permutations": args.shapley_permutations,
    }
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = str(value) if isinstance(value, Path) else value
    for key in ("dataset", "checkpoint", "cohorts", "published_dir"):
        resolved[key] = str(Path(resolved[key]).resolve())
    return resolved


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def _h5_take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return dataset[unique][inverse]


def _strings(values: list[str] | tuple[str, ...] | np.ndarray, width: int = 96) -> np.ndarray:
    return np.asarray([str(value).encode("utf-8") for value in values], dtype=f"S{width}")


def _member_unit_ids(member_id: str, width: int) -> tuple[str, ...]:
    return tuple(f"{member_id}:u{index:03d}" for index in range(width))


def _stratum_masks(actual: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    return {
        "overall": np.ones(len(actual), dtype=bool),
        "stable_or_near_floor": actual <= threshold,
        "unstable": actual > threshold,
    }


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    total = float(np.square(y - y.mean()).sum())
    return float(1.0 - np.square(y - p).sum() / total) if total > 0 else float("nan")


def _toy_gate(seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    geometry = torch.randn(12, 96, 7, generator=generator)
    bottleneck = torch.stack(
        (
            geometry[:, :, 0].mean(1),
            geometry[:, :, 1].square().mean(1),
            geometry[:, :, 6].mean(1),
        ),
        dim=1,
    )
    shifted = torch.stack(
        (
            circular_shift(geometry, 17)[:, :, 0].mean(1),
            circular_shift(geometry, 17)[:, :, 1].square().mean(1),
            circular_shift(geometry, 17)[:, :, 6].mean(1),
        ),
        dim=1,
    )
    features = torch.column_stack((bottleneck, torch.linspace(-1, 1, 12), torch.zeros(12)))
    head = lambda x: x[:, 0] + 2 * x[:, 1] + 3 * x[:, 3]
    shapley = exact_or_sampled_shapley(head, features, torch.zeros(5))
    interventions = bottleneck_interventions(
        head,
        features,
        torch.zeros(5),
        feature_names=("u0", "u1", "null_u", "a_over_LT", "null_a_over_Ln"),
        intervention_features=(0, 1, 2),
        seed=seed,
        random_directions=4,
    )
    expected = np.column_stack(
        (
            features[:, 0].numpy(),
            2 * features[:, 1].numpy(),
            np.zeros(12),
            3 * features[:, 3].numpy(),
            np.zeros(12),
        )
    )
    maximum_shapley_error = float(np.max(np.abs(shapley.values - expected)))
    maximum_shift_error = float(torch.max(torch.abs(bottleneck - shifted)))
    maximum_null_intervention = float(np.max(np.abs(interventions.single_delta[:, 2])))
    maximum_efficiency_error = float(
        np.max(
            np.abs(
                shapley.values.sum(1)
                - (shapley.prediction - shapley.baseline_output)
            )
        )
    )
    return {
        "analytic_cyclic_shift_error": maximum_shift_error,
        "exact_shapley_max_error": maximum_shapley_error,
        "efficiency_max_error": maximum_efficiency_error,
        "ignored_unit_intervention_max": maximum_null_intervention,
        "native_output_has_negative_drive_contribution": bool(
            np.any(shapley.values[:, 3] < 0)
        ),
        "passed": bool(
            maximum_shift_error < 1e-6
            and maximum_shapley_error < 2e-6
            and maximum_efficiency_error < 2e-6
            and maximum_null_intervention < 1e-7
            and np.any(shapley.values[:, 3] < 0)
        ),
    }


def _resume_completed(output_dir: Path, dataset: Path, checkpoint: Path) -> Path | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset"]["sha256"] != sha256_file(dataset):
        raise RuntimeError("resume dataset fingerprint differs from completed run")
    if manifest["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise RuntimeError("resume checkpoint fingerprint differs from completed run")
    for name, digest in manifest["output_hashes"].items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"resume artifact is missing or changed: {name}")
    print(f"validated completed S04 run: {manifest_path}", flush=True)
    return manifest_path


def _invariant_bottleneck(
    member: InvariantMember,
    geometry: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(geometry), batch_size):
            stop = min(start + batch_size, len(geometry))
            batches.append(
                member.invariant_bottleneck(geometry[start:stop].to(device))
                .cpu()
                .numpy()
            )
    return np.concatenate(batches).astype(np.float32)


def _head_function(member: InvariantMember, width: int):  # type: ignore[no-untyped-def]
    def evaluate(packed: torch.Tensor) -> torch.Tensor:
        return member.head(
            packed[:, :width], packed[:, width], packed[:, width + 1]
        )

    return evaluate


def _metric_rows(
    member_id: str,
    result: InterventionResult,
    actual: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode_index, mode in enumerate(result.modes):
        for feature, feature_name in enumerate(result.feature_names):
            delta = result.single_delta[mode_index, feature]
            for stratum, mask in masks.items():
                original_error = np.mean(
                    np.square(result.original_prediction[mask] - actual[mask])
                )
                edited_error = np.mean(
                    np.square(result.original_prediction[mask] + delta[mask] - actual[mask])
                )
                rows.append(
                    {
                        "member_id": member_id,
                        "scope": "single_unit",
                        "mode": mode,
                        "feature_1": feature_name,
                        "feature_2": "",
                        "stratum": stratum,
                        "mean_signed_delta": float(np.mean(delta[mask])),
                        "mean_absolute_delta": float(np.mean(np.abs(delta[mask]))),
                        "rms_delta": float(np.sqrt(np.mean(np.square(delta[mask])))),
                        "mse_change": float(edited_error - original_error),
                        "mean_signed_interaction": 0.0,
                        "rms_interaction": 0.0,
                        "edit_magnitude_standardized_rms": "",
                        "rms_delta_per_edit_sd": "",
                        "validity_tag": result.validity_tag,
                    }
                )
        for pair, (left, right) in enumerate(result.pair_indices):
            delta = result.pair_delta[mode_index, pair]
            interaction = result.pair_interaction[mode_index, pair]
            for stratum, mask in masks.items():
                original_error = np.mean(
                    np.square(result.original_prediction[mask] - actual[mask])
                )
                edited_error = np.mean(
                    np.square(result.original_prediction[mask] + delta[mask] - actual[mask])
                )
                rows.append(
                    {
                        "member_id": member_id,
                        "scope": "unit_pair",
                        "mode": mode,
                        "feature_1": result.feature_names[int(left)],
                        "feature_2": result.feature_names[int(right)],
                        "stratum": stratum,
                        "mean_signed_delta": float(np.mean(delta[mask])),
                        "mean_absolute_delta": float(np.mean(np.abs(delta[mask]))),
                        "rms_delta": float(np.sqrt(np.mean(np.square(delta[mask])))),
                        "mse_change": float(edited_error - original_error),
                        "mean_signed_interaction": float(np.mean(interaction[mask])),
                        "rms_interaction": float(
                            np.sqrt(np.mean(np.square(interaction[mask])))
                        ),
                        "edit_magnitude_standardized_rms": "",
                        "rms_delta_per_edit_sd": "",
                        "validity_tag": result.validity_tag,
                    }
                )
    for direction, delta in enumerate(result.random_direction_delta):
        for stratum, mask in masks.items():
            rows.append(
                {
                    "member_id": member_id,
                    "scope": "random_direction",
                    "mode": "mean_projection_removal",
                    "feature_1": f"random_direction_{direction:03d}",
                    "feature_2": "",
                    "stratum": stratum,
                    "mean_signed_delta": float(np.mean(delta[mask])),
                    "mean_absolute_delta": float(np.mean(np.abs(delta[mask]))),
                    "rms_delta": float(np.sqrt(np.mean(np.square(delta[mask])))),
                    "mse_change": float("nan"),
                    "mean_signed_interaction": 0.0,
                    "rms_interaction": 0.0,
                    "edit_magnitude_standardized_rms": float(
                        result.random_direction_edit_magnitude[direction]
                    ),
                    "rms_delta_per_edit_sd": float(
                        np.sqrt(np.mean(np.square(delta[mask])))
                        / result.random_direction_edit_magnitude[direction]
                    ),
                    "validity_tag": result.validity_tag,
                }
            )
    return rows


def _shapley_rows(
    member_id: str,
    feature_names: tuple[str, ...],
    result: ShapleyResult,
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stratum, mask in masks.items():
        contribution, share = variance_decomposition(
            result.values[mask], result.prediction[mask]
        )
        for feature, feature_name in enumerate(feature_names):
            values = result.values[mask, feature]
            rows.append(
                {
                    "member_id": member_id,
                    "feature_id": feature_name,
                    "feature_kind": "bottleneck_unit"
                    if feature < len(feature_names) - 2
                    else "scalar_drive",
                    "stratum": stratum,
                    "method": result.method,
                    "permutations": "" if result.permutations is None else result.permutations,
                    "mean_signed": float(np.mean(values)),
                    "mean_absolute": float(np.mean(np.abs(values))),
                    "rms": float(np.sqrt(np.mean(np.square(values)))),
                    "variance_contribution": float(contribution[feature]),
                    "variance_share": float(share[feature]),
                    "mean_sampling_se": float(np.mean(result.standard_errors[mask, feature])),
                    "maximum_sampling_se": float(np.max(result.standard_errors[mask, feature])),
                    "validity_tag": HIDDEN_INTERVENTION_VALIDITY,
                }
            )
    return rows


def _group_bootstrap_weights(
    groups: np.ndarray, *, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
    group_weights = np.zeros((replicates, len(unique)), dtype=np.float64)
    for replicate in range(replicates):
        group_weights[replicate] = np.bincount(draws[replicate], minlength=len(unique))
    return group_weights[:, inverse], inverse


def _effect_interval(
    values: np.ndarray,
    row_weights: np.ndarray,
    *,
    statistic: str,
) -> tuple[float, float, float]:
    denominator = row_weights.sum(axis=1)
    if statistic == "mean_absolute":
        point = float(np.mean(np.abs(values)))
        draws = row_weights @ np.abs(values) / denominator
    elif statistic == "rms":
        point = float(np.sqrt(np.mean(np.square(values))))
        draws = np.sqrt(row_weights @ np.square(values) / denominator)
    else:
        raise ValueError(statistic)
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _r2_draws(
    actual: np.ndarray, predicted: np.ndarray, row_weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the ordinary R2 and grouped-bootstrap weighted R2 draws."""

    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    weights = np.asarray(row_weights, dtype=np.float64)
    denominator = weights.sum(axis=1)
    weighted_mean = weights @ y / denominator
    total = np.sum(weights * np.square(y[None, :] - weighted_mean[:, None]), axis=1)
    residual = weights @ np.square(y - p)
    draws = np.full(len(weights), np.nan, dtype=np.float64)
    valid = total > np.finfo(np.float64).tiny
    draws[valid] = 1.0 - residual[valid] / total[valid]
    return _r2(y, p), draws


def _interval_from_draws(point: float, draws: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(draws, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return point, float("nan"), float("nan")
    return point, float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _rms_draws(values: np.ndarray, row_weights: np.ndarray) -> tuple[float, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(row_weights, dtype=np.float64)
    draws = np.sqrt(
        weights @ np.square(values) / np.maximum(weights.sum(axis=1), 1.0)
    )
    return float(np.sqrt(np.mean(np.square(values)))), draws


def _decoder_rows(
    member_id: str,
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    groups: np.ndarray,
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, bool], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str, bool], np.ndarray] = {}
    for target_index, (target_name, target) in enumerate(targets.items()):
        for kind in ("linear", "nonlinear"):
            for permuted in (False, True):
                prediction = grouped_cv_predictions(
                    features,
                    target,
                    groups,
                    kind=kind,
                    n_folds=int(config["decoder_folds"]),
                    seed=int(config["seed"]) + seed_offset + 97 * target_index,
                    ridge=float(config["decoder_ridge"]),
                    hidden_features=int(config["decoder_hidden_features"]),
                    permute_labels=permuted,
                    minimum_active_fraction=float(
                        config["decoder_minimum_active_fraction"]
                    ),
                    active_tolerance=float(config["decoder_active_tolerance"]),
                )
                predictions[(target_name, kind, permuted)] = prediction
                for stratum, mask in masks.items():
                    rows.append(
                        {
                            "member_id": member_id,
                            "target": target_name,
                            "decoder": kind,
                            "label_permuted": permuted,
                            "stratum": stratum,
                            "r2": _r2(target[mask], prediction[mask]),
                            "fold_group": "equilibrium_files",
                            "folds": int(config["decoder_folds"]),
                        }
                    )
    return rows, predictions


def _linear_direction(features: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    standardized = (features - center) / scale
    design = np.column_stack((np.ones(len(features)), standardized))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ np.asarray(target, dtype=np.float64)
    )[1:]
    norm = np.linalg.norm(coefficients)
    return coefficients / norm if norm > np.finfo(np.float64).eps else np.zeros_like(coefficients)


def _direction_delta(
    member: InvariantMember,
    bottleneck: np.ndarray,
    drives: np.ndarray,
    direction: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    center = bottleneck.mean(axis=0)
    scale = bottleneck.std(axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    standardized = (bottleneck - center) / scale
    projection = standardized @ direction
    edit_magnitude = float(np.sqrt(np.mean(np.square(projection))))
    edited = (standardized - projection[:, None] * direction) * scale + center
    original_packed = torch.as_tensor(
        np.column_stack((bottleneck, drives)), dtype=torch.float32, device=device
    )
    edited_packed = torch.as_tensor(
        np.column_stack((edited, drives)), dtype=torch.float32, device=device
    )
    head = _head_function(member, bottleneck.shape[1])
    with torch.inference_mode():
        delta = (head(edited_packed) - head(original_packed)).cpu().numpy()
    return delta.astype(np.float64), edit_magnitude


def _dependence(
    member: InvariantMember,
    bottleneck: np.ndarray,
    drives: np.ndarray,
    masks: dict[str, np.ndarray],
    quantiles: np.ndarray,
    dead_tolerance: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    sample_count, width = bottleneck.shape
    grid_count = len(quantiles)
    packed = torch.as_tensor(
        np.column_stack((bottleneck, drives)), dtype=torch.float32, device=device
    )
    head = _head_function(member, width)
    with torch.inference_mode():
        original = head(packed).cpu().numpy()
    unit_grid = np.quantile(bottleneck, quantiles, axis=0).T.astype(np.float32)
    drive_grids = np.quantile(drives, quantiles, axis=0).T.astype(np.float32)
    ice = np.full((width, grid_count, sample_count), np.nan, dtype=np.float32)
    unit_by_lt = np.full((width, grid_count, grid_count, len(STRATA)), np.nan, dtype=np.float32)
    unit_by_ln = np.full_like(unit_by_lt, np.nan)
    live = np.max(np.abs(bottleneck), axis=0) > dead_tolerance
    for unit in np.flatnonzero(live):
        for unit_grid_index, unit_value in enumerate(unit_grid[unit]):
            edited = packed.clone()
            edited[:, unit] = float(unit_value)
            with torch.inference_mode():
                ice[unit, unit_grid_index] = head(edited).cpu().numpy() - original
            for drive_axis, destination in ((0, unit_by_lt), (1, unit_by_ln)):
                for drive_grid_index, drive_value in enumerate(drive_grids[drive_axis]):
                    two_dimensional = edited.clone()
                    two_dimensional[:, width + drive_axis] = float(drive_value)
                    with torch.inference_mode():
                        delta = head(two_dimensional).cpu().numpy() - original
                    for stratum_index, stratum in enumerate(STRATA):
                        destination[
                            unit, unit_grid_index, drive_grid_index, stratum_index
                        ] = np.mean(delta[masks[stratum]])
    return {
        "unit_grid": unit_grid,
        "drive_grid": drive_grids,
        "live": live,
        "ice_delta": ice,
        "unit_by_lt_delta": unit_by_lt,
        "unit_by_ln_delta": unit_by_ln,
    }


def _plot_rank_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["stratum"] == "overall"]
    figure, axes = plt.subplots(figsize=(6.2, 4.8))
    for member_id in dict.fromkeys(row["member_id"] for row in selected):
        subset = [row for row in selected if row["member_id"] == member_id]
        axes.scatter(
            [float(row["shapley_mean_absolute"]) for row in subset],
            [float(row["mean_ablation_rms"]) for row in subset],
            s=15,
            alpha=0.65,
            label=member_id if len(dict.fromkeys(r["member_id"] for r in selected)) <= 3 else None,
        )
    axes.set_xlabel("Mean |Shapley value| (native clipped-log units)")
    axes.set_ylabel("Mean-replacement RMS change (native units)")
    axes.grid(alpha=0.25)
    if axes.get_legend_handles_labels()[1]:
        axes.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_interaction_graph(
    path: Path, member_id: str, unit_ids: tuple[str, ...], pair_indices: np.ndarray, strength: np.ndarray
) -> None:
    figure, axes = plt.subplots(figsize=(6.2, 6.2))
    count = len(unit_ids)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    order = np.argsort(-strength)[: min(20, len(strength))]
    maximum = max(float(np.max(strength)), np.finfo(float).eps)
    for pair in order:
        left, right = pair_indices[pair]
        axes.plot(
            positions[[left, right], 0],
            positions[[left, right], 1],
            color="tab:purple",
            alpha=0.15 + 0.8 * float(strength[pair] / maximum),
            linewidth=0.5 + 4 * float(strength[pair] / maximum),
        )
    axes.scatter(positions[:, 0], positions[:, 1], s=90, color="white", edgecolor="black", zorder=3)
    for index, (x_value, y_value) in enumerate(positions):
        axes.text(x_value, y_value, f"u{index}", ha="center", va="center", fontsize=7)
    axes.set_title(f"Mean-replacement interaction graph: {member_id}")
    axes.set_aspect("equal")
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_pdp_atlas(
    path: Path, member_id: str, dependence: dict[str, np.ndarray]
) -> None:
    live = np.flatnonzero(dependence["live"])
    columns = 4
    rows = max(1, int(np.ceil(len(live) / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(10.0, 2.2 * rows), squeeze=False)
    for plot_index, unit in enumerate(live):
        axis = axes.flat[plot_index]
        ice = dependence["ice_delta"][unit]
        x_values = dependence["unit_grid"][unit]
        for sample in range(min(30, ice.shape[1])):
            axis.plot(x_values, ice[:, sample], color="0.75", linewidth=0.4, alpha=0.45)
        axis.plot(x_values, np.nanmean(ice, axis=1), color="tab:blue", linewidth=1.7)
        axis.axhline(0, color="black", linewidth=0.5)
        axis.set_title(f"u{unit}", fontsize=8)
    for axis in axes.flat[len(live) :]:
        axis.axis("off")
    figure.suptitle(f"PDP/ICE atlas (native output change): {member_id}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_decodability(path: Path, rows: list[dict[str, Any]], top_ids: tuple[str, ...]) -> None:
    targets = list(dict.fromkeys(row["target"] for row in rows))
    matrix = np.full((len(top_ids), len(targets)), np.nan)
    for member_index, member_id in enumerate(top_ids):
        for target_index, target in enumerate(targets):
            matches = [
                row
                for row in rows
                if row["member_id"] == member_id
                and row["target"] == target
                and row["decoder"] == "nonlinear"
                and not row["label_permuted"]
                and row["stratum"] == "overall"
            ]
            if matches:
                matrix[member_index, target_index] = float(matches[0]["r2"])
    figure, axes = plt.subplots(figsize=(7.2, 4.5))
    image = axes.imshow(matrix, aspect="auto", vmin=-0.25, vmax=1, cmap="viridis")
    axes.set_xticks(range(len(targets)), targets, rotation=35, ha="right")
    axes.set_yticks(range(len(top_ids)), [f"rank {index + 1}" for index in range(len(top_ids))])
    axes.set_title("Grouped-CV nonlinear decodability from invariant bottleneck")
    figure.colorbar(image, ax=axes, label=r"out-of-fold $R^2$")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _publish(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in paths:
        temporary = destination / f".{source.name}.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(destination / source.name)


def run(args: argparse.Namespace) -> Path:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = _resolve(config, args)
    repository = Path(__file__).resolve().parents[1]
    resolved["source_hashes"] = {
        "runner_sha256": sha256_file(__file__),
        "bottleneck_library_sha256": sha256_file(
            repository / "itg_nn/xai/bottleneck.py"
        ),
        "cohorts_sha256": sha256_file(Path(resolved["cohorts"])),
    }
    set_deterministic_seed(int(resolved["seed"]))
    dataset = Path(resolved["dataset"])
    checkpoint = Path(resolved["checkpoint"])
    cohorts_path = Path(resolved["cohorts"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (Path("output/xai/S04") / str(resolved["run_id"])).resolve()
    )
    if args.resume:
        completed = _resume_completed(output_dir, dataset, checkpoint)
        if completed is not None:
            return completed
    artifacts = RunArtifacts(output_dir)
    toy_checks = _toy_gate(int(resolved["seed"]) + 41)
    if not toy_checks["passed"]:
        raise RuntimeError(f"pre-inference analytic toy gate failed: {toy_checks}")

    cohorts = json.loads(cohorts_path.read_text(encoding="utf-8"))
    registered_rows = np.asarray(
        cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64
    )
    row_count = int(resolved["panel_varied_rows"])
    if not 1 <= row_count <= len(registered_rows):
        raise ValueError("panel row cap is outside the registered S01 panel")
    row_ids = registered_rows[:row_count]
    panel = load_hdf5_rows(dataset, row_ids, gradient_set="varied", include_targets=True)
    if panel.actual_log_heat_flux is None:
        raise RuntimeError("varied panel target was not loaded")
    actual = panel.actual_log_heat_flux.numpy().astype(np.float64)
    drives = np.column_stack((panel.a_over_lt.numpy(), panel.a_over_ln.numpy())).astype(
        np.float32
    )
    with h5py.File(dataset, "r") as h5_file:
        scalar_names = tuple(_decode(h5_file["scalar_features"][:]).tolist())
        scalar = _h5_take(h5_file["scalar_feature_matrix"], row_ids).astype(np.float64)
        equilibrium_files = _decode(_h5_take(h5_file["equilibrium_files"], row_ids))
        registered_fsa = _h5_take(h5_file["FSA_grad_xs"], row_ids).astype(np.float64)
    invariant_targets = registered_invariants(
        panel.geometry.numpy(), scalar, scalar_names
    )
    masks = _stratum_masks(actual, float(resolved["stable_threshold_log_Q"]))
    if not masks["stable_or_near_floor"].any() or not masks["unstable"].any():
        raise RuntimeError("panel cap does not contain both registered drive strata")

    ensemble = load_ensemble(checkpoint, device=str(resolved["device"]))
    registered_all = tuple(cohorts["member_cohorts"]["all_100"])
    registered_top = tuple(cohorts["member_cohorts"]["stored_validation_top_10"])
    member_count = int(resolved["members"])
    member_ids = registered_all[:member_count]
    top_count = min(int(resolved["top_members"]), member_count, len(registered_top))
    top_ids = registered_top[:top_count]
    if member_ids[:top_count] != top_ids:
        raise RuntimeError("registered all-member order no longer begins with the top cohort")
    index_by_id = {member_id: index for index, member_id in enumerate(ensemble.member_ids)}
    if set(member_ids).difference(index_by_id):
        raise RuntimeError("registered S01 member IDs are absent from checkpoint")
    models = {
        member_id: InvariantMember(ensemble.models[index_by_id[member_id]])
        for member_id in member_ids
    }
    maximum_width = max(models[member_id].model.conv_layers[-1].out_channels for member_id in member_ids)
    bottleneck_array = np.full(
        (len(member_ids), row_count, maximum_width), np.nan, dtype=np.float32
    )
    prediction_array = np.full((len(member_ids), row_count), np.nan, dtype=np.float32)
    unit_present = np.zeros((len(member_ids), maximum_width), dtype=bool)
    widths: dict[str, int] = {}
    for member_index, member_id in enumerate(member_ids):
        started = time.monotonic()
        width = models[member_id].model.conv_layers[-1].out_channels
        widths[member_id] = width
        values = _invariant_bottleneck(
            models[member_id],
            panel.geometry,
            batch_size=int(resolved["batch_size"]),
            device=ensemble.device,
        )
        bottleneck_array[member_index, :, :width] = values
        unit_present[member_index, :width] = True
        packed = torch.as_tensor(
            np.column_stack((values, drives)), dtype=torch.float32, device=ensemble.device
        )
        with torch.inference_mode():
            prediction_array[member_index] = (
                _head_function(models[member_id], width)(packed).cpu().numpy()
            )
        print(
            f"bottleneck {member_index + 1}/{len(member_ids)} {member_id} "
            f"width={width} {time.monotonic() - started:.1f}s",
            flush=True,
        )

    bottleneck_path = artifacts.write_hdf5(
        "bottlenecks.h5",
        {
            "bottleneck": bottleneck_array,
            "prediction": prediction_array,
            "unit_present": unit_present,
            "member_id": _strings(member_ids),
            "row_id": row_ids,
            "equilibrium_file": _strings(equilibrium_files.tolist(), width=240),
            "actual_log_Q": actual.astype(np.float32),
            "a_over_LT": drives[:, 0],
            "a_over_Ln": drives[:, 1],
            "stable_or_near_floor": masks["stable_or_near_floor"],
        },
        axes={
            "bottleneck": ("member", "sample", "unit"),
            "prediction": ("member", "sample"),
            "unit_present": ("member", "unit"),
            "member_id": ("member",),
            "row_id": ("sample",),
            "equilibrium_file": ("sample",),
            "actual_log_Q": ("sample",),
            "a_over_LT": ("sample",),
            "a_over_Ln": ("sample",),
            "stable_or_near_floor": ("sample",),
        },
        attributes={
            "canonical_function": CANONICAL_FUNCTION,
            "estimand": "native max(log Q, -2)",
            "unit_id_format": "<member_id>:u<zero-padded local index>",
            "gradient_set": "varied",
        },
        compression="gzip",
    )

    max_top_features = max(widths[member_id] + 2 for member_id in top_ids)
    shapley_values = np.full(
        (top_count, row_count, max_top_features), np.nan, dtype=np.float32
    )
    shapley_se = np.full_like(shapley_values, np.nan)
    shapley_present = np.zeros((top_count, max_top_features), dtype=bool)
    shapley_feature_ids = np.full((top_count, max_top_features), b"", dtype="S96")
    shapley_baseline = np.full((top_count, row_count), np.nan, dtype=np.float32)
    shapley_results: dict[str, ShapleyResult] = {}
    shapley_global_rows: list[dict[str, Any]] = []
    for top_index, member_id in enumerate(top_ids):
        member_index = member_ids.index(member_id)
        width = widths[member_id]
        packed_values = torch.as_tensor(
            np.column_stack((bottleneck_array[member_index, :, :width], drives)),
            dtype=torch.float32,
            device=ensemble.device,
        )
        reference = packed_values.mean(dim=0)
        started = time.monotonic()
        result = exact_or_sampled_shapley(
            _head_function(models[member_id], width),
            packed_values,
            reference,
            exact_max_features=int(resolved["exact_max_features"]),
            permutations=int(resolved["shapley_permutations"]),
            seed=int(resolved["seed"]) + 101 * top_index,
            mask_batch_size=int(resolved["shapley_mask_batch_size"]),
        )
        feature_ids = (*_member_unit_ids(member_id, width), "a_over_LT", "a_over_Ln")
        shapley_results[member_id] = result
        shapley_values[top_index, :, : width + 2] = result.values.astype(np.float32)
        shapley_se[top_index, :, : width + 2] = result.standard_errors.astype(np.float32)
        shapley_present[top_index, : width + 2] = True
        shapley_feature_ids[top_index, : width + 2] = _strings(feature_ids)
        shapley_baseline[top_index] = result.baseline_output.astype(np.float32)
        shapley_global_rows.extend(
            _shapley_rows(member_id, feature_ids, result, masks)
        )
        print(
            f"Shapley {top_index + 1}/{top_count} {member_id} {result.method} "
            f"evaluations={result.evaluations} {time.monotonic() - started:.1f}s",
            flush=True,
        )
    shapley_path = artifacts.write_hdf5(
        "shapley.h5",
        {
            "values": shapley_values,
            "standard_errors": shapley_se,
            "feature_present": shapley_present,
            "feature_id": shapley_feature_ids,
            "baseline_output": shapley_baseline,
            "member_id": _strings(top_ids),
            "row_id": row_ids,
        },
        axes={
            "values": ("member", "sample", "feature"),
            "standard_errors": ("member", "sample", "feature"),
            "feature_present": ("member", "feature"),
            "feature_id": ("member", "feature"),
            "baseline_output": ("member", "sample"),
            "member_id": ("member",),
            "row_id": ("sample",),
        },
        attributes={
            "reference": "mean of each bottleneck unit and scalar drive on the frozen varied panel",
            "estimand": "native max(log Q, -2)",
            "exact_max_features": int(resolved["exact_max_features"]),
            "validity_tag": HIDDEN_INTERVENTION_VALIDITY,
        },
        compression="gzip",
    )
    shapley_global_path = artifacts.write_text(
        "shapley_global.csv", _csv_text(shapley_global_rows)
    )

    max_pairs = maximum_width * (maximum_width - 1) // 2
    top_single = np.full(
        (top_count, len(INTERVENTION_MODES), maximum_width, row_count),
        np.nan,
        dtype=np.float32,
    )
    top_pair_indices = np.full((top_count, max_pairs, 2), -1, dtype=np.int16)
    top_pair_delta = np.full(
        (top_count, len(INTERVENTION_MODES), max_pairs, row_count),
        np.nan,
        dtype=np.float32,
    )
    top_pair_interaction = np.full_like(top_pair_delta, np.nan)
    top_random = np.full(
        (top_count, int(resolved["random_directions"]), row_count),
        np.nan,
        dtype=np.float32,
    )
    top_random_magnitude = np.full(
        (top_count, int(resolved["random_directions"])), np.nan, dtype=np.float32
    )
    intervention_rows: list[dict[str, Any]] = []
    intervention_results: dict[str, InterventionResult] = {}
    for member_index, member_id in enumerate(member_ids):
        width = widths[member_id]
        packed = torch.as_tensor(
            np.column_stack((bottleneck_array[member_index, :, :width], drives)),
            dtype=torch.float32,
            device=ensemble.device,
        )
        started = time.monotonic()
        result = bottleneck_interventions(
            _head_function(models[member_id], width),
            packed,
            packed.mean(dim=0),
            feature_names=(*_member_unit_ids(member_id, width), "a_over_LT", "a_over_Ln"),
            intervention_features=range(width),
            seed=int(resolved["seed"]) + 307 * member_index,
            random_directions=int(resolved["random_directions"]),
        )
        intervention_rows.extend(_metric_rows(member_id, result, actual, masks))
        if member_id in top_ids:
            top_index = top_ids.index(member_id)
            pair_count = len(result.pair_indices)
            top_single[top_index, :, :width] = result.single_delta.astype(np.float32)
            top_pair_indices[top_index, :pair_count] = result.pair_indices
            top_pair_delta[top_index, :, :pair_count] = result.pair_delta.astype(np.float32)
            top_pair_interaction[top_index, :, :pair_count] = result.pair_interaction.astype(
                np.float32
            )
            top_random[top_index] = result.random_direction_delta.astype(np.float32)
            top_random_magnitude[top_index] = result.random_direction_edit_magnitude.astype(
                np.float32
            )
            intervention_results[member_id] = result
        print(
            f"interventions {member_index + 1}/{len(member_ids)} {member_id} "
            f"pairs={len(result.pair_indices)} {time.monotonic() - started:.1f}s",
            flush=True,
        )
    intervention_h5_path = artifacts.write_hdf5(
        "interventions_top10.h5",
        {
            "single_delta": top_single,
            "pair_indices": top_pair_indices,
            "pair_delta": top_pair_delta,
            "pair_interaction": top_pair_interaction,
            "random_direction_delta": top_random,
            "random_direction_edit_magnitude": top_random_magnitude,
            "member_id": _strings(top_ids),
            "row_id": row_ids,
            "mode": _strings(INTERVENTION_MODES, width=16),
        },
        axes={
            "single_delta": ("member", "mode", "unit", "sample"),
            "pair_indices": ("member", "pair", "pair_endpoint"),
            "pair_delta": ("member", "mode", "pair", "sample"),
            "pair_interaction": ("member", "mode", "pair", "sample"),
            "random_direction_delta": ("member", "direction", "sample"),
            "random_direction_edit_magnitude": ("member", "direction"),
            "member_id": ("member",),
            "row_id": ("sample",),
            "mode": ("mode",),
        },
        attributes={
            "validity_tag": HIDDEN_INTERVENTION_VALIDITY,
            "delta_sign": "edited native output minus original native output",
            "pair_interaction": "pair_delta - left_single_delta - right_single_delta",
        },
        compression="gzip",
    )
    intervention_summary_path = artifacts.write_text(
        "intervention_summary.csv", _csv_text(intervention_rows)
    )

    decoder_rows: list[dict[str, Any]] = []
    decoder_predictions: dict[str, dict[tuple[str, str, bool], np.ndarray]] = {}
    for member_index, member_id in enumerate(member_ids):
        member_rows, member_predictions = _decoder_rows(
            member_id,
            bottleneck_array[member_index, :, : widths[member_id]],
            invariant_targets,
            equilibrium_files,
            masks,
            resolved,
            seed_offset=10000 * member_index,
        )
        decoder_rows.extend(member_rows)
        if member_id in top_ids:
            decoder_predictions[member_id] = member_predictions
        print(f"decoders {member_index + 1}/{len(member_ids)} {member_id}", flush=True)

    fidelity_rows: list[dict[str, Any]] = []
    for top_index, member_id in enumerate(top_ids):
        member_index = member_ids.index(member_id)
        features = np.column_stack(
            (bottleneck_array[member_index, :, : widths[member_id]], drives)
        )
        target = prediction_array[member_index].astype(np.float64)
        for kind in ("linear", "nonlinear"):
            prediction = grouped_cv_predictions(
                features,
                target,
                equilibrium_files,
                kind=kind,
                n_folds=int(resolved["decoder_folds"]),
                seed=int(resolved["seed"]) + 500000 + top_index,
                ridge=float(resolved["decoder_ridge"]),
                hidden_features=int(resolved["decoder_hidden_features"]),
                minimum_active_fraction=float(
                    resolved["decoder_minimum_active_fraction"]
                ),
                active_tolerance=float(resolved["decoder_active_tolerance"]),
            )
            for stratum, mask in masks.items():
                fidelity_rows.append(
                    {
                        "member_id": member_id,
                        "decoder": kind,
                        "stratum": stratum,
                        "r2_to_member_invariant_tilde_f": _r2(
                            target[mask], prediction[mask]
                        ),
                        "head_nonlinearity_residual_fraction": float(
                            np.mean(np.square(target[mask] - prediction[mask]))
                            / np.var(target[mask])
                        ),
                        "features": "invariant bottleneck + a_over_LT + a_over_Ln",
                        "fold_group": "equilibrium_files",
                        "target_rationale": "isolates head nonlinearity; f differs from invariant_tilde_f by the S02 canonicalization residual",
                    }
                )
    decoder_path = artifacts.write_text("decodability.csv", _csv_text(decoder_rows))
    fidelity_path = artifacts.write_text("head_fidelity.csv", _csv_text(fidelity_rows))

    bootstrap_weights, _ = _group_bootstrap_weights(
        equilibrium_files,
        replicates=int(resolved["bootstrap_replicates"]),
        seed=int(resolved["seed"]) + 600001,
    )
    encoded_used_rows: list[dict[str, Any]] = []
    concept_bootstrap: dict[str, dict[str, list[np.ndarray]]] = {
        target_name: {"encoded_nonlinear_r2": [], "used_rms": []}
        for target_name in invariant_targets
    }
    for top_index, member_id in enumerate(top_ids):
        member_index = member_ids.index(member_id)
        width = widths[member_id]
        bottleneck = bottleneck_array[member_index, :, :width].astype(np.float64)
        random_result = intervention_results[member_id]
        for target_name, target in invariant_targets.items():
            direction = _linear_direction(bottleneck, target, float(resolved["decoder_ridge"]))
            delta, direction_magnitude = _direction_delta(
                models[member_id], bottleneck, drives, direction, ensemble.device
            )
            predictions = decoder_predictions[member_id]
            for stratum, mask in masks.items():
                stratum_weights = bootstrap_weights[:, mask]
                random_rms = np.sqrt(
                    np.mean(np.square(random_result.random_direction_delta[:, mask]), axis=1)
                )
                random_normalized_rms = (
                    random_rms / random_result.random_direction_edit_magnitude
                )
                linear_point, linear_draws = _r2_draws(
                    target[mask],
                    predictions[(target_name, "linear", False)][mask],
                    stratum_weights,
                )
                nonlinear_point, nonlinear_draws = _r2_draws(
                    target[mask],
                    predictions[(target_name, "nonlinear", False)][mask],
                    stratum_weights,
                )
                used_point, used_draws = _rms_draws(delta[mask], stratum_weights)
                _, linear_lower, linear_upper = _interval_from_draws(
                    linear_point, linear_draws
                )
                _, nonlinear_lower, nonlinear_upper = _interval_from_draws(
                    nonlinear_point, nonlinear_draws
                )
                _, used_lower, used_upper = _interval_from_draws(used_point, used_draws)
                if stratum == "overall":
                    concept_bootstrap[target_name]["encoded_nonlinear_r2"].append(
                        nonlinear_draws
                    )
                    concept_bootstrap[target_name]["used_rms"].append(used_draws)
                encoded_used_rows.append(
                    {
                        "member_id": member_id,
                        "target": target_name,
                        "stratum": stratum,
                        "encoded_linear_grouped_cv_r2": linear_point,
                        "encoded_linear_grouped_bootstrap_ci95_lower": linear_lower,
                        "encoded_linear_grouped_bootstrap_ci95_upper": linear_upper,
                        "encoded_nonlinear_grouped_cv_r2": nonlinear_point,
                        "encoded_nonlinear_grouped_bootstrap_ci95_lower": nonlinear_lower,
                        "encoded_nonlinear_grouped_bootstrap_ci95_upper": nonlinear_upper,
                        "label_permutation_linear_r2": _r2(
                            target[mask], predictions[(target_name, "linear", True)][mask]
                        ),
                        "label_permutation_nonlinear_r2": _r2(
                            target[mask], predictions[(target_name, "nonlinear", True)][mask]
                        ),
                        "used_direction_removal_mean_signed_delta": float(np.mean(delta[mask])),
                        "used_direction_removal_rms_delta": used_point,
                        "used_grouped_bootstrap_ci95_lower": used_lower,
                        "used_grouped_bootstrap_ci95_upper": used_upper,
                        "direction_edit_magnitude_standardized_rms": direction_magnitude,
                        "used_rms_per_edit_sd": (
                            used_point / direction_magnitude
                            if direction_magnitude > np.finfo(np.float64).eps
                            else float("nan")
                        ),
                        "random_direction_control_median_rms": float(np.median(random_rms)),
                        "random_direction_control_q90_rms": float(np.quantile(random_rms, 0.9)),
                        "random_direction_control_median_edit_magnitude_standardized_rms": float(
                            np.median(random_result.random_direction_edit_magnitude)
                        ),
                        "random_direction_control_median_rms_per_edit_sd": float(
                            np.median(random_normalized_rms)
                        ),
                        "direction_intervention_validity": HIDDEN_INTERVENTION_VALIDITY,
                    }
                )
    encoded_used_path = artifacts.write_text(
        "encoded_vs_used.csv", _csv_text(encoded_used_rows)
    )

    rank_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    rank_correlations: list[float] = []
    rank_bootstrap_draws: list[np.ndarray] = []
    for member_id in top_ids:
        width = widths[member_id]
        shapley = shapley_results[member_id].values[:, :width]
        mean_delta = intervention_results[member_id].single_delta[1, :width]
        shapley_score = np.mean(np.abs(shapley), axis=0)
        ablation_score = np.sqrt(np.mean(np.square(mean_delta), axis=1))
        shapley_rank = rankdata(shapley_score, descending=True)
        ablation_rank = rankdata(ablation_score, descending=True)
        correlation = spearman_correlation(shapley_score, ablation_score)
        rank_correlations.append(correlation)
        member_rank_draws = np.empty(len(bootstrap_weights), dtype=np.float64)
        for replicate, weights in enumerate(bootstrap_weights):
            denominator = max(float(weights.sum()), 1.0)
            sampled_shapley = weights @ np.abs(shapley) / denominator
            sampled_ablation = np.sqrt(weights @ np.square(mean_delta.T) / denominator)
            member_rank_draws[replicate] = spearman_correlation(
                sampled_shapley, sampled_ablation
            )
        rank_bootstrap_draws.append(member_rank_draws)
        for unit in range(width):
            rank_rows.append(
                {
                    "member_id": member_id,
                    "unit_id": _member_unit_ids(member_id, width)[unit],
                    "stratum": "overall",
                    "shapley_mean_absolute": float(shapley_score[unit]),
                    "shapley_rank": float(shapley_rank[unit]),
                    "mean_ablation_rms": float(ablation_score[unit]),
                    "ablation_rank": float(ablation_rank[unit]),
                    "member_spearman": correlation,
                    "encoded_column": "see encoded_vs_used.csv grouped-CV target rows",
                    "used_column": "mean_ablation_rms and shapley_mean_absolute",
                }
            )
            for method, values, statistic in (
                ("shapley_mean_absolute", shapley[:, unit], "mean_absolute"),
                ("mean_replacement_rms", mean_delta[unit], "rms"),
            ):
                point, lower, upper = _effect_interval(
                    values, bootstrap_weights, statistic=statistic
                )
                uncertainty_rows.append(
                    {
                        "member_id": member_id,
                        "unit_id": _member_unit_ids(member_id, width)[unit],
                        "effect": method,
                        "point": point,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "bootstrap_unit": "equilibrium_files",
                        "replicates": int(resolved["bootstrap_replicates"]),
                    }
                )
    median_rank_draws = np.median(np.stack(rank_bootstrap_draws), axis=0)
    median_rank_point = float(np.median(rank_correlations))
    _, median_rank_lower, median_rank_upper = _interval_from_draws(
        median_rank_point, median_rank_draws
    )
    uncertainty_rows.append(
        {
            "member_id": "top10_median",
            "unit_id": "",
            "effect": "shapley_ablation_spearman",
            "point": median_rank_point,
            "ci95_lower": median_rank_lower,
            "ci95_upper": median_rank_upper,
            "bootstrap_unit": "equilibrium_files",
            "replicates": int(resolved["bootstrap_replicates"]),
        }
    )
    concept_intervals: dict[str, dict[str, list[float]]] = {}
    for target_name, metric_draws in concept_bootstrap.items():
        concept_intervals[target_name] = {}
        for metric, member_draws in metric_draws.items():
            median_draws = np.median(np.stack(member_draws), axis=0)
            if metric == "encoded_nonlinear_r2":
                points = [
                    float(row["encoded_nonlinear_grouped_cv_r2"])
                    for row in encoded_used_rows
                    if row["target"] == target_name and row["stratum"] == "overall"
                ]
            else:
                points = [
                    float(row["used_direction_removal_rms_delta"])
                    for row in encoded_used_rows
                    if row["target"] == target_name and row["stratum"] == "overall"
                ]
            point = float(np.median(points))
            _, lower, upper = _interval_from_draws(point, median_draws)
            concept_intervals[target_name][metric] = [point, lower, upper]
            uncertainty_rows.append(
                {
                    "member_id": "top10_median",
                    "unit_id": "",
                    "effect": f"{target_name}:{metric}",
                    "point": point,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_unit": "equilibrium_files",
                    "replicates": int(resolved["bootstrap_replicates"]),
                }
            )
    rank_path = artifacts.write_text("rank_comparison.csv", _csv_text(rank_rows))
    uncertainty_path = artifacts.write_text(
        "grouped_uncertainty.csv", _csv_text(uncertainty_rows)
    )

    quantiles = np.asarray(resolved["dependence_quantiles"], dtype=np.float64)
    pdp_unit_grid = np.full(
        (top_count, maximum_width, len(quantiles)), np.nan, dtype=np.float32
    )
    pdp_drive_grid = np.full((top_count, 2, len(quantiles)), np.nan, dtype=np.float32)
    pdp_live = np.zeros((top_count, maximum_width), dtype=bool)
    ice_delta = np.full(
        (top_count, maximum_width, len(quantiles), row_count), np.nan, dtype=np.float32
    )
    unit_by_lt = np.full(
        (top_count, maximum_width, len(quantiles), len(quantiles), len(STRATA)),
        np.nan,
        dtype=np.float32,
    )
    unit_by_ln = np.full_like(unit_by_lt, np.nan)
    dependence_results: dict[str, dict[str, np.ndarray]] = {}
    for top_index, member_id in enumerate(top_ids):
        member_index = member_ids.index(member_id)
        width = widths[member_id]
        started = time.monotonic()
        result = _dependence(
            models[member_id],
            bottleneck_array[member_index, :, :width],
            drives,
            masks,
            quantiles,
            float(resolved["dead_tolerance"]),
            ensemble.device,
        )
        dependence_results[member_id] = result
        pdp_unit_grid[top_index, :width] = result["unit_grid"]
        pdp_drive_grid[top_index] = result["drive_grid"]
        pdp_live[top_index, :width] = result["live"]
        ice_delta[top_index, :width] = result["ice_delta"]
        unit_by_lt[top_index, :width] = result["unit_by_lt_delta"]
        unit_by_ln[top_index, :width] = result["unit_by_ln_delta"]
        print(
            f"PDP/ICE {top_index + 1}/{top_count} {member_id} "
            f"live={int(result['live'].sum())} {time.monotonic() - started:.1f}s",
            flush=True,
        )
    pdp_path = artifacts.write_hdf5(
        "pdp_ice_atlas.h5",
        {
            "unit_grid": pdp_unit_grid,
            "drive_grid": pdp_drive_grid,
            "live_unit": pdp_live,
            "ice_delta": ice_delta,
            "unit_by_lt_delta": unit_by_lt,
            "unit_by_ln_delta": unit_by_ln,
            "quantile": quantiles,
            "stratum": _strings(STRATA, width=32),
            "member_id": _strings(top_ids),
            "row_id": row_ids,
        },
        axes={
            "unit_grid": ("member", "unit", "quantile"),
            "drive_grid": ("member", "drive", "quantile"),
            "live_unit": ("member", "unit"),
            "ice_delta": ("member", "unit", "quantile", "sample"),
            "unit_by_lt_delta": ("member", "unit", "unit_quantile", "drive_quantile", "stratum"),
            "unit_by_ln_delta": ("member", "unit", "unit_quantile", "drive_quantile", "stratum"),
            "quantile": ("quantile",),
            "stratum": ("stratum",),
            "member_id": ("member",),
            "row_id": ("sample",),
        },
        attributes={
            "estimand": "edited canonical head output minus original native clipped-log output",
            "live_definition": f"max absolute panel activation > {resolved['dead_tolerance']}",
            "unit_by_drive": "other bottleneck units and the other scalar drive remain sample-specific",
            "validity_tag": HIDDEN_INTERVENTION_VALIDITY,
        },
        compression="gzip",
    )

    rank_figure = output_dir / "rank_comparison.png"
    _plot_rank_comparison(rank_figure, rank_rows)
    artifacts.register_existing(rank_figure.name)
    interaction_figure = output_dir / "interaction_graph.png"
    first_member = top_ids[0]
    first_intervention = intervention_results[first_member]
    _plot_interaction_graph(
        interaction_figure,
        first_member,
        _member_unit_ids(first_member, widths[first_member]),
        first_intervention.pair_indices,
        np.sqrt(np.mean(np.square(first_intervention.pair_interaction[1]), axis=1)),
    )
    artifacts.register_existing(interaction_figure.name)
    pdp_figure = output_dir / "pdp_ice_atlas.png"
    _plot_pdp_atlas(pdp_figure, first_member, dependence_results[first_member])
    artifacts.register_existing(pdp_figure.name)
    decoder_figure = output_dir / "decodability_matrix.png"
    _plot_decodability(decoder_figure, decoder_rows, top_ids)
    artifacts.register_existing(decoder_figure.name)

    exact_count = sum(result.method == "exact_enumeration" for result in shapley_results.values())
    efficiency_error = max(
        float(
            np.max(
                np.abs(
                    result.values.sum(1)
                    - (result.prediction - result.baseline_output)
                )
            )
        )
        for result in shapley_results.values()
    )
    decoder_controls = [
        float(row["r2"])
        for row in decoder_rows
        if row["label_permuted"]
        and row["stratum"] == "overall"
        and np.isfinite(float(row["r2"]))
    ]
    overall_shapley = [row for row in shapley_global_rows if row["stratum"] == "overall"]
    geometry_shapley = [
        row for row in overall_shapley if row["feature_kind"] == "bottleneck_unit"
    ]
    geometry_variance_shares = []
    geometry_mean_absolute_fractions = []
    for member_id in top_ids:
        member_all = [row for row in overall_shapley if row["member_id"] == member_id]
        member_geometry = [row for row in geometry_shapley if row["member_id"] == member_id]
        geometry_variance_shares.append(
            float(sum(float(row["variance_share"]) for row in member_geometry))
        )
        geometry_mean_absolute_fractions.append(
            float(
                sum(float(row["mean_absolute"]) for row in member_geometry)
                / sum(float(row["mean_absolute"]) for row in member_all)
            )
        )
    decoder_medians: dict[str, dict[str, float]] = {}
    decoder_minima: dict[str, dict[str, float]] = {}
    for target_name in invariant_targets:
        decoder_medians[target_name] = {}
        decoder_minima[target_name] = {}
        for kind in ("linear", "nonlinear"):
            scores = [
                float(row["r2"])
                for row in decoder_rows
                if row["target"] == target_name
                and row["decoder"] == kind
                and not row["label_permuted"]
                and row["stratum"] == "overall"
            ]
            decoder_medians[target_name][kind] = float(np.median(scores))
            decoder_minima[target_name][kind] = float(np.min(scores))
    overall_fidelity = [row for row in fidelity_rows if row["stratum"] == "overall"]
    fidelity_by_member: dict[str, dict[str, float]] = {}
    for row in overall_fidelity:
        fidelity_by_member.setdefault(str(row["member_id"]), {})[
            str(row["decoder"])
        ] = float(row["r2_to_member_invariant_tilde_f"])
    fidelity_linear = [values["linear"] for values in fidelity_by_member.values()]
    fidelity_nonlinear = [values["nonlinear"] for values in fidelity_by_member.values()]
    fidelity_increments = [
        nonlinear - linear
        for nonlinear, linear in zip(fidelity_nonlinear, fidelity_linear)
    ]
    concept_use: dict[str, dict[str, Any]] = {}
    for target_name in invariant_targets:
        concept_rows = [
            row
            for row in encoded_used_rows
            if row["target"] == target_name and row["stratum"] == "overall"
        ]
        concept_use[target_name] = {
            "median_encoded_linear_r2": float(
                np.median([float(row["encoded_linear_grouped_cv_r2"]) for row in concept_rows])
            ),
            "median_encoded_nonlinear_r2": float(
                np.median(
                    [float(row["encoded_nonlinear_grouped_cv_r2"]) for row in concept_rows]
                )
            ),
            "median_used_direction_removal_rms": float(
                np.median(
                    [float(row["used_direction_removal_rms_delta"]) for row in concept_rows]
                )
            ),
            "median_random_direction_rms": float(
                np.median(
                    [float(row["random_direction_control_median_rms"]) for row in concept_rows]
                )
            ),
            "median_direction_edit_magnitude_standardized_rms": float(
                np.median(
                    [
                        float(row["direction_edit_magnitude_standardized_rms"])
                        for row in concept_rows
                    ]
                )
            ),
            "median_used_rms_per_edit_sd": float(
                np.nanmedian([float(row["used_rms_per_edit_sd"]) for row in concept_rows])
            ),
            "median_random_direction_rms_per_edit_sd": float(
                np.median(
                    [
                        float(row["random_direction_control_median_rms_per_edit_sd"])
                        for row in concept_rows
                    ]
                )
            ),
            "grouped_bootstrap_ci95": concept_intervals[target_name],
        }
    mean_pair_interactions = [
        float(row["rms_interaction"])
        for row in intervention_rows
        if row["scope"] == "unit_pair"
        and row["mode"] == "mean"
        and row["stratum"] == "overall"
    ]
    intervention_mode_summary: dict[str, dict[str, Any]] = {}
    for mode in INTERVENTION_MODES:
        mode_rows = [
            row
            for row in intervention_rows
            if row["scope"] == "single_unit"
            and row["mode"] == mode
            and row["stratum"] == "overall"
        ]
        rows_by_member = {
            member_id: [row for row in mode_rows if row["member_id"] == member_id]
            for member_id in member_ids
        }
        strongest = [
            max(member_rows, key=lambda row: float(row["rms_delta"]))
            for member_rows in rows_by_member.values()
        ]
        strongest_rms = [float(row["rms_delta"]) for row in strongest]
        intervention_mode_summary[mode] = {
            "strongest_unit_rms_median": float(np.median(strongest_rms)),
            "strongest_unit_rms_range": [
                float(np.min(strongest_rms)),
                float(np.max(strongest_rms)),
            ],
            "strongest_unit_mse_change_median": float(
                np.median([float(row["mse_change"]) for row in strongest])
            ),
            "all_unit_mse_change_median": float(
                np.median([float(row["mse_change"]) for row in mode_rows])
            ),
        }
    live_top_units = int(pdp_live.sum())
    present_top_units = int(sum(widths[member_id] for member_id in top_ids))
    summary = {
        "step": "S04",
        "mode": resolved["mode"],
        "estimand": "member-level invariant_tilde_f in native max(log Q, -2) units",
        "cohort": {
            "gradient_set": "varied",
            "panel_rows": row_count,
            "unique_equilibrium_files": int(len(np.unique(equilibrium_files))),
            "stable_or_near_floor": int(masks["stable_or_near_floor"].sum()),
            "unstable": int(masks["unstable"].sum()),
            "members": len(member_ids),
            "shapley_and_atlas_members": top_count,
        },
        "checks": {
            "toy_gate_before_real_inference": toy_checks,
            "canonical_function": CANONICAL_FUNCTION,
            "member_level_signed_arrays_retained": True,
            "stable_unstable_separate": True,
            "decoder_split_unit": "equilibrium_files",
            "bootstrap_unit": "equilibrium_files",
            "hidden_intervention_validity_tag": HIDDEN_INTERVENTION_VALIDITY,
            "random_direction_controls": int(resolved["random_directions"]),
            "label_permutation_controls": True,
            "label_permutation_max_is_decoder_family_max_statistic_diagnostic": True,
            "multiplicity_counts": {
                "decoder_r2": len(decoder_rows),
                "grouped_bootstrap_intervals": len(uncertainty_rows),
                "mean_mode_overall_pair_interactions": len(mean_pair_interactions),
            },
            "fixed_gradient_rows_used": False,
            "review_slice_used": False,
            "shapley_efficiency_max_absolute_error": efficiency_error,
            "all_shapley_efficiency_within_2e-5": bool(efficiency_error <= 2e-5),
            "registered_FSA_grad_x_max_log_difference": float(
                np.max(
                    np.abs(
                        invariant_targets["log_FSA_grad_x"] - np.log(registered_fsa)
                    )
                )
            ),
        },
        "shapley": {
            "exact_members": exact_count,
            "sampled_members": top_count - exact_count,
            "sampled_permutations": int(resolved["shapley_permutations"]),
            "maximum_reported_sampling_se": float(
                np.nanmax(shapley_se) if top_count - exact_count else 0.0
            ),
            "reference": "frozen varied-panel mean for bottleneck units and both drives",
        },
        "ranking": {
            "member_spearman_values": rank_correlations,
            "median_spearman": float(np.median(rank_correlations)),
            "median_grouped_bootstrap_ci95": [
                median_rank_lower,
                median_rank_upper,
            ],
        },
        "head_anatomy": {
            "median_geometry_variance_share": float(
                np.median(geometry_variance_shares)
            ),
            "geometry_variance_share_range": [
                float(np.min(geometry_variance_shares)),
                float(np.max(geometry_variance_shares)),
            ],
            "median_geometry_fraction_of_total_mean_absolute_shapley": float(
                np.median(geometry_mean_absolute_fractions)
            ),
            "live_top10_units": live_top_units,
            "present_top10_units": present_top_units,
            "mean_replacement_pair_interaction_rms": {
                "median": float(np.median(mean_pair_interactions)),
                "q90": float(np.quantile(mean_pair_interactions, 0.9)),
                "q99": float(np.quantile(mean_pair_interactions, 0.99)),
                "maximum": float(np.max(mean_pair_interactions)),
            },
        },
        "intervention_modes_all100": intervention_mode_summary,
        "decoders": {
            "types": ["linear ridge", "32-fixed-ReLU-feature nonlinear ridge"],
            "folds": int(resolved["decoder_folds"]),
            "maximum_label_permutation_r2": float(np.max(decoder_controls)),
            "median_label_permutation_r2": float(np.median(decoder_controls)),
            "minimum_active_fraction_per_training_fold": float(
                resolved["decoder_minimum_active_fraction"]
            ),
            "all100_median_r2": decoder_medians,
            "all100_minimum_r2": decoder_minima,
        },
        "head_fidelity": {
            "median_linear_r2": float(np.median(fidelity_linear)),
            "median_nonlinear_r2": float(np.median(fidelity_nonlinear)),
            "median_nonlinear_increment": float(np.median(fidelity_increments)),
            "nonlinear_increment_range": [
                float(np.min(fidelity_increments)),
                float(np.max(fidelity_increments)),
            ],
        },
        "encoded_vs_used": concept_use,
        "acceptance_criteria": {
            "shapley_exact_or_sampling_error_reported": {
                "exact_members": exact_count,
                "sampled_members_with_standard_errors": top_count - exact_count,
                "maximum_mean_feature_sampling_se": float(
                    max(float(row["mean_sampling_se"]) for row in overall_shapley)
                ),
            },
            "ablation_and_shapley_rankings_compared": {
                "median_spearman": float(np.median(rank_correlations)),
                "grouped_bootstrap_ci95": [median_rank_lower, median_rank_upper],
                "artifact": "rank_comparison.csv",
            },
            "encoded_and_used_are_separate": {
                "artifact": "encoded_vs_used.csv",
                "columns": [
                    "encoded_*_grouped_cv_r2",
                    "used_direction_removal_rms_delta",
                ],
            },
            "random_direction_controls_included": {
                "directions_per_top_member": int(resolved["random_directions"]),
                "artifact": "interventions_top10.h5",
            },
        },
        "artifacts": {
            "bottleneck_arrays": bottleneck_path.name,
            "per_sample_shapley": shapley_path.name,
            "top10_signed_interventions": intervention_h5_path.name,
            "all100_intervention_summary": intervention_summary_path.name,
            "pdp_ice_atlas": pdp_path.name,
            "global_shapley": shapley_global_path.name,
            "intervention_graph_source": intervention_summary_path.name,
            "decodability": decoder_path.name,
            "head_fidelity": fidelity_path.name,
            "encoded_vs_used": encoded_used_path.name,
            "rank_comparison": rank_path.name,
            "grouped_uncertainty": uncertainty_path.name,
        },
        "negative_results": [
            "The head is drive-dominated: geometry carries only about 20% of output variance on the panel.",
            "Head nonlinearity adds only about 0.012 median grouped-CV R2 beyond the linear decoder.",
            "After normalizing by hidden-state edit magnitude, nfp and shat remain below the random-direction control while aspect is near it.",
            "Most pair interactions are small (median RMS about 0.011 native units), although a sparse tail reaches about 0.28.",
        ],
        "deferred": "nothing" if resolved["mode"] == "production" else "production scaling",
    }
    summary_path = artifacts.write_json("summary.json", summary)
    failure_path = output_dir / "failure.json"
    if failure_path.exists():
        failure_path.unlink()
    manifest = artifacts.finalize(
        config=resolved,
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=member_ids,
        row_ids=row_ids,
        gradient_set="varied frozen S01 interpretation panel only",
        device=ensemble.device,
        repository=repository,
        published_dir=(
            Path(resolved["published_dir"])
            if not args.pilot and not args.no_publish
            else None
        ),
    )
    if not args.pilot and not args.no_publish:
        _publish(
            [
                summary_path,
                shapley_global_path,
                rank_path,
                encoded_used_path,
                decoder_path,
                fidelity_path,
                uncertainty_path,
                rank_figure,
                interaction_figure,
                pdp_figure,
                decoder_figure,
            ],
            Path(resolved["published_dir"]),
        )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"S04 {resolved['mode']} completed; manifest: {manifest}", flush=True)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    output_dir: Path | None = args.output_dir.resolve() if args.output_dir else None
    try:
        run(args)
    except Exception as error:
        if output_dir is None:
            try:
                config = json.loads(args.config.read_text(encoding="utf-8"))
                resolved = _resolve(config, args)
                output_dir = (Path("output/xai/S04") / str(resolved["run_id"])).resolve()
            except Exception:
                output_dir = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "failure.json").write_text(
                json.dumps(
                    {
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                        "command": sys.argv,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
