#!/usr/bin/env python3
"""Execute S02 symmetry, invariant-model, density, and bottleneck analyses."""

from __future__ import annotations

import argparse
import csv
import gzip
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

from itg_nn.data import InferenceData, load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.audit import regression_metrics, spearman_correlation
from itg_nn.xai.runtime import iter_inference_batches, set_deterministic_seed
from itg_nn.xai.symmetry import (
    InvariantMember,
    circular_shift,
    normalized_parity_mismatch,
    receptive_field_blocks,
    reverse_parallel,
    stellarator_parity,
)


FUNCTION_NAMES = ("original_f", "shift_averaged_bar_f", "invariant_tilde_f")
CHANNEL_NAMES = (
    "bmag",
    "gbdrift",
    "cvdrift",
    "gbdrift0_over_shat",
    "gds2",
    "gds21_over_shat",
    "gds22_over_shat_squared",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/xai/S02_symmetry.json"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--published-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--phase-chunk", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--members", type=int)
    parser.add_argument("--rows", type=int, help="Reference-row prefix length")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    return parser


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _h5_take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    """Read arbitrary stable row IDs while preserving their registered order."""

    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return dataset[unique][inverse]


def _concatenate_data(first: InferenceData, second: InferenceData) -> InferenceData:
    targets = None
    if first.actual_log_heat_flux is not None and second.actual_log_heat_flux is not None:
        targets = torch.cat((first.actual_log_heat_flux, second.actual_log_heat_flux))
    return InferenceData(
        geometry=torch.cat((first.geometry, second.geometry)),
        a_over_lt=torch.cat((first.a_over_lt, second.a_over_lt)),
        a_over_ln=torch.cat((first.a_over_ln, second.a_over_ln)),
        row_indices=np.concatenate((first.row_indices, second.row_indices)),
        actual_log_heat_flux=targets,
    )


def _transformed_data(data: InferenceData, transform) -> InferenceData:
    return InferenceData(
        geometry=transform(data.geometry),
        a_over_lt=data.a_over_lt,
        a_over_ln=data.a_over_ln,
        row_indices=data.row_indices,
        actual_log_heat_flux=data.actual_log_heat_flux,
    )


def _predict_original(
    member: InvariantMember, data: InferenceData, batch_size: int, device: torch.device
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in iter_inference_batches(data, batch_size):
            outputs.append(
                member.original(
                    batch.geometry.to(device),
                    batch.a_over_lt.to(device),
                    batch.a_over_ln.to(device),
                ).cpu().numpy()
            )
    return np.concatenate(outputs)


def _predict_phases(
    member: InvariantMember,
    data: InferenceData,
    shifts: tuple[int, ...],
    batch_size: int,
    phase_chunk: int,
    device: torch.device,
) -> np.ndarray:
    result = np.empty((len(data.row_indices), len(shifts)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(data.row_indices), batch_size):
            stop = min(start + batch_size, len(data.row_indices))
            geometry = data.geometry[start:stop].to(device)
            a_over_lt = data.a_over_lt[start:stop].to(device)
            a_over_ln = data.a_over_ln[start:stop].to(device)
            for phase_start in range(0, len(shifts), phase_chunk):
                phase_stop = min(phase_start + phase_chunk, len(shifts))
                selected = shifts[phase_start:phase_stop]
                shifted = torch.cat(
                    [circular_shift(geometry, phase) for phase in selected], dim=0
                )
                repeat = len(selected)
                output = member.original(
                    shifted, a_over_lt.repeat(repeat), a_over_ln.repeat(repeat)
                )
                result[start:stop, phase_start:phase_stop] = (
                    output.reshape(repeat, stop - start).transpose(0, 1).cpu().numpy()
                )
    return result


def _predict_invariant(
    member: InvariantMember, data: InferenceData, batch_size: int, device: torch.device
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in iter_inference_batches(data, batch_size):
            outputs.append(
                member.invariant(
                    batch.geometry.to(device),
                    batch.a_over_lt.to(device),
                    batch.a_over_ln.to(device),
                ).cpu().numpy()
            )
    return np.concatenate(outputs)


def _change_metrics(reference: np.ndarray, changed: np.ndarray) -> dict[str, float]:
    difference = np.asarray(changed, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "mean_signed_change": float(np.mean(difference)),
        "mean_absolute_change": float(np.mean(np.abs(difference))),
        "rms_change": float(np.sqrt(np.mean(np.square(difference)))),
        "max_absolute_change": float(np.max(np.abs(difference))),
        "relative_l2_change": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(np.asarray(reference, dtype=np.float64)), np.finfo(float).tiny)
        ),
    }


def _accuracy_rows(
    actual: np.ndarray,
    predictions: np.ndarray,
    member_ids: tuple[str, ...],
    timings: np.ndarray,
    stable_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stability = {
        "all": np.ones(len(actual), dtype=bool),
        "stable_near_floor": actual <= stable_threshold,
        "unstable": actual > stable_threshold,
    }
    entities = [
        (member_id, predictions[index], timings[index], "single_member")
        for index, member_id in enumerate(member_ids)
    ]
    entities.append(
        (
            "ensemble_mean",
            predictions.mean(axis=0),
            timings.sum(axis=0),
            "sequential_all_members",
        )
    )
    for entity, values, cost, cost_scope in entities:
        for function_index, function_name in enumerate(FUNCTION_NAMES):
            for stratum, mask in stability.items():
                metrics = regression_metrics(actual[mask], values[function_index, mask])
                residual = values[function_index, mask].astype(np.float64) - actual[mask]
                rows.append(
                    {
                        "entity": entity,
                        "function": function_name,
                        "stratum": stratum,
                        **metrics,
                        "residual_std": float(np.std(residual, ddof=1)),
                        "cost_scope": cost_scope,
                        "wall_seconds": float(cost[function_index]),
                        "microseconds_per_sample": float(cost[function_index] * 1e6 / len(actual)),
                    }
                )
    return rows


def _grouped_bootstrap_summary(
    actual: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired equilibrium bootstrap for ensemble function differences."""

    unique, inverse = np.unique(groups, return_inverse=True)
    group_rows = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    generator = np.random.default_rng(seed)
    values = {"bar_f_minus_f_r2": [], "tilde_f_minus_f_r2": [], "bar_f_minus_f_residual_std": [], "tilde_f_minus_f_residual_std": []}
    ensemble = predictions.mean(axis=0)
    for _ in range(replicates):
        chosen = generator.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([group_rows[index] for index in chosen])
        y = actual[indices]
        metrics = [regression_metrics(y, ensemble[f, indices])["r2"] for f in range(3)]
        residual_std = [float(np.std(ensemble[f, indices] - y, ddof=1)) for f in range(3)]
        values["bar_f_minus_f_r2"].append(metrics[1] - metrics[0])
        values["tilde_f_minus_f_r2"].append(metrics[2] - metrics[0])
        values["bar_f_minus_f_residual_std"].append(residual_std[1] - residual_std[0])
        values["tilde_f_minus_f_residual_std"].append(residual_std[2] - residual_std[0])
    return {
        "unit": "equilibrium_files",
        "groups": int(len(unique)),
        "replicates": int(replicates),
        **{
            name: {
                "median": float(np.median(samples)),
                "ci_95": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
            }
            for name, samples in values.items()
        },
    }


def _member_grouped_bootstrap_rows(
    actual: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    member_ids: tuple[str, ...],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Paired grouped intervals for each member's residual-std change."""

    unique, inverse = np.unique(groups, return_inverse=True)
    residual = predictions.astype(np.float64) - actual[None, None, :]
    flat = residual.reshape(-1, residual.shape[-1])
    group_sum = np.zeros((flat.shape[0], len(unique)), dtype=np.float64)
    group_sum_squares = np.zeros_like(group_sum)
    row_index = np.broadcast_to(inverse, flat.shape)
    np.add.at(group_sum, (np.arange(flat.shape[0])[:, None], row_index), flat)
    np.add.at(
        group_sum_squares,
        (np.arange(flat.shape[0])[:, None], row_index),
        np.square(flat),
    )
    group_sizes = np.bincount(inverse, minlength=len(unique)).astype(np.float64)

    generator = np.random.default_rng(seed)
    weights = np.empty((len(unique), replicates), dtype=np.float64)
    for replicate in range(replicates):
        chosen = generator.integers(0, len(unique), size=len(unique))
        weights[:, replicate] = np.bincount(chosen, minlength=len(unique))
    sample_sizes = group_sizes @ weights
    sums = group_sum @ weights
    sums_squares = group_sum_squares @ weights
    variance = (sums_squares - np.square(sums) / sample_sizes) / (sample_sizes - 1)
    standard_deviation = np.sqrt(np.maximum(variance, 0)).reshape(
        len(member_ids), len(FUNCTION_NAMES), replicates
    )

    original_point = np.std(residual[:, 0], axis=1, ddof=1)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "unit": "equilibrium_files",
        "groups": int(len(unique)),
        "replicates": int(replicates),
    }
    for function_index, function_name in enumerate(FUNCTION_NAMES[1:], start=1):
        deltas = standard_deviation[:, function_index] - standard_deviation[:, 0]
        point = np.std(residual[:, function_index], axis=1, ddof=1) - original_point
        lower, upper = np.quantile(deltas, (0.025, 0.975), axis=1)
        probability_improved = np.mean(deltas < 0, axis=1)
        for member_index, member_id in enumerate(member_ids):
            rows.append(
                {
                    "member_id": member_id,
                    "function": function_name,
                    "point_delta_residual_std_vs_f": float(point[member_index]),
                    "bootstrap_median_delta": float(np.median(deltas[member_index])),
                    "ci_95_lower": float(lower[member_index]),
                    "ci_95_upper": float(upper[member_index]),
                    "probability_improved": float(probability_improved[member_index]),
                }
            )
        summary[function_name] = {
            "point_improved_members": int(np.sum(point < 0)),
            "members_with_ci_upper_below_zero": int(np.sum(upper < 0)),
            "probability_improved_minimum": float(np.min(probability_improved)),
            "probability_improved_median": float(np.median(probability_improved)),
        }
    return rows, summary


def _panel_strata(
    data: InferenceData, varied_count: int, stable_threshold: float
) -> dict[tuple[str, str], np.ndarray]:
    """Return non-pooled gradient-set and stability masks for the paired panel."""

    if data.actual_log_heat_flux is None:
        raise RuntimeError("panel targets unavailable")
    actual = data.actual_log_heat_flux.numpy()
    gradient_masks = {
        "varied": np.arange(len(actual)) < varied_count,
        "fixed": np.arange(len(actual)) >= varied_count,
    }
    strata: dict[tuple[str, str], np.ndarray] = {}
    for gradient_set, gradient_mask in gradient_masks.items():
        strata[(gradient_set, "all")] = gradient_mask
        strata[(gradient_set, "stable_near_floor")] = gradient_mask & (
            actual <= stable_threshold
        )
        strata[(gradient_set, "unstable")] = gradient_mask & (
            actual > stable_threshold
        )
    return strata


def _initialize_prediction_file(
    path: Path,
    member_ids: tuple[str, ...],
    panel_data: InferenceData,
    reference_data: InferenceData,
    source_hashes: dict[str, str],
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(temporary, "w") as h5_file:
        h5_file.attrs["member_ids"] = json.dumps(member_ids)
        h5_file.attrs["dataset_sha256"] = source_hashes["dataset"]
        h5_file.attrs["checkpoint_sha256"] = source_hashes["checkpoint"]
        h5_file.attrs["estimand"] = "native max(log Q, -2)"
        h5_file.create_dataset("member_id", data=np.asarray(member_ids, dtype="S"))
        h5_file.create_dataset("panel_row_id", data=panel_data.row_indices)
        h5_file.create_dataset("reference_row_id", data=reference_data.row_indices)
        h5_file.create_dataset("complete", data=np.zeros(len(member_ids), dtype=bool))
        h5_file.create_dataset(
            "panel_shift_prediction", shape=(len(member_ids), len(panel_data.row_indices), 96), dtype="f4"
        )
        h5_file.create_dataset(
            "reference_prediction", shape=(len(member_ids), 3, len(reference_data.row_indices)), dtype="f4"
        )
        h5_file.create_dataset(
            "parity_prediction", shape=(len(member_ids), 3, len(panel_data.row_indices)), dtype="f4"
        )
        h5_file.create_dataset("wall_seconds", shape=(len(member_ids), 3), dtype="f8")
        h5_file["panel_shift_prediction"].attrs["axes"] = json.dumps(["member", "sample", "shift"])
        h5_file["reference_prediction"].attrs["axes"] = json.dumps(["member", "function", "sample"])
        h5_file["parity_prediction"].attrs["axes"] = json.dumps(["member", "transform", "sample"])
    temporary.replace(path)
    return h5py.File(path, "r+")


def _open_prediction_file(
    path: Path,
    member_ids: tuple[str, ...],
    panel_data: InferenceData,
    reference_data: InferenceData,
    source_hashes: dict[str, str],
    resume: bool,
) -> h5py.File:
    if not path.exists():
        return _initialize_prediction_file(path, member_ids, panel_data, reference_data, source_hashes)
    if not resume:
        raise FileExistsError(f"{path} exists; pass --resume or select a new --output-dir")
    h5_file = h5py.File(path, "r+")
    checks = (
        (json.loads(h5_file.attrs["member_ids"]) == list(member_ids), "member IDs"),
        (np.array_equal(h5_file["panel_row_id"][:], panel_data.row_indices), "panel rows"),
        (np.array_equal(h5_file["reference_row_id"][:], reference_data.row_indices), "reference rows"),
        (h5_file.attrs["dataset_sha256"] == source_hashes["dataset"], "dataset hash"),
        (h5_file.attrs["checkpoint_sha256"] == source_hashes["checkpoint"], "checkpoint hash"),
    )
    failed = [name for passed, name in checks if not passed]
    if failed:
        h5_file.close()
        raise RuntimeError(f"resume artifact mismatch: {', '.join(failed)}")
    return h5_file


def _density_census_and_checks(
    ensemble,
    member_ids: tuple[str, ...],
    panel_data: InferenceData,
    validation: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    unit_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    index_by_id = {member_id: index for index, member_id in enumerate(ensemble.member_ids)}
    check_count = min(int(config["density_check_rows"]), len(panel_data.row_indices))
    check_geometry = panel_data.geometry[:check_count].to(ensemble.device)
    dead_counts: list[int] = []
    near_dead_counts: list[int] = []
    original_dead_counts: list[int] = []
    original_near_dead_counts: list[int] = []
    widths: list[int] = []
    with torch.inference_mode():
        for rank, member_id in enumerate(member_ids, start=1):
            wrapped = InvariantMember(ensemble.models[index_by_id[member_id]])
            values: list[np.ndarray] = []
            original_values: list[np.ndarray] = []
            for batch in iter_inference_batches(panel_data, int(config["batch_size"])):
                geometry = batch.geometry.to(ensemble.device)
                values.append(wrapped.equivariant_density(geometry).cpu().numpy())
                original_values.append(wrapped.bottleneck(geometry).cpu().numpy())
            density = np.concatenate(values, axis=0)
            original_bottleneck = np.concatenate(original_values, axis=0)
            width = density.shape[1]
            widths.append(width)
            dead_count = 0
            near_dead_count = 0
            original_dead_count = 0
            original_near_dead_count = 0
            for unit in range(width):
                flat = density[:, unit, :].astype(np.float64).ravel()
                original_flat = original_bottleneck[:, unit].astype(np.float64)
                active_fraction = float(np.mean(flat > 0))
                dead = bool(np.max(np.abs(flat)) <= 1e-12)
                near_dead = bool(not dead and active_fraction < float(config["near_dead_active_fraction"]))
                original_active_fraction = float(np.mean(original_flat > 0))
                original_dead = bool(np.max(np.abs(original_flat)) <= 1e-12)
                original_near_dead = bool(
                    not original_dead
                    and original_active_fraction
                    < float(config["near_dead_active_fraction"])
                )
                dead_count += int(dead)
                near_dead_count += int(near_dead)
                original_dead_count += int(original_dead)
                original_near_dead_count += int(original_near_dead)
                unit_rows.append(
                    {
                        "member_id": member_id,
                        "stored_validation_rank": rank,
                        "stored_validation_r2": validation[rank - 1],
                        "unit": unit,
                        "dead": dead,
                        "near_dead": near_dead,
                        "active_fraction": active_fraction,
                        "zero_fraction": float(np.mean(flat == 0)),
                        "mean": float(np.mean(flat)),
                        "std": float(np.std(flat, ddof=1)),
                        "q01": float(np.quantile(flat, 0.01)),
                        "median": float(np.median(flat)),
                        "q99": float(np.quantile(flat, 0.99)),
                        "maximum": float(np.max(flat)),
                        "original_dead": original_dead,
                        "original_near_dead": original_near_dead,
                        "original_active_fraction": original_active_fraction,
                        "original_mean": float(np.mean(original_flat)),
                        "original_std": float(np.std(original_flat, ddof=1)),
                        "original_q01": float(np.quantile(original_flat, 0.01)),
                        "original_median": float(np.median(original_flat)),
                        "original_q99": float(np.quantile(original_flat, 0.99)),
                        "original_maximum": float(np.max(original_flat)),
                    }
                )
            dead_counts.append(dead_count)
            near_dead_counts.append(near_dead_count)
            original_dead_counts.append(original_dead_count)
            original_near_dead_counts.append(original_near_dead_count)

            density_check = wrapped.equivariant_density(check_geometry)
            original_map = wrapped.bottleneck_map(check_geometry)
            phase_mean = torch.stack(
                [wrapped.bottleneck(circular_shift(check_geometry, phase)) for phase in range(32)]
            ).mean(dim=0)
            shifted_density = wrapped.equivariant_density(circular_shift(check_geometry, 1))
            mean_error = torch.abs(density_check.mean(-1) - phase_mean)
            equivariance_error = torch.abs(shifted_density - torch.roll(density_check, 1, -1))
            alignment_error = torch.abs(density_check[..., ::32] - original_map)
            atol = float(config["exact_atol"])
            rtol = float(config["exact_rtol"])
            check_rows.append(
                {
                    "member_id": member_id,
                    "mean_identity_max_abs": float(mean_error.max().cpu()),
                    "mean_identity_relative_l2": float(
                        torch.linalg.vector_norm(mean_error)
                        / torch.linalg.vector_norm(phase_mean).clamp_min(torch.finfo(phase_mean.dtype).tiny)
                    ),
                    "equivariance_max_abs": float(equivariance_error.max().cpu()),
                    "equivariance_relative_l2": float(
                        torch.linalg.vector_norm(equivariance_error)
                        / torch.linalg.vector_norm(density_check).clamp_min(torch.finfo(density_check.dtype).tiny)
                    ),
                    "alignment_max_abs": float(alignment_error.max().cpu()),
                    "alignment_relative_l2": float(
                        torch.linalg.vector_norm(alignment_error)
                        / torch.linalg.vector_norm(original_map).clamp_min(
                            torch.finfo(original_map.dtype).tiny
                        )
                    ),
                    "mean_identity_pass": bool(
                        torch.allclose(
                            density_check.mean(-1), phase_mean, atol=atol, rtol=rtol
                        )
                    ),
                    "equivariance_pass": bool(
                        torch.allclose(
                            shifted_density,
                            torch.roll(density_check, 1, -1),
                            atol=atol,
                            rtol=rtol,
                        )
                    ),
                    "alignment_pass": bool(
                        torch.allclose(
                            density_check[..., ::32], original_map, atol=atol, rtol=rtol
                        )
                    ),
                }
            )
            print(f"density census complete: {rank}/{len(member_ids)} {member_id}", flush=True)
    def correlation(values: list[int]) -> float | None:
        return None if len(set(values)) < 2 else spearman_correlation(np.asarray(values), validation)
    summary = {
        "dead_definition": "maximum absolute rho on the S01 panel <= 1e-12",
        "near_dead_definition": f"non-dead with active rho fraction < {config['near_dead_active_fraction']}",
        "width_range": [int(min(widths)), int(max(widths))],
        "dead_unit_count_total": int(sum(dead_counts)),
        "near_dead_unit_count_total": int(sum(near_dead_counts)),
        "original_dead_definition": "maximum absolute native bottleneck u on the S01 panel <= 1e-12",
        "original_near_dead_definition": f"non-dead native u with active sample fraction < {config['near_dead_active_fraction']}",
        "original_dead_unit_count_total": int(sum(original_dead_counts)),
        "original_near_dead_unit_count_total": int(sum(original_near_dead_counts)),
        "spearman_with_stored_validation_r2": {
            "width": correlation(widths),
            "dead_count": correlation(dead_counts),
            "near_dead_count": correlation(near_dead_counts),
            "original_dead_count": correlation(original_dead_counts),
            "original_near_dead_count": correlation(original_near_dead_counts),
        },
    }
    return unit_rows, check_rows, summary


def _plot_shift(path: Path, symmetry_rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(figsize=(7.2, 3.8))
    member_rows = [
        row
        for row in symmetry_rows
        if row["entity_type"] == "member"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "all"
    ]
    shifts = np.arange(96)
    rows_by_member: dict[str, list[dict[str, Any]]] = {}
    for row in member_rows:
        rows_by_member.setdefault(row["entity"], []).append(row)
    matrix = np.asarray(
        [
            [row["rms_change_over_residual_std"] for row in rows_by_member[member]]
            for member in rows_by_member
        ]
    )
    axes.plot(shifts, np.median(matrix, axis=0), color="black", label="member median")
    axes.fill_between(shifts, np.quantile(matrix, 0.1, axis=0), np.quantile(matrix, 0.9, axis=0), alpha=0.25, label="10–90% members")
    axes.axvline(32, color="tab:red", linestyle="--", linewidth=0.8)
    axes.axvline(64, color="tab:red", linestyle="--", linewidth=0.8)
    axes.set(
        xlabel="Circular shift (grid points)",
        ylabel="RMS change / own varied-reference residual std",
    )
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_accuracy(path: Path, accuracy_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in accuracy_rows if row["entity"] != "ensemble_mean" and row["stratum"] == "all"]
    by_function = {name: np.asarray([row["residual_std"] for row in rows if row["function"] == name]) for name in FUNCTION_NAMES}
    figure, axes = plt.subplots(figsize=(5.8, 3.8))
    axes.boxplot(
        [by_function[name] for name in FUNCTION_NAMES],
        labels=["f", "bar f", "tilde f"],
        showfliers=False,
    )
    axes.set_ylabel("Reference residual std (clipped-log units)")
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _publish(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if any(path.name == "shift_symmetry_summary.csv" for path in paths):
        (destination / "shift_symmetry.csv").unlink(missing_ok=True)
        (destination / "shift_symmetry.csv.gz").unlink(missing_ok=True)
    for source in paths:
        target = destination / source.name
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)


def _write_gzip_csv(
    artifacts: RunArtifacts, name: str, rows: list[dict[str, Any]]
) -> Path:
    """Write a compressed CSV atomically and register it in the run manifest."""

    path = artifacts.output_dir / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
        stream.write(_csv_text(rows))
    temporary.replace(path)
    return artifacts.register_existing(name)


def _shift_summary_rows(symmetry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact member quantiles and ensemble rows for the committed report."""

    result: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in symmetry_rows:
        key = (row["gradient_set"], row["stratum"], row["n"], row["shift"])
        grouped.setdefault(key, []).append(row)
    for (gradient_set, stratum, count, shift), selected in grouped.items():
        members = [row for row in selected if row["entity_type"] == "member"]
        member_rms = np.asarray([row["rms_change"] for row in members])
        member_ratio = np.asarray(
            [
                row["rms_change_over_residual_std"]
                for row in members
                if row["rms_change_over_residual_std"] is not None
            ]
        )
        base = {
            "gradient_set": gradient_set,
            "stratum": stratum,
            "n": count,
            "shift": shift,
            "exact_pooling_subgroup": shift in (0, 32, 64),
        }
        result.append(
            {
                **base,
                "entity": "member_distribution",
                "rms_change_q10": float(np.quantile(member_rms, 0.1)),
                "rms_change_median": float(np.median(member_rms)),
                "rms_change_q90": float(np.quantile(member_rms, 0.9)),
                "rms_over_residual_std_q10": (
                    float(np.quantile(member_ratio, 0.1))
                    if len(member_ratio)
                    else None
                ),
                "rms_over_residual_std_median": (
                    float(np.median(member_ratio)) if len(member_ratio) else None
                ),
                "rms_over_residual_std_q90": (
                    float(np.quantile(member_ratio, 0.9))
                    if len(member_ratio)
                    else None
                ),
            }
        )
        for row in selected:
            if row["entity_type"] != "ensemble":
                continue
            ratio = row["rms_change_over_residual_std"]
            result.append(
                {
                    **base,
                    "entity": row["entity"],
                    "rms_change_q10": row["rms_change"],
                    "rms_change_median": row["rms_change"],
                    "rms_change_q90": row["rms_change"],
                    "rms_over_residual_std_q10": ratio,
                    "rms_over_residual_std_median": ratio,
                    "rms_over_residual_std_q90": ratio,
                }
            )
    return result


def run(config: dict[str, Any], args: argparse.Namespace) -> Path:
    resolved = dict(config)
    pilot = bool(args.pilot)
    if pilot:
        resolved.update(config["pilot"])
        resolved["mode"] = "pilot"
    else:
        resolved["mode"] = "registered_production"
    resolved["dataset"] = str((args.dataset or Path(config["dataset"])).resolve())
    resolved["checkpoint"] = str((args.checkpoint or Path(config["checkpoint"])).resolve())
    resolved["cohorts"] = str((args.cohorts or Path(config["cohorts"])).resolve())
    resolved["published_dir"] = str((args.published_dir or Path(config["published_dir"])).resolve())
    resolved["device"] = args.device or config["device"]
    resolved["batch_size"] = args.batch_size or int(config["batch_size"])
    resolved["phase_chunk"] = args.phase_chunk or int(config["phase_chunk"])
    resolved["seed"] = args.seed if args.seed is not None else int(config["seed"])
    if args.members is not None:
        resolved["members"] = args.members
    if args.rows is not None:
        resolved["reference_rows"] = args.rows
    resolved["resume"] = bool(args.resume)
    if int(resolved.get("members", 100)) not in range(1, 101):
        raise ValueError("--members must be in [1, 100]")
    if int(resolved.get("reference_rows", 1)) < 1:
        raise ValueError("--rows must be positive")
    if int(resolved["phase_chunk"]) < 1 or int(resolved["phase_chunk"]) > 96:
        raise ValueError("--phase-chunk must be in [1, 96]")

    dataset = Path(resolved["dataset"])
    checkpoint = Path(resolved["checkpoint"])
    cohorts_path = Path(resolved["cohorts"])
    output_dir = (args.output_dir or Path("output/xai/S02") / str(resolved["run_id"])).resolve()
    artifacts = RunArtifacts(output_dir)
    set_deterministic_seed(int(resolved["seed"]))
    cohorts = json.loads(cohorts_path.read_text(encoding="utf-8"))
    panel_rows = np.asarray(cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64)
    if "panel_varied_rows" in resolved:
        panel_rows = panel_rows[: int(resolved["panel_varied_rows"])]
    reference_rows = np.asarray(cohorts["reference_varied"]["row_ids"], dtype=np.int64)
    if "reference_rows" in resolved:
        reference_rows = reference_rows[: int(resolved["reference_rows"])]
    varied_panel = load_hdf5_rows(dataset, panel_rows, gradient_set="varied", include_targets=True)
    fixed_panel = load_hdf5_rows(dataset, panel_rows, gradient_set="fixed", include_targets=True)
    panel_data = _concatenate_data(varied_panel, fixed_panel)
    reference_data = load_hdf5_rows(dataset, reference_rows, gradient_set="varied", include_targets=True)
    if reference_data.actual_log_heat_flux is None:
        raise RuntimeError("reference targets unavailable")

    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    ranked = sorted(bundle["members"], key=lambda member: (-float(member["validation_r2"]), str(member["id"])))
    ranked = ranked[: int(resolved.get("members", len(ranked)))]
    member_ids = tuple(str(member["id"]) for member in ranked)
    validation = np.asarray([float(member["validation_r2"]) for member in ranked])
    ensemble = load_ensemble(checkpoint, device=resolved["device"])
    index_by_id = {member_id: index for index, member_id in enumerate(ensemble.member_ids)}
    source_hashes = {"dataset": sha256_file(dataset), "checkpoint": sha256_file(checkpoint)}
    prediction_path = output_dir / "predictions.h5"
    h5_file = _open_prediction_file(prediction_path, member_ids, panel_data, reference_data, source_hashes, bool(args.resume))
    try:
        for member_index, member_id in enumerate(member_ids):
            if bool(h5_file["complete"][member_index]):
                print(f"resumed completed member {member_index + 1}/{len(member_ids)} {member_id}", flush=True)
                continue
            wrapped = InvariantMember(ensemble.models[index_by_id[member_id]])
            start = time.perf_counter()
            original = _predict_original(wrapped, reference_data, int(resolved["batch_size"]), ensemble.device)
            original_seconds = time.perf_counter() - start
            start = time.perf_counter()
            reference_phases = _predict_phases(wrapped, reference_data, tuple(range(32)), int(resolved["batch_size"]), int(resolved["phase_chunk"]), ensemble.device)
            bar_f = reference_phases.mean(axis=1)
            bar_seconds = time.perf_counter() - start
            start = time.perf_counter()
            tilde_f = _predict_invariant(wrapped, reference_data, int(resolved["batch_size"]), ensemble.device)
            tilde_seconds = time.perf_counter() - start
            panel_shifts = _predict_phases(wrapped, panel_data, tuple(range(96)), int(resolved["batch_size"]), int(resolved["phase_chunk"]), ensemble.device)
            parity = _predict_original(wrapped, _transformed_data(panel_data, stellarator_parity), int(resolved["batch_size"]), ensemble.device)
            wrong_parity = _predict_original(wrapped, _transformed_data(panel_data, reverse_parallel), int(resolved["batch_size"]), ensemble.device)
            h5_file["reference_prediction"][member_index] = np.stack((original, bar_f, tilde_f))
            h5_file["panel_shift_prediction"][member_index] = panel_shifts
            h5_file["parity_prediction"][member_index] = np.stack((panel_shifts[:, 0], parity, wrong_parity))
            h5_file["wall_seconds"][member_index] = (original_seconds, bar_seconds, tilde_seconds)
            h5_file["complete"][member_index] = True
            h5_file.flush()
            elapsed = original_seconds + bar_seconds + tilde_seconds
            print(f"member complete: {member_index + 1}/{len(member_ids)} {member_id}; reference functions {elapsed:.2f}s", flush=True)
    finally:
        h5_file.close()

    with h5py.File(prediction_path, "r") as completed:
        if not np.all(completed["complete"][:]):
            raise RuntimeError("prediction artifact is incomplete")
        panel_shift_prediction = completed["panel_shift_prediction"][:]
        reference_prediction = completed["reference_prediction"][:]
        parity_prediction = completed["parity_prediction"][:]
        timings = completed["wall_seconds"][:]
    actual = reference_data.actual_log_heat_flux.numpy()
    accuracy_rows = _accuracy_rows(actual, reference_prediction, member_ids, timings, float(resolved["stable_threshold_log_Q"]))
    residual_std = {
        (row["entity"], row["stratum"]): row["residual_std"]
        for row in accuracy_rows
        if row["function"] == "original_f"
    }
    panel_strata = _panel_strata(
        panel_data, len(panel_rows), float(resolved["stable_threshold_log_Q"])
    )

    symmetry_rows: list[dict[str, Any]] = []
    symmetry_entities = [(member_id, panel_shift_prediction[index], "member") for index, member_id in enumerate(member_ids)]
    symmetry_entities.extend((
        ("ensemble_mean", panel_shift_prediction.mean(axis=0), "ensemble"),
        ("ensemble_spread", panel_shift_prediction.std(axis=0), "ensemble"),
    ))
    for entity, values, entity_type in symmetry_entities:
        for (gradient_set, stratum), mask in panel_strata.items():
            for shift in range(96):
                metrics = _change_metrics(values[mask, 0], values[mask, shift])
                own_residual = (
                    residual_std.get((entity, stratum))
                    if gradient_set == "varied"
                    else None
                )
                symmetry_rows.append(
                    {
                        "entity": entity,
                        "entity_type": entity_type,
                        "gradient_set": gradient_set,
                        "stratum": stratum,
                        "n": int(np.sum(mask)),
                        "shift": shift,
                        "exact_pooling_subgroup": shift in (0, 32, 64),
                        **metrics,
                        "original_reference_residual_std": own_residual,
                        "rms_change_over_residual_std": (
                            metrics["rms_change"] / own_residual
                            if own_residual
                            else None
                        ),
                    }
                )

    phase_rows: list[dict[str, Any]] = []
    for entity, values, entity_type in symmetry_entities:
        for (gradient_set, stratum), mask in panel_strata.items():
            metrics = _change_metrics(
                values[mask, :32].mean(axis=1), values[mask].mean(axis=1)
            )
            phase_rows.append(
                {
                    "entity": entity,
                    "entity_type": entity_type,
                    "gradient_set": gradient_set,
                    "stratum": stratum,
                    "n": int(np.sum(mask)),
                    **metrics,
                }
            )

    parity_rows: list[dict[str, Any]] = []
    parity_entities = [(member_id, parity_prediction[index], "member") for index, member_id in enumerate(member_ids)]
    parity_entities.extend((
        ("ensemble_mean", parity_prediction.mean(axis=0), "ensemble"),
        ("ensemble_spread", parity_prediction.std(axis=0), "ensemble"),
    ))
    for entity, values, entity_type in parity_entities:
        for (gradient_set, stratum), mask in panel_strata.items():
            for transform_index, transform_name in (
                (1, "stellarator_parity"),
                (2, "plain_reversal_control"),
            ):
                metrics = _change_metrics(values[0, mask], values[transform_index, mask])
                own_residual = (
                    residual_std.get((entity, stratum))
                    if gradient_set == "varied"
                    else None
                )
                parity_rows.append(
                    {
                        "entity": entity,
                        "entity_type": entity_type,
                        "gradient_set": gradient_set,
                        "stratum": stratum,
                        "n": int(np.sum(mask)),
                        "transform": transform_name,
                        **metrics,
                        "original_reference_residual_std": own_residual,
                        "rms_change_over_residual_std": (
                            metrics["rms_change"] / own_residual
                            if own_residual
                            else None
                        ),
                    }
                )

    unit_rows, density_rows, census_summary = _density_census_and_checks(ensemble, member_ids, panel_data, validation, resolved)
    rf_rows: list[dict[str, Any]] = []
    for rank, member in enumerate(ranked, start=1):
        for block in receptive_field_blocks(tuple(member["architecture"]["kernel_sizes"])):
            rf_rows.append({"member_id": str(member["id"]), "stored_validation_rank": rank, "stored_validation_r2": float(member["validation_r2"]), **block.__dict__})

    with h5py.File(dataset, "r") as source:
        equilibrium_bytes = _h5_take(source["equilibrium_files"], reference_rows)
        equilibrium = np.asarray(
            [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in equilibrium_bytes]
        )
    correct_mismatch = normalized_parity_mismatch(reference_data.geometry.numpy())
    reversed_geometry = reverse_parallel(reference_data.geometry).numpy()
    original_geometry = reference_data.geometry.numpy().astype(np.float64)
    wrong_mismatch = np.mean(np.square(reversed_geometry - original_geometry), axis=(0, 1)) / np.maximum(np.var(original_geometry, axis=(0, 1)), np.finfo(float).tiny)
    parity_data_rows = [
        {
            "cohort": "varied_reference",
            "n": len(reference_rows),
            "channel": index,
            "channel_name": CHANNEL_NAMES[index],
            "stellarator_parity_normalized_mse": float(correct_mismatch[index]),
            "plain_reversal_normalized_mse": float(wrong_mismatch[index]),
        }
        for index in range(7)
    ]

    exact_rows = [
        row
        for row in symmetry_rows
        if row["entity_type"] == "member"
        and row["stratum"] == "all"
        and row["shift"] in (32, 64)
    ]
    arbitrary_rows = [
        row
        for row in symmetry_rows
        if row["entity_type"] == "member"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "all"
        and row["shift"] not in (0, 32, 64)
    ]
    exact_tolerance = float(resolved["exact_atol"])
    relative_tolerance = float(resolved["exact_rtol"])
    subgroup_pass = bool(
        np.allclose(
            panel_shift_prediction[:, :, (0,)],
            panel_shift_prediction[:, :, (32, 64)],
            atol=exact_tolerance,
            rtol=relative_tolerance,
        )
    )
    phase_pass = bool(
        np.allclose(
            panel_shift_prediction[:, :, :32].mean(axis=2),
            panel_shift_prediction.mean(axis=2),
            atol=exact_tolerance,
            rtol=relative_tolerance,
        )
    )
    checks = {
        "exact_subgroup_max_abs": float(max(row["max_absolute_change"] for row in exact_rows)),
        "exact_subgroup_pass": subgroup_pass,
        "phase_32_vs_96_max_abs": float(max(row["max_absolute_change"] for row in phase_rows)),
        "phase_32_vs_96_pass": phase_pass,
        "density_mean_identity_max_abs": float(max(row["mean_identity_max_abs"] for row in density_rows)),
        "density_equivariance_max_abs": float(max(row["equivariance_max_abs"] for row in density_rows)),
        "density_alignment_max_abs": float(max(row["alignment_max_abs"] for row in density_rows)),
        "density_mean_identity_pass": bool(
            all(row["mean_identity_pass"] for row in density_rows)
        ),
        "density_equivariance_pass": bool(
            all(row["equivariance_pass"] for row in density_rows)
        ),
        "density_alignment_pass": bool(
            all(row["alignment_pass"] for row in density_rows)
        ),
    }
    checks["density_exactness_pass"] = bool(
        checks["density_mean_identity_pass"]
        and checks["density_equivariance_pass"]
        and checks["density_alignment_pass"]
    )
    if not all((checks["exact_subgroup_pass"], checks["phase_32_vs_96_pass"], checks["density_exactness_pass"])):
        raise RuntimeError(f"registered exactness check failed: {checks}")

    reference_ensemble = reference_prediction.mean(axis=0)
    member_bootstrap_rows, member_bootstrap_summary = _member_grouped_bootstrap_rows(
        actual,
        reference_prediction,
        equilibrium,
        member_ids,
        int(resolved["bootstrap_replicates"]),
        int(resolved["seed"]) + 2,
    )
    ensemble_accuracy = {
        function: next(row for row in accuracy_rows if row["entity"] == "ensemble_mean" and row["function"] == function and row["stratum"] == "all")
        for function in FUNCTION_NAMES
    }
    summary = {
        "mode": resolved["mode"],
        "estimand": "native max(log Q, -2)",
        "members": len(member_ids),
        "panel_samples": len(panel_data.row_indices),
        "panel_varied_rows": len(panel_rows),
        "panel_fixed_rows": len(panel_rows),
        "reference_varied_rows": len(reference_rows),
        "checks": checks,
        "arbitrary_shift_member_rms_over_residual_std": {
            "median": float(np.median([row["rms_change_over_residual_std"] for row in arbitrary_rows])),
            "q10": float(np.quantile([row["rms_change_over_residual_std"] for row in arbitrary_rows], 0.1)),
            "q90": float(np.quantile([row["rms_change_over_residual_std"] for row in arbitrary_rows], 0.9)),
        },
        "ensemble_accuracy": ensemble_accuracy,
        "paired_grouped_bootstrap": _grouped_bootstrap_summary(actual, reference_prediction, equilibrium, int(resolved["bootstrap_replicates"]), int(resolved["seed"]) + 1),
        "paired_member_grouped_bootstrap": member_bootstrap_summary,
        "bottleneck_census": census_summary,
        "canonical_decision_status": resolved["canonical_decision"],
        "canonical_function": resolved["canonical_function"],
        "canonical_decision_basis": (
            "tilde_f is exactly invariant, supplies mean_z(rho)=bar_u, improves "
            "all 100 individual-member residual standard deviations in the registered "
            "production run, and has no resolved ensemble-accuracy penalty under the "
            "grouped bootstrap"
        ),
        "reference_prediction_standard_deviation": {FUNCTION_NAMES[index]: float(np.std(reference_ensemble[index], ddof=1)) for index in range(3)},
    }

    (artifacts.output_dir / "shift_symmetry.csv").unlink(missing_ok=True)
    _write_gzip_csv(artifacts, "shift_symmetry.csv.gz", symmetry_rows)
    paths = [
        artifacts.write_text("accuracy.csv", _csv_text(accuracy_rows)),
        artifacts.write_text(
            "shift_symmetry_summary.csv", _csv_text(_shift_summary_rows(symmetry_rows))
        ),
        artifacts.write_text("phase_average_exactness.csv", _csv_text(phase_rows)),
        artifacts.write_text("parity_symmetry.csv", _csv_text(parity_rows)),
        artifacts.write_text("parity_data_mismatch.csv", _csv_text(parity_data_rows)),
        artifacts.write_text("receptive_fields.csv", _csv_text(rf_rows)),
        artifacts.write_text("bottleneck_units.csv", _csv_text(unit_rows)),
        artifacts.write_text("density_exactness.csv", _csv_text(density_rows)),
        artifacts.write_text("member_grouped_bootstrap.csv", _csv_text(member_bootstrap_rows)),
        artifacts.write_json("summary.json", summary),
    ]
    artifacts.register_existing("predictions.h5")
    shift_plot = output_dir / "shift_symmetry.png"
    accuracy_plot = output_dir / "accuracy_comparison.png"
    _plot_shift(shift_plot, symmetry_rows)
    _plot_accuracy(accuracy_plot, accuracy_rows)
    artifacts.register_existing(shift_plot.name)
    artifacts.register_existing(accuracy_plot.name)
    paths.extend((shift_plot, accuracy_plot))
    failure_path = output_dir / "failure.json"
    if failure_path.exists():
        failure_path.unlink()
    manifest = artifacts.finalize(
        config=resolved,
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=member_ids,
        row_ids=panel_data.row_indices,
        gradient_set="fixed+varied panel; varied reference",
        device=ensemble.device,
        repository=Path(__file__).resolve().parents[1],
    )
    if not pilot and not args.no_publish:
        _publish(paths, Path(resolved["published_dir"]))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"S02 {resolved['mode']} completed; manifest: {manifest}", flush=True)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        run(config, args)
    except Exception as error:
        run_id = config.get("pilot", {}).get("run_id") if args.pilot else config.get("run_id")
        output = (args.output_dir or Path("output/xai/S02") / str(run_id)).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.json").write_text(json.dumps({"exception": repr(error), "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
