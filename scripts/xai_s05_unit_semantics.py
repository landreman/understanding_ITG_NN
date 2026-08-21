#!/usr/bin/env python3
"""Execute S05's bottleneck-unit semantic-density experiment."""

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
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import InvariantMember, circular_shift, receptive_field_blocks
from itg_nn.xai.unit_semantics import (
    NATURAL_EXEMPLAR_VALIDITY,
    cluster_natural_exemplars,
    extract_wrapped_patches,
    first_layer_transfer,
    native_output_comparison,
    physics_concept_traces,
    select_natural_exemplars,
    shift_consistency_error,
    unit_concept_alignment,
)


STRATA = ("overall", "stable_or_near_floor", "unstable")
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
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xai/S05_unit_semantics.json")
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
    }
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = str(value) if isinstance(value, Path) else value
    for key in (
        "dataset",
        "checkpoint",
        "cohorts",
        "channel_scales",
        "s04_ranking",
        "published_dir",
    ):
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


def _strings(values: Any, width: int = 128) -> np.ndarray:
    return np.asarray([str(value).encode("utf-8") for value in values], dtype=f"S{width}")


def _stratum_masks(actual: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    stable = np.asarray(actual) <= threshold
    return {
        "overall": np.ones(len(actual), dtype=bool),
        "stable_or_near_floor": stable,
        "unstable": ~stable,
    }


def _load_channel_scales(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows.sort(key=lambda row: int(row["channel"]))
    names = tuple(row["channel_name"] for row in rows)
    if names != CHANNEL_NAMES:
        raise RuntimeError(f"S01 channel scale order changed: {names}")
    return np.asarray([float(row["robust_sigma_iqr"]) for row in rows])


def _load_s04_importance(path: Path) -> dict[str, dict[str, float]]:
    importance: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        importance[row["unit_id"]] = {
            "s04_shapley_mean_absolute": float(row["shapley_mean_absolute"]),
            "s04_mean_ablation_rms": float(row["mean_ablation_rms"]),
            "s04_shapley_rank": float(row["shapley_rank"]),
        }
    return importance


def _toy_gate(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(16, 96))
    null = rng.normal(size=(16, 96))
    density = np.roll(signal, 7, axis=1)
    result = unit_concept_alignment(
        density,
        np.stack((signal, null), axis=1),
        concept_names=("analytic_signal", "null_control"),
        channel_magnitude_controls=rng.normal(size=(16, 7, 96)),
        sparsity=0.1,
        groups=np.asarray([f"eq{index // 2}" for index in range(16)]),
        bootstrap_replicates=100,
        seed=seed,
    )
    signal_row, null_row = result.rows
    shifted_error = shift_consistency_error(
        density, np.roll(density, 13, axis=1), shift=13, position_axis=1
    )
    native = native_output_comparison(
        np.asarray([-2.0, -1.0, 0.0, 1.0]),
        np.asarray([-1.9, -1.2, -0.1, 1.2]),
        stable_or_near_floor=np.asarray([True, False, False, False]),
    )[0]
    passed = bool(
        signal_row["best_lag"] == 7
        and abs(float(signal_row["lag_correlation"]) - 1.0) < 1e-12
        and abs(float(null_row["lag_correlation"])) < 0.2
        and shifted_error == 0
        and float(native["signed_delta_min"]) < 0
    )
    return {
        "known_concept_best_lag": signal_row["best_lag"],
        "known_concept_lag_correlation": signal_row["lag_correlation"],
        "null_lag_correlation": null_row["lag_correlation"],
        "shift_consistency_error": shifted_error,
        "native_signed_delta_min": native["signed_delta_min"],
        "passed": passed,
    }


def _resume_completed(
    output_dir: Path, dataset: Path, checkpoint: Path
) -> Path | None:
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
    print(f"validated completed S05 run: {manifest_path}", flush=True)
    return manifest_path


def _member_density(
    model: InvariantMember,
    geometry: torch.Tensor,
    a_over_lt: torch.Tensor,
    a_over_ln: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    density: list[np.ndarray] = []
    native_bottleneck: list[np.ndarray] = []
    original: list[np.ndarray] = []
    invariant: list[np.ndarray] = []
    for start in range(0, len(geometry), batch_size):
        stop = min(start + batch_size, len(geometry))
        x = geometry[start:stop].to(device)
        lt = a_over_lt[start:stop].to(device)
        ln = a_over_ln[start:stop].to(device)
        with torch.inference_mode():
            density.append(model.equivariant_density(x).cpu().numpy())
            native_bottleneck.append(model.bottleneck(x).cpu().numpy())
            original.append(model.original(x, lt, ln).cpu().numpy())
            invariant.append(model.invariant(x, lt, ln).cpu().numpy())
    return {
        "density": np.concatenate(density),
        "native_bottleneck": np.concatenate(native_bottleneck),
        "original": np.concatenate(original),
        "invariant": np.concatenate(invariant),
    }


def _density_summary_rows(
    member_id: str,
    density: np.ndarray,
    live: np.ndarray,
    masks: dict[str, np.ndarray],
    importance: dict[str, dict[str, float]],
    tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in np.flatnonzero(live):
        unit_id = f"{member_id}:u{unit:03d}"
        for stratum, mask in masks.items():
            values = density[mask, unit]
            maxima = values.max(axis=1)
            active_positions = np.sum(values > tolerance, axis=1)
            halfmax_positions = np.sum(
                values >= (0.5 * maxima[:, None]), axis=1
            )
            flat = values.ravel()
            rows.append(
                {
                    "member_id": member_id,
                    "unit_id": unit_id,
                    "unit_index": int(unit),
                    "stratum": stratum,
                    "n_equilibrium_files": int(mask.sum()),
                    "signed_mean": float(flat.mean()),
                    "standard_deviation": float(flat.std()),
                    "q50": float(np.quantile(flat, 0.5)),
                    "q90": float(np.quantile(flat, 0.9)),
                    "q99": float(np.quantile(flat, 0.99)),
                    "maximum": float(flat.max()),
                    "active_fraction": float(np.mean(flat > tolerance)),
                    "median_active_positions": float(np.median(active_positions)),
                    "median_halfmax_positions": float(np.median(halfmax_positions)),
                    "q90_halfmax_positions": float(np.quantile(halfmax_positions, 0.9)),
                    **importance.get(unit_id, {
                        "s04_shapley_mean_absolute": float("nan"),
                        "s04_mean_ablation_rms": float("nan"),
                        "s04_shapley_rank": float("nan"),
                    }),
                }
            )
    return rows


def _native_bottleneck_rows(
    member_id: str,
    native: np.ndarray,
    invariant_mean: np.ndarray,
    live: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in np.flatnonzero(live):
        for stratum, mask in masks.items():
            delta = invariant_mean[mask, unit] - native[mask, unit]
            left = invariant_mean[mask, unit]
            right = native[mask, unit]
            correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(right) > 0 else float("nan")
            rows.append(
                {
                    "member_id": member_id,
                    "unit_id": f"{member_id}:u{unit:03d}",
                    "stratum": stratum,
                    "n": int(mask.sum()),
                    "signed_invariant_minus_native_mean": float(delta.mean()),
                    "rms_invariant_minus_native": float(np.sqrt(np.mean(np.square(delta)))),
                    "pearson_invariant_vs_native": correlation,
                    "native_object": "original_f strided pre-GAP bottleneck",
                    "invariant_object": "mean_z rho feeding invariant_tilde_f",
                }
            )
    return rows


def _publish(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in paths:
        temporary = destination / f".{source.name}.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(destination / source.name)


def _plot_density_atlas(
    path: Path,
    member_ids: tuple[str, ...],
    densities: list[np.ndarray],
    live_masks: list[np.ndarray],
) -> None:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for member_id, density, live in zip(member_ids, densities, live_masks):
        for unit in np.flatnonzero(live):
            values = density[:, unit]
            centers = values.argmax(axis=1)
            aligned = np.stack(
                [np.roll(row, 48 - int(center)) for row, center in zip(values, centers)]
            )
            scale = np.maximum(aligned.max(axis=1, keepdims=True), 1e-12)
            rows.append(np.median(aligned / scale, axis=0))
            labels.append(f"r{member_ids.index(member_id) + 1}:u{unit:03d}")
    matrix = np.stack(rows)
    figure, axis = plt.subplots(figsize=(9.5, max(4.0, 0.22 * len(rows))))
    image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=1)
    axis.axvline(48, color="cyan", linewidth=0.6)
    axis.set_yticks(range(len(labels)), labels, fontsize=6)
    axis.set_xticks([0, 24, 48, 72, 95], [-48, -24, 0, 24, 47])
    axis.set_xlabel("position relative to each natural activation maximum")
    axis.set_title("Median max-normalized equivariant unit densities")
    figure.colorbar(image, ax=axis, label="relative activation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_filter_catalog(
    path: Path, transfer_rows: list[dict[str, Any]], member_ids: tuple[str, ...]
) -> None:
    figure, axes = plt.subplots(len(member_ids), 1, figsize=(9, 3.2 * len(member_ids)), squeeze=False)
    for member_index, member_id in enumerate(member_ids):
        selected = [row for row in transfer_rows if row["member_id"] == member_id]
        filters = sorted({int(row["filter_index"]) for row in selected})
        matrix = np.full((len(filters), 7), np.nan)
        for row in selected:
            matrix[int(row["filter_index"]), int(row["channel_index"])] = float(row["spectral_centroid"])
        axis = axes[member_index, 0]
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=48)
        axis.set_xticks(range(7), CHANNEL_NAMES, rotation=30, ha="right", fontsize=7)
        axis.set_ylabel("first-layer filter")
        axis.set_title(f"{member_id}: Fourier-transfer centroid")
        figure.colorbar(image, ax=axis, label="Fourier index")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_motif_clusters(
    path: Path,
    cluster_rows: list[dict[str, Any]],
) -> None:
    units = list(dict.fromkeys(row["unit_id"] for row in cluster_rows))
    separation = []
    balance = []
    for unit_id in units:
        rows = [row for row in cluster_rows if row["unit_id"] == unit_id]
        separation.append(float(rows[0]["two_cluster_separation_ratio"]))
        counts = [int(row["cluster_size"]) for row in rows]
        balance.append(min(counts) / max(sum(counts), 1))
    figure, axis = plt.subplots(figsize=(8.2, 4.0))
    axis.scatter(separation, balance, s=24, alpha=0.75)
    for index, unit_id in enumerate(units):
        axis.annotate(unit_id.split(":")[-1], (separation[index], balance[index]), fontsize=6)
    axis.set_xlabel("between-center distance / median within-cluster distance")
    axis.set_ylabel("minor motif fraction")
    axis.set_title("Natural-exemplar motif coherence and possible polysemanticity")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = _resolve(config, args)
    repository = Path(__file__).resolve().parents[1]
    resolved["source_hashes"] = {
        "runner_sha256": sha256_file(__file__),
        "unit_semantics_library_sha256": sha256_file(repository / "itg_nn/xai/unit_semantics.py"),
        "cohorts_sha256": sha256_file(resolved["cohorts"]),
        "channel_scales_sha256": sha256_file(resolved["channel_scales"]),
        "s04_ranking_sha256": sha256_file(resolved["s04_ranking"]),
    }
    set_deterministic_seed(int(resolved["seed"]))
    dataset = Path(resolved["dataset"])
    checkpoint = Path(resolved["checkpoint"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (Path("output/xai/S05") / str(resolved["run_id"])).resolve()
    )
    if args.resume:
        completed = _resume_completed(output_dir, dataset, checkpoint)
        if completed is not None:
            return completed
    artifacts = RunArtifacts(output_dir)
    progress: dict[str, Any] = {"completed_members": [], "completed_units": []}
    try:
        toy_checks = _toy_gate(int(resolved["seed"]) + 41)
        if not toy_checks["passed"]:
            raise RuntimeError(f"pre-inference analytic cyclic toy failed: {toy_checks}")

        cohorts = json.loads(Path(resolved["cohorts"]).read_text(encoding="utf-8"))
        registered_rows = np.asarray(
            cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64
        )
        row_count = int(resolved["panel_varied_rows"])
        if not 1 <= row_count <= len(registered_rows):
            raise ValueError("panel row cap is outside the registered S01 panel")
        row_ids = registered_rows[:row_count]
        panel = load_hdf5_rows(
            dataset, row_ids, gradient_set="varied", include_targets=True
        )
        if panel.actual_log_heat_flux is None:
            raise RuntimeError("varied panel targets were not loaded")
        actual = panel.actual_log_heat_flux.numpy().astype(np.float64)
        masks = _stratum_masks(actual, float(resolved["stable_threshold_log_Q"]))
        if not masks["stable_or_near_floor"].any() or not masks["unstable"].any():
            raise RuntimeError("panel cap must contain stable and unstable rows")
        with h5py.File(dataset, "r") as h5_file:
            equilibrium_files = _decode(_h5_take(h5_file["equilibrium_files"], row_ids))

        channel_scales = _load_channel_scales(Path(resolved["channel_scales"]))
        concepts = physics_concept_traces(
            panel.geometry.numpy(),
            channel_scales=channel_scales,
            window_widths=tuple(resolved["window_widths"]),
        )
        s04_importance = _load_s04_importance(Path(resolved["s04_ranking"]))
        ensemble = load_ensemble(checkpoint, device=str(resolved["device"]))
        all_ids = tuple(cohorts["member_cohorts"]["all_100"])
        member_ids = all_ids[: int(resolved["members"])]
        if not member_ids:
            raise ValueError("at least one member is required")
        index_by_id = {member_id: index for index, member_id in enumerate(ensemble.member_ids)}
        models = {
            member_id: InvariantMember(ensemble.models[index_by_id[member_id]])
            for member_id in member_ids
        }

        member_results: list[dict[str, np.ndarray]] = []
        widths: list[int] = []
        live_masks: list[np.ndarray] = []
        density_rows: list[dict[str, Any]] = []
        native_output_rows: list[dict[str, Any]] = []
        native_bottleneck_rows: list[dict[str, Any]] = []
        shift_rows: list[dict[str, Any]] = []
        receptive_rows: list[dict[str, Any]] = []
        for member_index, member_id in enumerate(member_ids):
            started = time.monotonic()
            result = _member_density(
                models[member_id],
                panel.geometry,
                panel.a_over_lt,
                panel.a_over_ln,
                batch_size=int(resolved["batch_size"]),
                device=ensemble.device,
            )
            member_results.append(result)
            width = result["density"].shape[1]
            widths.append(width)
            live = np.max(np.abs(result["density"]), axis=(0, 2)) > float(resolved["dead_tolerance"])
            live_masks.append(live)
            density_rows.extend(
                _density_summary_rows(
                    member_id,
                    result["density"],
                    live,
                    masks,
                    s04_importance,
                    float(resolved["dead_tolerance"]),
                )
            )
            for row in native_output_comparison(
                result["original"],
                result["invariant"],
                stable_or_near_floor=masks["stable_or_near_floor"],
            ):
                native_output_rows.append({"member_id": member_id, **row})
            native_bottleneck_rows.extend(
                _native_bottleneck_rows(
                    member_id,
                    result["native_bottleneck"],
                    result["density"].mean(axis=2),
                    live,
                    masks,
                )
            )
            rf = receptive_field_blocks(
                tuple(int(layer.kernel_size[0]) for layer in models[member_id].model.conv_layers)
            )[-1]
            for unit in np.flatnonzero(live):
                receptive_rows.append(
                    {
                        "member_id": member_id,
                        "unit_id": f"{member_id}:u{unit:03d}",
                        "formal_span": rf.formal_span,
                        "left_extent": rf.left_extent,
                        "right_extent": rf.right_extent,
                        "center_offset": rf.center_offset,
                        "unique_periodic_positions": rf.unique_periodic_positions,
                        "wraps": rf.wraps,
                        "globally_connected": rf.globally_connected,
                        "coordinate_convention": "integer offsets -left_extent through +right_extent; wrapped modulo 96",
                    }
                )

            check_count = min(int(resolved["shift_check_rows"]), row_count)
            base_geometry = panel.geometry[:check_count].to(ensemble.device)
            base_lt = panel.a_over_lt[:check_count].to(ensemble.device)
            base_ln = panel.a_over_ln[:check_count].to(ensemble.device)
            base_density = result["density"][:check_count]
            base_concepts = concepts.values[:check_count]
            for shift in resolved["shift_checks"]:
                shifted_geometry = circular_shift(base_geometry, int(shift))
                with torch.inference_mode():
                    shifted_density = models[member_id].equivariant_density(shifted_geometry).cpu().numpy()
                    shifted_invariant = models[member_id].invariant(shifted_geometry, base_lt, base_ln).cpu().numpy()
                    shifted_original = models[member_id].original(shifted_geometry, base_lt, base_ln).cpu().numpy()
                shifted_concepts = physics_concept_traces(
                    shifted_geometry.cpu().numpy(),
                    channel_scales=channel_scales,
                    window_widths=tuple(resolved["window_widths"]),
                )
                shift_rows.append(
                    {
                        "member_id": member_id,
                        "shift": int(shift),
                        "rows": check_count,
                        "density_equivariance_max_abs": shift_consistency_error(
                            base_density, shifted_density, shift=int(shift), position_axis=2
                        ),
                        "concept_equivariance_max_abs": shift_consistency_error(
                            base_concepts, shifted_concepts.values, shift=int(shift), position_axis=2
                        ),
                        "invariant_tilde_f_max_abs_change": float(
                            np.max(np.abs(shifted_invariant - result["invariant"][:check_count]))
                        ),
                        "original_f_rms_change": float(
                            np.sqrt(np.mean(np.square(shifted_original - result["original"][:check_count])))
                        ),
                        "estimand": "native max(log Q, -2)",
                        "transform_tag": "exact-symmetry",
                    }
                )
            progress["completed_members"].append(member_id)
            artifacts.write_json("progress.json", progress)
            print(
                f"density {member_index + 1}/{len(member_ids)} {member_id}: "
                f"{int(live.sum())}/{width} live units in {time.monotonic() - started:.1f}s",
                flush=True,
            )

        maximum_width = max(widths)
        density_array = np.full(
            (len(member_ids), row_count, maximum_width, 96), np.nan, dtype=np.float32
        )
        native_array = np.full(
            (len(member_ids), row_count, maximum_width), np.nan, dtype=np.float32
        )
        original_array = np.full((len(member_ids), row_count), np.nan, dtype=np.float32)
        invariant_array = np.full_like(original_array, np.nan)
        present = np.zeros((len(member_ids), maximum_width), dtype=bool)
        for index, (result, width) in enumerate(zip(member_results, widths)):
            density_array[index, :, :width] = result["density"].astype(np.float32)
            native_array[index, :, :width] = result["native_bottleneck"].astype(np.float32)
            original_array[index] = result["original"].astype(np.float32)
            invariant_array[index] = result["invariant"].astype(np.float32)
            present[index, :width] = True
        densities_path = artifacts.write_hdf5(
            "densities.h5",
            {
                "rho": density_array,
                "native_bottleneck": native_array,
                "original_f": original_array,
                "invariant_tilde_f": invariant_array,
                "unit_present": present,
                "member_id": _strings(member_ids),
                "row_id": row_ids,
                "equilibrium_file": _strings(equilibrium_files, width=240),
                "actual_log_Q": actual.astype(np.float32),
            },
            axes={
                "rho": ("member", "sample", "unit", "position"),
                "native_bottleneck": ("member", "sample", "unit"),
                "original_f": ("member", "sample"),
                "invariant_tilde_f": ("member", "sample"),
                "unit_present": ("member", "unit"),
                "member_id": ("member",),
                "row_id": ("sample",),
                "equilibrium_file": ("sample",),
                "actual_log_Q": ("sample",),
            },
            attributes={
                "canonical_function": "invariant_tilde_f",
                "estimand": "native max(log Q, -2)",
                "density_validity": "exact-symmetry equivariant internal representation",
            },
            compression="gzip",
        )
        concepts_path = artifacts.write_hdf5(
            "concept_traces.h5",
            {
                "trace": concepts.values.astype(np.float32),
                "concept_name": _strings(concepts.names),
                "row_id": row_ids,
                "channel_robust_scale_iqr": channel_scales,
            },
            axes={
                "trace": ("sample", "concept", "position"),
                "concept_name": ("concept",),
                "row_id": ("sample",),
                "channel_robust_scale_iqr": ("channel",),
            },
            attributes={
                "validity_tag": concepts.validity_tag,
                "paper_vocabulary": "identity, abs, derivative, Heaviside, powers, multiply/divide by B, circular mean",
            },
            compression="gzip",
        )

        alignment_rows: list[dict[str, Any]] = []
        motif_rows: list[dict[str, Any]] = []
        exemplar_rows: list[dict[str, Any]] = []
        cluster_rows: list[dict[str, Any]] = []
        patch_values: list[np.ndarray] = []
        patch_centers: list[np.ndarray] = []
        patch_dispersion: list[np.ndarray] = []
        patch_ids: list[str] = []
        controls = np.moveaxis(np.abs(panel.geometry.numpy().astype(np.float64)), 2, 1)
        for member_index, member_id in enumerate(member_ids):
            density = member_results[member_index]["density"]
            rf = receptive_field_blocks(
                tuple(int(layer.kernel_size[0]) for layer in models[member_id].model.conv_layers)
            )[-1]
            offsets = np.arange(-rf.left_extent, rf.right_extent + 1, dtype=np.int64)
            for unit in np.flatnonzero(live_masks[member_index]):
                unit_id = f"{member_id}:u{unit:03d}"
                overall_result = None
                for stratum, mask in masks.items():
                    alignment = unit_concept_alignment(
                        density[mask, unit],
                        concepts.values[mask],
                        concept_names=concepts.names,
                        channel_magnitude_controls=controls[mask],
                        sparsity=float(resolved["alignment_sparsity"]),
                        groups=equilibrium_files[mask],
                        bootstrap_replicates=int(resolved["bootstrap_replicates"]),
                        seed=int(resolved["seed"]) + 1009 * member_index + 17 * unit + STRATA.index(stratum),
                    )
                    if stratum == "overall":
                        overall_result = alignment
                    for row in alignment.rows:
                        alignment_rows.append(
                            {
                                "member_id": member_id,
                                "unit_id": unit_id,
                                "stratum": stratum,
                                **row,
                            }
                        )
                assert overall_result is not None
                winner = max(
                    overall_result.rows,
                    key=lambda row: abs(float(row["lag_correlation"])),
                )
                recurrence = float(winner["bootstrap_recurrence"])
                abs_correlation = abs(float(winner["lag_correlation"]))
                supported = bool(
                    recurrence >= float(resolved["motif_min_recurrence"])
                    and abs_correlation >= float(resolved["motif_min_abs_lag_correlation"])
                )
                exemplars = select_natural_exemplars(
                    density[:, unit],
                    equilibrium_files,
                    count=int(resolved["exemplars_per_unit"]),
                )
                patches = extract_wrapped_patches(
                    panel.geometry.numpy(), exemplars.sample_indices, exemplars.centers, offsets
                )
                clusters = cluster_natural_exemplars(
                    patches.values,
                    clusters=int(resolved["motif_clusters"]),
                    seed=int(resolved["seed"]) + 4001 * member_index + unit,
                )
                flattened = patches.values.reshape(len(patches.values), -1)
                center_flat = clusters.centers.reshape(len(clusters.centers), -1)
                if len(center_flat) == 2:
                    between = float(np.linalg.norm(center_flat[0] - center_flat[1]))
                else:
                    between = float("nan")
                within = []
                for exemplar_index, cluster in enumerate(clusters.assignment):
                    within.append(float(np.linalg.norm(flattened[exemplar_index] - center_flat[cluster])))
                separation = between / max(float(np.median(within)), 1e-12)
                motif_rows.append(
                    {
                        "member_id": member_id,
                        "unit_id": unit_id,
                        "motif_status": "supported_named_motif" if supported else "unresolved_named_concept",
                        "claimed_concept": winner["concept"] if supported else "none",
                        "best_observed_concept": winner["concept"],
                        "lag": winner["best_lag"],
                        "lag_correlation": winner["lag_correlation"],
                        "lag_correlation_ci95_lower": winner["lag_correlation_ci95_lower"],
                        "lag_correlation_ci95_upper": winner["lag_correlation_ci95_upper"],
                        "bootstrap_recurrence": recurrence,
                        "bootstrap_unit": "equilibrium_files",
                        "natural_exemplar_count": len(exemplars.sample_indices),
                        "independent_exemplar_equilibria": len(np.unique(equilibrium_files[exemplars.sample_indices])),
                        "formal_receptive_field_span": rf.formal_span,
                        "receptive_field_left": -rf.left_extent,
                        "receptive_field_right": rf.right_extent,
                        "unique_periodic_positions": rf.unique_periodic_positions,
                        "shift_consistency_max_abs": max(
                            float(row["density_equivariance_max_abs"])
                            for row in shift_rows if row["member_id"] == member_id
                        ),
                        "natural_only": True,
                        "synthetic_optimization_used": False,
                        "validity_tag": NATURAL_EXEMPLAR_VALIDITY,
                    }
                )
                for rank, (sample, center, activation, cluster) in enumerate(
                    zip(
                        exemplars.sample_indices,
                        exemplars.centers,
                        exemplars.activations,
                        clusters.assignment,
                    ),
                    start=1,
                ):
                    exemplar_rows.append(
                        {
                            "member_id": member_id,
                            "unit_id": unit_id,
                            "exemplar_rank": rank,
                            "row_id": int(row_ids[sample]),
                            "equilibrium_file": equilibrium_files[sample],
                            "activation_center_position": int(center),
                            "activation": float(activation),
                            "cluster": int(cluster),
                            "alignment_operation": patches.alignment_operation,
                            "receptive_field_offsets": f"{-rf.left_extent}:{rf.right_extent}",
                            "validity_tag": patches.validity_tag,
                        }
                    )
                for cluster in range(len(clusters.centers)):
                    cluster_rows.append(
                        {
                            "member_id": member_id,
                            "unit_id": unit_id,
                            "cluster": cluster,
                            "cluster_size": int(np.sum(clusters.assignment == cluster)),
                            "cluster_fraction": float(np.mean(clusters.assignment == cluster)),
                            "two_cluster_separation_ratio": separation,
                            "center_statistic": "coordinatewise median",
                            "dispersion_statistic": "coordinatewise median absolute deviation",
                            "validity_tag": NATURAL_EXEMPLAR_VALIDITY,
                        }
                    )
                patch_values.append(patches.values.astype(np.float32))
                patch_centers.append(clusters.centers.astype(np.float32))
                patch_dispersion.append(clusters.dispersion.astype(np.float32))
                patch_ids.append(unit_id)
                progress["completed_units"].append(unit_id)
                artifacts.write_json("progress.json", progress)
                print(
                    f"alignment {len(progress['completed_units'])}/"
                    f"{sum(int(mask.sum()) for mask in live_masks)} {unit_id}",
                    flush=True,
                )

        max_span = max(array.shape[2] for array in patch_values)
        exemplar_count = int(resolved["exemplars_per_unit"])
        cluster_count = int(resolved["motif_clusters"])
        patch_array = np.full(
            (len(patch_values), exemplar_count, 7, max_span), np.nan, dtype=np.float32
        )
        center_array = np.full(
            (len(patch_values), cluster_count, 7, max_span), np.nan, dtype=np.float32
        )
        dispersion_array = np.full_like(center_array, np.nan)
        spans = np.zeros(len(patch_values), dtype=np.int32)
        for index, (values, centers, dispersion) in enumerate(
            zip(patch_values, patch_centers, patch_dispersion)
        ):
            span = values.shape[2]
            spans[index] = span
            patch_array[index, :, :, :span] = values
            center_array[index, :, :, :span] = centers
            dispersion_array[index, :, :, :span] = dispersion
        motifs_h5_path = artifacts.write_hdf5(
            "natural_motifs.h5",
            {
                "patch": patch_array,
                "cluster_center": center_array,
                "cluster_dispersion": dispersion_array,
                "formal_span": spans,
                "unit_id": _strings(patch_ids),
            },
            axes={
                "patch": ("unit", "exemplar", "channel", "formal_offset"),
                "cluster_center": ("unit", "cluster", "channel", "formal_offset"),
                "cluster_dispersion": ("unit", "cluster", "channel", "formal_offset"),
                "formal_span": ("unit",),
                "unit_id": ("unit",),
            },
            attributes={
                "validity_tag": NATURAL_EXEMPLAR_VALIDITY,
                "alignment_operation": "joint_circular_roll_to_activation_center",
                "optimized_synthetic_inputs": False,
            },
            compression="gzip",
        )

        kernel_rows: list[dict[str, Any]] = []
        transfer_rows: list[dict[str, Any]] = []
        transfer_arrays: list[np.ndarray] = []
        maximum_filters = max(model.model.conv_layers[0].out_channels for model in models.values())
        transfer_h5 = np.full((len(member_ids), maximum_filters, 7, 49), np.nan, dtype=np.float32)
        for member_index, member_id in enumerate(member_ids):
            weights = models[member_id].model.conv_layers[0].weight.detach().cpu().numpy()
            result = first_layer_transfer(weights, grid_size=96)
            transfer_h5[member_index, : len(weights)] = result.amplitude.astype(np.float32)
            transfer_arrays.append(result.amplitude)
            for filter_index in range(weights.shape[0]):
                for channel in range(7):
                    for kernel_position, weight in enumerate(weights[filter_index, channel]):
                        kernel_rows.append(
                            {
                                "member_id": member_id,
                                "filter_index": filter_index,
                                "channel_index": channel,
                                "channel_name": CHANNEL_NAMES[channel],
                                "kernel_position": kernel_position,
                                "signed_weight": float(weight),
                            }
                        )
                    amplitude = result.amplitude[filter_index, channel]
                    total = float(amplitude.sum())
                    transfer_rows.append(
                        {
                            "member_id": member_id,
                            "filter_index": filter_index,
                            "channel_index": channel,
                            "channel_name": CHANNEL_NAMES[channel],
                            "kernel_width": weights.shape[2],
                            "dc_amplitude": float(amplitude[0]),
                            "peak_frequency_index": int(np.argmax(amplitude)),
                            "spectral_centroid": float(
                                np.dot(result.frequency_index, amplitude) / total
                            ) if total > 0 else float("nan"),
                            "kernel_l1": float(np.sum(np.abs(weights[filter_index, channel]))),
                            "kernel_l2": float(np.linalg.norm(weights[filter_index, channel])),
                        }
                    )
        filters_h5_path = artifacts.write_hdf5(
            "first_layer_transfer.h5",
            {
                "amplitude": transfer_h5,
                "frequency_index": np.arange(49),
                "member_id": _strings(member_ids),
                "channel_name": _strings(CHANNEL_NAMES),
            },
            axes={
                "amplitude": ("member", "filter", "channel", "frequency"),
                "frequency_index": ("frequency",),
                "member_id": ("member",),
                "channel_name": ("channel",),
            },
            attributes={"catalog": "first trained convolutional layer"},
            compression="gzip",
        )

        small_paths = [
            artifacts.write_text("unit_density_summary.csv", _csv_text(density_rows)),
            artifacts.write_text("unit_concept_alignment.csv", _csv_text(alignment_rows)),
            artifacts.write_text("unit_motifs.csv", _csv_text(motif_rows)),
            artifacts.write_text("natural_exemplars.csv", _csv_text(exemplar_rows)),
            artifacts.write_text("motif_clusters.csv", _csv_text(cluster_rows)),
            artifacts.write_text("receptive_fields.csv", _csv_text(receptive_rows)),
            artifacts.write_text("native_function_comparison.csv", _csv_text(native_output_rows)),
            artifacts.write_text("native_bottleneck_comparison.csv", _csv_text(native_bottleneck_rows)),
            artifacts.write_text("shift_consistency.csv", _csv_text(shift_rows)),
            artifacts.write_text("first_layer_kernels.csv", _csv_text(kernel_rows)),
            artifacts.write_text("first_layer_transfer.csv", _csv_text(transfer_rows)),
        ]
        density_plot = output_dir / "density_atlas.png"
        _plot_density_atlas(
            density_plot,
            member_ids,
            [result["density"] for result in member_results],
            live_masks,
        )
        artifacts.register_existing(density_plot.name)
        filter_plot = output_dir / "filter_transfer_catalog.png"
        _plot_filter_catalog(filter_plot, transfer_rows, member_ids)
        artifacts.register_existing(filter_plot.name)
        motif_plot = output_dir / "motif_clusters.png"
        _plot_motif_clusters(motif_plot, cluster_rows)
        artifacts.register_existing(motif_plot.name)
        small_paths.extend((density_plot, filter_plot, motif_plot))

        supported_rows = [row for row in motif_rows if row["motif_status"] == "supported_named_motif"]
        overall_density = [row for row in density_rows if row["stratum"] == "overall"]
        summary = {
            "step": "S05",
            "mode": resolved["mode"],
            "estimand": "member-level signed rho for invariant_tilde_f plus native original_f comparison in max(log Q, -2)",
            "cohort": {
                "gradient_set": "varied",
                "panel_rows": row_count,
                "unique_equilibrium_files": int(len(np.unique(equilibrium_files))),
                "stable_or_near_floor": int(masks["stable_or_near_floor"].sum()),
                "unstable": int(masks["unstable"].sum()),
            },
            "members": list(member_ids),
            "live_units": {member_id: int(live.sum()) for member_id, live in zip(member_ids, live_masks)},
            "concept_count": len(concepts.names),
            "supported_named_motifs": len(supported_rows),
            "unresolved_units": len(motif_rows) - len(supported_rows),
            "maximum_density_shift_error": float(max(row["density_equivariance_max_abs"] for row in shift_rows)),
            "maximum_concept_shift_error": float(max(row["concept_equivariance_max_abs"] for row in shift_rows)),
            "maximum_invariant_output_shift_error": float(max(row["invariant_tilde_f_max_abs_change"] for row in shift_rows)),
            "median_halfmax_support_positions": float(np.median([row["median_halfmax_positions"] for row in overall_density])),
            "natural_exemplars_per_unit": int(resolved["exemplars_per_unit"]),
            "bootstrap": {
                "unit": "equilibrium_files",
                "replicates": int(resolved["bootstrap_replicates"]),
            },
            "toy_checks": toy_checks,
            "validity_tags": {
                "density_shift": "exact-symmetry",
                "concept_alignment": NATURAL_EXEMPLAR_VALIDITY,
                "natural_exemplars": NATURAL_EXEMPLAR_VALIDITY,
            },
            "artifacts": {
                "large": [densities_path.name, concepts_path.name, motifs_h5_path.name, filters_h5_path.name],
                "published": [path.name for path in small_paths],
            },
            "deferred": [
                "window surrogate for unresolved units: both MVD members have globally connected 96-position final receptive fields, so the nominal 672-input surrogate is not a small local regression; defer model choice to a later step rather than tune it on the registered panel",
                "NMF or sparse dictionary learning: natural two-cluster summaries are coherent enough to retain as the preregistered first diagnostic",
            ],
        }
        summary_path = artifacts.write_json("summary.json", summary)
        small_paths.append(summary_path)
        if not args.no_publish and resolved["mode"] == "production":
            _publish(small_paths, Path(resolved["published_dir"]))

        manifest = artifacts.finalize(
            config=resolved,
            dataset=dataset,
            checkpoint=checkpoint,
            member_ids=member_ids,
            row_ids=row_ids,
            gradient_set="varied",
            device=ensemble.device,
            repository=repository,
            command=sys.argv,
            published_dir=(
                None
                if args.no_publish or resolved["mode"] != "production"
                else Path(resolved["published_dir"])
            ),
        )
        print(f"S05 complete: {manifest}", flush=True)
        return manifest
    except Exception as error:
        failure = {
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "progress": progress,
        }
        artifacts.write_json("failure.json", failure)
        raise


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
