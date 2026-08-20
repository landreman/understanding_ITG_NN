#!/usr/bin/env python3
"""Execute the registered S01 dataset, ranking, and cohort audit."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from itg_nn.data import (
    SPLIT_NAMES,
    load_hdf5_rows,
    reference_split_assignments,
    reference_test_rows,
)
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.audit import (
    STABLE_THRESHOLD,
    channel_correlation_statistics,
    flux_regimes,
    grouped_bootstrap,
    median_normalized_power_spectrum,
    performance_rows,
    quantile_bins,
    regression_metrics,
    robust_channel_statistics,
    row_bootstrap_r2,
    select_panel_rows,
    spearman_correlation,
    top_k_members,
)
from itg_nn.xai.members import MemberPredictor
from itg_nn.xai.runtime import iter_inference_batches, set_deterministic_seed


CHANNEL_NAMES = (
    "bmag",
    "gbdrift",
    "cvdrift",
    "gbdrift0_over_shat",
    "gds2",
    "gds21_over_shat",
    "gds22_over_shat_squared",
)
REFERENCE_ENSEMBLE_R2 = 0.989310659379


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/xai/S01_audit.json"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--published-dir", type=Path, default=Path("reports/xai/S01_artifacts"))
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--members", type=int, help="Validation-ranked member count (pilot/debug)"
    )
    parser.add_argument(
        "--rows", type=int, help="Reference-row prefix length (pilot/debug)"
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse a validated prediction artifact"
    )
    parser.add_argument("--no-publish", action="store_true")
    return parser


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def _h5_take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    """Read arbitrary source rows while satisfying h5py's increasing-index rule."""

    unique_rows, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return dataset[unique_rows][inverse]


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _split_audit(dataset: Path, assignments: dict[str, np.ndarray]) -> dict[str, Any]:
    fixed = assignments["fixed"]
    varied = assignments["varied"]
    paired = (fixed >= 0) & (varied >= 0)
    cross = paired & (fixed != varied)
    pair_matrix = {
        f"fixed_{SPLIT_NAMES[i]}__varied_{SPLIT_NAMES[j]}": int(
            np.sum(paired & (fixed == i) & (varied == j))
        )
        for i in range(3)
        for j in range(3)
    }
    with h5py.File(dataset, "r") as h5_file:
        equilibrium_files = _decode(h5_file["equilibrium_files"][:])
    combined_assignment = np.concatenate((fixed, varied))
    combined_equilibrium = np.concatenate((equilibrium_files, equilibrium_files))
    positive = combined_assignment >= 0
    unique_equilibrium, inverse = np.unique(combined_equilibrium[positive], return_inverse=True)
    split_bits = np.left_shift(1, combined_assignment[positive].astype(np.int64))
    masks = np.zeros(len(unique_equilibrium), dtype=np.int8)
    np.bitwise_or.at(masks, inverse, split_bits.astype(np.int8))
    split_count = np.asarray([int(mask).bit_count() for mask in masks])
    row_crosses = split_count[inverse] > 1
    varied_test = varied == 2
    _, full_inverse = np.unique(equilibrium_files, return_inverse=True)
    equilibrium_masks = np.zeros(len(np.unique(equilibrium_files)), dtype=np.int8)
    # Map masks again on the source-row equilibrium ordering.
    for split_index in range(3):
        source_present = (fixed == split_index) | (varied == split_index)
        np.bitwise_or.at(
            equilibrium_masks,
            full_inverse[source_present],
            np.int8(1 << split_index),
        )
    varied_test_masks = equilibrium_masks[full_inverse[varied_test]]
    return {
        "split_counts": {
            gradient_set: {
                **{name: int(np.sum(values == index)) for index, name in enumerate(SPLIT_NAMES)},
                "excluded_nonpositive": int(np.sum(values < 0)),
            }
            for gradient_set, values in assignments.items()
        },
        "positive_fixed_varied_identity_pairs": int(paired.sum()),
        "identity_pairs_crossing_splits": int(cross.sum()),
        "identity_pair_crossing_fraction": float(cross.sum() / paired.sum()),
        "identity_pair_split_matrix": pair_matrix,
        "equilibrium_files_total": int(len(unique_equilibrium)),
        "equilibrium_files_in_multiple_splits": int(np.sum(split_count > 1)),
        "equilibrium_files_in_all_three_splits": int(np.sum(split_count == 3)),
        "combined_positive_samples_in_cross_split_equilibria": int(row_crosses.sum()),
        "combined_positive_sample_cross_split_fraction": float(row_crosses.mean()),
        "varied_test_rows_with_equilibrium_in_train": int(np.sum((varied_test_masks & 1) > 0)),
        "varied_test_rows_with_equilibrium_in_validation": int(np.sum((varied_test_masks & 2) > 0)),
        "varied_test_rows_with_fixed_pair_outside_test": int(
            np.sum(varied_test & (fixed != 2))
        ),
        "geometry_storage_note": (
            "Both simulation groups reference the same root raw_feature_tensor row; "
            "paired geometry equality is exact by construction."
        ),
    }


def _equilibrium_in_train(
    dataset: Path, assignments: dict[str, np.ndarray], source_rows: np.ndarray
) -> np.ndarray:
    """Identify whether each source row's equilibrium occurs in legacy training."""

    with h5py.File(dataset, "r") as h5_file:
        all_equilibria = _decode(h5_file["equilibrium_files"][:])
    training_rows = (assignments["fixed"] == 0) | (assignments["varied"] == 0)
    training_equilibria = np.unique(all_equilibria[training_rows])
    return np.isin(
        all_equilibria[np.asarray(source_rows, dtype=np.int64)], training_equilibria
    )


def _validate_resume_sources(
    output_dir: Path, dataset: Path, checkpoint: Path
) -> None:
    """Reject cached predictions unless their immutable inputs hash-match."""

    manifest_path = output_dir.resolve() / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("cannot validate cached predictions without manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for label, path in (("dataset", dataset), ("checkpoint", checkpoint)):
        expected = manifest.get(label, {}).get("sha256")
        if not expected:
            raise RuntimeError(f"cached manifest has no {label} SHA-256")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"cached {label} SHA-256 does not match the resolved {label}"
            )


def _predict(
    ensemble,
    member_ids: tuple[str, ...],
    data,
    batch_size: int,
) -> np.ndarray:
    predictor = MemberPredictor.from_ensemble(ensemble, member_ids).eval()
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(iter_inference_batches(data, batch_size), start=1):
            batches.append(
                predictor(
                    batch.geometry.to(ensemble.device),
                    batch.a_over_lt.to(ensemble.device),
                    batch.a_over_ln.to(ensemble.device),
                ).cpu()
            )
            if batch_index % 10 == 0:
                print(f"prediction batches complete: {batch_index}", flush=True)
    return torch.cat(batches, dim=1).numpy()


def _identity_audit(dataset: Path, *, full: bool) -> dict[str, Any]:
    max_absolute = 0.0
    max_relative = 0.0
    squared = 0.0
    expected_squared = 0.0
    relative_squared = 0.0
    count = 0
    relative_errors: list[np.ndarray] = []
    outliers: list[dict[str, float | int]] = []
    chunk_size = 2048
    with h5py.File(dataset, "r") as h5_file:
        geometry = h5_file["raw_feature_tensor"]
        registered = h5_file["FSA_grad_xs"]
        stop_all = len(geometry) if full else min(4096, len(geometry))
        for start in range(0, stop_all, chunk_size):
            stop = min(start + chunk_size, stop_all)
            values = geometry[start:stop]
            bmag = values[:, :, 0]
            calculated = np.mean(np.sqrt(values[:, :, 6]) / bmag, axis=1) / np.mean(
                1.0 / bmag, axis=1
            )
            expected = registered[start:stop].astype(np.float64)
            difference = calculated - expected
            relative = np.abs(difference) / np.maximum(np.abs(expected), np.finfo(float).tiny)
            relative_errors.append(relative)
            for offset in np.flatnonzero(relative > 5e-7):
                geometry_row = values[offset]
                outliers.append(
                    {
                        "row_id": int(start + offset),
                        "relative_error": float(relative[offset]),
                        "absolute_error": float(abs(difference[offset])),
                        "calculated_FSA_grad_x": float(calculated[offset]),
                        "registered_FSA_grad_x": float(expected[offset]),
                        "minimum_bmag": float(np.min(geometry_row[:, 0])),
                        "minimum_channel_6": float(np.min(geometry_row[:, 6])),
                        "maximum_gds2": float(np.max(geometry_row[:, 4])),
                        "all_geometry_finite": bool(np.isfinite(geometry_row).all()),
                        "degenerate_geometry_detected": bool(
                            not np.isfinite(geometry_row).all()
                            or np.min(geometry_row[:, 0]) <= 0
                            or np.min(geometry_row[:, 6]) < 0
                        ),
                    }
                )
            max_absolute = max(max_absolute, float(np.max(np.abs(difference))))
            max_relative = max(max_relative, float(np.max(relative)))
            squared += float(np.square(difference).sum())
            expected_squared += float(np.square(expected).sum())
            relative_squared += float(np.square(difference / expected).sum())
            count += len(difference)
    relative_l2 = float(np.sqrt(squared / expected_squared))
    all_relative = np.concatenate(relative_errors)
    return {
        "rows_checked": count,
        "scope": "full_dataset" if full else "pilot_prefix",
        "formula": "mean_z(sqrt(channel_6)/channel_0) / mean_z(1/channel_0)",
        "max_absolute_error": max_absolute,
        "max_relative_error": max_relative,
        "relative_l2_error": relative_l2,
        "rms_relative_error": float(np.sqrt(relative_squared / count)),
        "relative_error_quantiles": {
            "q50": float(np.quantile(all_relative, 0.5)),
            "q95": float(np.quantile(all_relative, 0.95)),
            "q99": float(np.quantile(all_relative, 0.99)),
            "q99_9": float(np.quantile(all_relative, 0.999)),
        },
        "rows_above_5e-7_relative": outliers,
        "rmse": float(np.sqrt(squared / count)),
        "passed_1e-7_relative_l2": bool(relative_l2 <= 1e-7),
        "precision_note": (
            "The registered field is float32 while geometry is float64; max-row "
            "relative error is also retained and is not used as the aggregate tolerance."
        ),
        "registered_feature": "log_FSA_grad_x",
    }


def _panel_artifact(
    dataset: Path,
    varied_rows: np.ndarray,
    assignments: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], list[dict[str, Any]]]:
    arrays: dict[str, list[np.ndarray]] = {}
    metadata_rows: list[dict[str, Any]] = []
    with h5py.File(dataset, "r") as h5_file:
        geometry = h5_file["raw_feature_tensor"][varied_rows].astype(np.float32)
        scalar_names = _decode(h5_file["scalar_features"][:])
        scalar = h5_file["scalar_feature_matrix"][varied_rows].astype(np.float32)
        equilibrium_class = h5_file["equilibrium_class"][varied_rows].astype(np.int16)
        equilibrium_files = _decode(h5_file["equilibrium_files"][varied_rows])
        tube_files = _decode(h5_file["tube_files"][varied_rows])
        quasr = h5_file["QUASR_IDs"][varied_rows]
        fsa = h5_file["FSA_grad_xs"][varied_rows].astype(np.float32)
        for gradient_code, gradient_set in enumerate(("varied", "fixed")):
            group = h5_file[f"{gradient_set}_gradient_simulations"]
            target_q = group["Q_avgs"][varied_rows]
            positive = target_q > 0
            target_log = np.full(len(target_q), np.nan, dtype=np.float32)
            target_log[positive] = np.maximum(np.log(target_q[positive]), -2.0)
            # The value the checkpoint actually sees. Fixed rows are supplied
            # at their physical +3; see reports/xai/S03_fixed_gradient_decision.md.
            model_lt = group["a_over_LT"][varied_rows].astype(np.float32)
            block = {
                "row_id": varied_rows,
                "gradient_set_code": np.full(len(varied_rows), gradient_code, dtype=np.int8),
                "legacy_split_code": assignments[gradient_set][varied_rows],
                "geometry": geometry,
                "actual_log_Q": target_log,
                "a_over_LT_model": model_lt,
                "a_over_Ln_model": group["a_over_Ln"][varied_rows].astype(np.float32),
                "Q_stds": group["Q_stds"][varied_rows].astype(np.float32),
                "Q_avgs_vs_z": group["Q_avgs_vs_z"][varied_rows].astype(np.float32),
                "zonal_phi2_amplitudes": group["zonal_phi2_amplitudes"][
                    varied_rows
                ].astype(np.float32),
                "Q_avgs_divided_by_FSA_grad_x": group[
                    "Q_avgs_divided_by_FSA_grad_x"
                ][varied_rows].astype(np.float32),
                "scalar_feature_matrix": scalar,
                "FSA_grad_x": fsa,
                "log_FSA_grad_x": np.log(fsa).astype(np.float32),
                "equilibrium_class": equilibrium_class,
                "QUASR_ID": quasr,
                "equilibrium_file": np.asarray(equilibrium_files, dtype="S"),
                "tube_file": np.asarray(tube_files, dtype="S"),
            }
            for key, value in block.items():
                arrays.setdefault(key, []).append(value)
            for index, row_id in enumerate(varied_rows):
                row: dict[str, Any] = {
                    "stable_id": f"{gradient_set}:{int(row_id)}",
                    "gradient_set": gradient_set,
                    "split": SPLIT_NAMES[int(assignments[gradient_set][row_id])],
                    "row_id": int(row_id),
                    "equilibrium_file": equilibrium_files[index],
                    "tube_file": tube_files[index],
                    "equilibrium_class": int(equilibrium_class[index]),
                    "actual_log_Q": float(target_log[index]),
                    "a_over_LT_model": float(model_lt[index]),
                    "a_over_Ln_model": float(group["a_over_Ln"][row_id]),
                    "FSA_grad_x": float(fsa[index]),
                }
                row.update({name: float(scalar[index, j]) for j, name in enumerate(scalar_names)})
                metadata_rows.append(row)
    combined = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    axes = {
        "row_id": ("sample",),
        "gradient_set_code": ("sample",),
        "legacy_split_code": ("sample",),
        "geometry": ("sample", "z", "channel"),
        "actual_log_Q": ("sample",),
        "a_over_LT_model": ("sample",),
        "a_over_Ln_model": ("sample",),
        "Q_stds": ("sample",),
        "Q_avgs_vs_z": ("sample", "z"),
        "zonal_phi2_amplitudes": ("sample",),
        "Q_avgs_divided_by_FSA_grad_x": ("sample",),
        "scalar_feature_matrix": ("sample", "scalar_feature"),
        "FSA_grad_x": ("sample",),
        "log_FSA_grad_x": ("sample",),
        "equilibrium_class": ("sample",),
        "QUASR_ID": ("sample",),
        "equilibrium_file": ("sample",),
        "tube_file": ("sample",),
    }
    return combined, axes, metadata_rows


def _plot_ranking(
    path: Path,
    validation_r2: np.ndarray,
    heldout_r2: np.ndarray,
    ci: np.ndarray,
) -> None:
    figure, axes = plt.subplots(figsize=(6.2, 4.2))
    axes.errorbar(
        validation_r2,
        heldout_r2,
        yerr=np.maximum(
            0.0, np.vstack((heldout_r2 - ci[:, 0], ci[:, 1] - heldout_r2))
        ),
        fmt="o",
        markersize=2.8,
        linewidth=0.5,
        alpha=0.65,
    )
    axes.set_xlabel(r"Stored validation $R^2$")
    axes.set_ylabel(r"Reference-test $R^2$ (95% grouped bootstrap CI)")
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_spectra(path: Path, spectrum: np.ndarray) -> None:
    figure, axes = plt.subplots(figsize=(6.2, 4.2))
    modes = np.arange(1, spectrum.shape[1] + 1)
    for channel, name in enumerate(CHANNEL_NAMES):
        axes.semilogy(modes, spectrum[channel], label=f"{channel}: {name}")
    axes.set_xlabel("Parallel rFFT mode (DC excluded)")
    axes.set_ylabel("Median per-sample fraction of non-DC power")
    axes.legend(fontsize=6, ncol=2)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _publish(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in paths:
        target = destination / source.name
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)


def run(config: dict[str, Any], args: argparse.Namespace) -> Path:
    pilot = bool(args.pilot)
    resolved = dict(config)
    resolved["dataset"] = str((args.dataset or Path(config["dataset"])).resolve())
    resolved["checkpoint"] = str((args.checkpoint or Path(config["checkpoint"])).resolve())
    resolved["device"] = args.device or config["device"]
    resolved["batch_size"] = args.batch_size or int(config["batch_size"])
    resolved["seed"] = args.seed if args.seed is not None else int(config["seed"])
    if pilot:
        resolved.update(config["pilot"])
        resolved["mode"] = "pilot"
    else:
        resolved["mode"] = "registered_production"
    if args.members is not None:
        if not 1 <= args.members <= 100:
            raise ValueError("--members must be in [1, 100]")
        resolved["members"] = args.members
    if args.rows is not None:
        if args.rows < 1:
            raise ValueError("--rows must be positive")
        resolved["reference_rows"] = args.rows
    resolved["resume"] = bool(args.resume)
    dataset = Path(resolved["dataset"])
    checkpoint = Path(resolved["checkpoint"])
    run_id = str(resolved["run_id"])
    output_dir = args.output_dir or Path("output/xai/S01") / run_id
    artifacts = RunArtifacts(output_dir)
    set_deterministic_seed(int(resolved["seed"]))

    assignments = reference_split_assignments(dataset, seed=int(resolved["split_seed"]))
    split_audit = _split_audit(dataset, assignments)
    reference_rows = reference_test_rows(dataset, seed=int(resolved["split_seed"]))
    if len(reference_rows) != int(config["expected_varied_reference_rows"]):
        raise RuntimeError(f"reference cohort has {len(reference_rows)} rows, expected 9785")
    analysis_rows = reference_rows[: int(resolved.get("reference_rows", len(reference_rows)))]
    data = load_hdf5_rows(dataset, analysis_rows, gradient_set="varied", include_targets=True)
    if data.actual_log_heat_flux is None:
        raise RuntimeError("reference targets were not loaded")
    actual = data.actual_log_heat_flux.numpy()

    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_members = bundle["members"]
    validation_ranked = sorted(
        checkpoint_members,
        key=lambda member: (-float(member["validation_r2"]), str(member["id"])),
    )
    selected_members = validation_ranked[: int(resolved.get("members", len(validation_ranked)))]
    member_ids = tuple(str(member["id"]) for member in selected_members)
    validation_r2 = np.asarray([float(member["validation_r2"]) for member in selected_members])
    ensemble = load_ensemble(checkpoint, device=resolved["device"])
    prediction_cache = output_dir.resolve() / "reference_predictions.h5"
    if args.resume and prediction_cache.is_file():
        _validate_resume_sources(output_dir, dataset, checkpoint)
        with h5py.File(prediction_cache, "r") as h5_file:
            cached_rows = h5_file["row_id"][:]
            cached_predictions = h5_file["member_prediction_log_Q"][:]
            cached_member_ids = tuple(json.loads(h5_file.attrs["member_ids"]))
        if not np.array_equal(cached_rows, analysis_rows):
            raise RuntimeError("cached prediction row IDs do not match the resolved cohort")
        if cached_member_ids != member_ids:
            raise RuntimeError("cached prediction member IDs do not match the resolved cohort")
        predictions = cached_predictions
        print(f"resumed predictions from {prediction_cache}", flush=True)
    else:
        predictions = _predict(ensemble, member_ids, data, int(resolved["batch_size"]))
    ensemble_prediction = predictions.mean(axis=0)
    ensemble_metrics = regression_metrics(actual, ensemble_prediction)
    is_full_registered_run = (
        not pilot and len(analysis_rows) == len(reference_rows) and len(member_ids) == 100
    )
    if is_full_registered_run:
        tolerance = float(config["ensemble_r2_tolerance"])
        expected = float(config.get("expected_ensemble_r2", REFERENCE_ENSEMBLE_R2))
        if abs(ensemble_metrics["r2"] - expected) > tolerance:
            raise RuntimeError(
                f"ensemble R2 {ensemble_metrics['r2']:.12f} differs from {expected:.12f}"
            )

    with h5py.File(dataset, "r") as h5_file:
        equilibrium_files = _decode(_h5_take(h5_file["equilibrium_files"], analysis_rows))
        equilibrium_class = _h5_take(h5_file["equilibrium_class"], analysis_rows).astype(np.int16)
        scalar_names = _decode(h5_file["scalar_features"][:])
        reference_geometry_float64 = _h5_take(
            h5_file["raw_feature_tensor"], analysis_rows
        )
    flux_labels, flux_definition = flux_regimes(actual)
    lt_bin, lt_cuts = quantile_bins(data.a_over_lt.numpy(), 3)
    ln_bin, ln_cuts = quantile_bins(data.a_over_ln.numpy(), 3)
    stability = np.where(actual <= STABLE_THRESHOLD, "stable_near_floor", "unstable")
    fixed_pair_split = np.asarray(
        [SPLIT_NAMES[int(value)] for value in assignments["fixed"][analysis_rows]]
    )
    equilibrium_in_train = _equilibrium_in_train(dataset, assignments, analysis_rows)
    strata = {
        "stability": stability,
        "flux_regime": flux_labels,
        "a_over_LT_tertile": lt_bin,
        "a_over_Ln_tertile": ln_bin,
        "equilibrium_class": equilibrium_class,
        "fixed_pair_split": fixed_pair_split,
        "equilibrium_in_train": equilibrium_in_train,
    }
    overall_rows, stratified_rows = performance_rows(actual, predictions, member_ids, strata)
    heldout_r2 = np.asarray([row["r2"] for row in overall_rows[1:]])

    bootstrap = grouped_bootstrap(
        actual,
        predictions,
        equilibrium_files,
        replicates=int(resolved["bootstrap_replicates"]),
        seed=int(resolved["seed"]) + 1,
    )
    ci = np.quantile(bootstrap.r2, (0.025, 0.975), axis=0).T
    heldout_rank = np.argsort(np.argsort(-heldout_r2, kind="mergesort"), kind="mergesort") + 1
    top10_membership = np.zeros_like(bootstrap.r2, dtype=bool)
    top_k = min(10, len(member_ids))
    for replicate in range(len(bootstrap.r2)):
        top10_membership[
            replicate, top_k_members(bootstrap.r2[replicate], top_k)
        ] = True
    ranking_rows: list[dict[str, Any]] = []
    for index, member_id in enumerate(member_ids):
        ranking_rows.append(
            {
                "member_id": member_id,
                "stored_validation_rank": index + 1,
                "stored_validation_r2": validation_r2[index],
                "heldout_rank": int(heldout_rank[index]),
                "heldout_r2": heldout_r2[index],
                "heldout_r2_ci_low": ci[index, 0],
                "heldout_r2_ci_high": ci[index, 1],
                "bootstrap_rank_median": float(np.median(bootstrap.ranks[:, index])),
                "bootstrap_rank_ci_low": float(np.quantile(bootstrap.ranks[:, index], 0.025)),
                "bootstrap_rank_ci_high": float(np.quantile(bootstrap.ranks[:, index], 0.975)),
                "bootstrap_probability_top10": float(np.mean(top10_membership[:, index])),
            }
        )
    stored_top10_reproduction = None
    if len(member_ids) == 100:
        stored_top10 = set(range(10))
        reproduced = [
            set(np.flatnonzero(top10_membership[replicate])) == stored_top10
            for replicate in range(len(bootstrap.ranks))
        ]
        stored_top10_reproduction = float(np.mean(reproduced))

    row_bootstrap = row_bootstrap_r2(
        actual,
        ensemble_prediction,
        replicates=int(resolved["bootstrap_replicates"]),
        seed=int(resolved["seed"]) + 2,
    )
    group_bootstrap_ensemble = grouped_bootstrap(
        actual,
        ensemble_prediction[None, :],
        equilibrium_files,
        replicates=int(resolved["bootstrap_replicates"]),
        seed=int(resolved["seed"]) + 2,
    ).r2[:, 0]
    tube_se = float(np.std(row_bootstrap, ddof=1))
    group_se = float(np.std(group_bootstrap_ensemble, ddof=1))
    _, group_sizes = np.unique(equilibrium_files, return_counts=True)
    perfect_correlation_design_effect = float(np.square(group_sizes).sum() / len(actual))

    absolute_error = np.abs(ensemble_prediction - actual)
    disagreement = predictions.std(axis=0)
    effective_panel_size = min(
        int(resolved["panel_varied_rows"]), len(np.unique(equilibrium_files))
    )
    resolved["effective_panel_varied_rows"] = effective_panel_size
    panel_rows, panel_sampling = select_panel_rows(
        analysis_rows,
        equilibrium_files,
        equilibrium_class,
        flux_labels,
        lt_bin,
        ln_bin,
        absolute_error,
        disagreement,
        panel_size=effective_panel_size,
        seed=int(resolved["seed"]) + 3,
    )
    panel_lookup = {int(row): index for index, row in enumerate(analysis_rows)}
    panel_indices = np.asarray([panel_lookup[int(row)] for row in panel_rows])
    panel_arrays, panel_axes, panel_metadata = _panel_artifact(
        dataset, panel_rows, assignments
    )
    panel_geometry_unique = panel_arrays["geometry"][: len(panel_rows)]
    channel_stats = robust_channel_statistics(reference_geometry_float64)
    for row in channel_stats:
        row["channel_name"] = CHANNEL_NAMES[int(row["channel"])]
    channel_correlations = channel_correlation_statistics(panel_geometry_unique)
    spectrum = median_normalized_power_spectrum(panel_geometry_unique)
    identity = _identity_audit(dataset, full=not pilot)

    panel_samples = [
        {
            "stable_id": f"{gradient_set}:{int(row)}",
            "split": SPLIT_NAMES[int(assignments[gradient_set][row])],
        }
        for gradient_set in ("varied", "fixed")
        for row in panel_rows
    ]

    cohorts = {
        "schema_version": 2,
        "stable_id_schema": "<gradient_set>:<zero-based HDF5 row_id>",
        "reference_varied": {
            "split": "legacy_test",
            "row_ids": [int(row) for row in reference_rows],
            "count": int(len(reference_rows)),
        },
        "interpretation_panel": {
            "varied_row_ids": [int(row) for row in panel_rows],
            "fixed_row_ids": [int(row) for row in panel_rows],
            "stable_ids": [f"varied:{int(row)}" for row in panel_rows]
            + [f"fixed:{int(row)}" for row in panel_rows],
            "samples": panel_samples,
            "count": int(2 * len(panel_rows)),
            "pairing": "same raw_feature_tensor row; varied then fixed",
            "sampling": panel_sampling,
        },
        "member_cohorts": {
            "stored_validation_top_10": [str(member["id"]) for member in validation_ranked[:10]],
            "stored_validation_ranks_11_50": [
                str(member["id"]) for member in validation_ranked[10:50]
            ],
            "stored_validation_ranks_51_100": [
                str(member["id"]) for member in validation_ranked[50:]
            ],
            "all_100": [str(member["id"]) for member in validation_ranked],
            "ensemble": "arithmetic mean of all 100 native clipped-log member outputs",
        },
    }
    summary = {
        "mode": resolved["mode"],
        "estimand": "native max(log Q, -2)",
        "reference_rows_analyzed": int(len(analysis_rows)),
        "reference_rows_registered": int(len(reference_rows)),
        "members_analyzed": len(member_ids),
        "ensemble_metrics": ensemble_metrics,
        "reference_r2_absolute_difference": float(
            abs(ensemble_metrics["r2"] - float(config["expected_ensemble_r2"]))
        ),
        "stored_validation_vs_heldout_spearman": spearman_correlation(validation_r2, heldout_r2),
        "stored_top10_exact_reproduction_probability": stored_top10_reproduction,
        "stored_top10_zero_success_95pct_upper_bound": (
            float(1.0 - 0.05 ** (1.0 / len(bootstrap.r2)))
            if stored_top10_reproduction == 0.0
            else None
        ),
        "bootstrap": {
            "unit": "equilibrium_files",
            "groups": bootstrap.group_count,
            "replicates": int(resolved["bootstrap_replicates"]),
            "single_se_relative_mc_error_approx": float(
                np.sqrt(1.0 / (2.0 * (len(bootstrap.r2) - 1)))
            ),
            "se_ratio_relative_mc_error_approx": float(
                np.sqrt(1.0 / (len(bootstrap.r2) - 1))
            ),
        },
        "tube_bootstrap_r2_se": tube_se,
        "equilibrium_grouped_bootstrap_r2_se": group_se,
        "grouped_to_tube_bootstrap_se_ratio": group_se / tube_se,
        "reference_tubes_per_equilibrium_file": float(len(actual) / len(group_sizes)),
        "nominal_independent_unit_count_inflation": float(len(actual) / len(group_sizes)),
        "perfect_within_equilibrium_design_effect": perfect_correlation_design_effect,
        "perfect_within_equilibrium_se_inflation": float(
            np.sqrt(perfect_correlation_design_effect)
        ),
        "split_audit": split_audit,
        "flux_regime_definition": flux_definition,
        "flux_regime_counts_reference": {
            str(value): int(np.sum(flux_labels == value))
            for value in np.unique(flux_labels)
        },
        "leakage_performance_ensemble": {
            "fixed_pair_split": {
                str(level): regression_metrics(
                    actual[fixed_pair_split == level],
                    ensemble_prediction[fixed_pair_split == level],
                )
                for level in np.unique(fixed_pair_split)
            },
            "equilibrium_in_train": {
                str(level): regression_metrics(
                    actual[equilibrium_in_train == level],
                    ensemble_prediction[equilibrium_in_train == level],
                )
                for level in np.unique(equilibrium_in_train)
            },
        },
        "gradient_tertile_cuts": {
            "a_over_LT": [float(value) for value in lt_cuts],
            "a_over_Ln": [float(value) for value in ln_cuts],
        },
        "panel": {
            "varied_rows": int(len(panel_rows)),
            "fixed_rows": int(len(panel_rows)),
            "unique_equilibrium_files": int(len(np.unique(equilibrium_files[panel_indices]))),
            "flux_regime_counts_varied": {
                str(value): int(np.sum(flux_labels[panel_indices] == value))
                for value in np.unique(flux_labels[panel_indices])
            },
            "equilibrium_class_counts_varied": {
                str(value): int(np.sum(equilibrium_class[panel_indices] == value))
                for value in np.unique(equilibrium_class[panel_indices])
            },
            "fixed_pair_split_counts": {
                SPLIT_NAMES[index]: int(
                    np.sum(assignments["fixed"][panel_rows] == index)
                )
                for index in range(len(SPLIT_NAMES))
            },
            "near_threshold_reference_population": int(
                np.sum(flux_labels == "near_threshold")
            ),
            "near_threshold_panel_count": int(
                np.sum(flux_labels[panel_indices] == "near_threshold")
            ),
            "near_threshold_analysis_limited": bool(
                np.sum(flux_labels == "near_threshold") < 30
            ),
            "large_error_count": int(
                np.sum(absolute_error[panel_indices] >= panel_sampling["error_top_decile_cut"])
            ),
            "high_disagreement_count": int(
                np.sum(disagreement[panel_indices] >= panel_sampling["disagreement_top_decile_cut"])
            ),
        },
        "fsa_grad_x_identity": identity,
        "scalar_feature_names": [str(value) for value in scalar_names],
    }

    artifacts.write_hdf5(
        "reference_predictions.h5",
        {
            "member_prediction_log_Q": predictions,
            "ensemble_prediction_log_Q": ensemble_prediction,
            "ensemble_spread_log_Q": disagreement,
            "actual_log_Q": actual,
            "row_id": analysis_rows,
        },
        axes={
            "member_prediction_log_Q": ("member", "sample"),
            "ensemble_prediction_log_Q": ("sample",),
            "ensemble_spread_log_Q": ("sample",),
            "actual_log_Q": ("sample",),
            "row_id": ("sample",),
        },
        attributes={"member_ids": list(member_ids), "estimand": "max(log Q, -2)"},
    )
    artifacts.write_hdf5(
        "interpretation_panel.h5",
        panel_arrays,
        axes=panel_axes,
        attributes={
            "gradient_set_codes": {"0": "varied", "1": "fixed"},
            "legacy_split_codes": {str(index): name for index, name in enumerate(SPLIT_NAMES)},
            "channel_names": list(CHANNEL_NAMES),
            "scalar_feature_names": [str(value) for value in scalar_names],
            "sample_order": "varied rows followed by paired fixed rows",
        },
    )
    artifacts.write_hdf5(
        "bootstrap.h5",
        {
            "r2": bootstrap.r2.astype(np.float32),
            "mse": bootstrap.mse.astype(np.float32),
            "bias": bootstrap.bias.astype(np.float32),
            "rank": bootstrap.ranks.astype(np.float32),
            "member_id": np.asarray(member_ids, dtype="S"),
        },
        axes={
            "r2": ("bootstrap", "member"),
            "mse": ("bootstrap", "member"),
            "bias": ("bootstrap", "member"),
            "rank": ("bootstrap", "member"),
            "member_id": ("member",),
        },
        attributes={"resampling_unit": "equilibrium_files"},
    )
    artifacts.write_hdf5(
        "panel_geometry_summary.h5",
        {
            **{
                name: values.astype(np.float32)
                for name, values in channel_correlations.items()
            },
            "median_normalized_non_dc_power_spectrum": spectrum.astype(np.float32),
            "fourier_mode": np.arange(1, spectrum.shape[1] + 1, dtype=np.int16),
        },
        axes={
            **{
                name: ("channel", "channel") for name in channel_correlations
            },
            "median_normalized_non_dc_power_spectrum": ("channel", "fourier_mode"),
            "fourier_mode": ("fourier_mode",),
        },
        attributes={"channel_names": list(CHANNEL_NAMES)},
    )
    summary_path = artifacts.write_json("summary.json", summary)
    cohorts_path = artifacts.write_json("cohorts.json", cohorts)
    overall_path = artifacts.write_text("member_performance.csv", _csv_text(overall_rows))
    strata_path = artifacts.write_text("stratified_performance.csv", _csv_text(stratified_rows))
    ranking_path = artifacts.write_text("bootstrap_ranking.csv", _csv_text(ranking_rows))
    channel_path = artifacts.write_text("channel_robust_scales.csv", _csv_text(channel_stats))
    panel_metadata_path = artifacts.write_text(
        "panel_metadata.csv", _csv_text(panel_metadata)
    )
    ranking_plot = output_dir.resolve() / "ranking_uncertainty.png"
    spectrum_plot = output_dir.resolve() / "panel_fourier_spectra.png"
    _plot_ranking(ranking_plot, validation_r2, heldout_r2, ci)
    _plot_spectra(spectrum_plot, spectrum)
    artifacts.register_existing(ranking_plot.name)
    artifacts.register_existing(spectrum_plot.name)

    publish_paths = [
        summary_path,
        cohorts_path,
        overall_path,
        strata_path,
        ranking_path,
        channel_path,
        panel_metadata_path,
        ranking_plot,
        spectrum_plot,
    ]
    manifest = artifacts.finalize(
        config=resolved,
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=member_ids,
        row_ids=panel_rows,
        gradient_set="fixed+varied",
        device=ensemble.device,
        repository=Path(__file__).resolve().parents[1],
    )
    if not pilot and not args.no_publish:
        _publish(publish_paths, args.published_dir)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"S01 {resolved['mode']} completed; manifest: {manifest}", flush=True)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(config, args)


if __name__ == "__main__":
    main()
