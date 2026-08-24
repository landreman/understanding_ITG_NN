#!/usr/bin/env python3
"""Explain ensemble disagreement and diagnose held-out common-mode failures."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from itg_nn.data import load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.disagreement import (
    diagnostic_association_rows,
    ensemble_spread,
    failure_categories,
    grouped_crossfit_ridge,
    member_residuals,
    paired_outcome_association_rows,
    perturbation_effect_rows,
    robust_scaled_channel_gradient,
    spread_input_gradient,
)
from itg_nn.xai.perturbations import (
    ScaledPCASupport,
    ValidityTag,
    independent_channel_shifts,
    random_joint_shift,
)
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import CANONICAL_FUNCTION, InvariantMember
from itg_nn.xai.unit_semantics import physics_concept_traces


NATIVE_ESTIMAND = "native max(log Q, -2)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/xai/S11_disagreement.json"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--published-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--members", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    return parser


def _resolve(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if args.pilot:
        resolved.update(config["pilot"])
    resolved["mode"] = "pilot" if args.pilot else "production"
    for name in ("device", "seed", "members", "batch_size"):
        value = getattr(args, name)
        if value is not None:
            resolved[name] = value
    if args.rows is not None:
        resolved["panel_varied_rows"] = args.rows
        resolved["gradient_rows"] = min(int(resolved["gradient_rows"]), args.rows)
        resolved["review_rows"] = min(int(resolved["review_rows"]), args.rows)
    for value, name in (
        (args.dataset, "dataset"),
        (args.checkpoint, "checkpoint"),
        (args.published_dir, "published_dir"),
    ):
        if value is not None:
            resolved[name] = str(value)
    resolved["source_hashes"] = {
        "runner": sha256_file(Path(__file__)),
        "library": sha256_file(Path(__file__).parents[1] / "itg_nn/xai/disagreement.py"),
        "config": sha256_file(args.config),
    }
    return resolved


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def _h5_take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return dataset[unique][inverse]


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
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


def _channel_scales(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["channel"]))
    scales = np.asarray([float(row["robust_sigma_iqr"]) for row in rows], dtype=np.float64)
    if scales.shape != (7,) or np.any(scales <= 0):
        raise ValueError("S01 must provide seven positive robust IQR scales")
    return scales


def _stratified_indices(stable: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count >= len(stable):
        return np.arange(len(stable), dtype=np.int64)
    rng = np.random.default_rng(seed)
    stable_rows, unstable_rows = np.flatnonzero(stable), np.flatnonzero(~stable)
    stable_count = min(len(stable_rows), max(1, round(count * stable.mean())))
    unstable_count = count - stable_count
    if unstable_count > len(unstable_rows):
        unstable_count = len(unstable_rows)
        stable_count = count - unstable_count
    chosen = np.concatenate((
        rng.choice(stable_rows, stable_count, replace=False),
        rng.choice(unstable_rows, unstable_count, replace=False),
    ))
    return np.sort(chosen.astype(np.int64))


def _panel_metadata(dataset: Path, row_ids: np.ndarray) -> dict[str, np.ndarray]:
    with h5py.File(dataset, "r") as handle:
        scalar_names = _decode(handle["scalar_features"][:])
        scalar = _h5_take(handle["scalar_feature_matrix"], row_ids).astype(np.float64)
        scalar_by_name = {name: scalar[:, index] for index, name in enumerate(scalar_names)}
        result = {
            "equilibrium_file": _decode(_h5_take(handle["equilibrium_files"], row_ids)),
            "equilibrium_class": _h5_take(handle["equilibrium_class"], row_ids).astype(np.int64),
            "q_stds": _h5_take(handle["varied_gradient_simulations/Q_stds"], row_ids).astype(np.float64),
        }
    for name in ("nfp", "iota", "shat", "d_pressure_d_s", "aspect", "rho"):
        if name not in scalar_by_name:
            raise RuntimeError(f"missing registered scalar feature {name}")
        result[name] = scalar_by_name[name]
    result["aspect_over_rho"] = result["aspect"] / result["rho"]
    return result


def _support_warning(
    dataset: Path,
    panel_geometry: np.ndarray,
    panel_groups: np.ndarray,
    *,
    count: int,
    fit_fraction: float,
    components: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    with h5py.File(dataset, "r") as handle:
        all_groups = _decode(handle["equilibrium_files"][:])
        eligible = np.flatnonzero(~np.isin(all_groups, np.unique(panel_groups)))
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        seen: set[str] = set()
        for position in rng.permutation(eligible):
            group = all_groups[position]
            if group in seen:
                continue
            selected.append(int(position))
            seen.add(group)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise RuntimeError("not enough equilibrium-unique support rows")
        support_geometry = _h5_take(handle["raw_feature_tensor"], np.asarray(selected)).astype(np.float64)
    split = int(count * fit_fraction)
    support = ScaledPCASupport.fit(
        support_geometry[:split], support_geometry[split:], components=components
    )
    return support.score(panel_geometry)["warning_score"], {
        "fit_rows": split,
        "heldout_rows": count - split,
    }


def _member_models(ensemble: Any, member_ids: Sequence[str]) -> dict[str, InvariantMember]:
    index = {member_id: position for position, member_id in enumerate(ensemble.member_ids)}
    return {member_id: InvariantMember(ensemble.models[index[member_id]]) for member_id in member_ids}


def _predict(
    models: Mapping[str, InvariantMember],
    member_ids: Sequence[str],
    geometry: torch.Tensor,
    a_lt: torch.Tensor,
    a_ln: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    original: bool = False,
) -> np.ndarray:
    output = np.empty((len(member_ids), len(geometry)), dtype=np.float32)
    with torch.inference_mode():
        for member_index, member_id in enumerate(member_ids):
            chunks = []
            model = models[member_id]
            for start in range(0, len(geometry), batch_size):
                stop = min(start + batch_size, len(geometry))
                x = geometry[start:stop].to(device)
                function = model.original if original else model.invariant
                chunks.append(function(x, a_lt[start:stop].to(device), a_ln[start:stop].to(device)).cpu().numpy())
            output[member_index] = np.concatenate(chunks)
    return output


def _spread_gradients(
    models: Mapping[str, InvariantMember],
    member_ids: Sequence[str],
    geometry: torch.Tensor,
    a_lt: torch.Tensor,
    a_ln: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions, gradients = [], []
    for start in range(0, len(geometry), batch_size):
        stop = min(start + batch_size, len(geometry))
        x = geometry[start:stop].to(device).detach().requires_grad_(True)
        outputs = torch.stack(
            [models[member_id].invariant(x, a_lt[start:stop].to(device), a_ln[start:stop].to(device)) for member_id in member_ids],
            dim=0,
        )
        result = spread_input_gradient(outputs, x)
        predictions.append(outputs.detach().cpu().numpy())
        gradients.append(result.gradient.detach().cpu().numpy())
        print(f"spread gradient rows {stop}/{len(geometry)}", flush=True)
    return np.concatenate(predictions, axis=1), np.concatenate(gradients, axis=0)


def _member_gradients_and_bottlenecks(
    models: Mapping[str, InvariantMember],
    member_ids: Sequence[str],
    geometry: torch.Tensor,
    a_lt: torch.Tensor,
    a_ln: torch.Tensor,
    gradient_indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    gradient_output = np.empty((len(member_ids), len(gradient_indices), 96, 7), dtype=np.float32)
    bottlenecks: list[np.ndarray] = []
    for member_index, member_id in enumerate(member_ids):
        model = models[member_id]
        bottleneck_chunks = []
        with torch.inference_mode():
            for start in range(0, len(geometry), batch_size):
                stop = min(start + batch_size, len(geometry))
                bottleneck_chunks.append(model.invariant_bottleneck(geometry[start:stop].to(device)).cpu().numpy())
        bottlenecks.append(np.concatenate(bottleneck_chunks).astype(np.float64))
        for start in range(0, len(gradient_indices), batch_size):
            positions = gradient_indices[start : start + batch_size]
            x = geometry[positions].to(device).detach().requires_grad_(True)
            output = model.invariant(x, a_lt[positions].to(device), a_ln[positions].to(device))
            gradient = torch.autograd.grad(output.sum(), x)[0]
            gradient_output[member_index, start : start + len(positions)] = gradient.detach().cpu().numpy()
        print(f"member gradient/bottleneck {member_index + 1}/{len(member_ids)} {member_id}", flush=True)
    return gradient_output, bottlenecks


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    center = array.mean(axis=0)
    scale = array.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return (array - center) / scale


def _motif_dispersion(
    path: Path, member_ids: Sequence[str], bottlenecks: Sequence[np.ndarray]
) -> np.ndarray:
    activation = {member_id: _standardize_columns(values) for member_id, values in zip(member_ids, bottlenecks)}
    motif_values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            traces = []
            for unit_id in row["unit_ids"].split("|"):
                member_id, unit = unit_id.rsplit(":u", 1)
                if member_id in activation:
                    traces.append(activation[member_id][:, int(unit)])
            if len(traces) >= 2:
                motif_values.append(np.std(np.stack(traces), axis=0, ddof=0))
    if not motif_values:
        return np.zeros(len(bottlenecks[0]), dtype=np.float64)
    return np.mean(np.stack(motif_values), axis=0)


def _concept_dispersion(
    geometry: np.ndarray,
    scales: np.ndarray,
    bottlenecks: Sequence[np.ndarray],
) -> np.ndarray:
    concepts = _standardize_columns(physics_concept_traces(geometry, channel_scales=scales).values.mean(axis=2))
    member_scores = []
    for values in bottlenecks:
        standardized = _standardize_columns(values)
        correlation = standardized.T @ concepts / len(concepts)
        norm = np.sqrt(np.sum(np.square(correlation), axis=0))
        norm[norm == 0] = 1.0
        member_scores.append((standardized @ correlation) / norm)
    return np.std(np.stack(member_scores), axis=0, ddof=0).mean(axis=1)


def _gradient_summary(
    spread_gradient: np.ndarray,
    member_gradient: np.ndarray,
    stable: np.ndarray,
    gradient_indices: np.ndarray,
    scales: np.ndarray,
    member_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scaled_spread = robust_scaled_channel_gradient(spread_gradient, scales)
    scaled_member = robust_scaled_channel_gradient(member_gradient, scales)
    sources: list[tuple[str, str, np.ndarray, np.ndarray]] = [
        ("ensemble_spread", "all_members", scaled_spread[None, ...], stable),
        ("ensemble_mean_prediction", "mean_top_gradient_members", scaled_member.mean(axis=0, keepdims=True), stable[gradient_indices]),
    ]
    sources.extend(
        ("member_residual", member_id, scaled_member[index : index + 1], stable[gradient_indices])
        for index, member_id in enumerate(member_ids)
    )
    for outcome, member_id, values, source_stable in sources:
        for regime, mask in (
            ("all", np.ones(len(source_stable), dtype=bool)),
            ("stable_or_near_floor", source_stable),
            ("unstable", ~source_stable),
        ):
            for channel in range(7):
                selected = values[:, mask, :, channel]
                rows.append({
                    "outcome": outcome,
                    "member_id": member_id,
                    "regime": regime,
                    "channel": channel,
                    "sample_count": int(mask.sum()),
                    "signed_mean_robust_scaled_gradient": float(selected.mean()),
                    "mean_absolute_robust_scaled_gradient": float(np.abs(selected).mean()),
                    "signs_retained": True,
                    "estimand": NATIVE_ESTIMAND,
                    "gradient_identity": "d(member residual)/dX = d(member native prediction)/dX" if outcome == "member_residual" else "direct_autograd",
                })
    return rows


def _perturbation_summary(
    reference: np.ndarray,
    edited_by_name: Mapping[str, np.ndarray],
    validity: Mapping[str, ValidityTag],
    stable: np.ndarray,
    member_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference_spread = ensemble_spread(reference)
    for name, edited in edited_by_name.items():
        difference = edited - reference
        spread_difference = ensemble_spread(edited) - reference_spread
        for regime, mask in (
            ("all", np.ones(len(stable), dtype=bool)),
            ("stable_or_near_floor", stable),
            ("unstable", ~stable),
        ):
            for index, member_id in enumerate(member_ids):
                values = difference[index, mask]
                rows.append({
                    "perturbation": name,
                    "validity": validity[name].value,
                    "outcome": "member_native_prediction",
                    "member_id": member_id,
                    "regime": regime,
                    "sample_count": int(mask.sum()),
                    "signed_mean_change_native": float(values.mean()),
                    "rms_change_native": float(np.sqrt(np.mean(np.square(values)))),
                    "estimand": NATIVE_ESTIMAND,
                })
            values = spread_difference[mask]
            rows.append({
                "perturbation": name,
                "validity": validity[name].value,
                "outcome": "ensemble_spread",
                "member_id": "all_members_population_std",
                "regime": regime,
                "sample_count": int(mask.sum()),
                "signed_mean_change_native": float(values.mean()),
                "rms_change_native": float(np.sqrt(np.mean(np.square(values)))),
                "estimand": NATIVE_ESTIMAND,
            })
    return rows


def _native_row_diagnostics(
    predictions: np.ndarray, target: np.ndarray, original_shift_signed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute native ensemble diagnostics without cancelling member shift magnitudes."""
    spread = ensemble_spread(predictions)
    residuals = member_residuals(predictions, target)
    ensemble_mean = predictions.mean(axis=0)
    ensemble_error = np.abs(ensemble_mean - target)
    member_symmetry = np.abs(original_shift_signed).mean(axis=0)
    return spread, residuals, ensemble_mean, ensemble_error, member_symmetry


def _crossfit_rows(
    features: Mapping[str, np.ndarray],
    classes: np.ndarray,
    outcomes: Mapping[str, np.ndarray],
    groups: np.ndarray,
    *,
    folds: int,
    alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    continuous = np.column_stack([features[name] for name in features])
    class_values = sorted(np.unique(classes).tolist())
    one_hot = (
        np.column_stack([classes == value for value in class_values[1:]]).astype(float)
        if len(class_values) > 1
        else np.empty((len(classes), 0), dtype=float)
    )
    design = np.column_stack((continuous, one_hot))
    rows = []
    for index, (name, outcome) in enumerate(outcomes.items()):
        result = grouped_crossfit_ridge(
            design, outcome, groups, folds=folds, alpha=alpha, seed=seed + index
        )
        residual = outcome - result.predictions
        denominator = np.sum(np.square(outcome - outcome.mean()))
        rows.append({
            "outcome": name,
            "heldout_r2": float(1 - np.sum(np.square(residual)) / denominator) if denominator > 0 else float("nan"),
            "heldout_mae_native": float(np.mean(np.abs(residual))),
            "folds": folds,
            "split_unit": "equilibrium_files",
            "feature_selection": "none_frozen_before_residual_analysis",
            "features": "|".join([*features, "equilibrium_class_one_hot"]),
            "ridge_alpha": alpha,
        })
    return rows


def _category_rows(categories: np.ndarray, stable: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for regime, mask in (
        ("all", np.ones(len(stable), dtype=bool)),
        ("stable_or_near_floor", stable),
        ("unstable", ~stable),
    ):
        for category in (
            "high_spread_low_error", "high_spread_high_error", "common_mode_failure", "unanimous_success"
        ):
            count = int(np.sum(mask & (categories == category)))
            rows.append({"regime": regime, "category": category, "count": count, "fraction": count / int(mask.sum())})
    return rows


def _threshold_sensitivity_rows(
    spread: np.ndarray,
    error: np.ndarray,
    stable: np.ndarray,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for spread_factor in config["threshold_sensitivity_factors"]:
        for error_factor in config["threshold_sensitivity_factors"]:
            spread_threshold = float(config["high_spread_threshold_native"]) * float(spread_factor)
            error_threshold = float(config["high_error_threshold_native"]) * float(error_factor)
            categories = failure_categories(
                spread,
                error,
                high_spread_threshold=spread_threshold,
                high_error_threshold=error_threshold,
            )
            for regime, mask in (
                ("all", np.ones(len(stable), dtype=bool)),
                ("stable_or_near_floor", stable),
                ("unstable", ~stable),
            ):
                rows.append({
                    "spread_threshold_native": spread_threshold,
                    "error_threshold_native": error_threshold,
                    "regime": regime,
                    "sample_count": int(mask.sum()),
                    "common_mode_failure_count": int(np.sum(mask & (categories == "common_mode_failure"))),
                    "common_mode_failure_fraction": float(np.mean(categories[mask] == "common_mode_failure")),
                    "threshold_status": (
                        "registered_primary"
                        if spread_factor == 1.0 and error_factor == 1.0
                        else "post_review_sensitivity"
                    ),
                })
    return rows


def _class_rows(
    classes: np.ndarray, stable: np.ndarray, spread: np.ndarray, error: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for regime, regime_mask in (
        ("all", np.ones(len(stable), dtype=bool)),
        ("stable_or_near_floor", stable),
        ("unstable", ~stable),
    ):
        for class_value in sorted(np.unique(classes).tolist()):
            mask = regime_mask & (classes == class_value)
            rows.append({
                "regime": regime,
                "equilibrium_class": int(class_value),
                "sample_count": int(mask.sum()),
                "mean_ensemble_spread_native": float(spread[mask].mean()) if mask.any() else float("nan"),
                "mean_ensemble_absolute_error_native": float(error[mask].mean()) if mask.any() else float("nan"),
            })
    return rows


def _plot_failure_atlas(
    path: Path, spread: np.ndarray, error: np.ndarray, categories: np.ndarray, category_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    colors = {
        "high_spread_low_error": "#3b82f6", "high_spread_high_error": "#ef4444",
        "common_mode_failure": "#7c3aed", "unanimous_success": "#16a34a",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for category, color in colors.items():
        mask = categories == category
        axes[0].scatter(spread[mask], error[mask], s=14, alpha=0.65, label=category.replace("_", " "), color=color)
    axes[0].axvline(float(config["high_spread_threshold_native"]), color="black", linestyle="--")
    axes[0].axhline(float(config["high_error_threshold_native"]), color="black", linestyle="--")
    axes[0].set(xlabel="ensemble spread (native units)", ylabel="ensemble absolute error (native units)")
    axes[0].legend(fontsize=7)
    all_rows = [row for row in category_rows if row["regime"] == "all"]
    axes[1].bar([row["category"].replace("_", "\n") for row in all_rows], [row["count"] for row in all_rows], color=[colors[row["category"]] for row in all_rows])
    axes[1].set_ylabel("panel rows")
    axes[1].tick_params(axis="x", labelsize=7)
    figure.suptitle("S11 disagreement and common-mode failure atlas")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _publish(paths: Sequence[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, destination / path.name)


def _resume(output_dir: Path, resolved: Mapping[str, Any], dataset: Path, checkpoint: Path) -> bool:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("config") != dict(resolved):
        raise RuntimeError("resume config or source hashes differ")
    if manifest.get("dataset", {}).get("sha256") != sha256_file(dataset):
        raise RuntimeError("resume dataset fingerprint differs")
    if manifest.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint):
        raise RuntimeError("resume checkpoint fingerprint differs")
    for name, expected in manifest.get("output_hashes", {}).items():
        if not (output_dir / name).is_file() or sha256_file(output_dir / name) != expected:
            raise RuntimeError(f"resume output hash mismatch: {name}")
    print("S11 resume validated config, sources, inputs, and output hashes", flush=True)
    return True


def run(args: argparse.Namespace) -> Path:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = _resolve(config, args)
    if resolved["canonical_function"] != CANONICAL_FUNCTION or resolved["estimand"] != NATIVE_ESTIMAND:
        raise ValueError("S11 must explain the S02 canonical native output")
    if resolved["resampling_unit"] != "equilibrium_files":
        raise ValueError("S11 must group by equilibrium_files")
    set_deterministic_seed(int(resolved["seed"]))
    dataset, checkpoint = Path(resolved["dataset"]), Path(resolved["checkpoint"])
    output_dir = args.output_dir.resolve() if args.output_dir else (Path("output/xai/S11") / resolved["run_id"]).resolve()
    if args.resume and _resume(output_dir, resolved, dataset, checkpoint):
        return output_dir
    artifacts = RunArtifacts(output_dir)
    cohorts = json.loads(Path(resolved["cohorts"]).read_text(encoding="utf-8"))
    registered = np.asarray(cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64)
    row_ids = registered[: int(resolved["panel_varied_rows"])]
    panel = load_hdf5_rows(dataset, row_ids, gradient_set="varied", include_targets=True)
    if panel.actual_log_heat_flux is None:
        raise RuntimeError("native varied-gradient targets are required")
    target = panel.actual_log_heat_flux.numpy().astype(np.float64)
    stable = target <= float(resolved["stable_threshold_log_Q"])
    if not stable.any() or stable.all():
        raise RuntimeError("both stable/near-floor and unstable rows are required")
    metadata = _panel_metadata(dataset, row_ids)
    groups = metadata["equilibrium_file"]
    if len(np.unique(groups)) != len(groups):
        raise RuntimeError("the frozen S01 panel must contain one row per equilibrium")
    scales = _channel_scales(Path(resolved["channel_scales"]))
    support_warning, support_counts = _support_warning(
        dataset, panel.geometry.numpy(), groups,
        count=int(resolved["support_reference_rows"]),
        fit_fraction=float(resolved["support_fit_fraction"]),
        components=int(resolved["support_components"]), seed=int(resolved["seed"]) + 100,
    )
    ensemble = load_ensemble(checkpoint, device=resolved["device"])
    all_ids = tuple(cohorts["member_cohorts"]["all_100"])
    member_ids = all_ids[: int(resolved["members"])]
    if len(member_ids) < 2:
        raise ValueError("ensemble disagreement requires at least two members")
    models = _member_models(ensemble, member_ids)
    predictions, spread_gradient = _spread_gradients(
        models, member_ids, panel.geometry, panel.a_over_lt, panel.a_over_ln,
        batch_size=int(resolved["batch_size"]), device=ensemble.device,
    )
    spread, residuals, ensemble_mean, ensemble_error, _ = _native_row_diagnostics(
        predictions, target, np.zeros((1, predictions.shape[1]), dtype=np.float64)
    )
    categories = failure_categories(
        spread, ensemble_error,
        high_spread_threshold=float(resolved["high_spread_threshold_native"]),
        high_error_threshold=float(resolved["high_error_threshold_native"]),
    )
    gradient_ids = member_ids[: min(int(resolved["gradient_members"]), len(member_ids))]
    gradient_indices = _stratified_indices(stable, int(resolved["gradient_rows"]), int(resolved["seed"]) + 1)
    member_gradient, bottlenecks = _member_gradients_and_bottlenecks(
        models, gradient_ids, panel.geometry, panel.a_over_lt, panel.a_over_ln, gradient_indices,
        batch_size=int(resolved["batch_size"]), device=ensemble.device,
    )
    motif_dispersion = _motif_dispersion(Path(resolved["motif_catalog"]), gradient_ids, bottlenecks)
    concept_dispersion = _concept_dispersion(panel.geometry.numpy(), scales, bottlenecks)
    exact_geometry = random_joint_shift(panel.geometry, seed=int(resolved["seed"]) + 2)
    off_manifold_geometry = independent_channel_shifts(panel.geometry, seed=int(resolved["seed"]) + 3)
    exact_predictions = _predict(models, member_ids, exact_geometry, panel.a_over_lt, panel.a_over_ln, batch_size=int(resolved["batch_size"]), device=ensemble.device)
    off_manifold_predictions = _predict(models, member_ids, off_manifold_geometry, panel.a_over_lt, panel.a_over_ln, batch_size=int(resolved["batch_size"]), device=ensemble.device)
    original_ids = gradient_ids
    original_reference = _predict(models, original_ids, panel.geometry, panel.a_over_lt, panel.a_over_ln, batch_size=int(resolved["batch_size"]), device=ensemble.device, original=True)
    original_shifted = _predict(models, original_ids, exact_geometry, panel.a_over_lt, panel.a_over_ln, batch_size=int(resolved["batch_size"]), device=ensemble.device, original=True)
    original_shift_signed = original_shifted - original_reference
    symmetry_error = np.abs(original_shift_signed.mean(axis=0))
    _, _, _, _, symmetry_member_mean_absolute = _native_row_diagnostics(
        predictions, target, original_shift_signed
    )
    diagnostic_features = {
        "support_warning_score": support_warning,
        "a_over_lt": panel.a_over_lt.numpy().astype(np.float64),
        "a_over_ln": panel.a_over_ln.numpy().astype(np.float64),
        "q_stds": metadata["q_stds"],
        "symmetry_error": symmetry_error,
        "motif_activation_dispersion": motif_dispersion,
        "concept_activation_dispersion": concept_dispersion,
        "nfp": metadata["nfp"], "iota": metadata["iota"], "shat": metadata["shat"],
        "d_pressure_d_s": metadata["d_pressure_d_s"], "aspect": metadata["aspect"],
        "rho": metadata["rho"], "aspect_over_rho": metadata["aspect_over_rho"],
    }
    expected = [name for name in resolved["diagnostic_features"] if name != "equilibrium_class"]
    if list(diagnostic_features) != expected:
        raise RuntimeError("runner diagnostic features differ from the frozen config")
    outcomes = {"ensemble_spread": spread, "ensemble_absolute_error": ensemble_error}
    regimes = {
        "all": np.ones(len(stable), dtype=bool),
        "stable_or_near_floor": stable,
        "unstable": ~stable,
    }
    association_rows = diagnostic_association_rows(
        diagnostic_features, outcomes, groups, regimes,
        replicates=int(resolved["bootstrap_replicates"]), seed=int(resolved["seed"]) + 10,
    )
    member_symmetry_rows = diagnostic_association_rows(
        {"member_mean_absolute_shift_error_top10": symmetry_member_mean_absolute},
        outcomes,
        groups,
        regimes,
        replicates=int(resolved["bootstrap_replicates"]),
        seed=int(resolved["seed"]) + 15,
    )
    spread_error_rows = paired_outcome_association_rows(
        spread,
        ensemble_error,
        groups,
        regimes,
        left_name="ensemble_spread",
        right_name="ensemble_absolute_error",
        replicates=int(resolved["bootstrap_replicates"]),
        seed=int(resolved["seed"]) + 17,
    )
    crossfit_rows = _crossfit_rows(
        diagnostic_features, metadata["equilibrium_class"], outcomes, groups,
        folds=int(resolved["crossfit_folds"]), alpha=float(resolved["crossfit_ridge_alpha"]), seed=int(resolved["seed"]) + 20,
    )
    gradient_rows = _gradient_summary(spread_gradient, member_gradient, stable, gradient_indices, scales, gradient_ids)
    edited = {"random_joint_shift": exact_predictions, "independent_channel_shifts": off_manifold_predictions}
    validity = {"random_joint_shift": ValidityTag.EXACT_SYMMETRY, "independent_channel_shifts": ValidityTag.OFF_MANIFOLD}
    perturbation_rows = _perturbation_summary(predictions, edited, validity, stable, member_ids)
    signed_effect_rows = []
    for name, values in edited.items():
        signed_effect_rows.extend(perturbation_effect_rows(
            predictions, values, member_ids=member_ids, row_ids=row_ids,
            perturbation=name, validity=validity[name],
        ))
    category_rows = _category_rows(categories, stable)
    threshold_rows = _threshold_sensitivity_rows(spread, ensemble_error, stable, resolved)
    class_rows = _class_rows(metadata["equilibrium_class"], stable, spread, ensemble_error)
    row_diagnostics = []
    for position, row_id in enumerate(row_ids):
        row_diagnostics.append({
            "row_id": int(row_id), "equilibrium_file": groups[position],
            "equilibrium_class": int(metadata["equilibrium_class"][position]),
            "stability": "stable_or_near_floor" if stable[position] else "unstable",
            "target_native": target[position], "ensemble_mean_native": ensemble_mean[position],
            "ensemble_spread_native": spread[position], "ensemble_absolute_error_native": ensemble_error[position],
            "failure_category": categories[position],
            "member_mean_absolute_shift_error_top10": symmetry_member_mean_absolute[position],
            **{name: values[position] for name, values in diagnostic_features.items()},
        })
    summary = {
        "step": "S11", "run_id": resolved["run_id"], "estimand": NATIVE_ESTIMAND,
        "canonical_function": CANONICAL_FUNCTION, "members": len(member_ids), "rows": len(row_ids),
        "stable_or_near_floor_rows": int(stable.sum()), "unstable_rows": int((~stable).sum()),
        "ensemble_spread_median_native": float(np.median(spread)),
        "ensemble_absolute_error_median_native": float(np.median(ensemble_error)),
        "common_mode_failure_rows": int(np.sum(categories == "common_mode_failure")),
        "common_mode_failure_fraction": float(np.mean(categories == "common_mode_failure")),
        "high_spread_high_error_rows": int(np.sum(categories == "high_spread_high_error")),
        "spread_error_spearman": float(spread_error_rows[0]["spearman"]),
        "spread_error_pearson": float(spread_error_rows[0]["pearson"]),
        "model_spread_interpretation": "member dispersion, not a confidence interval",
        "residual_feature_selection": resolved["feature_selection"],
        "support_reference": support_counts,
        "perturbation_validity": {name: tag.value for name, tag in validity.items()},
        "deferred": ["task 3 detailed case-study narratives", "task 4 opposing-strategy cancellation analysis"],
    }
    row_path = artifacts.write_text("row_diagnostics.csv", _csv_text(row_diagnostics))
    association_path = artifacts.write_text("diagnostic_associations.csv", _csv_text(association_rows))
    member_symmetry_path = artifacts.write_text("member_symmetry_associations.csv", _csv_text(member_symmetry_rows))
    spread_error_path = artifacts.write_text("spread_error_associations.csv", _csv_text(spread_error_rows))
    crossfit_path = artifacts.write_text("crossfit_diagnostics.csv", _csv_text(crossfit_rows))
    gradient_path = artifacts.write_text("gradient_summary.csv", _csv_text(gradient_rows))
    perturbation_path = artifacts.write_text("perturbation_summary.csv", _csv_text(perturbation_rows))
    category_path = artifacts.write_text("failure_categories.csv", _csv_text(category_rows))
    threshold_path = artifacts.write_text("failure_threshold_sensitivity.csv", _csv_text(threshold_rows))
    class_path = artifacts.write_text("equilibrium_class_diagnostics.csv", _csv_text(class_rows))
    summary_path = artifacts.write_json("summary.json", summary)
    full_h5 = artifacts.write_hdf5(
        "member_level_diagnostics.h5",
        {
            "member_id": np.asarray([value.encode() for value in member_ids]),
            "row_id": row_ids, "gradient_row_id": row_ids[gradient_indices],
            "prediction_native": predictions.astype(np.float32),
            "member_residual_native": residuals.astype(np.float32),
            "ensemble_spread_gradient": spread_gradient.astype(np.float32),
            "member_residual_gradient": member_gradient.astype(np.float32),
            "original_shift_signed_native": original_shift_signed.astype(np.float32),
            "perturbed_prediction_native": np.stack([edited[name] for name in edited]).astype(np.float32),
            "perturbation": np.asarray([name.encode() for name in edited]),
        },
        axes={
            "member_id": ("member",), "row_id": ("sample",), "gradient_row_id": ("gradient_sample",),
            "prediction_native": ("member", "sample"), "member_residual_native": ("member", "sample"),
            "ensemble_spread_gradient": ("sample", "z", "channel"),
            "member_residual_gradient": ("gradient_member", "gradient_sample", "z", "channel"),
            "original_shift_signed_native": ("gradient_member", "sample"),
            "perturbed_prediction_native": ("perturbation", "member", "sample"), "perturbation": ("perturbation",),
        },
        attributes={"estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION, "signs_retained": True}, compression="gzip",
    )
    signed_path = artifacts.write_text("signed_perturbation_effects.csv", _csv_text(signed_effect_rows))
    symmetry_signed_rows = []
    for member_index, member_id in enumerate(original_ids):
        for sample_index, row_id in enumerate(row_ids):
            symmetry_signed_rows.append({
                "member_id": member_id,
                "row_id": int(row_id),
                "function": "original_f",
                "perturbation": "random_joint_shift",
                "validity": ValidityTag.EXACT_SYMMETRY.value,
                "signed_change_native": float(original_shift_signed[member_index, sample_index]),
                "absolute_change_native": float(abs(original_shift_signed[member_index, sample_index])),
                "estimand": NATIVE_ESTIMAND,
            })
    symmetry_signed_path = artifacts.write_text(
        "signed_member_symmetry_changes.csv", _csv_text(symmetry_signed_rows)
    )
    review_indices = gradient_indices[: int(resolved["review_rows"])]
    member_gradient_lookup = {int(position): index for index, position in enumerate(gradient_indices)}
    review_gradient_positions = np.asarray([member_gradient_lookup[int(position)] for position in review_indices])
    review_path = artifacts.write_hdf5(
        "selected_review_diagnostics.h5",
        {
            "member_id": np.asarray([value.encode() for value in member_ids]),
            "gradient_member_id": np.asarray([value.encode() for value in gradient_ids]),
            "row_id": row_ids[review_indices],
            "prediction_native": predictions[:, review_indices].astype(np.float32),
            "ensemble_spread_gradient": spread_gradient[review_indices].astype(np.float32),
            "member_residual_gradient": member_gradient[:, review_gradient_positions].astype(np.float32),
            "original_shift_signed_native": original_shift_signed[:, review_indices].astype(np.float32),
        },
        axes={
            "member_id": ("member",), "gradient_member_id": ("gradient_member",), "row_id": ("sample",),
            "prediction_native": ("member", "sample"), "ensemble_spread_gradient": ("sample", "z", "channel"),
            "member_residual_gradient": ("gradient_member", "sample", "z", "channel"),
            "original_shift_signed_native": ("gradient_member", "sample"),
        }, attributes={"estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION, "signs_retained": True}, compression="gzip",
    )
    figure_path = output_dir / "failure_atlas.png"
    _plot_failure_atlas(figure_path, spread, ensemble_error, categories, category_rows, resolved)
    artifacts.register_existing(figure_path.name)
    publish_paths = [
        row_path, association_path, member_symmetry_path, spread_error_path,
        crossfit_path, gradient_path, perturbation_path, category_path,
        threshold_path, class_path, summary_path, symmetry_signed_path,
        review_path, figure_path,
    ]
    published_dir = None if args.no_publish else Path(resolved["published_dir"])
    if published_dir is not None:
        _publish(publish_paths, published_dir)
    artifacts.finalize(
        config=resolved, dataset=dataset, checkpoint=checkpoint, member_ids=member_ids,
        row_ids=row_ids, gradient_set="varied frozen S01 interpretation panel",
        device=ensemble.device, repository=Path(__file__).resolve().parents[1],
        published_dir=published_dir,
        extra_manifest={
            "gradient_member_ids": list(gradient_ids), "gradient_row_ids": row_ids[gradient_indices].tolist(),
            "review_row_ids": row_ids[review_indices].tolist(), "large_member_level_artifacts": [full_h5.name, signed_path.name],
            "reviewer_recomputable_artifact": review_path.name,
        },
    )
    return output_dir


def main() -> None:
    output = run(build_parser().parse_args())
    print(output)


if __name__ == "__main__":
    main()
