#!/usr/bin/env python3
"""Run S12 invariant-feature EBM distillation."""

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

from itg_nn.data import load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.distillation import (
    DistillationResult,
    grouped_ebm_crossfit,
    grouped_r2_difference_interval,
    grouped_r2_interval,
    grouped_term_recurrence,
    invariant_feature_table,
)
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import CANONICAL_FUNCTION, InvariantMember


NATIVE_ESTIMAND = "native max(log Q, -2)"
OBSERVED = "observed-comparison"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xai/S12_distillation.json")
    )
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
    for value, name in (
        (args.dataset, "dataset"),
        (args.checkpoint, "checkpoint"),
        (args.published_dir, "published_dir"),
    ):
        if value is not None:
            resolved[name] = str(value)
    return resolved


def _csv_text(rows: list[dict[str, Any]]) -> str:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
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
    unique, inverse = np.unique(rows, return_inverse=True)
    return dataset[unique][inverse]


def _channel_scales(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    ordered = sorted(rows, key=lambda row: int(row["channel"]))
    scales = np.asarray([float(row["iqr"]) for row in ordered], dtype=np.float64)
    if scales.shape != (7,) or np.any(scales <= 0):
        raise RuntimeError(
            "S01 channel scale artifact must provide seven positive IQRs"
        )
    return scales


def _metadata(
    dataset: Path, rows: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    with h5py.File(dataset, "r") as handle:
        scalar_names = tuple(_decode(handle["scalar_features"][:]))
        scalars = _h5_take(handle["scalar_feature_matrix"], rows).astype(np.float64)
        groups = _decode(_h5_take(handle["equilibrium_files"], rows))
    return scalars, scalar_names, groups


def _member_values(
    member: InvariantMember,
    panel,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    bottlenecks: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(panel.geometry), batch_size):
            stop = start + batch_size
            geometry = panel.geometry[start:stop].to(device)
            a_lt = panel.a_over_lt[start:stop].to(device)
            a_ln = panel.a_over_ln[start:stop].to(device)
            bottlenecks.append(member.invariant_bottleneck(geometry).cpu().numpy())
            predictions.append(member(geometry, a_lt, a_ln).cpu().numpy())
    return (
        np.concatenate(bottlenecks).astype(np.float64),
        np.concatenate(predictions).astype(np.float64),
    )


def _factory(
    resolved: dict[str, Any], feature_names: tuple[str, ...]
) -> Callable[..., Any]:
    from interpret.glassbox import ExplainableBoostingRegressor

    def make(*, seed: int, interactions=()):
        return ExplainableBoostingRegressor(
            feature_names=list(feature_names),
            interactions=list(interactions),
            max_bins=int(resolved["ebm_max_bins"]),
            max_interaction_bins=int(resolved["ebm_max_interaction_bins"]),
            validation_size=0.15,
            outer_bags=int(resolved["ebm_outer_bags"]),
            inner_bags=0,
            learning_rate=float(resolved["ebm_learning_rate"]),
            max_rounds=int(resolved["ebm_max_rounds"]),
            min_samples_leaf=int(resolved["ebm_min_samples_leaf"]),
            random_state=int(seed),
            n_jobs=1,
        )

    return make


def _r2(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    y = target[mask]
    p = prediction[mask]
    denominator = float(np.sum(np.square(y - y.mean())))
    return (
        float("nan")
        if denominator <= 0
        else 1.0 - float(np.sum(np.square(y - p)) / denominator)
    )


def _fidelity_rows(
    target_kind: str,
    target_id: str,
    target: np.ndarray,
    result: DistillationResult,
    stable: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for regime, mask in (
        ("all", np.ones(len(target), dtype=bool)),
        ("stable_or_near_floor", stable),
        ("unstable", ~stable),
    ):
        residual = result.prediction[mask] - target[mask]
        row = {
            "target_kind": target_kind,
            "target_id": target_id,
            "regime": regime,
            "rows": int(mask.sum()),
            "held_out_r2": _r2(target, result.prediction, mask),
            "held_out_mse": float(np.mean(np.square(residual))),
            "bias_formula_minus_target": float(np.mean(residual)),
            "residual_std": float(np.std(residual)),
            "split_unit": "equilibrium_files",
            "model": "ExplainableBoostingRegressor",
            "model_version": "interpret-core==0.7.8",
            "estimand": NATIVE_ESTIMAND,
            "canonical_function": CANONICAL_FUNCTION,
            "validity_tag": OBSERVED,
            "held_out_r2_ci95_lower": "",
            "held_out_r2_ci95_upper": "",
            "fidelity_bootstrap_replicates": "",
            "fidelity_bootstrap_unit": "",
        }
        if regime == "all" and bootstrap_replicates:
            if groups is None:
                raise ValueError("groups are required for fidelity intervals")
            _, lower, upper = grouped_r2_interval(
                target,
                result.prediction,
                groups,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            row.update(
                {
                    "held_out_r2_ci95_lower": lower,
                    "held_out_r2_ci95_upper": upper,
                    "fidelity_bootstrap_replicates": bootstrap_replicates,
                    "fidelity_bootstrap_unit": "equilibrium_files",
                }
            )
        rows.append(row)
    return rows


def _nested_subset_spec(
    spec: dict[str, Any],
    feature_names: tuple[str, ...],
    registered_interactions: list[tuple[str, str]],
) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[int, int], ...]]:
    names = feature_names if spec["features"] == "all" else tuple(spec["features"])
    missing = set(names) - set(feature_names)
    if missing:
        raise ValueError(
            f"nested feature set contains unknown features: {sorted(missing)}"
        )
    interaction_names = (
        registered_interactions
        if spec["interactions"] == "registered"
        else [tuple(pair) for pair in spec["interactions"]]
    )
    if any(
        left not in names or right not in names for left, right in interaction_names
    ):
        raise ValueError("nested interactions must refer to features in their subset")
    positions = np.asarray(
        [feature_names.index(name) for name in names], dtype=np.int64
    )
    interaction_indices = tuple(
        (names.index(left), names.index(right)) for left, right in interaction_names
    )
    return positions, names, interaction_indices


def _importance_rows(
    target_kind: str,
    target_id: str,
    result: DistillationResult,
    top_k: int,
) -> list[dict[str, Any]]:
    by_feature = {
        feature: np.asarray(
            [
                float(row["importance"])
                for row in result.term_rows
                if row["term_name"] == feature
            ]
        )
        for feature in result.feature_names
    }
    top_by_fold = []
    for fold in np.unique(result.fold):
        fold_rows = [row for row in result.term_rows if int(row["fold"]) == int(fold)]
        fold_rows.sort(key=lambda row: -float(row["importance"]))
        top_by_fold.append({str(row["term_name"]) for row in fold_rows[:top_k]})
    return [
        {
            "target_kind": target_kind,
            "target_id": target_id,
            "feature_name": feature,
            "mean_fold_importance": float(np.mean(by_feature[feature])),
            "std_fold_importance": float(np.std(by_feature[feature])),
            "top_k_fold_recurrence": float(
                np.mean([feature in selected for selected in top_by_fold])
            ),
            "folds": len(top_by_fold),
            "split_unit": "equilibrium_files",
            "importance_method": "mean_absolute_EBM_term_contribution",
            "validity_tag": OBSERVED,
        }
        for feature in result.feature_names
    ]


def _global_effect_rows(
    target_kind: str,
    target_id: str,
    estimator: Any,
) -> list[dict[str, Any]]:
    explanation = estimator.explain_global()
    rows: list[dict[str, Any]] = []
    for term_index, term_name in enumerate(estimator.term_names_):
        data = explanation.data(term_index)
        scores = np.asarray(data["scores"], dtype=np.float64)
        if data["type"] == "univariate":
            indices = np.unique(
                np.linspace(0, len(scores) - 1, min(32, len(scores)), dtype=int)
            )
            for index in indices:
                point = data["names"][index]
                score = scores[index]
                rows.append(
                    {
                        "target_kind": target_kind,
                        "target_id": target_id,
                        "term_name": term_name,
                        "term_kind": "main_effect",
                        "x": float(point),
                        "y": "",
                        "signed_effect_native_units": float(score),
                        "estimand": NATIVE_ESTIMAND,
                    }
                )
        else:
            # Interpret exposes interaction bin edges in ``*_names`` and one
            # score per interval.  The names can therefore be one element
            # longer than the score axis; serialize only scored coordinates.
            left = data["left_names"][: scores.shape[0]]
            right = data["right_names"][: scores.shape[1]]
            left_indices = np.unique(
                np.linspace(0, len(left) - 1, min(8, len(left)), dtype=int)
            )
            right_indices = np.unique(
                np.linspace(0, len(right) - 1, min(8, len(right)), dtype=int)
            )
            for left_index in left_indices:
                x_value = left[left_index]
                for right_index in right_indices:
                    y_value = right[right_index]
                    rows.append(
                        {
                            "target_kind": target_kind,
                            "target_id": target_id,
                            "term_name": term_name,
                            "term_kind": "pairwise_interaction",
                            "x": float(x_value),
                            "y": float(y_value),
                            "signed_effect_native_units": float(
                                scores[left_index, right_index]
                            ),
                            "estimand": NATIVE_ESTIMAND,
                        }
                    )
    return rows


def _plot_effects(
    path: Path, effects: list[dict[str, Any]], target_ids: list[str]
) -> None:
    figure, axes = plt.subplots(
        len(target_ids), 2, figsize=(12, 3.5 * len(target_ids)), squeeze=False
    )
    for row_index, target_id in enumerate(target_ids):
        selected = [
            row
            for row in effects
            if row["target_id"] == target_id and row["term_kind"] == "main_effect"
        ]
        terms = list(dict.fromkeys(str(row["term_name"]) for row in selected))[:2]
        for column, term in enumerate(terms):
            term_rows = [row for row in selected if row["term_name"] == term]
            axes[row_index, column].plot(
                [float(row["x"]) for row in term_rows],
                [float(row["signed_effect_native_units"]) for row in term_rows],
            )
            axes[row_index, column].set_title(f"{target_id}: {term}")
            axes[row_index, column].set_ylabel("signed EBM effect (native units)")
            axes[row_index, column].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    repository = Path(__file__).resolve().parents[1]
    resolved = _resolve(json.loads(args.config.read_text(encoding="utf-8")), args)
    dataset = Path(resolved["dataset"]).resolve()
    checkpoint = (repository / resolved["checkpoint"]).resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (repository / "output/xai/S12" / resolved["run_id"]).resolve()
    )
    published = (
        None
        if args.no_publish or resolved["mode"] == "pilot"
        else (repository / resolved["published_dir"]).resolve()
    )
    if args.resume and (output_dir / "manifest.json").is_file():
        manifest = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest["dataset"]["sha256"] != sha256_file(dataset) or manifest[
            "checkpoint"
        ]["sha256"] != sha256_file(checkpoint):
            raise RuntimeError("resume input fingerprint changed")
        return output_dir

    set_deterministic_seed(int(resolved["seed"]))
    registry = json.loads(
        (repository / resolved["cohorts"]).read_text(encoding="utf-8")
    )
    row_ids = np.asarray(
        registry["interpretation_panel"]["varied_row_ids"][
            : int(resolved["panel_varied_rows"])
        ],
        dtype=np.int64,
    )
    panel = load_hdf5_rows(
        dataset, row_ids, gradient_set="varied", include_targets=True
    )
    if panel.actual_log_heat_flux is None:
        raise RuntimeError("true clipped-log targets were not loaded")
    scalars, scalar_names, groups = _metadata(dataset, row_ids)
    if len(np.unique(groups)) != len(row_ids):
        raise RuntimeError(
            "registered panel must contain one tube per equilibrium_files"
        )
    feature_table = invariant_feature_table(
        panel.geometry.numpy(),
        scalars,
        scalar_names,
        panel.a_over_lt.numpy(),
        panel.a_over_ln.numpy(),
        channel_scales=_channel_scales(repository / resolved["channel_scales"]),
    )
    if feature_table.version != resolved["feature_table_version"]:
        raise RuntimeError("feature-table implementation and config versions differ")
    feature_names = feature_table.names
    interaction_names = [tuple(pair) for pair in resolved["registered_interactions"]]
    interaction_indices = tuple(
        (feature_names.index(left), feature_names.index(right))
        for left, right in interaction_names
    )
    make_estimator = _factory(resolved, feature_names)

    ensemble = load_ensemble(checkpoint, device=str(resolved["device"]))
    member_ids = list(
        registry["member_cohorts"]["stored_validation_top_10"][
            : int(resolved["members"])
        ]
    )
    member_index = {member: index for index, member in enumerate(ensemble.member_ids)}
    member_predictions: dict[str, np.ndarray] = {}
    bottleneck_targets: list[tuple[str, str, np.ndarray]] = []
    for member_id in member_ids:
        model = InvariantMember(ensemble.models[member_index[member_id]])
        bottleneck, prediction = _member_values(
            model,
            panel,
            batch_size=int(resolved["batch_size"]),
            device=ensemble.device,
        )
        member_predictions[member_id] = prediction
        for unit in range(bottleneck.shape[1]):
            bottleneck_targets.append(
                ("bottleneck_unit", f"{member_id}:u{unit:03d}", bottleneck[:, unit])
            )

    true_target = panel.actual_log_heat_flux.numpy().astype(np.float64)
    stable = true_target <= float(resolved["stable_threshold_log_Q"])
    primary_targets = [
        ("member_output", member_id, member_predictions[member_id])
        for member_id in member_ids
    ]
    primary_targets.extend(
        [
            (
                "ensemble_mean_output",
                "stored_validation_top_member_mean",
                np.mean(np.stack(list(member_predictions.values())), axis=0),
            ),
            ("true_clipped_log_Q", "GX_true_target", true_target),
        ]
    )

    artifacts = RunArtifacts(output_dir)
    artifacts.write_text(
        "feature_registry.csv", _csv_text(list(feature_table.definitions))
    )
    fidelity_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    recurrence_rows: list[dict[str, Any]] = []
    primary_results: dict[tuple[str, str], DistillationResult] = {}
    primary_target_seeds: dict[tuple[str, str], int] = {}

    targets = [*bottleneck_targets, *primary_targets]
    for target_number, (target_kind, target_id, target) in enumerate(targets):
        target_seed = int(resolved["seed"]) + target_number * 100
        result = grouped_ebm_crossfit(
            feature_table.values,
            target,
            groups,
            feature_names=feature_names,
            folds=int(resolved["outer_folds"]),
            seed=target_seed,
            interactions=interaction_indices,
            estimator_factory=make_estimator,
        )
        fidelity_rows.extend(
            _fidelity_rows(
                target_kind,
                target_id,
                target,
                result,
                stable,
                groups=groups,
                bootstrap_replicates=(
                    int(resolved["fidelity_bootstrap_replicates"])
                    if target_kind != "bottleneck_unit"
                    else 0
                ),
                bootstrap_seed=int(resolved["seed"]) + 70000 + target_number,
            )
        )
        importance_rows.extend(
            _importance_rows(
                target_kind, target_id, result, int(resolved["top_k_features"])
            )
        )
        if target_kind != "bottleneck_unit":
            primary_results[(target_kind, target_id)] = result
            primary_target_seeds[(target_kind, target_id)] = target_seed
            for position, row_id in enumerate(row_ids):
                residual_rows.append(
                    {
                        "target_kind": target_kind,
                        "target_id": target_id,
                        "row_id": int(row_id),
                        "equilibrium_file": groups[position],
                        "stable_or_near_floor": bool(stable[position]),
                        "target_native_value": float(target[position]),
                        "ebm_oof_prediction": float(result.prediction[position]),
                        "signed_residual_target_minus_formula": float(
                            target[position] - result.prediction[position]
                        ),
                        "fold": int(result.fold[position]),
                        "split_unit": "equilibrium_files",
                    }
                )
            recurrence = grouped_term_recurrence(
                feature_table.values,
                target,
                groups,
                feature_names=feature_names,
                replicates=int(resolved["bootstrap_replicates"]),
                seed=int(resolved["seed"]) + 50000 + target_number * 100,
                top_k=int(resolved["top_k_features"]),
                estimator_factory=make_estimator,
            )
            recurrence_rows.extend(
                {"target_kind": target_kind, "target_id": target_id, **row}
                for row in recurrence
            )
            final_estimator = make_estimator(
                seed=int(resolved["seed"]) + 90000 + target_number,
                interactions=interaction_indices,
            )
            final_estimator.fit(feature_table.values, target)
            effect_rows.extend(
                _global_effect_rows(target_kind, target_id, final_estimator)
            )
        print(
            f"{target_number + 1}/{len(targets)} {target_kind} {target_id} "
            f"R2={result.held_out_r2:.4f}",
            flush=True,
        )

    subset_rows: list[dict[str, Any]] = []
    subset_summary: dict[str, dict[str, float]] = {}
    fidelity_replicates = int(resolved["fidelity_bootstrap_replicates"])
    for target_number, (target_kind, target_id, target) in enumerate(primary_targets):
        key = (target_kind, target_id)
        predictions: dict[str, np.ndarray] = {}
        subset_metadata: dict[str, tuple[int, int]] = {}
        for subset_number, spec in enumerate(resolved["nested_feature_sets"]):
            positions, names, subset_interactions = _nested_subset_spec(
                spec, feature_names, interaction_names
            )
            if spec["name"] == "all_17_registered_interactions":
                result = primary_results[key]
            else:
                result = grouped_ebm_crossfit(
                    feature_table.values[:, positions],
                    target,
                    groups,
                    feature_names=names,
                    folds=int(resolved["outer_folds"]),
                    seed=primary_target_seeds[key],
                    interactions=subset_interactions,
                    estimator_factory=_factory(resolved, names),
                )
            predictions[spec["name"]] = result.prediction
            subset_metadata[spec["name"]] = (len(names), len(subset_interactions))
            print(
                f"subset {target_number + 1}/{len(primary_targets)} "
                f"{target_id} {spec['name']} R2={result.held_out_r2:.4f}",
                flush=True,
            )
        baseline = predictions["baseline_trio"]
        subset_summary[target_id] = {}
        for subset_number, spec in enumerate(resolved["nested_feature_sets"]):
            name = spec["name"]
            point, lower, upper = grouped_r2_interval(
                target,
                predictions[name],
                groups,
                replicates=fidelity_replicates,
                seed=int(resolved["seed"])
                + 80000
                + target_number * 100
                + subset_number,
            )
            gain, gain_lower, gain_upper = grouped_r2_difference_interval(
                target,
                predictions[name],
                baseline,
                groups,
                replicates=fidelity_replicates,
                seed=int(resolved["seed"])
                + 85000
                + target_number * 100
                + subset_number,
            )
            feature_count, interaction_count = subset_metadata[name]
            subset_rows.append(
                {
                    "target_kind": target_kind,
                    "target_id": target_id,
                    "concept_set": name,
                    "feature_count": feature_count,
                    "registered_interaction_count": interaction_count,
                    "held_out_r2": point,
                    "held_out_r2_ci95_lower": lower,
                    "held_out_r2_ci95_upper": upper,
                    "gain_over_baseline_trio": gain,
                    "gain_ci95_lower": gain_lower,
                    "gain_ci95_upper": gain_upper,
                    "bootstrap_replicates": fidelity_replicates,
                    "bootstrap_unit": "equilibrium_files",
                    "split_unit": "equilibrium_files",
                    "model": "ExplainableBoostingRegressor",
                    "model_version": "interpret-core==0.7.8",
                    "estimand": NATIVE_ESTIMAND,
                    "validity_tag": OBSERVED,
                }
            )
            subset_summary[target_id][name] = point

    artifacts.write_text("fidelity.csv", _csv_text(fidelity_rows))
    artifacts.write_text("subset_fidelity.csv", _csv_text(subset_rows))
    artifacts.write_text("term_importance.csv", _csv_text(importance_rows))
    artifacts.write_text("term_recurrence.csv", _csv_text(recurrence_rows))
    artifacts.write_text("primary_residuals.csv", _csv_text(residual_rows))
    artifacts.write_text("ebm_effects.csv", _csv_text(effect_rows))
    figure = output_dir / "ebm_effects.png"
    _plot_effects(
        figure, effect_rows, [target_id for _, target_id, _ in primary_targets]
    )
    artifacts.register_existing(figure.name)

    all_fidelity = [row for row in fidelity_rows if row["regime"] == "all"]
    member_fidelity = [
        row for row in all_fidelity if row["target_kind"] == "member_output"
    ]
    unit_fidelity = [
        row for row in all_fidelity if row["target_kind"] == "bottleneck_unit"
    ]
    true_fidelity = next(
        row for row in all_fidelity if row["target_kind"] == "true_clipped_log_Q"
    )
    summary = {
        "step": "S12",
        "run_id": resolved["run_id"],
        "mode": resolved["mode"],
        "estimand": NATIVE_ESTIMAND,
        "canonical_function": CANONICAL_FUNCTION,
        "feature_table_version": feature_table.version,
        "feature_count": len(feature_names),
        "registered_interaction_count": len(interaction_indices),
        "members": member_ids,
        "cohort": {
            "rows": len(row_ids),
            "unique_equilibrium_files": len(np.unique(groups)),
            "stable_or_near_floor": int(stable.sum()),
            "unstable": int((~stable).sum()),
        },
        "member_output_r2": {
            row["target_id"]: row["held_out_r2"] for row in member_fidelity
        },
        "median_member_output_r2": float(
            np.median([float(row["held_out_r2"]) for row in member_fidelity])
        ),
        "bottleneck_unit_count": len(unit_fidelity),
        "median_bottleneck_unit_r2": float(
            np.nanmedian([float(row["held_out_r2"]) for row in unit_fidelity])
        ),
        "bottleneck_units_r2_at_least_0_8": int(
            np.sum([float(row["held_out_r2"]) >= 0.8 for row in unit_fidelity])
        ),
        "true_clipped_log_Q_r2": float(true_fidelity["held_out_r2"]),
        "nested_subset_r2": subset_summary,
        "bootstrap": {
            "unit": "equilibrium_files",
            "replicates": int(resolved["bootstrap_replicates"]),
            "fidelity_replicates": fidelity_replicates,
        },
        "model": "ExplainableBoostingRegressor",
        "model_version": "interpret-core==0.7.8",
        "symbolic_regression": resolved["symbolic_regression"],
    }
    artifacts.write_json("summary.json", summary)
    resolved["script_sha256"] = sha256_file(__file__)
    resolved["distillation_module_sha256"] = sha256_file(
        repository / "itg_nn/xai/distillation.py"
    )
    resolved["symmetry_module_sha256"] = sha256_file(
        repository / "itg_nn/xai/symmetry.py"
    )
    resolved["bottleneck_module_sha256"] = sha256_file(
        repository / "itg_nn/xai/bottleneck.py"
    )
    resolved["feature_registry_sha256"] = sha256_file(
        output_dir / "feature_registry.csv"
    )
    artifacts.finalize(
        config=resolved,
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=member_ids,
        row_ids=row_ids,
        gradient_set="varied frozen S01 interpretation panel",
        device=ensemble.device,
        repository=repository,
        command=sys.argv,
        published_dir=published,
        extra_manifest={
            "feature_table_version": feature_table.version,
            "split_unit": "equilibrium_files",
            "target_axes": [
                "bottleneck_unit",
                "member_output",
                "ensemble_mean",
                "true_clipped_log_Q",
            ],
        },
    )
    if published is not None:
        for name in (
            "feature_registry.csv",
            "fidelity.csv",
            "subset_fidelity.csv",
            "term_importance.csv",
            "term_recurrence.csv",
            "primary_residuals.csv",
            "ebm_effects.csv",
            "ebm_effects.png",
            "summary.json",
        ):
            (published / name).write_bytes((output_dir / name).read_bytes())
    return output_dir


def main() -> int:
    print(run(build_parser().parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
