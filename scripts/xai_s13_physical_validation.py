#!/usr/bin/env python3
"""Run S13 natural experiments and prospective GX test design."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from itg_nn.data import clipped_log_heat_flux, load_hdf5_rows
from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.distillation import grouped_ebm_crossfit, invariant_feature_table
from itg_nn.xai.physical_validation import (
    cross_fitted_aipw,
    equilibrium_grouped_matches,
    grouped_bootstrap_interval,
    residual_rank_association,
)
from itg_nn.xai.runtime import set_deterministic_seed


NATIVE_ESTIMAND = "native max(log Q, -2)"
OBSERVED = "observed-comparison"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xai/S13_physical_validation.json")
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
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
        resolved["panel_rows"] = args.rows
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
        raise RuntimeError("S01 must provide seven positive robust channel scales")
    return scales


def _stratified_positions(
    classes: np.ndarray, target: np.ndarray, count: int, seed: int
) -> np.ndarray:
    if count >= len(target):
        return np.arange(len(target), dtype=np.int64)
    stable = target <= -1.9
    rng = np.random.default_rng(seed)
    strata = [(value, flag) for value in np.unique(classes) for flag in (False, True)]
    selected: list[int] = []
    for value, flag in strata:
        positions = np.flatnonzero((classes == value) & (stable == flag))
        share = max(1, round(count * len(positions) / len(target)))
        selected.extend(rng.choice(positions, min(share, len(positions)), replace=False))
    if len(selected) < count:
        remaining = np.setdiff1d(np.arange(len(target)), np.asarray(selected))
        selected.extend(rng.choice(remaining, count - len(selected), replace=False))
    return np.asarray(sorted(selected[:count]), dtype=np.int64)


def _load_metadata(
    dataset: Path, rows: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    with h5py.File(dataset, "r") as handle:
        names = tuple(_decode(handle["scalar_features"][:]))
        scalars = _h5_take(handle["scalar_feature_matrix"], rows).astype(np.float64)
        groups = _decode(_h5_take(handle["equilibrium_files"], rows))
        classes = _h5_take(handle["equilibrium_class"], rows).astype(np.int16)
    return scalars, names, groups, classes


def _physical_outcomes(dataset: Path, rows: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(dataset, "r") as handle:
        for gradient_set in ("fixed", "varied"):
            group = handle[f"{gradient_set}_gradient_simulations"]
            q = _h5_take(group["Q_avgs"], rows).astype(np.float64)
            q_std = _h5_take(group["Q_stds"], rows).astype(np.float64)
            zonal = _h5_take(group["zonal_phi2_amplitudes"], rows).astype(np.float64)
            q_z = _h5_take(group["Q_avgs_vs_z"], rows).astype(np.float64)
            tiny = np.finfo(np.float64).tiny
            result[gradient_set] = {
                "target_native": clipped_log_heat_flux(q).astype(np.float64),
                "log_Q": np.log(q),
                "log10_Q_stds": np.log10(np.maximum(q_std, tiny)),
                "log10_zonal_phi2": np.log10(np.maximum(zonal, tiny)),
                "Qz_localization": np.max(np.abs(q_z), axis=1)
                / np.maximum(np.mean(np.abs(q_z), axis=1), tiny),
                "Qz_positive_fraction": np.mean(q_z > 0, axis=1),
                "a_over_LT": _h5_take(group["a_over_LT"], rows).astype(np.float64),
                "a_over_Ln": _h5_take(group["a_over_Ln"], rows).astype(np.float64),
            }
    return result


def _nuisance_matrix(
    requested: list[str],
    candidate: str,
    feature_names: tuple[str, ...],
    feature_values: np.ndarray,
    scalar_names: tuple[str, ...],
    scalars: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for name in requested:
        if name == candidate:
            continue
        if name in feature_names:
            column = feature_values[:, feature_names.index(name)]
        elif name in scalar_names:
            column = scalars[:, scalar_names.index(name)]
        else:
            raise ValueError(f"unknown nuisance feature {name}")
        if np.std(column) > np.finfo(float).eps:
            columns.append(column)
            names.append(name)
    for value in np.unique(classes)[1:]:
        columns.append((classes == value).astype(np.float64))
        names.append(f"equilibrium_class_{value}")
    return np.column_stack(columns), tuple(names)


def _regimes(target: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    stable = target <= -1.9
    return (
        ("all", np.ones(len(target), dtype=bool)),
        ("stable_or_near_floor", stable),
        ("unstable", ~stable),
    )


def _association_rows(
    candidates: list[str],
    feature_names: tuple[str, ...],
    feature_values: np.ndarray,
    outcomes: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = outcomes["target_native"]
    for candidate_index, candidate in enumerate(candidates):
        values = feature_values[:, feature_names.index(candidate)]
        for outcome_index, (outcome_name, outcome) in enumerate(outcomes.items()):
            if outcome_name in {"a_over_LT", "a_over_Ln"}:
                continue
            for regime_index, (regime, mask) in enumerate(_regimes(target)):
                if mask.sum() < 8:
                    continue
                estimate, lower, upper = residual_rank_association(
                    values[mask],
                    outcome[mask],
                    groups[mask],
                    replicates=replicates,
                    seed=seed + 100 * candidate_index + 10 * outcome_index + regime_index,
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "outcome": outcome_name,
                        "gradient_set": "fixed",
                        "regime": regime,
                        "rows": int(mask.sum()),
                        "spearman_rho": estimate,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "bootstrap_replicates": replicates,
                        "bootstrap_unit": "equilibrium_files",
                        "validity_tag": OBSERVED,
                        "claim_grade": "observational-physical",
                    }
                )
    return rows


def _matched_rows(
    candidates: list[str],
    feature_names: tuple[str, ...],
    feature_values: np.ndarray,
    scalars: np.ndarray,
    scalar_names: tuple[str, ...],
    groups: np.ndarray,
    classes: np.ndarray,
    outcomes: dict[str, np.ndarray],
    row_ids: np.ndarray,
    resolved: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    target = outcomes["target_native"]
    replicates = int(resolved["bootstrap_replicates"])
    for candidate_index, candidate in enumerate(candidates):
        exposure = feature_values[:, feature_names.index(candidate)]
        nuisance, nuisance_names = _nuisance_matrix(
            list(resolved["matching_nuisance_features"]),
            candidate,
            feature_names,
            feature_values,
            scalar_names,
            scalars,
            classes,
        )
        match = equilibrium_grouped_matches(
            exposure,
            nuisance,
            groups,
            high_quantile=float(resolved["match_high_quantile"]),
            low_quantile=float(resolved["match_low_quantile"]),
            caliper=float(resolved["match_caliper_iqr_units"]),
        )
        high, low = match.high_positions, match.low_positions
        for pair_index, (high_position, low_position) in enumerate(zip(high, low)):
            row: dict[str, Any] = {
                "candidate": candidate,
                "pair_id": f"{candidate}:{pair_index:04d}",
                "high_row_id": int(row_ids[high_position]),
                "low_row_id": int(row_ids[low_position]),
                "high_equilibrium_file": groups[high_position],
                "low_equilibrium_file": groups[low_position],
                "nuisance_distance_iqr_units": float(match.distance[pair_index]),
                "candidate_high": float(exposure[high_position]),
                "candidate_low": float(exposure[low_position]),
                "candidate_contrast": float(match.exposure_contrast[pair_index]),
                "high_stable_or_near_floor": bool(target[high_position] <= -1.9),
                "low_stable_or_near_floor": bool(target[low_position] <= -1.9),
                "max_abs_nuisance_smd_before": float(
                    np.max(np.abs(match.nuisance_smd_before))
                ),
                "max_abs_nuisance_smd_after": float(
                    np.max(np.abs(match.nuisance_smd_after))
                ),
                "nuisance_features": "|".join(nuisance_names),
                "validity_tag": OBSERVED,
            }
            for outcome_name, outcome in outcomes.items():
                if outcome_name not in {"a_over_LT", "a_over_Ln"}:
                    row[f"{outcome_name}_high_minus_low"] = float(
                        outcome[high_position] - outcome[low_position]
                    )
            pair_rows.append(row)

        pair_stable = (target[high] <= -1.9) | (target[low] <= -1.9)
        pair_regimes = (
            ("all", np.ones(len(high), dtype=bool)),
            ("either_stable_or_near_floor", pair_stable),
            ("both_unstable", ~pair_stable),
        )
        for outcome_index, (outcome_name, outcome) in enumerate(outcomes.items()):
            if outcome_name in {"a_over_LT", "a_over_Ln"}:
                continue
            differences = outcome[high] - outcome[low]
            for regime_index, (regime, mask) in enumerate(pair_regimes):
                if mask.sum() < 3:
                    continue
                point, lower, upper = grouped_bootstrap_interval(
                    differences[mask],
                    np.arange(len(differences))[mask],
                    replicates=replicates,
                    seed=int(resolved["seed"])
                    + 1000
                    + 100 * candidate_index
                    + 10 * outcome_index
                    + regime_index,
                )
                effect_rows.append(
                    {
                        "candidate": candidate,
                        "outcome": outcome_name,
                        "regime": regime,
                        "matched_pairs": int(mask.sum()),
                        "mean_high_minus_low": point,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "median_candidate_contrast": float(
                            np.median(match.exposure_contrast[mask])
                        ),
                        "median_nuisance_distance_iqr_units": float(
                            np.median(match.distance[mask])
                        ),
                        "max_abs_nuisance_smd_before": float(
                            np.max(np.abs(match.nuisance_smd_before))
                        ),
                        "max_abs_nuisance_smd_after": float(
                            np.max(np.abs(match.nuisance_smd_after))
                        ),
                        "bootstrap_replicates": replicates,
                        "bootstrap_unit": "disjoint matched pair of equilibrium_files",
                        "validity_tag": OBSERVED,
                        "causal_claim_permitted": False,
                    }
                )

        low_cut = np.quantile(exposure, float(resolved["match_low_quantile"]))
        high_cut = np.quantile(exposure, float(resolved["match_high_quantile"]))
        tails = (exposure <= low_cut) | (exposure >= high_cut)
        treated = exposure >= high_cut
        for outcome_index, outcome_name in enumerate(
            ("target_native", "log10_Q_stds", "log10_zonal_phi2", "Qz_localization")
        ):
            outcome = outcomes[outcome_name]
            for regime_index, (regime, regime_mask) in enumerate(
                (("all", np.ones(len(target), dtype=bool)), ("unstable", target > -1.9))
            ):
                mask = tails & regime_mask
                if mask.sum() < 30 or np.unique(treated[mask]).size < 2:
                    continue
                result = cross_fitted_aipw(
                    treated[mask],
                    outcome[mask],
                    nuisance[mask],
                    groups[mask],
                    folds=int(resolved["crossfit_folds"]),
                    seed=int(resolved["seed"])
                    + 2000
                    + 100 * candidate_index
                    + 10 * outcome_index
                    + regime_index,
                    propensity_clip=float(resolved["propensity_clip"]),
                )
                point, lower, upper = grouped_bootstrap_interval(
                    result.influence,
                    groups[mask],
                    replicates=replicates,
                    seed=int(resolved["seed"])
                    + 3000
                    + 100 * candidate_index
                    + 10 * outcome_index
                    + regime_index,
                )
                sensitivity_rows.append(
                    {
                        "candidate": candidate,
                        "outcome": outcome_name,
                        "regime": regime,
                        "tail_rows": int(mask.sum()),
                        "aipw_high_minus_low": point,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "overlap_fraction": result.overlap_fraction,
                        "propensity_clip": resolved["propensity_clip"],
                        "split_unit": result.split_unit,
                        "bootstrap_unit": "equilibrium_files",
                        "validity_tag": result.validity_tag,
                        "causal_claim_permitted": False,
                    }
                )
    return pair_rows, effect_rows, sensitivity_rows


def _ebm_factory(resolved: dict[str, Any], names: tuple[str, ...]):
    from interpret.glassbox import ExplainableBoostingRegressor

    def make(*, seed: int, interactions=()):
        return ExplainableBoostingRegressor(
            feature_names=list(names),
            interactions=list(interactions),
            max_bins=int(resolved["ebm_max_bins"]),
            outer_bags=int(resolved["ebm_outer_bags"]),
            inner_bags=0,
            max_rounds=int(resolved["ebm_max_rounds"]),
            learning_rate=0.03,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=1,
        )

    return make


def _r2(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    selected = target[mask]
    denominator = float(np.sum(np.square(selected - selected.mean())))
    return (
        float("nan")
        if denominator <= 0
        else 1.0
        - float(np.sum(np.square(selected - prediction[mask])) / denominator)
    )


def _residual_rows(
    candidates: list[str],
    feature_names: tuple[str, ...],
    feature_values_by_panel: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    groups: np.ndarray,
    resolved: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replicates = int(resolved["bootstrap_replicates"])
    for panel_index, gradient_set in enumerate(("fixed", "varied")):
        feature_values = feature_values_by_panel[gradient_set]
        target = targets[gradient_set]
        for baseline_index, (baseline_name, baseline_features) in enumerate(
            resolved["paper_baselines"].items()
        ):
            baseline_names = tuple(baseline_features)
            baseline_indices = [feature_names.index(name) for name in baseline_names]
            baseline = grouped_ebm_crossfit(
                feature_values[:, baseline_indices],
                target,
                groups,
                feature_names=baseline_names,
                folds=int(resolved["crossfit_folds"]),
                seed=int(resolved["seed"]) + 4000 + 100 * panel_index + baseline_index,
                interactions=(),
                estimator_factory=_ebm_factory(resolved, baseline_names),
            )
            for candidate_index, candidate in enumerate(candidates):
                if candidate in baseline_names:
                    continue
                augmented_names = (*baseline_names, candidate)
                augmented_indices = [feature_names.index(name) for name in augmented_names]
                augmented = grouped_ebm_crossfit(
                    feature_values[:, augmented_indices],
                    target,
                    groups,
                    feature_names=augmented_names,
                    folds=int(resolved["crossfit_folds"]),
                    seed=int(resolved["seed"])
                    + 4000
                    + 100 * panel_index
                    + baseline_index,
                    interactions=(),
                    estimator_factory=_ebm_factory(resolved, augmented_names),
                )
                candidate_values = feature_values[:, feature_names.index(candidate)]
                for regime_index, (regime, mask) in enumerate(_regimes(target)):
                    if mask.sum() < 8:
                        continue
                    baseline_residual = target - baseline.prediction
                    baseline_sq = np.square(baseline_residual[mask])
                    augmented_sq = np.square(augmented.prediction[mask] - target[mask])

                    def improvement(values: np.ndarray) -> float:
                        return float(np.mean(values[:, 0] - values[:, 1]))

                    delta_mse, lower, upper = grouped_bootstrap_interval(
                        np.column_stack((baseline_sq, augmented_sq)),
                        groups[mask],
                        replicates=replicates,
                        seed=int(resolved["seed"])
                        + 5000
                        + 1000 * panel_index
                        + 100 * baseline_index
                        + 10 * candidate_index
                        + regime_index,
                        statistic=improvement,
                    )
                    rho, rho_lower, rho_upper = residual_rank_association(
                        candidate_values[mask],
                        baseline_residual[mask],
                        groups[mask],
                        replicates=replicates,
                        seed=int(resolved["seed"])
                        + 6000
                        + 1000 * panel_index
                        + 100 * baseline_index
                        + 10 * candidate_index
                        + regime_index,
                    )
                    rows.append(
                        {
                            "gradient_set": gradient_set,
                            "baseline": baseline_name,
                            "candidate": candidate,
                            "regime": regime,
                            "rows": int(mask.sum()),
                            "baseline_r2": _r2(target, baseline.prediction, mask),
                            "augmented_r2": _r2(target, augmented.prediction, mask),
                            "delta_r2": _r2(target, augmented.prediction, mask)
                            - _r2(target, baseline.prediction, mask),
                            "baseline_mse": float(np.mean(baseline_sq)),
                            "augmented_mse": float(np.mean(augmented_sq)),
                            "mse_improvement": delta_mse,
                            "mse_improvement_ci95_lower": lower,
                            "mse_improvement_ci95_upper": upper,
                            "candidate_vs_baseline_residual_rho": rho,
                            "residual_rho_ci95_lower": rho_lower,
                            "residual_rho_ci95_upper": rho_upper,
                            "model": "ExplainableBoostingRegressor",
                            "split_unit": "equilibrium_files",
                            "bootstrap_unit": "equilibrium_files",
                            "estimand": NATIVE_ESTIMAND,
                            "validity_tag": OBSERVED,
                        }
                    )
    return rows


def _candidate_ranking(
    candidates: list[str],
    effects: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    residuals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        matched = next(
            row
            for row in effects
            if row["candidate"] == candidate
            and row["outcome"] == "target_native"
            and row["regime"] == "all"
        )
        aipw = next(
            row
            for row in sensitivity
            if row["candidate"] == candidate
            and row["outcome"] == "target_native"
            and row["regime"] == "all"
        )
        residual_candidates = [
            row
            for row in residuals
            if row["candidate"] == candidate
            and row["gradient_set"] == "fixed"
            and row["regime"] == "all"
        ]
        matched_resolved = matched["ci95_lower"] * matched["ci95_upper"] > 0
        aipw_resolved = aipw["ci95_lower"] * aipw["ci95_upper"] > 0
        same_sign = np.sign(matched["mean_high_minus_low"]) == np.sign(
            aipw["aipw_high_minus_low"]
        )
        balance_acceptable = matched["max_abs_nuisance_smd_after"] <= 0.5
        overlap_acceptable = aipw["overlap_fraction"] >= 0.8
        residual_by_baseline = {
            str(row["baseline"]): row for row in residual_candidates
        }
        f_q_residual = residual_by_baseline.get("f_Q_baseline")
        paper_residual = residual_by_baseline.get("paper_selected")
        residual_for_ranking = paper_residual or f_q_residual
        if residual_for_ranking is None:
            raise RuntimeError(f"no fixed-panel residual result for {candidate}")
        residual_gain = float(residual_for_ranking["mse_improvement"])
        residual_gain_resolved = (
            float(residual_for_ranking["mse_improvement_ci95_lower"]) > 0
        )
        evidence_score = (
            int(matched_resolved)
            + int(aipw_resolved)
            + int(same_sign)
            + int(residual_gain_resolved)
            + int(balance_acceptable)
            + int(overlap_acceptable)
        )
        if (
            matched_resolved
            and aipw_resolved
            and same_sign
            and residual_gain_resolved
            and balance_acceptable
            and overlap_acceptable
        ):
            grade = "intervention-ready"
        elif matched_resolved or aipw_resolved:
            grade = "observational-physical"
        else:
            grade = "model-mechanistic"
        rows.append(
            {
                "candidate": candidate,
                "rank_score": evidence_score,
                "claim_grade": grade,
                "matched_native_effect": matched["mean_high_minus_low"],
                "matched_native_ci95_lower": matched["ci95_lower"],
                "matched_native_ci95_upper": matched["ci95_upper"],
                "aipw_native_effect": aipw["aipw_high_minus_low"],
                "aipw_native_ci95_lower": aipw["ci95_lower"],
                "aipw_native_ci95_upper": aipw["ci95_upper"],
                "matched_aipw_same_sign": bool(same_sign),
                "max_abs_nuisance_smd_after": matched[
                    "max_abs_nuisance_smd_after"
                ],
                "balance_threshold": 0.5,
                "balance_acceptable": bool(balance_acceptable),
                "aipw_overlap_fraction": aipw["overlap_fraction"],
                "overlap_threshold": 0.8,
                "overlap_acceptable": bool(overlap_acceptable),
                "ranking_residual_baseline": residual_for_ranking["baseline"],
                "ranking_residual_mse_improvement": residual_gain,
                "ranking_residual_mse_improvement_ci95_lower": residual_for_ranking[
                    "mse_improvement_ci95_lower"
                ],
                "ranking_residual_mse_improvement_ci95_upper": residual_for_ranking[
                    "mse_improvement_ci95_upper"
                ],
                "ranking_residual_gain_resolved": bool(residual_gain_resolved),
                "f_Q_baseline_mse_improvement": (
                    "" if f_q_residual is None else f_q_residual["mse_improvement"]
                ),
                "paper_selected_mse_improvement": (
                    "" if paper_residual is None else paper_residual["mse_improvement"]
                ),
                "remaining_confounding_visible": True,
                "causal_claim_permitted": False,
            }
        )
    return sorted(rows, key=lambda row: (-int(row["rank_score"]), row["candidate"]))


def _contradictory_cases(
    pair_rows: list[dict[str, Any]], ranking: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for rank_row in ranking:
        candidate = rank_row["candidate"]
        expected = np.sign(float(rank_row["matched_native_effect"])) or 1.0
        selected = [row for row in pair_rows if row["candidate"] == candidate]
        selected.sort(
            key=lambda row: expected * float(row["target_native_high_minus_low"]),
            reverse=True,
        )
        supporting = selected[:count]
        contradicting = sorted(
            selected,
            key=lambda row: expected * float(row["target_native_high_minus_low"]),
        )[:count]
        for case_type, case_rows in (
            ("supporting", supporting),
            ("contradicting", contradicting),
        ):
            for order, row in enumerate(case_rows, 1):
                result.append(
                    {
                        "candidate": candidate,
                        "case_type": case_type,
                        "order": order,
                        "expected_population_sign": int(expected),
                        **row,
                    }
                )
    return result


def _gx_spec(ranking: list[dict[str, Any]], design: dict[str, Any]) -> dict[str, Any]:
    selected = ranking[:2]
    standard_runs = (
        int(design["anchor_equilibria"])
        * int(design["candidate_directions"])
        * int(design["signed_steps_per_direction"])
        * len(design["drive_points"])
    )
    base = (
        standard_runs * float(design["standard_run_node_hours"])
        + int(design["convergence_cases"])
        * float(design["convergence_run_node_hours"])
        + float(design["vmec_search_node_hours"])
    )
    total = base * (1.0 + float(design["contingency_fraction"]))
    interventions = []
    for candidate in selected:
        interventions.append(
            {
                "candidate": candidate["candidate"],
                "expected_native_log_Q_sign": int(
                    np.sign(float(candidate["matched_native_effect"])) or 1
                ),
                "construction": (
                    "For each anchor, continue VMEC boundary coefficients in both signs "
                    "while recomputing force-balanced equilibria; optimize the named invariant "
                    "direction and constrain the competing candidate, log_f_Q, aspect, iota, "
                    "shat, beta proxy, and nfp to their anchor tolerances. Never edit GX geometry "
                    "channels independently."
                ),
                "validity_tag": "prospective equilibrium-consistent intervention",
            }
        )
    return {
        "status": "proposal_only_researcher_approval_required",
        "decision_gate": (
            "Do not generate equilibria or launch GX until the researcher approves this "
            "intervention and budget."
        ),
        "interventions": interventions,
        "anchors": {
            "count": int(design["anchor_equilibria"]),
            "selection": (
                "one typical unstable matched pair from each of three represented equilibrium "
                "classes, excluding S11 common-mode failures and extreme Q_stds"
            ),
        },
        "drives": design["drive_points"],
        "controls": [
            "VMEC continuation returning to the anchor boundary (zero-step null)",
            "candidate-orthogonal continuation with the primary invariant constrained",
            "paired plus/minus steps of equal boundary-coefficient norm",
            "original anchor GX rerun to measure numerical repeatability",
        ],
        "gx_resolution_and_convergence": [
            "use the dataset's registered periodic flux-tube setup for the standard run",
            "repeat all anchor controls and the largest response in each direction at doubled parallel and perpendicular resolution",
            "double nonlinear averaging time and require the response to exceed two combined Q_stds standard errors",
            "require sign agreement and <=20% effect-size drift between standard and convergence runs",
        ],
        "compute_estimate": {
            "standard_runs": standard_runs,
            "standard_node_hours": standard_runs
            * float(design["standard_run_node_hours"]),
            "convergence_runs": int(design["convergence_cases"]),
            "convergence_node_hours": int(design["convergence_cases"])
            * float(design["convergence_run_node_hours"]),
            "vmec_search_node_hours": float(design["vmec_search_node_hours"]),
            "contingency_fraction": float(design["contingency_fraction"]),
            "total_perlmutter_node_hours": total,
            "basis": (
                "planning envelope, not a measured Perlmutter pilot: 0.5 node-hour per "
                "standard GX run and 2.0 per doubled-resolution/longer-average run; benchmark "
                "one standard and one convergence case before allocation"
            ),
        },
    }


def _plot(path: Path, ranking: list[dict[str, Any]], residuals: list[dict[str, Any]]) -> None:
    names = [str(row["candidate"]) for row in ranking]
    matched = np.asarray([float(row["matched_native_effect"]) for row in ranking])
    matched_low = np.asarray([float(row["matched_native_ci95_lower"]) for row in ranking])
    matched_high = np.asarray([float(row["matched_native_ci95_upper"]) for row in ranking])
    aipw = np.asarray([float(row["aipw_native_effect"]) for row in ranking])
    aipw_low = np.asarray([float(row["aipw_native_ci95_lower"]) for row in ranking])
    aipw_high = np.asarray([float(row["aipw_native_ci95_upper"]) for row in ranking])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    y = np.arange(len(names))
    axes[0].errorbar(
        matched,
        y - 0.08,
        xerr=np.vstack((matched - matched_low, matched_high - matched)),
        fmt="o",
        label="matched pairs",
    )
    axes[0].errorbar(
        aipw,
        y + 0.08,
        xerr=np.vstack((aipw - aipw_low, aipw_high - aipw)),
        fmt="s",
        label="adjusted tail contrast",
    )
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_yticks(y, names)
    axes[0].set_xlabel("high-minus-low GX target (native units)")
    axes[0].set_title("Observed fixed-drive contrasts")
    axes[0].legend()
    selected = [
        row
        for row in residuals
        if row["gradient_set"] == "fixed"
        and row["baseline"] == "paper_selected"
        and row["regime"] == "unstable"
    ]
    axes[1].barh(
        [str(row["candidate"]) for row in selected],
        [float(row["mse_improvement"]) for row in selected],
    )
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("out-of-fold MSE improvement (native units squared)")
    axes[1].set_title("Added value beyond paper-selected features")
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
        else (repository / "output/xai/S13" / resolved["run_id"]).resolve()
    )
    published = (
        None
        if args.no_publish or resolved["mode"] == "pilot"
        else (repository / resolved["published_dir"]).resolve()
    )
    if args.resume and (output_dir / "manifest.json").is_file():
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["dataset"]["sha256"] != sha256_file(dataset):
            raise RuntimeError("resume dataset fingerprint changed")
        return output_dir

    set_deterministic_seed(int(resolved["seed"]))
    registry = json.loads((repository / resolved["cohorts"]).read_text(encoding="utf-8"))
    registered_rows = np.asarray(
        registry["interpretation_panel"]["varied_row_ids"], dtype=np.int64
    )
    with h5py.File(dataset, "r") as handle:
        classes_all = _h5_take(handle["equilibrium_class"], registered_rows).astype(np.int16)
        fixed_q = _h5_take(
            handle["fixed_gradient_simulations/Q_avgs"], registered_rows
        ).astype(np.float64)
    fixed_target_all = clipped_log_heat_flux(fixed_q).astype(np.float64)
    positions = _stratified_positions(
        classes_all,
        fixed_target_all,
        int(resolved["panel_rows"]),
        int(resolved["seed"]),
    )
    row_ids = registered_rows[positions]
    scalars, scalar_names, groups, classes = _load_metadata(dataset, row_ids)
    if len(np.unique(groups)) != len(groups):
        raise RuntimeError("S13 panel must retain one row per equilibrium_files")
    outcomes = _physical_outcomes(dataset, row_ids)
    panels = {
        gradient_set: load_hdf5_rows(
            dataset, row_ids, gradient_set=gradient_set, include_targets=True
        )
        for gradient_set in ("fixed", "varied")
    }
    if not (
        np.allclose(outcomes["fixed"]["a_over_LT"], 3.0)
        and np.allclose(outcomes["fixed"]["a_over_Ln"], 0.9)
    ):
        raise RuntimeError("fixed-gradient panel does not use physical drives (3, 0.9)")
    scales = _channel_scales(repository / resolved["channel_scales"])
    feature_tables = {
        gradient_set: invariant_feature_table(
            panel.geometry.numpy(),
            scalars,
            scalar_names,
            panel.a_over_lt.numpy(),
            panel.a_over_ln.numpy(),
            channel_scales=scales,
        )
        for gradient_set, panel in panels.items()
    }
    feature_names = feature_tables["fixed"].names
    if feature_tables["fixed"].version != resolved["feature_table_version"]:
        raise RuntimeError("S12 feature table version changed")
    candidates = list(resolved["candidate_features"])
    if set(candidates) - set(feature_names):
        raise ValueError("candidate registry contains unknown features")

    artifacts = RunArtifacts(output_dir)
    association_rows = _association_rows(
        candidates,
        feature_names,
        feature_tables["fixed"].values,
        outcomes["fixed"],
        groups,
        replicates=int(resolved["bootstrap_replicates"]),
        seed=int(resolved["seed"]),
    )
    pair_rows, effect_rows, sensitivity_rows = _matched_rows(
        candidates,
        feature_names,
        feature_tables["fixed"].values,
        scalars,
        scalar_names,
        groups,
        classes,
        outcomes["fixed"],
        row_ids,
        resolved,
    )
    residual_rows = _residual_rows(
        candidates,
        feature_names,
        {key: table.values for key, table in feature_tables.items()},
        {key: value["target_native"] for key, value in outcomes.items()},
        groups,
        resolved,
    )
    ranking = _candidate_ranking(candidates, effect_rows, sensitivity_rows, residual_rows)
    contradictions = _contradictory_cases(
        pair_rows, ranking, int(resolved["contradictory_cases_per_candidate"])
    )
    gx_spec = _gx_spec(ranking, resolved["gx_design"])

    for name, rows in (
        ("fixed_associations.csv", association_rows),
        ("matched_pairs.csv", pair_rows),
        ("matched_effects.csv", effect_rows),
        ("doubly_robust_sensitivity.csv", sensitivity_rows),
        ("residual_validation.csv", residual_rows),
        ("candidate_ranking.csv", ranking),
        ("contradictory_cases.csv", contradictions),
    ):
        artifacts.write_text(name, _csv_text(rows))
    artifacts.write_json("gx_experiment_spec.json", gx_spec)
    plot_path = output_dir / "natural_experiment_atlas.png"
    _plot(plot_path, ranking, residual_rows)
    artifacts.register_existing(plot_path.name)
    summary = {
        "run_id": resolved["run_id"],
        "estimand": NATIVE_ESTIMAND,
        "rows": len(row_ids),
        "fixed_stable_or_near_floor_rows": int(
            np.sum(outcomes["fixed"]["target_native"] <= -1.9)
        ),
        "varied_stable_or_near_floor_rows": int(
            np.sum(outcomes["varied"]["target_native"] <= -1.9)
        ),
        "candidate_ranking": ranking,
        "gx_compute_estimate": gx_spec["compute_estimate"],
        "decision_gate": gx_spec["decision_gate"],
        "claim_taxonomy": [
            "model-mechanistic",
            "observational-physical",
            "intervention-ready",
        ],
        "causal_claims_made": False,
        "invalid_perturbations_used": False,
        "matching": {
            "tail_quantiles": [
                resolved["match_low_quantile"],
                resolved["match_high_quantile"],
            ],
            "caliper_iqr_units": resolved["match_caliper_iqr_units"],
            "remaining_imbalance_published": True,
        },
    }
    artifacts.write_json("summary.json", summary)
    resolved["script_sha256"] = sha256_file(__file__)
    resolved["physical_validation_module_sha256"] = sha256_file(
        repository / "itg_nn/xai/physical_validation.py"
    )
    resolved["distillation_module_sha256"] = sha256_file(
        repository / "itg_nn/xai/distillation.py"
    )
    member_ids = list(
        registry["member_cohorts"]["stored_validation_top_10"][: int(resolved["members"])]
    )
    artifacts.finalize(
        config=resolved,
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=member_ids,
        row_ids=row_ids,
        gradient_set="fixed and varied S01 interpretation panel",
        device=resolved["device"],
        repository=repository,
        command=sys.argv,
        published_dir=published,
        extra_manifest={
            "feature_table_version": feature_tables["fixed"].version,
            "split_unit": "equilibrium_files",
            "model_outputs_computed": False,
            "physical_quantities": [
                "Q_avgs",
                "Q_avgs_vs_z",
                "Q_stds",
                "zonal_phi2_amplitudes",
            ],
            "perturbation_validity": "observed comparisons only; GX interventions are proposal-only",
        },
    )
    if published is not None:
        for name in (
            "fixed_associations.csv",
            "matched_pairs.csv",
            "matched_effects.csv",
            "doubly_robust_sensitivity.csv",
            "residual_validation.csv",
            "candidate_ranking.csv",
            "contradictory_cases.csv",
            "gx_experiment_spec.json",
            "natural_experiment_atlas.png",
            "summary.json",
        ):
            (published / name).write_bytes((output_dir / name).read_bytes())
    return output_dir


def main() -> int:
    print(run(build_parser().parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
