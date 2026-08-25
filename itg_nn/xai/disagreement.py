"""S11 ensemble-disagreement and held-out failure diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from itg_nn.xai.audit import spearman_correlation
from itg_nn.xai.perturbations import ValidityTag


@dataclass(frozen=True)
class SpreadGradient:
    spread: torch.Tensor
    gradient: torch.Tensor


@dataclass(frozen=True)
class CrossfitResult:
    predictions: np.ndarray
    fold_ids: np.ndarray


def ensemble_spread(predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("predictions must have shape (member, sample) with at least two members")
    if not np.isfinite(values).all():
        raise ValueError("predictions must be finite")
    return np.std(values, axis=0, ddof=0)


def spread_input_gradient(
    member_outputs: torch.Tensor, inputs: torch.Tensor
) -> SpreadGradient:
    if member_outputs.ndim != 2 or member_outputs.shape[0] < 2:
        raise ValueError("member_outputs must have shape (member, sample)")
    if not inputs.requires_grad:
        raise ValueError("inputs must require gradients")
    if member_outputs.shape[1] != inputs.shape[0]:
        raise ValueError("member_outputs and inputs have incompatible sample axes")
    spread = torch.std(member_outputs, dim=0, correction=0)
    gradient = torch.autograd.grad(spread.sum(), inputs)[0]
    return SpreadGradient(spread=spread, gradient=gradient)


def member_residuals(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or truth.ndim != 1 or values.shape[1] != len(truth):
        raise ValueError("predictions and targets must have shapes (member, sample) and (sample,)")
    if not np.isfinite(values).all() or not np.isfinite(truth).all():
        raise ValueError("predictions and targets must be finite")
    return values - truth[None, :]


def robust_scaled_channel_gradient(
    gradients: np.ndarray, channel_scales: np.ndarray
) -> np.ndarray:
    values = np.asarray(gradients, dtype=np.float64)
    scales = np.asarray(channel_scales, dtype=np.float64)
    if values.ndim < 2 or scales.ndim != 1 or values.shape[-1] != len(scales):
        raise ValueError("the last gradient axis must match the one-dimensional channel scales")
    if not np.isfinite(values).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("gradients must be finite and channel scales finite and positive")
    return values * scales.reshape((1,) * (values.ndim - 1) + (-1,))


def failure_categories(
    spread: np.ndarray,
    ensemble_absolute_error: np.ndarray,
    *,
    high_spread_threshold: float,
    high_error_threshold: float,
) -> np.ndarray:
    values = np.asarray(spread, dtype=np.float64)
    errors = np.asarray(ensemble_absolute_error, dtype=np.float64)
    if values.shape != errors.shape or values.ndim != 1:
        raise ValueError("spread and error must be matching one-dimensional arrays")
    if high_spread_threshold <= 0 or high_error_threshold <= 0:
        raise ValueError("failure thresholds must be positive native-unit values")
    high_spread = values >= high_spread_threshold
    high_error = errors >= high_error_threshold
    labels = np.empty(len(values), dtype="U24")
    labels[high_spread & ~high_error] = "high_spread_low_error"
    labels[high_spread & high_error] = "high_spread_high_error"
    labels[~high_spread & high_error] = "common_mode_failure"
    labels[~high_spread & ~high_error] = "unanimous_success"
    return labels


def grouped_fold_ids(
    groups: Sequence[str], *, folds: int, seed: int
) -> np.ndarray:
    values = np.asarray(groups).astype(str)
    if values.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    unique, inverse = np.unique(values, return_inverse=True)
    if not 2 <= folds <= len(unique):
        raise ValueError("folds must be between two and the number of unique groups")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique))
    group_folds = np.empty(len(unique), dtype=np.int64)
    group_folds[shuffled] = np.arange(len(unique)) % folds
    return group_folds[inverse]


def grouped_crossfit_ridge(
    features: np.ndarray,
    outcome: np.ndarray,
    groups: Sequence[str],
    *,
    folds: int,
    alpha: float,
    seed: int,
) -> CrossfitResult:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    group_values = np.asarray(groups).astype(str)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(group_values) != len(y):
        raise ValueError("features, outcome, and groups have incompatible shapes")
    if alpha < 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("crossfit inputs must be finite and alpha nonnegative")
    fold_ids = grouped_fold_ids(group_values, folds=folds, seed=seed)
    predictions = np.empty(len(y), dtype=np.float64)
    for fold in range(folds):
        test = fold_ids == fold
        train = ~test
        center = x[train].mean(axis=0)
        scale = x[train].std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        train_x = (x[train] - center) / scale
        test_x = (x[test] - center) / scale
        design = np.column_stack((np.ones(train.sum()), train_x))
        penalty = np.eye(design.shape[1]) * float(alpha)
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y[train])
        predictions[test] = np.column_stack((np.ones(test.sum()), test_x)) @ coefficient
    return CrossfitResult(predictions=predictions, fold_ids=fold_ids)


def grouped_bootstrap_spearman(
    feature: np.ndarray,
    outcome: np.ndarray,
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    x = np.asarray(feature, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    group_values = np.asarray(groups).astype(str)
    if x.ndim != 1 or y.shape != x.shape or group_values.shape != x.shape:
        raise ValueError("feature, outcome, and groups must be matching one-dimensional arrays")
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    unique = np.unique(group_values)
    positions = [np.flatnonzero(group_values == group) for group in unique]
    rng = np.random.default_rng(seed)
    correlations = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        drawn = rng.integers(0, len(unique), size=len(unique))
        selected = np.concatenate([positions[index] for index in drawn])
        if np.ptp(x[selected]) == 0 or np.ptp(y[selected]) == 0:
            correlations[replicate] = np.nan
        else:
            correlations[replicate] = spearman_correlation(x[selected], y[selected])
    return correlations


def paired_outcome_association_rows(
    left: np.ndarray,
    right: np.ndarray,
    groups: Sequence[str],
    regimes: Mapping[str, np.ndarray],
    *,
    left_name: str,
    right_name: str,
    replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    """Relate two outcomes with rank and linear association by output regime."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    group_values = np.asarray(groups).astype(str)
    if x.ndim != 1 or y.shape != x.shape or group_values.shape != x.shape:
        raise ValueError("left, right, and groups must be matching one-dimensional arrays")
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    rows: list[dict[str, object]] = []
    for regime_index, (regime, mask_value) in enumerate(regimes.items()):
        mask = np.asarray(mask_value, dtype=bool)
        if mask.shape != x.shape or mask.sum() < 2:
            raise ValueError("every regime must select at least two aligned rows")
        selected_x, selected_y = x[mask], y[mask]
        selected_groups = group_values[mask]
        unique = np.unique(selected_groups)
        positions = [np.flatnonzero(selected_groups == group) for group in unique]
        rng = np.random.default_rng(seed + regime_index)
        spearman_draws = np.empty(replicates, dtype=np.float64)
        pearson_draws = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            drawn = rng.integers(0, len(unique), size=len(unique))
            selected = np.concatenate([positions[index] for index in drawn])
            if np.ptp(selected_x[selected]) == 0 or np.ptp(selected_y[selected]) == 0:
                spearman_draws[replicate] = np.nan
                pearson_draws[replicate] = np.nan
            else:
                spearman_draws[replicate] = spearman_correlation(
                    selected_x[selected], selected_y[selected]
                )
                pearson_draws[replicate] = np.corrcoef(
                    selected_x[selected], selected_y[selected]
                )[0, 1]
        row: dict[str, object] = {
            "left_outcome": left_name,
            "right_outcome": right_name,
            "regime": str(regime),
            "sample_count": int(mask.sum()),
            "spearman": spearman_correlation(selected_x, selected_y),
            "pearson": float(np.corrcoef(selected_x, selected_y)[0, 1]),
            "resampling_unit": "equilibrium_files",
            "interval_kind": "grouped_resample_sensitivity_interval",
        }
        for name, draws in (("spearman", spearman_draws), ("pearson", pearson_draws)):
            finite = draws[np.isfinite(draws)]
            lower, upper = np.quantile(finite, (0.025, 0.975))
            row[f"{name}_interval_lower"] = float(lower)
            row[f"{name}_interval_upper"] = float(upper)
            row[f"{name}_finite_resamples"] = int(len(finite))
        rows.append(row)
    return rows


def diagnostic_association_rows(
    features: Mapping[str, np.ndarray],
    outcomes: Mapping[str, np.ndarray],
    groups: Sequence[str],
    regimes: Mapping[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    """Report every frozen feature/outcome association without residual selection."""
    group_values = np.asarray(groups).astype(str)
    count = len(group_values)
    arrays = {
        **{f"feature:{key}": np.asarray(value, dtype=np.float64) for key, value in features.items()},
        **{f"outcome:{key}": np.asarray(value, dtype=np.float64) for key, value in outcomes.items()},
        **{f"regime:{key}": np.asarray(value, dtype=bool) for key, value in regimes.items()},
    }
    if any(value.ndim != 1 or len(value) != count for value in arrays.values()):
        raise ValueError("all diagnostic arrays must match the one-dimensional group axis")
    rows: list[dict[str, object]] = []
    combination = 0
    for regime_name, mask_value in regimes.items():
        mask = np.asarray(mask_value, dtype=bool)
        for feature_name, feature_value in features.items():
            x = np.asarray(feature_value, dtype=np.float64)[mask]
            for outcome_name, outcome_value in outcomes.items():
                y = np.asarray(outcome_value, dtype=np.float64)[mask]
                selected_groups = group_values[mask]
                variable = len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0
                point = spearman_correlation(x, y) if variable else float("nan")
                if variable and len(np.unique(selected_groups)) >= 2:
                    draws = grouped_bootstrap_spearman(
                        x,
                        y,
                        selected_groups,
                        replicates=replicates,
                        seed=seed + combination,
                    )
                    finite = draws[np.isfinite(draws)]
                else:
                    finite = np.asarray([], dtype=np.float64)
                lower, upper = (
                    np.quantile(finite, (0.025, 0.975))
                    if len(finite)
                    else (float("nan"), float("nan"))
                )
                rows.append(
                    {
                        "feature": str(feature_name),
                        "outcome": str(outcome_name),
                        "regime": str(regime_name),
                        "sample_count": int(mask.sum()),
                        "spearman": float(point),
                        "interval_lower": float(lower),
                        "interval_upper": float(upper),
                        "finite_resamples": int(len(finite)),
                        "resampling_unit": "equilibrium_files",
                        "interval_kind": "grouped_resample_sensitivity_interval",
                        "feature_selection": "none_frozen_before_residual_analysis",
                    }
                )
                combination += 1
    return rows


def perturbation_effect_rows(
    reference: np.ndarray,
    perturbed: np.ndarray,
    *,
    member_ids: Sequence[str],
    row_ids: Sequence[int],
    perturbation: str,
    validity: ValidityTag,
) -> list[dict[str, object]]:
    baseline = np.asarray(reference, dtype=np.float64)
    edited = np.asarray(perturbed, dtype=np.float64)
    if baseline.shape != edited.shape or baseline.ndim != 2:
        raise ValueError("reference and perturbed must share shape (member, sample)")
    if baseline.shape != (len(member_ids), len(row_ids)):
        raise ValueError("member_ids and row_ids do not match prediction axes")
    rows: list[dict[str, object]] = []
    difference = edited - baseline
    for member_index, member_id in enumerate(member_ids):
        for sample_index, row_id in enumerate(row_ids):
            rows.append(
                {
                    "member_id": str(member_id),
                    "row_id": int(row_id),
                    "perturbation": str(perturbation),
                    "validity": validity.value,
                    "signed_change_native": float(difference[member_index, sample_index]),
                    "estimand": "max(log Q, -2)",
                }
            )
    return rows
