"""Natural-experiment estimators for S13 physical validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .distillation import grouped_folds


@dataclass(frozen=True)
class MatchResult:
    """Equilibrium-disjoint high/low candidate-feature matches."""

    high_positions: np.ndarray
    low_positions: np.ndarray
    distance: np.ndarray
    exposure_contrast: np.ndarray
    nuisance_smd_before: np.ndarray
    nuisance_smd_after: np.ndarray
    split_unit: str = "equilibrium_files"
    validity_tag: str = "observed-comparison"


@dataclass(frozen=True)
class CrossFitResult:
    """Out-of-fold predictions and residuals for a grouped regression."""

    prediction: np.ndarray
    residual: np.ndarray
    fold: np.ndarray
    r2: float
    mse: float
    split_unit: str = "equilibrium_files"
    estimand: str = "native max(log Q, -2)"


@dataclass(frozen=True)
class AIPWResult:
    """Cross-fitted augmented inverse-probability contrast."""

    estimate: float
    influence: np.ndarray
    propensity: np.ndarray
    fold: np.ndarray
    overlap_fraction: float
    split_unit: str = "equilibrium_files"
    validity_tag: str = "observed-comparison"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return stable one-based average ranks without an optional dependency."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("rank input must be one-dimensional")
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def equilibrium_grouped_matches(
    exposure: np.ndarray,
    nuisance: np.ndarray,
    groups: np.ndarray,
    *,
    high_quantile: float = 0.75,
    low_quantile: float = 0.25,
    caliper: float | None = None,
) -> MatchResult:
    """Match high- and low-exposure equilibria on robust-scaled nuisances."""

    values = np.asarray(exposure, dtype=np.float64)
    covariates = np.asarray(nuisance, dtype=np.float64)
    group_values = np.asarray(groups)
    if values.ndim != 1 or covariates.ndim != 2:
        raise ValueError("exposure must be one-dimensional and nuisance two-dimensional")
    if len(values) != len(covariates) or group_values.shape != values.shape:
        raise ValueError("exposure, nuisance, and groups must be row aligned")
    if not 0 < low_quantile < high_quantile < 1:
        raise ValueError("quantiles must satisfy 0 < low < high < 1")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(covariates)):
        raise ValueError("matching inputs must be finite")

    low = np.flatnonzero(values <= np.quantile(values, low_quantile))
    high = np.flatnonzero(values >= np.quantile(values, high_quantile))
    if not len(low) or not len(high):
        raise ValueError("both exposure tails must contain rows")
    center = np.median(covariates, axis=0)
    q25, q75 = np.quantile(covariates, (0.25, 0.75), axis=0)
    scale = q75 - q25
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    standardized = (covariates - center) / scale
    distance = np.sqrt(
        np.mean(
            np.square(standardized[high, None, :] - standardized[None, low, :]),
            axis=2,
        )
    )
    candidates = [
        (float(distance[i, j]), int(high[i]), int(low[j]))
        for i in range(len(high))
        for j in range(len(low))
        if group_values[high[i]] != group_values[low[j]]
        and (caliper is None or distance[i, j] <= caliper)
    ]
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_rows: set[int] = set()
    used_groups: set[object] = set()
    selected: list[tuple[float, int, int]] = []
    for item in candidates:
        _, high_position, low_position = item
        high_group = group_values[high_position].item()
        low_group = group_values[low_position].item()
        if (
            high_position in used_rows
            or low_position in used_rows
            or high_group in used_groups
            or low_group in used_groups
        ):
            continue
        selected.append(item)
        used_rows.update((high_position, low_position))
        used_groups.update((high_group, low_group))
    if not selected:
        raise ValueError("no equilibrium-disjoint matches satisfy the caliper")
    selected_distance = np.asarray([item[0] for item in selected])
    selected_high = np.asarray([item[1] for item in selected], dtype=np.int64)
    selected_low = np.asarray([item[2] for item in selected], dtype=np.int64)

    def smd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        pooled = np.sqrt((np.var(left, axis=0) + np.var(right, axis=0)) / 2.0)
        return (np.mean(left, axis=0) - np.mean(right, axis=0)) / np.where(
            pooled > np.finfo(float).eps, pooled, 1.0
        )

    return MatchResult(
        high_positions=selected_high,
        low_positions=selected_low,
        distance=selected_distance,
        exposure_contrast=values[selected_high] - values[selected_low],
        nuisance_smd_before=smd(covariates[high], covariates[low]),
        nuisance_smd_after=smd(covariates[selected_high], covariates[selected_low]),
    )


def grouped_ridge_crossfit(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
    alpha: float = 1.0,
) -> CrossFitResult:
    """Fit a deterministic equilibrium-grouped ridge model out of fold."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    group_values = np.asarray(groups)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != len(y):
        raise ValueError("features must be a row-aligned matrix")
    if group_values.shape != y.shape or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("target and groups must be finite and row aligned")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    fold = grouped_folds(group_values, folds, seed)
    prediction = np.empty_like(y)
    for fold_index in range(folds):
        train = fold != fold_index
        test = ~train
        center = np.mean(x[train], axis=0)
        scale = np.std(x[train], axis=0)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        train_x = (x[train] - center) / scale
        test_x = (x[test] - center) / scale
        design = np.column_stack((np.ones(train.sum()), train_x))
        penalty = np.eye(design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y[train])
        prediction[test] = np.column_stack((np.ones(test.sum()), test_x)) @ coefficient
    residual = y - prediction
    denominator = float(np.sum(np.square(y - y.mean())))
    r2 = float("nan") if denominator <= 0 else 1 - float(np.sum(residual**2) / denominator)
    return CrossFitResult(
        prediction=prediction,
        residual=residual,
        fold=fold,
        r2=r2,
        mse=float(np.mean(np.square(residual))),
    )


def cross_fitted_aipw(
    treated: np.ndarray,
    outcome: np.ndarray,
    nuisance: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
    propensity_clip: float = 0.05,
) -> AIPWResult:
    """Estimate an adjusted observed high/low contrast with grouped cross-fitting."""

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    treatment = np.asarray(treated, dtype=bool)
    y = np.asarray(outcome, dtype=np.float64)
    x = np.asarray(nuisance, dtype=np.float64)
    group_values = np.asarray(groups)
    if treatment.ndim != 1 or y.shape != treatment.shape or x.shape[0] != len(y):
        raise ValueError("treatment, outcome, and nuisance must be row aligned")
    if x.ndim != 2 or group_values.shape != y.shape:
        raise ValueError("nuisance must be a matrix and groups a row vector")
    if not 0 < propensity_clip < 0.5:
        raise ValueError("propensity_clip must lie between zero and one half")
    fold = grouped_folds(group_values, folds, seed)
    propensity = np.empty(len(y), dtype=np.float64)
    mu0 = np.empty(len(y), dtype=np.float64)
    mu1 = np.empty(len(y), dtype=np.float64)
    for fold_index in range(folds):
        train = fold != fold_index
        test = ~train
        if np.unique(treatment[train]).size != 2:
            raise ValueError("each training fold must contain both treatment levels")
        propensity_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, random_state=seed + fold_index, max_iter=2000),
        ).fit(x[train], treatment[train])
        propensity[test] = propensity_model.predict_proba(x[test])[:, 1]
        for level, destination in ((False, mu0), (True, mu1)):
            selected = train & (treatment == level)
            if selected.sum() < 2:
                raise ValueError("each treatment level needs two training rows")
            outcome_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                x[selected], y[selected]
            )
            destination[test] = outcome_model.predict(x[test])
    overlap = float(
        np.mean(
            (propensity >= propensity_clip)
            & (propensity <= 1.0 - propensity_clip)
        )
    )
    clipped = np.clip(propensity, propensity_clip, 1.0 - propensity_clip)
    treatment_float = treatment.astype(np.float64)
    influence = (
        mu1
        - mu0
        + treatment_float * (y - mu1) / clipped
        - (1.0 - treatment_float) * (y - mu0) / (1.0 - clipped)
    )
    return AIPWResult(
        estimate=float(np.mean(influence)),
        influence=influence,
        propensity=propensity,
        fold=fold,
        overlap_fraction=overlap,
    )


def grouped_bootstrap_interval(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float, float]:
    """Return a point estimate and whole-equilibrium bootstrap interval."""

    array = np.asarray(values)
    group_values = np.asarray(groups)
    if array.ndim < 1 or len(array) != len(group_values) or group_values.ndim != 1:
        raise ValueError("values and groups must share their first axis")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    unique = np.unique(group_values)
    positions = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([positions[group] for group in sampled])
        draws.append(float(statistic(array[selected])))
    point = float(statistic(array))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return point, float(lower), float(upper)


def residual_rank_association(
    candidate: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """Spearman association with a whole-equilibrium bootstrap interval."""

    x = np.asarray(candidate, dtype=np.float64)
    y = np.asarray(residual, dtype=np.float64)
    group_values = np.asarray(groups)
    if x.shape != y.shape or x.ndim != 1 or group_values.shape != x.shape:
        raise ValueError("candidate, residual, and groups must be aligned vectors")

    def spearman(selected: np.ndarray) -> float:
        left = _average_ranks(selected[:, 0])
        right = _average_ranks(selected[:, 1])
        if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    return grouped_bootstrap_interval(
        np.column_stack((x, y)),
        group_values,
        replicates=replicates,
        seed=seed,
        statistic=spearman,
    )
