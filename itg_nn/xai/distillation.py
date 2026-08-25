"""Invariant feature distillation estimators for S12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .bottleneck import registered_invariants


@dataclass(frozen=True)
class InvariantFeatureTable:
    values: np.ndarray
    names: tuple[str, ...]
    definitions: tuple[dict[str, Any], ...]
    version: str


@dataclass(frozen=True)
class DistillationResult:
    prediction: np.ndarray
    fold: np.ndarray
    held_out_r2: float
    held_out_mse: float
    feature_importance: np.ndarray
    feature_names: tuple[str, ...]
    term_rows: tuple[dict[str, Any], ...]
    split_unit: str = "equilibrium_files"
    estimand: str = "native max(log Q, -2)"


def invariant_feature_table(
    geometry: np.ndarray,
    scalar_features: np.ndarray,
    scalar_feature_names: Sequence[str],
    a_over_lt: np.ndarray,
    a_over_ln: np.ndarray,
    *,
    channel_scales: np.ndarray,
) -> InvariantFeatureTable:
    values = np.asarray(geometry, dtype=np.float64)
    scalars = np.asarray(scalar_features, dtype=np.float64)
    a_lt = np.asarray(a_over_lt, dtype=np.float64)
    a_ln = np.asarray(a_over_ln, dtype=np.float64)
    scales = np.asarray(channel_scales, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (96, 7):
        raise ValueError("geometry must have shape (sample, 96, 7)")
    if scalars.ndim != 2 or len(scalars) != len(values):
        raise ValueError("scalar features must be a row-aligned matrix")
    if a_lt.shape != (len(values),) or a_ln.shape != (len(values),):
        raise ValueError("drive arrays must be row-aligned")
    if scales.shape != (7,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("seven positive robust channel scales are required")

    invariant = registered_invariants(values, scalars, scalar_feature_names)
    bmag = values[:, :, 0]
    curvature = values[:, :, 2]
    geodesic = values[:, :, 3]
    grad_x = np.sqrt(values[:, :, 6])
    if np.any(bmag <= 0) or np.any(values[:, :, 6] < 0):
        raise ValueError("feature formulas require B > 0 and gds22 >= 0")
    bad = (curvature > 0).astype(np.float64)
    bad_compression_trace = bad * grad_x**3 / bmag
    paper_bad_trace = bad * values[:, :, 6] / bmag
    geodesic_trace = np.abs(geodesic) * grad_x / bmag
    dimensionless = values / scales.reshape(1, 1, 7)
    derivative = 48.0 * (
        np.roll(dimensionless, -1, axis=1)
        - np.roll(dimensionless, 1, axis=1)
    )
    roughness = np.sqrt(np.mean(np.square(derivative), axis=(1, 2)))
    colocation = np.mean(bad_compression_trace, axis=1)
    shear_ratio = values[:, :, 5] / np.maximum(values[:, :, 6], np.finfo(float).tiny)
    local_shear = 48.0 * (
        np.roll(shear_ratio, -1, axis=1) - np.roll(shear_ratio, 1, axis=1)
    )

    def expected_k(trace: np.ndarray) -> np.ndarray:
        centered = trace - trace.mean(axis=1, keepdims=True)
        amplitude = np.abs(np.fft.rfft(centered, axis=1))
        frequency = np.arange(amplitude.shape[1], dtype=np.float64)
        return np.sum(amplitude * frequency[None, :], axis=1) / np.maximum(
            np.sum(amplitude, axis=1), np.finfo(float).tiny
        )

    def circular_mean(trace: np.ndarray, width: int) -> np.ndarray:
        left = width // 2
        return np.mean(
            np.stack(
                [np.roll(trace, shift, axis=1) for shift in range(-left, left + 1)],
                axis=0,
            ),
            axis=0,
        )

    columns = (
        ("a_over_LT", a_lt, "drive", "a/L_T"),
        ("a_over_Ln", a_ln, "drive", "a/L_n"),
        ("log_f_Q", invariant["log_f_Q"], "paper", "log mean((H(cvdrift)+0.2)|grad x|^3/B)"),
        ("f_stab", invariant["f_stab"], "paper", "mean((H(gbdrift)+0.4)|grad x|/sqrt(B))"),
        ("log_FSA_grad_x", invariant["log_FSA_grad_x"], "paper", "log flux-surface-average(|grad x|)"),
        ("bad_curvature_compression", np.var(paper_bad_trace, axis=1), "paper/S05/S08", "variance(H(cvdrift)|grad x|^2/B)"),
        ("paper_bad_curvature_feature_mean_square", np.mean(np.square(paper_bad_trace), axis=1), "paper", "mean-square(H(cvdrift)|grad x|^2/B)"),
        ("geodesic_curvature_abs_mean", np.mean(np.abs(geodesic), axis=1), "paper/S05/S08", "mean(abs(radial drift / geodesic curvature))"),
        ("geodesic_curvature_compression", np.mean(geodesic_trace, axis=1), "paper/S05", "mean(abs(geodesic curvature)|grad x|/B)"),
        ("parallel_roughness_iqr_scaled", roughness, "S05", "RMS_z,channel(d/dz(channel/IQR_channel))"),
        ("cross_channel_colocation", colocation, "S09", "mean(H(cvdrift)|grad x|^3/B)"),
        ("bmag_parallel_expected_k", expected_k(bmag), "paper/S05", "expected non-DC Fourier mode of B"),
        ("compression_parallel_expected_k", expected_k(grad_x), "paper/S05", "expected non-DC Fourier mode of |grad x|"),
        ("curvature_parallel_expected_k", expected_k(curvature), "paper/S05", "expected non-DC Fourier mode of cvdrift"),
        ("local_shear_abs_mean", np.mean(np.abs(local_shear), axis=1), "paper/S05", "mean(abs(d/dz(gds21/gds22)))"),
        ("f_Q_integrand_w25_peak", np.max(circular_mean((bad + 0.2) * grad_x**3 / bmag, 25), axis=1), "S05", "max_z circular-mean_25(f_Q integrand)"),
        ("geodesic_abs_w25_peak", np.max(circular_mean(np.abs(geodesic), 25), axis=1), "S05", "max_z circular-mean_25(abs(geodesic curvature))"),
    )
    matrix = np.column_stack([column for _, column, _, _ in columns])
    if np.any(~np.isfinite(matrix)):
        raise ValueError("invariant feature table contains non-finite values")
    definitions = tuple(
        {
            "feature_name": name,
            "formula": formula,
            "source_vocabulary": origin,
            "cyclic_reduction": "invariant",
            "validity_tag": "observed-comparison",
            "version": "S12-v1",
        }
        for name, _, origin, formula in columns
    )
    return InvariantFeatureTable(
        values=matrix,
        names=tuple(name for name, _, _, _ in columns),
        definitions=definitions,
        version="S12-v1",
    )


def grouped_folds(groups: np.ndarray, count: int, seed: int) -> np.ndarray:
    group_values = np.asarray(groups)
    if group_values.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    unique = np.unique(group_values)
    if count < 2 or len(unique) < count:
        raise ValueError("fold count requires at least that many unique groups")
    shuffled = np.random.default_rng(seed).permutation(unique)
    assignment = {group: position % count for position, group in enumerate(shuffled)}
    return np.asarray([assignment[group] for group in group_values], dtype=np.int16)


def _default_estimator_factory(*, seed: int, interactions=()):
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
    except ImportError as error:
        raise ImportError(
            "Explainable Boosting production fits require the optional xai dependency "
            "'interpret'"
        ) from error
    return ExplainableBoostingRegressor(
        interactions=list(interactions),
        max_bins=128,
        max_interaction_bins=32,
        validation_size=0.15,
        outer_bags=8,
        inner_bags=0,
        learning_rate=0.03,
        max_rounds=1000,
        min_samples_leaf=3,
        random_state=int(seed),
        n_jobs=1,
    )


def _main_importance(estimator: Any, feature_count: int) -> np.ndarray:
    if hasattr(estimator, "term_importances"):
        importance = np.asarray(estimator.term_importances(), dtype=np.float64)
    elif hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=np.float64)
    else:
        raise TypeError("estimator does not expose term importances")
    if len(importance) < feature_count:
        raise ValueError("estimator returned fewer importances than input features")
    return importance[:feature_count]


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum(np.square(target - target.mean())))
    return float("nan") if denominator <= 0 else 1.0 - float(
        np.sum(np.square(target - prediction)) / denominator
    )


def grouped_ebm_crossfit(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str],
    folds: int,
    seed: int,
    interactions: Sequence[tuple[int, int]],
    estimator_factory: Callable[..., Any] | None = None,
) -> DistillationResult:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    group_values = np.asarray(groups)
    names = tuple(str(name) for name in feature_names)
    interaction_pairs = tuple((int(left), int(right)) for left, right in interactions)
    if x.ndim != 2 or x.shape != (len(y), len(names)) or group_values.shape != y.shape:
        raise ValueError("features, target, names, and groups must align")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("features and target must be finite")
    if any(left == right or min(left, right) < 0 or max(left, right) >= x.shape[1] for left, right in interaction_pairs):
        raise ValueError("interaction indices must name two distinct input features")
    make_estimator = estimator_factory or _default_estimator_factory
    fold = grouped_folds(group_values, folds, seed)
    prediction = np.empty_like(y)
    fold_importance: list[np.ndarray] = []
    term_rows: list[dict[str, Any]] = []
    for fold_index in range(folds):
        train = fold != fold_index
        test = ~train
        estimator = make_estimator(
            seed=seed + fold_index,
            interactions=interaction_pairs,
        )
        estimator.fit(x[train], y[train])
        prediction[test] = np.asarray(estimator.predict(x[test]), dtype=np.float64)
        importance = _main_importance(estimator, x.shape[1])
        fold_importance.append(importance)
        term_rows.extend(
            {
                "fold": fold_index,
                "term_name": name,
                "term_kind": "main_effect",
                "importance": float(importance[position]),
                "split_unit": "equilibrium_files",
            }
            for position, name in enumerate(names)
        )
    importance = np.mean(np.stack(fold_importance), axis=0)
    residual = prediction - y
    return DistillationResult(
        prediction=prediction,
        fold=fold,
        held_out_r2=_r2(y, prediction),
        held_out_mse=float(np.mean(np.square(residual))),
        feature_importance=importance,
        feature_names=names,
        term_rows=tuple(term_rows),
    )


def grouped_term_recurrence(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str],
    replicates: int,
    seed: int,
    top_k: int,
    estimator_factory: Callable[..., Any] | None = None,
) -> tuple[dict[str, float], ...]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    group_values = np.asarray(groups)
    names = tuple(str(name) for name in feature_names)
    if x.shape != (len(y), len(names)) or group_values.shape != y.shape:
        raise ValueError("features, target, groups, and names must align")
    if replicates < 1 or top_k < 1 or top_k > x.shape[1]:
        raise ValueError("replicates and top_k must be positive and valid")
    make_estimator = estimator_factory or _default_estimator_factory
    unique = np.unique(group_values)
    positions = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(seed)
    selected = np.zeros((replicates, x.shape[1]), dtype=bool)
    importances = np.empty((replicates, x.shape[1]), dtype=np.float64)
    for replicate in range(replicates):
        chosen = rng.choice(unique, len(unique), replace=True)
        sampled = np.concatenate([positions[group] for group in chosen])
        estimator = make_estimator(seed=seed + replicate, interactions=())
        estimator.fit(x[sampled], y[sampled])
        importance = _main_importance(estimator, x.shape[1])
        importances[replicate] = importance
        ranking = np.argsort(-importance, kind="stable")[:top_k]
        selected[replicate, ranking] = True
    return tuple(
        {
            "feature_name": name,
            "top_k_recurrence": float(np.mean(selected[:, index])),
            "median_importance": float(np.median(importances[:, index])),
            "importance_ci95_lower": float(np.quantile(importances[:, index], 0.025)),
            "importance_ci95_upper": float(np.quantile(importances[:, index], 0.975)),
            "bootstrap_unit": "equilibrium_files",
        }
        for index, name in enumerate(names)
    )


def expression_pareto_frontier(
    candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    ordered = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda row: (int(row["complexity"]), -float(row["held_out_r2"])),
    )
    frontier: list[dict[str, Any]] = []
    best_fidelity = -np.inf
    for row in ordered:
        fidelity = float(row["held_out_r2"])
        if fidelity > best_fidelity:
            frontier.append(row)
            best_fidelity = fidelity
    return tuple(frontier)


def expression_recurrence(
    bootstrap_expressions: Sequence[Sequence[str]],
) -> tuple[dict[str, Any], ...]:
    draws = [set(str(expression) for expression in draw) for draw in bootstrap_expressions]
    if not draws:
        raise ValueError("at least one bootstrap expression set is required")
    expressions = sorted(set().union(*draws))
    return tuple(
        {
            "expression": expression,
            "recurrence": float(np.mean([expression in draw for draw in draws])),
            "bootstrap_replicates": len(draws),
            "bootstrap_unit": "equilibrium_files",
        }
        for expression in expressions
    )
