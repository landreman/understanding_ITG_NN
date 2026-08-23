"""Concept-set completeness and drive-interaction estimators for S09."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CompletenessResult:
    name: str
    held_out_r2: float
    increment_r2: float
    increment_ci95_lower: float
    increment_ci95_upper: float
    selection_stability: float
    prediction: np.ndarray
    fold: np.ndarray
    selected_penalties: np.ndarray
    split_unit: str = "equilibrium_files"


@dataclass(frozen=True)
class DirectionalEffect:
    concept: str
    bin_index: int
    drive_lower: float
    drive_upper: float
    rows: int
    slope: float
    ci95_lower: float
    ci95_upper: float
    validity_tag: str = "observed-comparison"
    bootstrap_unit: str = "equilibrium_files"


@dataclass(frozen=True)
class IntegratedHessianTerm:
    drive_index: int
    concept: str
    mixed_derivative: float
    ci95_lower: float
    ci95_upper: float
    mean_signed_integrated_term: float
    mean_absolute_integrated_term: float
    validity_tag: str = "observed-comparison"
    split_unit: str = "equilibrium_files"
    bootstrap_unit: str = "equilibrium_files"


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    return float("nan") if denominator <= 0 else 1.0 - float(np.sum((target - prediction) ** 2)) / denominator


def _folds(groups: np.ndarray, count: int, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    if count < 2 or len(unique) < count:
        raise ValueError("fold count requires at least that many unique groups")
    shuffled = np.random.default_rng(seed).permutation(unique)
    assignment = {group: index % count for index, group in enumerate(shuffled)}
    return np.asarray([assignment[group] for group in groups], dtype=np.int16)


def _expand(values: np.ndarray) -> np.ndarray:
    """Main effects, squares, and the two registered drive interactions."""
    x = np.asarray(values, dtype=np.float64)
    columns = [x, x * x]
    if x.shape[1] > 2:
        columns.append(np.column_stack([x[:, drive] * x[:, feature] for drive in range(min(2, x.shape[1])) for feature in range(2, x.shape[1])]))
    return np.column_stack(columns)


def _ridge_fit(train_x: np.ndarray, train_y: np.ndarray, penalty: float):
    mean = train_x.mean(0)
    scale = train_x.std(0)
    scale[scale < 1e-10] = 1.0
    x = (train_x - mean) / scale
    y_mean = float(train_y.mean())
    y = train_y - y_mean
    gram = x.T @ x
    coefficient = np.linalg.solve(gram + penalty * np.eye(x.shape[1]), x.T @ y)
    return mean, scale, coefficient, y_mean


def _predict(model, values: np.ndarray) -> np.ndarray:
    mean, scale, coefficient, intercept = model
    return (values - mean) / scale @ coefficient + intercept


def _nested_prediction(values: np.ndarray, target: np.ndarray, groups: np.ndarray, *, outer_fold: np.ndarray, inner_folds: int, penalties: Sequence[float], seed: int) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.empty_like(target)
    selected: list[float] = []
    for outer in np.unique(outer_fold):
        train = outer_fold != outer
        test = ~train
        inner = _folds(groups[train], inner_folds, seed + int(outer) + 1)
        scores = []
        for penalty in penalties:
            inner_prediction = np.empty(np.count_nonzero(train), dtype=np.float64)
            for fold in np.unique(inner):
                fit = inner != fold
                model = _ridge_fit(values[train][fit], target[train][fit], float(penalty))
                inner_prediction[inner == fold] = _predict(model, values[train][inner == fold])
            scores.append(_r2(target[train], inner_prediction))
        best = int(np.nanargmax(scores))
        selected.append(float(penalties[best]))
        model = _ridge_fit(values[train], target[train], selected[-1])
        prediction[test] = _predict(model, values[test])
    return prediction, np.asarray(selected)


def _group_positions(groups: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    return np.concatenate([np.flatnonzero(groups == group) for group in chosen])


def grouped_completeness(
    feature_sets: Mapping[str, np.ndarray], target: np.ndarray, groups: np.ndarray,
    *, outer_folds: int, inner_folds: int, penalties: Sequence[float], seed: int,
    bootstrap_replicates: int,
) -> list[CompletenessResult]:
    """Fit nested quadratic-ridge decoders with equilibrium-held-out predictions."""
    y = np.asarray(target, dtype=np.float64)
    group_values = np.asarray(groups)
    if y.ndim != 1 or group_values.shape != y.shape or len(np.unique(group_values)) < outer_folds:
        raise ValueError("target and groups must align and provide enough unique groups")
    if not feature_sets or not penalties:
        raise ValueError("feature_sets and penalties must be non-empty")
    fold = _folds(group_values, outer_folds, seed)
    unique = np.unique(group_values)
    positions_by_group = {group: np.flatnonzero(group_values == group) for group in unique}
    rng = np.random.default_rng(seed + 65537)
    results: list[CompletenessResult] = []
    previous_prediction = np.full_like(y, y.mean())
    for set_index, (name, raw) in enumerate(feature_sets.items()):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or len(raw) != len(y):
            raise ValueError("every feature set must be a row-aligned matrix")
        values = _expand(raw)
        prediction, selected = _nested_prediction(values, y, group_values, outer_fold=fold, inner_folds=inner_folds, penalties=penalties, seed=seed + 100 * set_index)
        current_r2 = _r2(y, prediction)
        previous_r2 = _r2(y, previous_prediction)
        increment = current_r2 - previous_r2
        draws = np.empty(bootstrap_replicates)
        for draw in range(bootstrap_replicates):
            positions = np.concatenate([positions_by_group[group] for group in rng.choice(unique, len(unique), replace=True)])
            draws[draw] = _r2(y[positions], prediction[positions]) - _r2(y[positions], previous_prediction[positions])
        lower, upper = np.quantile(draws, (0.025, 0.975))
        results.append(CompletenessResult(name, current_r2, increment, float(lower), float(upper), float(np.mean(draws > 0)), prediction, fold.copy(), selected))
        previous_prediction = prediction
    return results


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - x.mean()
    denominator = float(centered @ centered)
    return 0.0 if denominator <= 1e-20 else float(centered @ (y - y.mean()) / denominator)


def stratified_directional_effects(
    concepts: Mapping[str, np.ndarray], drive: np.ndarray, target: np.ndarray,
    groups: np.ndarray, *, bins: int, bootstrap_replicates: int, seed: int,
) -> list[DirectionalEffect]:
    """Estimate observed concept slopes within drive quantile strata."""
    drive_values = np.asarray(drive, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    group_values = np.asarray(groups)
    edges = np.quantile(drive_values, np.linspace(0, 1, bins + 1))
    unique = np.unique(group_values)
    positions_by_group = {group: np.flatnonzero(group_values == group) for group in unique}
    rows: list[DirectionalEffect] = []
    for concept_index, (name, raw) in enumerate(concepts.items()):
        x = np.asarray(raw, dtype=np.float64)
        for bin_index in range(bins):
            mask = (drive_values >= edges[bin_index]) & ((drive_values <= edges[bin_index + 1]) if bin_index == bins - 1 else (drive_values < edges[bin_index + 1]))
            slope = _slope(x[mask], y[mask])
            present = np.unique(group_values[mask])
            rng = np.random.default_rng(seed + 1000 * concept_index + bin_index)
            draws = np.empty(bootstrap_replicates)
            for draw in range(bootstrap_replicates):
                positions = np.concatenate([positions_by_group[group] for group in rng.choice(present, len(present), replace=True)])
                positions = positions[mask[positions]]
                draws[draw] = _slope(x[positions], y[positions])
            lower, upper = np.quantile(draws, (0.025, 0.975))
            rows.append(DirectionalEffect(name, bin_index, float(edges[bin_index]), float(edges[bin_index + 1]), int(mask.sum()), slope, float(lower), float(upper)))
    return rows


def grouped_integrated_hessian(
    values: np.ndarray, target: np.ndarray, groups: np.ndarray, *,
    concept_names: Sequence[str], outer_folds: int, inner_folds: int,
    penalties: Sequence[float], bootstrap_replicates: int, seed: int,
) -> list[IntegratedHessianTerm]:
    """Estimate selected concept-by-drive mixed derivatives out of fold.

    The first two columns are the registered drives; remaining columns are
    named concepts. Four finite decoder evaluations give the exact mixed term
    of the declared quadratic decoder. Integrated terms use the panel medians
    as the observed-comparison background.
    """
    raw = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    group_values = np.asarray(groups)
    if raw.shape != (len(y), len(concept_names) + 2):
        raise ValueError("values must contain two drives followed by named concepts")
    fold = _folds(group_values, outer_folds, seed)
    expanded = _expand(raw)
    _, selected = _nested_prediction(expanded, y, group_values, outer_fold=fold, inner_folds=inner_folds, penalties=penalties, seed=seed)
    center = np.median(raw, axis=0)
    scale = np.subtract(*np.quantile(raw, (0.75, 0.25), axis=0))
    scale[scale < 1e-8] = 1.0
    unique = np.unique(group_values)
    positions_by_group = {group: np.flatnonzero(group_values == group) for group in unique}
    result: list[IntegratedHessianTerm] = []
    for drive in range(2):
        for concept_offset, concept in enumerate(concept_names):
            concept_index = concept_offset + 2
            mixed = np.empty(len(y), dtype=np.float64)
            h_drive = 0.1 * scale[drive]
            h_concept = 0.1 * scale[concept_index]
            for outer in np.unique(fold):
                train = fold != outer
                test = ~train
                model = _ridge_fit(expanded[train], y[train], float(selected[int(outer)]))
                evaluations = []
                for drive_sign, concept_sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    changed = raw[test].copy()
                    changed[:, drive] += drive_sign * h_drive
                    changed[:, concept_index] += concept_sign * h_concept
                    evaluations.append(_predict(model, _expand(changed)))
                mixed[test] = (evaluations[0] - evaluations[1] - evaluations[2] + evaluations[3]) / (4 * h_drive * h_concept)
            integrated = mixed * (raw[:, drive] - center[drive]) * (raw[:, concept_index] - center[concept_index])
            rng = np.random.default_rng(seed + drive * 1000 + concept_offset)
            draws = np.empty(bootstrap_replicates)
            for draw in range(bootstrap_replicates):
                positions = np.concatenate([positions_by_group[group] for group in rng.choice(unique, len(unique), replace=True)])
                draws[draw] = float(np.mean(mixed[positions]))
            lower, upper = np.quantile(draws, (0.025, 0.975))
            result.append(IntegratedHessianTerm(drive, str(concept), float(np.mean(mixed)), float(lower), float(upper), float(np.mean(integrated)), float(np.mean(np.abs(integrated)))))
    return result
