"""Cross-member representation and causal-signature comparisons for S10."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


HIDDEN_INTERVENTION_VALIDITY = "deliberately_off_manifold_diagnostic"


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Return linear centered-kernel alignment for two feature matrices."""

    x = _feature_matrix(left, "left")
    y = _feature_matrix(right, "right")
    if len(x) != len(y):
        raise ValueError("left and right must contain identical examples")
    # The Gram form is algebraically identical to the feature-cross-product
    # form and is dramatically cheaper for flattened spatial layers where the
    # feature count is much larger than the probe-example count.
    left_gram = _normalized_centered_gram(x)
    right_gram = _normalized_centered_gram(y)
    return float(np.dot(left_gram, right_gram))


def residualize_against_covariates(
    values: np.ndarray, covariates: np.ndarray
) -> np.ndarray:
    """Remove a linear covariate fit while retaining each feature axis."""

    y = _feature_matrix(values, "values")
    x = _feature_matrix(covariates, "covariates")
    if len(y) != len(x):
        raise ValueError("values and covariates must contain identical rows")
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ coefficients


def functional_similarity(
    left_activations: np.ndarray,
    right_activations: np.ndarray,
    *,
    covariates: np.ndarray,
    left_auxiliary: np.ndarray,
    right_auxiliary: np.ndarray,
    left_effects: np.ndarray,
    right_effects: np.ndarray,
    component_weights: Sequence[float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a unit-by-unit similarity matrix from registered signatures."""

    left = _feature_matrix(left_activations, "left_activations")
    right = _feature_matrix(right_activations, "right_activations")
    if len(left) != len(right):
        raise ValueError("activation matrices must use identical panel rows")
    cov = _feature_matrix(covariates, "covariates")
    if len(cov) != len(left):
        raise ValueError("covariates must use the activation rows")
    left_aux = _unit_signatures(left_auxiliary, left.shape[1], "left_auxiliary")
    right_aux = _unit_signatures(right_auxiliary, right.shape[1], "right_auxiliary")
    left_fx = _unit_signatures(left_effects, left.shape[1], "left_effects")
    right_fx = _unit_signatures(right_effects, right.shape[1], "right_effects")
    weights = np.asarray(tuple(component_weights), dtype=np.float64)
    if weights.shape != (4,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("four nonnegative component weights summing to one are required")

    raw = np.clip(_column_correlation(left, right), 0.0, 1.0)
    residual = np.clip(
        _column_correlation(
            residualize_against_covariates(left, cov),
            residualize_against_covariates(right, cov),
        ),
        0.0,
        1.0,
    )
    auxiliary = np.clip(_row_cosine(left_aux, right_aux), 0.0, 1.0)
    effects = np.clip(_row_cosine(left_fx, right_fx), 0.0, 1.0)
    pieces = {
        "activation_raw": raw,
        "activation_residual": residual,
        "auxiliary": auxiliary,
        "causal_effect": effects,
    }
    stacked = np.stack(tuple(pieces.values()), axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0)), pieces


def match_units(
    similarity: np.ndarray, *, minimum_similarity: float
) -> tuple[tuple[int, int, float], ...]:
    """Maximum-weight one-to-one assignment with below-threshold units unmatched."""

    scores = np.asarray(similarity, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.all(np.isfinite(scores)):
        raise ValueError("similarity must be a finite nonempty matrix")
    if not 0.0 <= minimum_similarity <= 1.0:
        raise ValueError("minimum_similarity must lie in [0, 1]")
    # A private dummy column for every left unit allows every unit to remain
    # unmatched.  The Hungarian solver maximizes the global, not greedy, score.
    augmented = np.column_stack(
        (scores, np.full((scores.shape[0], scores.shape[0]), minimum_similarity))
    )
    assignment = _hungarian_minimize(1.0 - augmented)
    matches = []
    for left_index, right_index in enumerate(assignment):
        if right_index < scores.shape[1] and scores[left_index, right_index] >= minimum_similarity:
            matches.append((left_index, int(right_index), float(scores[left_index, right_index])))
    return tuple(matches)


def grouped_bootstrap_match_recurrence(
    left_activations: np.ndarray,
    right_activations: np.ndarray,
    *,
    groups: np.ndarray,
    covariates: np.ndarray,
    left_auxiliary: np.ndarray,
    right_auxiliary: np.ndarray,
    left_effects: np.ndarray,
    right_effects: np.ndarray,
    component_weights: Sequence[float],
    minimum_similarity: float,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Return pair recurrence from resampling complete equilibrium groups."""

    left = _feature_matrix(left_activations, "left_activations")
    right = _feature_matrix(right_activations, "right_activations")
    group_values = np.asarray(groups)
    cov = _feature_matrix(covariates, "covariates")
    if group_values.ndim != 1 or len(group_values) != len(left):
        raise ValueError("groups must contain one equilibrium ID per row")
    if len(right) != len(left) or len(cov) != len(left):
        raise ValueError("all row-level arrays must use identical rows")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    unique = np.unique(group_values)
    rng = np.random.default_rng(seed)
    counts = np.zeros((left.shape[1], right.shape[1]), dtype=np.float64)
    for _ in range(replicates):
        row_weights = _group_bootstrap_row_weights(group_values, rng)
        score = _weighted_functional_similarity(
            left, right, covariates=cov, row_weights=row_weights,
            left_auxiliary=left_auxiliary, right_auxiliary=right_auxiliary,
            left_effects=left_effects, right_effects=right_effects,
            component_weights=component_weights,
        )
        for left_index, right_index, _score in match_units(
            score, minimum_similarity=minimum_similarity
        ):
            counts[left_index, right_index] += 1.0
    return counts / float(replicates)


def _group_bootstrap_row_weights(
    groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw whole groups and give every row in one group equal multiplicity."""

    group_values = np.asarray(groups)
    if group_values.ndim != 1 or not len(group_values):
        raise ValueError("groups must be a nonempty vector")
    unique, inverse = np.unique(group_values, return_inverse=True)
    sampled = rng.integers(0, len(unique), size=len(unique))
    group_counts = np.bincount(sampled, minlength=len(unique)).astype(np.float64)
    return group_counts[inverse]


def mean_replacement_effects(
    representations: np.ndarray,
    head: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Signed native-output changes after replacing one unit by its panel mean."""

    values = _feature_matrix(representations, "representations")
    prediction = np.asarray(head(values), dtype=np.float64)
    if prediction.shape == (len(values), 1):
        prediction = prediction[:, 0]
    if prediction.shape != (len(values),) or not np.all(np.isfinite(prediction)):
        raise ValueError("head must return one finite native output per row")
    effects = np.empty_like(values)
    reference = values.mean(axis=0)
    for unit in range(values.shape[1]):
        edited = values.copy()
        edited[:, unit] = reference[unit]
        changed = np.asarray(head(edited), dtype=np.float64).reshape(-1)
        if changed.shape != prediction.shape or not np.all(np.isfinite(changed)):
            raise ValueError("head must retain one finite output per edited row")
        effects[:, unit] = prediction - changed
    return effects


def member_distance_matrix(feature_blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Combine standardized prediction/attribution/causal/concept feature blocks."""

    if not feature_blocks:
        raise ValueError("at least one evidence block is required")
    transformed = []
    member_count = None
    for block_index, block in enumerate(feature_blocks):
        values = _feature_matrix(block, f"feature_blocks[{block_index}]")
        if member_count is None:
            member_count = len(values)
        elif len(values) != member_count:
            raise ValueError("every evidence block must use identical members")
        median = np.median(values, axis=0)
        scale = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
        active = scale > np.finfo(float).eps
        if np.any(active):
            standardized = (values[:, active] - median[active]) / scale[active]
            transformed.append(standardized / np.sqrt(standardized.shape[1]))
    if not transformed:
        return np.zeros((int(member_count or 0), int(member_count or 0)))
    combined = np.concatenate(transformed, axis=1) / np.sqrt(len(transformed))
    differences = combined[:, None, :] - combined[None, :, :]
    return np.sqrt(np.square(differences).sum(axis=2))


def grouped_bootstrap_cka(
    representations: Sequence[np.ndarray],
    *,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise CKA and equilibrium-grouped 95% bootstrap intervals."""

    matrices = tuple(
        _feature_matrix(values, f"representations[{index}]")
        for index, values in enumerate(representations)
    )
    if not matrices:
        raise ValueError("at least one representation is required")
    sample_count = len(matrices[0])
    if any(len(values) != sample_count for values in matrices):
        raise ValueError("representations must use identical examples")
    group_values = np.asarray(groups)
    if group_values.ndim != 1 or len(group_values) != sample_count:
        raise ValueError("groups must contain one equilibrium ID per example")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    raw_grams = tuple(values @ values.T for values in matrices)

    def matrix_for(indices: np.ndarray) -> np.ndarray:
        vectors = np.stack([
            _normalized_centered_gram_matrix(gram[np.ix_(indices, indices)])
            for gram in raw_grams
        ])
        return np.clip(vectors @ vectors.T, 0.0, 1.0)

    point = matrix_for(np.arange(sample_count))
    unique = np.unique(group_values)
    rows_by_group = [np.flatnonzero(group_values == group) for group in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(matrices), len(matrices)), dtype=np.float64)
    for replicate in range(replicates):
        chosen = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([rows_by_group[index] for index in chosen])
        draws[replicate] = matrix_for(indices)
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    return point, lower, upper


def _feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return matrix


def _normalized_centered_gram(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    return _normalized_centered_gram_matrix(centered @ centered.T)


def _normalized_centered_gram_matrix(gram: np.ndarray) -> np.ndarray:
    gram = (
        gram
        - gram.mean(axis=0, keepdims=True)
        - gram.mean(axis=1, keepdims=True)
        + gram.mean()
    )
    norm = np.linalg.norm(gram)
    if norm <= np.finfo(float).eps:
        return np.zeros(gram.size, dtype=np.float64)
    return (gram / norm).ravel()


def _weighted_functional_similarity(
    left: np.ndarray,
    right: np.ndarray,
    *,
    covariates: np.ndarray,
    row_weights: np.ndarray,
    left_auxiliary: np.ndarray,
    right_auxiliary: np.ndarray,
    left_effects: np.ndarray,
    right_effects: np.ndarray,
    component_weights: Sequence[float],
) -> np.ndarray:
    weights = np.asarray(row_weights, dtype=np.float64)
    component = np.asarray(tuple(component_weights), dtype=np.float64)
    raw = np.clip(_weighted_column_correlation(left, right, weights), 0.0, 1.0)
    residual = np.clip(
        _weighted_column_correlation(
            _weighted_residualize(left, covariates, weights),
            _weighted_residualize(right, covariates, weights),
            weights,
        ), 0.0, 1.0,
    )
    auxiliary = np.clip(
        _row_cosine(np.asarray(left_auxiliary), np.asarray(right_auxiliary)), 0.0, 1.0
    )
    effects = np.clip(
        _row_cosine(np.asarray(left_effects), np.asarray(right_effects)), 0.0, 1.0
    )
    return np.tensordot(component, np.stack((raw, residual, auxiliary, effects)), axes=(0, 0))


def _weighted_residualize(
    values: np.ndarray, covariates: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    design = np.column_stack((np.ones(len(covariates)), covariates))
    normal = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights[:, None] * values)
    coefficients = np.linalg.pinv(normal, hermitian=True) @ rhs
    return values - design @ coefficients


def _weighted_column_correlation(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    total = weights.sum()
    if total <= 0:
        return np.zeros((left.shape[1], right.shape[1]))
    left_centered = left - (weights[:, None] * left).sum(0) / total
    right_centered = right - (weights[:, None] * right).sum(0) / total
    numerator = left_centered.T @ (weights[:, None] * right_centered)
    left_norm = np.sqrt((weights[:, None] * np.square(left_centered)).sum(0))
    right_norm = np.sqrt((weights[:, None] * np.square(right_centered)).sum(0))
    denominator = left_norm[:, None] * right_norm[None, :]
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def _unit_signatures(values: np.ndarray, units: int, name: str) -> np.ndarray:
    matrix = _feature_matrix(values, name)
    if len(matrix) != units:
        raise ValueError(f"{name} must have one row per unit")
    return matrix


def _column_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = left - left.mean(axis=0, keepdims=True)
    y = right - right.mean(axis=0, keepdims=True)
    numerator = x.T @ y
    denominator = np.linalg.norm(x, axis=0)[:, None] * np.linalg.norm(y, axis=0)[None, :]
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(float).eps,
    )


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[1] != right.shape[1]:
        raise ValueError("paired signature families must have identical columns")
    denominator = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
    return np.divide(
        left @ right.T,
        denominator,
        out=np.zeros((len(left), len(right)), dtype=np.float64),
        where=denominator > np.finfo(float).eps,
    )


def _hungarian_minimize(cost: np.ndarray) -> np.ndarray:
    """Rectangular Hungarian assignment for rows <= columns."""

    values = np.asarray(cost, dtype=np.float64)
    rows, columns = values.shape
    if rows > columns:
        raise ValueError("Hungarian solver requires rows <= columns")
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    p = np.zeros(columns + 1, dtype=np.int64)
    way = np.zeros(columns + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        p[0] = row
        minimum = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = values[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = np.full(rows, -1, dtype=np.int64)
    for column in range(1, columns + 1):
        if p[column] != 0:
            assignment[p[column] - 1] = column - 1
    return assignment
