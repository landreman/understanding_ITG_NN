"""S07 alignment of learned spatial signals with held-out GX diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


AlignmentMode = Literal["signed", "positive_contribution"]


@dataclass(frozen=True)
class CircularAlignment:
    """A cyclic learned-signal/GX-profile comparison."""

    mode: AlignmentMode
    best_lag: int
    rank_correlation: float
    rank_ci_lower: float
    rank_ci_upper: float
    lag_recurrence: float
    overlap: float
    overlap_chance: float
    overlap_enrichment: float
    cross_correlation_by_lag: np.ndarray
    rank_correlation_by_lag: np.ndarray
    per_sample_rank_correlation: np.ndarray
    per_sample_overlap: np.ndarray
    bootstrap_rank_correlation: np.ndarray
    bootstrap_best_lag: np.ndarray
    bootstrap_group: str


@dataclass(frozen=True)
class ScalarAssociation:
    """A grouped rank association between a learned summary and GX scalar."""

    spearman_rho: float
    ci_lower: float
    ci_upper: float
    bootstrap_rho: np.ndarray
    bootstrap_group: str


@dataclass(frozen=True)
class PairedDifference:
    """A paired grouped-bootstrap mean difference in native units."""

    estimate: float
    ci_lower: float
    ci_upper: float
    bootstrap_difference: np.ndarray
    bootstrap_group: str
    estimand: str


def circular_alignment(
    learned: np.ndarray,
    gx_profile: np.ndarray,
    groups: np.ndarray,
    *,
    mode: AlignmentMode,
    sparsity: float,
    bootstrap_replicates: int,
    seed: int,
) -> CircularAlignment:
    """Compare learned and physical profiles over all 96 circular lags."""

    left, right, group_values = _validated_profiles(learned, gx_profile, groups)
    if mode not in ("signed", "positive_contribution"):
        raise ValueError(f"unknown alignment mode: {mode!r}")
    if not 0 < sparsity <= 1:
        raise ValueError("sparsity must lie in (0, 1]")
    if bootstrap_replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    if mode == "positive_contribution":
        left = np.maximum(left, 0.0)
        right = np.maximum(right, 0.0)

    left_rank = _rank_rows(left)
    right_rank = _rank_rows(right)
    sample_count, grid_size = left.shape
    per_sample_rank = np.empty((sample_count, grid_size), dtype=np.float64)
    per_sample_cross = np.empty_like(per_sample_rank)
    for lag in range(grid_size):
        per_sample_rank[:, lag] = _row_correlation(
            left_rank, np.roll(right_rank, lag, axis=1)
        )
        per_sample_cross[:, lag] = _row_correlation(left, np.roll(right, lag, axis=1))
    rank_by_lag = per_sample_rank.mean(axis=0)
    cross_by_lag = per_sample_cross.mean(axis=0)
    best_lag = int(np.argmax(np.abs(rank_by_lag)))

    weights = _grouped_bootstrap_weights(
        group_values, replicates=bootstrap_replicates, seed=seed
    )
    denominators = weights.sum(axis=1, keepdims=True)
    bootstrap_by_lag = weights @ per_sample_rank / denominators
    bootstrap_best = np.argmax(np.abs(bootstrap_by_lag), axis=1).astype(np.int16)
    bootstrap_fixed = bootstrap_by_lag[:, best_lag]

    aligned = np.roll(right, best_lag, axis=1)
    oriented = aligned
    if mode == "signed" and rank_by_lag[best_lag] < 0:
        oriented = -oriented
    left_mask = _upper_mask(left, sparsity)
    right_mask = _upper_mask(oriented, sparsity)
    intersection = np.count_nonzero(left_mask & right_mask, axis=1)
    right_count = np.count_nonzero(right_mask, axis=1)
    left_count = np.count_nonzero(left_mask, axis=1)
    overlap = intersection / np.maximum(right_count, 1)
    chance = left_count / grid_size

    return CircularAlignment(
        mode=mode,
        best_lag=best_lag,
        rank_correlation=float(rank_by_lag[best_lag]),
        rank_ci_lower=float(np.quantile(bootstrap_fixed, 0.025)),
        rank_ci_upper=float(np.quantile(bootstrap_fixed, 0.975)),
        lag_recurrence=float(np.mean(bootstrap_best == best_lag)),
        overlap=float(overlap.mean()),
        overlap_chance=float(chance.mean()),
        overlap_enrichment=float(
            overlap.mean() / max(chance.mean(), np.finfo(np.float64).eps)
        ),
        cross_correlation_by_lag=cross_by_lag,
        rank_correlation_by_lag=rank_by_lag,
        per_sample_rank_correlation=per_sample_rank[:, best_lag],
        per_sample_overlap=overlap,
        bootstrap_rank_correlation=bootstrap_fixed,
        bootstrap_best_lag=bootstrap_best,
        bootstrap_group="equilibrium_files",
    )


def scalar_rank_association(
    learned_summary: np.ndarray,
    gx_scalar: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> ScalarAssociation:
    """Rank-correlate sample-level learned summaries with a held-out GX scalar."""

    left, right, group_values = _validated_vectors(learned_summary, gx_scalar, groups)
    if bootstrap_replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    estimate = _spearman(left, right)
    draws = _grouped_row_draws(group_values, replicates=bootstrap_replicates, seed=seed)
    bootstrap = np.asarray(
        [_spearman(left[draw], right[draw]) for draw in draws], dtype=np.float64
    )
    return ScalarAssociation(
        spearman_rho=estimate,
        ci_lower=float(np.quantile(bootstrap, 0.025)),
        ci_upper=float(np.quantile(bootstrap, 0.975)),
        bootstrap_rho=bootstrap,
        bootstrap_group="equilibrium_files",
    )


def paired_native_difference(
    fixed: np.ndarray,
    varied: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_replicates: int,
    seed: int,
    estimand: str = "native max(log Q, -2)",
) -> PairedDifference:
    """Return fixed-minus-varied paired effects without exponentiating predictions."""

    left, right, group_values = _validated_vectors(fixed, varied, groups)
    if bootstrap_replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    difference = left - right
    weights = _grouped_bootstrap_weights(
        group_values, replicates=bootstrap_replicates, seed=seed
    )
    bootstrap = weights @ difference / weights.sum(axis=1)
    return PairedDifference(
        estimate=float(difference.mean()),
        ci_lower=float(np.quantile(bootstrap, 0.025)),
        ci_upper=float(np.quantile(bootstrap, 0.975)),
        bootstrap_difference=bootstrap,
        bootstrap_group="equilibrium_files",
        estimand=str(estimand),
    )


def select_balanced_case_studies(
    scores: np.ndarray,
    row_ids: np.ndarray,
    groups: np.ndarray,
    *,
    per_direction: int,
    expected_sign: int = 1,
) -> tuple[dict[str, object], ...]:
    """Select equal numbers of distinct-equilibrium supports and contradictions."""

    values = np.asarray(scores, dtype=np.float64)
    rows = np.asarray(row_ids, dtype=np.int64)
    group_values = _validated_groups(groups, len(values))
    if rows.shape != values.shape:
        raise ValueError("scores and row IDs must have the same shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("case-study scores must be finite")
    if per_direction < 1:
        raise ValueError("per_direction must be positive")
    if expected_sign not in (-1, 1):
        raise ValueError("expected_sign must be -1 or +1")

    chosen: list[dict[str, object]] = []
    used_groups: set[str] = set()
    oriented = values * expected_sign
    for case_type, order in (
        ("supporting", np.argsort(-oriented, kind="stable")),
        ("contradicting", np.argsort(oriented, kind="stable")),
    ):
        count = 0
        for index in order:
            group = str(group_values[index])
            if group in used_groups:
                continue
            chosen.append(
                {
                    "case_type": case_type,
                    "row_id": int(rows[index]),
                    "equilibrium_file": group,
                    "score": float(values[index]),
                    "oriented_score": float(oriented[index]),
                    "expected_sign": int(expected_sign),
                }
            )
            used_groups.add(group)
            count += 1
            if count == per_direction:
                break
        if count != per_direction:
            raise ValueError("insufficient distinct equilibria for balanced cases")
    return tuple(chosen)


def _validated_profiles(
    learned: np.ndarray, gx_profile: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(learned, dtype=np.float64)
    right = np.asarray(gx_profile, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            "learned and GX profiles must have the same two-dimensional shape"
        )
    if left.shape[1] != 96:
        raise ValueError("spatial profiles must have 96 periodic positions")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("spatial profiles must be finite")
    return left, right, _validated_groups(groups, len(left))


def _validated_vectors(
    left: np.ndarray, right: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.ndim != 1:
        raise ValueError("paired vectors must have the same shape")
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        raise ValueError("paired vectors must be finite")
    return left_values, right_values, _validated_groups(groups, len(left_values))


def _validated_groups(groups: np.ndarray, sample_count: int) -> np.ndarray:
    values = np.asarray(groups)
    if values.shape != (sample_count,):
        raise ValueError("equilibrium groups must match the sample axis")
    if values.dtype.kind not in "SUO":
        raise ValueError(
            "equilibrium groups must be equilibrium_files labels, not row indices"
        )
    decoded = np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )
    if np.any(decoded == ""):
        raise ValueError("equilibrium_files labels must be nonempty")
    return decoded


def _average_ranks(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(len(flat), dtype=np.float64)
    start = 0
    while start < len(flat):
        stop = start + 1
        while stop < len(flat) and flat[order[stop]] == flat[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _rank_rows(values: np.ndarray) -> np.ndarray:
    return np.stack([_average_ranks(row) for row in values], axis=0)


def _row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(np.square(left_centered), axis=1)
        * np.sum(np.square(right_centered), axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).eps,
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        _row_correlation(_average_ranks(left)[None, :], _average_ranks(right)[None, :])[
            0
        ]
    )


def _upper_mask(values: np.ndarray, sparsity: float) -> np.ndarray:
    count = max(1, int(np.ceil(values.shape[1] * sparsity)))
    threshold = np.partition(values, values.shape[1] - count, axis=1)[
        :, values.shape[1] - count
    ]
    return values >= threshold[:, None]


def _group_rows(groups: np.ndarray) -> tuple[np.ndarray, ...]:
    unique = np.unique(groups)
    return tuple(np.flatnonzero(groups == group) for group in unique)


def _grouped_bootstrap_weights(
    groups: np.ndarray, *, replicates: int, seed: int
) -> np.ndarray:
    group_rows = _group_rows(groups)
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(groups)), dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(group_rows), size=len(group_rows))
        counts = np.bincount(selected, minlength=len(group_rows))
        for group_index, count in enumerate(counts):
            weights[replicate, group_rows[group_index]] = count
    return weights


def _grouped_row_draws(
    groups: np.ndarray, *, replicates: int, seed: int
) -> tuple[np.ndarray, ...]:
    group_rows = _group_rows(groups)
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for _ in range(replicates):
        selected = rng.integers(0, len(group_rows), size=len(group_rows))
        draws.append(np.concatenate([group_rows[index] for index in selected]))
    return tuple(draws)
