"""Dataset, ranking, and panel utilities for the registered S01 audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


STABLE_THRESHOLD = -1.9
RESIDUAL_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def rankdata(values: np.ndarray, *, descending: bool = False) -> np.ndarray:
    """Return one-based average ranks without requiring SciPy."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("rankdata expects one-dimensional values")
    order = np.argsort(-array if descending else array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties."""

    x = rankdata(np.asarray(left))
    y = rankdata(np.asarray(right))
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Metrics in the native clipped-log target space."""

    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("actual and predicted must be matching one-dimensional arrays")
    residual = p - y
    sse = float(np.square(residual).sum())
    centered = y - y.mean()
    tss = float(np.square(centered).sum())
    result = {
        "n": float(len(y)),
        "r2": float(1.0 - sse / tss) if tss > 0 else float("nan"),
        "mse": float(np.mean(np.square(residual))),
        "bias": float(np.mean(residual)),
    }
    for quantile, value in zip(RESIDUAL_QUANTILES, np.quantile(residual, RESIDUAL_QUANTILES)):
        result[f"residual_q{int(quantile * 100):02d}"] = float(value)
    return result


def quantile_bins(values: np.ndarray, count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Stable equal-frequency bins and their registered interior cut points."""

    values = np.asarray(values, dtype=np.float64)
    cuts = np.quantile(values, np.linspace(0, 1, count + 1)[1:-1])
    return np.searchsorted(cuts, values, side="right").astype(np.int8), cuts


def flux_regimes(actual: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Separate exact floor, near threshold, and unstable flux tertiles."""

    y = np.asarray(actual, dtype=np.float64)
    labels = np.empty(len(y), dtype="U24")
    floor = y == -2.0
    near = (y > -2.0) & (y <= STABLE_THRESHOLD)
    unstable = y > STABLE_THRESHOLD
    labels[floor] = "stable_floor"
    labels[near] = "near_threshold"
    unstable_bins, cuts = quantile_bins(y[unstable], 3)
    unstable_labels = np.asarray(("low_flux", "medium_flux", "high_flux"))
    labels[unstable] = unstable_labels[unstable_bins]
    return labels, {
        "stable_floor": "actual clipped log Q == -2",
        "near_threshold": f"-2 < actual clipped log Q <= {STABLE_THRESHOLD}",
        "unstable_flux_tertile_cuts": [float(value) for value in cuts],
    }


def performance_rows(
    actual: np.ndarray,
    member_predictions: np.ndarray,
    member_ids: Sequence[str],
    strata: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create tidy overall and stratified native-output performance tables."""

    predictions = np.asarray(member_predictions)
    if predictions.shape != (len(member_ids), len(actual)):
        raise ValueError("predictions must have shape (member, sample)")
    overall: list[dict[str, Any]] = []
    stratified: list[dict[str, Any]] = []
    all_ids = ("ensemble_mean", *member_ids)
    all_predictions = np.vstack((predictions.mean(axis=0), predictions))
    for member_id, predicted in zip(all_ids, all_predictions):
        overall.append({"member_id": member_id, **regression_metrics(actual, predicted)})
        for variable, levels in strata.items():
            levels_array = np.asarray(levels)
            for level in sorted(np.unique(levels_array).tolist(), key=str):
                selected = levels_array == level
                metrics = regression_metrics(actual[selected], predicted[selected])
                stratified.append(
                    {
                        "member_id": member_id,
                        "stratifier": variable,
                        "level": str(level),
                        **metrics,
                    }
                )
    return overall, stratified


def _group_sums(
    actual: np.ndarray, predictions: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_count = len(unique_groups)
    counts = np.bincount(inverse, minlength=group_count).astype(np.float64)
    sum_y = np.bincount(inverse, weights=actual, minlength=group_count)
    sum_y2 = np.bincount(inverse, weights=np.square(actual), minlength=group_count)
    sum_residual = np.empty((group_count, predictions.shape[0]), dtype=np.float64)
    sum_squared_residual = np.empty_like(sum_residual)
    for index, predicted in enumerate(predictions):
        residual = predicted - actual
        sum_residual[:, index] = np.bincount(
            inverse, weights=residual, minlength=group_count
        )
        sum_squared_residual[:, index] = np.bincount(
            inverse, weights=np.square(residual), minlength=group_count
        )
    return unique_groups, counts, sum_y, sum_y2, sum_residual, sum_squared_residual


@dataclass(frozen=True)
class BootstrapResult:
    r2: np.ndarray
    mse: np.ndarray
    bias: np.ndarray
    ranks: np.ndarray
    group_count: int


def grouped_bootstrap(
    actual: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    """Bootstrap whole equilibrium files and retain member-level results."""

    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(y) or len(groups) != len(y):
        raise ValueError("bootstrap arrays have incompatible shapes")
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    group_data = _group_sums(y, p, np.asarray(groups))
    _, n_by_group, y_by_group, y2_by_group, residual_by_group, sse_by_group = group_data
    group_count = len(n_by_group)
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, group_count), dtype=np.float64)
    draws = rng.integers(0, group_count, size=(replicates, group_count))
    for replicate in range(replicates):
        weights[replicate] = np.bincount(draws[replicate], minlength=group_count)
    sample_n = weights @ n_by_group
    sum_y = weights @ y_by_group
    sum_y2 = weights @ y2_by_group
    tss = sum_y2 - np.square(sum_y) / sample_n
    sse = weights @ sse_by_group
    residual = weights @ residual_by_group
    r2 = 1.0 - sse / tss[:, None]
    mse = sse / sample_n[:, None]
    bias = residual / sample_n[:, None]
    ranks = np.vstack([rankdata(row, descending=True) for row in r2])
    return BootstrapResult(r2=r2, mse=mse, bias=bias, ranks=ranks, group_count=group_count)


def top_k_members(values: np.ndarray, k: int) -> np.ndarray:
    """Return exactly ``k`` member indices, breaking ties by member order."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("top_k_members expects one-dimensional values")
    if not 0 <= k <= len(array):
        raise ValueError("k must be between zero and the number of values")
    return np.argsort(-array, kind="mergesort")[:k]


def row_bootstrap_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Tube-level bootstrap control used only to quantify pseudoreplication."""

    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    rng = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(y), size=len(y))
        output[replicate] = regression_metrics(y[selected], p[selected])["r2"]
    return output


def select_panel_rows(
    row_ids: np.ndarray,
    equilibrium_files: np.ndarray,
    equilibrium_class: np.ndarray,
    flux_regime: np.ndarray,
    lt_bin: np.ndarray,
    ln_bin: np.ndarray,
    absolute_error: np.ndarray,
    disagreement: np.ndarray,
    *,
    panel_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one varied row per equilibrium with coverage-weighted sampling."""

    arrays = tuple(
        np.asarray(value)
        for value in (
            row_ids,
            equilibrium_files,
            equilibrium_class,
            flux_regime,
            lt_bin,
            ln_bin,
            absolute_error,
            disagreement,
        )
    )
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("panel arrays must have equal lengths")
    if panel_size < 1:
        raise ValueError("panel_size must be positive")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(row_ids))
    # One randomly ordered representative per equilibrium makes the sampling
    # unit explicit and prevents the panel itself from pseudoreplicating tubes.
    representatives: list[int] = []
    seen: set[str] = set()
    for index in order:
        group = str(equilibrium_files[index])
        if group not in seen:
            representatives.append(int(index))
            seen.add(group)
    candidates = np.asarray(representatives, dtype=np.int64)
    if panel_size > len(candidates):
        raise ValueError("panel_size exceeds the number of unique equilibria")

    error_cut = float(np.quantile(absolute_error, 0.9))
    disagreement_cut = float(np.quantile(disagreement, 0.9))
    chosen: list[int] = []
    chosen_set: set[int] = set()

    def add_balanced(mask: np.ndarray, quota: int) -> None:
        eligible = candidates[mask[candidates]]
        if not len(eligible):
            return
        target = min(panel_size, len(chosen) + quota)
        shuffled = eligible[rng.permutation(len(eligible))]
        # Round-robin over equilibrium classes rather than allowing the largest
        # class to consume the diagnostic quota.
        by_class = {
            value: list(shuffled[equilibrium_class[shuffled] == value])
            for value in np.unique(equilibrium_class[shuffled])
        }
        while len(chosen) < target and any(by_class.values()):
            for value in sorted(by_class):
                while by_class[value]:
                    index = int(by_class[value].pop())
                    if index not in chosen_set:
                        chosen.append(index)
                        chosen_set.add(index)
                        break
                if len(chosen) >= target:
                    break

    diagnostic_quota = max(1, panel_size // 10)
    add_balanced(absolute_error >= error_cut, diagnostic_quota)
    add_balanced(disagreement >= disagreement_cut, diagnostic_quota)

    remaining = np.asarray(
        [index for index in candidates if int(index) not in chosen_set], dtype=np.int64
    )
    cells = np.asarray(
        [
            f"{equilibrium_class[i]}|{flux_regime[i]}|{lt_bin[i]}|{ln_bin[i]}"
            for i in remaining
        ]
    )
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    probabilities = 1.0 / counts[inverse]
    probabilities /= probabilities.sum()
    needed = panel_size - len(chosen)
    sampled = rng.choice(remaining, size=needed, replace=False, p=probabilities)
    chosen.extend(int(index) for index in sampled)
    selected = np.asarray(chosen, dtype=np.int64)
    selected = selected[np.argsort(row_ids[selected])]
    return row_ids[selected].astype(np.int64), {
        "sampling_unit": "equilibrium_files",
        "maximum_rows_per_equilibrium": 1,
        "error_top_decile_cut": error_cut,
        "disagreement_top_decile_cut": disagreement_cut,
        "diagnostic_quota_per_stage": diagnostic_quota,
        "diagnostic_stages": ["top_decile_absolute_error", "top_decile_disagreement"],
        "seed": seed,
    }


def robust_channel_statistics(geometry: np.ndarray) -> list[dict[str, float]]:
    """Robust scales over sample and position for each physical channel."""

    values = np.asarray(geometry, dtype=np.float64).reshape(-1, geometry.shape[-1])
    output: list[dict[str, float]] = []
    for channel in range(values.shape[1]):
        column = values[:, channel]
        median = float(np.median(column))
        mad = float(np.median(np.abs(column - median)))
        q01, q25, q75, q99 = np.quantile(column, (0.01, 0.25, 0.75, 0.99))
        output.append(
            {
                "channel": channel,
                "median": median,
                "mad": mad,
                "robust_sigma_mad": 1.4826 * mad,
                "iqr": float(q75 - q25),
                "robust_sigma_iqr": float((q75 - q25) / 1.349),
                "q01": float(q01),
                "q99": float(q99),
                "max_abs": float(np.max(np.abs(column))),
            }
        )
    return output


def channel_correlation_statistics(geometry: np.ndarray) -> dict[str, np.ndarray]:
    """Separate local co-location from between-tube channel covariation."""

    values = np.asarray(geometry, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("geometry must have shape (sample, position, channel)")
    pooled = np.corrcoef(values.reshape(-1, values.shape[-1]), rowvar=False)
    tube_means = values.mean(axis=1)
    between_tube = np.corrcoef(tube_means, rowvar=False)
    within_tube_values = values - tube_means[:, None, :]
    within_tube = np.corrcoef(
        within_tube_values.reshape(-1, values.shape[-1]), rowvar=False
    )
    return {
        "pooled_channel_correlation": pooled,
        "within_tube_channel_correlation": within_tube,
        "between_tube_mean_channel_correlation": between_tube,
    }


def median_normalized_power_spectrum(geometry: np.ndarray) -> np.ndarray:
    """Median per-sample spectral shape after dropping DC and normalizing power."""

    values = np.asarray(geometry, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("geometry must have shape (sample, position, channel)")
    power = np.square(np.abs(np.fft.rfft(values, axis=1)))[:, 1:, :]
    normalizer = power.sum(axis=1, keepdims=True)
    normalized = np.divide(
        power,
        normalizer,
        out=np.zeros_like(power),
        where=normalizer > np.finfo(np.float64).tiny,
    )
    return np.median(normalized, axis=0).T
