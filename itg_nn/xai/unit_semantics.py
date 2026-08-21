"""S05 unit-density semantics and natural-exemplar analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .audit import rankdata


NATURAL_EXEMPLAR_VALIDITY = "observed-comparison"


@dataclass(frozen=True)
class ConceptTraces:
    values: np.ndarray
    names: tuple[str, ...]
    validity_tag: str


@dataclass(frozen=True)
class AlignmentResult:
    rows: tuple[dict[str, Any], ...]
    recurrence: dict[str, float]
    bootstrap_group: str


@dataclass(frozen=True)
class WrappedPatches:
    values: np.ndarray
    source_positions: np.ndarray
    alignment_operation: str
    validity_tag: str


@dataclass(frozen=True)
class NaturalExemplars:
    sample_indices: np.ndarray
    centers: np.ndarray
    activations: np.ndarray
    selection_unit: str


@dataclass(frozen=True)
class MotifClusters:
    assignment: np.ndarray
    centers: np.ndarray
    dispersion: np.ndarray


@dataclass(frozen=True)
class FilterTransfer:
    kernels: np.ndarray
    amplitude: np.ndarray
    frequency_index: np.ndarray


def physics_concept_traces(
    geometry: np.ndarray,
    *,
    channel_scales: np.ndarray,
    window_widths: Sequence[int] = (1, 9, 25),
) -> ConceptTraces:
    """Build the preregistered pointwise and circular-window concept vocabulary.

    Channel conventions follow the checkpoint input order used by the paper:
    ``(B, gbdrift, cvdrift, gbdrift0/shat, gds2, gds21/shat,
    gds22/shat**2)``.  In these coordinates ``sqrt(gds22/shat**2)`` is
    ``|grad x|`` and ``cvdrift > 0`` is bad curvature.  Derivatives are with
    respect to a unit-period coordinate, hence the factor of 96.
    """

    values = np.asarray(geometry, dtype=np.float64)
    scales = np.asarray(channel_scales, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (96, 7):
        raise ValueError("geometry must have shape (sample, 96, 7)")
    if scales.shape != (7,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("seven positive robust channel scales are required")
    widths = tuple(dict.fromkeys(int(width) for width in window_widths))
    if not widths or any(width < 1 or width > 96 for width in widths):
        raise ValueError("window widths must lie in [1, 96]")
    if np.any(values[:, :, 0] <= 0) or np.any(values[:, :, 6] < 0):
        raise ValueError("concept formulas require B > 0 and gds22 >= 0")

    bmag = values[:, :, 0]
    curvature = values[:, :, 2]
    radial_drift = values[:, :, 3]
    grad_x = np.sqrt(values[:, :, 6])
    bad = (curvature > 0).astype(np.float64)
    shear_ratio = values[:, :, 5] / np.maximum(values[:, :, 6], np.finfo(float).tiny)
    local_shear = 48.0 * (
        np.roll(shear_ratio, -1, axis=1) - np.roll(shear_ratio, 1, axis=1)
    )
    dimensionless = values / scales.reshape(1, 1, 7)
    dimensionless_dz = 48.0 * (
        np.roll(dimensionless, -1, axis=1)
        - np.roll(dimensionless, 1, axis=1)
    )
    parallel_scale = np.sqrt(np.mean(np.square(dimensionless_dz), axis=2))
    bmag_dz = 48.0 * (np.roll(bmag, -1, axis=1) - np.roll(bmag, 1, axis=1))

    base: list[tuple[str, np.ndarray]] = [
        ("bmag", bmag),
        ("inverse_bmag", 1.0 / bmag),
        ("bad_curvature", bad),
        ("good_curvature", 1.0 - bad),
        ("curvature_drift", curvature),
        ("abs_curvature_drift", np.abs(curvature)),
        ("radial_drift_geodesic_curvature", radial_drift),
        ("abs_radial_drift_geodesic_curvature", np.abs(radial_drift)),
        ("compression_abs_grad_x", grad_x),
        ("compression_abs_grad_x_p2", np.square(grad_x)),
        ("compression_abs_grad_x_p3", grad_x**3),
        ("compression_abs_grad_x_p4", grad_x**4),
        ("local_shear_dz_gds21_over_gds22", local_shear),
        ("abs_local_shear", np.abs(local_shear)),
        ("bmag_extremum_strength_abs_dz", np.abs(bmag_dz)),
        ("bmag_local_minimum", (
            (bmag <= np.roll(bmag, 1, axis=1))
            & (bmag < np.roll(bmag, -1, axis=1))
        ).astype(np.float64)),
        ("parallel_scale_local_dimensionless", parallel_scale),
    ]
    fourier_width = max((width for width in widths if width > 1), default=25)
    for name, trace in (
        ("bmag", bmag),
        ("compression", grad_x),
        ("curvature", curvature),
    ):
        base.append(
            (
                f"parallel_fourier_expected_k_{name}_w{fourier_width}",
                _local_expected_fourier_k(trace, fourier_width),
            )
        )
    for power in (1, 2, 3, 4):
        base.append(
            (
                f"bad_curvature_compression_p{power}_Bm1",
                bad * grad_x**power / bmag,
            )
        )
    base.append(("f_Q_integrand_p3_Bm1", (bad + 0.2) * grad_x**3 / bmag))

    names: list[str] = []
    traces: list[np.ndarray] = []
    for name, trace in base:
        names.append(name)
        traces.append(trace)
        for width in widths:
            if width == 1:
                continue
            names.append(f"{name}__mean_w{width}")
            traces.append(_circular_mean(trace, width))
    return ConceptTraces(
        values=np.stack(traces, axis=1),
        names=tuple(names),
        validity_tag=NATURAL_EXEMPLAR_VALIDITY,
    )


def unit_concept_alignment(
    density: np.ndarray,
    concept_traces: np.ndarray,
    *,
    concept_names: Sequence[str],
    channel_magnitude_controls: np.ndarray,
    sparsity: float,
    groups: np.ndarray | None = None,
    bootstrap_replicates: int = 0,
    seed: int = 0,
) -> AlignmentResult:
    """Compare one signed unit density with named traces on a cyclic grid.

    Zero-lag Spearman correlation is retained explicitly.  Overlap and partial
    rank association use the separately reported best circular lag, so a
    translated match is neither discarded nor silently treated as zero-lag.
    """

    target = np.asarray(density, dtype=np.float64)
    traces = np.asarray(concept_traces, dtype=np.float64)
    controls = np.asarray(channel_magnitude_controls, dtype=np.float64)
    names = tuple(str(name) for name in concept_names)
    if target.ndim != 2 or target.shape[1] != 96:
        raise ValueError("density must have shape (sample, 96)")
    if traces.shape != (target.shape[0], len(names), 96):
        raise ValueError("concept traces must have shape (sample, concept, 96)")
    if controls.shape != (target.shape[0], 7, 96):
        raise ValueError("channel controls must have shape (sample, 7, 96)")
    if not 0 < sparsity <= 0.5:
        raise ValueError("sparsity must lie in (0, 0.5]")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(traces)):
        raise ValueError("density and concept traces must be finite")

    sample_count, concept_count, grid_size = traces.shape
    target_centered = target - target.mean(axis=1, keepdims=True)
    trace_centered = traces - traces.mean(axis=2, keepdims=True)
    target_norm = np.linalg.norm(target_centered, axis=1)
    trace_norm = np.linalg.norm(trace_centered, axis=2)
    numerator = np.fft.irfft(
        np.fft.rfft(target_centered, axis=1)[:, None, :]
        * np.conjugate(np.fft.rfft(trace_centered, axis=2)),
        n=grid_size,
        axis=2,
    )
    denominator = target_norm[:, None, None] * trace_norm[:, :, None]
    defined = (
        (target_norm[:, None] > np.finfo(float).eps)
        & (trace_norm > np.finfo(float).eps)
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(float).eps,
    )
    mean_by_lag = correlations.mean(axis=0)
    best_indices = np.argmax(np.abs(mean_by_lag), axis=1)
    best_lags = np.where(best_indices <= grid_size // 2, best_indices, best_indices - grid_size)
    per_sample_lag = correlations[
        np.arange(sample_count)[:, None],
        np.arange(concept_count)[None, :],
        best_indices[None, :],
    ]

    target_ranks = _rank_last(target)
    trace_ranks = _rank_last(traces)
    zero_lag_rank = _row_correlations(target_ranks[:, None, :], trace_ranks).mean(axis=0)
    aligned = np.empty_like(traces)
    aligned_ranks = np.empty_like(trace_ranks)
    for concept, lag in enumerate(best_lags):
        aligned[:, concept] = np.roll(traces[:, concept], int(lag), axis=1)
        aligned_ranks[:, concept] = np.roll(trace_ranks[:, concept], int(lag), axis=1)

    selected_count = max(1, int(np.ceil(sparsity * grid_size)))
    overlap = np.empty((sample_count, concept_count), dtype=np.float64)
    overlap_baseline = np.empty_like(overlap)
    density_mask_count = np.empty_like(overlap)
    concept_mask_count = np.empty_like(overlap)
    density_threshold = np.partition(
        target, grid_size - selected_count, axis=1
    )[:, grid_size - selected_count]
    target_mask = target >= density_threshold[:, None]
    for concept in range(concept_count):
        oriented = aligned[:, concept] * np.where(
            mean_by_lag[concept, best_indices[concept]] >= 0, 1.0, -1.0
        )
        concept_threshold = np.partition(
            oriented, grid_size - selected_count, axis=1
        )[:, grid_size - selected_count]
        concept_mask = oriented >= concept_threshold[:, None]
        for sample in range(sample_count):
            target_count = int(target_mask[sample].sum())
            trace_count = int(concept_mask[sample].sum())
            intersection = int(
                np.count_nonzero(target_mask[sample] & concept_mask[sample])
            )
            density_mask_count[sample, concept] = target_count
            concept_mask_count[sample, concept] = trace_count
            overlap[sample, concept] = intersection / max(trace_count, 1)
            overlap_baseline[sample, concept] = target_count / grid_size

    control_ranks = _rank_last(controls)
    partial = np.zeros((sample_count, concept_count), dtype=np.float64)
    for sample in range(sample_count):
        design = np.column_stack(
            (np.ones(grid_size), control_ranks[sample].T)
        )
        q, _ = np.linalg.qr(design, mode="reduced")
        y = target_ranks[sample]
        y_residual = y - q @ (q.T @ y)
        x = aligned_ranks[sample].T
        x_residual = x - q @ (q.T @ x)
        partial[sample] = _row_correlations(
            y_residual.reshape(1, 1, -1), x_residual.T.reshape(1, concept_count, -1)
        )[0]

    bootstrap_group = "none"
    recurrence: dict[str, float] = {}
    bootstrap_draws: np.ndarray | None = None
    if bootstrap_replicates:
        if groups is None:
            raise ValueError("equilibrium groups are required for bootstrap recurrence")
        weights, _ = grouped_bootstrap_weights(
            np.asarray(groups), replicates=bootstrap_replicates, seed=seed
        )
        denominators = weights.sum(axis=1, keepdims=True)
        bootstrap_draws = weights @ per_sample_lag / denominators
        winner = np.argmax(np.abs(bootstrap_draws), axis=1)
        recurrence = {
            name: float(np.mean(winner == index)) for index, name in enumerate(names)
        }
        bootstrap_group = "equilibrium_files"

    rows: list[dict[str, Any]] = []
    for concept, name in enumerate(names):
        best_index = int(best_indices[concept])
        row: dict[str, Any] = {
            "concept": name,
            "rank_correlation_zero_lag": float(zero_lag_rank[concept]),
            "lag_correlation_zero_lag": float(mean_by_lag[concept, 0]),
            "overlap_at_fixed_sparsity_tie_inclusive": float(
                overlap[:, concept].mean()
            ),
            "overlap_chance_baseline": float(
                overlap_baseline[:, concept].mean()
            ),
            "overlap_enrichment": float(
                overlap[:, concept].mean()
                / max(overlap_baseline[:, concept].mean(), np.finfo(float).eps)
            ),
            "density_mask_mean_count": float(
                density_mask_count[:, concept].mean()
            ),
            "concept_mask_mean_count": float(
                concept_mask_count[:, concept].mean()
            ),
            "sparsity": float(sparsity),
            "best_lag": int(best_lags[concept]),
            "lag_correlation": float(mean_by_lag[concept, best_index]),
            "lag_correlation_defined_rows": float(
                per_sample_lag[defined[:, concept], concept].mean()
            ) if np.any(defined[:, concept]) else float("nan"),
            "n_rows_with_defined_correlation": int(defined[:, concept].sum()),
            "defined_correlation_fraction": float(defined[:, concept].mean()),
            "partial_rank_correlation_density_local_controls": float(
                partial[:, concept].mean()
            ),
            "partial_control_position": "density_position_not_lagged_concept_source",
            "validity_tag": NATURAL_EXEMPLAR_VALIDITY,
            "bootstrap_group": bootstrap_group,
            "bootstrap_recurrence": float(recurrence.get(name, np.nan)),
        }
        if bootstrap_draws is not None:
            row["lag_correlation_ci95_lower"] = float(
                np.quantile(bootstrap_draws[:, concept], 0.025)
            )
            row["lag_correlation_ci95_upper"] = float(
                np.quantile(bootstrap_draws[:, concept], 0.975)
            )
        else:
            row["lag_correlation_ci95_lower"] = float("nan")
            row["lag_correlation_ci95_upper"] = float("nan")
        rows.append(row)
    return AlignmentResult(tuple(rows), recurrence, bootstrap_group)


def grouped_bootstrap_weights(
    groups: np.ndarray, *, replicates: int, seed: int
) -> tuple[np.ndarray, int]:
    labels = np.asarray(groups)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("groups must be a non-empty one-dimensional array")
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    unique, inverse = np.unique(labels, return_inverse=True)
    rng = np.random.default_rng(seed)
    group_weights = np.zeros((replicates, len(unique)), dtype=np.float64)
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
    for replicate in range(replicates):
        group_weights[replicate] = np.bincount(
            draws[replicate], minlength=len(unique)
        )
    return group_weights[:, inverse], len(unique)


def shift_consistency_error(
    reference: np.ndarray,
    shifted: np.ndarray,
    *,
    shift: int,
    position_axis: int,
) -> float:
    baseline = np.asarray(reference)
    transformed = np.asarray(shifted)
    if baseline.shape != transformed.shape:
        raise ValueError("reference and shifted arrays must have matching shapes")
    expected = np.roll(baseline, int(shift), axis=position_axis)
    return float(np.max(np.abs(transformed - expected)))


def extract_wrapped_patches(
    geometry: np.ndarray,
    sample_indices: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
) -> WrappedPatches:
    values = np.asarray(geometry)
    samples = np.asarray(sample_indices, dtype=np.int64)
    activation_centers = np.asarray(centers, dtype=np.int64)
    relative = np.asarray(offsets, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (96, 7):
        raise ValueError("geometry must have shape (sample, 96, 7)")
    if samples.ndim != 1 or activation_centers.shape != samples.shape:
        raise ValueError("sample indices and centers must be matching vectors")
    if relative.ndim != 1 or len(relative) == 0:
        raise ValueError("offsets must be a non-empty vector")
    if len(samples) and (samples.min() < 0 or samples.max() >= len(values)):
        raise IndexError("sample index outside geometry")
    positions = (activation_centers[:, None] + relative[None, :]) % 96
    patches = np.empty((len(samples), 7, len(relative)), dtype=values.dtype)
    for exemplar, (sample, source_positions) in enumerate(zip(samples, positions)):
        patches[exemplar] = values[sample, source_positions].T
    return WrappedPatches(
        patches,
        positions,
        "joint_circular_roll_to_activation_center",
        NATURAL_EXEMPLAR_VALIDITY,
    )


def select_natural_exemplars(
    density: np.ndarray, groups: np.ndarray, *, count: int
) -> NaturalExemplars:
    values = np.asarray(density, dtype=np.float64)
    labels = np.asarray(groups)
    if values.ndim != 2 or values.shape[1] != 96 or labels.shape != (len(values),):
        raise ValueError("density must be (sample, 96) with one group per sample")
    if count < 2:
        raise ValueError("at least two natural exemplars are required")
    maxima = values.max(axis=1)
    centers = values.argmax(axis=1)
    order = np.argsort(-maxima, kind="mergesort")
    chosen: list[int] = []
    seen: set[Any] = set()
    for sample in order:
        group = labels[sample].item() if hasattr(labels[sample], "item") else labels[sample]
        if group in seen:
            continue
        chosen.append(int(sample))
        seen.add(group)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise ValueError("fewer independent equilibrium groups than requested exemplars")
    selected = np.asarray(chosen, dtype=np.int64)
    return NaturalExemplars(
        selected,
        centers[selected].astype(np.int64),
        maxima[selected],
        "equilibrium_files",
    )


def cluster_natural_exemplars(
    patches: np.ndarray, *, clusters: int, seed: int
) -> MotifClusters:
    values = np.asarray(patches, dtype=np.float64)
    if values.ndim != 3 or len(values) < clusters or clusters < 1:
        raise ValueError("patches must be (exemplar, channel, offset) with clusters <= exemplars")
    flattened = values.reshape(len(values), -1)
    median = np.median(flattened, axis=0)
    scale = np.median(np.abs(flattened - median), axis=0)
    fallback = np.std(flattened, axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(fallback > 1e-12, fallback, 1.0))
    standardized = (flattened - median) / scale
    rng = np.random.default_rng(seed)
    center_indices = [int(rng.integers(0, len(values)))]
    while len(center_indices) < clusters:
        distance = np.min(
            np.stack(
                [np.sum(np.square(standardized - standardized[index]), axis=1) for index in center_indices]
            ),
            axis=0,
        )
        distance[center_indices] = -1
        center_indices.append(int(np.argmax(distance)))
    centers_standardized = standardized[center_indices].copy()
    assignment = np.zeros(len(values), dtype=np.int64)
    for _ in range(100):
        distances = np.sum(
            np.square(standardized[:, None, :] - centers_standardized[None, :, :]),
            axis=2,
        )
        updated_assignment = np.argmin(distances, axis=1)
        if np.array_equal(updated_assignment, assignment) and _ > 0:
            break
        assignment = updated_assignment
        for cluster in range(clusters):
            selected = standardized[assignment == cluster]
            if len(selected):
                centers_standardized[cluster] = selected.mean(axis=0)
            else:
                nearest = np.min(distances, axis=1)
                centers_standardized[cluster] = standardized[int(np.argmax(nearest))]

    centers = np.empty((clusters, *values.shape[1:]), dtype=np.float64)
    dispersion = np.empty_like(centers)
    for cluster in range(clusters):
        selected = values[assignment == cluster]
        if not len(selected):
            centers[cluster] = np.nan
            dispersion[cluster] = np.nan
            continue
        centers[cluster] = np.median(selected, axis=0)
        dispersion[cluster] = np.median(
            np.abs(selected - centers[cluster]), axis=0
        )
    return MotifClusters(assignment, centers, dispersion)


def first_layer_transfer(weights: np.ndarray, *, grid_size: int = 96) -> FilterTransfer:
    kernels = np.asarray(weights, dtype=np.float64)
    if kernels.ndim != 3 or kernels.shape[1] != 7:
        raise ValueError("weights must have shape (filter, 7, kernel_position)")
    if grid_size < kernels.shape[2]:
        raise ValueError("grid size must be at least the kernel width")
    transfer = np.fft.rfft(kernels, n=grid_size, axis=2)
    return FilterTransfer(
        kernels.copy(), np.abs(transfer), np.arange(grid_size // 2 + 1)
    )


def native_output_comparison(
    original: np.ndarray,
    invariant: np.ndarray,
    *,
    stable_or_near_floor: np.ndarray,
) -> list[dict[str, Any]]:
    original_values = np.asarray(original, dtype=np.float64)
    invariant_values = np.asarray(invariant, dtype=np.float64)
    stable = np.asarray(stable_or_near_floor, dtype=bool)
    if (
        original_values.ndim != 1
        or invariant_values.shape != original_values.shape
        or stable.shape != original_values.shape
    ):
        raise ValueError("native outputs and stable mask must be matching vectors")
    masks = {
        "overall": np.ones(len(stable), dtype=bool),
        "stable_or_near_floor": stable,
        "unstable": ~stable,
    }
    rows: list[dict[str, Any]] = []
    for stratum, mask in masks.items():
        if not np.any(mask):
            continue
        delta = invariant_values[mask] - original_values[mask]
        rows.append(
            {
                "stratum": stratum,
                "n": int(mask.sum()),
                "original_mean": float(original_values[mask].mean()),
                "invariant_mean": float(invariant_values[mask].mean()),
                "signed_delta_mean": float(delta.mean()),
                "signed_delta_min": float(delta.min()),
                "signed_delta_max": float(delta.max()),
                "delta_rms": float(np.sqrt(np.mean(np.square(delta)))),
                "delta_mean_absolute": float(np.mean(np.abs(delta))),
                "estimand": "native max(log Q, -2)",
                "delta_sign": "invariant_tilde_f minus original_f",
            }
        )
    return rows


def row_permutation_selection_null(
    density: np.ndarray,
    concept_traces: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> np.ndarray:
    """Maximum |correlation| over all concepts/lags after permuting sample pairs."""

    target = np.asarray(density, dtype=np.float64)
    traces = np.asarray(concept_traces, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 96:
        raise ValueError("density must have shape (sample, 96)")
    if traces.ndim != 3 or traces.shape[0] != len(target) or traces.shape[2] != 96:
        raise ValueError("concept traces must have shape (sample, concept, 96)")
    if permutations < 2:
        raise ValueError("at least two row permutations are required")
    trace_centered = traces - traces.mean(axis=2, keepdims=True)
    trace_fft = np.conjugate(np.fft.rfft(trace_centered, axis=2))
    trace_norm = np.linalg.norm(trace_centered, axis=2)
    rng = np.random.default_rng(seed)
    maxima = np.empty(permutations, dtype=np.float64)
    for permutation in range(permutations):
        permuted = target[rng.permutation(len(target))]
        centered = permuted - permuted.mean(axis=1, keepdims=True)
        numerator = np.fft.irfft(
            np.fft.rfft(centered, axis=1)[:, None, :] * trace_fft,
            n=96,
            axis=2,
        )
        denominator = np.linalg.norm(centered, axis=1)[:, None, None] * trace_norm[:, :, None]
        correlations = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > np.finfo(float).eps,
        )
        maxima[permutation] = np.max(np.abs(correlations.mean(axis=0)))
    return maxima


def _circular_mean(values: np.ndarray, width: int) -> np.ndarray:
    left = (width - 1) // 2
    offsets = range(-left, width - left)
    return np.mean([np.roll(values, -offset, axis=1) for offset in offsets], axis=0)


def _local_expected_fourier_k(values: np.ndarray, width: int) -> np.ndarray:
    """Expected non-DC Fourier index in a wrapped window around every position."""

    left = (width - 1) // 2
    offsets = range(-left, width - left)
    windows = np.stack(
        [np.roll(values, -offset, axis=1) for offset in offsets], axis=2
    )
    windows = windows - windows.mean(axis=2, keepdims=True)
    amplitude = np.abs(np.fft.rfft(windows, axis=2))[:, :, 1:]
    frequencies = np.arange(1, amplitude.shape[2] + 1, dtype=np.float64) * (
        96.0 / width
    )
    denominator = amplitude.sum(axis=2)
    return np.divide(
        amplitude @ frequencies,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.finfo(float).eps,
    )


def _rank_last(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    output = np.empty_like(array)
    flattened = array.reshape(-1, array.shape[-1])
    ranked = output.reshape(-1, array.shape[-1])
    for row, source in enumerate(flattened):
        ranked[row] = rankdata(source)
    return output


def _row_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x_centered = x - x.mean(axis=-1, keepdims=True)
    y_centered = y - y.mean(axis=-1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=-1)
    denominator = np.linalg.norm(x_centered, axis=-1) * np.linalg.norm(
        y_centered, axis=-1
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(float).eps,
    )
