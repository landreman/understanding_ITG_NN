"""Grouped sparse concept probes and representation-direction use tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from itg_nn.xai.symmetry import InvariantMember, _circular_same_convolution


@dataclass(frozen=True)
class SparseProbeResult:
    predictions: np.ndarray
    fold: np.ndarray
    coefficients: np.ndarray
    intercept: float
    held_out_r2: float
    nonzero_fraction: float
    selected_penalties: np.ndarray


@dataclass(frozen=True)
class MatchedExtremes:
    high: np.ndarray
    low: np.ndarray
    validity_tag: str


@dataclass(frozen=True)
class DirectionUse:
    mean_directional_derivative: float
    positive_fraction: float
    intervention_rms: float
    random_intervention_rms_median: float
    random_intervention_rms_q95: float
    scale_matched_random_rms_median: float
    scale_matched_random_rms_q95: float
    orthogonal_complement_ablation_rms: float
    intervention_rms_stable_or_near_floor: float
    intervention_rms_unstable: float
    random_intervention_rms_median_stable_or_near_floor: float
    random_intervention_rms_median_unstable: float
    scale_matched_random_rms_median_stable_or_near_floor: float
    scale_matched_random_rms_median_unstable: float
    validity_tag: str


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return false-discovery-rate adjusted p-values for one test family."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a one-dimensional array in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def invariant_layer_maps(
    member: InvariantMember, geometry: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Return every full-resolution canonical convolutional representation."""

    hidden = geometry.transpose(1, 2)
    dilation = 1
    maps: list[torch.Tensor] = []
    for convolution in member.model.conv_layers:
        hidden = F.relu(_circular_same_convolution(hidden, convolution, dilation))
        hidden = torch.maximum(hidden, torch.roll(hidden, -dilation, dims=-1))
        maps.append(hidden)
        dilation *= 2
    return tuple(maps)


def canonical_output_from_layer(
    member: InvariantMember,
    layer_index: int,
    hidden: torch.Tensor,
    a_over_lt: torch.Tensor,
    a_over_ln: torch.Tensor,
) -> torch.Tensor:
    """Continue S02's canonical network from an intervened layer map."""

    if not 0 <= layer_index < len(member.model.conv_layers):
        raise ValueError("layer_index is outside the convolutional stack")
    dilation = 2 ** (layer_index + 1)
    for convolution in member.model.conv_layers[layer_index + 1 :]:
        hidden = F.relu(_circular_same_convolution(hidden, convolution, dilation))
        hidden = torch.maximum(hidden, torch.roll(hidden, -dilation, dims=-1))
        dilation *= 2
    return member.head(hidden.mean(-1), a_over_lt, a_over_ln)


def _group_folds(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    if folds < 2 or folds > len(unique):
        raise ValueError("fold count must be between 2 and the number of groups")
    shuffled = np.random.default_rng(seed).permutation(unique)
    assignment = {group: index % folds for index, group in enumerate(shuffled)}
    return np.asarray([assignment[group] for group in groups], dtype=np.int16)


def _soft_threshold(value: float, penalty: float) -> float:
    return float(np.sign(value) * max(abs(value) - penalty, 0.0))


def _lasso_fit(
    values: np.ndarray, target: np.ndarray, penalty: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x = (values - mean) / scale
    y_mean = float(target.mean())
    y = target - y_mean
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    residual = y.copy()
    squared_norm = np.mean(x * x, axis=0)
    for _ in range(2000):
        largest_change = 0.0
        for feature in range(x.shape[1]):
            residual += x[:, feature] * coefficients[feature]
            correlation = float(np.mean(x[:, feature] * residual))
            updated = _soft_threshold(correlation, penalty) / max(
                squared_norm[feature], 1e-12
            )
            residual -= x[:, feature] * updated
            largest_change = max(largest_change, abs(updated - coefficients[feature]))
            coefficients[feature] = updated
        if largest_change < 1e-9:
            break
    raw = coefficients / scale
    intercept = y_mean - float(mean @ raw)
    return raw, intercept, mean, scale


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator <= 0:
        return float("nan")
    return 1.0 - float(np.sum((target - prediction) ** 2)) / denominator


def _permute_by_group(target: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    unique = np.unique(groups)
    source = np.random.default_rng(seed).permutation(unique)
    result = np.empty_like(target)
    for destination_group, source_group in zip(unique, source):
        destination = np.flatnonzero(groups == destination_group)
        source_values = target[groups == source_group]
        result[destination] = np.resize(source_values, len(destination))
    return result


def grouped_nested_sparse_probe(
    representation: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    outer_folds: int,
    inner_folds: int,
    penalties: Sequence[float],
    seed: int,
    permute_target: bool = False,
) -> SparseProbeResult:
    x = np.asarray(representation, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    group_values = np.asarray(groups)
    if x.ndim != 2 or y.shape != (len(x),) or group_values.shape != y.shape:
        raise ValueError("representation, target, and groups have incompatible shapes")
    if not penalties or any(value < 0 for value in penalties):
        raise ValueError("penalties must be a non-empty nonnegative sequence")
    if permute_target:
        y = _permute_by_group(y, group_values, seed + 104729)
    fold = _group_folds(group_values, outer_folds, seed)
    predictions = np.empty_like(y)
    selected: list[float] = []
    for outer in range(outer_folds):
        train = fold != outer
        test = ~train
        inner = _group_folds(group_values[train], inner_folds, seed + outer + 1)
        scores: list[float] = []
        for penalty in penalties:
            inner_prediction = np.empty(np.count_nonzero(train), dtype=np.float64)
            for inner_fold in range(inner_folds):
                inner_train = inner != inner_fold
                coefficient, intercept, _, _ = _lasso_fit(
                    x[train][inner_train], y[train][inner_train], float(penalty)
                )
                inner_prediction[inner == inner_fold] = (
                    x[train][inner == inner_fold] @ coefficient + intercept
                )
            scores.append(_r2(y[train], inner_prediction))
        best_score = float(np.nanmax(scores))
        # Prefer the sparsest model whose inner-CV score is practically tied
        # with the maximum.  This is the deterministic analogue of a one-SE rule.
        eligible = [
            index for index, score in enumerate(scores) if score >= best_score - 0.01
        ]
        best = max(eligible, key=lambda index: float(penalties[index]))
        selected.append(float(penalties[best]))
        coefficient, intercept, _, _ = _lasso_fit(x[train], y[train], selected[-1])
        predictions[test] = x[test] @ coefficient + intercept
    final_penalty = float(np.median(selected))
    coefficients, intercept, _, _ = _lasso_fit(x, y, final_penalty)
    return SparseProbeResult(
        predictions=predictions,
        fold=fold,
        coefficients=coefficients,
        intercept=intercept,
        held_out_r2=_r2(y, predictions),
        nonzero_fraction=float(np.mean(coefficients != 0)),
        selected_penalties=np.asarray(selected),
    )


def matched_extremes(
    concept: np.ndarray,
    nuisance: np.ndarray,
    groups: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> MatchedExtremes:
    values = np.asarray(concept, dtype=np.float64)
    nuisance_values = np.asarray(nuisance)
    group_values = np.asarray(groups)
    if nuisance_values.ndim != 2 or len(values) != len(nuisance_values):
        raise ValueError("concept and nuisance values have incompatible shapes")
    if not 0 < fraction <= 0.5:
        raise ValueError("fraction must lie in (0, 0.5]")
    continuous_all = nuisance_values[:, 1:].astype(np.float64)
    continuous_center = np.median(continuous_all, axis=0)
    continuous_scale = np.subtract(
        *np.quantile(continuous_all, [0.75, 0.25], axis=0)
    )
    continuous_scale[continuous_scale < 1e-12] = 1.0
    standardized = (continuous_all - continuous_center) / continuous_scale
    strata = np.unique(nuisance_values[:, 0])
    indicators = (
        np.column_stack(
            [nuisance_values[:, 0] == stratum for stratum in strata[1:]]
        ).astype(np.float64)
        if len(strata) > 1
        else np.empty((len(values), 0), dtype=np.float64)
    )
    design = np.column_stack((np.ones(len(values)), standardized, indicators))
    penalty = np.eye(design.shape[1], dtype=np.float64) * 1e-8
    penalty[0, 0] = 0.0
    fitted = design @ np.linalg.solve(
        design.T @ design + penalty, design.T @ values
    )
    residual_concept = values - fitted
    # The first nuisance is the exact matching stratum (equilibrium class in S08).
    # Choose high examples by concept rank, then nearest unused counterexamples
    # from the lower half using robustly scaled continuous nuisances.
    rng = np.random.default_rng(seed)
    high: list[int] = []
    low: list[int] = []
    for stratum in np.unique(nuisance_values[:, 0]):
        positions = np.flatnonzero(nuisance_values[:, 0] == stratum)
        jitter = rng.uniform(0, 1e-12, size=len(positions))
        order = positions[
            np.argsort(residual_concept[positions] + jitter, kind="stable")
        ]
        count = max(1, int(np.floor(fraction * len(order))))
        selected_high = order[-count:]
        candidates = order[: max(count, len(order) // 2)].tolist()
        continuous = nuisance_values[positions, 1:].astype(np.float64)
        scale = np.subtract(*np.quantile(continuous, [0.75, 0.25], axis=0))
        scale[scale < 1e-12] = 1.0
        for high_position in rng.permutation(selected_high):
            distances = np.sum(
                ((nuisance_values[candidates, 1:] - nuisance_values[high_position, 1:]) / scale)
                ** 2,
                axis=1,
            )
            chosen = candidates.pop(int(np.argmin(distances)))
            high.append(int(high_position))
            low.append(int(chosen))
    high_array = np.asarray(high, dtype=np.int64)
    low_array = np.asarray(low, dtype=np.int64)
    if set(group_values[high_array]) & set(group_values[low_array]):
        raise ValueError("matched extremes reuse an equilibrium group")
    return MatchedExtremes(high_array, low_array, "observed-comparison")


def representation_direction_use(
    output_from_representation: Callable[[torch.Tensor], torch.Tensor],
    representation: torch.Tensor,
    concept_direction: torch.Tensor,
    *,
    random_directions: int,
    intervention_scale: float,
    seed: int,
    feature_scale: torch.Tensor | None = None,
    stable_mask: torch.Tensor | None = None,
) -> DirectionUse:
    if random_directions < 1 or intervention_scale <= 0:
        raise ValueError("random_directions and intervention_scale must be positive")
    base = representation.detach().clone().requires_grad_(True)
    direction = concept_direction.to(base).reshape(-1)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
    output = output_from_representation(base)
    gradient = torch.autograd.grad(output.sum(), base)[0]
    derivatives = gradient @ direction
    with torch.no_grad():
        plus = output_from_representation(base.detach() + intervention_scale * direction)
        minus = output_from_representation(base.detach() - intervention_scale * direction)
        intervention = (plus - minus) / 2
        generator = torch.Generator(device=base.device).manual_seed(seed)
        control_effects: list[torch.Tensor] = []
        scale_matched_control_effects: list[torch.Tensor] = []
        scale = (
            torch.ones_like(direction)
            if feature_scale is None
            else feature_scale.to(base).reshape(-1)
        )
        for _ in range(random_directions):
            random = torch.randn(
                direction.shape,
                generator=generator,
                device=base.device,
                dtype=base.dtype,
            )
            random /= torch.linalg.vector_norm(random).clamp_min(1e-12)
            random_plus = output_from_representation(
                base.detach() + intervention_scale * random
            )
            random_minus = output_from_representation(
                base.detach() - intervention_scale * random
            )
            control_effects.append((random_plus - random_minus) / 2)
            scale_matched = torch.randn(
                direction.shape,
                generator=generator,
                device=base.device,
                dtype=base.dtype,
            ) * scale
            scale_matched /= torch.linalg.vector_norm(scale_matched).clamp_min(1e-12)
            scale_plus = output_from_representation(
                base.detach() + intervention_scale * scale_matched
            )
            scale_minus = output_from_representation(
                base.detach() - intervention_scale * scale_matched
            )
            scale_matched_control_effects.append((scale_plus - scale_minus) / 2)
        center = base.detach().mean(dim=0, keepdim=True)
        coordinate = (base.detach() - center) @ direction
        orthogonal = base.detach() - coordinate[:, None] * direction[None, :]
        orthogonal_output = output_from_representation(orthogonal)
        orthogonal_effect = torch.sqrt(torch.mean((orthogonal_output - output.detach()) ** 2))
        control_matrix = torch.stack(control_effects)
        scale_control_matrix = torch.stack(scale_matched_control_effects)
        controls = torch.sqrt(torch.mean(control_matrix**2, dim=1)).cpu().numpy()
        scale_matched_controls = torch.sqrt(
            torch.mean(scale_control_matrix**2, dim=1)
        ).cpu().numpy()
        mask = (
            torch.zeros(len(base), dtype=torch.bool, device=base.device)
            if stable_mask is None
            else stable_mask.to(device=base.device, dtype=torch.bool)
        )
        if mask.shape != (len(base),):
            raise ValueError("stable_mask must have one value per representation row")

        def rms(values: torch.Tensor, selection: torch.Tensor) -> float:
            return (
                float(torch.sqrt(torch.mean(values[selection] ** 2)))
                if bool(selection.any())
                else float("nan")
            )

        def median_control_rms(values: torch.Tensor, selection: torch.Tensor) -> float:
            if not bool(selection.any()):
                return float("nan")
            per_direction = torch.sqrt(torch.mean(values[:, selection] ** 2, dim=1))
            return float(np.median(per_direction.cpu().numpy()))
    return DirectionUse(
        mean_directional_derivative=float(derivatives.mean()),
        positive_fraction=float((derivatives > 0).float().mean()),
        intervention_rms=float(torch.sqrt(torch.mean(intervention**2))),
        random_intervention_rms_median=float(np.median(controls)),
        random_intervention_rms_q95=float(np.quantile(controls, 0.95)),
        scale_matched_random_rms_median=float(np.median(scale_matched_controls)),
        scale_matched_random_rms_q95=float(np.quantile(scale_matched_controls, 0.95)),
        orthogonal_complement_ablation_rms=float(orthogonal_effect),
        intervention_rms_stable_or_near_floor=rms(intervention, mask),
        intervention_rms_unstable=rms(intervention, ~mask),
        random_intervention_rms_median_stable_or_near_floor=median_control_rms(
            control_matrix, mask
        ),
        random_intervention_rms_median_unstable=median_control_rms(
            control_matrix, ~mask
        ),
        scale_matched_random_rms_median_stable_or_near_floor=median_control_rms(
            scale_control_matrix, mask
        ),
        scale_matched_random_rms_median_unstable=median_control_rms(
            scale_control_matrix, ~mask
        ),
        validity_tag="deliberately_off_manifold_diagnostic",
    )


def uniform_edit_gradient(
    output_from_layer_map: Callable[[torch.Tensor], torch.Tensor],
    layer_map: torch.Tensor,
) -> torch.Tensor:
    """Gradient for adding one hidden-channel direction at every position."""

    base = layer_map.detach().clone().requires_grad_(True)
    output = output_from_layer_map(base)
    # Uniform addition repeats a channel edit at every spatial position, so the
    # chain rule sums (and preserves the sign of) all positionwise gradients.
    return torch.autograd.grad(output.sum(), base)[0].sum(-1)
