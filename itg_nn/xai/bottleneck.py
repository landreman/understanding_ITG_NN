"""Invariant-bottleneck attribution, interventions, and grouped decoders."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch


HeadFunction = Callable[[torch.Tensor], torch.Tensor]
HIDDEN_INTERVENTION_VALIDITY = "deliberately_off_manifold_diagnostic"


@dataclass(frozen=True)
class ShapleyResult:
    values: np.ndarray
    standard_errors: np.ndarray
    baseline_output: np.ndarray
    prediction: np.ndarray
    method: str
    evaluations: int
    permutations: int | None


@dataclass(frozen=True)
class InterventionResult:
    modes: tuple[str, ...]
    feature_names: tuple[str, ...]
    original_prediction: np.ndarray
    single_delta: np.ndarray
    pair_indices: np.ndarray
    pair_delta: np.ndarray
    pair_interaction: np.ndarray
    random_direction_delta: np.ndarray
    random_direction_edit_magnitude: np.ndarray
    validity_tag: str


def exact_or_sampled_shapley(
    head: HeadFunction,
    inputs: torch.Tensor,
    reference: torch.Tensor,
    *,
    exact_max_features: int = 20,
    permutations: int = 256,
    seed: int = 0,
    mask_batch_size: int = 64,
) -> ShapleyResult:
    """Attribute a head output from a fixed cohort-mean reference."""

    values, baseline = _validated_head_inputs(inputs, reference)
    feature_count = values.shape[1]
    if exact_max_features < 1:
        raise ValueError("exact_max_features must be positive")
    if mask_batch_size < 1:
        raise ValueError("mask_batch_size must be positive")
    with torch.inference_mode():
        prediction = _head_output(head, values)
        baseline_output = _head_output(head, baseline.expand_as(values))
        if feature_count <= exact_max_features:
            attribution, evaluations = _exact_shapley(
                head, values, baseline, mask_batch_size=mask_batch_size
            )
            standard_errors = torch.zeros_like(attribution)
            method = "exact_enumeration"
            permutation_count: int | None = None
        else:
            if permutations < 2:
                raise ValueError("sampled Shapley requires at least two permutations")
            try:
                attribution, standard_errors, evaluations = _captum_sampled_shapley(
                    head,
                    values,
                    baseline,
                    permutations=permutations,
                    seed=seed,
                    perturbations_per_eval=mask_batch_size,
                )
                method = "captum_shapley_value_sampling"
            except ModuleNotFoundError:
                # Core CI deliberately omits XAI extras. The tested fallback is
                # the same permutation estimator; registered production uses
                # Captum and records that method in every output row.
                attribution, standard_errors, evaluations = _sampled_shapley(
                    head, values, baseline, permutations=permutations, seed=seed
                )
                method = "permutation_sampling_fallback"
            permutation_count = permutations
    return ShapleyResult(
        values=attribution.cpu().numpy().astype(np.float64),
        standard_errors=standard_errors.cpu().numpy().astype(np.float64),
        baseline_output=baseline_output.cpu().numpy().astype(np.float64),
        prediction=prediction.cpu().numpy().astype(np.float64),
        method=method,
        evaluations=evaluations,
        permutations=permutation_count,
    )


def _validated_head_inputs(
    inputs: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(inputs)
    baseline = torch.as_tensor(reference, dtype=values.dtype, device=values.device)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0:
        raise ValueError("inputs must have shape (nonempty sample, feature)")
    if baseline.ndim == 2 and baseline.shape[0] == 1:
        baseline = baseline[0]
    if baseline.ndim != 1 or baseline.shape[0] != values.shape[1]:
        raise ValueError("reference must have one value per feature")
    if not torch.isfinite(values).all() or not torch.isfinite(baseline).all():
        raise ValueError("head inputs and reference must be finite")
    return values, baseline


def _head_output(head: HeadFunction, values: torch.Tensor) -> torch.Tensor:
    output = head(values)
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.ndim != 1 or len(output) != len(values):
        raise ValueError("head must return one scalar per input row")
    return output


def _coalition_outputs(
    head: HeadFunction,
    values: torch.Tensor,
    baseline: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    sample_count, feature_count = values.shape
    expanded = torch.where(
        masks[:, None, :], values[None, :, :], baseline[None, None, :]
    )
    output = _head_output(head, expanded.reshape(-1, feature_count))
    return output.reshape(len(masks), sample_count)


def _exact_shapley(
    head: HeadFunction,
    values: torch.Tensor,
    baseline: torch.Tensor,
    *,
    mask_batch_size: int,
) -> tuple[torch.Tensor, int]:
    feature_count = values.shape[1]
    coalition_count = 1 << feature_count
    coalition_outputs = torch.empty(
        (coalition_count, len(values)), dtype=values.dtype, device=values.device
    )
    bit_positions = torch.arange(feature_count, device=values.device)
    for start in range(0, coalition_count, mask_batch_size):
        stop = min(start + mask_batch_size, coalition_count)
        integers = torch.arange(start, stop, device=values.device)
        masks = integers[:, None].bitwise_and(1 << bit_positions).ne(0)
        coalition_outputs[start:stop] = _coalition_outputs(
            head, values, baseline, masks
        )

    attribution = torch.zeros_like(values)
    integer_masks = np.arange(coalition_count, dtype=np.int64)
    for feature in range(feature_count):
        without_feature = integer_masks[(integer_masks & (1 << feature)) == 0]
        sizes = np.fromiter(
            (int(mask).bit_count() for mask in without_feature),
            dtype=np.int64,
            count=len(without_feature),
        )
        weights = np.asarray(
            [1.0 / (feature_count * math.comb(feature_count - 1, int(size))) for size in sizes],
            dtype=np.float64,
        )
        lower = torch.as_tensor(without_feature, device=values.device)
        upper = lower | (1 << feature)
        marginal = coalition_outputs[upper] - coalition_outputs[lower]
        weight_tensor = torch.as_tensor(weights, dtype=values.dtype, device=values.device)
        attribution[:, feature] = torch.sum(weight_tensor[:, None] * marginal, dim=0)
    return attribution, coalition_count


def _sampled_shapley(
    head: HeadFunction,
    values: torch.Tensor,
    baseline: torch.Tensor,
    *,
    permutations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    feature_count = values.shape[1]
    generator = np.random.default_rng(seed)
    mean = torch.zeros_like(values)
    sum_squared_difference = torch.zeros_like(values)
    for replicate in range(permutations):
        order = generator.permutation(feature_count)
        state = baseline.expand_as(values).clone()
        previous = _head_output(head, state)
        marginal = torch.empty_like(values)
        for feature in order:
            state[:, int(feature)] = values[:, int(feature)]
            current = _head_output(head, state)
            marginal[:, int(feature)] = current - previous
            previous = current
        difference = marginal - mean
        mean += difference / float(replicate + 1)
        sum_squared_difference += difference * (marginal - mean)
    standard_error = torch.sqrt(
        torch.clamp(sum_squared_difference / float(permutations - 1), min=0)
        / float(permutations)
    )
    return mean, standard_error, permutations * (feature_count + 1)


def _captum_sampled_shapley(
    head: HeadFunction,
    values: torch.Tensor,
    baseline: torch.Tensor,
    *,
    permutations: int,
    seed: int,
    perturbations_per_eval: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    from captum.attr import ShapleyValueSampling

    block_count = min(16, max(2, permutations // 4))
    block_sizes = np.full(block_count, permutations // block_count, dtype=np.int64)
    block_sizes[: permutations % block_count] += 1
    estimates: list[torch.Tensor] = []
    attribution = ShapleyValueSampling(head)
    feature_mask = torch.arange(
        values.shape[1], dtype=torch.long, device=values.device
    ).reshape(1, -1)
    with torch.random.fork_rng(devices=[]):
        for block, block_size in enumerate(block_sizes):
            torch.manual_seed(seed + 104729 * block)
            estimates.append(
                attribution.attribute(
                    values,
                    baselines=baseline.reshape(1, -1),
                    feature_mask=feature_mask,
                    n_samples=int(block_size),
                    perturbations_per_eval=max(1, int(perturbations_per_eval)),
                    show_progress=False,
                )
            )
    stacked = torch.stack(estimates)
    weights = torch.as_tensor(
        block_sizes / block_sizes.sum(), dtype=values.dtype, device=values.device
    )
    mean = torch.sum(weights[:, None, None] * stacked, dim=0)
    centered = stacked - mean
    # Unequal-block weighted standard error. Production uses equal blocks for
    # the registered 256 permutations; this remains defined for CLI overrides.
    effective_blocks = 1.0 / float(np.square(weights.cpu().numpy()).sum())
    weighted_variance = torch.sum(weights[:, None, None] * centered.square(), dim=0)
    standard_error = torch.sqrt(weighted_variance / max(effective_blocks - 1.0, 1.0))
    return mean, standard_error, permutations * (values.shape[1] + 1)


def variance_decomposition(
    shapley_values: np.ndarray, predictions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed output-variance contributions and shares by feature."""

    values = np.asarray(shapley_values, dtype=np.float64)
    output = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or output.shape != (len(values),):
        raise ValueError("Shapley values and predictions have incompatible shapes")
    centered_output = output - output.mean()
    variance = float(np.mean(np.square(centered_output)))
    if variance <= np.finfo(np.float64).tiny:
        raise ValueError("output variance must be positive")
    centered_values = values - values.mean(axis=0, keepdims=True)
    contribution = np.mean(centered_values * centered_output[:, None], axis=0)
    return contribution, contribution / variance


def bottleneck_interventions(
    head: HeadFunction,
    inputs: torch.Tensor,
    reference: torch.Tensor,
    *,
    feature_names: Sequence[str],
    intervention_features: Sequence[int],
    seed: int = 0,
    random_directions: int = 8,
) -> InterventionResult:
    """Run signed unit, pair, and random-direction hidden interventions."""

    values, baseline = _validated_head_inputs(inputs, reference)
    names = tuple(str(name) for name in feature_names)
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must uniquely name every feature")
    selected = tuple(int(index) for index in intervention_features)
    if len(set(selected)) != len(selected) or any(
        index < 0 or index >= values.shape[1] for index in selected
    ):
        raise ValueError("invalid or repeated intervention feature")
    if random_directions < 1:
        raise ValueError("random_directions must be positive")
    rng = np.random.default_rng(seed)
    replacements: list[torch.Tensor] = []
    replacements.append(torch.zeros_like(values))
    replacements.append(baseline.expand_as(values).clone())
    resampled = values.clone()
    for feature in selected:
        order = torch.as_tensor(
            rng.permutation(len(values)), dtype=torch.long, device=values.device
        )
        resampled[:, feature] = values[order, feature]
    replacements.append(resampled)
    modes = ("zero", "mean", "resample")
    pair_indices = np.asarray(
        [(left, right) for offset, left in enumerate(selected) for right in selected[offset + 1 :]],
        dtype=np.int64,
    ).reshape(-1, 2)
    with torch.inference_mode():
        original = _head_output(head, values)
        single = torch.empty(
            (len(modes), len(selected), len(values)),
            dtype=values.dtype,
            device=values.device,
        )
        pair_delta = torch.empty(
            (len(modes), len(pair_indices), len(values)),
            dtype=values.dtype,
            device=values.device,
        )
        pair_interaction = torch.empty_like(pair_delta)
        selected_position = {feature: position for position, feature in enumerate(selected)}
        for mode_index, replacement in enumerate(replacements):
            for position, feature in enumerate(selected):
                edited = values.clone()
                edited[:, feature] = replacement[:, feature]
                single[mode_index, position] = _head_output(head, edited) - original
            for pair_position, (left, right) in enumerate(pair_indices):
                edited = values.clone()
                edited[:, left] = replacement[:, left]
                edited[:, right] = replacement[:, right]
                delta = _head_output(head, edited) - original
                pair_delta[mode_index, pair_position] = delta
                pair_interaction[mode_index, pair_position] = (
                    delta
                    - single[mode_index, selected_position[int(left)]]
                    - single[mode_index, selected_position[int(right)]]
                )

        subset = values[:, selected]
        center = baseline[list(selected)]
        scale = subset.std(dim=0, unbiased=False).clamp_min(
            torch.finfo(values.dtype).eps
        )
        standardized = (subset - center) / scale
        direction_effects = torch.empty(
            (random_directions, len(values)), dtype=values.dtype, device=values.device
        )
        direction_magnitudes = torch.empty(
            random_directions, dtype=values.dtype, device=values.device
        )
        for direction_index in range(random_directions):
            direction = torch.as_tensor(
                rng.normal(size=len(selected)), dtype=values.dtype, device=values.device
            )
            direction /= torch.linalg.vector_norm(direction).clamp_min(
                torch.finfo(values.dtype).eps
            )
            projection = standardized @ direction
            direction_magnitudes[direction_index] = torch.sqrt(
                torch.mean(projection.square())
            )
            edited = values.clone()
            edited[:, selected] = (
                standardized - projection[:, None] * direction[None, :]
            ) * scale + center
            direction_effects[direction_index] = _head_output(head, edited) - original
    return InterventionResult(
        modes=modes,
        feature_names=tuple(names[index] for index in selected),
        original_prediction=original.cpu().numpy().astype(np.float64),
        single_delta=single.cpu().numpy().astype(np.float64),
        pair_indices=pair_indices,
        pair_delta=pair_delta.cpu().numpy().astype(np.float64),
        pair_interaction=pair_interaction.cpu().numpy().astype(np.float64),
        random_direction_delta=direction_effects.cpu().numpy().astype(np.float64),
        random_direction_edit_magnitude=direction_magnitudes.cpu()
        .numpy()
        .astype(np.float64),
        validity_tag=HIDDEN_INTERVENTION_VALIDITY,
    )


def grouped_folds(
    groups: Sequence[object] | np.ndarray, *, n_folds: int, seed: int
) -> tuple[np.ndarray, ...]:
    """Return test-row indices for deterministic equilibrium-grouped folds."""

    labels = np.asarray(groups)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("groups must be a nonempty one-dimensional array")
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if not 2 <= n_folds <= len(unique):
        raise ValueError("n_folds must be between two and the unique group count")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    fold_groups: list[list[int]] = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    for group_index in order[np.argsort(-counts[order], kind="stable")]:
        fold = int(np.argmin(fold_sizes))
        fold_groups[fold].append(int(group_index))
        fold_sizes[fold] += counts[group_index]
    return tuple(
        np.flatnonzero(np.isin(inverse, group_indices)).astype(np.int64)
        for group_indices in fold_groups
    )


def grouped_cv_predictions(
    features: np.ndarray,
    target: np.ndarray,
    groups: Sequence[object] | np.ndarray,
    *,
    kind: str,
    n_folds: int,
    seed: int,
    ridge: float = 1e-3,
    hidden_features: int = 32,
    permute_labels: bool = False,
    minimum_active_fraction: float = 0.01,
    active_tolerance: float = 1e-8,
) -> np.ndarray:
    """Return out-of-fold predictions from a linear or small nonlinear decoder."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    labels = np.asarray(groups)
    if x.ndim != 2 or y.shape != (len(x),) or labels.shape != (len(x),):
        raise ValueError("decoder arrays have incompatible shapes")
    if kind not in {"linear", "nonlinear"}:
        raise ValueError("kind must be 'linear' or 'nonlinear'")
    if ridge <= 0 or hidden_features < 1:
        raise ValueError("ridge and hidden_features must be positive")
    if not 0 <= minimum_active_fraction < 1 or active_tolerance < 0:
        raise ValueError("invalid active-fraction decoder guard")
    folds = grouped_folds(labels, n_folds=n_folds, seed=seed)
    fit_target = _group_permuted_target(y, labels, seed + 1009) if permute_labels else y
    predictions = np.empty(len(y), dtype=np.float64)
    all_rows = np.arange(len(y))
    for fold_index, test_rows in enumerate(folds):
        train_rows = np.setdiff1d(all_rows, test_rows, assume_unique=True)
        mean = x[train_rows].mean(axis=0)
        scale = x[train_rows].std(axis=0)
        supported = (
            np.mean(np.abs(x[train_rows]) > active_tolerance, axis=0)
            > minimum_active_fraction
        )
        near_dead = scale <= 1e-6 * np.maximum(1.0, np.abs(mean))
        supported &= ~near_dead
        scale[~supported] = 1.0
        train_x = (x[train_rows] - mean) / scale
        test_x = (x[test_rows] - mean) / scale
        train_x[:, ~supported] = 0.0
        test_x[:, ~supported] = 0.0
        if kind == "nonlinear":
            rng = np.random.default_rng(seed + 7919 * (fold_index + 1))
            weights = rng.normal(
                scale=1.0 / math.sqrt(max(1, x.shape[1])),
                size=(x.shape[1], hidden_features),
            )
            bias = rng.uniform(-1.0, 1.0, size=hidden_features)
            train_hidden = np.maximum(train_x @ weights + bias, 0.0)
            test_hidden = np.maximum(test_x @ weights + bias, 0.0)
            hidden_mean = train_hidden.mean(axis=0)
            hidden_scale = train_hidden.std(axis=0)
            hidden_supported = (
                np.mean(np.abs(train_hidden) > active_tolerance, axis=0)
                > minimum_active_fraction
            )
            hidden_near_dead = hidden_scale <= 1e-6 * np.maximum(
                1.0, np.abs(hidden_mean)
            )
            hidden_supported &= ~hidden_near_dead
            hidden_scale[~hidden_supported] = 1.0
            train_x = np.column_stack(
                (train_x, (train_hidden - hidden_mean) / hidden_scale)
            )
            test_x = np.column_stack(
                (test_x, (test_hidden - hidden_mean) / hidden_scale)
            )
            hidden_offset = train_x.shape[1] - hidden_features
            unsupported_hidden = hidden_offset + np.flatnonzero(~hidden_supported)
            train_x[:, unsupported_hidden] = 0.0
            test_x[:, unsupported_hidden] = 0.0
        predictions[test_rows] = _ridge_predict(
            train_x, fit_target[train_rows], test_x, ridge=ridge
        )
    return predictions


def _group_permuted_target(
    target: np.ndarray, groups: np.ndarray, seed: int
) -> np.ndarray:
    unique, inverse = np.unique(groups, return_inverse=True)
    group_means = np.asarray(
        [target[inverse == index].mean() for index in range(len(unique))]
    )
    order = np.random.default_rng(seed).permutation(len(unique))
    return group_means[order][inverse]


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(train_x)), train_x))
    test_design = np.column_stack((np.ones(len(test_x)), test_x))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return test_design @ coefficients


def registered_invariants(
    geometry: np.ndarray,
    scalar_features: np.ndarray,
    scalar_feature_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Compute S04's registered geometric targets and simple controls."""

    values = np.asarray(geometry, dtype=np.float64)
    scalars = np.asarray(scalar_features, dtype=np.float64)
    names = tuple(str(name) for name in scalar_feature_names)
    if values.ndim != 3 or values.shape[1:] != (96, 7):
        raise ValueError("geometry must have shape (sample, 96, 7)")
    if scalars.ndim != 2 or len(scalars) != len(values) or scalars.shape[1] != len(names):
        raise ValueError("scalar feature matrix and names do not match")
    required = {"shat", "nfp", "aspect"}
    if not required.issubset(names):
        raise ValueError(f"missing registered scalar controls: {sorted(required - set(names))}")
    bmag = values[:, :, 0]
    grad_x = np.sqrt(values[:, :, 6])
    if np.any(bmag <= 0) or np.any(values[:, :, 6] < 0):
        raise ValueError("registered invariant formulas require B > 0 and |grad x|^2 >= 0")
    fsa_grad_x = np.mean(grad_x / bmag, axis=1) / np.mean(1.0 / bmag, axis=1)
    f_q = np.mean(
        ((values[:, :, 2] > 0).astype(np.float64) + 0.2)
        * grad_x**3
        / bmag,
        axis=1,
    )
    f_stab = np.mean(
        ((values[:, :, 1] > 0).astype(np.float64) + 0.4)
        * grad_x
        / np.sqrt(bmag),
        axis=1,
    )
    if np.any(f_q <= 0):
        raise ValueError("f_Q must be positive before taking its logarithm")
    index = {name: position for position, name in enumerate(names)}
    return {
        "log_FSA_grad_x": np.log(fsa_grad_x),
        "log_f_Q": np.log(f_q),
        "f_stab": f_stab,
        "shat": scalars[:, index["shat"]].copy(),
        "nfp": scalars[:, index["nfp"]].copy(),
        "aspect": scalars[:, index["aspect"]].copy(),
    }
