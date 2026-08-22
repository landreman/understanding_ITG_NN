"""Input-attribution estimators and quantitative benchmark diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import time
from typing import Any, Literal

import numpy as np
import torch

from .perturbations import ValidityTag


GeometryForward = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AttributionMap:
    """A signed or magnitude-only member/sample/channel/position map."""

    values: torch.Tensor
    method: str
    validity: ValidityTag
    signed: bool
    runtime_seconds: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GroupedBootstrapResult:
    """A point estimate and equilibrium-grouped bootstrap distribution."""

    estimate: float
    lower: float
    upper: float
    samples: np.ndarray
    resampling_unit: str = "equilibrium_files"


def native_scaled_gradients(
    forward: GeometryForward, inputs: torch.Tensor, robust_scales: torch.Tensor
) -> AttributionMap:
    started = time.monotonic()
    _validate_geometry(inputs)
    scales = _validate_scales(robust_scales, inputs)
    gradients = _input_gradients(forward, inputs)
    return AttributionMap(
        values=gradients * scales.reshape(1, 1, -1),
        method="signed_robust_scaled_gradient",
        validity=ValidityTag.PLAUSIBLY_LOCAL,
        signed=True,
        runtime_seconds=time.monotonic() - started,
        metadata={
            "estimand": "native max(log Q, -2)",
            "scale": "robust_per_channel",
            "map_units": "native_output_per_robust_channel_scale",
            "contribution_valued": False,
        },
    )


def integrated_gradients(
    forward: GeometryForward,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    *,
    steps: int = 32,
    backend: Literal["auto", "captum", "fallback"] = "auto",
) -> AttributionMap:
    started = time.monotonic()
    values, reference = _validated_inputs_and_baseline(inputs, baseline)
    if steps < 2:
        raise ValueError("integrated gradients requires at least two path points")
    selected_backend = _resolve_backend(backend)
    if selected_backend == "captum":
        from captum.attr import IntegratedGradients

        estimator = IntegratedGradients(forward)
        attribution = estimator.attribute(
            values,
            baselines=reference,
            n_steps=int(steps),
            method="gausslegendre",
        ).detach()
    else:
        difference = values - reference
        total = torch.zeros_like(values)
        for step, alpha in enumerate(torch.linspace(0, 1, steps, device=values.device)):
            gradient = _input_gradients(forward, reference + alpha * difference)
            weight = 0.5 if step in (0, steps - 1) else 1.0
            total += weight * gradient
        attribution = difference * total / (steps - 1)
    return AttributionMap(
        values=attribution,
        method=f"integrated_gradients_{selected_backend}",
        validity=ValidityTag.OFF_MANIFOLD,
        signed=True,
        runtime_seconds=time.monotonic() - started,
        metadata={
            "estimand": "native max(log Q, -2)",
            "path_points": int(steps),
            "baseline_validity": ValidityTag.OFF_MANIFOLD.value,
            "contribution_valued": True,
        },
    )


def expected_gradients(
    forward: GeometryForward,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    *,
    samples: int,
    seed: int,
    backend: Literal["auto", "captum", "fallback"] = "auto",
) -> AttributionMap:
    started = time.monotonic()
    _validate_geometry(inputs)
    if baselines.ndim != 3 or baselines.shape[1:] != inputs.shape[1:]:
        raise ValueError("baselines must have shape (background, z, channel)")
    if samples < 1 or len(baselines) < 1:
        raise ValueError("expected gradients requires samples and backgrounds")
    baselines = baselines.to(device=inputs.device, dtype=inputs.dtype)
    selected_backend = _resolve_backend(backend)
    if selected_backend == "captum":
        from captum.attr import GradientShap

        panel_size = len(inputs)

        def captum_forward(expanded: torch.Tensor) -> torch.Tensor:
            if len(expanded) % panel_size:
                raise ValueError("Captum batch is not a multiple of panel rows")
            repetitions = len(expanded) // panel_size
            if repetitions == 1:
                return _forward_vector(forward, expanded)
            step_major = (
                expanded.reshape(panel_size, repetitions, *expanded.shape[1:])
                .transpose(0, 1)
                .reshape_as(expanded)
            )
            output = _forward_vector(forward, step_major)
            return output.reshape(repetitions, panel_size).transpose(0, 1).reshape(-1)

        devices = [inputs.device.index] if inputs.is_cuda else []
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                # Captum 0.9 GradientShap draws path coefficients with NumPy's
                # global RNG, while NoiseTunnel draws with PyTorch. Seed both
                # and restore NumPy so this estimator is deterministic without
                # contaminating later benchmark randomness.
                np.random.seed(int(seed))
                attribution = GradientShap(captum_forward).attribute(
                    inputs, baselines=baselines, n_samples=int(samples), stdevs=0.0
                ).detach()
        finally:
            np.random.set_state(numpy_state)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        total = torch.zeros_like(inputs)
        for _ in range(samples):
            indices = torch.randint(
                0, len(baselines), (len(inputs),), generator=generator
            ).to(inputs.device)
            reference = baselines.index_select(0, indices)
            alpha = torch.rand((len(inputs), 1, 1), generator=generator).to(
                device=inputs.device, dtype=inputs.dtype
            )
            point = reference + alpha * (inputs - reference)
            total += (inputs - reference) * _input_gradients(forward, point)
        attribution = total / samples
    return AttributionMap(
        values=attribution,
        method=f"expected_gradients_{selected_backend}",
        validity=ValidityTag.OFF_MANIFOLD,
        signed=True,
        runtime_seconds=time.monotonic() - started,
        metadata={
            "estimand": "native max(log Q, -2)",
            "samples": int(samples),
            "seed": int(seed),
            "background_count": int(len(baselines)),
            "contribution_valued": True,
            "batch_layout_adapter": (
                "captum_sample_major_to_step_major"
                if selected_backend == "captum"
                else "none"
            ),
        },
    )


def vargrad(
    forward: GeometryForward,
    inputs: torch.Tensor,
    *,
    robust_scales: torch.Tensor,
    samples: int,
    noise_fraction: float,
    seed: int,
) -> AttributionMap:
    started = time.monotonic()
    _validate_geometry(inputs)
    scales = _validate_scales(robust_scales, inputs)
    if samples < 2 or noise_fraction <= 0:
        raise ValueError("VarGrad requires at least two positive-noise samples")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gradients = []
    for _ in range(samples):
        noise = torch.randn(inputs.shape, generator=generator, dtype=inputs.dtype).to(
            inputs.device
        )
        noisy = inputs + noise * scales.reshape(1, 1, -1) * float(noise_fraction)
        gradients.append(
            _input_gradients(forward, noisy) * scales.reshape(1, 1, -1)
        )
    attribution = torch.stack(gradients).var(dim=0, unbiased=False)
    return AttributionMap(
        values=attribution,
        method="vargrad_robust_noise",
        validity=ValidityTag.PLAUSIBLY_LOCAL,
        signed=False,
        runtime_seconds=time.monotonic() - started,
        metadata={
            "estimand": "native max(log Q, -2)",
            "samples": int(samples),
            "seed": int(seed),
            "noise_fraction_of_robust_scale": float(noise_fraction),
            "contribution_valued": False,
        },
    )


def completeness_residual(
    forward: GeometryForward,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    attribution: torch.Tensor,
) -> torch.Tensor:
    values, reference = _validated_inputs_and_baseline(inputs, baseline)
    if attribution.shape != values.shape:
        raise ValueError("attribution and input shapes must match")
    with torch.no_grad():
        difference = _forward_vector(forward, values) - _forward_vector(forward, reference)
    return attribution.sum(dim=(1, 2)) - difference


def attribution_equivariance_error(
    attributor: Callable[[torch.Tensor], AttributionMap],
    inputs: torch.Tensor,
    *,
    shift: int,
    shifted_attributor: Callable[[torch.Tensor], AttributionMap] | None = None,
) -> float:
    _validate_geometry(inputs)
    reference = attributor(inputs).values
    shifted = torch.roll(inputs, shifts=int(shift), dims=1)
    shifted_map = (shifted_attributor or attributor)(shifted).values
    expected = torch.roll(reference, shifts=int(shift), dims=1)
    difference_rms = torch.sqrt(torch.mean((shifted_map - expected).square()))
    scale = torch.sqrt(torch.mean(expected.square())).clamp_min(
        torch.finfo(expected.dtype).eps
    )
    return float((difference_rms / scale).detach().cpu())


def cyclic_grouped_occlusion(
    forward: GeometryForward,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    *,
    window: int,
    stride: int,
) -> AttributionMap:
    started = time.monotonic()
    values, reference = _validated_inputs_and_baseline(inputs, baseline)
    grid_size, channel_count = values.shape[1:]
    if not 1 <= window <= grid_size or stride < 1:
        raise ValueError("invalid cyclic occlusion window or stride")
    with torch.no_grad():
        original = _forward_vector(forward, values)
    attribution = torch.zeros_like(values)
    coverage = torch.zeros_like(values)
    for channel in range(channel_count):
        for start in range(0, grid_size, stride):
            positions = (torch.arange(window, device=values.device) + start).remainder(
                grid_size
            )
            perturbed = values.clone()
            perturbed[:, positions, channel] = reference[:, positions, channel]
            with torch.no_grad():
                effect = original - _forward_vector(forward, perturbed)
            share = effect.reshape(-1, 1) / window
            attribution[:, positions, channel] += share
            coverage[:, positions, channel] += 1
    attribution /= coverage.clamp_min(1)
    return AttributionMap(
        values=attribution,
        method=f"cyclic_grouped_occlusion_w{window}_s{stride}",
        validity=ValidityTag.PLAUSIBLY_LOCAL,
        signed=True,
        runtime_seconds=time.monotonic() - started,
        metadata={
            "estimand": "native max(log Q, -2)",
            "window": int(window),
            "stride": int(stride),
            "wraparound": True,
            "contribution_valued": True,
        },
    )


def temporal_saliency_rescale(attribution: AttributionMap) -> AttributionMap:
    started = time.monotonic()
    values = attribution.values
    _validate_geometry(values)
    magnitude = values.abs()
    channel = magnitude.mean(dim=1)
    position = magnitude.mean(dim=2)
    joint = position.unsqueeze(2) * channel.unsqueeze(1)
    source_total = magnitude.sum(dim=(1, 2), keepdim=True)
    joint_total = joint.sum(dim=(1, 2), keepdim=True).clamp_min(
        torch.finfo(values.dtype).eps
    )
    rescaled = torch.sign(values) * joint * source_total / joint_total
    rescaled = torch.where(source_total > 0, rescaled, torch.zeros_like(rescaled))
    return AttributionMap(
        values=rescaled,
        method=f"temporal_saliency_rescaled__{attribution.method}",
        validity=attribution.validity,
        signed=attribution.signed,
        runtime_seconds=attribution.runtime_seconds + time.monotonic() - started,
        metadata={
            **attribution.metadata,
            "channel_marginal": "mean_absolute_over_z",
            "position_marginal": "mean_absolute_over_channel",
            "absolute_mass_preserved": True,
        },
    )


def periodic_extremal_mask(
    forward: GeometryForward,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    *,
    area_fraction: float,
    steps: int,
    learning_rate: float,
    seed: int,
) -> AttributionMap:
    started = time.monotonic()
    del seed  # Kept in the uniform estimator API; this optimizer uses no RNG.
    values, reference = _validated_inputs_and_baseline(inputs, baseline)
    if not 0 < area_fraction <= 1 or steps < 1 or learning_rate <= 0:
        raise ValueError("invalid periodic extremal-mask configuration")
    # A contribution initialization makes the optimization stable on small
    # pilot cohorts while the learned mask remains the returned estimand.
    gradient = _input_gradients(forward, values)
    score = (gradient * (values - reference)).abs()
    maximum = score.amax(dim=(1, 2), keepdim=True).clamp_min(
        torch.finfo(values.dtype).eps
    )
    logits = (-6 + 6 * score / maximum).detach()
    logits.requires_grad_(True)
    optimizer = torch.optim.Adam((logits,), lr=float(learning_rate))
    with torch.no_grad():
        original = _forward_vector(forward, values)
    for _ in range(steps):
        optimizer.zero_grad()
        mask = torch.sigmoid(logits)
        deleted = values * (1 - mask) + reference * mask
        effect = (_forward_vector(forward, deleted) - original).abs().mean()
        area_loss = (mask.mean(dim=(1, 2)) - float(area_fraction)).square().mean()
        periodic_tv = (mask - torch.roll(mask, shifts=1, dims=1)).abs().mean()
        entropy = (mask * (1 - mask)).mean()
        loss = -effect + 100.0 * area_loss + 0.01 * periodic_tv + 0.001 * entropy
        loss.backward()
        optimizer.step()
    result = torch.sigmoid(logits.detach())
    return AttributionMap(
        values=result,
        method="periodic_extremal_mask",
        validity=ValidityTag.OFF_MANIFOLD,
        signed=False,
        runtime_seconds=time.monotonic() - started,
        metadata={
            "estimand": "native max(log Q, -2)",
            "area_fraction": float(area_fraction),
            "optimization_steps": int(steps),
            "seed_used": False,
            "deterministic_optimizer": True,
            "periodic_total_variation": True,
            "replacement_path": ValidityTag.OFF_MANIFOLD.value,
            "contribution_valued": False,
        },
    )


def deletion_insertion_curves(
    forward: GeometryForward,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    attribution: torch.Tensor,
    *,
    fractions: Sequence[float],
    robust_scales: torch.Tensor,
    seed: int,
    support_scorer: Callable[[np.ndarray], Mapping[str, np.ndarray]] | None = None,
    include_per_sample: bool = False,
) -> list[dict[str, Any]]:
    values, reference = _validated_inputs_and_baseline(inputs, baseline)
    if attribution.shape != values.shape:
        raise ValueError("attribution and inputs must have equal shape")
    scales = _validate_scales(robust_scales, values)
    fractions_array = np.asarray(fractions, dtype=np.float64)
    if (
        fractions_array.ndim != 1
        or len(fractions_array) < 2
        or np.any((fractions_array < 0) | (fractions_array > 1))
        or np.any(np.diff(fractions_array) < 0)
    ):
        raise ValueError("fractions must be a sorted sequence within [0, 1]")
    flat_count = values.shape[1] * values.shape[2]
    importance_order = attribution.abs().reshape(len(values), -1).argsort(
        dim=1, descending=True, stable=True
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_order = torch.stack(
        [torch.randperm(flat_count, generator=generator) for _ in range(len(values))]
    ).to(values.device)
    flat_values = values.reshape(len(values), flat_count)
    flat_reference = reference.reshape(len(values), flat_count)

    def edited(order: torch.Tensor, count: int, *, insert: bool) -> torch.Tensor:
        mask = torch.zeros_like(flat_values, dtype=torch.bool)
        if count:
            mask.scatter_(1, order[:, :count], True)
        selected = torch.where(mask, flat_values, flat_reference)
        output = selected if insert else torch.where(mask, flat_reference, flat_values)
        return output.reshape_as(values)

    def drift(left: torch.Tensor, right: torch.Tensor) -> float:
        standardized = (left - right) / scales.reshape(1, 1, -1)
        return float(torch.sqrt(standardized.square().mean()).detach().cpu())

    def support_warning(geometry: torch.Tensor) -> float:
        if support_scorer is None:
            return float("nan")
        score = support_scorer(geometry.detach().cpu().numpy())
        return float(np.mean(np.asarray(score["warning_score"], dtype=np.float64)))

    with torch.no_grad():
        original_values = _forward_vector(forward, values).detach().cpu().numpy()
        baseline_values = _forward_vector(forward, reference).detach().cpu().numpy()
        original_output = float(np.mean(original_values))
        baseline_output = float(np.mean(baseline_values))
    rows: list[dict[str, Any]] = []
    for fraction in fractions_array:
        count = min(flat_count, int(math.ceil(float(fraction) * flat_count)))
        deletion = edited(importance_order, count, insert=False)
        insertion = edited(importance_order, count, insert=True)
        random_deletion = edited(random_order, count, insert=False)
        random_insertion = edited(random_order, count, insert=True)
        with torch.no_grad():
            output_arrays = {
                "deletion_output": _forward_vector(forward, deletion)
                .detach()
                .cpu()
                .numpy(),
                "insertion_output": _forward_vector(forward, insertion)
                .detach()
                .cpu()
                .numpy(),
                "random_deletion_output": _forward_vector(forward, random_deletion)
                .detach()
                .cpu()
                .numpy(),
                "random_insertion_output": _forward_vector(forward, random_insertion)
                .detach()
                .cpu()
                .numpy(),
            }
            row: dict[str, Any] = {
                "fraction": float(fraction),
                "original_output": original_output,
                "baseline_output": baseline_output,
                **{
                    key: float(np.mean(array))
                    for key, array in output_arrays.items()
                },
                "deletion_support_drift_rms": drift(deletion, values),
                "insertion_support_drift_rms": drift(insertion, reference),
                "random_deletion_support_drift_rms": drift(random_deletion, values),
                "random_insertion_support_drift_rms": drift(
                    random_insertion, reference
                ),
                "deletion_support_warning": support_warning(deletion),
                "insertion_support_warning": support_warning(insertion),
                "random_deletion_support_warning": support_warning(random_deletion),
                "random_insertion_support_warning": support_warning(random_insertion),
                "validity_tag": ValidityTag.OFF_MANIFOLD.value,
            }
            if include_per_sample:
                row["original_output_per_sample"] = original_values
                row["baseline_output_per_sample"] = baseline_values
                row.update(
                    {
                        f"{key}_per_sample": array
                        for key, array in output_arrays.items()
                    }
                )
            rows.append(row)
    return rows


def toy_recovery(
    attribution: torch.Tensor,
    *,
    relevant_channels: Sequence[int],
    relevant_positions: Sequence[int],
) -> dict[str, float]:
    _validate_geometry(attribution)
    channel_count = attribution.shape[2]
    grid_size = attribution.shape[1]
    relevant_channel_set = {int(value) for value in relevant_channels}
    relevant_position_set = {int(value) % grid_size for value in relevant_positions}
    if not relevant_channel_set or not relevant_position_set:
        raise ValueError("toy recovery needs relevant channels and positions")
    if min(relevant_channel_set) < 0 or max(relevant_channel_set) >= channel_count:
        raise ValueError("relevant channel is outside attribution")
    channel_score = attribution.abs().mean(dim=(0, 1))
    top_channel = int(torch.argmax(channel_score))
    position_score = attribution.abs().mean(dim=(0, 2)).detach().cpu().numpy()
    order = np.argsort(-position_score, kind="stable")
    hits = 0
    precision_sum = 0.0
    for rank, position in enumerate(order, start=1):
        if int(position) in relevant_position_set:
            hits += 1
            precision_sum += hits / rank
    return {
        "channel_top1": float(top_channel in relevant_channel_set),
        "position_average_precision": precision_sum / len(relevant_position_set),
    }


def grouped_bootstrap_mean(
    values: np.ndarray,
    equilibrium_files: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> GroupedBootstrapResult:
    observations = np.asarray(values, dtype=np.float64)
    groups = np.asarray(equilibrium_files)
    if observations.ndim != 1 or groups.ndim != 1 or len(observations) != len(groups):
        raise ValueError("values and equilibrium_files must be equal-length vectors")
    if len(observations) < 1 or replicates < 1 or not np.isfinite(observations).all():
        raise ValueError("grouped bootstrap inputs must be finite and nonempty")
    unique_groups = np.unique(groups)
    group_rows = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(int(seed))
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        chosen = rng.integers(0, len(group_rows), size=len(group_rows))
        sampled_rows = np.concatenate([group_rows[index] for index in chosen])
        samples[replicate] = observations[sampled_rows].mean()
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return GroupedBootstrapResult(
        estimate=float(observations.mean()),
        lower=float(lower),
        upper=float(upper),
        samples=samples,
    )


def absolute_rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Spearman agreement between absolute attribution maps."""

    if left.shape != right.shape or left.numel() < 2:
        raise ValueError("rank-correlation maps must be equal and nontrivial")
    left_values = left.detach().abs().cpu().numpy().reshape(-1)
    right_values = right.detach().abs().cpu().numpy().reshape(-1)
    left_rank = _average_ranks(left_values)
    right_rank = _average_ranks(right_values)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return float(np.array_equal(left_values, right_values))
    return float(left_centered @ right_centered / denominator)


def perturbation_infidelity(
    forward: GeometryForward,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    attribution: torch.Tensor,
    *,
    trials: int,
    removal_fraction: float,
    seed: int,
) -> float:
    """Normalized mask infidelity for contribution-valued maps.

    Each trial removes a random set of cells along the supplied baseline path.
    The summed removed attribution is compared with the resulting native-output
    change, so this diagnostic has the same units for every method.
    """

    values, reference = _validated_inputs_and_baseline(inputs, baseline)
    if attribution.shape != values.shape or trials < 1:
        raise ValueError("invalid infidelity attribution or trial count")
    if not 0 < removal_fraction <= 1:
        raise ValueError("removal_fraction must be in (0, 1]")
    feature_count = values.shape[1] * values.shape[2]
    remove_count = max(1, int(math.ceil(feature_count * removal_fraction)))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        original = _forward_vector(forward, values)
    squared_error = []
    squared_effect = []
    for _ in range(trials):
        mask = torch.zeros((len(values), feature_count), dtype=torch.bool)
        for sample in range(len(values)):
            selected = torch.randperm(feature_count, generator=generator)[:remove_count]
            mask[sample, selected] = True
        mask = mask.to(values.device).reshape_as(values)
        perturbed = torch.where(mask, reference, values)
        with torch.no_grad():
            effect = original - _forward_vector(forward, perturbed)
        predicted = torch.where(mask, attribution, torch.zeros_like(attribution)).sum(
            dim=(1, 2)
        )
        squared_error.append((predicted - effect).square())
        squared_effect.append(effect.square())
    numerator = torch.cat(squared_error).mean()
    denominator = torch.cat(squared_effect).mean().clamp_min(
        torch.finfo(values.dtype).eps
    )
    return float((numerator / denominator).cpu())


def attribution_sensitivity(
    attributor: Callable[[torch.Tensor], AttributionMap],
    inputs: torch.Tensor,
    *,
    robust_scales: torch.Tensor,
    trials: int,
    noise_fraction: float,
    seed: int,
) -> float:
    """Relative RMS map change under robust-scaled local input noise."""

    _validate_geometry(inputs)
    scales = _validate_scales(robust_scales, inputs)
    if trials < 1 or noise_fraction <= 0:
        raise ValueError("sensitivity requires positive trials and noise")
    reference = attributor(inputs).values
    denominator = torch.sqrt(reference.square().mean()).clamp_min(
        torch.finfo(reference.dtype).eps
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    changes = []
    for _ in range(trials):
        noise = torch.randn(inputs.shape, generator=generator, dtype=inputs.dtype).to(
            inputs.device
        )
        perturbed = inputs + noise * scales.reshape(1, 1, -1) * noise_fraction
        changed = attributor(perturbed).values
        changes.append(torch.sqrt((changed - reference).square().mean()) / denominator)
    return float(torch.stack(changes).mean().detach().cpu())


def curve_area(rows: Sequence[dict[str, float | str]], key: str) -> float:
    """Trapezoidal area under a dose curve after native-output normalization."""

    if len(rows) < 2:
        raise ValueError("curve area needs at least two rows")
    fraction = np.asarray([float(row["fraction"]) for row in rows])
    values = np.asarray([float(row[key]) for row in rows])
    baseline = float(rows[0]["baseline_output"])
    original = float(rows[0]["original_output"])
    scale = original - baseline
    if abs(scale) <= np.finfo(np.float64).eps:
        return float("nan")
    normalized = (values - baseline) / scale
    segment_area = 0.5 * (normalized[:-1] + normalized[1:]) * np.diff(fraction)
    return float(np.sum(segment_area))


def attribution_sparsity(attribution: torch.Tensor, *, mass_fraction: float = 0.9) -> float:
    """Median fraction of cells needed to carry a requested absolute mass."""

    _validate_geometry(attribution)
    if not 0 < mass_fraction <= 1:
        raise ValueError("mass_fraction must be in (0, 1]")
    flattened = attribution.detach().abs().reshape(len(attribution), -1)
    sorted_values = torch.sort(flattened, dim=1, descending=True).values
    totals = sorted_values.sum(dim=1, keepdim=True)
    cumulative = sorted_values.cumsum(dim=1)
    threshold = mass_fraction * totals
    counts = (cumulative < threshold).sum(dim=1) + 1
    counts = torch.where(totals[:, 0] > 0, counts, torch.full_like(counts, len(sorted_values[0])))
    return float((counts.float() / sorted_values.shape[1]).median().cpu())


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _validate_geometry(values: torch.Tensor) -> None:
    if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("geometry must have shape (sample, z, channel)")
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("geometry must be finite floating point")


def _validate_scales(scales: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(scales, dtype=inputs.dtype, device=inputs.device)
    if result.shape != (inputs.shape[2],) or not torch.isfinite(result).all():
        raise ValueError("robust scales must be a finite channel vector")
    if torch.any(result <= 0):
        raise ValueError("robust scales must be positive")
    return result


def _validated_inputs_and_baseline(
    inputs: torch.Tensor, baseline: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_geometry(inputs)
    reference = torch.as_tensor(baseline, dtype=inputs.dtype, device=inputs.device)
    if reference.ndim != 3 or reference.shape[1:] != inputs.shape[1:]:
        raise ValueError("baseline must have shape (1|sample, z, channel)")
    if len(reference) == 1:
        reference = reference.expand_as(inputs)
    if reference.shape != inputs.shape or not torch.isfinite(reference).all():
        raise ValueError("baseline must broadcast exactly across samples")
    return inputs, reference


def _forward_vector(forward: GeometryForward, values: torch.Tensor) -> torch.Tensor:
    output = forward(values)
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if output.shape != (len(values),):
        raise ValueError("forward must return one scalar per sample")
    return output


def _input_gradients(forward: GeometryForward, values: torch.Tensor) -> torch.Tensor:
    differentiable = values.detach().clone().requires_grad_(True)
    output = _forward_vector(forward, differentiable)
    gradient = torch.autograd.grad(output.sum(), differentiable)[0]
    return gradient.detach()


def _resolve_backend(
    backend: Literal["auto", "captum", "fallback"],
) -> Literal["captum", "fallback"]:
    if backend == "fallback":
        return "fallback"
    try:
        import captum.attr  # noqa: F401
    except ImportError:
        if backend == "captum":
            raise
        return "fallback"
    return "captum"
