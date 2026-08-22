"""Scaling, consensus, and uncertainty helpers for S06b attribution maps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class SignedConsensus:
    member_signed: np.ndarray
    median_signed: np.ndarray
    q25_signed: np.ndarray
    q75_signed: np.ndarray
    median_absolute: np.ndarray
    sign_agreement: np.ndarray
    member_channel_importance: np.ndarray
    pairwise_rank_correlations: np.ndarray
    median_rank_agreement: float


@dataclass(frozen=True)
class HierarchicalBootstrap:
    estimate: float
    lower: float
    upper: float
    samples: np.ndarray
    resampling_units: tuple[str, str] = ("members", "equilibrium_files")


@dataclass(frozen=True)
class ScalarSensitivity:
    values: np.ndarray
    signed: bool = True
    estimand: str = "native max(log Q, -2)"
    scale: str = "robust_per_scalar_drive"


@dataclass(frozen=True)
class ValidationStabilityCorrelation:
    stability: np.ndarray
    spearman_rho: float
    metric: str = "spearman_validation_r2_vs_median_map_rank_agreement"


def independent_sign_agreement_null(member_count: int) -> float:
    """Expected majority agreement when member signs are independent fair coins."""

    if member_count < 1:
        raise ValueError("member_count must be positive")
    return float(
        sum(
            max(positive, member_count - positive)
            * math.comb(member_count, positive)
            for positive in range(member_count + 1)
        )
        / (member_count * 2**member_count)
    )


def signed_consensus(maps: np.ndarray) -> SignedConsensus:
    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 4 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("maps must be finite (member, sample, channel, z) values")

    # Preserve each member's signed mean before any ensemble aggregation.  This
    # makes an even split between positive and negative mechanisms visible as a
    # zero median with 0.5 sign agreement, rather than as a falsely weak map.
    member_signed = values.mean(axis=1)
    median_signed = np.median(member_signed, axis=0)
    q25_signed, q75_signed = np.quantile(member_signed, (0.25, 0.75), axis=0)
    member_absolute = np.abs(values).mean(axis=1)
    median_absolute = np.median(member_absolute, axis=0)
    positive = np.mean(member_signed > 0, axis=0)
    negative = np.mean(member_signed < 0, axis=0)
    sign_agreement = np.maximum(positive, negative)
    sign_agreement = np.where(np.all(member_signed == 0, axis=0), 1.0, sign_agreement)

    member_channel_importance = np.abs(values).mean(axis=(1, 3))
    correlations = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            correlations.append(
                _spearman(member_channel_importance[left], member_channel_importance[right])
            )
    pairwise = np.asarray(correlations, dtype=np.float64)
    median_rank = float(np.median(pairwise)) if len(pairwise) else 1.0
    return SignedConsensus(
        member_signed=member_signed,
        median_signed=median_signed,
        q25_signed=q25_signed,
        q75_signed=q75_signed,
        median_absolute=median_absolute,
        sign_agreement=sign_agreement,
        member_channel_importance=member_channel_importance,
        pairwise_rank_correlations=pairwise,
        median_rank_agreement=median_rank,
    )


def hierarchical_group_bootstrap(
    values: np.ndarray,
    equilibrium_files: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> HierarchicalBootstrap:
    observations = np.asarray(values, dtype=np.float64)
    groups = np.asarray(equilibrium_files)
    if observations.ndim != 2 or groups.ndim != 1:
        raise ValueError("values must be (member, sample) and groups must be a vector")
    if observations.shape[1] != len(groups) or not np.isfinite(observations).all():
        raise ValueError("sample and group axes must agree and values must be finite")
    if observations.shape[0] < 1 or len(groups) < 1 or replicates < 1:
        raise ValueError("bootstrap inputs and replicate count must be nonempty")

    unique_groups = np.unique(groups)
    group_rows = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        member_draw = rng.integers(0, observations.shape[0], size=observations.shape[0])
        group_draw = rng.integers(0, len(group_rows), size=len(group_rows))
        row_draw = np.concatenate([group_rows[index] for index in group_draw])
        samples[replicate] = observations[np.ix_(member_draw, row_draw)].mean()
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return HierarchicalBootstrap(
        estimate=float(observations.mean()),
        lower=float(lower),
        upper=float(upper),
        samples=samples,
    )


def native_scalar_sensitivities(
    forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    geometry: torch.Tensor,
    a_over_lt: torch.Tensor,
    a_over_ln: torch.Tensor,
    *,
    robust_scales: np.ndarray,
) -> ScalarSensitivity:
    if geometry.ndim != 3 or len(geometry) < 1:
        raise ValueError("geometry must have shape (sample, z, channel)")
    if a_over_lt.ndim != 1 or a_over_ln.ndim != 1:
        raise ValueError("scalar drives must be vectors")
    if len(a_over_lt) != len(geometry) or len(a_over_ln) != len(geometry):
        raise ValueError("geometry and drive sample axes must agree")
    scales = np.asarray(robust_scales, dtype=np.float64)
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("robust_scales must contain two finite positive values")

    drive_lt = a_over_lt.detach().clone().requires_grad_(True)
    drive_ln = a_over_ln.detach().clone().requires_grad_(True)
    output = forward(geometry, drive_lt, drive_ln).reshape(-1)
    if len(output) != len(geometry):
        raise ValueError("forward must return one native output per sample")
    gradient_lt, gradient_ln = torch.autograd.grad(
        output.sum(), (drive_lt, drive_ln), create_graph=False
    )
    gradients = torch.stack((gradient_lt, gradient_ln), dim=1).detach().cpu().numpy()
    return ScalarSensitivity(values=gradients * scales.reshape(1, 2))


def build_stratification_masks(
    *,
    gradient_set: np.ndarray,
    target: np.ndarray,
    a_over_lt: np.ndarray,
    a_over_ln: np.ndarray,
    equilibrium_class: np.ndarray,
    stable_threshold: float,
    member_absolute_error: np.ndarray | None = None,
    ensemble_spread: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    gradient = np.asarray(gradient_set).astype(str)
    targets = np.asarray(target, dtype=np.float64)
    drive_lt = np.asarray(a_over_lt, dtype=np.float64)
    drive_ln = np.asarray(a_over_ln, dtype=np.float64)
    classes = np.asarray(equilibrium_class)
    count = len(gradient)
    if any(len(array) != count for array in (targets, drive_lt, drive_ln, classes)):
        raise ValueError("all stratification arrays must share a sample axis")
    if set(np.unique(gradient)) - {"fixed", "varied"}:
        raise ValueError("gradient_set must contain only fixed or varied")
    if not all(np.isfinite(array).all() for array in (targets, drive_lt, drive_ln)):
        raise ValueError("numeric stratification arrays must be finite")

    optional_fields: dict[str, np.ndarray] = {}
    for field, optional in (
        ("member_absolute_error", member_absolute_error),
        ("ensemble_spread", ensemble_spread),
    ):
        if optional is None:
            continue
        values = np.asarray(optional, dtype=np.float64)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"{field} must be a finite sample vector")
        optional_fields[field] = values

    masks: dict[str, np.ndarray] = {}
    for name in ("varied", "fixed"):
        base = gradient == name
        unstable = base & (targets > float(stable_threshold))
        masks[f"gradient_set={name}|all"] = base
        masks[f"gradient_set={name}|stability=stable_or_near_floor"] = base & (
            targets <= float(stable_threshold)
        )
        masks[f"gradient_set={name}|stability=unstable"] = unstable
        if np.count_nonzero(unstable) >= 3 and np.ptp(targets[unstable]) > 0:
            cut1, cut2 = np.quantile(targets[unstable], (1 / 3, 2 / 3))
            for label, condition in (
                ("low_unstable", targets <= cut1),
                ("medium_unstable", (targets > cut1) & (targets <= cut2)),
                ("high_unstable", targets > cut2),
            ):
                masks[f"gradient_set={name}|flux={label}"] = unstable & condition
        for field, values in (("a_over_lt", drive_lt), ("a_over_ln", drive_ln)):
            if np.count_nonzero(base) >= 3 and np.ptp(values[base]) > 0:
                # Keep the registered full-panel tertile boundaries, then
                # exclude floor rows from every feature-level summary.
                cut1, cut2 = np.quantile(values[base], (1 / 3, 2 / 3))
                bins = (
                    ("low", values <= cut1),
                    ("medium", (values > cut1) & (values <= cut2)),
                    ("high", values > cut2),
                )
                for label, condition in bins:
                    masks[f"gradient_set={name}|{field}={label}"] = unstable & condition
        for class_value in np.unique(classes[unstable]):
            masks[f"gradient_set={name}|equilibrium_class={class_value}"] = unstable & (
                classes == class_value
            )
        for field, values in optional_fields.items():
            if np.count_nonzero(base) < 3 or np.ptp(values[base]) == 0:
                continue
            cut1, cut2 = np.quantile(values[base], (1 / 3, 2 / 3))
            for label, condition in (
                ("low", values <= cut1),
                ("medium", (values > cut1) & (values <= cut2)),
                ("high", values > cut2),
            ):
                masks[f"gradient_set={name}|{field}={label}"] = unstable & condition
    return masks


def validation_stability_correlation(
    member_channel_importance: np.ndarray, validation_r2: np.ndarray
) -> ValidationStabilityCorrelation:
    importance = np.asarray(member_channel_importance, dtype=np.float64)
    scores = np.asarray(validation_r2, dtype=np.float64)
    if importance.ndim != 2 or scores.shape != (importance.shape[0],):
        raise ValueError("importance must be (member, feature) with one score per member")
    if min(importance.shape) < 2 or not np.isfinite(importance).all() or not np.isfinite(scores).all():
        raise ValueError("rank correlation inputs must be finite and nontrivial")
    consensus = np.median(importance, axis=0)
    stability = np.asarray([_spearman(row, consensus) for row in importance])
    return ValidationStabilityCorrelation(
        stability=stability,
        spearman_rho=_spearman(scores, stability),
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.asarray(left).shape != np.asarray(right).shape or np.asarray(left).ndim != 1:
        raise ValueError("Spearman inputs must be equal vectors")
    left_rank = _average_ranks(np.asarray(left))
    right_rank = _average_ranks(np.asarray(right))
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    if denominator == 0:
        return float(np.array_equal(left, right))
    return float(left_rank @ right_rank / denominator)
