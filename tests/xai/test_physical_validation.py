from __future__ import annotations

import numpy as np

from itg_nn.xai.physical_validation import (
    cross_fitted_aipw,
    equilibrium_grouped_matches,
    grouped_bootstrap_interval,
    grouped_ridge_crossfit,
    residual_rank_association,
)


def _cyclic_toy(rows: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic cyclic signal: the mean and peak are invariant to phase."""

    rng = np.random.default_rng(14)
    z = 2 * np.pi * np.arange(96) / 96
    phase = rng.uniform(0, 2 * np.pi, rows)
    mean_signal = rng.uniform(0.2, 1.2, rows)
    peak_signal = rng.uniform(0.0, 1.0, rows)
    profile = (
        mean_signal[:, None]
        + peak_signal[:, None] * np.maximum(np.cos(z[None, :] + phase[:, None]), 0)
    )
    null = rng.normal(size=rows)
    return profile, peak_signal, null


def test_grouped_residual_crossfit_recovers_cyclic_candidate_and_null_control() -> None:
    profile, peak_signal, null = _cyclic_toy()
    shifted = np.roll(profile, 23, axis=1)
    mean_feature = profile.mean(axis=1)
    peak_feature = profile.max(axis=1) - profile.mean(axis=1)
    np.testing.assert_allclose(mean_feature, shifted.mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(profile.max(axis=1), shifted.max(axis=1), atol=1e-12)

    native = np.maximum(-2.0, -1.4 + 0.7 * mean_feature + 1.3 * peak_signal)
    groups = np.repeat(np.arange(len(native) // 2), 2)
    baseline = grouped_ridge_crossfit(
        mean_feature[:, None], native, groups, folds=5, seed=9, alpha=1e-8
    )
    candidate = grouped_ridge_crossfit(
        np.column_stack((mean_feature, peak_feature)),
        native,
        groups,
        folds=5,
        seed=9,
        alpha=1e-8,
    )
    negative_control = grouped_ridge_crossfit(
        np.column_stack((mean_feature, null)),
        native,
        groups,
        folds=5,
        seed=9,
        alpha=1e-8,
    )
    assert candidate.r2 > 0.999
    assert candidate.r2 - baseline.r2 > 0.2
    assert negative_control.r2 - baseline.r2 < 0.02
    assert candidate.estimand == "native max(log Q, -2)"
    assert not np.allclose(candidate.prediction, np.exp(native))
    for fold in np.unique(candidate.fold):
        held_out = groups[candidate.fold == fold]
        training = groups[candidate.fold != fold]
        assert set(held_out).isdisjoint(training)


def test_matching_is_equilibrium_disjoint_and_improves_nuisance_balance() -> None:
    rng = np.random.default_rng(4)
    nuisance = rng.normal(size=(80, 2))
    exposure = nuisance[:, 0] + rng.normal(scale=0.7, size=80)
    groups = np.asarray([f"eq-{index}" for index in range(80)])
    result = equilibrium_grouped_matches(
        exposure,
        nuisance,
        groups,
        high_quantile=0.70,
        low_quantile=0.30,
        caliper=1.10,
    )
    assert len(result.high_positions) >= 10
    assert len(np.unique(np.r_[result.high_positions, result.low_positions])) == 2 * len(
        result.high_positions
    )
    assert set(groups[result.high_positions]).isdisjoint(groups[result.low_positions])
    assert np.all(result.exposure_contrast > 0)
    assert np.nanmax(np.abs(result.nuisance_smd_after)) < np.nanmax(
        np.abs(result.nuisance_smd_before)
    )
    assert result.split_unit == "equilibrium_files"
    assert result.validity_tag == "observed-comparison"

    rescaled_nuisance = nuisance.copy()
    rescaled_nuisance[:, 1] *= 1000
    rescaled = equilibrium_grouped_matches(
        exposure,
        rescaled_nuisance,
        groups,
        high_quantile=0.70,
        low_quantile=0.30,
        caliper=1.10,
    )
    np.testing.assert_array_equal(result.high_positions, rescaled.high_positions)
    np.testing.assert_array_equal(result.low_positions, rescaled.low_positions)
    np.testing.assert_allclose(result.distance, rescaled.distance)


def test_aipw_recovers_adjusted_observed_contrast_and_is_deterministic() -> None:
    rng = np.random.default_rng(18)
    nuisance = rng.normal(size=(500, 3))
    probability = 1.0 / (1.0 + np.exp(-0.8 * nuisance[:, 0] + 0.4 * nuisance[:, 1]))
    treated = rng.binomial(1, probability).astype(bool)
    outcome = 1.25 * treated + 2.0 * nuisance[:, 0] - nuisance[:, 1]
    groups = np.repeat(np.arange(250), 2)
    first = cross_fitted_aipw(
        treated, outcome, nuisance, groups, folds=5, seed=31
    )
    second = cross_fitted_aipw(
        treated, outcome, nuisance, groups, folds=5, seed=31
    )
    assert first.estimate == second.estimate
    np.testing.assert_array_equal(first.fold, second.fold)
    assert abs(first.estimate - 1.25) < 0.15
    assert first.overlap_fraction > 0.9
    assert first.method == "in_repo_logistic_irls_plus_ridge"
    for fold in np.unique(first.fold):
        assert set(groups[first.fold == fold]).isdisjoint(groups[first.fold != fold])


def test_grouped_bootstrap_resamples_complete_equilibria() -> None:
    values = np.asarray([1.0, 3.0, 2.0, 4.0, 7.0, 9.0])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    first = grouped_bootstrap_interval(values, groups, replicates=100, seed=12)
    second = grouped_bootstrap_interval(values, groups, replicates=100, seed=12)
    assert first == second
    assert first[1] <= first[0] <= first[2]
    assert first[0] == np.mean(values)


def test_residual_rank_association_retains_sign_and_null() -> None:
    rng = np.random.default_rng(22)
    candidate = np.linspace(-1.0, 1.0, 80)
    residual = -2.0 * candidate + 0.01 * rng.normal(size=80)
    groups = np.repeat(np.arange(40), 2)
    estimate, lower, upper = residual_rank_association(
        candidate, residual, groups, replicates=200, seed=5
    )
    assert estimate < -0.99
    assert upper < 0
    null_estimate, null_lower, null_upper = residual_rank_association(
        rng.permutation(candidate), residual, groups, replicates=200, seed=5
    )
    assert abs(null_estimate) < 0.2
    assert null_lower < 0 < null_upper
