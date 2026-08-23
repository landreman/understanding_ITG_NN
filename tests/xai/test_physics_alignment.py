from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from itg_nn.xai.physics_alignment import (
    _average_ranks,
    _grouped_row_draws,
    _row_correlation,
    _spearman,
    circular_alignment,
    lag_selection_permutation_null,
    paired_native_difference,
    scalar_rank_association,
    select_balanced_case_studies,
)
from scripts.xai_s07_physics_alignment import (
    _alignment_row,
    _association_bootstrap_stable,
    _lag_within_tolerance_recurrence,
    _pair_strata,
    _position_source,
    _select_alignment_case_studies,
    _signed_lag,
    _strata,
    _top_mass_fraction,
    _zonal_summary_associations,
)


def _cyclic_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(701)
    physical = rng.normal(size=(24, 96))
    learned = np.roll(physical, 7, axis=1)
    groups = np.repeat(np.asarray([f"eq{index}" for index in range(12)]), 2)
    return learned, physical, groups


def test_circular_alignment_recovers_known_lag_overlap_and_null_control() -> None:
    learned, physical, groups = _cyclic_fixture()
    result = circular_alignment(
        learned,
        physical,
        groups,
        mode="signed",
        sparsity=0.1,
        bootstrap_replicates=80,
        seed=19,
    )

    assert result.best_lag == 7
    assert result.rank_correlation == pytest.approx(1.0, abs=1e-12)
    assert result.rank_ci_lower <= result.rank_correlation <= result.rank_ci_upper
    assert result.per_sample_rank_correlation.mean() == pytest.approx(
        result.rank_correlation, abs=1e-12
    )
    assert result.cross_correlation_by_lag[7] == pytest.approx(1.0, abs=1e-12)
    assert result.overlap == pytest.approx(1.0)
    assert result.lag_recurrence == pytest.approx(1.0)
    assert result.bootstrap_group == "equilibrium_files"

    nonlinear = circular_alignment(
        np.roll(physical**3, 7, axis=1),
        physical,
        groups,
        mode="signed",
        sparsity=0.1,
        bootstrap_replicates=20,
        seed=20,
    )
    expected_cross = np.mean(
        [
            np.corrcoef(np.roll(physical[index] ** 3, 7), np.roll(physical[index], 7))[
                0, 1
            ]
            for index in range(len(physical))
        ]
    )
    assert nonlinear.rank_correlation_by_lag[7] == pytest.approx(1.0, abs=1e-12)
    assert nonlinear.cross_correlation_by_lag[7] == pytest.approx(expected_cross)
    assert nonlinear.cross_correlation_by_lag[7] < 0.9

    rng = np.random.default_rng(702)
    null = rng.normal(size=learned.shape)
    control = circular_alignment(
        null,
        physical,
        groups,
        mode="signed",
        sparsity=0.1,
        bootstrap_replicates=80,
        seed=19,
    )
    assert abs(control.rank_correlation) < 0.12
    assert control.overlap_enrichment < 1.5
    assert control.rank_ci_lower == pytest.approx(
        np.quantile(control.bootstrap_rank_correlation, 0.025)
    )
    assert control.rank_ci_upper == pytest.approx(
        np.quantile(control.bootstrap_rank_correlation, 0.975)
    )
    assert len(np.unique(control.bootstrap_best_lag)) > 1
    assert 0 < control.lag_recurrence < 1


def test_alignment_is_joint_shift_invariant_and_positive_mode_clips_signs() -> None:
    learned, physical, groups = _cyclic_fixture()
    signed = circular_alignment(
        -learned,
        physical,
        groups,
        mode="signed",
        sparsity=0.1,
        bootstrap_replicates=40,
        seed=33,
    )
    positive = circular_alignment(
        -learned,
        physical,
        groups,
        mode="positive_contribution",
        sparsity=0.1,
        bootstrap_replicates=40,
        seed=33,
    )
    shifted = circular_alignment(
        np.roll(-learned, 19, axis=1),
        np.roll(physical, 19, axis=1),
        groups,
        mode="signed",
        sparsity=0.1,
        bootstrap_replicates=40,
        seed=33,
    )

    assert signed.best_lag == shifted.best_lag == 7
    assert signed.rank_correlation == pytest.approx(-1.0, abs=1e-12)
    assert signed.overlap == pytest.approx(1.0)
    assert signed.overlap_orientation == (
        "gx_profile_sign_flipped_to_match_negative_association"
    )
    assert shifted.rank_correlation == pytest.approx(signed.rank_correlation)
    np.testing.assert_allclose(
        shifted.rank_correlation_by_lag, signed.rank_correlation_by_lag
    )
    assert positive.mode == "positive_contribution"
    assert positive.rank_correlation > -0.9
    assert positive.overlap_orientation == "gx_profile_unflipped"
    assert positive.overlap < 0.1


def test_tie_inclusive_overlap_and_lag_selection_null_are_pinned() -> None:
    tied = np.zeros((12, 96), dtype=np.float64)
    groups = np.asarray([f"eq{index}" for index in range(len(tied))])
    tied_result = circular_alignment(
        tied,
        tied,
        groups,
        mode="signed",
        sparsity=0.1,
        bootstrap_replicates=20,
        seed=44,
    )
    assert tied_result.overlap == pytest.approx(1.0)
    assert tied_result.overlap_chance == pytest.approx(1.0)
    assert tied_result.overlap_enrichment == pytest.approx(1.0)
    assert tied_result.learned_constant_profile_count == 12
    assert tied_result.learned_active_profile_count == 0
    assert tied_result.learned_constant_profile_fraction == pytest.approx(1.0)
    assert tied_result.learned_mask_width_mean == pytest.approx(96.0)
    assert tied_result.gx_mask_width_mean == pytest.approx(96.0)

    mixed_learned, mixed_physical, mixed_groups = _cyclic_fixture()
    mixed_learned[::2] = 0.0
    mixed_result = circular_alignment(
        mixed_learned,
        mixed_physical,
        mixed_groups,
        mode="signed",
        sparsity=0.1,
        bootstrap_replicates=20,
        seed=44,
    )
    assert mixed_result.learned_constant_profile_count == 12
    assert mixed_result.learned_active_profile_count == 12
    assert mixed_result.rank_correlation == pytest.approx(0.5, abs=1e-12)
    assert mixed_result.circular_spearman_active_learned_profiles == pytest.approx(
        1.0, abs=1e-12
    )
    assert mixed_result.learned_mask_width_mean == pytest.approx(53.0)
    assert mixed_result.gx_mask_width_mean == pytest.approx(10.0)
    assert mixed_result.overlap_enrichment == pytest.approx(
        1.8113207547169812, abs=1e-12
    )
    np.testing.assert_allclose(
        mixed_result.per_sample_overlap, np.ones(len(mixed_learned)), atol=0.0
    )

    learned, physical, paired_groups = _cyclic_fixture()
    signal_null = lag_selection_permutation_null(
        learned,
        physical,
        paired_groups,
        mode="signed",
        permutations=40,
        seed=45,
    )
    random_null = lag_selection_permutation_null(
        np.random.default_rng(46).normal(size=learned.shape),
        physical,
        paired_groups,
        mode="signed",
        permutations=40,
        seed=45,
    )
    assert signal_null.q95 < 0.55
    assert random_null.q95 < 0.2
    assert signal_null.q95 == pytest.approx(
        np.quantile(signal_null.max_abs_rank_correlation, 0.95)
    )
    assert signal_null.maximum == pytest.approx(
        signal_null.max_abs_rank_correlation.max()
    )
    assert signal_null.maximum > signal_null.max_abs_rank_correlation.mean()
    assert signal_null.permutation_group == "equilibrium_files"
    np.testing.assert_array_equal(
        signal_null.max_abs_rank_correlation,
        lag_selection_permutation_null(
            learned,
            physical,
            paired_groups,
            mode="signed",
            permutations=40,
            seed=45,
        ).max_abs_rank_correlation,
    )

    # Independent slow oracle: repeat the same group permutations, but let the
    # observed-statistic implementation search all 96 lags explicitly. This
    # fails if the fast null silently evaluates lag zero only.
    group_rows = tuple(
        np.flatnonzero(paired_groups == group) for group in np.unique(paired_groups)
    )
    rng = np.random.default_rng(45)
    direct_maxima = []
    for _ in range(40):
        group_permutation = rng.permutation(len(group_rows))
        pairing = np.empty(len(paired_groups), dtype=np.int64)
        for left_group, right_group in enumerate(group_permutation):
            pairing[group_rows[left_group]] = group_rows[right_group]
        direct = circular_alignment(
            learned,
            physical[pairing],
            paired_groups,
            mode="signed",
            sparsity=0.1,
            bootstrap_replicates=2,
            seed=1,
        )
        direct_maxima.append(abs(direct.rank_correlation))
    np.testing.assert_allclose(
        signal_null.max_abs_rank_correlation,
        np.asarray(direct_maxima),
        atol=1e-12,
    )


def test_average_ranks_pin_midrank_ties_and_constant_row_convention() -> None:
    # In each four-position block, the long low plateau in `left` has a
    # mid-rank correlation of exactly zero with `right`. Stable ordinal ranks
    # (which break ties by position) instead give a large nonzero value.
    left = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), 24)
    right = np.tile(np.asarray([0.0, 1.0, 2.0, 1.0]), 24)
    tied = _row_correlation(
        _average_ranks(left)[None, :], _average_ranks(right)[None, :]
    )
    assert tied[0] == pytest.approx(0.0, abs=1e-12)

    constant = np.zeros(96, dtype=np.float64)
    constant_result = _row_correlation(
        _average_ranks(constant)[None, :], _average_ranks(right)[None, :]
    )
    assert constant_result[0] == pytest.approx(0.0, abs=1e-12)


def test_signed_position_marginal_and_stability_strata_are_pinned() -> None:
    channels = np.asarray([[[1.0, -2.0, 3.0], [-1.0, 1.0, -4.0]]])
    np.testing.assert_array_equal(
        _position_source(channels, "ig_low_pass", "signed"),
        np.asarray([[0.0, -1.0, -1.0]]),
    )
    np.testing.assert_array_equal(
        _position_source(channels, "ig_low_pass", "positive_contribution"),
        np.asarray([[1.0, 1.0, 3.0]]),
    )
    with pytest.raises(ValueError, match="magnitude-only"):
        _position_source(channels, "periodic_mask", "signed")

    strata = _strata(np.asarray([-2.0, -1.9, -1.0, 0.0]), threshold=-1.9)
    assert tuple(label for label, _ in strata) == (
        "all",
        "stable_or_near_floor",
        "unstable",
    )
    np.testing.assert_array_equal(strata[0][1], np.asarray([True, True, True, True]))
    np.testing.assert_array_equal(strata[1][1], np.asarray([True, True, False, False]))
    np.testing.assert_array_equal(strata[2][1], np.asarray([False, False, True, True]))

    pair_strata = _pair_strata(
        np.asarray([-2.0, -1.0, -2.0, -1.0]),
        np.asarray([-1.0, -2.0, -2.0, -1.0]),
        threshold=-1.9,
    )
    assert tuple(label for label, _ in pair_strata) == (
        "all",
        "either_stable_or_near_floor",
        "both_unstable",
    )
    np.testing.assert_array_equal(
        pair_strata[1][1], np.asarray([True, True, True, False])
    )
    np.testing.assert_array_equal(
        pair_strata[2][1], np.asarray([False, False, False, True])
    )

    assert _signed_lag(0) == 0
    assert _signed_lag(48) == 48
    assert _signed_lag(60) == -36
    assert _signed_lag(95) == -1

    mass = np.asarray([[[1.0] * 9 + [9.0]]])
    np.testing.assert_allclose(_top_mass_fraction(mass), np.asarray([0.5]))


def test_grouped_bootstraps_and_case_selection_are_deterministic() -> None:
    learned, physical, groups = _cyclic_fixture()
    first = circular_alignment(
        learned,
        physical,
        groups,
        mode="signed",
        sparsity=0.1,
        bootstrap_replicates=50,
        seed=81,
    )
    second = circular_alignment(
        learned,
        physical,
        groups,
        mode="signed",
        sparsity=0.1,
        bootstrap_replicates=50,
        seed=81,
    )
    np.testing.assert_array_equal(
        first.bootstrap_rank_correlation, second.bootstrap_rank_correlation
    )
    np.testing.assert_array_equal(first.bootstrap_best_lag, second.bootstrap_best_lag)

    scores = np.asarray([0.9, 0.8, -0.9, -0.8, 0.7, -0.7, 0.6, -0.6])
    row_ids = np.arange(100, 108)
    case_groups = np.asarray(["a", "a", "b", "c", "d", "e", "f", "g"])
    cases = select_balanced_case_studies(scores, row_ids, case_groups, per_direction=2)
    assert [case["case_type"] for case in cases].count("supporting") == 2
    assert [case["case_type"] for case in cases].count("contradicting") == 2
    assert len({case["equilibrium_file"] for case in cases}) == 4
    assert cases == select_balanced_case_studies(
        scores, row_ids, case_groups, per_direction=2
    )
    negative_cases = select_balanced_case_studies(
        scores, row_ids, case_groups, per_direction=2, expected_sign=-1
    )
    assert negative_cases[0]["case_type"] == "supporting"
    assert float(negative_cases[0]["score"]) < 0

    negative_alignment = replace(
        first,
        rank_correlation=-0.25,
        per_sample_rank_correlation=scores,
    )
    aligned_cases = _select_alignment_case_studies(
        negative_alignment,
        row_ids,
        case_groups,
        per_direction=2,
    )
    supporting = [
        float(case["score"])
        for case in aligned_cases
        if case["case_type"] == "supporting"
    ]
    assert supporting == [-0.9, -0.8]
    assert {int(case["expected_sign"]) for case in aligned_cases} == {-1}


def test_lag_stability_wrap_tolerance_and_reported_flags_are_pinned() -> None:
    learned, physical, groups = _cyclic_fixture()
    base = circular_alignment(
        learned,
        physical,
        groups,
        mode="signed",
        sparsity=0.1,
        bootstrap_replicates=4,
        seed=82,
    )
    wrapped = replace(
        base,
        best_lag=1,
        bootstrap_best_lag=np.asarray([95, 94, 5, 50], dtype=np.int16),
    )
    assert _lag_within_tolerance_recurrence(wrapped, 4) == pytest.approx(0.75)
    assert _lag_within_tolerance_recurrence(wrapped, 3) == pytest.approx(0.5)

    failing = replace(
        wrapped,
        rank_ci_lower=-0.1,
        rank_ci_upper=0.2,
        bootstrap_best_lag=np.asarray([50, 50, 50, 50], dtype=np.int16),
    )
    assert _association_bootstrap_stable(failing) is False
    row = _alignment_row(
        {},
        failing,
        sample_count=len(groups),
        sparsity=0.1,
        bootstrap_replicates=4,
        minimum_recurrence=0.5,
        lag_tolerance=4,
    )
    assert row["lag_bootstrap_stable"] is False
    assert row["association_bootstrap_stable"] is False


def test_scalar_and_paired_results_use_whole_equilibria_and_native_units() -> None:
    groups = np.repeat(np.asarray([f"eq{index}" for index in range(10)]), 2)
    learned = np.repeat(np.linspace(-2.0, 3.0, 10), 2)
    zonal = np.exp(learned)
    association = scalar_rank_association(
        learned,
        zonal,
        groups,
        bootstrap_replicates=60,
        seed=9,
    )
    assert association.spearman_rho == pytest.approx(1.0)
    assert association.bootstrap_group == "equilibrium_files"

    sibling_groups = np.asarray(["eq0", "eq0", "eq1", "eq1", "eq2", "eq2"])
    grouped_draws = _grouped_row_draws(sibling_groups, replicates=4, seed=123)
    expected_draws = (
        np.asarray([0, 1, 4, 5, 2, 3]),
        np.asarray([0, 1, 4, 5, 0, 1]),
        np.asarray([0, 1, 0, 1, 2, 3]),
        np.asarray([0, 1, 2, 3, 4, 5]),
    )
    for actual, expected_draw in zip(grouped_draws, expected_draws, strict=True):
        np.testing.assert_array_equal(actual, expected_draw)

    moving_left = np.asarray([0.0, 8.0, 2.0, -3.0, 4.0, 1.0])
    moving_right = np.asarray([5.0, 0.0, 3.0, 7.0, -2.0, 4.0])
    moving = scalar_rank_association(
        moving_left,
        moving_right,
        sibling_groups,
        bootstrap_replicates=4,
        seed=123,
    )
    expected_bootstrap = np.asarray(
        [
            _spearman(moving_left[draw], moving_right[draw])
            for draw in expected_draws
        ]
    )
    np.testing.assert_allclose(moving.bootstrap_rho, expected_bootstrap, atol=1e-12)
    assert len(np.unique(moving.bootstrap_rho)) > 1

    split_values = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    split_zonal = np.asarray([9.0, 8.0, 7.0, 6.0, 1.0, 5.0, 2.0, 6.0, 3.0, 4.0])
    split_groups = np.asarray([f"split{index}" for index in range(10)])
    split = _zonal_summary_associations(
        split_values,
        split_zonal,
        split_groups,
        bootstrap_replicates=80,
        seed=27,
    )
    assert split.zero_count == 4
    assert split.active_count == 6
    assert split.zero_fraction == pytest.approx(0.4)
    assert split.active is not None
    assert split.pooled.spearman_rho != pytest.approx(split.active.spearman_rho)
    assert split.active.spearman_rho == pytest.approx(0.37142857142857144)
    assert split.active.ci_lower < 0 < split.active.ci_upper
    assert split.pooled_bootstrap_stable is True
    assert split.active_bootstrap_stable is False

    monotone = _zonal_summary_associations(
        np.asarray([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        np.arange(8, dtype=np.float64),
        np.asarray([f"monotone{index}" for index in range(8)]),
        bootstrap_replicates=80,
        seed=28,
    )
    assert monotone.active is not None
    assert monotone.active_bootstrap_stable is True

    neutral = _zonal_summary_associations(
        np.arange(6, dtype=np.float64),
        np.asarray([0.0, 3.0, 1.0, 5.0, 2.0, 4.0]),
        np.asarray([f"neutral{index}" for index in range(6)]),
        bootstrap_replicates=80,
        seed=30,
    )
    assert neutral.pooled.ci_lower < 0 < neutral.pooled.ci_upper
    assert neutral.pooled_bootstrap_stable is False

    guarded = _zonal_summary_associations(
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([2.0, 1.0, 0.0]),
        np.asarray(["guard0", "guard1", "guard2"]),
        bootstrap_replicates=20,
        seed=29,
    )
    assert guarded.zero_count == 2
    assert guarded.active_count == 1
    assert guarded.active is None
    assert guarded.active_bootstrap_stable is False

    fixed_native = np.asarray([-2.0, -1.0, 0.0, 1.0] * 5)
    varied_native = fixed_native - 0.5
    paired = paired_native_difference(
        fixed_native,
        varied_native,
        groups,
        bootstrap_replicates=60,
        seed=9,
    )
    assert paired.estimate == pytest.approx(0.5)
    assert paired.ci_lower == pytest.approx(0.5)
    assert paired.ci_upper == pytest.approx(0.5)
    assert paired.bootstrap_group == "equilibrium_files"
    assert paired.estimand == "native max(log Q, -2)"
    assert paired.estimate != pytest.approx(
        float(np.mean(np.exp(fixed_native) - np.exp(varied_native)))
    )

    # Pin the resampling unit itself, not just the label attached to the result.
    # Each selected equilibrium must carry both of its rows with the same weight.
    grouped_labels = np.asarray(["eq0", "eq0", "eq1", "eq1", "eq2", "eq2"])
    grouped_difference = np.asarray([0.0, 0.0, 2.0, 2.0, 10.0, 10.0])
    grouped = paired_native_difference(
        grouped_difference,
        np.zeros_like(grouped_difference),
        grouped_labels,
        bootstrap_replicates=12,
        seed=31,
    )
    rng = np.random.default_rng(31)
    group_means = np.asarray([0.0, 2.0, 10.0])
    expected = np.asarray(
        [group_means[rng.integers(0, 3, size=3)].mean() for _ in range(12)]
    )
    np.testing.assert_allclose(grouped.bootstrap_difference, expected)
    assert grouped.estimate == pytest.approx(4.0)
    assert grouped.estimate != pytest.approx(np.median(grouped_difference))
    assert grouped.ci_lower == pytest.approx(
        np.quantile(grouped.bootstrap_difference, 0.025)
    )
    assert grouped.ci_upper == pytest.approx(
        np.quantile(grouped.bootstrap_difference, 0.975)
    )
    assert grouped.ci_lower < grouped.ci_upper


def test_alignment_rejects_row_bootstrap_labels_and_misaligned_pairs() -> None:
    learned, physical, groups = _cyclic_fixture()
    with pytest.raises(ValueError, match="equilibrium"):
        circular_alignment(
            learned,
            physical,
            np.arange(len(groups)),
            mode="signed",
            sparsity=0.1,
            bootstrap_replicates=20,
            seed=4,
        )
    with pytest.raises(ValueError, match="same shape"):
        paired_native_difference(
            np.ones(5),
            np.ones(4),
            np.asarray(["a"] * 5),
            bootstrap_replicates=20,
            seed=4,
        )
