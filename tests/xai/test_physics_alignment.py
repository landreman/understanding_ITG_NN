from __future__ import annotations

import numpy as np
import pytest

from itg_nn.xai.physics_alignment import (
    circular_alignment,
    paired_native_difference,
    scalar_rank_association,
    select_balanced_case_studies,
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
    assert result.cross_correlation_by_lag[7] == pytest.approx(1.0, abs=1e-12)
    assert result.overlap == pytest.approx(1.0)
    assert result.lag_recurrence == pytest.approx(1.0)
    assert result.bootstrap_group == "equilibrium_files"

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
    assert shifted.rank_correlation == pytest.approx(signed.rank_correlation)
    np.testing.assert_allclose(
        shifted.rank_correlation_by_lag, signed.rank_correlation_by_lag
    )
    assert positive.mode == "positive_contribution"
    assert positive.rank_correlation > -0.9


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
