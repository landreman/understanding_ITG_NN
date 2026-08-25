import numpy as np
import pytest
import torch

from itg_nn.xai.disagreement import (
    diagnostic_association_rows,
    ensemble_spread,
    failure_categories,
    grouped_bootstrap_spearman,
    grouped_crossfit_ridge,
    grouped_fold_ids,
    member_residuals,
    paired_outcome_association_rows,
    perturbation_effect_rows,
    robust_scaled_channel_gradient,
    spread_input_gradient,
)
from itg_nn.xai.perturbations import ValidityTag


def test_cyclic_toy_spread_gradient_has_closed_form_and_null_channel():
    geometry = torch.tensor(
        [
            [[1.0, 2.0, 9.0], [3.0, 4.0, -8.0], [5.0, 6.0, 7.0]],
            [[2.0, -3.0, 5.0], [4.0, -1.0, 6.0], [6.0, -2.0, 7.0]],
        ],
        requires_grad=True,
    )
    means = geometry.mean(dim=1)
    member_outputs = torch.stack(
        (means[:, 0] + means[:, 1], means[:, 0] - means[:, 1]), dim=0
    )
    result = spread_input_gradient(member_outputs, geometry)

    expected_spread = means[:, 1].abs()
    expected_gradient = torch.zeros_like(geometry)
    expected_gradient[:, :, 1] = means[:, 1].sign()[:, None] / geometry.shape[1]
    assert torch.allclose(result.spread, expected_spread)
    assert torch.allclose(result.gradient, expected_gradient)

    shifted = torch.roll(geometry.detach(), shifts=1, dims=1).requires_grad_(True)
    shifted_means = shifted.mean(dim=1)
    shifted_outputs = torch.stack(
        (shifted_means[:, 0] + shifted_means[:, 1], shifted_means[:, 0] - shifted_means[:, 1]),
        dim=0,
    )
    shifted_result = spread_input_gradient(shifted_outputs, shifted)
    assert torch.allclose(shifted_result.spread, result.spread)
    assert torch.allclose(shifted_result.gradient, torch.roll(result.gradient, 1, 1))


def test_native_member_residuals_are_not_exponentiated_and_spread_is_population_std():
    predictions = np.asarray([[-2.0, 0.0], [-1.0, -0.5]])
    target = np.asarray([-2.0, -1.0])
    np.testing.assert_allclose(member_residuals(predictions, target), [[0.0, 1.0], [1.0, 0.5]])
    np.testing.assert_allclose(ensemble_spread(predictions), np.std(predictions, axis=0, ddof=0))


def test_robust_channel_scale_retains_member_sample_position_and_sign():
    gradients = np.asarray(
        [
            [[[-1.0, 2.0], [3.0, -4.0]]],
            [[[5.0, -6.0], [-7.0, 8.0]]],
        ]
    )
    scaled = robust_scaled_channel_gradient(gradients, np.asarray([2.0, 0.5]))
    np.testing.assert_allclose(
        scaled,
        np.asarray(
            [
                [[[-2.0, 1.0], [6.0, -2.0]]],
                [[[10.0, -3.0], [-14.0, 4.0]]],
            ]
        ),
    )


def test_common_mode_failure_is_named_by_fixed_native_unit_thresholds():
    labels = failure_categories(
        np.asarray([0.2, 0.2, 0.05, 0.05]),
        np.asarray([0.1, 0.8, 0.8, 0.1]),
        high_spread_threshold=0.1,
        high_error_threshold=0.5,
    )
    assert labels.tolist() == [
        "high_spread_low_error",
        "high_spread_high_error",
        "common_mode_failure",
        "unanimous_success",
    ]


def test_grouped_folds_never_split_sibling_tubes_and_crossfit_is_deterministic():
    groups = np.asarray(["eq0", "eq0", "eq1", "eq1", "eq2", "eq2", "eq3", "eq3"])
    folds = grouped_fold_ids(groups, folds=4, seed=19)
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1

    features = np.column_stack((np.arange(8.0), np.asarray([0, 1] * 4, dtype=float)))
    outcome = 1.5 * features[:, 0] - 0.25 * features[:, 1]
    first = grouped_crossfit_ridge(features, outcome, groups, folds=4, alpha=1e-6, seed=19)
    second = grouped_crossfit_ridge(features, outcome, groups, folds=4, alpha=1e-6, seed=19)
    np.testing.assert_array_equal(first.fold_ids, folds)
    np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
    np.testing.assert_allclose(first.predictions, second.predictions)
    assert np.isfinite(first.predictions).all()


def test_grouped_bootstrap_uses_one_multiplicity_per_equilibrium_and_is_deterministic():
    feature = np.asarray([0.0, 0.4, 1.0, 1.8, 2.0, 3.2, 4.0, 5.0])
    outcome = np.asarray([0.2, 1.0, 0.7, 2.4, 2.8, 2.2, 5.2, 3.9])
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    first = grouped_bootstrap_spearman(feature, outcome, groups, replicates=25, seed=7)
    second = grouped_bootstrap_spearman(feature, outcome, groups, replicates=25, seed=7)
    np.testing.assert_allclose(first, second, equal_nan=True)
    np.testing.assert_allclose(
        first[:5],
        [0.6, 0.7804878048780488, 0.9024390243902439,
         0.7804878048780488, 0.9024390243902439],
    )


def test_spread_error_association_reports_rank_and_linear_grouped_ranges_by_regime():
    spread = np.asarray([0.1, 0.2, 0.15, 0.4, 0.3, 0.7, 0.6, 0.8])
    error = np.asarray([0.05, 0.3, 0.1, 0.5, 0.2, 0.9, 0.4, 1.0])
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    rows = paired_outcome_association_rows(
        spread,
        error,
        groups,
        {"all": np.ones(8, dtype=bool), "stable": np.arange(8) < 4, "unstable": np.arange(8) >= 4},
        left_name="ensemble_spread",
        right_name="ensemble_absolute_error",
        replicates=30,
        seed=9,
    )
    assert len(rows) == 3
    assert {row["resampling_unit"] for row in rows} == {"equilibrium_files"}
    assert all(row["spearman_interval_lower"] <= row["spearman"] <= row["spearman_interval_upper"] for row in rows)
    assert all(row["pearson_interval_lower"] <= row["pearson"] <= row["pearson_interval_upper"] for row in rows)
    np.testing.assert_allclose(
        [rows[0]["spearman_interval_lower"], rows[0]["spearman_interval_upper"],
         rows[0]["pearson_interval_lower"], rows[0]["pearson_interval_upper"]],
        [0.8, 1.0, 0.8753582466450451, 0.9708723126837387],
    )


def test_diagnostic_table_reports_every_frozen_feature_without_confidence_language():
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    regimes = {
        "all": np.ones(6, dtype=bool),
        "stable_or_near_floor": np.asarray([1, 1, 0, 0, 0, 0], dtype=bool),
        "unstable": np.asarray([0, 0, 1, 1, 1, 1], dtype=bool),
    }
    rows = diagnostic_association_rows(
        {"support_distance": np.arange(6.0), "q_stds": np.arange(6.0)[::-1]},
        {"ensemble_spread": np.linspace(0.0, 1.0, 6), "ensemble_absolute_error": np.ones(6)},
        groups,
        regimes,
        replicates=20,
        seed=11,
    )
    assert len(rows) == 2 * 2 * 3
    assert {(row["feature"], row["outcome"]) for row in rows} == {
        ("support_distance", "ensemble_spread"),
        ("support_distance", "ensemble_absolute_error"),
        ("q_stds", "ensemble_spread"),
        ("q_stds", "ensemble_absolute_error"),
    }
    assert {row["resampling_unit"] for row in rows} == {"equilibrium_files"}
    assert {row["interval_kind"] for row in rows} == {"grouped_resample_sensitivity_interval"}
    assert {row["feature_selection"] for row in rows} == {"none_frozen_before_residual_analysis"}
    assert all("confidence" not in key.lower() for row in rows for key in row)


def test_perturbation_rows_retain_signed_member_effects_and_validity_tag():
    reference = np.asarray([[1.0, 2.0], [4.0, 8.0]])
    perturbed = np.asarray([[0.0, 3.0], [6.0, 5.0]])
    rows = perturbation_effect_rows(
        reference,
        perturbed,
        member_ids=("m0", "m1"),
        row_ids=(10, 20),
        perturbation="independent_channel_shift",
        validity=ValidityTag.OFF_MANIFOLD,
    )
    assert [row["signed_change_native"] for row in rows] == [-1.0, 1.0, 2.0, -3.0]
    assert {row["validity"] for row in rows} == {ValidityTag.OFF_MANIFOLD.value}
    assert {row["estimand"] for row in rows} == {"max(log Q, -2)"}
