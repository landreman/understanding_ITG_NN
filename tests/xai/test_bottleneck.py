from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from itg_nn.xai.bottleneck import (
    HIDDEN_INTERVENTION_VALIDITY,
    InterventionResult,
    ShapleyResult,
    bottleneck_interventions,
    exact_or_sampled_shapley,
    grouped_cv_predictions,
    grouped_folds,
    registered_invariants,
    variance_decomposition,
)
from scripts.xai_s04_bottleneck import (
    _direction_delta,
    _encoded_used_row,
    _effect_interval,
    _group_bootstrap_weights,
    _interval_from_draws,
    _metric_rows,
    _r2_draws,
    _rms_draws,
    _shapley_rows,
    _stratum_masks,
)


def _periodic_bottleneck(geometry: torch.Tensor) -> torch.Tensor:
    """Analytic shift-invariant summary with one deliberately ignored unit."""

    return torch.stack(
        (
            geometry[:, :, 0].mean(dim=1),
            geometry[:, :, 1].square().mean(dim=1),
            geometry[:, :, 6].mean(dim=1),
        ),
        dim=1,
    )


def test_exact_shapley_recovers_analytic_cyclic_toy_and_native_output() -> None:
    generator = torch.Generator().manual_seed(12)
    geometry = torch.randn(9, 96, 7, generator=generator)
    bottleneck = _periodic_bottleneck(geometry)
    a_over_lt = torch.linspace(-1.0, 1.0, len(geometry))
    a_over_ln = torch.linspace(0.5, -0.5, len(geometry))
    features = torch.column_stack((bottleneck, a_over_lt, a_over_ln))
    reference = torch.zeros(5)

    def native_head(values: torch.Tensor) -> torch.Tensor:
        return values[:, 0] + 2 * values[:, 1] + 3 * values[:, 3]

    result = exact_or_sampled_shapley(native_head, features, reference)
    expected = torch.column_stack(
        (features[:, 0], 2 * features[:, 1], torch.zeros(9), 3 * features[:, 3], torch.zeros(9))
    ).numpy()
    np.testing.assert_allclose(result.values, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        result.values.sum(1),
        result.prediction - result.baseline_output,
        rtol=2e-6,
        atol=2e-6,
    )
    assert result.method == "exact_enumeration"
    assert np.count_nonzero(result.standard_errors) == 0
    # Exponentiating the native clipped-log output would fail this signed identity.
    assert np.any(result.values[:, 3] < 0)


def test_exact_shapley_splits_a_pair_interaction_equally() -> None:
    values = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
    result = exact_or_sampled_shapley(
        lambda x: x[:, 0] * x[:, 1], values, torch.zeros(2)
    )
    expected = 0.5 * values.prod(dim=1).numpy()
    np.testing.assert_allclose(result.values[:, 0], expected)
    np.testing.assert_allclose(result.values[:, 1], expected)


def test_sampled_shapley_is_deterministic_reports_se_and_is_efficient() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(11, 5, generator=generator)
    head = lambda x: x[:, 0] * x[:, 1] + x[:, 2].square() - 0.5 * x[:, 4]
    kwargs = dict(exact_max_features=2, permutations=80, seed=31)
    first = exact_or_sampled_shapley(head, values, torch.zeros(5), **kwargs)
    second = exact_or_sampled_shapley(head, values, torch.zeros(5), **kwargs)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.standard_errors, second.standard_errors)
    np.testing.assert_allclose(
        first.values.sum(1), first.prediction - first.baseline_output, atol=2e-6
    )
    assert first.method in {
        "captum_shapley_value_sampling",
        "permutation_sampling_fallback",
    }
    assert first.permutations == 80
    assert np.all(first.standard_errors >= 0)


def test_captum_sampled_shapley_contract_is_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class FakeShapleyValueSampling:
        def __init__(self, forward_func: object) -> None:
            self.forward_func = forward_func

        def attribute(self, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
            calls.append(int(kwargs["n_samples"]))
            baseline = torch.as_tensor(kwargs["baselines"], dtype=inputs.dtype)
            return inputs - baseline + 0.01 * len(calls)

    captum = types.ModuleType("captum")
    captum_attr = types.ModuleType("captum.attr")
    captum_attr.ShapleyValueSampling = FakeShapleyValueSampling  # type: ignore[attr-defined]
    captum.attr = captum_attr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "captum", captum)
    monkeypatch.setitem(sys.modules, "captum.attr", captum_attr)

    values = torch.arange(30, dtype=torch.float32).reshape(6, 5) / 10
    result = exact_or_sampled_shapley(
        lambda x: x.sum(1),
        values,
        torch.zeros(5),
        exact_max_features=2,
        permutations=34,
        seed=3,
    )
    assert result.method == "captum_shapley_value_sampling"
    assert len(calls) == 8
    assert sum(calls) == 34
    weights = np.asarray(calls, dtype=np.float64) / sum(calls)
    offsets = 0.01 * np.arange(1, len(calls) + 1)
    expected_offset = float(weights @ offsets)
    effective_blocks = 1.0 / np.square(weights).sum()
    expected_se = np.sqrt(
        np.sum(weights * np.square(offsets - expected_offset))
        / (effective_blocks - 1.0)
    )
    np.testing.assert_allclose(result.values, values.numpy() + expected_offset, atol=1e-6)
    np.testing.assert_allclose(result.standard_errors, expected_se, atol=1e-7)
    assert expected_se > 0


def test_variance_shares_sum_to_one_and_retain_signed_suppressors() -> None:
    shapley = np.asarray(
        [[-2.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [2.0, -1.0]]
    )
    prediction = shapley.sum(axis=1)
    contribution, share = variance_decomposition(shapley, prediction)
    assert contribution.shape == share.shape == (2,)
    assert share.sum() == pytest.approx(1.0)
    assert share[1] < 0


def test_interventions_keep_signed_member_sample_effects_and_null_unit() -> None:
    generator = torch.Generator().manual_seed(18)
    values = torch.randn(15, 4, generator=generator)
    reference = values.mean(dim=0)
    result = bottleneck_interventions(
        lambda x: x[:, 0] + 2 * x[:, 1] + x[:, 0] * x[:, 1],
        values,
        reference,
        feature_names=("m:u000", "m:u001", "m:u002", "m:u003"),
        intervention_features=range(4),
        seed=9,
        random_directions=3,
    )
    assert result.single_delta.shape == (3, 4, 15)
    assert result.pair_delta.shape == result.pair_interaction.shape == (3, 6, 15)
    assert result.random_direction_delta.shape == (3, 15)
    assert result.random_direction_edit_magnitude.shape == (3,)
    rng = np.random.default_rng(9)
    for _ in range(values.shape[1]):
        rng.permutation(len(values))
    standardized = (values.numpy() - reference.numpy()) / values.numpy().std(axis=0)
    expected_magnitudes = []
    for _ in range(3):
        direction = rng.normal(size=values.shape[1])
        direction /= np.linalg.norm(direction)
        expected_magnitudes.append(
            np.sqrt(np.mean(np.square(standardized @ direction)))
        )
    np.testing.assert_allclose(
        result.random_direction_edit_magnitude, expected_magnitudes, rtol=1e-6
    )
    assert not np.allclose(result.random_direction_edit_magnitude, 1.0)
    np.testing.assert_allclose(result.single_delta[:, 2:], 0.0, atol=1e-7)
    pair = np.flatnonzero(np.all(result.pair_indices == (0, 1), axis=1))[0]
    assert np.sqrt(np.mean(np.square(result.pair_interaction[1, pair]))) > 0.1
    assert result.validity_tag == HIDDEN_INTERVENTION_VALIDITY


def test_grouped_folds_never_split_equilibria_and_are_deterministic() -> None:
    groups = np.repeat(np.asarray(["eq0", "eq1", "eq2", "eq3", "eq4", "eq5"]), 2)
    first = grouped_folds(groups, n_folds=3, seed=4)
    second = grouped_folds(groups, n_folds=3, seed=4)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert sorted(np.concatenate(first).tolist()) == list(range(len(groups)))
    for test_rows in first:
        train_rows = np.setdiff1d(np.arange(len(groups)), test_rows)
        assert set(groups[test_rows]).isdisjoint(groups[train_rows])


def test_grouped_decoders_find_signal_and_label_permutation_is_null() -> None:
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(60), 2)
    group_features = rng.normal(size=(60, 3))
    features = np.repeat(group_features, 2, axis=0)
    target = features[:, 0] - 0.75 * features[:, 1] + 0.1 * rng.normal(size=120)
    prediction = grouped_cv_predictions(
        features, target, groups, kind="linear", n_folds=5, seed=8
    )
    control = grouped_cv_predictions(
        features,
        target,
        groups,
        kind="linear",
        n_folds=5,
        seed=8,
        permute_labels=True,
    )
    r2 = 1 - np.square(prediction - target).sum() / np.square(target - target.mean()).sum()
    control_r2 = 1 - np.square(control - target).sum() / np.square(target - target.mean()).sum()
    assert r2 > 0.95
    assert control_r2 < 0.2


def test_grouped_decoder_does_not_amplify_near_dead_fold_features() -> None:
    rng = np.random.default_rng(41)
    groups = np.arange(2000)
    features = np.column_stack((rng.normal(size=2000), np.zeros(2000)))
    held_out = grouped_folds(groups, n_folds=5, seed=12)[0]
    training = np.setdiff1d(np.arange(len(groups)), held_out)
    rare_training = training[:8]  # 0.5% active in this fold: S02 near-dead.
    features[rare_training, 1] = 0.01
    # Active on enough held-out rows to look globally supported, but still only
    # 0.5% active in this fold's training rows. A full-dataset support mask leaks.
    features[held_out[:30], 1] = 2.0
    target = 0.75 * features[:, 0] + rng.normal(scale=0.05, size=2000)
    target[rare_training] += 5.0
    for kind in ("linear", "nonlinear"):
        prediction = grouped_cv_predictions(
            features, target, groups, kind=kind, n_folds=5, seed=12, ridge=1.0
        )
        assert np.isfinite(prediction).all()
        assert np.max(np.abs(prediction)) < 5.0
        r2 = 1 - np.square(prediction - target).sum() / np.square(target - target.mean()).sum()
        assert r2 > 0.8
        unsupported_control = grouped_cv_predictions(
            features,
            target,
            groups,
            kind=kind,
            n_folds=5,
            seed=12,
            ridge=1.0,
            minimum_active_fraction=0.0,
        )
        assert np.max(np.abs(unsupported_control)) > 50.0


def test_registered_invariants_match_closed_form_geometry() -> None:
    geometry = np.zeros((3, 96, 7), dtype=np.float64)
    geometry[:, :, 0] = 4.0
    geometry[:, :, 1] = np.asarray([-1.0, 1.0, 1.0])[:, None]
    geometry[:, :, 2] = np.asarray([-1.0, 1.0, 1.0])[:, None]
    geometry[:, :, 6] = 9.0
    scalar = np.asarray([[2.0, 0.1, -0.5, 0.0, 7.0], [3.0, 0.2, 0.0, 0.0, 8.0], [4.0, 0.3, 0.5, 0.0, 9.0]])
    result = registered_invariants(
        geometry, scalar, ("nfp", "iota", "shat", "d_pressure_d_s", "aspect")
    )
    np.testing.assert_allclose(result["log_FSA_grad_x"], np.log(3.0))
    np.testing.assert_allclose(result["log_f_Q"], np.log(np.asarray([1.35, 8.1, 8.1])))
    np.testing.assert_allclose(result["f_stab"], np.asarray([0.6, 2.1, 2.1]))
    np.testing.assert_array_equal(result["shat"], scalar[:, 2])
    np.testing.assert_array_equal(result["nfp"], scalar[:, 0])
    np.testing.assert_array_equal(result["aspect"], scalar[:, 4])


def test_runner_statistics_preserve_groups_strata_signs_and_95_percent_interval() -> None:
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    weights, inverse = _group_bootstrap_weights(groups, replicates=200, seed=17)
    assert weights.shape == (200, 6)
    assert np.array_equal(inverse, np.asarray([0, 0, 1, 1, 2, 2]))
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    np.testing.assert_array_equal(weights[:, 2], weights[:, 3])
    np.testing.assert_array_equal(weights[:, 4], weights[:, 5])

    actual = np.asarray([-2.0, -1.9, -1.89, 0.0, 0.4, 1.0])
    masks = _stratum_masks(actual, -1.9)
    np.testing.assert_array_equal(
        masks["stable_or_near_floor"], np.asarray([True, True, False, False, False, False])
    )
    assert not np.any(masks["stable_or_near_floor"] & masks["unstable"])
    assert np.all(masks["stable_or_near_floor"] | masks["unstable"])

    values = np.asarray([1.0, -2.0, 3.0, -4.0, 2.0, -1.0])
    point, lower, upper = _effect_interval(values, weights, statistic="mean_absolute")
    draws = weights @ np.abs(values) / weights.sum(axis=1)
    assert point == pytest.approx(np.mean(np.abs(values)))
    assert lower == pytest.approx(np.quantile(draws, 0.025))
    assert upper == pytest.approx(np.quantile(draws, 0.975))

    suppressor = np.asarray([-2.0, -1.0, 1.0, 2.0])
    prediction = np.asarray([4.0, 2.0, 0.0, 0.5])
    signed = np.column_stack((prediction - suppressor, suppressor))
    shapley_result = ShapleyResult(
        values=signed,
        standard_errors=np.zeros_like(signed),
        baseline_output=np.zeros(4),
        prediction=prediction,
        method="synthetic",
        evaluations=0,
        permutations=None,
    )
    rows = _shapley_rows(
        "toy", ("toy:u000", "toy:u001"), shapley_result, _stratum_masks(np.arange(4), 1)
    )
    suppressor_row = next(
        row
        for row in rows
        if row["stratum"] == "overall" and row["feature_id"] == "toy:u001"
    )
    expected_contribution = np.mean(
        (suppressor - suppressor.mean()) * (prediction - prediction.mean())
    )
    absolute_contribution = np.mean(
        (np.abs(suppressor) - np.abs(suppressor).mean())
        * (prediction - prediction.mean())
    )
    assert expected_contribution < 0 < absolute_contribution
    assert float(suppressor_row["variance_contribution"]) == pytest.approx(
        expected_contribution
    )
    assert suppressor_row["validity_tag"] == HIDDEN_INTERVENTION_VALIDITY


def test_runner_direction_delta_uses_edited_minus_original_sign_and_records_size() -> None:
    class LinearMember:
        def head(
            self, bottleneck: torch.Tensor, a_over_lt: torch.Tensor, a_over_ln: torch.Tensor
        ) -> torch.Tensor:
            return bottleneck[:, 0] + 0.0 * (a_over_lt + a_over_ln)

    bottleneck = np.asarray([[-2.0, -2.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
    drives = np.zeros((4, 2))
    direction = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    delta, magnitude = _direction_delta(
        LinearMember(),  # type: ignore[arg-type]
        bottleneck,
        drives,
        direction,
        torch.device("cpu"),
    )
    center = bottleneck.mean(axis=0)
    scale = bottleneck.std(axis=0)
    standardized = (bottleneck - center) / scale
    projection = standardized @ direction
    edited = (standardized - projection[:, None] * direction) * scale + center
    np.testing.assert_allclose(delta, edited[:, 0] - bottleneck[:, 0])
    assert magnitude == pytest.approx(np.sqrt(np.mean(np.square(projection))))
    assert magnitude != pytest.approx(1.0)


def test_runner_acceptance_intervals_use_grouped_weights_and_95_percent_quantiles() -> None:
    actual = np.asarray([-3.0, -1.0, 2.0, 5.0])
    predicted = np.asarray([-2.5, -0.5, 1.0, 4.0])
    values = np.asarray([1.0, 2.0, 4.0, 8.0])
    row_weights = np.asarray(
        [
            [3.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )

    r2_point, r2_draws = _r2_draws(actual, predicted, row_weights)
    denominator = row_weights.sum(axis=1)
    weighted_mean = row_weights @ actual / denominator
    total = np.sum(
        row_weights * np.square(actual[None, :] - weighted_mean[:, None]), axis=1
    )
    expected_r2_draws = 1.0 - (
        row_weights @ np.square(actual - predicted)
    ) / total
    np.testing.assert_allclose(r2_draws, expected_r2_draws)
    assert r2_point == pytest.approx(
        1.0
        - np.square(actual - predicted).sum()
        / np.square(actual - actual.mean()).sum()
    )

    rms_point, rms_draws = _rms_draws(values, row_weights)
    expected_rms_draws = np.sqrt(
        row_weights @ np.square(values) / row_weights.sum(axis=1)
    )
    np.testing.assert_allclose(rms_draws, expected_rms_draws)
    assert np.std(rms_draws) > 0.5
    assert rms_point == pytest.approx(np.sqrt(np.mean(np.square(values))))

    point, lower, upper = _interval_from_draws(rms_point, rms_draws)
    assert point == rms_point
    assert lower == pytest.approx(np.quantile(expected_rms_draws, 0.025))
    assert upper == pytest.approx(np.quantile(expected_rms_draws, 0.975))
    assert lower != pytest.approx(np.quantile(expected_rms_draws, 0.25))
    assert upper != pytest.approx(np.quantile(expected_rms_draws, 0.75))


def test_runner_row_builders_apply_normalization_and_signed_mse_change() -> None:
    original = np.asarray([2.0, 1.0, -1.0, -2.0])
    single_delta = -original
    pair_delta = -0.5 * original
    result = InterventionResult(
        modes=("mean",),
        feature_names=("toy:u000", "toy:u001"),
        original_prediction=original,
        single_delta=np.asarray([[single_delta, np.zeros(4)]]),
        pair_indices=np.asarray([[0, 1]]),
        pair_delta=np.asarray([[pair_delta]]),
        pair_interaction=np.zeros((1, 1, 4)),
        random_direction_delta=np.asarray([[1.0, 1.0, 1.0, 1.0]]),
        random_direction_edit_magnitude=np.asarray([0.5]),
        validity_tag=HIDDEN_INTERVENTION_VALIDITY,
    )
    rows = _metric_rows("toy", result, np.zeros(4), {"overall": np.ones(4, bool)})
    single_row = next(
        row
        for row in rows
        if row["scope"] == "single_unit" and row["feature_1"] == "toy:u000"
    )
    pair_row = next(row for row in rows if row["scope"] == "unit_pair")
    original_mse = np.mean(np.square(original))
    assert float(single_row["mse_change"]) == pytest.approx(-original_mse)
    assert float(pair_row["mse_change"]) == pytest.approx(-0.75 * original_mse)

    target = np.asarray([-3.0, -1.0, 2.0, 5.0])
    delta = np.asarray([-3.0, -1.0, 2.0, 4.0])
    random_delta = np.asarray([[1.0, 1.0, 1.0, 1.0], [0.0, 2.0, 0.0, 2.0]])
    random_magnitude = np.asarray([0.5, 2.0])
    row, _, _ = _encoded_used_row(
        member_id="toy",
        target_name="concept",
        stratum="overall",
        target=target,
        linear_prediction=target + 0.5,
        nonlinear_prediction=target + 0.25,
        permuted_linear_prediction=np.zeros(4),
        permuted_nonlinear_prediction=np.ones(4),
        delta=delta,
        direction_magnitude=1.25,
        random_direction_delta=random_delta,
        random_direction_edit_magnitude=random_magnitude,
        row_weights=np.eye(4),
    )
    used_rms = np.sqrt(np.mean(np.square(delta)))
    random_rms = np.sqrt(np.mean(np.square(random_delta), axis=1))
    assert float(row["used_rms_per_edit_sd"]) == pytest.approx(used_rms / 1.25)
    assert float(row["random_direction_control_median_rms_per_edit_sd"]) == pytest.approx(
        np.median(random_rms / random_magnitude)
    )
    assert float(row["used_rms_per_edit_sd"]) != pytest.approx(used_rms * 1.25)
