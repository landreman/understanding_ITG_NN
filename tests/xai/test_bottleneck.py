from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from itg_nn.xai.bottleneck import (
    HIDDEN_INTERVENTION_VALIDITY,
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
    _effect_interval,
    _group_bootstrap_weights,
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
            return inputs - baseline

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
    np.testing.assert_allclose(result.values, values.numpy())
    np.testing.assert_allclose(result.standard_errors, 0.0, atol=1e-6)


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
    assert np.all(result.random_direction_edit_magnitude > 0)
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
    features[held_out[0], 1] = 2.0
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

    signed = np.asarray([[-2.0, 1.0], [-1.0, 0.5], [1.0, 0.0], [2.0, -0.5]])
    shapley_result = ShapleyResult(
        values=signed,
        standard_errors=np.zeros_like(signed),
        baseline_output=np.zeros(4),
        prediction=signed.sum(axis=1),
        method="synthetic",
        evaluations=0,
        permutations=None,
    )
    rows = _shapley_rows(
        "toy", ("toy:u000", "toy:u001"), shapley_result, _stratum_masks(np.arange(4), 1)
    )
    suppressor = next(
        row
        for row in rows
        if row["stratum"] == "overall" and row["feature_id"] == "toy:u001"
    )
    assert float(suppressor["variance_contribution"]) < 0
    assert suppressor["validity_tag"] == HIDDEN_INTERVENTION_VALIDITY


def test_runner_direction_delta_uses_edited_minus_original_sign_and_records_size() -> None:
    class LinearMember:
        def head(
            self, bottleneck: torch.Tensor, a_over_lt: torch.Tensor, a_over_ln: torch.Tensor
        ) -> torch.Tensor:
            return bottleneck[:, 0] + 0.0 * (a_over_lt + a_over_ln)

    bottleneck = np.asarray([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    drives = np.zeros((4, 2))
    delta, magnitude = _direction_delta(
        LinearMember(),  # type: ignore[arg-type]
        bottleneck,
        drives,
        np.asarray([1.0, 0.0]),
        torch.device("cpu"),
    )
    np.testing.assert_allclose(delta, -bottleneck[:, 0])
    assert magnitude == pytest.approx(1.0)
