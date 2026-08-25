from __future__ import annotations

import numpy as np

from itg_nn.xai.distillation import (
    expression_pareto_frontier,
    expression_recurrence,
    grouped_ebm_crossfit,
    grouped_folds,
    grouped_term_recurrence,
    invariant_feature_table,
)


class LinearToyEstimator:
    """Small deterministic stand-in that exercises the production cross-fit wiring."""

    def __init__(self, *, seed: int, interactions=()) -> None:
        self.seed = seed
        self.interactions = tuple(interactions)

    def fit(self, values: np.ndarray, target: np.ndarray):
        design = [np.ones(len(values)), *values.T]
        design.extend(values[:, left] * values[:, right] for left, right in self.interactions)
        matrix = np.column_stack(design)
        self.coefficient = np.linalg.lstsq(matrix, target, rcond=None)[0]
        self.feature_importances_ = np.abs(self.coefficient[1 : values.shape[1] + 1])
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        design = [np.ones(len(values)), *values.T]
        design.extend(values[:, left] * values[:, right] for left, right in self.interactions)
        return np.column_stack(design) @ self.coefficient


def _factory(*, seed: int, interactions=()):
    return LinearToyEstimator(seed=seed, interactions=interactions)


def _cyclic_geometry(rows: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = 2 * np.pi * np.arange(96) / 96
    phase = np.linspace(0, 2 * np.pi, rows, endpoint=False)
    amplitude = np.linspace(0.5, 2.0, rows)
    geometry = np.zeros((rows, 96, 7), dtype=np.float64)
    geometry[:, :, 0] = 2.0 + 0.1 * np.cos(z[None, :] + phase[:, None])
    geometry[:, :, 1] = np.sin(z[None, :] + phase[:, None])
    geometry[:, :, 2] = amplitude[:, None] * np.cos(z[None, :] + phase[:, None])
    geometry[:, :, 3] = 0.2 * np.sin(2 * z[None, :] - phase[:, None])
    geometry[:, :, 4] = 1.0
    geometry[:, :, 5] = 0.1 * np.sin(3 * z[None, :])
    geometry[:, :, 6] = (1.0 + 0.1 * amplitude[:, None] * np.maximum(np.cos(z), 0)) ** 2
    scalar = np.column_stack((np.ones(rows), np.linspace(-1, 1, rows), np.full(rows, 5.0)))
    return geometry, scalar, amplitude


def test_feature_table_is_exactly_cyclic_invariant_and_uses_robust_scales() -> None:
    geometry, scalar, _ = _cyclic_geometry()
    drives = np.linspace(1.5, 5.0, len(geometry))
    table = invariant_feature_table(
        geometry,
        scalar,
        ("nfp", "shat", "aspect"),
        drives,
        drives / 3,
        channel_scales=np.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
    )
    shifted = invariant_feature_table(
        np.roll(geometry, 17, axis=1),
        scalar,
        ("nfp", "shat", "aspect"),
        drives,
        drives / 3,
        channel_scales=np.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
    )
    np.testing.assert_allclose(table.values, shifted.values, atol=1e-12)
    assert table.version == "S12-v1"
    assert "parallel_roughness_iqr_scaled" in table.names
    roughness = table.values[:, table.names.index("parallel_roughness_iqr_scaled")]
    rescaled = invariant_feature_table(
        geometry * np.asarray([1, 1, 1, 1, 1, 1, 1000])[None, None, :],
        scalar,
        ("nfp", "shat", "aspect"),
        drives,
        drives / 3,
        channel_scales=np.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64000.0]),
    )
    np.testing.assert_allclose(
        roughness,
        rescaled.values[:, rescaled.names.index("parallel_roughness_iqr_scaled")],
        atol=1e-12,
    )
    assert {row["validity_tag"] for row in table.definitions} == {"observed-comparison"}


def test_grouped_crossfit_recovers_analytic_cyclic_signal_and_null_control() -> None:
    geometry, scalar, amplitude = _cyclic_geometry(60)
    drives = np.linspace(2.0, 5.0, len(geometry))
    table = invariant_feature_table(
        geometry,
        scalar,
        ("nfp", "shat", "aspect"),
        drives,
        drives * 0.2,
        channel_scales=np.ones(7),
    )
    signal = table.values[:, table.names.index("bad_curvature_compression")]
    null = np.random.default_rng(4).normal(size=len(signal))
    values = np.column_stack((signal, drives, null))
    # This is already the native clipped-log quantity. Exponentiating it would
    # fail the exact held-out control and is therefore not an equivalent target.
    native = np.maximum(-2.0, -1.5 + 0.8 * signal + 0.2 * drives)
    groups = np.repeat(np.arange(30), 2)
    result = grouped_ebm_crossfit(
        values,
        native,
        groups,
        feature_names=("cyclic_signal", "a_over_LT", "permuted_null"),
        folds=5,
        seed=17,
        interactions=(),
        estimator_factory=_factory,
    )
    assert result.held_out_r2 > 0.999999
    assert result.estimand == "native max(log Q, -2)"
    assert result.feature_importance[0] > 10 * result.feature_importance[2]
    assert not np.allclose(result.prediction, np.exp(native))
    for fold in np.unique(result.fold):
        assert set(groups[result.fold == fold]).isdisjoint(groups[result.fold != fold])


def test_grouped_folds_keep_sibling_tubes_together_and_are_deterministic() -> None:
    groups = np.asarray(["eq0", "eq0", "eq1", "eq2", "eq2", "eq3", "eq4"])
    first = grouped_folds(groups, 3, 91)
    second = grouped_folds(groups, 3, 91)
    np.testing.assert_array_equal(first, second)
    assert first[0] == first[1]
    assert first[3] == first[4]


def test_bootstrap_recurrence_resamples_equilibria_and_retains_null() -> None:
    rng = np.random.default_rng(11)
    group_values = rng.normal(size=(40, 2))
    features = np.repeat(group_values, 2, axis=0)
    groups = np.repeat(np.arange(40), 2)
    target = 2.0 * features[:, 0] + 0.01 * rng.normal(size=len(features))
    first = grouped_term_recurrence(
        features,
        target,
        groups,
        feature_names=("signal", "null"),
        replicates=30,
        seed=29,
        top_k=1,
        estimator_factory=_factory,
    )
    second = grouped_term_recurrence(
        features,
        target,
        groups,
        feature_names=("signal", "null"),
        replicates=30,
        seed=29,
        top_k=1,
        estimator_factory=_factory,
    )
    assert first == second
    recurrence = {row["feature_name"]: row["top_k_recurrence"] for row in first}
    assert recurrence["signal"] == 1.0
    assert recurrence["null"] == 0.0
    assert {row["bootstrap_unit"] for row in first} == {"equilibrium_files"}


def test_symbolic_pareto_and_recurrence_keep_negative_controls_visible() -> None:
    candidates = [
        {"expression": "mean_signal", "complexity": 1, "held_out_r2": 0.80},
        {"expression": "2 * mean_signal", "complexity": 3, "held_out_r2": 0.95},
        {"expression": "noise", "complexity": 2, "held_out_r2": -0.10},
        {"expression": "redundant", "complexity": 5, "held_out_r2": 0.90},
    ]
    frontier = expression_pareto_frontier(candidates)
    assert [row["expression"] for row in frontier] == [
        "mean_signal",
        "2 * mean_signal",
    ]
    assert candidates[2]["held_out_r2"] < 0
    recurrence = expression_recurrence(
        [
            ["mean_signal", "2 * mean_signal"],
            ["mean_signal", "noise"],
            ["mean_signal"],
        ]
    )
    by_expression = {row["expression"]: row["recurrence"] for row in recurrence}
    assert by_expression == {
        "mean_signal": 1.0,
        "2 * mean_signal": 1 / 3,
        "noise": 1 / 3,
    }
    assert {row["bootstrap_unit"] for row in recurrence} == {"equilibrium_files"}
