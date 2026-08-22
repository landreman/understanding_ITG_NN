from __future__ import annotations

import numpy as np
import torch

from itg_nn.xai.attribution import integrated_gradients, toy_recovery
from itg_nn.xai.attribution_scaled import (
    build_stratification_masks,
    hierarchical_group_bootstrap,
    native_scalar_sensitivities,
    signed_consensus,
    validation_stability_correlation,
)
from itg_nn.xai.toys import PeriodicWindowToy


def test_scaled_attribution_recovers_wrapped_toy_and_keeps_null_channel_zero() -> None:
    toy = PeriodicWindowToy(channel=2, start=92, width=8)
    geometry = torch.zeros(6, 96, 7)
    geometry[:, toy.positions, 2] = torch.linspace(0.5, 1.0, len(toy.positions))
    geometry[:, :, 6] = 50.0  # large ignored channel: a required null control
    zeros = torch.zeros(len(geometry))

    result = integrated_gradients(
        lambda values: toy(values, zeros, zeros).squeeze(1),
        geometry,
        torch.zeros_like(geometry),
        steps=16,
        backend="fallback",
    )
    recovery = toy_recovery(
        result.values,
        relevant_channels=toy.expectation.channels,
        relevant_positions=toy.expectation.positions,
    )

    assert recovery == {"channel_top1": 1.0, "position_average_precision": 1.0}
    assert torch.count_nonzero(result.values[:, :, 6]) == 0
    assert result.metadata["estimand"] == "native max(log Q, -2)"


def test_signed_consensus_is_member_first_and_exposes_opposing_mechanisms() -> None:
    maps = np.zeros((4, 3, 3, 4), dtype=np.float64)
    maps[:2, :, 0] = 2.0
    maps[2:, :, 0] = -2.0
    maps[:, :, 1] = np.asarray([1.0, 2.0, 3.0, 4.0])[:, None, None]
    # Channel 2 is an exact null control.

    result = signed_consensus(maps)

    assert result.member_signed.shape == (4, 3, 4)
    assert np.all(result.median_signed[0] == 0.0)
    assert np.all(result.sign_agreement[0] == 0.5)
    assert np.all(result.median_signed[1] == 2.5)
    assert np.all(result.sign_agreement[1] == 1.0)
    assert np.all(result.median_absolute[2] == 0.0)
    assert np.all(result.sign_agreement[2] == 1.0)
    assert len(result.pairwise_rank_correlations) == 6


def test_hierarchical_bootstrap_resamples_members_and_whole_equilibria() -> None:
    values = np.asarray(
        [
            [0.0, 4.0, 10.0, 20.0],
            [2.0, 6.0, 12.0, 22.0],
        ]
    )
    equilibrium_files = np.asarray(["shared", "shared", "b", "c"])

    first = hierarchical_group_bootstrap(
        values, equilibrium_files, replicates=40, seed=17
    )
    second = hierarchical_group_bootstrap(
        values, equilibrium_files, replicates=40, seed=17
    )

    assert first.resampling_units == ("members", "equilibrium_files")
    assert first.estimate == np.mean(values)
    np.testing.assert_array_equal(first.samples, second.samples)
    rng = np.random.default_rng(17)
    group_rows = (np.asarray([2]), np.asarray([3]), np.asarray([0, 1]))
    expected = []
    for _ in range(40):
        member_draw = rng.integers(0, 2, size=2)
        group_draw = rng.integers(0, 3, size=3)
        row_draw = np.concatenate([group_rows[index] for index in group_draw])
        expected.append(values[np.ix_(member_draw, row_draw)].mean())
    np.testing.assert_array_equal(first.samples, expected)
    assert first.lower <= first.estimate <= first.upper


def test_native_scalar_sensitivity_explains_clipped_log_not_exponentiated_output() -> None:
    geometry = torch.zeros(3, 8, 2)
    drive_lt = torch.tensor([1.0, -5.0, 2.0])
    drive_ln = torch.tensor([0.5, 0.0, -0.5])

    def native_forward(
        values: torch.Tensor, lt: torch.Tensor, ln: torch.Tensor
    ) -> torch.Tensor:
        del values
        return torch.clamp(lt + 2.0 * ln, min=-2.0)

    result = native_scalar_sensitivities(
        native_forward,
        geometry,
        drive_lt,
        drive_ln,
        robust_scales=np.asarray([3.0, 5.0]),
    )

    np.testing.assert_allclose(
        result.values,
        np.asarray([[3.0, 10.0], [0.0, 0.0], [3.0, 10.0]]),
    )
    assert result.signed is True
    assert result.estimand == "native max(log Q, -2)"
    assert result.scale == "robust_per_scalar_drive"


def test_strata_never_pool_gradient_sets_or_floor_rows() -> None:
    masks = build_stratification_masks(
        gradient_set=np.asarray(["varied", "varied", "fixed", "fixed"]),
        target=np.asarray([-2.0, 1.0, -2.0, 2.0]),
        a_over_lt=np.asarray([1.0, 3.0, 3.0, 3.0]),
        a_over_ln=np.asarray([0.1, 0.9, 0.9, 0.9]),
        equilibrium_class=np.asarray([0, 1, 0, 1]),
        stable_threshold=-1.9,
    )

    for gradient_set in ("varied", "fixed"):
        stable = masks[f"gradient_set={gradient_set}|stability=stable_or_near_floor"]
        unstable = masks[f"gradient_set={gradient_set}|stability=unstable"]
        assert not np.any(stable & unstable)
        assert np.array_equal(
            stable | unstable, masks[f"gradient_set={gradient_set}|all"]
        )
    assert not np.any(
        masks["gradient_set=varied|all"] & masks["gradient_set=fixed|all"]
    )


def test_validation_stability_correlation_is_deterministic_and_rank_based() -> None:
    # Two middle members deliberately tie in agreement with the median map, so
    # the test also pins average-rank handling rather than assuming strict ranks.
    member_channel_importance = np.asarray(
        [
            [5.0, 4.0, 3.0, 2.0],
            [5.0, 4.0, 2.0, 3.0],
            [4.0, 5.0, 2.0, 3.0],
            [1.0, 2.0, 4.0, 5.0],
        ]
    )
    validation_r2 = np.asarray([0.99, 0.98, 0.97, 0.96])

    first = validation_stability_correlation(member_channel_importance, validation_r2)
    second = validation_stability_correlation(member_channel_importance, validation_r2)

    np.testing.assert_array_equal(first.stability, second.stability)
    assert first.spearman_rho == second.spearman_rho
    assert np.isclose(first.spearman_rho, 0.6324555320336759)
    assert first.metric == "spearman_validation_r2_vs_median_map_rank_agreement"
