from __future__ import annotations

import numpy as np
import torch

from itg_nn.xai.concepts import (
    canonical_output_from_layer,
    grouped_nested_sparse_probe,
    invariant_layer_maps,
    matched_extremes,
    representation_direction_use,
)
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.symmetry import InvariantMember, circular_shift


def _cyclic_fixture(seed: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(48), 2)
    phase = rng.uniform(0, 2 * np.pi, size=48)[groups]
    z = 2 * np.pi * np.arange(96) / 96
    signal = np.cos(z[None, :] - phase[:, None])
    nuisance = rng.normal(size=(len(groups), 3))
    # Shift-invariant known concept: amplitude of Fourier mode 1.
    amplitude = rng.uniform(0.3, 2.0, size=48)[groups]
    traces = amplitude[:, None] * signal + 0.05 * rng.normal(size=signal.shape)
    representation = np.column_stack(
        (np.mean(traces * signal, axis=1), nuisance, rng.normal(size=len(groups)))
    )
    return representation, amplitude, groups


def test_nested_sparse_probe_recovers_cyclic_concept_and_permutation_null() -> None:
    representation, target, groups = _cyclic_fixture()
    fitted = grouped_nested_sparse_probe(
        representation,
        target,
        groups,
        outer_folds=4,
        inner_folds=3,
        penalties=(1e-4, 1e-2, 1e-1),
        seed=17,
    )
    permuted = grouped_nested_sparse_probe(
        representation,
        target,
        groups,
        outer_folds=4,
        inner_folds=3,
        penalties=(1e-4, 1e-2, 1e-1),
        seed=17,
        permute_target=True,
    )
    assert fitted.held_out_r2 > 0.9
    assert permuted.held_out_r2 < 0.2
    assert fitted.nonzero_fraction < 0.8
    np.testing.assert_allclose(fitted.predictions, grouped_nested_sparse_probe(
        representation, target, groups, outer_folds=4, inner_folds=3,
        penalties=(1e-4, 1e-2, 1e-1), seed=17,
    ).predictions)


def test_grouped_folds_never_split_an_equilibrium() -> None:
    representation, target, groups = _cyclic_fixture()
    result = grouped_nested_sparse_probe(
        representation, target, groups, outer_folds=4, inner_folds=2,
        penalties=(1e-3, 1e-2), seed=4,
    )
    for fold in np.unique(result.fold):
        test_groups = set(groups[result.fold == fold])
        train_groups = set(groups[result.fold != fold])
        assert test_groups.isdisjoint(train_groups)


def test_matched_extremes_balance_nuisances_and_keep_groups_disjoint() -> None:
    rng = np.random.default_rng(3)
    groups = np.arange(80)
    concept = np.linspace(-2, 2, 80)
    nuisance = np.column_stack((np.repeat(np.arange(4), 20), rng.normal(size=80)))
    match = matched_extremes(concept, nuisance, groups, fraction=0.25, seed=8)
    assert len(match.high) == len(match.low)
    assert set(groups[match.high]).isdisjoint(groups[match.low])
    assert set(nuisance[match.high, 0]) == set(nuisance[match.low, 0])
    assert match.validity_tag == "observed-comparison"


def test_matched_extremes_residualize_a_strong_continuous_confounder() -> None:
    rng = np.random.default_rng(31)
    groups = np.arange(400)
    drive = rng.normal(size=400)
    geometry_scale = rng.normal(size=400)
    nuisance = np.column_stack((np.repeat(np.arange(4), 100), drive, geometry_scale))
    concept = 4.0 * drive - 2.0 * geometry_scale + rng.normal(size=400)
    match = matched_extremes(concept, nuisance, groups, fraction=0.2, seed=11)
    for column in (1, 2):
        high = nuisance[match.high, column]
        low = nuisance[match.low, column]
        pooled = np.sqrt((high.var() + low.var()) / 2)
        assert abs((high.mean() - low.mean()) / pooled) < 0.2


def test_directional_use_explains_native_output_and_beats_random_controls() -> None:
    torch.manual_seed(2)
    representation = torch.randn(64, 5)
    concept_direction = torch.tensor([1.0, 0, 0, 0, 0])

    def native_output(values: torch.Tensor) -> torch.Tensor:
        # This is already the native scalar; exponentiating would change the derivative.
        return 2.5 * values[:, 0] - 0.2 * values[:, 1]

    use = representation_direction_use(
        native_output,
        representation,
        concept_direction,
        random_directions=32,
        intervention_scale=0.2,
        seed=12,
    )
    assert use.mean_directional_derivative == np.float64(2.5)
    assert use.intervention_rms > 2.0 * use.random_intervention_rms_median
    assert use.validity_tag == "deliberately_off_manifold_diagnostic"


def test_canonical_layer_representations_are_shift_invariant_and_continue_exactly() -> None:
    ensemble = load_ensemble("models/cyclic_ensemble_pre2.pt", device="cpu")
    member = InvariantMember(ensemble.models[0])
    torch.manual_seed(13)
    geometry = torch.randn(3, 96, 7)
    a_over_lt = torch.tensor([2.0, 3.0, 4.0])
    a_over_ln = torch.tensor([0.0, 0.5, 1.0])
    maps = invariant_layer_maps(member, geometry)
    shifted = invariant_layer_maps(member, circular_shift(geometry, 17))
    for base, moved in zip(maps, shifted):
        torch.testing.assert_close(base.mean(-1), moved.mean(-1), atol=2e-5, rtol=2e-5)
    continued = canonical_output_from_layer(
        member, 2, maps[2], a_over_lt, a_over_ln
    )
    torch.testing.assert_close(continued, member.invariant(geometry, a_over_lt, a_over_ln))
