from __future__ import annotations

import numpy as np

from itg_nn.xai.cross_model import (
    HIDDEN_INTERVENTION_VALIDITY,
    functional_similarity,
    grouped_bootstrap_match_recurrence,
    grouped_bootstrap_cka,
    linear_cka,
    match_units,
    mean_replacement_effects,
    member_distance_matrix,
    residualize_against_covariates,
)
from itg_nn.xai.cross_model import _group_bootstrap_row_weights


def _cyclic_toy(seed: int = 4):
    rng = np.random.default_rng(seed)
    geometry = rng.normal(size=(72, 96, 3))
    # Closed-form cyclic invariants: neither changes under a joint roll.
    first = geometry[:, :, 0].mean(axis=1)
    second = np.square(geometry[:, :, 1]).mean(axis=1)
    nuisance = rng.normal(size=len(geometry))
    left = np.column_stack((first, second))
    right = np.column_stack((2.0 * second + 0.2, 3.0 * first - 0.4, nuisance))
    groups = np.repeat(np.arange(24), 3)
    target = 2.5 * first - 0.75 * second
    return geometry, left, right, groups, target


def test_linear_cka_recovers_rotated_representation_and_null_control():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(80, 4))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    assert linear_cka(values, values @ rotation) > 1.0 - 1e-12
    assert linear_cka(values, rng.normal(size=(80, 4))) < 0.15


def test_cyclic_toy_matching_recovers_permutation_and_leaves_null_unmatched():
    geometry, left, right, _, target = _cyclic_toy()
    rolled = np.roll(geometry, 17, axis=1)
    assert np.allclose(geometry[:, :, 0].mean(1), rolled[:, :, 0].mean(1))
    left_aux = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    right_aux = np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, -1.0]])
    left_effect = np.asarray([[2.5, 1.5], [-0.75, 0.4]])
    right_effect = np.asarray([[-0.75, 0.4], [2.5, 1.5], [0.0, 0.0]])
    score, pieces = functional_similarity(
        left,
        right,
        covariates=target[:, None],
        left_auxiliary=left_aux,
        right_auxiliary=right_aux,
        left_effects=left_effect,
        right_effects=right_effect,
        component_weights=(0.35, 0.25, 0.2, 0.2),
    )
    matches = match_units(score, minimum_similarity=0.7)
    assert {(left_i, right_i) for left_i, right_i, _ in matches} == {(0, 1), (1, 0)}
    assert all(right_i != 2 for _, right_i, _ in matches)
    assert pieces["activation_residual"][0, 1] > 0.99
    assert pieces["activation_residual"][1, 0] > 0.99


def test_flux_residualization_rejects_label_only_correspondence():
    rng = np.random.default_rng(7)
    flux = rng.normal(size=120)
    left = flux[:, None] + 0.01 * rng.normal(size=(120, 1))
    right = flux[:, None] + 0.01 * rng.normal(size=(120, 1))
    residual_left = residualize_against_covariates(left, flux[:, None])
    residual_right = residualize_against_covariates(right, flux[:, None])
    assert np.corrcoef(left[:, 0], right[:, 0])[0, 1] > 0.99
    assert abs(np.corrcoef(residual_left[:, 0], residual_right[:, 0])[0, 1]) < 0.25
    score, pieces = functional_similarity(
        left,
        right,
        covariates=flux[:, None],
        left_auxiliary=np.ones((1, 1)),
        right_auxiliary=np.ones((1, 1)),
        left_effects=np.ones((1, 1)),
        right_effects=np.ones((1, 1)),
        component_weights=(0.0, 1.0, 0.0, 0.0),
    )
    assert pieces["activation_raw"][0, 0] > 0.99
    assert pieces["activation_residual"][0, 0] < 0.25
    assert score[0, 0] == pieces["activation_residual"][0, 0]


def test_grouped_bootstrap_recurs_by_equilibrium_and_is_deterministic():
    _, left, right, groups, target = _cyclic_toy()
    kwargs = dict(
        groups=groups,
        covariates=target[:, None],
        left_auxiliary=np.eye(2),
        right_auxiliary=np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, -1.0]]),
        left_effects=np.asarray([[2.5], [-0.75]]),
        right_effects=np.asarray([[-0.75], [2.5], [0.0]]),
        component_weights=(0.35, 0.25, 0.2, 0.2),
        minimum_similarity=0.7,
        replicates=80,
        seed=9,
    )
    first = grouped_bootstrap_match_recurrence(left, right, **kwargs)
    second_run = grouped_bootstrap_match_recurrence(left, right, **kwargs)
    assert np.array_equal(first, second_run)
    assert first[0, 1] > 0.9 and first[1, 0] > 0.9
    assert np.all(first[:, 2] < 0.1)


def test_grouped_bootstrap_residualization_rejects_label_only_recurrence():
    rng = np.random.default_rng(71)
    flux = rng.normal(size=120)
    left = flux[:, None] + 0.05 * rng.normal(size=(120, 1))
    right = flux[:, None] + 0.05 * rng.normal(size=(120, 1))
    recurrence = grouped_bootstrap_match_recurrence(
        left,
        right,
        groups=np.repeat(np.arange(40), 3),
        covariates=flux[:, None],
        left_auxiliary=np.ones((1, 1)),
        right_auxiliary=np.ones((1, 1)),
        left_effects=np.ones((1, 1)),
        right_effects=np.ones((1, 1)),
        component_weights=(0.0, 1.0, 0.0, 0.0),
        minimum_similarity=0.7,
        replicates=60,
        seed=17,
    )
    assert recurrence[0, 0] < 0.1


def test_group_bootstrap_gives_sibling_tubes_identical_multiplicity():
    groups = np.asarray(["eq0", "eq0", "eq1", "eq2", "eq2", "eq2"])
    weights = _group_bootstrap_row_weights(groups, np.random.default_rng(5))
    assert weights[0] == weights[1]
    assert weights[3] == weights[4] == weights[5]
    assert weights.sum() > 0


def test_native_signed_ablation_is_not_exponentiated_and_keeps_regimes():
    _, left, _, _, _ = _cyclic_toy()

    def native_head(values: np.ndarray) -> np.ndarray:
        return np.maximum(2.5 * values[:, 0] - 0.75 * values[:, 1], -2.0)

    effects = mean_replacement_effects(left, native_head)
    expected = native_head(left) - native_head(
        np.column_stack((np.full(len(left), left[:, 0].mean()), left[:, 1]))
    )
    assert effects.shape == (len(left), 2)
    assert np.allclose(effects[:, 0], expected)
    assert not np.allclose(effects[:, 0], np.exp(native_head(left)) - np.exp(native_head(left) - expected))
    assert HIDDEN_INTERVENTION_VALIDITY == "deliberately_off_manifold_diagnostic"


def test_member_distance_uses_each_registered_evidence_block():
    predictions = np.asarray([[0.0, 1.0], [0.0, 1.1], [2.0, -1.0]])
    causal = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 1.0]])
    concepts = np.asarray([[0.4], [0.45], [-0.6]])
    distance = member_distance_matrix((predictions, causal, concepts))
    assert distance.shape == (3, 3)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert distance[0, 1] < distance[0, 2]


def test_member_distance_is_invariant_to_each_block_natural_scale():
    small = np.asarray([[0.0], [1.0], [2.0]])
    large = 1000.0 * small[:, ::-1]
    registered = member_distance_matrix((small, large))
    rescaled = member_distance_matrix((small, 0.001 * large))
    assert np.allclose(registered, rescaled)


def test_grouped_cka_bootstrap_keeps_rotated_pair_and_null_separate():
    rng = np.random.default_rng(33)
    base = rng.normal(size=(60, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    representations = (base, base @ rotation, rng.normal(size=(60, 3)))
    point, lower, upper = grouped_bootstrap_cka(
        representations,
        groups=np.repeat(np.arange(20), 3),
        replicates=40,
        seed=12,
    )
    assert np.allclose(np.diag(point), 1.0)
    assert point[0, 1] > 1.0 - 1e-12
    assert lower[0, 1] > 1.0 - 1e-12
    assert upper[0, 2] < 0.5
