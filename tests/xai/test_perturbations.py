from __future__ import annotations

import numpy as np
import pytest
import torch

from itg_nn.xai.perturbations import (
    ReferenceBackgrounds,
    RobustPCASupport,
    attenuate_fourier_band,
    block_permutation,
    independent_channel_shifts,
    joint_permutation,
    low_pass,
    phase_scramble,
    replace_channel,
    robust_constant_profile,
    wrapped_window_mask,
)
from itg_nn.xai.toys import ColocationToy, FourierBandToy, PeriodicPermutationToy


def _geometry(samples: int = 8) -> torch.Tensor:
    return torch.randn(samples, 96, 7, generator=torch.Generator().manual_seed(12))


def test_wrapped_windows_have_constant_support_and_no_boundary_artifact() -> None:
    first = wrapped_window_mask(96, start=92, length=11)
    shifted = wrapped_window_mask(96, start=3, length=11)
    assert first.sum() == shifted.sum() == 11
    assert torch.equal(torch.roll(first, shifts=7), shifted)


def test_seeded_random_operators_are_deterministic_and_preserve_registered_marginals() -> None:
    geometry = _geometry()
    for operator in (
        lambda value: joint_permutation(value, seed=4),
        lambda value: block_permutation(value, 8, seed=4),
        lambda value: independent_channel_shifts(value, seed=4),
        lambda value: phase_scramble(value, seed=4, independent_channels=True),
    ):
        assert torch.equal(operator(geometry), operator(geometry))

    permuted = joint_permutation(geometry, seed=4)
    torch.testing.assert_close(
        geometry.sort(dim=1).values, permuted.sort(dim=1).values, rtol=0, atol=0
    )
    scrambled = phase_scramble(geometry, seed=4, independent_channels=True)
    torch.testing.assert_close(
        torch.fft.rfft(geometry, dim=1).abs(),
        torch.fft.rfft(scrambled, dim=1).abs(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_toys_rank_relevant_structure_above_matched_controls() -> None:
    geometry = _geometry(24)
    gradients = torch.zeros(len(geometry))
    permutation_toy = PeriodicPermutationToy()
    reference = permutation_toy(geometry, gradients, gradients)
    joint = permutation_toy(joint_permutation(geometry, seed=8), gradients, gradients)
    independent = permutation_toy(
        independent_channel_shifts(geometry, seed=8), gradients, gradients
    )
    assert torch.sqrt(torch.mean(torch.square(joint - reference))) < 1e-5
    assert torch.sqrt(torch.mean(torch.square(independent - reference))) > 0.1

    blocked = permutation_toy(
        block_permutation(geometry, 8, seed=8), gradients, gradients
    )
    torch.testing.assert_close(reference, blocked, rtol=1e-6, atol=1e-6)

    colocation = ColocationToy()
    reference = colocation(geometry, gradients, gradients)
    joint = colocation(joint_permutation(geometry, seed=9), gradients, gradients)
    independent = colocation(
        independent_channel_shifts(geometry, seed=9), gradients, gradients
    )
    assert torch.sqrt(torch.mean(torch.square(joint - reference))) < 1e-5
    assert torch.sqrt(torch.mean(torch.square(independent - reference))) > 0.1

    signal = torch.sin(2 * torch.pi * 3 * torch.arange(96) / 96)
    band_geometry = torch.zeros(4, 96, 7)
    band_geometry[:, :, 4] = signal
    fourier = FourierBandToy(channel=4, band=3)
    reference = fourier(band_geometry, torch.zeros(4), torch.zeros(4))
    relevant = fourier(
        attenuate_fourier_band(
            band_geometry, minimum_frequency=3, maximum_frequency=3, dose=1
        ),
        torch.zeros(4),
        torch.zeros(4),
    )
    control = fourier(
        attenuate_fourier_band(
            band_geometry, minimum_frequency=8, maximum_frequency=12, dose=1
        ),
        torch.zeros(4),
        torch.zeros(4),
    )
    assert torch.mean(torch.abs(reference - relevant)) > 10
    torch.testing.assert_close(reference, control, rtol=1e-6, atol=1e-6)


def test_reference_backgrounds_are_nonzero_matched_and_channel_local() -> None:
    geometry = _geometry(10)
    geometry[:, :, 0] += 4
    geometry[:, :, 4] += 8
    gradients = np.column_stack((np.arange(10), np.arange(10) / 2))
    classes = np.repeat((0, 1), 5)
    rows = np.arange(10)
    backgrounds = ReferenceBackgrounds(geometry, gradients, classes, rows)
    constant = backgrounds.constant()
    assert constant.shape == (1, 96, 7)
    assert torch.all(constant[:, :, 0] > 0)
    matched = backgrounds.matched_observed(
        gradients[:2], classes[:2], source_row_ids=rows[:2]
    )
    assert matched.shape == (2, 96, 7)
    assert not torch.equal(matched[0], geometry[0])
    profile = backgrounds.conditional_channel_profile(
        4, gradients[:2], classes[:2], neighbours=2, source_row_ids=rows[:2]
    )
    replaced = replace_channel(geometry[:2], 4, profile)
    torch.testing.assert_close(replaced[:, :, :4], geometry[:2, :, :4])
    torch.testing.assert_close(replaced[:, :, 5:], geometry[:2, :, 5:])
    assert not torch.equal(replaced[:, :, 4], geometry[:2, :, 4])


def test_low_pass_and_support_score_behave_analytically() -> None:
    geometry = _geometry(40).numpy()
    fit = geometry[:24]
    heldout = geometry[24:32]
    support = RobustPCASupport.fit(fit, heldout, components=6)
    in_support = support.score(geometry[32:36])
    far = support.score(geometry[32:36] + 100)
    assert np.median(far["warning_score"]) > np.median(in_support["warning_score"])
    assert np.all((far["warning_score"] >= 0) & (far["warning_score"] <= 1))
    shifted = np.roll(geometry[32:36], shift=17, axis=1)
    original_score = support.score(geometry[32:36])
    shifted_score = support.score(shifted)
    np.testing.assert_allclose(
        original_score["reconstruction_rms"], shifted_score["reconstruction_rms"],
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        original_score["nearest_distance"], shifted_score["nearest_distance"],
        rtol=1e-12, atol=1e-12,
    )

    values = torch.zeros(1, 96, 7)
    z = torch.arange(96)
    values[0, :, 0] = torch.sin(2 * torch.pi * 2 * z / 96)
    values[0, :, 1] = torch.sin(2 * torch.pi * 20 * z / 96)
    filtered = low_pass(values, 4)
    torch.testing.assert_close(filtered[:, :, 0], values[:, :, 0], rtol=1e-5, atol=1e-5)
    assert filtered[:, :, 1].abs().max() < 1e-5


def test_invalid_operator_arguments_fail_loudly() -> None:
    geometry = _geometry(2)
    with pytest.raises(ValueError):
        block_permutation(geometry, 5, seed=1)
    with pytest.raises(ValueError):
        wrapped_window_mask(96, start=0, length=97)
    with pytest.raises(ValueError):
        replace_channel(geometry, 8, torch.zeros(2, 96))
