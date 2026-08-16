"""Analytic periodic controls with known relevant features for XAI validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ToyExpectation:
    """Known input channels/positions that should outrank controls."""

    channels: tuple[int, ...]
    positions: tuple[int, ...] = ()
    fourier_bands: tuple[int, ...] = ()


class PeriodicWindowToy(nn.Module):
    """A regressor depending only on one wrapped channel window."""

    expectation: ToyExpectation

    def __init__(self, *, channel: int = 2, start: int = 92, width: int = 8) -> None:
        super().__init__()
        if not 0 <= channel < 7 or width < 1:
            raise ValueError("invalid toy channel or window width")
        self.channel = channel
        self.positions = tuple((start + offset) % 96 for offset in range(width))
        self.expectation = ToyExpectation(channels=(channel,), positions=self.positions)

    def forward(
        self, geometry: torch.Tensor, a_over_lt: torch.Tensor, a_over_ln: torch.Tensor
    ) -> torch.Tensor:
        del a_over_lt, a_over_ln
        return geometry[:, self.positions, self.channel].mean(dim=1, keepdim=True)


class ColocationToy(nn.Module):
    """A periodic regressor whose signal requires two channels to co-locate."""

    expectation = ToyExpectation(channels=(1, 5))

    def forward(
        self, geometry: torch.Tensor, a_over_lt: torch.Tensor, a_over_ln: torch.Tensor
    ) -> torch.Tensor:
        del a_over_lt, a_over_ln
        return (geometry[:, :, 1] * geometry[:, :, 5]).mean(dim=1, keepdim=True)


class FourierBandToy(nn.Module):
    """A regressor depending on the magnitude of a selected periodic Fourier band."""

    def __init__(self, *, channel: int = 4, band: int = 3) -> None:
        super().__init__()
        if not 0 <= channel < 7 or not 1 <= band <= 48:
            raise ValueError("invalid toy channel or Fourier band")
        self.channel = channel
        self.band = band
        self.expectation = ToyExpectation(channels=(channel,), fourier_bands=(band,))

    def forward(
        self, geometry: torch.Tensor, a_over_lt: torch.Tensor, a_over_ln: torch.Tensor
    ) -> torch.Tensor:
        del a_over_lt, a_over_ln
        spectrum = torch.fft.rfft(geometry[:, :, self.channel], dim=1)
        return spectrum[:, self.band].abs().square().unsqueeze(1) / 96


class PeriodicPermutationToy(nn.Module):
    """A regressor invariant to any joint permutation of position vectors.

    Independent channel permutations generally change the result, making this a
    control for separating use of spatial order from use of cross-channel
    co-location in S03's perturbation ladder.
    """

    expectation = ToyExpectation(channels=(1, 4, 6))

    def forward(
        self, geometry: torch.Tensor, a_over_lt: torch.Tensor, a_over_ln: torch.Tensor
    ) -> torch.Tensor:
        del a_over_lt, a_over_ln
        pointwise = geometry[:, :, 1] * geometry[:, :, 4] + geometry[:, :, 6].square()
        return pointwise.mean(dim=1, keepdim=True)
