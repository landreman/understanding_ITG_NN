from __future__ import annotations

import numpy as np
import pytest
import torch

from itg_nn.model import Architecture, CyclicInvariantNet
from itg_nn.xai.symmetry import (
    CANONICAL_FUNCTION,
    InvariantMember,
    circular_shift,
    receptive_field_blocks,
    reverse_parallel,
    stellarator_parity,
)


@pytest.mark.parametrize("kernel_sizes", [(3, 3, 3, 3, 3), (4, 5, 2, 3, 4)])
def test_atrous_density_matches_phase_average_and_is_equivariant(kernel_sizes) -> None:
    generator = torch.Generator().manual_seed(19)
    model = CyclicInvariantNet(
        Architecture(
            kernel_sizes=kernel_sizes,
            convolution_channels=(4, 5, 6, 7, 8),
            dense_dimensions=(9, 10),
        )
    ).eval()
    member = InvariantMember(model)
    geometry = torch.randn(2, 96, 7, generator=generator)
    gradients = torch.randn(2, generator=generator)

    phase_bottlenecks = torch.stack(
        [member.bottleneck(circular_shift(geometry, phase)) for phase in range(32)]
    ).mean(dim=0)
    density = member.equivariant_density(geometry)
    torch.testing.assert_close(density.mean(dim=-1), phase_bottlenecks, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        member.invariant(geometry, gradients, gradients),
        member.head(phase_bottlenecks, gradients, gradients),
        rtol=2e-6,
        atol=2e-6,
    )
    assert CANONICAL_FUNCTION == "invariant_tilde_f"
    torch.testing.assert_close(
        member(geometry, gradients, gradients),
        member.invariant(geometry, gradients, gradients),
        rtol=0,
        atol=0,
    )
    shifted_density = member.equivariant_density(circular_shift(geometry, 7))
    torch.testing.assert_close(
        shifted_density,
        torch.roll(density, shifts=7, dims=-1),
        rtol=2e-6,
        atol=2e-6,
    )


def test_exact_pooling_subgroup_and_32_equals_96_phase_average() -> None:
    generator = torch.Generator().manual_seed(2)
    model = CyclicInvariantNet(
        Architecture(
            kernel_sizes=(4, 5, 2, 3, 4),
            convolution_channels=(4, 5, 6, 7, 8),
            dense_dimensions=(9, 10),
        )
    ).eval()
    member = InvariantMember(model)
    geometry = torch.randn(3, 96, 7, generator=generator)
    gradient = torch.randn(3, generator=generator)
    outputs = member.phase_outputs(geometry, gradient, gradient, range(96))
    torch.testing.assert_close(outputs[:, 0], outputs[:, 32], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(outputs[:, 0], outputs[:, 64], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        outputs[:, :32].mean(1), outputs.mean(1), rtol=2e-6, atol=2e-6
    )


def test_parity_is_an_involution_with_expected_odd_channels() -> None:
    values = torch.arange(96 * 7, dtype=torch.float32).reshape(1, 96, 7)
    torch.testing.assert_close(reverse_parallel(reverse_parallel(values)), values)
    torch.testing.assert_close(stellarator_parity(stellarator_parity(values)), values)
    transformed = stellarator_parity(values)
    reversed_values = reverse_parallel(values)
    torch.testing.assert_close(transformed[:, :, (0, 1, 2, 4, 6)], reversed_values[:, :, (0, 1, 2, 4, 6)])
    torch.testing.assert_close(transformed[:, :, (3, 5)], -reversed_values[:, :, (3, 5)])


def test_receptive_fields_include_even_kernel_offset_and_periodic_saturation() -> None:
    rows = receptive_field_blocks((13, 5, 3, 8, 5))
    assert [row.formal_span for row in rows] == [14, 24, 36, 100, 180]
    assert rows[0].center_offset == pytest.approx(0.5)
    assert rows[3].wraps
    assert rows[3].globally_connected
    assert rows[3].unique_periodic_positions == 96
    assert np.all(np.diff([row.unique_periodic_positions for row in rows]) >= 0)
