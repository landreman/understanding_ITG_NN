"""Exact cyclic symmetrisation and full-resolution bottleneck densities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from itg_nn.model import CyclicInvariantNet, N_Z_POINTS


N_POOLING_PHASES = 32
PARITY_ODD_CHANNELS = (3, 5)
CANONICAL_FUNCTION = "invariant_tilde_f"


def circular_shift(geometry: torch.Tensor, shift: int) -> torch.Tensor:
    """Apply ``S_shift`` on the parallel-position axis."""

    return torch.roll(geometry, shifts=int(shift), dims=1)


def reverse_parallel(geometry: torch.Tensor) -> torch.Tensor:
    """Return the grid-anchored periodic reversal ``z -> -z``."""

    indices = torch.remainder(
        -torch.arange(geometry.shape[1], device=geometry.device), geometry.shape[1]
    )
    return geometry.index_select(1, indices)


def stellarator_parity(geometry: torch.Tensor) -> torch.Tensor:
    """Reverse ``z`` and change sign on the two parity-odd geometry channels."""

    transformed = reverse_parallel(geometry).clone()
    transformed[:, :, PARITY_ODD_CHANNELS] *= -1
    return transformed


def _circular_same_convolution(
    values: torch.Tensor, convolution: nn.Conv1d, dilation: int
) -> torch.Tensor:
    """Apply a trained cross-correlation with circular SAME padding and dilation.

    Explicit modular indices support padding wider than the 96-point input at
    late dilated layers.  PyTorch's circular ``pad`` rejects such repeated wraps.
    """

    kernel = int(convolution.kernel_size[0])
    total_padding = dilation * (kernel - 1)
    # Match Conv1d(padding="same") on the coarser grid exactly.  For an even
    # kernel the trained layer puts the extra cell on the right *before* the
    # grid spacing is expanded by dilation; splitting total_padding directly
    # would translate the full-resolution density by dilation / 2.
    left_padding = dilation * ((kernel - 1) // 2)
    right_padding = total_padding - left_padding
    indices = torch.arange(
        -left_padding,
        values.shape[-1] + right_padding,
        device=values.device,
    ).remainder(values.shape[-1])
    padded = values.index_select(-1, indices)
    return F.conv1d(
        padded,
        convolution.weight,
        convolution.bias,
        stride=1,
        padding=0,
        dilation=dilation,
        groups=convolution.groups,
    )


class InvariantMember(nn.Module):
    """Register ``f``, shift-averaged ``bar_f``, ``tilde_f``, and ``rho``.

    ``rho`` is evaluated with a stride-one à trous chain.  Each layer retains
    all 96 pooling phases by dilating both its convolution and its two-point max
    pool by the cumulative stride.  Its position mean is therefore the exact
    32-phase bottleneck average used by ``tilde_f``.
    """

    def __init__(self, model: CyclicInvariantNet) -> None:
        super().__init__()
        self.model = model

    def bottleneck(self, geometry: torch.Tensor) -> torch.Tensor:
        """Return the original member bottleneck after strided pooling."""

        return self.bottleneck_map(geometry).mean(dim=-1)

    def bottleneck_map(self, geometry: torch.Tensor) -> torch.Tensor:
        """Return the trained stride-32 pre-GAP map with three positions."""

        hidden = geometry.transpose(1, 2)
        for convolution, pooling in zip(self.model.conv_layers, self.model.pool_layers):
            hidden = pooling(F.relu(convolution(hidden)))
        return hidden

    def head(
        self,
        bottleneck: torch.Tensor,
        a_over_lt: torch.Tensor,
        a_over_ln: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the trained MLP head to a supplied bottleneck."""

        hidden = torch.cat(
            (
                bottleneck,
                a_over_lt.reshape(-1, 1),
                a_over_ln.reshape(-1, 1),
            ),
            dim=1,
        )
        for dense_layer in self.model.fc_layers:
            hidden = F.relu(dense_layer(hidden))
        return self.model.output_layer(hidden).squeeze(-1)

    def original(
        self,
        geometry: torch.Tensor,
        a_over_lt: torch.Tensor,
        a_over_ln: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the trained member's native clipped-log output ``f``."""

        return self.model(geometry, a_over_lt, a_over_ln).squeeze(-1)

    def equivariant_density(self, geometry: torch.Tensor) -> torch.Tensor:
        """Return full-resolution ``rho`` with shape ``(sample, unit, z)``."""

        if geometry.ndim != 3 or geometry.shape[1] != N_Z_POINTS:
            raise ValueError("geometry must have shape (sample, 96, channel)")
        hidden = geometry.transpose(1, 2)
        dilation = 1
        for convolution in self.model.conv_layers:
            hidden = F.relu(_circular_same_convolution(hidden, convolution, dilation))
            hidden = torch.maximum(hidden, torch.roll(hidden, shifts=-dilation, dims=-1))
            dilation *= 2
        return hidden

    def invariant_bottleneck(self, geometry: torch.Tensor) -> torch.Tensor:
        """Return ``bar_u = mean_z rho``."""

        return self.equivariant_density(geometry).mean(dim=-1)

    def invariant(
        self,
        geometry: torch.Tensor,
        a_over_lt: torch.Tensor,
        a_over_ln: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate ``tilde_f = MLP(bar_u, a/L_T, a/L_n)``."""

        return self.head(self.invariant_bottleneck(geometry), a_over_lt, a_over_ln)

    def forward(
        self,
        geometry: torch.Tensor,
        a_over_lt: torch.Tensor,
        a_over_ln: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the researcher-confirmed canonical function ``tilde_f``."""

        return self.invariant(geometry, a_over_lt, a_over_ln)

    def phase_outputs(
        self,
        geometry: torch.Tensor,
        a_over_lt: torch.Tensor,
        a_over_ln: torch.Tensor,
        phases: Iterable[int] = range(N_POOLING_PHASES),
    ) -> torch.Tensor:
        """Evaluate native outputs at shifts, returning ``(sample, phase)``."""

        outputs = [
            self.original(circular_shift(geometry, phase), a_over_lt, a_over_ln)
            for phase in phases
        ]
        return torch.stack(outputs, dim=1)

    def shift_averaged(
        self,
        geometry: torch.Tensor,
        a_over_lt: torch.Tensor,
        a_over_ln: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate ``bar_f`` over the 32 distinct pooling phases."""

        return self.phase_outputs(geometry, a_over_lt, a_over_ln).mean(dim=1)


@dataclass(frozen=True)
class ReceptiveFieldBlock:
    """Exact structural receptive-field geometry after one conv/pool block."""

    block: int
    kernel_size: int
    output_stride: int
    formal_span: int
    left_extent: int
    right_extent: int
    center_offset: float
    unique_periodic_positions: int
    wraps: bool
    globally_connected: bool


def receptive_field_blocks(
    kernel_sizes: Sequence[int], grid_size: int = N_Z_POINTS
) -> tuple[ReceptiveFieldBlock, ...]:
    """Compute formal and unique periodic receptive fields, including asymmetry."""

    offsets = {0}
    stride = 1
    rows: list[ReceptiveFieldBlock] = []
    for block, kernel_size in enumerate(kernel_sizes, start=1):
        left = (int(kernel_size) - 1) // 2
        convolution_offsets = {
            stride * (tap - left) for tap in range(int(kernel_size))
        }
        block_offsets = convolution_offsets | {
            value + stride for value in convolution_offsets
        }
        offsets = {old + new for old in offsets for new in block_offsets}
        minimum = min(offsets)
        maximum = max(offsets)
        unique = len({value % grid_size for value in offsets})
        stride *= 2
        rows.append(
            ReceptiveFieldBlock(
                block=block,
                kernel_size=int(kernel_size),
                output_stride=stride,
                formal_span=maximum - minimum + 1,
                left_extent=-minimum,
                right_extent=maximum,
                center_offset=(minimum + maximum) / 2,
                unique_periodic_positions=unique,
                wraps=(maximum - minimum + 1) > grid_size,
                globally_connected=unique == grid_size,
            )
        )
    return tuple(rows)


def normalized_parity_mismatch(geometry: np.ndarray) -> np.ndarray:
    """Return per-channel MSE(transform, original) divided by channel variance."""

    values = torch.as_tensor(np.asarray(geometry), dtype=torch.float64)
    transformed = stellarator_parity(values).numpy()
    difference_mse = np.mean(np.square(transformed - np.asarray(geometry)), axis=(0, 1))
    variance = np.var(np.asarray(geometry), axis=(0, 1))
    return difference_mse / np.maximum(variance, np.finfo(np.float64).tiny)
