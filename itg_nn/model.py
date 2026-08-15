"""Neural-network architecture used by the inference ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


N_GEOMETRY_FEATURES = 7
N_Z_POINTS = 96
N_CONVOLUTION_LAYERS = 5
N_DENSE_LAYERS = 2


@dataclass(frozen=True)
class Architecture:
    """The dimensions that vary among members of the trained ensemble."""

    kernel_sizes: tuple[int, int, int, int, int]
    convolution_channels: tuple[int, int, int, int, int]
    dense_dimensions: tuple[int, int]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Architecture:
        """Create an architecture from checkpoint metadata."""

        return cls(
            kernel_sizes=tuple(int(value) for value in values["kernel_sizes"]),
            convolution_channels=tuple(
                int(value) for value in values["convolution_channels"]
            ),
            dense_dimensions=tuple(
                int(value) for value in values["dense_dimensions"]
            ),
        )

    def as_dict(self) -> dict[str, list[int]]:
        """Return serialization-safe checkpoint metadata."""

        return {
            "kernel_sizes": list(self.kernel_sizes),
            "convolution_channels": list(self.convolution_channels),
            "dense_dimensions": list(self.dense_dimensions),
        }

    def __post_init__(self) -> None:
        if len(self.kernel_sizes) != N_CONVOLUTION_LAYERS:
            raise ValueError("Exactly five convolution kernel sizes are required")
        if len(self.convolution_channels) != N_CONVOLUTION_LAYERS:
            raise ValueError("Exactly five convolution channel counts are required")
        if len(self.dense_dimensions) != N_DENSE_LAYERS:
            raise ValueError("Exactly two dense-layer dimensions are required")
        if any(value <= 0 for value in (*self.kernel_sizes, *self.convolution_channels)):
            raise ValueError("Kernel sizes and channel counts must be positive")
        if any(value <= 0 for value in self.dense_dimensions):
            raise ValueError("Dense-layer dimensions must be positive")


class CyclicInvariantNet(nn.Module):
    """Circular-convolution network for geometry and two scalar gradients.

    This is the inference-only form of the trained architecture. All selected
    ensemble members omit batch normalization and use the same seven geometry
    channels, five convolution/pooling stages, two scalar gradients, and two
    dense layers. Training-only switches are intentionally absent.
    """

    def __init__(self, architecture: Architecture) -> None:
        super().__init__()
        channels = (N_GEOMETRY_FEATURES, *architecture.convolution_channels)

        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                channels[index],
                channels[index + 1],
                kernel_size=architecture.kernel_sizes[index],
                padding="same",
                padding_mode="circular",
            )
            for index in range(N_CONVOLUTION_LAYERS)
        )
        self.pool_layers = nn.ModuleList(
            nn.MaxPool1d(kernel_size=2, stride=2)
            for _ in range(N_CONVOLUTION_LAYERS)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        dense_dimensions = (
            architecture.convolution_channels[-1] + 2,
            *architecture.dense_dimensions,
        )
        self.fc_layers = nn.ModuleList(
            nn.Linear(dense_dimensions[index], dense_dimensions[index + 1])
            for index in range(N_DENSE_LAYERS)
        )
        self.output_layer = nn.Linear(architecture.dense_dimensions[-1], 1)

    def forward(
        self,
        geometry: torch.Tensor,
        a_over_lt: torch.Tensor,
        a_over_ln: torch.Tensor,
    ) -> torch.Tensor:
        """Predict clipped log heat flux for a batch.

        ``geometry`` has shape ``(batch, 96, 7)``. The scalar arrays each have
        shape ``(batch,)`` and contain the normalized temperature and density
        gradients, respectively.
        """

        if geometry.ndim != 3 or geometry.shape[1:] != (
            N_Z_POINTS,
            N_GEOMETRY_FEATURES,
        ):
            raise ValueError(
                "geometry must have shape (batch, 96, 7); "
                f"received {tuple(geometry.shape)}"
            )

        hidden = geometry.transpose(1, 2)
        for convolution, pooling in zip(self.conv_layers, self.pool_layers):
            hidden = pooling(F.relu(convolution(hidden)))

        hidden = self.global_avg_pool(hidden).flatten(start_dim=1)
        hidden = torch.cat(
            (hidden, a_over_lt.reshape(-1, 1), a_over_ln.reshape(-1, 1)), dim=1
        )
        for dense_layer in self.fc_layers:
            hidden = F.relu(dense_layer(hidden))
        return self.output_layer(hidden)
