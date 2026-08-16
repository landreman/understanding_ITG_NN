"""Module-form equivalent of the legacy functional-ReLU inference network."""

from __future__ import annotations

import copy

import torch
from torch import nn

from itg_nn.model import Architecture, CyclicInvariantNet, N_CONVOLUTION_LAYERS


class ModuleCyclicInvariantNet(nn.Module):
    """An attribution-only version with named :class:`~torch.nn.ReLU` modules.

    Parameters and convolution/pooling operations are copied from an original
    ``CyclicInvariantNet``.  Its module names make Captum and hooks explicit;
    the forward arithmetic deliberately preserves the validated inference path.
    """

    def __init__(self, architecture: Architecture) -> None:
        super().__init__()
        channels = (7, *architecture.convolution_channels)
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
        self.relu_layers = nn.ModuleList(nn.ReLU() for _ in range(N_CONVOLUTION_LAYERS))
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
            for index in range(2)
        )
        self.dense_relus = nn.ModuleList(nn.ReLU() for _ in range(2))
        self.output_layer = nn.Linear(architecture.dense_dimensions[-1], 1)

    @classmethod
    def from_inference_model(
        cls, model: CyclicInvariantNet
    ) -> "ModuleCyclicInvariantNet":
        """Clone an original model into the module-form attribution network."""

        architecture = Architecture(
            kernel_sizes=tuple(layer.kernel_size[0] for layer in model.conv_layers),
            convolution_channels=tuple(layer.out_channels for layer in model.conv_layers),
            dense_dimensions=tuple(layer.out_features for layer in model.fc_layers),
        )
        wrapped = cls(architecture)
        wrapped.conv_layers = copy.deepcopy(model.conv_layers)
        wrapped.pool_layers = copy.deepcopy(model.pool_layers)
        wrapped.global_avg_pool = copy.deepcopy(model.global_avg_pool)
        wrapped.fc_layers = copy.deepcopy(model.fc_layers)
        wrapped.output_layer = copy.deepcopy(model.output_layer)
        wrapped.to(next(model.parameters()).device)
        wrapped.eval()
        return wrapped

    def forward(
        self,
        geometry: torch.Tensor,
        a_over_lt: torch.Tensor,
        a_over_ln: torch.Tensor,
    ) -> torch.Tensor:
        hidden = geometry.transpose(1, 2)
        for convolution, relu, pooling in zip(
            self.conv_layers, self.relu_layers, self.pool_layers
        ):
            hidden = pooling(relu(convolution(hidden)))
        hidden = self.global_avg_pool(hidden).flatten(start_dim=1)
        hidden = torch.cat(
            (hidden, a_over_lt.reshape(-1, 1), a_over_ln.reshape(-1, 1)), dim=1
        )
        for dense_layer, relu in zip(self.fc_layers, self.dense_relus):
            hidden = relu(dense_layer(hidden))
        return self.output_layer(hidden)
