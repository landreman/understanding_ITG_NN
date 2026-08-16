"""Explicit activation capture with stable module names and no hidden globals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager

import torch
from torch import nn


def _named_module(model: nn.Module, name: str) -> nn.Module:
    try:
        return dict(model.named_modules())[name]
    except KeyError as error:
        raise ValueError(f"unknown module name {name!r}") from error


class ActivationCapture(AbstractContextManager["ActivationCapture"]):
    """Capture requested forward activations while retaining tensor axes.

    By default activations are detached and copied to CPU, suitable for an atlas
    or artifact.  ``detach=False`` is available for a later gradient-based step.
    """

    def __init__(
        self,
        model: nn.Module,
        module_names: Sequence[str],
        *,
        detach: bool = True,
    ) -> None:
        if not module_names:
            raise ValueError("at least one module name is required")
        self.detach = detach
        self.values: dict[str, torch.Tensor] = {}
        self._handles = [
            _named_module(model, name).register_forward_hook(self._hook(name))
            for name in module_names
        ]

    def _hook(self, name: str):  # type: ignore[no-untyped-def]
        def capture(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"module {name!r} did not return a tensor")
            self.values[name] = output.detach().cpu().clone() if self.detach else output

        return capture

    def __exit__(self, *_: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def activations(self) -> Mapping[str, torch.Tensor]:
        """The most recent activation for each requested module."""

        return self.values
