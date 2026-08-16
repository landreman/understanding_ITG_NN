"""Deterministic setup and transparent mini-batching for XAI experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from itg_nn.data import InferenceData


def set_deterministic_seed(seed: int) -> None:
    """Seed all local random generators used by CPU pilot calculations."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class InferenceBatch:
    """A source-addressable batch retaining the original stable row IDs."""

    geometry: torch.Tensor
    a_over_lt: torch.Tensor
    a_over_ln: torch.Tensor
    row_indices: np.ndarray


def iter_inference_batches(
    data: InferenceData, batch_size: int
) -> Iterator[InferenceBatch]:
    """Yield source-addressable batches without shuffling sample order."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(data.row_indices), batch_size):
        stop = min(start + batch_size, len(data.row_indices))
        yield InferenceBatch(
            geometry=data.geometry[start:stop],
            a_over_lt=data.a_over_lt[start:stop],
            a_over_ln=data.a_over_ln[start:stop],
            row_indices=data.row_indices[start:stop],
        )
