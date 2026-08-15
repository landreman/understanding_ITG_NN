"""Checkpoint loading and batched ensemble inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .model import Architecture, CyclicInvariantNet


CHECKPOINT_FORMAT_VERSION = 1
TARGET_TRANSFORM = "log_clip_min_-2"


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve an explicit device or select the best available accelerator."""

    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class EnsemblePrediction:
    """Mean and model-to-model spread in the network's log target space."""

    mean_log_heat_flux: np.ndarray
    std_log_heat_flux: np.ndarray
    member_count: int

    @property
    def mean_heat_flux(self) -> np.ndarray:
        return np.exp(self.mean_log_heat_flux)

    @property
    def lower_heat_flux(self) -> np.ndarray:
        return np.exp(self.mean_log_heat_flux - self.std_log_heat_flux)

    @property
    def upper_heat_flux(self) -> np.ndarray:
        return np.exp(self.mean_log_heat_flux + self.std_log_heat_flux)


@dataclass
class Ensemble:
    """Loaded inference ensemble and its member identifiers."""

    models: Sequence[CyclicInvariantNet]
    member_ids: tuple[str, ...]
    device: torch.device

    def predict(
        self,
        geometry: torch.Tensor,
        a_over_lt: torch.Tensor,
        a_over_ln: torch.Tensor,
        *,
        batch_size: int = 1024,
    ) -> EnsemblePrediction:
        """Run all ensemble members in batches and aggregate their predictions."""

        sample_count = len(geometry)
        if sample_count == 0:
            raise ValueError("At least one sample is required for inference")
        if len(a_over_lt) != sample_count or len(a_over_ln) != sample_count:
            raise ValueError("Geometry and gradient sample counts do not match")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        member_predictions = np.empty(
            (sample_count, len(self.models)), dtype=np.float32
        )
        with torch.inference_mode():
            for member_index, model in enumerate(self.models):
                for start in range(0, sample_count, batch_size):
                    stop = min(start + batch_size, sample_count)
                    output = model(
                        geometry[start:stop].to(self.device),
                        a_over_lt[start:stop].to(self.device),
                        a_over_ln[start:stop].to(self.device),
                    )
                    member_predictions[start:stop, member_index] = (
                        output.squeeze(1).cpu().numpy()
                    )

        return EnsemblePrediction(
            mean_log_heat_flux=member_predictions.mean(axis=1),
            std_log_heat_flux=member_predictions.std(axis=1),
            member_count=len(self.models),
        )


def load_ensemble(
    checkpoint_path: str | Path, *, device: str | torch.device = "auto"
) -> Ensemble:
    """Load the consolidated, inference-only ensemble checkpoint."""

    resolved_device = resolve_device(device)
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported ensemble checkpoint format")
    if bundle.get("target_transform") != TARGET_TRANSFORM:
        raise ValueError("Checkpoint target transform is not supported")

    models: list[CyclicInvariantNet] = []
    member_ids: list[str] = []
    for member in bundle["members"]:
        architecture = Architecture.from_mapping(member["architecture"])
        model = CyclicInvariantNet(architecture)
        model.load_state_dict(member["state_dict"], strict=True)
        model.to(resolved_device)
        model.eval()
        models.append(model)
        member_ids.append(member["id"])

    if not models:
        raise ValueError("Ensemble checkpoint contains no models")
    return Ensemble(models=models, member_ids=tuple(member_ids), device=resolved_device)
