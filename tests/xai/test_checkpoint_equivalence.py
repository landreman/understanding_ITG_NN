from __future__ import annotations

from pathlib import Path

import torch

from itg_nn.ensemble import load_ensemble
from itg_nn.xai.module_model import ModuleCyclicInvariantNet


CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "cyclic_ensemble_pre2.pt"


def test_all_checkpoint_members_match_module_form_bit_for_bit() -> None:
    """Guard the all-member equivalence contract required before LRP/Captum use."""

    torch.manual_seed(20260816)
    ensemble = load_ensemble(CHECKPOINT, device="cpu")
    geometry = torch.randn(2, 96, 7)
    a_over_lt = torch.randn(2)
    a_over_ln = torch.randn(2)
    with torch.inference_mode():
        for original in ensemble.models:
            wrapped = ModuleCyclicInvariantNet.from_inference_model(original)
            assert torch.equal(
                original(geometry, a_over_lt, a_over_ln),
                wrapped(geometry, a_over_lt, a_over_ln),
            )
