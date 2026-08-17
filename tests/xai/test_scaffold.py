from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from itg_nn.ensemble import Ensemble
from itg_nn.model import Architecture, CyclicInvariantNet
from itg_nn.xai.activations import ActivationCapture
from itg_nn.xai.artifacts import RunArtifacts
from itg_nn.xai.config import MemberSelectionConfig
from itg_nn.xai.members import MemberPredictor, select_member_ids
from itg_nn.xai.module_model import ModuleCyclicInvariantNet
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.toys import (
    ColocationToy,
    FourierBandToy,
    PeriodicPermutationToy,
    PeriodicWindowToy,
)


def test_s03_review_artifact_script_imports_as_a_module() -> None:
    module = importlib.import_module("scripts.xai_s03_review2_artifacts")
    assert callable(module.main)


def test_s03_compact_summary_reports_a_capped_expanded_cohort() -> None:
    module = importlib.import_module("scripts.xai_s03_ladder")
    rows = [
        {
            "member_id": f"m{index}",
            "function": "invariant_tilde_f",
            "gradient_set": "varied",
            "stratum": "all",
            "replicate": 0,
            "family": "joint_shift",
            "parameter": "shift=32",
            "dose": 1.0,
            "rms_change_over_residual_std": float(index + 1),
            "bootstrap_ci95_lower": float(index) + 0.5,
            "bootstrap_ci95_upper": float(index) + 1.5,
        }
        for index in range(5)
    ]
    summary = module._compact_ladder_summary(rows, {"m0", "m1"})
    joint_shift = next(row for row in summary if row["item"] == "joint_shift")
    assert joint_shift["expanded_member_count"] == 5
    assert joint_shift["expanded_cohort_median"] == 3.0


def _architecture() -> Architecture:
    return Architecture(
        kernel_sizes=(3, 3, 3, 3, 3),
        convolution_channels=(4, 5, 6, 7, 8),
        dense_dimensions=(9, 10),
    )


def test_validation_member_selection_never_uses_test_metrics() -> None:
    members = [
        {"id": "a", "validation_r2": 0.5, "test_r2": 0.99},
        {"id": "b", "validation_r2": 0.9, "test_r2": -1.0},
        {"id": "c", "validation_r2": 0.9, "test_r2": 0.0},
    ]
    selected = select_member_ids(
        members, MemberSelectionConfig(kind="top_validation", count=2)
    )
    assert selected == ("b", "c")


def test_member_predictor_preserves_individual_native_outputs() -> None:
    models = [CyclicInvariantNet(_architecture()).eval() for _ in range(2)]
    ensemble = Ensemble(
        models=models, member_ids=("first", "second"), device=torch.device("cpu")
    )
    predictor = MemberPredictor.from_ensemble(ensemble, ("second", "first"))
    geometry = torch.randn(3, 96, 7)
    gradients = torch.randn(3)
    actual = predictor(geometry, gradients, gradients)
    expected = torch.stack(
        (
            models[1](geometry, gradients, gradients).squeeze(1),
            models[0](geometry, gradients, gradients).squeeze(1),
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_module_form_matches_original_bit_for_bit_and_can_be_captured() -> None:
    set_deterministic_seed(11)
    original = CyclicInvariantNet(_architecture()).eval()
    wrapped = ModuleCyclicInvariantNet.from_inference_model(original)
    geometry = torch.randn(3, 96, 7)
    gradients = torch.randn(3)
    with ActivationCapture(
        wrapped, ("conv_layers.0", "relu_layers.0", "pool_layers.0")
    ) as capture:
        actual = wrapped(geometry, gradients, gradients)
    expected = original(geometry, gradients, gradients)
    assert torch.equal(actual, expected)
    assert capture.activations["conv_layers.0"].shape[-1] == 96
    assert capture.activations["pool_layers.0"].shape[-1] == 48


def test_periodic_toys_have_known_controls() -> None:
    geometry = torch.zeros(1, 96, 7)
    geometry[0, 95, 2] = 8.0
    window = PeriodicWindowToy(channel=2, start=92, width=8)
    assert window(geometry, torch.zeros(1), torch.zeros(1)).item() == pytest.approx(1.0)

    geometry[0, :, 1] = 2.0
    geometry[0, :, 5] = 3.0
    assert ColocationToy()(geometry, torch.zeros(1), torch.zeros(1)).item() == pytest.approx(6.0)

    signal = torch.sin(2 * torch.pi * 3 * torch.arange(96) / 96)
    geometry[0, :, 4] = signal
    result = FourierBandToy(channel=4, band=3)(geometry, torch.zeros(1), torch.zeros(1))
    assert result.item() > 20


def test_permutation_toy_is_joint_permutation_invariant_but_uses_colocation() -> None:
    generator = torch.Generator().manual_seed(7)
    geometry = torch.randn(3, 96, 7, generator=generator)
    gradients = torch.zeros(3)
    toy = PeriodicPermutationToy()
    expected = toy(geometry, gradients, gradients)
    permutation = torch.randperm(96, generator=generator)
    joint = toy(geometry[:, permutation], gradients, gradients)
    torch.testing.assert_close(expected, joint, rtol=1e-6, atol=1e-7)
    independently_permuted = geometry.clone()
    independently_permuted[:, :, 4] = independently_permuted[
        :, torch.randperm(96, generator=generator), 4
    ]
    actual = toy(independently_permuted, gradients, gradients)
    assert not torch.allclose(expected, actual)


def test_artifact_manifest_records_required_provenance(tmp_path) -> None:
    dataset = tmp_path / "dataset.bin"
    checkpoint = tmp_path / "checkpoint.bin"
    dataset.write_bytes(b"data")
    checkpoint.write_bytes(b"checkpoint")
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.write_hdf5(
        "values.h5",
        {"values": np.zeros((2, 3), dtype=np.float32)},
        axes={"values": ("member", "sample")},
    )
    artifacts.write_json("check.json", {"passed": True})
    artifacts.write_text("notes.csv", "value\n1\n")
    manifest_path = artifacts.finalize(
        config={"seed": 4, "full": "config"},
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=("member-1",),
        row_ids=(7, 8, 9),
        gradient_set="varied",
        device="cpu",
        repository=tmp_path,
        command=("xai-smoke",),
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset"]["sha256"]
    assert manifest["checkpoint"]["sha256"]
    assert manifest["output_hashes"].keys() == {
        "check.json",
        "notes.csv",
        "values.h5",
    }
    assert manifest["member_ids"] == ["member-1"]
    assert manifest["row_ids"] == [7, 8, 9]
