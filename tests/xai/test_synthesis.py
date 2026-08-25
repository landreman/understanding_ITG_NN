from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from itg_nn.xai.synthesis import (
    MATRIX_EVIDENCE_COLUMNS,
    NATIVE_ESTIMAND,
    validate_claim_register,
    validate_evidence_ledger,
    validate_evidence_matrix,
    validate_reproducibility_index,
)
from itg_nn.xai.toys import ColocationToy


def _evidence(
    evidence_id: str,
    candidate_id: str,
    matrix_column: str,
    method_family: str,
    *,
    direction: str = "supports",
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "candidate_id": candidate_id,
        "matrix_column": matrix_column,
        "source_step": "STOY",
        "source_artifact": "toy.csv",
        "source_selector": "candidate=toy",
        "source_fields": "signed_effect",
        "source_record_count": 1,
        "method_family": method_family,
        "direction": direction,
        "estimand": NATIVE_ESTIMAND,
        "function_scope": "invariant_tilde_f",
        "cohort": "analytic cyclic toy",
        "regime": "all",
        "validity_tag": "exact-symmetry" if direction == "null_control" else "plausibly-local",
        "intervention": "joint circular shift" if direction == "null_control" else "analytic channel edit",
        "intervention_executed": True,
        "machine_readable": True,
        "summary": "analytic result",
    }


def _matrix(candidate_id: str, evidence_ids: dict[str, str]) -> dict[str, str]:
    row = {
        "candidate_id": candidate_id,
        "hypothesis": "analytic cyclic candidate",
        "status": "supported",
        "claim_grade": "model-mechanistic",
        "function_scope": "invariant_tilde_f",
        "regime": "all",
        "uncertainty": "closed form",
        "negative_results": "ignored channel is exactly null",
    }
    row.update({column: evidence_ids.get(column, "not_tested") for column in MATRIX_EVIDENCE_COLUMNS})
    return row


def test_analytic_cyclic_toy_triangulates_signal_and_keeps_null_control() -> None:
    geometry = torch.zeros(1, 96, 7, requires_grad=True)
    with torch.no_grad():
        geometry[0, :, 1] = torch.arange(96, dtype=torch.float32) / 96
        geometry[0, :, 5] = 2.0
        geometry[0, :, 0] = 3.0
    drives = torch.zeros(1)
    toy = ColocationToy()
    output = toy(geometry, drives, drives)
    output.sum().backward()
    assert geometry.grad is not None
    assert torch.count_nonzero(geometry.grad[..., 1]) == 96
    assert torch.count_nonzero(geometry.grad[..., 5]) == 95
    assert torch.count_nonzero(geometry.grad[..., 0]) == 0
    shifted = torch.roll(geometry.detach(), shifts=17, dims=1)
    torch.testing.assert_close(toy(shifted, drives, drives), output.detach())

    rows = [
        _evidence("toy-gradient", "colocation", "input_attribution", "gradient_path"),
        _evidence("toy-edit", "colocation", "supported_perturbation", "perturbation"),
        _evidence(
            "toy-null", "ignored_channel", "input_attribution", "gradient_path", direction="null_control"
        ),
    ]
    validated = validate_evidence_ledger(rows)
    assert len(validated) == 3
    matrix = _matrix(
        "colocation",
        {
            "input_attribution": "toy-gradient",
            "supported_perturbation": "toy-edit",
        },
    )
    assert validate_evidence_matrix([matrix], validated)[0]["status"] == "supported"


def test_native_output_and_machine_readable_source_are_required() -> None:
    row = _evidence("bad", "candidate", "input_attribution", "gradient_path")
    row["estimand"] = "Q"
    with pytest.raises(ValueError, match="native max"):
        validate_evidence_ledger([row])
    row["estimand"] = NATIVE_ESTIMAND
    row["machine_readable"] = False
    with pytest.raises(ValueError, match="machine-readable"):
        validate_evidence_ledger([row])


def test_headline_claim_requires_two_independent_method_families() -> None:
    evidence = [
        _evidence("one", "candidate", "input_attribution", "gradient_path"),
        _evidence("two", "candidate", "hidden_encoding", "gradient_path"),
    ]
    claim = {
        "claim_id": "C1",
        "headline": True,
        "claim_text": "candidate is supported",
        "status": "supported",
        "scope": "invariant_tilde_f",
        "causal_statement": False,
        "intervention": "not_causal",
        "evidence_ids": "one;two",
        "limitations": "toy only",
    }
    with pytest.raises(ValueError, match="independent method families"):
        validate_claim_register([claim], validate_evidence_ledger(evidence))
    evidence[1]["method_family"] = "hidden_intervention"
    validated = validate_claim_register([claim], validate_evidence_ledger(evidence))
    assert validated[0]["independent_method_family_count"] == 2


def test_causal_statement_requires_named_executed_intervention() -> None:
    evidence = [
        _evidence("one", "candidate", "input_attribution", "gradient_path"),
        _evidence("two", "candidate", "supported_perturbation", "perturbation"),
    ]
    claim = {
        "claim_id": "C1",
        "headline": True,
        "claim_text": "editing the candidate changes the native output",
        "status": "supported",
        "scope": "invariant_tilde_f",
        "causal_statement": True,
        "intervention": "not_causal",
        "evidence_ids": "one;two",
        "limitations": "toy only",
    }
    with pytest.raises(ValueError, match="intervention"):
        validate_claim_register([claim], validate_evidence_ledger(evidence))
    claim["intervention"] = "analytic channel edit"
    assert validate_claim_register([claim], validate_evidence_ledger(evidence))


def test_matrix_requires_every_plan_column_and_explicit_absence() -> None:
    evidence = validate_evidence_ledger(
        [_evidence("one", "candidate", "input_attribution", "gradient_path")]
    )
    row = _matrix("candidate", {"input_attribution": "one"})
    del row["zonal_flow"]
    with pytest.raises(ValueError, match="zonal_flow"):
        validate_evidence_matrix([row], evidence)
    row["zonal_flow"] = ""
    with pytest.raises(ValueError, match="explicit"):
        validate_evidence_matrix([row], evidence)
    row["zonal_flow"] = "not_tested"
    assert validate_evidence_matrix([row], evidence)


def test_reproducibility_index_checks_manifest_content_and_hash(tmp_path: Path) -> None:
    manifest = {
        "command": ["python", "run.py"],
        "config": {"run_id": "registered"},
        "checkpoint": {"sha256": "checkpoint"},
        "dataset": {"sha256": "dataset"},
        "device": "cpu",
        "git_commit": "abc123",
        "git_tree": "tree123",
        "git_dirty": False,
        "git_tracked_dirty": False,
        "gradient_set": "varied",
        "member_ids": ["m1"],
        "output_hashes": {"values.csv": "digest"},
        "package_versions": {"numpy": "1"},
        "python": {"version": "3"},
        "row_ids": [1],
        "seed": 7,
        "wall_time_seconds": 1.0,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {
        "run_key": "SXX",
        "step": "SXX",
        "run_id": "registered",
        "manifest_path": "manifest.json",
        "manifest_sha256": digest,
        "role": "registered_production",
        "recreates_claims": True,
        "caveat": "none",
    }
    validated = validate_reproducibility_index([row], repository=tmp_path)
    assert validated[0]["output_count"] == 1
    row["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        validate_reproducibility_index([row], repository=tmp_path)

