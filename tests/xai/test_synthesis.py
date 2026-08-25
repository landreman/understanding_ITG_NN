from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from itg_nn.xai.synthesis import (
    MATRIX_EVIDENCE_COLUMNS,
    NATIVE_ESTIMAND,
    attach_evidence_manifest_pins,
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
        "direction_rule": "analytic toy expectation",
        "direction_source": "test fixture",
        "program_estimand": NATIVE_ESTIMAND,
        "outcome": NATIVE_ESTIMAND,
        "outcome_source": "toy.csv:estimand",
        "function_scope": "invariant_tilde_f",
        "cohort": "analytic cyclic toy",
        "regime": "all",
        "validity_tag": "exact-symmetry" if direction == "null_control" else "plausibly-local",
        "intervention": "joint circular shift" if direction == "null_control" else "analytic channel edit",
        "intervention_executed": True,
        "machine_readable": True,
        "uncertainty_unit": "not_applicable_no_interval_selected",
        "uncertainty_unit_source": "not_applicable",
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
    row["program_estimand"] = "Q"
    with pytest.raises(ValueError, match="native max"):
        validate_evidence_ledger([row])
    row["program_estimand"] = NATIVE_ESTIMAND
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
        "candidate_ids": "candidate",
        "evidence_polarity": "supports",
        "physical_causal_statement": False,
        "physical_intervention": "not_causal",
        "evidence_ids": "one;two",
        "evidence_alignment": {"one": "corroborates", "two": "corroborates"},
        "evidence_conjunct": {"one": "candidate support", "two": "candidate support"},
        "limitations": "toy only",
    }
    with pytest.raises(ValueError, match="corroborating method families"):
        validate_claim_register([claim], validate_evidence_ledger(evidence))
    evidence[1]["method_family"] = "hidden_intervention"
    validated = validate_claim_register([claim], validate_evidence_ledger(evidence))
    assert validated[0]["corroborating_method_family_count"] == 2
    assert validated[0]["minimum_corroborating_families_per_conjunct"] == 2
    assert validated[0]["corroborating_source_step_count"] == 1
    assert validated[0]["corroborating_source_artifact_count"] == 1
    assert validated[0]["gate_margin"] == 0


def test_headline_uses_claim_alignment_not_candidate_direction() -> None:
    evidence = [
        _evidence("one", "candidate", "input_attribution", "gradient_path"),
        _evidence(
            "two",
            "candidate",
            "hidden_encoding",
            "hidden_intervention",
            direction="mixed",
        ),
    ]
    claim = {
        "claim_id": "C1",
        "headline": True,
        "claim_text": "candidate is supported",
        "status": "supported",
        "scope": "invariant_tilde_f",
        "candidate_ids": "candidate",
        "evidence_polarity": "supports",
        "physical_causal_statement": False,
        "physical_intervention": "not_causal",
        "evidence_ids": "one;two",
        "evidence_alignment": {"one": "corroborates", "two": "qualifies"},
        "evidence_conjunct": {"one": "candidate support", "two": "scope limit"},
        "limitations": "toy only",
    }
    with pytest.raises(ValueError, match="corroborating method families"):
        validate_claim_register([claim], validate_evidence_ledger(evidence))
    claim["evidence_alignment"]["two"] = "corroborates"
    validated = validate_claim_register([claim], validate_evidence_ledger(evidence))
    assert validated[0]["corroborating_method_family_count"] == 2
    assert validated[0]["corroborating_evidence_ids"] == "one;two"
    evidence[1]["candidate_id"] = "other"
    with pytest.raises(ValueError, match="outside its candidates"):
        validate_claim_register([claim], validate_evidence_ledger(evidence))


def test_qualifying_cross_step_evidence_does_not_inflate_corroborating_steps() -> None:
    one = _evidence("one", "candidate", "input_attribution", "gradient_path")
    two = _evidence("two", "candidate", "hidden_encoding", "hidden_probe")
    qualifier = _evidence("qualifier", "candidate", "distillation", "distillation")
    qualifier["source_step"] = "SOTHER"
    claim = {
        "claim_id": "C1",
        "headline": True,
        "claim_text": "candidate is supported with a cross-step limitation",
        "status": "supported",
        "scope": "toy",
        "candidate_ids": "candidate",
        "evidence_polarity": "supports",
        "physical_causal_statement": False,
        "physical_intervention": "not_causal",
        "evidence_ids": "one;two;qualifier",
        "evidence_alignment": {
            "one": "corroborates",
            "two": "corroborates",
            "qualifier": "qualifies",
        },
        "evidence_conjunct": {
            "one": "candidate support",
            "two": "candidate support",
            "qualifier": "scope limit",
        },
        "limitations": "toy only",
    }
    validated = validate_claim_register(
        [claim], validate_evidence_ledger([one, two, qualifier])
    )[0]
    assert validated["corroborating_source_step_count"] == 1
    assert validated["corroborating_source_steps"] == "STOY"


def test_corroborating_families_remain_separate_across_claim_conjuncts() -> None:
    evidence = validate_evidence_ledger(
        [
            _evidence("one", "candidate", "input_attribution", "gradient_path"),
            _evidence("two", "candidate", "hidden_encoding", "hidden_probe"),
        ]
    )
    claim = {
        "claim_id": "C1",
        "headline": True,
        "claim_text": "compound candidate claim",
        "status": "supported",
        "scope": "toy",
        "candidate_ids": "candidate",
        "evidence_polarity": "supports",
        "physical_causal_statement": False,
        "physical_intervention": "not_causal",
        "evidence_ids": "one;two",
        "evidence_alignment": {"one": "corroborates", "two": "corroborates"},
        "evidence_conjunct": {"one": "first conjunct", "two": "second conjunct"},
        "limitations": "toy only",
    }
    validated = validate_claim_register([claim], evidence)[0]
    assert validated["corroborating_method_family_count"] == 2
    assert validated["maximum_corroborating_families_per_conjunct"] == 1


def test_claim_alignment_and_conjunct_cover_exactly_the_linked_evidence() -> None:
    evidence = validate_evidence_ledger(
        [
            _evidence("one", "candidate", "input_attribution", "gradient_path"),
            _evidence("two", "candidate", "hidden_encoding", "hidden_probe"),
        ]
    )
    claim = {
        "claim_id": "C1",
        "headline": True,
        "claim_text": "compound candidate claim",
        "status": "supported",
        "scope": "toy",
        "candidate_ids": "candidate",
        "evidence_polarity": "supports",
        "physical_causal_statement": False,
        "physical_intervention": "not_causal",
        "evidence_ids": "one;two",
        "evidence_alignment": {"one": "corroborates", "two": "corroborates"},
        "evidence_conjunct": {"one": "first conjunct"},
        "limitations": "toy only",
    }
    with pytest.raises(ValueError, match="evidence_conjunct"):
        validate_claim_register([claim], evidence)
    claim["evidence_conjunct"]["two"] = "second conjunct"
    claim["evidence_alignment"]["extra"] = "corroborates"
    with pytest.raises(ValueError, match="evidence_alignment"):
        validate_claim_register([claim], evidence)


def test_unknown_claim_alignment_is_rejected() -> None:
    evidence = validate_evidence_ledger(
        [
            _evidence("one", "candidate", "input_attribution", "gradient_path"),
            _evidence("two", "candidate", "hidden_encoding", "hidden_probe"),
        ]
    )
    claim = {
        "claim_id": "C1",
        "headline": True,
        "claim_text": "candidate claim",
        "status": "supported",
        "scope": "toy",
        "candidate_ids": "candidate",
        "evidence_polarity": "supports",
        "physical_causal_statement": False,
        "physical_intervention": "not_causal",
        "evidence_ids": "one;two",
        "evidence_alignment": {"one": "corroborates", "two": "maybe"},
        "evidence_conjunct": {"one": "claim", "two": "claim"},
        "limitations": "toy only",
    }
    with pytest.raises(ValueError, match="invalid evidence_alignment"):
        validate_claim_register([claim], evidence)


def test_every_declared_candidate_needs_claim_aligned_evidence() -> None:
    evidence = validate_evidence_ledger(
        [
            _evidence("a-one", "candidate-a", "input_attribution", "gradient_path"),
            _evidence("a-two", "candidate-a", "hidden_encoding", "hidden_probe"),
            _evidence("b", "candidate-b", "distillation", "distillation"),
        ]
    )
    claim = {
        "claim_id": "C1",
        "headline": True,
        "claim_text": "two-candidate claim",
        "status": "supported",
        "scope": "toy",
        "candidate_ids": "candidate-a;candidate-b",
        "evidence_polarity": "supports",
        "physical_causal_statement": False,
        "physical_intervention": "not_causal",
        "evidence_ids": "a-one;a-two;b",
        "evidence_alignment": {
            "a-one": "corroborates",
            "a-two": "corroborates",
            "b": "qualifies",
        },
        "evidence_conjunct": {"a-one": "claim", "a-two": "claim", "b": "limit"},
        "limitations": "toy only",
    }
    with pytest.raises(ValueError, match="candidates without claim-aligned evidence"):
        validate_claim_register([claim], evidence)


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
        "candidate_ids": "candidate",
        "evidence_polarity": "supports",
        "physical_causal_statement": True,
        "physical_intervention": "not_causal",
        "evidence_ids": "one;two",
        "evidence_alignment": {"one": "corroborates", "two": "corroborates"},
        "evidence_conjunct": {"one": "physical effect", "two": "physical effect"},
        "limitations": "toy only",
    }
    with pytest.raises(ValueError, match="intervention"):
        validate_claim_register([claim], validate_evidence_ledger(evidence))
    claim["physical_intervention"] = "analytic channel edit"
    assert validate_claim_register([claim], validate_evidence_ledger(evidence))


def test_invalid_validity_tag_and_direction_are_rejected() -> None:
    row = _evidence("bad", "candidate", "input_attribution", "gradient_path")
    row["validity_tag"] = "looks_local"
    with pytest.raises(ValueError, match="validity tag"):
        validate_evidence_ledger([row])
    row["validity_tag"] = "plausibly-local"
    row["direction"] = "probably_supports"
    with pytest.raises(ValueError, match="direction"):
        validate_evidence_ledger([row])


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


def test_manifest_pin_guard_rejects_unpinned_evidence(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.csv"
    evidence_path.write_text("value\n1\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"output_hashes": {}}), encoding="utf-8")
    rows = [{"manifest_path": "manifest.json"}]
    evidence = [{"source_artifact": "evidence.csv"}]
    with pytest.raises(ValueError, match="lack a manifest content-hash pin"):
        attach_evidence_manifest_pins(
            rows, evidence, repository=tmp_path, require_all=True
        )
