from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from itg_nn.xai.synthesis import MATRIX_EVIDENCE_COLUMNS, NATIVE_ESTIMAND


ROOT = Path(__file__).parents[2]
ARTIFACTS = ROOT / "reports/xai/S14_artifacts"


def _rows(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader((ARTIFACTS / name).open(newline="", encoding="utf-8")))


def _one(rows: list[dict[str, str]], **keys: str) -> dict[str, str]:
    selected = [row for row in rows if all(row[key] == value for key, value in keys.items())]
    assert len(selected) == 1
    return selected[0]


def test_registered_manifest_hashes_all_synthesis_outputs() -> None:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["run_id"] == "synthesis-registered-evidence-s01-s13"
    assert manifest["run_id"] == "synthesis-registered-evidence-s01-s13"
    assert manifest["model_outputs_computed"] is False
    assert manifest["gx_outputs_computed"] is False
    assert manifest["review_slice_used"] is False
    assert manifest["source_manifest_count"] == 19
    assert manifest["source_run_manifest_count"] == 18
    assert manifest["source_evidence_artifact_count"] == 21
    expected_regular = {
        "evidence_ledger.csv",
        "evidence_matrix.csv",
        "claim_register.csv",
        "reproducibility_index.csv",
        "next_experiments.csv",
        "source_hashes.json",
        "summary.json",
    }
    expected_upstream = {
        "upstream_manifest_S00.json": "upstream_manifests/S00.json",
        "upstream_manifest_S01.json": "upstream_manifests/S01.json",
        "upstream_manifest_S02_ORIGINAL.json": "upstream_manifests/S02_original.json",
        "upstream_manifest_S03_PRIMARY.json": "upstream_manifests/S03_primary.json",
        "upstream_manifest_S03_PHASE.json": "upstream_manifests/S03_phase.json",
    }
    assert set(manifest["output_hashes"]) == expected_regular | set(expected_upstream)
    for name in expected_regular:
        assert hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest() == manifest[
            "output_hashes"
        ][name]
    for output_name, published_name in expected_upstream.items():
        assert hashlib.sha256((ARTIFACTS / published_name).read_bytes()).hexdigest() == manifest[
            "output_hashes"
        ][output_name]


def test_evidence_ledger_selectors_reproduce_every_source_value() -> None:
    ledger = _rows("evidence_ledger.csv")
    assert len(ledger) == 64
    assert len({row["evidence_id"] for row in ledger}) == 64
    assert {row["program_estimand"] for row in ledger} == {NATIVE_ESTIMAND}
    assert "estimand" not in ledger[0]
    assert {row["machine_readable"] for row in ledger} == {"True"}
    assert all(row["direction_rule"] for row in ledger)
    assert all(row["direction_source"] for row in ledger)
    assert {row["outcome"] for row in ledger} == {
        NATIVE_ESTIMAND,
        "target_native",
        "Qz_localization",
        "log10_zonal_phi2",
    }
    for row in ledger:
        if "ci95" in row["source_fields"] or "interval_" in row["source_fields"]:
            assert row["uncertainty_unit"] == "equilibrium_files"
            assert row["uncertainty_unit_source"] != "not_applicable"
    for evidence in ledger:
        source = ROOT / evidence["source_artifact"]
        selector = json.loads(evidence["source_selector"])
        fields = evidence["source_fields"].split(";")
        rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
        selected = [
            row
            for row in rows
            if all(row[key] == str(value) for key, value in selector.items())
        ]
        expected = [{field: row[field] for field in fields} for row in selected]
        assert len(expected) == int(evidence["source_record_count"])
        assert json.loads(evidence["source_values_json"]) == expected


def test_matrix_is_complete_and_keeps_negative_results_explicit() -> None:
    matrix = _rows("evidence_matrix.csv")
    assert len(matrix) == 11
    assert {row["status"] for row in matrix} == {
        "supported",
        "regime-dependent",
        "contradicted",
        "unresolved",
    }
    assert sum(row["status"] == "supported" for row in matrix) == 5
    assert sum(row["status"] == "regime-dependent" for row in matrix) == 3
    assert sum(row["status"] == "contradicted" for row in matrix) == 2
    assert sum(row["status"] == "unresolved" for row in matrix) == 1
    for row in matrix:
        assert row["negative_results"]
        assert row["uncertainty"]
        for column in MATRIX_EVIDENCE_COLUMNS:
            assert row[column]
    spread = _one(matrix, candidate_id="ensemble_spread_as_error_signal")
    assert "not a calibrated error bar" in spread["hypothesis"]
    q_z = _one(matrix, candidate_id="direct_Qz_spatial_focus")
    assert q_z["status"] == "contradicted"
    assert "E07_ATTRIBUTION_QZ_NULL" in q_z["input_attribution"]
    zonal = _one(matrix, candidate_id="zonal_flow_mechanism")
    assert zonal["status"] == "contradicted"


def test_every_headline_is_machine_readable_and_triangulated() -> None:
    claims = _rows("claim_register.csv")
    assert len(claims) == 9
    assert {row["headline"] for row in claims} == {"True"}
    assert {row["physical_causal_statement"] for row in claims} == {"False"}
    assert {row["physical_intervention"] for row in claims} == {"not_causal"}
    for row in claims:
        assert int(row["corroborating_method_family_count"]) >= 2
        assert len(row["corroborating_method_families"].split(";")) >= 2
        evidence_ids = set(row["evidence_ids"].split(";"))
        assert set(json.loads(row["evidence_alignment"])) == evidence_ids
        assert set(json.loads(row["evidence_conjunct"])) == evidence_ids
        assert int(row["corroborating_source_step_count"]) >= 1
        assert int(row["corroborating_source_artifact_count"]) >= 1
        assert int(row["gate_margin"]) == int(row["corroborating_method_family_count"]) - 2
        for path in row["machine_readable_sources"].split(";"):
            assert (ROOT / path).is_file()
    spread = _one(claims, claim_id="C08_SPREAD_NOT_ERROR_BAR")
    assert json.loads(spread["evidence_conjunct"]) == {
        "E11_COMMON_MODE_FAILURE": "not a calibrated guarantee",
        "E11_SPREAD_ERROR_ASSOCIATION": "error-ranking utility",
    }
    assert json.loads(spread["corroborating_family_counts_per_conjunct"]) == {
        "error-ranking utility": 1,
        "not a calibrated guarantee": 1,
    }
    assert spread["corroborating_source_step_count"] == "1"
    assert spread["corroborating_source_artifact_count"] == "2"
    direct_qz = _one(claims, claim_id="C09_DIRECT_QZ_FOCUS_REJECTED")
    assert set(direct_qz["corroborating_evidence_ids"].split(";")) == {
        "E07_DENSITY_QZ_CONTRADICTION",
        "E07_ATTRIBUTION_QZ_NULL",
    }


def test_headline_numbers_and_contradictions_are_pinned() -> None:
    evidence = _rows("evidence_ledger.csv")

    def values(evidence_id: str) -> list[dict[str, str]]:
        return json.loads(_one(evidence, evidence_id=evidence_id)["source_values_json"])

    shapley = values("E04_FQ_U001_SHAPLEY")[0]
    assert float(shapley["mean_absolute"]) == pytest.approx(0.4183336394447324)
    alignment = values("E03_ALIGNMENT_PERTURBATION")[0]
    assert float(alignment["top10_median"]) == pytest.approx(2.4131995623205214)
    channel = values("E06_CHANNEL_AGREEMENT")[0]
    assert float(channel["median_pairwise_channel_rank_agreement"]) == pytest.approx(
        0.9642857142857142
    )
    assert float(channel["mean_cell_sign_agreement"]) == pytest.approx(
        0.7485119047619048
    )
    q_z = sorted(float(row["circular_spearman"]) for row in values("E07_ATTRIBUTION_QZ_NULL"))
    assert q_z == pytest.approx(
        sorted([-0.02122218224703246, -0.012763898379259002, -0.01160833480844481])
    )
    zonal = values("E08_ZONAL_USE_NULL")
    assert len(zonal) == 15
    assert {row["use_claim_permitted"] for row in zonal} == {"False"}
    fidelity = sorted(float(row["held_out_r2"]) for row in values("E12_MEMBER_FIDELITY"))
    assert fidelity == pytest.approx(
        sorted([0.8603164016682042, 0.8560790733151917, 0.8635806956843547])
    )
    geodesic = values("E13_GEO_NATURAL")[0]
    assert geodesic["causal_claim_permitted"] == "False"
    assert geodesic["aipw_resolved_fold_count"] == "7"
    assert float(geodesic["max_abs_nuisance_smd_after"]) > 0.5
    assert float(geodesic["aipw_overlap_fraction"]) < 0.8
    expected_directions = {
        "E07_FQ_QZ": "mixed",
        "E08_COLOCATION_USE": "regime-dependent",
        "E08_GEO_USE": "regime-dependent",
        "E08_LOCAL_QZ_ENCODING": "mixed",
        "E08_LOCAL_QZ_USE": "regime-dependent",
        "E08_ZONAL_ENCODING_NULL": "contradicts",
        "E08_ZONAL_USE_NULL": "contradicts",
        "E11_COMMON_MODE_FAILURE": "contradicts",
        "E11_SPREAD_ERROR_ASSOCIATION": "regime-dependent",
        "E12_BAD_RECURRENCE": "mixed",
        "E12_GEO_RECURRENCE": "regime-dependent",
        "E13_BAD_NATURAL": "regime-dependent",
        "E13_FQ_NATURAL": "unresolved",
        "E13_FSTAB_NATURAL": "regime-dependent",
        "E13_GEO_NATURAL": "supports",
    }
    for evidence_id, direction in expected_directions.items():
        row = _one(evidence, evidence_id=evidence_id)
        assert row["direction"] == direction
        assert row["direction_source"].startswith("config.direction_rule:")
    assert {
        _one(evidence, evidence_id=evidence_id)["direction"]
        for evidence_id in ("E03_JOINT_SHIFT_EXACT_CONTROL", "E03_HIGH_BAND_CONTROL")
    } == {"null_control"}
    local_qz_use = values("E08_LOCAL_QZ_USE")
    assert len(local_qz_use) == 15
    assert sum(row["use_claim_permitted"] == "True" for row in local_qz_use) == 7


def test_reproducibility_index_resolves_and_hashes_every_run_manifest() -> None:
    rows = _rows("reproducibility_index.csv")
    assert len(rows) == 19
    phase = _one(rows, run_key="S03_PHASE")
    assert phase["recreates_claims"] == "False"
    assert phase["git_commit"] in {"", "None"}
    assert all(
        row["recreates_claims"] == "True" for row in rows if row["run_key"] != "S03_PHASE"
    )
    assert sum(row["is_run_manifest"] == "True" for row in rows) == 18
    assert {row["step"] for row in rows} >= {
        "S00",
        "S01",
        "S02",
        "S03",
        "S04",
        "S05",
        "S06a",
        "S06b",
        "S07",
        "S08",
        "S09",
        "S10",
        "S11",
        "S12",
        "S13",
    }
    for row in rows:
        path = ROOT / row["manifest_path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["manifest_sha256"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if row["role"] == "publication_verification":
            assert manifest["published_output_hashes"]
            assert "reports/xai/S03_artifacts/ladder_summary.csv" in row[
                "pins_evidence_artifacts"
            ]
        else:
            assert manifest["config"]["run_id"] == row["run_id"]
            assert manifest["command"]
            assert manifest["dataset"]["sha256"]
            assert manifest["checkpoint"]["sha256"]


def test_every_evidence_artifact_is_content_hash_pinned_by_an_indexed_manifest() -> None:
    source_hashes = json.loads(
        (ARTIFACTS / "source_hashes.json").read_text(encoding="utf-8")
    )["evidence_artifacts"]
    pinned = {
        path
        for row in _rows("reproducibility_index.csv")
        for path in row["pins_evidence_artifacts"].split(";")
        if path != "none"
    }
    assert pinned == set(source_hashes)


def test_smallest_next_calculation_is_vmec_only_before_gx() -> None:
    rows = _rows("next_experiments.csv")
    assert [int(row["priority"]) for row in rows] == [1, 2, 3, 4, 5]
    first = rows[0]
    assert first["experiment_id"] == "N01_VMEC_JACOBIAN_FEASIBILITY"
    assert "no GX runs" in first["estimated_cost"]
    assert "0.5 panel IQR" in first["minimum_success"]
    assert "0.1 panel IQR" in first["minimum_success"]
    assert "researcher approval" in first["decision_gate"]


def test_reports_publish_required_handoff_sections() -> None:
    final_report = (ROOT / "reports/xai/FINAL_REPORT.md").read_text(encoding="utf-8")
    executive = (ROOT / "reports/xai/S14_executive_summary.md").read_text(
        encoding="utf-8"
    )
    step_report = (ROOT / "reports/xai/S14_synthesis.md").read_text(encoding="utf-8")
    for report in (final_report, executive, step_report):
        assert "## Deferred" in report
        assert "## Reviewer reproduction" in report
    assert "## Acceptance criteria" in final_report
    assert "## Acceptance criteria" in step_report
    assert "Every headline conclusion links to machine-readable evidence" in final_report
    assert "Every causal statement identifies its intervention" in final_report
    assert "All runs can be recreated from manifests" in final_report
    assert "S14_artifacts/evidence_matrix.csv" in final_report
    assert "S14_artifacts/reproducibility_index.csv" in final_report
    assert "S14_artifacts/next_experiments.csv" in final_report
