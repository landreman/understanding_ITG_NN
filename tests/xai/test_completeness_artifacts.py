from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


ARTIFACTS = Path("reports/xai/S09_artifacts")


@pytest.mark.skipif(not (ARTIFACTS / "summary.json").is_file(), reason="S09 production artifacts not committed")
def test_s09_headlines_recompute_from_committed_tables():
    summary = json.loads((ARTIFACTS / "summary.json").read_text())
    rows = list(csv.DictReader((ARTIFACTS / "completeness.csv").open()))
    all_rows = [row for row in rows if row["regime"] == "all"]
    candidates = [float(row["held_out_r2"]) for row in all_rows if row["concept_set"] == "all_candidates"]
    baseline = [float(row["held_out_r2"]) for row in all_rows if row["concept_set"] == "paper_baseline"]
    assert sorted(candidates)[1] == pytest.approx(summary["median_all_candidates_r2"])
    assert sorted(c - b for c, b in zip(candidates, baseline))[1] == pytest.approx(summary["median_gain_over_paper_baseline"])
    assert all(float(row["gain_over_paper_baseline_ci95_lower"]) > 0 for row in all_rows if row["concept_set"] == "all_candidates")
    assert {row["regime"] for row in rows} == {"all", "stable_or_near_floor", "unstable"}
    assert all(row["estimand"] == "native max(log Q, -2)" for row in rows)


@pytest.mark.skipif(not (ARTIFACTS / "manifest.json").is_file(), reason="S09 production manifest not committed")
def test_s09_manifest_hashes_every_published_artifact():
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    assert manifest["config"]["production_compute_wall_time_seconds"] > 0
    for name, expected in manifest["output_hashes"].items():
        assert hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest() == expected


@pytest.mark.skipif(not (ARTIFACTS / "interaction_summary.csv").is_file(), reason="S09 interaction artifacts not committed")
def test_s09_interactions_retain_members_and_regimes():
    effects = list(csv.DictReader((ARTIFACTS / "interaction_effects.csv").open()))
    hessian = list(csv.DictReader((ARTIFACTS / "integrated_hessian_terms.csv").open()))
    assert len({row["member_id"] for row in effects}) == 3
    assert {row["regime"] for row in effects} == {"all", "stable_or_near_floor", "unstable"}
    assert {row["regime"] for row in hessian} == {"all", "stable_or_near_floor", "unstable"}
    assert all(row["perturbation_validity_tag"] == "observed-comparison" for row in effects)
