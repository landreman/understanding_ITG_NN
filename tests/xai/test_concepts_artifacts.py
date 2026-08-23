from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from itg_nn.xai.artifacts import sha256_file


ARTIFACTS = Path("reports/xai/S08_artifacts")


def _csv(name: str) -> list[dict[str, str]]:
    with (ARTIFACTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_registered_manifest_and_small_artifact_hashes_are_complete() -> None:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["run_id"] == "concept-probes-top3-panel1000"
    assert manifest["config"]["production_compute_wall_time_seconds"] == pytest.approx(
        2203.1035236670286
    )
    assert manifest["wall_time_seconds"] < 5
    assert manifest["git_tracked_dirty"] is False
    assert len(manifest["row_ids"]) == 1000
    assert len(set(manifest["row_ids"])) == 1000
    assert len(manifest["member_ids"]) == 3
    for name, expected in manifest["output_hashes"].items():
        assert sha256_file(ARTIFACTS / name) == expected


def test_encoding_and_use_are_separate_grouped_member_level_columns() -> None:
    matrix = _csv("encoding_use_matrix.csv")
    assert len(matrix) == 3 * 5 * 10
    assert Counter(row["member_id"] for row in matrix) == {
        "2864601_0.437": 50,
        "2864601_0.371": 50,
        "2864601_0.409": 50,
    }
    assert {row["outer_split_unit"] for row in matrix} == {"equilibrium_files"}
    assert {row["inner_split_unit"] for row in matrix} == {"equilibrium_files"}
    assert {row["estimand"] for row in matrix} == {"native max(log Q, -2)"}
    assert {row["canonical_function"] for row in matrix} == {"invariant_tilde_f"}
    assert all("encoded_r2_stable_or_near_floor" in row for row in matrix)
    assert all("mean_directional_derivative_unstable" in row for row in matrix)
    assert {(row["derivative_rows"], row["derivative_stable_rows"], row["derivative_unstable_rows"]) for row in matrix} == {("96", "25", "71")}
    assert {row["direction_source"] for row in matrix} == {
        "mean_of_five_paired_matched_counterexample_CAVs"
    }
    assert {(row["encoded_column"], row["used_column"]) for row in matrix} == {
        ("True", "True")
    }
    assert all("bootstrap_fdr_q_value" in row for row in matrix)
    assert all("orthogonal_complement_ablation_rms" in row for row in matrix)
    for suffix in ("stable_or_near_floor", "unstable"):
        assert all(f"intervention_rms_{suffix}" in row for row in matrix)
        assert all(
            f"intervention_to_scale_matched_random_ratio_{suffix}" in row
            for row in matrix
        )


def test_controls_and_claim_gate_preserve_the_zonal_contradiction() -> None:
    probes = _csv("probe_scores.csv")
    assert len(probes) == 165
    random_rows = [row for row in probes if row["concept"] == "random_concept_control"]
    assert len(random_rows) == 15
    matrix = _csv("encoding_use_matrix.csv")
    zonal = [row for row in matrix if row["concept"] == "log10_zonal_phi2"]
    assert len(zonal) == 15
    assert all(row["encoded_generalizes_by_equilibrium"] == "False" for row in zonal)
    assert all(row["use_claim_permitted"] == "False" for row in zonal)
    # The raw intervention control passes anyway: this contradiction is retained,
    # not converted into a use claim without held-out encoding.
    assert all(row["direction_intervention_beats_random"] == "True" for row in zonal)


def test_headline_acceptance_numbers_are_pinned_to_the_tables() -> None:
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    matrix = _csv("encoding_use_matrix.csv")
    assert summary["median_probe_r2"] == pytest.approx(0.7471185927044091)
    assert summary["median_permuted_r2"] == pytest.approx(-0.0020780937051435577)
    assert summary["median_random_concept_r2"] == pytest.approx(-0.0022791902498608962)
    assert summary["stable_counterexample_fraction"] == pytest.approx(146 / 150)
    assert summary["intervention_beats_random_fraction"] == pytest.approx(120 / 150)
    assert summary["intervention_beats_scale_matched_random_fraction"] == pytest.approx(119 / 150)
    assert summary["fdr_significant_fraction"] == pytest.approx(124 / 150)
    assert sum(row["use_claim_permitted"] == "True" for row in matrix) == 83
    assert summary["use_claim_permitted_fraction"] == pytest.approx(83 / 150)
    assert summary["cohort"] == {
        "panel_rows": 1000,
        "unique_equilibrium_files": 1000,
        "stable_or_near_floor": 240,
        "unstable": 760,
    }


def test_matched_examples_are_observed_and_equilibrium_disjoint() -> None:
    rows = _csv("matched_examples.csv")
    assert len(rows) == 3940
    assert {row["validity_tag"] for row in rows} == {"observed-comparison"}
    for concept in {row["concept"] for row in rows}:
        high = {
            row["equilibrium_file"]
            for row in rows
            if row["concept"] == concept and row["role"] == "high"
        }
        low = {
            row["equilibrium_file"]
            for row in rows
            if row["concept"] == concept and row["role"] == "low"
        }
        assert high.isdisjoint(low)

    balance = _csv("matching_balance.csv")
    assert len(balance) == 10
    failed = [row["concept"] for row in balance if row["balance_pass"] == "False"]
    assert failed == ["log10_zonal_phi2"]
    assert max(
        float(row["max_abs_smd"])
        for row in balance
        if row["concept"] != "log10_zonal_phi2"
    ) < 0.25
