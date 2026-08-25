from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
ARTIFACTS = ROOT / "reports/xai/S13_artifacts"


def _rows(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader((ARTIFACTS / name).open(newline="", encoding="utf-8")))


def _one(rows: list[dict[str, str]], **keys: str) -> dict[str, str]:
    selected = [row for row in rows if all(row[key] == value for key, value in keys.items())]
    assert len(selected) == 1
    return selected[0]


def test_registered_manifest_hashes_every_published_scientific_artifact() -> None:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["run_id"] == "physical-validation-panel1000"
    assert manifest["split_unit"] == "equilibrium_files"
    assert manifest["model_outputs_computed"] is False
    assert manifest["config"]["postmatch_balance_threshold"] == 0.5
    assert manifest["config"]["aipw_overlap_threshold"] == 0.8
    assert manifest["config"]["stable_threshold_log_Q"] == -1.9
    assert manifest["gradient_set"] == "fixed and varied S01 interpretation panel"
    assert len(manifest["row_ids"]) == len(set(manifest["row_ids"])) == 1000
    expected = {
        "fixed_associations.csv",
        "matched_pairs.csv",
        "matched_effects.csv",
        "doubly_robust_sensitivity.csv",
        "residual_validation.csv",
        "residual_fold_sensitivity.csv",
        "match_distance_sensitivity.csv",
        "candidate_ranking.csv",
        "contradictory_cases.csv",
        "gx_experiment_spec.json",
        "natural_experiment_atlas.png",
        "summary.json",
    }
    assert set(manifest["output_hashes"]) == expected
    for name, digest in manifest["output_hashes"].items():
        assert hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest() == digest


def test_headline_observed_contrasts_and_confounds_are_pinned() -> None:
    matched = _rows("matched_effects.csv")
    adjusted = _rows("doubly_robust_sensitivity.csv")
    geodesic = _one(
        matched,
        candidate="geodesic_curvature_compression",
        outcome="target_native",
        regime="all",
    )
    assert float(geodesic["mean_high_minus_low"]) == pytest.approx(1.316841864482161)
    assert float(geodesic["ci95_lower"]) == pytest.approx(1.1499375293833307)
    assert int(geodesic["matched_pairs"]) == 198
    assert float(geodesic["max_abs_nuisance_smd_after"]) == pytest.approx(
        1.068121850701722
    )
    geodesic_adjusted = _one(
        adjusted,
        candidate="geodesic_curvature_compression",
        outcome="target_native",
        regime="all",
    )
    assert float(geodesic_adjusted["aipw_high_minus_low"]) == pytest.approx(
        0.5588252059354347
    )
    assert float(geodesic_adjusted["overlap_fraction"]) == pytest.approx(0.478)
    assert geodesic_adjusted["method"] == (
        "in_repo_logistic_irls_plus_common_scale_ridge"
    )

    localized_match = _one(
        matched,
        candidate="f_Q_integrand_w25_peak",
        outcome="target_native",
        regime="both_unstable",
    )
    localized_adjusted = _one(
        adjusted,
        candidate="f_Q_integrand_w25_peak",
        outcome="target_native",
        regime="unstable",
    )
    assert float(localized_match["mean_high_minus_low"]) > 0
    assert float(localized_adjusted["aipw_high_minus_low"]) == pytest.approx(
        -0.01118220933455479
    )
    assert float(localized_adjusted["ci95_lower"]) < 0
    assert float(localized_adjusted["ci95_upper"]) > 0
    near_floor = _one(
        matched,
        candidate="geodesic_curvature_compression",
        outcome="target_native",
        regime="either_stable_or_near_floor",
    )
    assert int(near_floor["matched_pairs"]) < 20
    assert near_floor["interval_interpretable"] == "False"


def test_residual_validation_separates_fq_and_paper_baselines() -> None:
    rows = _rows("residual_validation.csv")
    geodesic_paper = _one(
        rows,
        gradient_set="fixed",
        baseline="paper_selected",
        candidate="geodesic_curvature_compression",
        regime="all",
    )
    assert float(geodesic_paper["delta_r2"]) == pytest.approx(0.013937293296463094)
    assert float(geodesic_paper["mse_improvement"]) == pytest.approx(
        0.018801234591043544
    )
    assert float(geodesic_paper["mse_improvement_ci95_lower"]) > 0
    localized_paper = _one(
        rows,
        gradient_set="fixed",
        baseline="paper_selected",
        candidate="f_Q_integrand_w25_peak",
        regime="all",
    )
    assert float(localized_paper["mse_improvement_ci95_lower"]) < 0
    assert float(localized_paper["mse_improvement_ci95_upper"]) > 0
    assert {row["estimand"] for row in rows} == {"native max(log Q, -2)"}
    assert {row["split_unit"] for row in rows} == {"equilibrium_files"}
    near_floor = [row for row in rows if row["regime"] == "stable_or_near_floor"]
    assert near_floor
    assert {row["r2_meaningful"] for row in near_floor} == {"False"}


def test_fold_and_match_distance_sensitivities_are_published() -> None:
    folds = _rows("residual_fold_sensitivity.csv")
    assert len(folds) == 21
    expected_resolved = {
        "geodesic_curvature_compression": 7,
        "f_Q_integrand_w25_peak": 4,
        "bad_curvature_compression": 5,
    }
    for candidate, expected in expected_resolved.items():
        selected = [row for row in folds if row["candidate"] == candidate]
        assert len(selected) == 7
        assert sum(row["mse_improvement_resolved"] == "True" for row in selected) == expected
        assert len({row["model_seed_held_fixed"] for row in selected}) == 1

    distance = _rows("match_distance_sensitivity.csv")
    assert len(distance) == 12
    for candidate in {row["candidate"] for row in distance}:
        all_pairs = _one(distance, candidate=candidate, distance_stratum="all")
        closest = _one(
            distance, candidate=candidate, distance_stratum="best_matched_quarter"
        )
        assert float(closest["mean_native_high_minus_low"]) > 0
        assert float(closest["mean_native_high_minus_low"]) < float(
            all_pairs["mean_native_high_minus_low"]
        )


def test_claim_grades_keep_balance_overlap_and_causality_limits_visible() -> None:
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["fixed_stable_or_near_floor_rows"] == 23
    assert summary["varied_stable_or_near_floor_rows"] == 240
    assert summary["estimand"] == "native max(log Q, -2)"
    assert not summary["causal_claims_made"]
    assert not summary["invalid_perturbations_used"]
    assert summary["claim_gates_applied"] == {
        "postmatch_balance_threshold": 0.5,
        "aipw_overlap_threshold": 0.8,
    }
    ranking = summary["candidate_ranking"]
    assert ranking[0]["candidate"] == "geodesic_curvature_compression"
    assert {row["claim_grade"] for row in ranking} == {"observational-physical"}
    assert not any(row["balance_acceptable"] for row in ranking)
    assert not any(row["overlap_acceptable"] for row in ranking)
    csv_ranking = _rows("candidate_ranking.csv")
    assert [row["candidate"] for row in csv_ranking] == [
        row["candidate"] for row in ranking
    ]
    f_stab = _one(csv_ranking, candidate="f_stab")
    assert f_stab["ranking_residual_baseline"] == (
        "not_applicable_candidate_in_baseline"
    )
    assert f_stab["ranking_residual_comparable"] == "False"
    localized = _one(csv_ranking, candidate="f_Q_integrand_w25_peak")
    assert f_stab["evidence_rank"] == localized["evidence_rank"] == "3"
    assert f_stab["rank_tied"] == localized["rank_tied"] == "True"


def test_pairs_are_equilibrium_disjoint_and_contradictions_are_balanced() -> None:
    pairs = _rows("matched_pairs.csv")
    for candidate in {row["candidate"] for row in pairs}:
        selected = [row for row in pairs if row["candidate"] == candidate]
        high = [row["high_equilibrium_file"] for row in selected]
        low = [row["low_equilibrium_file"] for row in selected]
        assert len(high) == len(set(high))
        assert len(low) == len(set(low))
        assert set(high).isdisjoint(low)
        assert all(float(row["candidate_contrast"]) > 0 for row in selected)
    contradictions = _rows("contradictory_cases.csv")
    assert len(contradictions) == 40
    for candidate in {row["candidate"] for row in contradictions}:
        selected = [row for row in contradictions if row["candidate"] == candidate]
        assert sum(row["case_type"] == "supporting" for row in selected) == 5
        assert sum(row["case_type"] == "contradicting" for row in selected) == 5


def test_gx_spec_is_proposal_only_with_auditable_planning_budget() -> None:
    spec = json.loads((ARTIFACTS / "gx_experiment_spec.json").read_text(encoding="utf-8"))
    assert spec["status"] == "proposal_only_researcher_approval_required"
    assert [row["candidate"] for row in spec["interventions"]] == [
        "geodesic_curvature_compression",
        "bad_curvature_compression",
    ]
    assert {row["validity_tag"] for row in spec["interventions"]} == {
        "plausibly-local"
    }
    budget = spec["compute_estimate"]
    assert budget["standard_runs"] == 24
    assert budget["standard_node_hours"] == 12
    assert budget["convergence_node_hours"] == 12
    assert budget["total_perlmutter_node_hours"] == 32.5
    assert "not a measured Perlmutter pilot" in budget["basis"]
    localized = spec["observed_candidate_separability"]["f_Q_integrand_w25_peak"]
    assert float(localized["spearman_with_log_f_Q"]) == pytest.approx(0.9524282444)
    assert float(
        localized["partial_spearman_with_native_target_given_log_f_Q"]
    ) == pytest.approx(0.1575229498)
    assert float(localized["linear_r2_exposure_from_registered_nuisances"]) > 0.95
    assert spec["pre_budget_feasibility_gate"][
        "required_before_researcher_budget_approval"
    ]
