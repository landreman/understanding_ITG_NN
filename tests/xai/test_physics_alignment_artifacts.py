from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from itg_nn.xai.artifacts import sha256_file


ARTIFACTS = Path("reports/xai/S07_artifacts")


def _csv(name: str) -> list[dict[str, str]]:
    with (ARTIFACTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_s07_manifest_hashes_published_artifacts_and_registered_inputs() -> None:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["step"] == "S07"
    assert manifest["config"]["mode"] == "production"
    assert manifest["config"]["run_id"] == "physics-alignment-top3-panel1000"
    assert manifest["gradient_set"] == "fixed_and_varied_separate"
    assert len(manifest["member_ids"]) == 3
    assert len(manifest["row_ids"]) == 2000
    assert len(set(manifest["row_ids"])) == 1000
    assert manifest["checkpoint"]["sha256"] == (
        "d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb"
    )
    assert manifest["dataset"]["sha256"] == (
        "9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad"
    )
    assert manifest["config"]["upstream_s06b_map_sha256"] == (
        "ab848646a7c0032fd3e2f40dab1498b3143114257ba19d36c6ae652175b8930b"
    )
    for name in manifest["config"]["published_artifacts"]:
        assert sha256_file(ARTIFACTS / name) == manifest["output_hashes"][name]


def test_s07_spatial_table_keeps_functions_strata_signs_lags_and_grouping() -> None:
    rows = _csv("spatial_alignment.csv")
    assert len(rows) == 216
    assert {row["source_family"] for row in rows} == {
        "s05_density",
        "s06_attribution",
    }
    assert {row["gradient_set"] for row in rows} == {"fixed", "varied"}
    assert {row["stratum"] for row in rows} == {
        "all",
        "stable_or_near_floor",
        "unstable",
    }
    attribution = [row for row in rows if row["source_family"] == "s06_attribution"]
    assert {row["function"] for row in attribution} == {
        "original_f",
        "invariant_tilde_f",
    }
    assert {row["mode"] for row in rows} == {
        "signed",
        "positive_contribution",
    }
    assert all(row["bootstrap_unit"] == "equilibrium_files" for row in rows)
    assert all(int(row["bootstrap_replicates"]) == 500 for row in rows)
    assert all(int(row["lag_stability_tolerance_positions"]) == 4 for row in rows)
    unstable = [row for row in rows if row["stratum"] == "unstable"]
    assert all(int(row["selection_null_permutations"]) == 200 for row in unstable)
    assert all(row["selection_null_q95"] for row in unstable)
    assert all(row["overlap_orientation"] for row in rows)
    assert all(row["validity_tag"] for row in rows)
    stable = [row for row in rows if row["stratum"] == "stable_or_near_floor"]
    assert all(row["feature_claims_permitted"] == "False" for row in stable)
    masks = [row for row in rows if row["method"] == "periodic_mask"]
    assert {row["mode"] for row in masks} == {"positive_contribution"}
    assert all(row["feature_claims_permitted"] == "False" for row in masks)
    assert all(row["plasma_claims_permitted"] == "False" for row in attribution)
    density_unstable = [
        row
        for row in rows
        if row["source_family"] == "s05_density" and row["stratum"] == "unstable"
    ]
    assert all(row["plasma_claims_permitted"] == "True" for row in density_unstable)


def test_s07_headline_spatial_results_pin_signed_and_positive_conclusions() -> None:
    rows = _csv("spatial_alignment.csv")

    def one(**expected: str) -> dict[str, str]:
        matches = [
            row
            for row in rows
            if all(row[key] == value for key, value in expected.items())
        ]
        assert len(matches) == 1
        return matches[0]

    density = one(
        source_id="2864601_0.437:u001",
        gradient_set="varied",
        stratum="unstable",
        mode="signed",
    )
    assert float(density["circular_spearman"]) == np.float64(-0.360511998599993)
    assert float(density["circular_spearman_ci95_upper"]) < -0.33
    assert int(density["best_lag"]) == 22
    assert float(density["best_lag_within_tolerance_recurrence"]) == 0.912
    assert float(density["overlap_enrichment"]) > 1.63
    assert density["overlap_orientation"] == (
        "gx_profile_sign_flipped_to_match_negative_association"
    )
    assert float(density["selection_null_q95"]) == np.float64(0.04321620846108337)
    assert density["lag_search_null_resolved"] == "True"
    assert density["association_bootstrap_stable"] == "True"
    assert density["lag_bootstrap_stable"] == "True"

    signed = one(
        source_family="s06_attribution",
        member_id="2864601_0.437",
        function="invariant_tilde_f",
        method="ig_low_pass",
        gradient_set="varied",
        stratum="unstable",
        mode="signed",
    )
    positive = one(
        source_family="s06_attribution",
        member_id="2864601_0.437",
        function="invariant_tilde_f",
        method="ig_low_pass",
        gradient_set="varied",
        stratum="unstable",
        mode="positive_contribution",
    )
    assert float(signed["circular_spearman"]) == np.float64(-0.02122218224703246)
    assert int(signed["best_lag"]) == -36
    assert float(positive["circular_spearman"]) == np.float64(0.2657020267671119)
    assert int(positive["best_lag"]) == 1
    assert float(positive["overlap_enrichment"]) > 2.38
    assert positive["overlap_orientation"] == "gx_profile_unflipped"
    assert signed["lag_search_null_resolved"] == "True"

    unresolved = one(
        source_family="s06_attribution",
        member_id="2864601_0.371",
        function="invariant_tilde_f",
        method="ig_low_pass",
        gradient_set="varied",
        stratum="unstable",
        mode="signed",
    )
    assert float(unresolved["circular_spearman"]) == np.float64(-0.012763898379259002)
    assert float(unresolved["selection_null_q95"]) > abs(
        float(unresolved["circular_spearman"])
    )
    assert unresolved["lag_search_null_resolved"] == "False"


def test_s07_zonal_pairing_cases_and_symmetry_keep_negative_results() -> None:
    zonal = _csv("zonal_association.csv")
    geodesic = [
        row
        for row in zonal
        if row["source_id"] == "2864601_0.437:u008"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
    ]
    assert len(geodesic) == 1
    assert float(geodesic[0]["spearman_rho"]) == np.float64(-0.12162944609886839)
    assert float(geodesic[0]["spearman_ci95_upper"]) < -0.06

    paired = _csv("paired_analysis.csv")
    physical = [
        row
        for row in paired
        if row["analysis_kind"] == "physical_Qz_fixed_vs_varied_same_geometry"
        and row["mode"] == "signed"
        and row["stratum"] == "all"
    ]
    assert len(physical) == 1
    assert int(physical[0]["best_lag"]) == 0
    assert float(physical[0]["circular_spearman"]) == np.float64(0.7355389542305631)
    assert float(physical[0]["lag_recurrence"]) == 1.0

    assert len(paired) == 138
    assert {row["stratum"] for row in paired} == {
        "all",
        "either_stable_or_near_floor",
        "both_unstable",
    }
    assert {(row["stratum"], int(row["sample_count"])) for row in paired} == {
        ("all", 1000),
        ("either_stable_or_near_floor", 251),
        ("both_unstable", 749),
    }
    attribution_pairs = [
        row for row in paired if row["source_family"] == "s06_attribution"
    ]
    assert all(
        row["validity_tag"] == "deliberately_off_manifold_diagnostic"
        for row in attribution_pairs
    )
    assert all(row["plasma_claims_permitted"] == "False" for row in attribution_pairs)
    primary_both_unstable = [
        row
        for row in attribution_pairs
        if row["method"] == "ig_low_pass" and row["stratum"] == "both_unstable"
    ]
    assert primary_both_unstable
    assert all(
        row["feature_claims_permitted"] == "True" for row in primary_both_unstable
    )

    cases = _csv("case_studies.csv")
    assert len(cases) == 20
    assert Counter((row["hypothesis"], row["case_type"]) for row in cases) == {
        ("bad_curvature_flux_compression", "supporting"): 5,
        ("bad_curvature_flux_compression", "contradicting"): 5,
        ("radial_drift_geodesic_curvature_zonal_flow", "supporting"): 5,
        ("radial_drift_geodesic_curvature_zonal_flow", "contradicting"): 5,
    }
    assert all(row["validity_tag"] == "observed-comparison" for row in cases)
    assert all(
        np.sign(float(row["score"])) == int(row["expected_sign"])
        for row in cases
        if row["case_type"] == "supporting"
    )
    assert all(
        np.sign(float(row["score"])) != int(row["expected_sign"])
        for row in cases
        if row["case_type"] == "contradicting"
    )

    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["symmetry"]["s07_joint_shift_lag_curve_max_abs_error"] == 0.0
    assert summary["stable_feature_claims_permitted"] is False


def test_s07_large_artifact_preserves_member_sample_and_position_axes() -> None:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    path = Path("output/xai/S07/physics-alignment-top3-panel1000/alignment_details.h5")
    if not path.is_file():
        return
    assert sha256_file(path) == manifest["output_hashes"]["alignment_details.h5"]
    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs["estimand"] == "native max(log Q, -2)"
        assert bool(h5_file.attrs["member_level_signed_before_aggregation"])
        assert not bool(h5_file.attrs["stable_feature_claims_permitted"])
        assert h5_file["density"].shape == (3, 1000, 3, 96)
        assert h5_file["prediction"].shape == (2, 3, 2, 1000)
        assert h5_file["q_vs_z"].shape == (2, 1000, 96)
        assert json.loads(h5_file["density"].attrs["axes"]) == [
            "member",
            "sample",
            "selected_unit",
            "z",
        ]
