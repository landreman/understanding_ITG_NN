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
    assert all(
        float(row["selection_null_max"]) >= float(row["selection_null_q95"])
        for row in unstable
    )
    assert all(row["overlap_orientation"] for row in rows)
    assert all(int(row["learned_constant_profile_count"]) >= 0 for row in rows)
    assert all(int(row["learned_active_profile_count"]) >= 0 for row in rows)
    assert all(
        int(row["learned_constant_profile_count"])
        + int(row["learned_active_profile_count"])
        == int(row["sample_count"])
        for row in rows
    )
    assert all(float(row["learned_mask_width_mean"]) >= 10.0 for row in rows)
    assert all(float(row["gx_mask_width_mean"]) >= 10.0 for row in rows)
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


def test_s07_lag_curve_artifact_preserves_rank_and_raw_value_curves() -> None:
    spatial = _csv("spatial_alignment.csv")
    curves = _csv("lag_curves.csv")
    key_fields = (
        "source_family",
        "source_id",
        "member_id",
        "function",
        "method",
        "gradient_set",
        "stratum",
        "mode",
    )
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in curves:
        grouped.setdefault(tuple(row[field] for field in key_fields), []).append(row)

    assert len(curves) == len(spatial) * 96
    assert len(grouped) == len(spatial)
    maximum_curve_difference = 0.0
    for row in spatial:
        key = tuple(row[field] for field in key_fields)
        curve = sorted(grouped[key], key=lambda item: int(item["lag_index_0_to_95"]))
        assert [int(item["lag_index_0_to_95"]) for item in curve] == list(range(96))
        rank = np.asarray(
            [float(item["mean_within_tube_spearman"]) for item in curve]
        )
        cross = np.asarray(
            [float(item["mean_within_tube_cross_correlation"]) for item in curve]
        )
        selected = int(row["best_lag_index_0_to_95"])
        assert int(np.argmax(np.abs(rank))) == selected
        assert rank[selected] == np.float64(row["circular_spearman"])
        maximum_curve_difference = max(
            maximum_curve_difference, float(np.max(np.abs(rank - cross)))
        )
    assert maximum_curve_difference > 0.09


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
    assert float(density["circular_spearman_ci95_lower"]) == np.float64(
        -0.3883653797328812
    )
    assert float(density["circular_spearman_ci95_upper"]) == np.float64(
        -0.332533286776065
    )
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
    assert int(density["learned_constant_profile_count"]) == 17
    assert int(density["learned_active_profile_count"]) == 743
    assert float(density["circular_spearman_active_learned_profiles"]) == np.float64(
        -0.3687605907617687
    )

    later_density = one(
        source_id="2864601_0.437:u008",
        gradient_set="varied",
        stratum="unstable",
        mode="signed",
    )
    assert (
        float(later_density["circular_spearman_ci95_lower"]),
        float(later_density["circular_spearman_ci95_upper"]),
    ) == (
        np.float64(-0.2943914186610519),
        np.float64(-0.24143675281600369),
    )

    mostly_silent = one(
        source_id="2864601_0.437:u003",
        gradient_set="varied",
        stratum="unstable",
        mode="signed",
    )
    assert int(mostly_silent["learned_constant_profile_count"]) == 605
    assert float(mostly_silent["learned_constant_profile_fraction"]) == np.float64(
        0.7960526315789473
    )
    assert float(mostly_silent["circular_spearman_active_learned_profiles"]) == (
        np.float64(0.1677936576993047)
    )
    assert float(mostly_silent["learned_mask_width_mean"]) == np.float64(
        80.84736842105264
    )

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

    original_signed = one(
        source_family="s06_attribution",
        member_id="2864601_0.437",
        function="original_f",
        method="ig_low_pass",
        gradient_set="varied",
        stratum="unstable",
        mode="signed",
    )
    original_positive = one(
        source_family="s06_attribution",
        member_id="2864601_0.437",
        function="original_f",
        method="ig_low_pass",
        gradient_set="varied",
        stratum="unstable",
        mode="positive_contribution",
    )
    assert float(original_signed["circular_spearman"]) == np.float64(
        -0.013272392231391003
    )
    assert float(original_positive["circular_spearman"]) == np.float64(
        0.2300460407141299
    )
    assert original_signed["circular_spearman"] != signed["circular_spearman"]
    assert original_positive["circular_spearman"] != positive["circular_spearman"]

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
    assert all(
        int(row["learned_zero_summary_count"])
        + int(row["learned_active_summary_count"])
        == int(row["sample_count"])
        for row in zonal
    )
    assert all(
        row["zero_summary_definition"] == "exact_zero_after_summary"
        for row in zonal
    )
    geodesic = [
        row
        for row in zonal
        if row["source_id"] == "2864601_0.437:u008"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
    ]
    assert len(geodesic) == 1
    assert float(geodesic[0]["spearman_rho"]) == np.float64(-0.12162944609886839)
    assert (
        float(geodesic[0]["spearman_ci95_lower"]),
        float(geodesic[0]["spearman_ci95_upper"]),
    ) == (
        np.float64(-0.18339575629260024),
        np.float64(-0.06012982420600932),
    )

    mostly_silent_zonal = [
        row
        for row in zonal
        if row["source_id"] == "2864601_0.437:u003"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
    ]
    assert len(mostly_silent_zonal) == 1
    silent_row = mostly_silent_zonal[0]
    assert int(silent_row["learned_zero_summary_count"]) == 605
    assert int(silent_row["learned_active_summary_count"]) == 155
    assert float(silent_row["spearman_active_summary"]) == np.float64(
        0.07426122264831943
    )
    assert (
        float(silent_row["spearman_active_summary_ci95_lower"]),
        float(silent_row["spearman_active_summary_ci95_upper"]),
    ) == (
        np.float64(-0.06860759699198943),
        np.float64(0.2267801678826823),
    )
    assert silent_row["active_summary_bootstrap_stable"] == "False"

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
    assert float(physical[0]["ci95_lower"]) == np.float64(0.7083975230968925)
    assert float(physical[0]["ci95_upper"]) == np.float64(0.7618774315953174)
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

    def paired_one(**expected: str) -> dict[str, str]:
        matches = [
            row
            for row in paired
            if all(row[key] == value for key, value in expected.items())
        ]
        assert len(matches) == 1
        return matches[0]

    observed_effect = paired_one(
        quantity="observed_clipped_log_Q",
        stratum="both_unstable",
    )
    assert float(observed_effect["estimate"]) == np.float64(0.9570316134145961)
    assert (
        float(observed_effect["ci95_lower"]),
        float(observed_effect["ci95_upper"]),
    ) == (
        np.float64(0.8264462663869981),
        np.float64(1.0759141859154944),
    )
    zonal_effect = paired_one(
        quantity="log10_zonal_phi2",
        stratum="both_unstable",
    )
    assert float(zonal_effect["estimate"]) == np.float64(0.6492487514893418)
    member_effect = paired_one(
        quantity="member_prediction",
        member_id="2864601_0.437",
        function="original_f",
        stratum="all",
    )
    assert float(member_effect["estimate"]) == np.float64(1.5445257106721402)
    attribution_effect = paired_one(
        quantity="learned_Qz_spatial_spearman",
        source_family="s06_attribution",
        member_id="2864601_0.371",
        function="invariant_tilde_f",
        method="ig_low_pass",
        mode="signed",
        stratum="both_unstable",
    )
    assert float(attribution_effect["estimate"]) == np.float64(
        0.026322870037569945
    )
    density_effect = paired_one(
        quantity="learned_Qz_spatial_spearman",
        source_family="s05_density",
        source_id="2864601_0.409:u021",
        mode="signed",
        stratum="all",
    )
    assert float(density_effect["estimate"]) == np.float64(0.27788018916618856)
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
    observed_gx = [
        row
        for row in paired
        if row["quantity"] in {"observed_clipped_log_Q", "log10_zonal_phi2"}
        and row["stratum"] == "both_unstable"
    ]
    assert len(observed_gx) == 2
    assert all(row["plasma_claims_permitted"] == "True" for row in observed_gx)
    predictions = [
        row
        for row in paired
        if row["quantity"] == "member_prediction" and row["stratum"] == "both_unstable"
    ]
    assert predictions
    assert all(row["plasma_claims_permitted"] == "False" for row in predictions)

    cases = _csv("case_studies.csv")
    assert len(cases) == 20
    assert Counter((row["hypothesis"], row["case_type"]) for row in cases) == {
        ("bad_curvature_flux_compression", "supporting"): 5,
        ("bad_curvature_flux_compression", "contradicting"): 5,
        ("radial_drift_geodesic_curvature_zonal_flow", "supporting"): 5,
        ("radial_drift_geodesic_curvature_zonal_flow", "contradicting"): 5,
    }
    assert all(row["validity_tag"] == "observed-comparison" for row in cases)
    assert all(row["feature_claims_permitted"] == "True" for row in cases)
    assert all(row["plasma_claims_permitted"] == "True" for row in cases)
    spatial = _csv("spatial_alignment.csv")
    population_sign = {
        (row["member_id"], row["source_id"]): int(
            np.sign(float(row["circular_spearman"]))
        )
        for row in spatial
        if row["source_family"] == "s05_density"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
        and row["mode"] == "signed"
    }
    assert all(
        int(row["expected_sign"])
        == population_sign[(row["member_id"], row["unit_id"])]
        for row in cases
    )
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

    spatial = _csv("spatial_alignment.csv")
    zonal_rows = _csv("zonal_association.csv")
    paired_rows = _csv("paired_analysis.csv")
    lag_rows = _csv("lag_curves.csv")
    summary_counts = summary["counts"]
    assert summary_counts == {
        "association_bootstrap_stable_spatial_rows": sum(
            row["association_bootstrap_stable"] == "True" for row in spatial
        ),
        "case_study_rows": len(cases),
        "lag_bootstrap_stable_spatial_rows": sum(
            row["lag_bootstrap_stable"] == "True" for row in spatial
        ),
        "lag_curve_rows": len(lag_rows),
        "paired_analysis_rows": len(paired_rows),
        "spatial_alignment_rows": len(spatial),
        "zonal_association_rows": len(zonal_rows),
    }
    assert summary_counts["lag_bootstrap_stable_spatial_rows"] == 161
    assert summary_counts["association_bootstrap_stable_spatial_rows"] == 203

    def assert_summary_row(block: dict[str, object], row: dict[str, str]) -> None:
        for key, expected in block.items():
            assert key in row
            if isinstance(expected, bool):
                actual: object = row[key] == "True"
            elif isinstance(expected, int):
                actual = int(row[key])
            elif isinstance(expected, float):
                actual = float(row[key])
            else:
                actual = row[key]
            assert actual == expected

    headline = summary["headline"]
    density_headline = headline["strongest_varied_unstable_density_Qz"]
    density_match = [
        row
        for row in spatial
        if row["source_id"] == density_headline["source_id"]
        and row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
        and row["mode"] == "signed"
    ]
    assert len(density_match) == 1
    assert density_headline["source_id"] == "2864601_0.437:u001"
    assert_summary_row(density_headline, density_match[0])

    ig_headline = headline["strongest_varied_unstable_canonical_ig_Qz"]
    ig_match = [
        row
        for row in spatial
        if row["source_family"] == "s06_attribution"
        and row["member_id"] == ig_headline["member_id"]
        and row["function"] == "invariant_tilde_f"
        and row["method"] == "ig_low_pass"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
        and row["mode"] == "signed"
    ]
    assert len(ig_match) == 1
    assert ig_headline["member_id"] == "2864601_0.437"
    assert_summary_row(ig_headline, ig_match[0])

    zonal_headline = headline["strongest_varied_unstable_zonal_association"]
    zonal_match = [
        row
        for row in zonal_rows
        if row["source_id"] == zonal_headline["source_id"]
        and row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
    ]
    assert len(zonal_match) == 1
    assert zonal_headline["source_id"] == "2864601_0.437:u003"
    assert_summary_row(zonal_headline, zonal_match[0])

    function_differences = headline[
        "canonical_original_ig_spearman_difference_by_member"
    ]
    assert {row["member_id"] for row in function_differences} == {
        "2864601_0.437",
        "2864601_0.371",
        "2864601_0.409",
    }
    differences = [row["canonical"] - row["original"] for row in function_differences]
    np.testing.assert_allclose(
        differences,
        [-0.007949790015641457, -0.005696698101801684, -0.002132017764995072],
    )
    assert all(value != 0.0 for value in differences)


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
