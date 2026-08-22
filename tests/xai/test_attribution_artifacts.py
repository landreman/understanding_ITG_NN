from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py

from itg_nn.xai.artifacts import sha256_file
from itg_nn.xai.perturbations import ValidityTag


ARTIFACTS = Path("reports/xai/S06a_artifacts")


def _metrics() -> list[dict[str, str]]:
    with (ARTIFACTS / "benchmark_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def test_s06a_published_manifest_and_small_artifact_hashes_are_exact() -> None:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["step"] == "S06a"
    assert manifest["config"]["mode"] == "production"
    assert manifest["gradient_set"] == "varied"
    assert manifest["member_ids"] == ["2864601_0.437"]
    assert len(manifest["row_ids"]) == 128
    assert manifest["checkpoint"]["sha256"] == (
        "d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb"
    )
    assert manifest["dataset"]["sha256"] == (
        "9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad"
    )
    assert manifest["config"]["attribution_module_sha256"] == sha256_file(
        "itg_nn/xai/attribution.py"
    )
    for name in (
        "benchmark_metrics.csv",
        "faithfulness_curves.csv",
        "grouped_uncertainty.csv",
        "ig_convergence.csv",
        "selected_methods.json",
        "selected_review_maps.h5",
        "robust_constant_background.csv",
        "summary.json",
        "toy_controls.json",
    ):
        assert sha256_file(ARTIFACTS / name) == manifest["output_hashes"][name]
    for name in ("pilot_candidate_metrics.csv", "pilot_selected_methods.json"):
        assert sha256_file(ARTIFACTS / name) == manifest[
            "supplementary_artifact_hashes"
        ][name]


def test_s06a_metrics_keep_functions_strata_signs_and_validity_machine_readable() -> None:
    rows = _metrics()
    assert {row["function"] for row in rows} == {
        "original_f",
        "invariant_tilde_f",
    }
    assert {row["stratum"] for row in rows} == {
        "all",
        "stable_or_near_floor",
        "unstable",
    }
    assert len(rows) == 2 * 12 * 3
    allowed = {tag.value for tag in ValidityTag}
    assert all(row["validity_tag"] in allowed for row in rows)
    assert all(row["baseline_validity_tag"] in allowed for row in rows)
    assert {row["signed"] for row in rows} == {"True", "False"}
    assert {row["contribution_valued"] for row in rows} == {"True", "False"}
    assert all(row["artifact_method"] for row in rows)
    expected = next(row for row in rows if row["method"] == "expected_gradients")
    assert expected["batch_layout_adapter"] == "captum_sample_major_to_step_major"
    mask = next(row for row in rows if row["method"] == "periodic_mask")
    assert mask["deterministic_optimizer"] == "True"
    assert mask["registered_baseline_convention"] == "fixed_matched_observed"
    median_mask = next(
        row for row in rows if row["method"] == "periodic_mask_robust_constant"
    )
    assert median_mask["registered_baseline_convention"] == "fixed_robust_constant"


def test_s06a_selected_methods_pass_both_faithfulness_directions_in_each_stratum() -> None:
    selection = json.loads(
        (ARTIFACTS / "selected_methods.json").read_text(encoding="utf-8")
    )
    assert selection["passed"] is True
    assert selection["primary_path_gradient"] == "ig_low_pass"
    assert selection["primary_perturbation"] == "periodic_mask"
    assert selection["perturbation_fallback_used"] is True
    assert selection["eligible"]["periodic_mask"] is False
    assert selection["eligible"]["periodic_mask_robust_constant"] is False
    assert selection["rule"]["maximum_parameter_randomization_correlation"] == 0.95
    assert selection["rule"]["faithfulness_strata"] == [
        "stable_or_near_floor",
        "unstable",
    ]
    assert selection["rule"]["control_aware_stratum"] == "unstable"
    assert selection["rule"]["control_aware_directions"] == [
        "deletion",
        "insertion",
    ]
    assert selection["rule"]["maximum_fixed_baseline_equivariance_relative_rms"] == 2e-5
    selected = {
        selection["primary_path_gradient"],
        selection["primary_perturbation"],
    }
    rows = [
        row
        for row in _metrics()
        if row["function"] == "invariant_tilde_f" and row["method"] in selected
    ]
    assert len(rows) == 6
    assert all(float(row["deletion_margin_vs_random"]) > 0 for row in rows)
    assert all(float(row["insertion_margin_vs_random"]) > 0 for row in rows)
    assert all(float(row["toy_channel_top1"]) == 1 for row in rows)
    assert all(float(row["toy_position_average_precision"]) >= 0.75 for row in rows)
    assert all(row["control_map"] == "absolute_input_minus_baseline" for row in rows)
    all_rows = {row["method"]: row for row in rows if row["stratum"] == "all"}
    assert all(
        row["cyclic_equivariance_baseline_convention"] == "co_shifted"
        for row in all_rows.values()
    )
    assert (
        float(
            all_rows["periodic_mask"][
                "cyclic_equivariance_fixed_baseline_relative_rms"
            ]
        )
        > 0.5
    )
    assert float(
        all_rows["periodic_mask"]["cyclic_equivariance_relative_rms"]
    ) < 1e-4
    assert float(
        all_rows["ig_low_pass"][
            "randomized_map_input_baseline_abs_rank_correlation"
        ]
    ) > float(
        all_rows["ig_low_pass"]["trained_map_input_baseline_abs_rank_correlation"]
    )
    assert all(float(row["parameter_randomization_correlation"]) < 0.95 for row in all_rows.values())


def test_s06a_pilot_selection_artifact_records_baseline_instability() -> None:
    pilot = json.loads(
        (ARTIFACTS / "pilot_selected_methods.json").read_text(encoding="utf-8")
    )
    production = json.loads(
        (ARTIFACTS / "selected_methods.json").read_text(encoding="utf-8")
    )
    assert pilot["passed"] is True
    assert pilot["primary_path_gradient"] == "ig_medoid"
    assert production["primary_path_gradient"] == "ig_low_pass"
    assert pilot["rule"]["faithfulness_strata"] == [
        "stable_or_near_floor",
        "unstable",
    ]
    assert production["rule"]["faithfulness_strata"] == [
        "stable_or_near_floor",
        "unstable",
    ]
    with (ARTIFACTS / "pilot_candidate_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        candidate_rows = list(csv.DictReader(handle))
    assert len(candidate_rows) == 7 * 3
    candidates = {
        (row["method"], row["stratum"]): row for row in candidate_rows
    }
    low_pass_all = candidates[("ig_low_pass", "all")]
    assert float(low_pass_all["deletion_margin_vs_random"]) < 0
    assert float(low_pass_all["insertion_margin_vs_random"]) > 0
    mask_stable = candidates[("periodic_mask", "stable_or_near_floor")]
    assert float(mask_stable["deletion_margin_vs_random"]) > 0
    assert float(mask_stable["insertion_margin_vs_random"]) < 0


def test_s06a_control_aware_production_signs_and_median_background_are_pinned() -> None:
    rows = _metrics()
    low_pass = {
        row["stratum"]: row
        for row in rows
        if row["function"] == "invariant_tilde_f" and row["method"] == "ig_low_pass"
    }
    assert float(
        low_pass["stable_or_near_floor"][
            "deletion_method_minus_control_map_gap_estimate"
        ]
    ) < 0
    assert float(
        low_pass["stable_or_near_floor"][
            "insertion_method_minus_control_map_gap_estimate"
        ]
    ) < 0
    assert float(
        low_pass["unstable"]["deletion_method_minus_control_map_gap_ci_lower"]
    ) > 0
    assert float(
        low_pass["unstable"]["insertion_method_minus_control_map_gap_ci_lower"]
    ) > 0

    median_mask = {
        row["stratum"]: row
        for row in rows
        if row["function"] == "invariant_tilde_f"
        and row["method"] == "periodic_mask_robust_constant"
    }
    assert float(median_mask["stable_or_near_floor"]["insertion_margin_vs_random"]) < 0
    assert float(
        median_mask["all"]["cyclic_equivariance_fixed_baseline_relative_rms"]
    ) > 2e-5

    with (ARTIFACTS / "robust_constant_background.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        background = list(csv.DictReader(handle))
    assert [int(row["channel"]) for row in background] == list(range(7))
    assert [float(row["z_constant_median"]) for row in background] == [
        1.0960736274719238,
        -0.06282701343297958,
        -0.02537703514099121,
        8.744433040043639e-16,
        1.8279118537902832,
        -8.811368759909226e-16,
        1.5146362781524658,
    ]


def test_s06a_selected_faithfulness_margins_have_grouped_intervals() -> None:
    with (ARTIFACTS / "grouped_uncertainty.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    margins = [
        row
        for row in rows
        if row["function"] == "invariant_tilde_f"
        and row["method"] in {"ig_low_pass", "periodic_mask"}
        and row["statistic"]
        in {"deletion_margin_vs_random", "insertion_margin_vs_random"}
    ]
    assert len(margins) == 2 * 2 * 3
    assert all(row["resampling_unit"] == "equilibrium_files" for row in margins)
    assert all(int(row["replicates"]) == 500 for row in margins)
    assert all(float(row["ci_lower"]) <= float(row["estimate"]) for row in margins)
    assert all(float(row["estimate"]) <= float(row["ci_upper"]) for row in margins)
    assert all(float(row["native_gap_ci_lower"]) <= float(row["native_gap_estimate"]) for row in margins)
    assert all(float(row["native_gap_estimate"]) <= float(row["native_gap_ci_upper"]) for row in margins)
    assert all(
        float(row["oriented_native_gap_ci_lower"])
        <= float(row["oriented_native_gap_estimate"])
        <= float(row["oriented_native_gap_ci_upper"])
        for row in margins
    )
    assert all(
        float(row["per_row_oriented_native_gap_ci_lower"])
        <= float(row["per_row_oriented_native_gap_estimate"])
        <= float(row["per_row_oriented_native_gap_ci_upper"])
        for row in margins
    )
    assert all(0 <= float(row["row_favouring_fraction_estimate"]) <= 1 for row in margins)
    assert all(row["control_map"] == "absolute_input_minus_baseline" for row in margins)
    assert all(
        float(row["control_map_per_row_oriented_native_gap_ci_lower"])
        <= float(row["control_map_per_row_oriented_native_gap_estimate"])
        <= float(row["control_map_per_row_oriented_native_gap_ci_upper"])
        for row in margins
    )
    assert all(
        float(row["method_minus_control_map_gap_ci_lower"])
        <= float(row["method_minus_control_map_gap_estimate"])
        <= float(row["method_minus_control_map_gap_ci_upper"])
        for row in margins
    )
    assert all(row["denominator_estimate"] for row in margins)
    assert all(0 <= float(row["denominator_negative_fraction"]) <= 1 for row in margins)
    assert all(
        0 <= float(row["denominator_abs_below_0_005_fraction"]) <= 1
        for row in margins
    )


def test_s06a_review_maps_are_native_member_level_and_axis_labeled() -> None:
    with h5py.File(ARTIFACTS / "selected_review_maps.h5", "r") as h5_file:
        assert h5_file.attrs["estimand"] == "native max(log Q, -2)"
        assert h5_file.attrs["research_source"].startswith("canonical external HDF5")
        assert h5_file["attribution"].shape == (2, 2, 1, 16, 7, 96)
        assert h5_file["canonical_minus_original"].shape == (2, 1, 16, 7, 96)
        assert len(h5_file["row_id"]) == 16
        assert json.loads(h5_file["attribution"].attrs["axes"]) == [
            "function",
            "method",
            "member",
            "sample",
            "channel",
            "z",
        ]
