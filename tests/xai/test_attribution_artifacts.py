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
    for name in (
        "benchmark_metrics.csv",
        "faithfulness_curves.csv",
        "grouped_uncertainty.csv",
        "ig_convergence.csv",
        "selected_methods.json",
        "selected_review_maps.h5",
        "summary.json",
        "toy_controls.json",
    ):
        assert sha256_file(ARTIFACTS / name) == manifest["output_hashes"][name]


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
    assert len(rows) == 2 * 11 * 3
    allowed = {tag.value for tag in ValidityTag}
    assert all(row["validity_tag"] in allowed for row in rows)
    assert all(row["baseline_validity_tag"] in allowed for row in rows)
    assert {row["signed"] for row in rows} == {"True", "False"}
    assert {row["contribution_valued"] for row in rows} == {"True", "False"}
    assert all(row["artifact_method"] for row in rows)


def test_s06a_selected_methods_pass_both_faithfulness_directions_in_each_stratum() -> None:
    selection = json.loads(
        (ARTIFACTS / "selected_methods.json").read_text(encoding="utf-8")
    )
    assert selection["passed"] is True
    assert selection["primary_path_gradient"] == "ig_low_pass"
    assert selection["primary_perturbation"] == "periodic_mask"
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
    all_rows = {row["method"]: row for row in rows if row["stratum"] == "all"}
    assert all(float(row["parameter_randomization_correlation"]) < 0.95 for row in all_rows.values())


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
