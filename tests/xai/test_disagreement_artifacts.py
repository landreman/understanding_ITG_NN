import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "reports" / "xai" / "S11_artifacts"


def _rows(name):
    with (ARTIFACTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_summary_reports_common_mode_failure_without_confidence_claim():
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "disagreement-all100-panel1000"
    assert summary["canonical_function"] == "invariant_tilde_f"
    assert summary["estimand"] == "native max(log Q, -2)"
    assert summary["members"] == 100
    assert summary["rows"] == 1000
    assert summary["stable_or_near_floor_rows"] == 240
    assert summary["unstable_rows"] == 760
    assert summary["common_mode_failure_rows"] == 8
    assert 0.7 < summary["spread_error_spearman"] < 0.8
    assert summary["model_spread_interpretation"] == "member dispersion, not a confidence interval"
    assert summary["residual_feature_selection"] == "none_frozen_before_residual_analysis"


def test_manifest_hashes_every_published_small_artifact():
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["member_ids"] and len(manifest["member_ids"]) == 100
    assert len(manifest["row_ids"]) == 1000
    assert manifest["gradient_set"] == "varied frozen S01 interpretation panel"
    assert manifest["checkpoint"]["sha256"] == "d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb"
    assert manifest["dataset"]["sha256"] == "9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad"
    for path in ARTIFACTS.iterdir():
        if path.name == "manifest.json":
            continue
        assert manifest["output_hashes"][path.name] == _sha256(path)


def test_review_proxy_retains_signed_member_and_spread_gradient_axes():
    with h5py.File(ARTIFACTS / "selected_review_diagnostics.h5", "r") as handle:
        assert handle["prediction_native"].shape == (100, 16)
        assert handle["ensemble_spread_gradient"].shape == (16, 96, 7)
        assert handle["member_residual_gradient"].shape == (10, 16, 96, 7)
        assert handle["original_shift_signed_native"].shape == (10, 16)
        assert json.loads(handle["prediction_native"].attrs["axes"]) == ["member", "sample"]
        assert json.loads(handle["ensemble_spread_gradient"].attrs["axes"]) == ["sample", "z", "channel"]
        assert json.loads(handle["member_residual_gradient"].attrs["axes"]) == [
            "gradient_member", "sample", "z", "channel"
        ]
        assert json.loads(handle["original_shift_signed_native"].attrs["axes"]) == [
            "gradient_member", "sample"
        ]
        assert bool(handle.attrs["signs_retained"])
        assert handle.attrs["estimand"] == "native max(log Q, -2)"


def test_diagnostics_use_frozen_features_and_equilibrium_resampling():
    rows = _rows("diagnostic_associations.csv")
    assert len(rows) == 14 * 2 * 3
    assert {row["outcome"] for row in rows} == {"ensemble_spread", "ensemble_absolute_error"}
    assert {row["regime"] for row in rows} == {"all", "stable_or_near_floor", "unstable"}
    assert {row["resampling_unit"] for row in rows} == {"equilibrium_files"}
    assert {row["feature_selection"] for row in rows} == {"none_frozen_before_residual_analysis"}
    assert {row["interval_kind"] for row in rows} == {"grouped_resample_sensitivity_interval"}

    crossfit = _rows("crossfit_diagnostics.csv")
    assert {row["split_unit"] for row in crossfit} == {"equilibrium_files"}
    assert {row["feature_selection"] for row in crossfit} == {"none_frozen_before_residual_analysis"}


def test_common_mode_counts_are_separate_by_output_regime():
    rows = _rows("failure_categories.csv")
    expected = {"all": 1000, "stable_or_near_floor": 240, "unstable": 760}
    for regime, count in expected.items():
        selected = [row for row in rows if row["regime"] == regime]
        assert sum(int(row["count"]) for row in selected) == count
    common = {row["regime"]: int(row["count"]) for row in rows if row["category"] == "common_mode_failure"}
    assert common == {"all": 8, "stable_or_near_floor": 2, "unstable": 6}

    sensitivity = _rows("failure_threshold_sensitivity.csv")
    assert len(sensitivity) == 27
    primary = [row for row in sensitivity if row["threshold_status"] == "registered_primary"]
    assert {row["regime"]: int(row["common_mode_failure_count"]) for row in primary} == common
    assert len({int(row["common_mode_failure_count"]) for row in sensitivity if row["regime"] == "all"}) > 1


def test_exact_shift_is_null_and_off_manifold_alignment_edit_is_not():
    rows = [
        row for row in _rows("perturbation_summary.csv")
        if row["outcome"] == "ensemble_spread" and row["regime"] == "all"
    ]
    by_name = {row["perturbation"]: row for row in rows}
    assert by_name["random_joint_shift"]["validity"] == "exact_symmetry"
    assert float(by_name["random_joint_shift"]["rms_change_native"]) < 2e-5
    assert by_name["independent_channel_shifts"]["validity"] == "deliberately_off_manifold_diagnostic"
    assert float(by_name["independent_channel_shifts"]["signed_mean_change_native"]) > 0.1


def test_gradient_table_keeps_variance_mean_and_member_residual_estimands_distinct():
    rows = _rows("gradient_summary.csv")
    assert {row["outcome"] for row in rows} == {
        "ensemble_spread", "ensemble_mean_prediction", "member_residual"
    }
    assert {row["regime"] for row in rows} == {"all", "stable_or_near_floor", "unstable"}
    assert {int(row["channel"]) for row in rows} == set(range(7))
    assert {row["signs_retained"] for row in rows} == {"True"}
    assert {row["estimand"] for row in rows} == {"native max(log Q, -2)"}


def test_member_symmetry_and_spread_error_diagnostics_are_published():
    symmetry = _rows("member_symmetry_associations.csv")
    assert len(symmetry) == 6
    assert {row["feature"] for row in symmetry} == {"member_mean_absolute_shift_error_top10"}
    assert {row["resampling_unit"] for row in symmetry} == {"equilibrium_files"}

    spread_error = _rows("spread_error_associations.csv")
    assert len(spread_error) == 3
    assert {row["left_outcome"] for row in spread_error} == {"ensemble_spread"}
    assert {row["right_outcome"] for row in spread_error} == {"ensemble_absolute_error"}
    assert all(float(row["spearman_interval_lower"]) <= float(row["spearman"]) <= float(row["spearman_interval_upper"]) for row in spread_error)

    signed = _rows("signed_member_symmetry_changes.csv")
    assert len(signed) == 10_000
    assert len({row["member_id"] for row in signed}) == 10
    assert {row["function"] for row in signed} == {"original_f"}
    assert {row["validity"] for row in signed} == {"exact_symmetry"}
    changes = np.asarray([float(row["signed_change_native"]) for row in signed])
    absolute = np.asarray([float(row["absolute_change_native"]) for row in signed])
    assert changes.min() < 0 < changes.max()
    np.testing.assert_allclose(absolute, np.abs(changes))
