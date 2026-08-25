from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "reports/xai/S12_artifacts"


def _rows(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader((ARTIFACTS / name).open(newline="", encoding="utf-8")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    return 1.0 - float(np.sum((target - prediction) ** 2)) / float(
        np.sum((target - target.mean()) ** 2)
    )


def test_s12_manifest_pins_inputs_sources_packages_and_outputs() -> None:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    assert manifest["config"]["run_id"] == "distillation-top3-panel1000"
    assert manifest["split_unit"] == "equilibrium_files"
    assert manifest["gradient_set"] == "varied frozen S01 interpretation panel"
    assert len(manifest["row_ids"]) == 1000
    assert len(set(manifest["row_ids"])) == 1000
    assert manifest["checkpoint"]["sha256"] == (
        "d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb"
    )
    assert manifest["dataset"]["sha256"] == (
        "9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad"
    )
    assert manifest["package_versions"]["interpret-core"] == "0.7.8"
    assert manifest["package_versions"]["numpy"] == "1.26.4"
    for name, digest in manifest["output_hashes"].items():
        assert _sha256(ARTIFACTS / name) == digest
    assert manifest["config"]["script_sha256"] == _sha256(
        ROOT / "scripts/xai_s12_distillation.py"
    )
    assert manifest["config"]["distillation_module_sha256"] == _sha256(
        ROOT / "itg_nn/xai/distillation.py"
    )


def test_s12_primary_fidelity_recomputes_from_signed_row_artifact() -> None:
    residuals = _rows("primary_residuals.csv")
    fidelity = _rows("fidelity.csv")
    all_fidelity = {
        (row["target_kind"], row["target_id"]): row
        for row in fidelity
        if row["regime"] == "all" and row["target_kind"] != "bottleneck_unit"
    }
    assert len(all_fidelity) == 5
    for key, expected in all_fidelity.items():
        selected = [
            row
            for row in residuals
            if (row["target_kind"], row["target_id"]) == key
        ]
        assert len(selected) == 1000
        assert len({row["equilibrium_file"] for row in selected}) == 1000
        target = np.asarray([float(row["target_native_value"]) for row in selected])
        prediction = np.asarray([float(row["ebm_oof_prediction"]) for row in selected])
        np.testing.assert_allclose(_r2(target, prediction), float(expected["held_out_r2"]))
        assert {row["split_unit"] for row in selected} == {"equilibrium_files"}
        assert np.sum([row["stable_or_near_floor"] == "True" for row in selected]) == 240


def test_s12_unit_attrition_and_feature_stability_are_not_hidden() -> None:
    fidelity = _rows("fidelity.csv")
    units = [
        row
        for row in fidelity
        if row["regime"] == "all" and row["target_kind"] == "bottleneck_unit"
    ]
    values = np.asarray([float(row["held_out_r2"]) for row in units])
    assert len(values) == 64
    assert np.isnan(values).sum() == 5
    assert np.sum(values >= 0.8) == 13
    np.testing.assert_allclose(np.nanmedian(values), 0.5942008791296616)
    recurrence = _rows("term_recurrence.csv")
    assert len(recurrence) == 5 * 17
    drives = [
        row
        for row in recurrence
        if row["feature_name"] in {"a_over_LT", "a_over_Ln"}
    ]
    assert len(drives) == 10
    assert {float(row["top_k_recurrence"]) for row in drives} == {1.0}
    assert {row["bootstrap_unit"] for row in recurrence} == {"equilibrium_files"}


def test_s12_registry_effects_regimes_and_pysr_deferral_are_explicit() -> None:
    registry = _rows("feature_registry.csv")
    assert len(registry) == 17
    assert {row["version"] for row in registry} == {"S12-v1"}
    assert {row["validity_tag"] for row in registry} == {"observed-comparison"}
    effects = _rows("ebm_effects.csv")
    interactions = {
        row["term_name"] for row in effects if row["term_kind"] == "pairwise_interaction"
    }
    assert interactions == {
        "a_over_LT & log_f_Q",
        "a_over_LT & bad_curvature_compression",
        "a_over_LT & geodesic_curvature_abs_mean",
        "a_over_Ln & bad_curvature_compression",
        "log_f_Q & geodesic_curvature_abs_mean",
    }
    fidelity = _rows("fidelity.csv")
    assert {row["regime"] for row in fidelity} == {
        "all",
        "stable_or_near_floor",
        "unstable",
    }
    summary = json.loads((ARTIFACTS / "summary.json").read_text())
    assert summary["symbolic_regression"]["status"] == "deferred_toolchain_incompatible"
    assert summary["symbolic_regression"]["requested_pysr_version"] == "1.5.10"
    assert summary["symbolic_regression"]["installed_julia_version"] == "1.12.6"
