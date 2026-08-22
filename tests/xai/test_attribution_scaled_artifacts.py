from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from itg_nn.xai.artifacts import sha256_file
from itg_nn.xai.perturbations import ValidityTag


ARTIFACTS = Path("reports/xai/S06b_artifacts")


def _csv(name: str) -> list[dict[str, str]]:
    with (ARTIFACTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_s06b_manifest_hashes_every_published_scientific_artifact() -> None:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["step"] == "S06b"
    assert manifest["config"]["mode"] == "production"
    assert manifest["gradient_set"] == "fixed_and_varied_separate"
    assert len(manifest["config"]["headline_member_ids"]) == 10
    assert len(manifest["config"]["sensitivity_member_ids"]) == 10
    assert len(manifest["row_ids"]) == 2000
    assert manifest["config"]["initial_production_wall_time_seconds"] == 4645.802209625021
    assert manifest["config"]["estimator_backends"] == {
        "ig_low_pass": "integrated_gradients_captum",
        "periodic_mask": "periodic_extremal_mask",
    }
    assert manifest["config"]["script_sha256"] == sha256_file(
        "scripts/xai_s06b_attribution.py"
    )
    assert manifest["config"]["scaled_module_sha256"] == sha256_file(
        "itg_nn/xai/attribution_scaled.py"
    )
    assert manifest["config"]["attribution_module_sha256"] == sha256_file(
        "itg_nn/xai/attribution.py"
    )
    assert manifest["checkpoint"]["sha256"] == (
        "d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb"
    )
    assert manifest["dataset"]["sha256"] == (
        "9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad"
    )
    for name in (
        "channel_consensus.csv",
        "hierarchical_uncertainty.csv",
        "member_agreement.csv",
        "rank_sensitivity.csv",
        "scalar_sensitivities.csv",
        "selected_review_maps.h5",
        "symmetry_checks.csv",
        "stratified_consensus.csv",
        "summary.json",
        "consensus_atlas.png",
    ):
        assert sha256_file(ARTIFACTS / name) == manifest["output_hashes"][name]


def test_s06b_review_maps_preserve_member_sample_channel_position_axes() -> None:
    with h5py.File(ARTIFACTS / "selected_review_maps.h5", "r") as h5_file:
        assert h5_file.attrs["estimand"] == "native max(log Q, -2)"
        assert h5_file.attrs["stable_feature_claims_permitted"] == "false"
        assert h5_file["attribution"].shape == (2, 2, 10, 2, 16, 7, 96)
        assert h5_file["canonical_minus_original"].shape == (2, 10, 2, 16, 7, 96)
        assert json.loads(h5_file["attribution"].attrs["axes"]) == [
            "function",
            "method",
            "member",
            "gradient_set",
            "sample",
            "channel",
            "z",
        ]
        assert [value.decode() for value in h5_file["method_name"][:]] == [
            "ig_low_pass",
            "periodic_mask",
        ]
        assert [value.decode() for value in h5_file["estimator_backend"][:]] == [
            "integrated_gradients_captum",
            "periodic_extremal_mask",
        ]
        assert h5_file["signed"][:].tolist() == [True, False]


def test_s06b_consensus_distinguishes_signed_absolute_and_agreement() -> None:
    rows = _csv("channel_consensus.csv")
    assert {row["function"] for row in rows} == {
        "original_f",
        "invariant_tilde_f",
    }
    assert {row["method"] for row in rows} == {"ig_low_pass", "periodic_mask"}
    assert {row["gradient_set"] for row in rows} == {"fixed", "varied"}
    assert {row["stratum"] for row in rows} == {
        "all",
        "stable_or_near_floor",
        "unstable",
    }
    assert len(rows) == 2 * 2 * 2 * 3 * 7
    assert {row["signed"] for row in rows} == {"True", "False"}
    assert all(row["median_signed"] != "" for row in rows)
    assert all(row["median_absolute"] != "" for row in rows)
    assert all(0.0 <= float(row["sign_agreement"]) <= 1.0 for row in rows)
    assert all(row["estimand"] == "native max(log Q, -2)" for row in rows)
    assert all(row["validity_tag"] in {tag.value for tag in ValidityTag} for row in rows)
    assert {row["estimator_backend"] for row in rows} == {
        "integrated_gradients_captum",
        "periodic_extremal_mask",
    }
    stable = [row for row in rows if row["stratum"] == "stable_or_near_floor"]
    assert all(row["feature_claims_permitted"] == "False" for row in stable)


def test_s06b_uncertainty_resamples_members_and_equilibrium_files() -> None:
    rows = _csv("hierarchical_uncertainty.csv")
    assert len(rows) == 2 * 2 * 2 * 3 * 7
    assert all(row["member_resampling_unit"] == "members" for row in rows)
    assert all(row["sample_resampling_unit"] == "equilibrium_files" for row in rows)
    assert all(int(row["replicates"]) == 500 for row in rows)
    assert all(
        float(row["ci_lower"]) <= float(row["estimate"]) <= float(row["ci_upper"])
        for row in rows
    )


def test_s06b_strata_rank_sensitivity_and_scalar_drives_are_complete() -> None:
    strata = _csv("stratified_consensus.csv")
    required = {
        "flux",
        "a_over_lt",
        "a_over_ln",
        "equilibrium_class",
        "member_absolute_error",
        "ensemble_spread",
    }
    assert required.issubset({row["stratifier"] for row in strata})
    assert all(row["gradient_set"] in {"fixed", "varied"} for row in strata)
    assert all(int(row["sample_count"]) > 0 for row in strata)
    assert all(int(row["sample_count_stable"]) == 0 for row in strata)
    assert all(row["feature_claims_permitted"] == "True" for row in strata)
    assert all(row["estimand"] == "native max(log Q, -2)" for row in strata)
    assert all(row["validity_tag"] == ValidityTag.OFF_MANIFOLD.value for row in strata)
    assert all(row["signed"] == "True" for row in strata)
    assert all(
        row["baseline_convention"]
        == "input-derived low-pass; deliberately off-manifold diagnostic"
        for row in strata
    )
    assert all(row["estimator_backend"] == "integrated_gradients_captum" for row in strata)

    agreement = _csv("member_agreement.csv")
    assert all(float(row["independent_sign_null"]) == 0.623046875 for row in agreement)
    assert {row["estimator_backend"] for row in agreement} == {
        "integrated_gradients_captum",
        "periodic_extremal_mask",
    }

    rank = _csv("rank_sensitivity.csv")
    assert len(rank) == 20
    assert {row["member_cohort"] for row in rank} == {
        "stored_validation_top_10",
        "stored_validation_ranks_11_50",
        "stored_validation_ranks_51_100",
    }
    assert all(row["stored_validation_r2"] for row in rank)
    assert all(row["canonical_map_rank_agreement"] for row in rank)

    scalar = _csv("scalar_sensitivities.csv")
    assert len(scalar) == 2 * 10 * 2 * 2
    assert {row["drive"] for row in scalar} == {"a_over_lt", "a_over_ln"}
    assert all(row["signed"] == "True" for row in scalar)
    assert all(row["scale"] == "robust_per_scalar_drive" for row in scalar)
    assert all(row["estimand"] == "native max(log Q, -2)" for row in scalar)

    symmetry = _csv("symmetry_checks.csv")
    assert len(symmetry) == 2 * 2 * 10
    canonical = [row for row in symmetry if row["function"] == "invariant_tilde_f"]
    assert all(row["baseline_convention"] for row in symmetry)
    assert all(float(row["prediction_invariance_relative_rms"]) >= 0 for row in symmetry)
    low_pass = [row for row in canonical if row["method"] == "ig_low_pass"]
    assert all(row["estimator_backend"] for row in symmetry)
    assert np.median(
        [float(row["co_shifted_equivariance_relative_rms"]) for row in low_pass]
    ) < 2e-6
    assert max(
        float(row["co_shifted_equivariance_relative_rms"]) for row in low_pass
    ) < 1e-4
    original_low_pass = [
        row
        for row in symmetry
        if row["function"] == "original_f" and row["method"] == "ig_low_pass"
    ]
    assert np.median(
        [float(row["co_shifted_equivariance_relative_rms"]) for row in original_low_pass]
    ) > 0.8
    canonical_mask = [row for row in canonical if row["method"] == "periodic_mask"]
    assert np.median(
        [float(row["fixed_baseline_equivariance_relative_rms"]) for row in canonical_mask]
    ) > 0.8
