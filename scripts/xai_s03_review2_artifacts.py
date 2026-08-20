#!/usr/bin/env python3
"""Regenerate small S03 review artifacts from a manifest-verified ladder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:  # Direct script execution puts scripts/ itself on sys.path.
    from xai_s03_ladder import (
        _analysis_masks,
        _channel_robust_scales,
        _compact_ladder_summary,
        _contrast_summary_rows,
        _csv_text,
        _family_summary,
        _load_panel,
        _load_support_reference,
        _robust_input_displacements,
        _specs,
        _support_rows,
        _transform,
    )
except ModuleNotFoundError:  # Package/module import from the repository root.
    from scripts.xai_s03_ladder import (
        _analysis_masks,
        _channel_robust_scales,
        _compact_ladder_summary,
        _contrast_summary_rows,
        _csv_text,
        _family_summary,
        _load_panel,
        _load_support_reference,
        _robust_input_displacements,
        _specs,
        _support_rows,
        _transform,
    )

from itg_nn.xai.perturbations import ReferenceBackgrounds, ScaledPCASupport


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    for row in rows:
        if "replicate" in row:
            row["replicate"] = int(row["replicate"])
        if "dose" in row:
            row["dose"] = float(row["dose"])
        if "path_dose" in row:
            row["path_dose"] = float(row["path_dose"])
    return rows


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _member_median(
    rows: list[dict[str, Any]],
    top_ids: set[str],
    *,
    family: str,
    function: str = "invariant_tilde_f",
    replicate: int = 0,
    dose: float | None = None,
) -> float:
    selected = [
        float(row["rms_change_over_residual_std"])
        for row in rows
        if row["member_id"] in top_ids
        and row["family"] == family
        and row["function"] == function
        and row["gradient_set"] == "varied"
        and row["stratum"] == "all"
        and int(row["replicate"]) == replicate
        and (dose is None or float(row["dose"]) == dose)
    ]
    if not selected:
        raise ValueError(f"no rows for {family=}, {function=}, {replicate=}, {dose=}")
    return float(np.median(selected))


def _verify_output(manifest: dict[str, Any], run: Path, name: str) -> Path:
    path = run / name
    expected = manifest["output_hashes"][name]
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{path} hash {actual} does not match manifest {expected}")
    return path


def _dose_enrichment(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[
    Any,
    dict[str, np.ndarray],
    ReferenceBackgrounds,
    ReferenceBackgrounds,
    dict[str, dict[str, str]],
]:
    dataset = Path(config["dataset"]).resolve()
    cohorts = json.loads(Path(config["cohorts"]).read_text(encoding="utf-8"))
    panel, metadata = _load_panel(
        dataset,
        cohorts,
        int(config["panel_varied_rows"]),
        float(config["stable_threshold_log_Q"]),
    )
    support_varied, support_fixed, support_metadata = _load_support_reference(
        dataset,
        cohorts,
        panel.row_indices[: int(config["panel_varied_rows"])],
        int(config["support_reference_rows"]),
        int(config["seed"]) + 41,
    )
    gradients_varied = np.column_stack(
        (support_varied.a_over_lt.numpy(), support_varied.a_over_ln.numpy())
    )
    gradients_fixed = np.column_stack(
        (support_fixed.a_over_lt.numpy(), support_fixed.a_over_ln.numpy())
    )
    varied_backgrounds = ReferenceBackgrounds(
        support_varied.geometry,
        gradients_varied,
        support_metadata["equilibrium_class"],
        support_varied.row_indices,
    )
    fixed_backgrounds = ReferenceBackgrounds(
        support_fixed.geometry,
        gradients_fixed,
        support_metadata["equilibrium_class"],
        support_fixed.row_indices,
    )
    masks = _analysis_masks(panel, metadata, float(config["stable_threshold_log_Q"]))
    channel_scales = _channel_robust_scales(Path(config["s01_channel_scales"]))
    doses: dict[tuple[str, str, str], dict[str, float]] = {}
    endpoint_hashes: dict[str, dict[str, str]] = {}
    varied_count = int(config["panel_varied_rows"])
    for spec in _specs(config):
        endpoint = _transform(
            spec,
            panel.geometry,
            panel,
            metadata,
            varied_backgrounds,
            fixed_backgrounds,
            config,
        )
        doses.update(
            _robust_input_displacements(
                panel.geometry, endpoint, spec, masks, channel_scales
            )
        )
        endpoint_hashes[spec.name] = {
            "varied_sha256": hashlib.sha256(
                np.ascontiguousarray(endpoint[:varied_count].numpy()).tobytes()
            ).hexdigest(),
            "full_paired_sha256": hashlib.sha256(
                np.ascontiguousarray(endpoint.numpy()).tobytes()
            ).hexdigest(),
        }
    for row in rows:
        values = doses[(row["perturbation"], row["gradient_set"], row["stratum"])]
        effect = row["rms_change_over_residual_std"]
        row["robust_input_displacement_rms"] = values["rms"]
        row["robust_input_displacement_median_abs"] = values["median_abs"]
        row["effect_per_robust_input_rms"] = (
            float(effect) / values["rms"] if effect != "" and values["rms"] > 0 else ""
        )
        row["effect_per_robust_input_median_abs"] = (
            float(effect) / values["median_abs"]
            if effect != "" and values["median_abs"] > 0
            else ""
        )
    return panel, metadata, varied_backgrounds, fixed_backgrounds, endpoint_hashes


def _load_support_model(path: Path) -> ScaledPCASupport:
    with np.load(path) as values:
        return ScaledPCASupport(
            channel_center=values["channel_center"],
            channel_scale=values["channel_scale"],
            feature_center=values["feature_center"],
            components=values["components"],
            fit_scores=values["fit_scores"],
            calibration_reconstruction=values["calibration_reconstruction"],
            calibration_nearest=values["calibration_nearest"],
        )


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _add_paired_equilibrium_bootstrap(
    contrast_rows: list[dict[str, Any]],
    ladder: list[dict[str, Any]],
    top_ids: set[str],
    panel: Any,
    source_predictions: Path,
    phase_predictions: Path,
    *,
    replicates: int,
    seed: int,
) -> None:
    """Add paired equilibrium bootstrap intervals to inferential contrasts."""

    varied_count = len(panel.geometry) // 2
    actual = panel.actual_log_heat_flux[:varied_count].numpy().astype(np.float64)
    draws = np.random.default_rng(seed).integers(
        0, varied_count, size=(replicates, varied_count)
    )

    def spec_name(family: str, replicate: int, parameter: str) -> str:
        matches = {
            str(row["perturbation"])
            for row in ladder
            if row["family"] == family
            and int(row["replicate"]) == replicate
            and (not parameter or row["parameter"] == parameter)
        }
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one perturbation for {family=}, {replicate=}, "
                f"{parameter=}; got {sorted(matches)}"
            )
        return matches.pop()

    for row in contrast_rows:
        if (
            row["contrast_type"]
            == "exact_symmetry_diagnostic_not_effect_size_control"
        ):
            row["equilibrium_bootstrap_ci95_lower"] = ""
            row["equilibrium_bootstrap_ci95_upper"] = ""
            row["equilibrium_bootstrap_unit"] = (
                "not_applicable_exact_symmetry_diagnostic"
            )
            continue
        phase = row["treatment_family"] in (
            "common_phase_scramble",
            "channel_phase_scramble",
        )
        path = phase_predictions if phase else source_predictions
        replicate = int(row["replicate"])
        treatment_name = spec_name(
            str(row["treatment_family"]),
            replicate,
            str(row["treatment_parameter"]),
        )
        control_name = spec_name(str(row["control_family"]), replicate, "")
        with h5py.File(path, "r") as handle:
            members = _decode(handle["member_id"][:])
            functions = _decode(handle["function"][:])
            perturbations = _decode(handle["perturbation"][:])
            function_index = functions.index("invariant_tilde_f")
            reference_index = perturbations.index("reference")
            treatment_index = perturbations.index(treatment_name)
            control_index = perturbations.index(control_name)
            member_bootstrap: list[np.ndarray] = []
            for member_id in sorted(top_ids):
                member_index = members.index(member_id)
                reference = handle["prediction"][
                    member_index, function_index, reference_index, :varied_count
                ].astype(np.float64)
                treatment = handle["prediction"][
                    member_index, function_index, treatment_index, :varied_count
                ].astype(np.float64)
                control = handle["prediction"][
                    member_index, function_index, control_index, :varied_count
                ].astype(np.float64)
                denominator = np.std(
                    (reference - actual)[draws], axis=1, ddof=1
                )
                treatment_rms = np.sqrt(
                    np.mean(np.square((treatment - reference)[draws]), axis=1)
                )
                control_rms = np.sqrt(
                    np.mean(np.square((control - reference)[draws]), axis=1)
                )
                member_bootstrap.append(
                    (treatment_rms - control_rms) / denominator
                )
        median_bootstrap = np.median(np.stack(member_bootstrap), axis=0)
        lower, upper = np.quantile(median_bootstrap, (0.025, 0.975))
        row["equilibrium_bootstrap_ci95_lower"] = float(lower)
        row["equilibrium_bootstrap_ci95_upper"] = float(upper)
        row["equilibrium_bootstrap_unit"] = (
            "paired_equilibrium_files;median_across_registered_top10_members"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--phase-run", type=Path, required=True)
    parser.add_argument(
        "--published-dir", type=Path, default=Path("reports/xai/S03_artifacts")
    )
    args = parser.parse_args()

    manifest_path = args.source_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ladder_path = _verify_output(manifest, args.source_run, "ladder.csv")
    support_path = _verify_output(manifest, args.source_run, "support.csv")
    source_baseline_path = _verify_output(
        manifest, args.source_run, "baseline_registry.json"
    )
    source_summary_path = _verify_output(manifest, args.source_run, "summary.json")
    source_predictions_path = _verify_output(
        manifest, args.source_run, "predictions.h5"
    )
    source_support_model_path = _verify_output(
        manifest, args.source_run, "support_model.npz"
    )
    source_endpoint_path = _verify_output(
        manifest, args.source_run, "operator_endpoint_hashes.json"
    )

    phase_manifest_path = args.phase_run / "manifest.json"
    phase_manifest = json.loads(phase_manifest_path.read_text(encoding="utf-8"))
    phase_ladder_path = _verify_output(
        phase_manifest, args.phase_run, "ladder.csv"
    )
    phase_summary_path = _verify_output(
        phase_manifest, args.phase_run, "summary.json"
    )
    phase_predictions_path = _verify_output(
        phase_manifest, args.phase_run, "predictions.h5"
    )
    phase_endpoint_path = _verify_output(
        phase_manifest, args.phase_run, "operator_endpoint_hashes.json"
    )

    ladder = _read_csv(ladder_path)
    phase_ladder = _read_csv(phase_ladder_path)
    ladder = [
        row
        for row in ladder
        if row["family"] not in ("common_phase_scramble", "channel_phase_scramble")
    ] + phase_ladder
    (
        panel,
        metadata,
        varied_backgrounds,
        fixed_backgrounds,
        endpoint_hashes,
    ) = _dose_enrichment(manifest["config"], ladder)
    support_model = _load_support_model(source_support_model_path)
    support = _support_rows(
        support_model,
        _specs(manifest["config"]),
        panel,
        metadata,
        varied_backgrounds,
        fixed_backgrounds,
        manifest["config"],
    )
    top_ids = {
        row["member_id"]
        for row in ladder
        if row["validation_cohort"] == "stored_validation_top10"
    }
    contrast_rows = _contrast_summary_rows(ladder, top_ids)
    _add_paired_equilibrium_bootstrap(
        contrast_rows,
        ladder,
        top_ids,
        panel,
        source_predictions_path,
        phase_predictions_path,
        replicates=int(manifest["config"]["bootstrap_replicates"]),
        seed=int(manifest["config"]["seed"]) + 307,
    )
    compact_rows = _compact_ladder_summary(ladder, top_ids)

    support_reference = next(
        row
        for row in support
        if float(row["path_dose"]) == 0 and int(row["replicate"]) == 0
    )
    amplitude = []
    for dose in (0.25, 0.5, 0.75, 1.0):
        selected = [
            row
            for row in ladder
            if row["member_id"] in top_ids
            and row["family"] == "amplitude_scaling"
            and row["function"] == "invariant_tilde_f"
            and row["gradient_set"] == "varied"
            and row["stratum"] == "all"
            and int(row["replicate"]) == 0
            and float(row["dose"]) == dose
        ]
        effects = [float(row["rms_change_over_residual_std"]) for row in selected]
        inputs = [float(row["robust_input_displacement_rms"]) for row in selected]
        efficiencies = [float(row["effect_per_robust_input_rms"]) for row in selected]
        median_inputs = [
            float(row["robust_input_displacement_median_abs"]) for row in selected
        ]
        median_efficiencies = [
            float(row["effect_per_robust_input_median_abs"]) for row in selected
        ]
        amplitude.append(
            {
                "dose": dose,
                "member_median_effect": float(np.median(effects)),
                "member_median_robust_input_rms": float(np.median(inputs)),
                "member_median_effect_per_robust_input_rms": float(
                    np.median(efficiencies)
                ),
                "member_median_robust_input_median_abs": float(
                    np.median(median_inputs)
                ),
                "member_median_effect_per_robust_input_median_abs": float(
                    np.median(median_efficiencies)
                ),
            }
        )

    summary: dict[str, Any] = {
        "source": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "ladder_sha256": _sha256(ladder_path),
            "support_sha256": _sha256(support_path),
            "source_git_commit": manifest["git_commit"],
            "matched_phase_manifest": str(phase_manifest_path.resolve()),
            "matched_phase_manifest_sha256": _sha256(phase_manifest_path),
            "matched_phase_ladder_sha256": _sha256(phase_ladder_path),
        },
        "scope": "varied-gradient rows only; fixed-gradient interpretations withdrawn",
        "support_warning_structural_null": {
            "median": math.sqrt(0.5),
            "fraction_above_0.95": 1 - 0.95**2,
            "observed_unperturbed_median": float(
                support_reference["warning_score_median"]
            ),
            "observed_unperturbed_fraction_above_0.95": float(
                support_reference["fraction_outside_heldout_central_95pct"]
            ),
        },
        "original_f_random_joint_shift_effect": _member_median(
            ladder, top_ids, family="random_joint_shift", function="original_f"
        ),
        "phase_member_medians_by_replicate": [
            {
                "replicate": replicate,
                "common_phase": _member_median(
                    ladder,
                    top_ids,
                    family="common_phase_scramble",
                    replicate=replicate,
                ),
                "channel_phase": _member_median(
                    ladder,
                    top_ids,
                    family="channel_phase_scramble",
                    replicate=replicate,
                ),
            }
            for replicate in range(3)
        ],
        "amplitude_scaling": amplitude,
        "contrasts": contrast_rows,
    }

    ladder_summary_path = args.published_dir / "ladder_summary.csv"
    contrasts_path = args.published_dir / "contrasts.csv"
    review_path = args.published_dir / "review2_summary.json"
    baseline_path = args.published_dir / "baseline_registry.json"
    summary_path = args.published_dir / "summary.json"
    support_published_path = args.published_dir / "support.csv"
    endpoint_published_path = args.published_dir / "operator_endpoint_hashes.json"
    derived_manifest_path = args.published_dir / "review3_manifest.json"

    baseline_registry = json.loads(source_baseline_path.read_text(encoding="utf-8"))
    baseline_registry["available"] = {
        "robust_constant": "per-channel reference median expanded over z",
        "matched_observed": "observed robust-gradient nearest neighbour within equilibrium class; source row excluded",
        "medoid": "observed geometry nearest the robust profile center, optionally within class",
        "low_pass": "rFFT truncation of the supplied analysed or background input, retaining DC through the registered cutoff",
        "conditional_channel_profile": "pointwise median profile of nearest class/gradient-matched observed rows; tied gradients reduce explicitly to class-only matching",
    }
    baseline_registry["production_real_data_usage"] = {
        "conditional_channel_profile": "used for S03 channel replacement",
        "robust_constant_matched_observed_medoid_low_pass": "API-only in S03; first real consumers are downstream steps",
    }
    baseline_registry["support_warning"] = (
        "robust per-channel scaling + ordinary SVD PCA + held-out nearest-neighbour "
        "distance; publish one-sided component percentiles as well as the legacy "
        "two-sided maximum; not proof of physical validity"
    )
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    phase_summary = json.loads(phase_summary_path.read_text(encoding="utf-8"))
    source_summary["support"] = baseline_registry
    source_summary["normalization"]["input_dose"] = (
        "S01 IQR/1.349 channel scales followed by both RMS and median-absolute "
        "aggregation; replacement dose uses only the edited channel; normalized "
        "rankings are sensitivity analyses"
    )
    source_summary["derived_review3"] = {
        "fixed_gradient_interpretations": "withdrawn",
        "matched_phase_run": str(phase_manifest_path.resolve()),
        "published_artifact_manifest": str(derived_manifest_path.resolve()),
    }
    source_summary["top10_canonical_varied_all_family_summary"] = _family_summary(
        ladder, top_ids
    )
    source_summary["checks"]["toy_controls"] = phase_summary["checks"][
        "toy_controls"
    ]
    source_repeats = source_summary["checks"]["full_batch_repeat_by_spec"]
    source_repeats = {
        name: passed
        for name, passed in source_repeats.items()
        if not name.startswith(("common_phase_scramble", "channel_phase_scramble"))
    }
    source_repeats.update(phase_summary["checks"]["full_batch_repeat_by_spec"])
    source_summary["checks"]["full_batch_repeat_by_spec"] = source_repeats
    source_summary["checks"]["deterministic_seeded_operators"] = bool(
        all(source_repeats.values())
    )
    source_summary["checks"]["support_at_every_spec_and_path_dose"] = (
        len(support)
        == len(_specs(manifest["config"]))
        * len(manifest["config"]["support_path_doses"])
    )
    source_summary["checks"]["registered_control_contrast_rows"] = len(
        contrast_rows
    )

    source_endpoint_hashes = json.loads(
        source_endpoint_path.read_text(encoding="utf-8")
    )
    phase_endpoint_hashes = json.loads(
        phase_endpoint_path.read_text(encoding="utf-8")
    )
    phase_names = set(phase_endpoint_hashes)
    endpoint_document = {
        "schema": "separate retained-varied and full-paired endpoint hashes",
        "retained_scope": "first panel_varied_rows entries; authoritative for S03 conclusions",
        "full_paired_scope": "varied plus withdrawn fixed twins; deterministic audit only",
        "operators": {
            name: {
                **values,
                "immutable_run_full_sha256": (
                    phase_endpoint_hashes[name]
                    if name in phase_names
                    else source_endpoint_hashes[name]
                ),
                "immutable_run": "corrected_phase" if name in phase_names else "source_full",
            }
            for name, values in endpoint_hashes.items()
        },
    }
    _atomic_text(ladder_summary_path, _csv_text(compact_rows))
    _atomic_text(contrasts_path, _csv_text(contrast_rows))
    _atomic_text(support_published_path, _csv_text(support))
    _atomic_text(
        endpoint_published_path,
        json.dumps(endpoint_document, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        baseline_path,
        json.dumps(baseline_registry, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        summary_path,
        json.dumps(source_summary, indent=2, sort_keys=True) + "\n",
    )
    summary["generated_artifacts"] = {
        "ladder_summary.csv": _sha256(ladder_summary_path),
        "contrasts.csv": _sha256(contrasts_path),
        "generator_sha256": _sha256(Path(__file__)),
    }
    _atomic_text(review_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    repository = Path(__file__).resolve().parents[1]
    dose_inputs = {
        "dataset": Path(manifest["config"]["dataset"]).resolve(),
        "cohorts": Path(manifest["config"]["cohorts"]).resolve(),
        "channel_scales": Path(
            manifest["config"]["s01_channel_scales"]
        ).resolve(),
        "ladder_code": repository / "scripts/xai_s03_ladder.py",
        "perturbation_code": repository / "itg_nn/xai/perturbations.py",
    }
    if _sha256(dose_inputs["dataset"]) != manifest["dataset"]["sha256"]:
        raise RuntimeError("current external dataset does not match source manifest")
    derived_manifest = {
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__)),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "phase_manifest": str(phase_manifest_path.resolve()),
        "phase_manifest_sha256": _sha256(phase_manifest_path),
        "dose_enrichment_input_hashes": {
            name: _sha256(path) for name, path in dose_inputs.items()
        },
        "verified_source_outputs": {
            "source_ladder.csv": _sha256(ladder_path),
            "source_support.csv": _sha256(support_path),
            "source_baseline_registry.json": _sha256(source_baseline_path),
            "source_summary.json": _sha256(source_summary_path),
            "matched_phase_ladder.csv": _sha256(phase_ladder_path),
            "source_predictions.h5": _sha256(source_predictions_path),
            "matched_phase_predictions.h5": _sha256(phase_predictions_path),
            "source_support_model.npz": _sha256(source_support_model_path),
            "source_operator_endpoint_hashes.json": _sha256(source_endpoint_path),
            "matched_phase_summary.json": _sha256(phase_summary_path),
            "matched_phase_operator_endpoint_hashes.json": _sha256(
                phase_endpoint_path
            ),
        },
        "published_output_hashes": {
            path.name: _sha256(path)
            for path in sorted(args.published_dir.iterdir())
            if path.is_file() and path != derived_manifest_path
        },
    }
    _atomic_text(
        derived_manifest_path,
        json.dumps(derived_manifest, indent=2, sort_keys=True) + "\n",
    )
    print(derived_manifest_path)


if __name__ == "__main__":
    main()
