#!/usr/bin/env python3
"""Run S09 concept completeness and geometry-drive interaction analysis."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from itg_nn.data import load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.completeness import grouped_completeness, grouped_integrated_hessian, stratified_directional_effects
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import CANONICAL_FUNCTION, InvariantMember
from xai_s08_concepts import _concept_scores


NATIVE_ESTIMAND = "native max(log Q, -2)"
OBSERVED = "observed-comparison"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/xai/S09_completeness.json"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--published-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--members", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    return parser


def _resolve(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if args.pilot:
        resolved.update(config["pilot"])
    resolved["mode"] = "pilot" if args.pilot else "production"
    for name in ("device", "seed", "members", "batch_size"):
        if getattr(args, name) is not None:
            resolved[name] = getattr(args, name)
    if args.rows is not None:
        resolved["panel_varied_rows"] = args.rows
    for value, name in ((args.dataset, "dataset"), (args.checkpoint, "checkpoint"), (args.published_dir, "published_dir")):
        if value is not None:
            resolved[name] = str(value)
    return resolved


def _csv_text(rows: list[dict[str, Any]]) -> str:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def _r2(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    y = target[mask]; p = prediction[mask]
    denominator = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if denominator <= 0 else 1.0 - float(np.sum((y - p) ** 2)) / denominator


def _error_metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    residual = prediction[mask] - target[mask]
    return {"mse": float(np.mean(residual ** 2)), "bias_decoder_minus_member": float(np.mean(residual)), "residual_std": float(np.std(residual))}


def _paired_gain(target: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, groups: np.ndarray, replicates: int, seed: int) -> tuple[float, float, float]:
    unique = np.unique(groups); positions_by_group = {group: np.flatnonzero(groups == group) for group in unique}; rng = np.random.default_rng(seed); draws = np.empty(replicates)
    for draw in range(replicates):
        chosen = rng.choice(unique, len(unique), replace=True)
        positions = np.concatenate([positions_by_group[group] for group in chosen])
        mask = np.ones(len(positions), dtype=bool)
        draws[draw] = _r2(target[positions], candidate[positions], mask) - _r2(target[positions], baseline[positions], mask)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(lower), float(upper), float(np.mean(draws > 0))


def _member_values(model: InvariantMember, panel, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    bottleneck, prediction = [], []
    with torch.inference_mode():
        for start in range(0, len(panel.geometry), batch_size):
            stop = start + batch_size
            geometry = panel.geometry[start:stop].to(device)
            a_lt = panel.a_over_lt[start:stop].to(device)
            a_ln = panel.a_over_ln[start:stop].to(device)
            bottleneck.append(model.invariant_bottleneck(geometry).cpu().numpy())
            prediction.append(model(geometry, a_lt, a_ln).cpu().numpy())
    return np.concatenate(bottleneck).astype(np.float64), np.concatenate(prediction).astype(np.float64)


def _plot(path: Path, rows: list[dict[str, Any]], member_ids: list[str]) -> None:
    names = [row["concept_set"] for row in rows if row["member_id"] == member_ids[0] and row["regime"] == "all"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for member in member_ids:
        selected = [row for row in rows if row["member_id"] == member and row["regime"] == "all"]
        axes[0].plot(range(len(selected)), [float(row["held_out_r2"]) for row in selected], marker="o", alpha=.75, label=member)
        axes[1].plot(range(len(selected)), [float(row["gain_over_paper_baseline"]) for row in selected], marker="o", alpha=.75)
    for axis, title in zip(axes, ("Held-out decoder fidelity", "Gain over paper baseline")):
        axis.set_xticks(range(len(names)), names, rotation=30, ha="right"); axis.set_title(title); axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _resume_postprocess(output_dir: Path, resolved: dict[str, Any], dataset: Path, checkpoint: Path, repository: Path, published: Path | None) -> Path:
    artifacts = RunArtifacts(output_dir)
    old = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if old["dataset"]["sha256"] != sha256_file(dataset) or old["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise RuntimeError("resume input fingerprint changed")
    completeness = list(csv.DictReader((output_dir / "completeness.csv").open(newline="", encoding="utf-8")))
    residuals = list(csv.DictReader((output_dir / "concept_residuals.csv").open(newline="", encoding="utf-8")))
    members = list(old["member_ids"])
    for row in completeness:
        if row["concept_set"] == "full_bottleneck": row["concept_set"] = "full_bottleneck_simple_decoder"
    for row in residuals:
        if row["concept_set"] == "full_bottleneck": row["concept_set"] = "full_bottleneck_simple_decoder"
    for member_number, member in enumerate(members):
        member_residuals = [row for row in residuals if row["member_id"] == member]
        ordered_names = list(dict.fromkeys(row["concept_set"] for row in member_residuals))
        by_set = {name: [row for row in member_residuals if row["concept_set"] == name] for name in ordered_names}
        baseline_rows = by_set["paper_baseline"]
        target = np.asarray([float(row["target_native_prediction"]) for row in baseline_rows])
        baseline = np.asarray([float(row["decoder_prediction"]) for row in baseline_rows])
        groups = np.asarray([row["equilibrium_file"] for row in baseline_rows])
        stable_local = np.asarray([row["stable_or_near_floor"] == "True" for row in baseline_rows])
        baseline_r2 = _r2(target, baseline, np.ones(len(target), dtype=bool))
        for set_number, (name, set_rows) in enumerate(by_set.items()):
            prediction = np.asarray([float(row["decoder_prediction"]) for row in set_rows])
            lower, upper, stability = _paired_gain(target, prediction, baseline, groups, int(resolved["direct_gain_bootstrap_replicates"]), int(resolved["seed"]) + member_number * 10000 + set_number)
            for row in completeness:
                if row["member_id"] == member and row["concept_set"] == name:
                    regime_mask = {"all": np.ones(len(target), dtype=bool), "stable_or_near_floor": stable_local, "unstable": ~stable_local}[row["regime"]]
                    row.update(_error_metrics(target, prediction, regime_mask))
                    if row["regime"] == "all":
                        row["completeness_relative_to_full_bottleneck"] = row["held_out_r2"]
                        row["gain_over_paper_baseline_ci95_lower"] = lower; row["gain_over_paper_baseline_ci95_upper"] = upper; row["gain_over_paper_baseline_selection_stability"] = stability
        if "full_bottleneck_exact_head" not in by_set:
            stable = np.asarray([row["stable_or_near_floor"] == "True" for row in baseline_rows])
            lower, upper, stability = _paired_gain(target, target, baseline, groups, int(resolved["direct_gain_bootstrap_replicates"]), int(resolved["seed"]) + member_number * 10000 + 999)
            for regime, mask in (("all", np.ones(len(target), dtype=bool)), ("stable_or_near_floor", stable), ("unstable", ~stable)):
                completeness.append({"member_id": member, "concept_set": "full_bottleneck_exact_head", "regime": regime, "held_out_r2": 1.0, "increment_r2_over_previous": "", "increment_ci95_lower": "", "increment_ci95_upper": "", "bootstrap_selection_stability": "", "gain_over_paper_baseline": 1.0 - baseline_r2 if regime == "all" else "", "gain_over_paper_baseline_ci95_lower": lower if regime == "all" else "", "gain_over_paper_baseline_ci95_upper": upper if regime == "all" else "", "gain_over_paper_baseline_selection_stability": stability if regime == "all" else "", "completeness_relative_to_full_bottleneck": 1.0 if regime == "all" else "", "rows": int(mask.sum()), "outer_split_unit": "not_fitted_exact_trained_head", "inner_split_unit": "not_fitted_exact_trained_head", "bootstrap_unit": "equilibrium_files", "decoder": "trained_canonical_head_exact_ceiling", "estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION, "validity_tag": OBSERVED})
            for row in baseline_rows:
                residuals.append({**row, "concept_set": "full_bottleneck_exact_head", "decoder_prediction": row["target_native_prediction"], "signed_residual": 0.0})
    panel = load_hdf5_rows(dataset, np.asarray(old["row_ids"], dtype=np.int64), gradient_set="varied", include_targets=True)
    scores, metadata = _concept_scores(dataset, np.asarray(old["row_ids"], dtype=np.int64), panel.geometry.numpy())
    stable_mask = panel.actual_log_heat_flux.numpy() <= float(resolved["stable_threshold_log_Q"])
    groups_all = metadata["equilibrium_file"]
    selected_names = ("log_f_Q", "f_stab", "log_compression", "bad_curvature", "geodesic_curvature", "parallel_scale", "cross_channel_colocation", "log_FSA_grad_x")
    interactions: list[dict[str, Any]] = []; hessian_rows: list[dict[str, Any]] = []
    for member_number, member in enumerate(members):
        baseline_rows = [row for row in residuals if row["member_id"] == member and row["concept_set"] == "paper_baseline"]
        target_all = np.asarray([float(row["target_native_prediction"]) for row in baseline_rows])
        selected_all = {name: scores[name] for name in selected_names}
        for regime_number, (regime, mask) in enumerate((("all", np.ones(len(target_all), dtype=bool)), ("stable_or_near_floor", stable_mask), ("unstable", ~stable_mask))):
            selected_regime = {name: values[mask] for name, values in selected_all.items()}; groups = groups_all[mask]; target = target_all[mask]
            for drive_name, drive in (("a_over_LT", metadata["a_over_lt"]), ("a_over_Ln", metadata["a_over_ln"])):
                effects = stratified_directional_effects(selected_regime, drive[mask], target, groups, bins=int(resolved["interaction_bins"]), bootstrap_replicates=int(resolved["bootstrap_replicates"]), seed=int(resolved["seed"]) + member_number * 10000 + regime_number * 2000 + (0 if drive_name == "a_over_LT" else 1000))
                for effect in effects: interactions.append({"member_id": member, "regime": regime, "drive": drive_name, **effect.__dict__, "estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION, "finite_difference_kind": "grouped_observed_directional_slope", "integrated_hessian_proxy": "quadratic_decoder_concept_by_drive_term", "perturbation_validity_tag": OBSERVED})
            values = np.column_stack((metadata["a_over_lt"][mask], metadata["a_over_ln"][mask], *selected_regime.values()))
            terms = grouped_integrated_hessian(values, target, groups, concept_names=selected_names, outer_folds=int(resolved["outer_folds"]), inner_folds=int(resolved["inner_folds"]), penalties=tuple(resolved["ridge_penalties"]), bootstrap_replicates=int(resolved["bootstrap_replicates"]), seed=int(resolved["seed"]) + member_number * 10000 + regime_number * 2000 + 1500)
            for term in terms: hessian_rows.append({"member_id": member, "regime": regime, "drive": ("a_over_LT" if term.drive_index == 0 else "a_over_Ln"), **term.__dict__, "estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION, "method": "held_out_quadratic_decoder_four_point_mixed_difference"})
    interaction_summary = []
    keys = sorted({(row["regime"], row["drive"], row["concept"], row["bin_index"]) for row in interactions})
    for regime, drive, concept, bin_index in keys:
        selected = [row for row in interactions if (row["regime"], row["drive"], row["concept"], row["bin_index"]) == (regime, drive, concept, bin_index)]
        slopes = np.asarray([float(row["slope"]) for row in selected])
        interaction_summary.append({"regime": regime, "drive": drive, "concept": concept, "bin_index": bin_index, "member_count": len(slopes), "median_slope": float(np.median(slopes)), "minimum_slope": float(slopes.min()), "maximum_slope": float(slopes.max()), "member_sign_agreement": float(max(np.mean(slopes > 0), np.mean(slopes < 0))), "source": "varied-gradient panel only", "validity_tag": OBSERVED})
    (output_dir / "completeness.csv").write_text(_csv_text(completeness), encoding="utf-8")
    (output_dir / "concept_residuals.csv").write_text(_csv_text(residuals), encoding="utf-8")
    (output_dir / "interaction_effects.csv").write_text(_csv_text(interactions), encoding="utf-8")
    (output_dir / "integrated_hessian_terms.csv").write_text(_csv_text(hessian_rows), encoding="utf-8")
    (output_dir / "interaction_summary.csv").write_text(_csv_text(interaction_summary), encoding="utf-8")
    _plot(output_dir / "completeness_curves.png", completeness, members)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    all_rows = [row for row in completeness if row["regime"] == "all"]
    summary["median_full_bottleneck_simple_decoder_r2"] = float(np.median([float(row["held_out_r2"]) for row in all_rows if row["concept_set"] == "full_bottleneck_simple_decoder"]))
    summary["median_full_bottleneck_r2"] = 1.0
    summary["median_completeness_relative_to_full_bottleneck"] = float(np.median([float(row["completeness_relative_to_full_bottleneck"]) for row in all_rows if row["concept_set"] == "all_candidates"]))
    summary["median_candidate_to_simple_bottleneck_decoder_ratio"] = summary["median_all_candidates_r2"] / summary["median_full_bottleneck_simple_decoder_r2"]
    summary["direct_gain_bootstrap"] = {"unit": "equilibrium_files", "replicates": int(resolved["direct_gain_bootstrap_replicates"])}
    summary["interaction_summary_rows"] = len(interaction_summary)
    summary["interaction_rows"] = len(interactions); summary["integrated_hessian_rows"] = len(hessian_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resolved["production_compute_wall_time_seconds"] = float(old["config"].get("production_compute_wall_time_seconds", old["wall_time_seconds"]))
    resolved["script_sha256"] = sha256_file(__file__); resolved["completeness_module_sha256"] = sha256_file(repository / "itg_nn/xai/completeness.py")
    for name in ("completeness.csv", "concept_residuals.csv", "interaction_effects.csv", "integrated_hessian_terms.csv", "interaction_summary.csv", "completeness_curves.png", "summary.json"): artifacts.register_existing(name)
    artifacts.finalize(config=resolved, dataset=dataset, checkpoint=checkpoint, member_ids=members, row_ids=old["row_ids"], gradient_set="varied", device=old["device"], repository=repository, command=sys.argv, published_dir=published)
    if published is not None:
        for name in ("completeness.csv", "concept_residuals.csv", "interaction_effects.csv", "integrated_hessian_terms.csv", "interaction_summary.csv", "completeness_curves.png", "summary.json"):
            (published / name).write_bytes((output_dir / name).read_bytes())
    return output_dir


def run(args: argparse.Namespace) -> Path:
    repository = Path(__file__).resolve().parents[1]
    resolved = _resolve(json.loads(args.config.read_text(encoding="utf-8")), args)
    dataset = Path(resolved["dataset"]).resolve()
    checkpoint = (repository / resolved["checkpoint"]).resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (repository / "output/xai/S09" / resolved["run_id"]).resolve()
    published = None if args.no_publish or resolved["mode"] == "pilot" else (repository / resolved["published_dir"]).resolve()
    if args.resume and (output_dir / "manifest.json").is_file():
        return _resume_postprocess(output_dir, resolved, dataset, checkpoint, repository, published)
    set_deterministic_seed(int(resolved["seed"]))
    registry = json.loads((repository / resolved["cohorts"]).read_text(encoding="utf-8"))
    rows = np.asarray(registry["interpretation_panel"]["varied_row_ids"][: int(resolved["panel_varied_rows"])], dtype=np.int64)
    panel = load_hdf5_rows(dataset, rows, gradient_set="varied", include_targets=True)
    if panel.actual_log_heat_flux is None:
        raise RuntimeError("varied targets were not loaded")
    scores, metadata = _concept_scores(dataset, rows, panel.geometry.numpy())
    groups = metadata["equilibrium_file"]
    if len(np.unique(groups)) != len(rows):
        raise RuntimeError("S09 panel must contain one row per equilibrium_files")
    stable = panel.actual_log_heat_flux.numpy() <= float(resolved["stable_threshold_log_Q"])
    ensemble = load_ensemble(checkpoint, device=str(resolved["device"]))
    member_ids = list(registry["member_cohorts"]["stored_validation_top_10"][: int(resolved["members"])])
    member_index = {member: index for index, member in enumerate(ensemble.member_ids)}
    artifacts = RunArtifacts(output_dir)
    completeness_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    hessian_rows: list[dict[str, Any]] = []
    concept_order = list(scores)
    for member_number, member_id in enumerate(member_ids):
        model = InvariantMember(ensemble.models[member_index[member_id]])
        bottleneck, target = _member_values(model, panel, int(resolved["batch_size"]), ensemble.device)
        drives = np.column_stack((metadata["a_over_lt"], metadata["a_over_ln"]))
        sets = {
            "paper_baseline": np.column_stack((drives, scores["log_f_Q"])),
            "paper_geometry": np.column_stack((drives, scores["log_f_Q"], scores["f_stab"], scores["log_FSA_grad_x"])),
            "spatial_geometry": np.column_stack((drives, *[scores[name] for name in concept_order[:8]])),
            "all_candidates": np.column_stack((drives, *[scores[name] for name in concept_order])),
            "full_bottleneck_simple_decoder": np.column_stack((drives, bottleneck)),
        }
        result = grouped_completeness(sets, target, groups, outer_folds=int(resolved["outer_folds"]), inner_folds=int(resolved["inner_folds"]), penalties=tuple(resolved["ridge_penalties"]), seed=int(resolved["seed"]) + member_number * 1000, bootstrap_replicates=int(resolved["bootstrap_replicates"]))
        full_r2 = 1.0
        baseline_r2 = result[0].held_out_r2
        baseline_prediction = result[0].prediction
        for entry in result:
            gain_lower, gain_upper, gain_stability = _paired_gain(target, entry.prediction, baseline_prediction, groups, int(resolved["direct_gain_bootstrap_replicates"]), int(resolved["seed"]) + member_number * 10000 + len(completeness_rows))
            for regime, mask in (("all", np.ones(len(rows), dtype=bool)), ("stable_or_near_floor", stable), ("unstable", ~stable)):
                completeness_rows.append({
                    "member_id": member_id, "concept_set": entry.name, "regime": regime,
                    "held_out_r2": _r2(target, entry.prediction, mask),
                    **_error_metrics(target, entry.prediction, mask),
                    "increment_r2_over_previous": entry.increment_r2 if regime == "all" else "",
                    "increment_ci95_lower": entry.increment_ci95_lower if regime == "all" else "",
                    "increment_ci95_upper": entry.increment_ci95_upper if regime == "all" else "",
                    "bootstrap_selection_stability": entry.selection_stability if regime == "all" else "",
                    "gain_over_paper_baseline": entry.held_out_r2 - baseline_r2 if regime == "all" else "",
                    "gain_over_paper_baseline_ci95_lower": gain_lower if regime == "all" else "",
                    "gain_over_paper_baseline_ci95_upper": gain_upper if regime == "all" else "",
                    "gain_over_paper_baseline_selection_stability": gain_stability if regime == "all" else "",
                    "completeness_relative_to_full_bottleneck": entry.held_out_r2 / full_r2 if regime == "all" else "",
                    "rows": int(mask.sum()), "outer_split_unit": "equilibrium_files", "inner_split_unit": "equilibrium_files",
                    "bootstrap_unit": "equilibrium_files", "decoder": "nested_quadratic_ridge", "estimand": NATIVE_ESTIMAND,
                    "canonical_function": CANONICAL_FUNCTION, "validity_tag": OBSERVED,
                })
            for position in range(len(rows)):
                residual_rows.append({"member_id": member_id, "row_id": int(rows[position]), "equilibrium_file": groups[position], "stable_or_near_floor": bool(stable[position]), "concept_set": entry.name, "target_native_prediction": float(target[position]), "decoder_prediction": float(entry.prediction[position]), "signed_residual": float(target[position] - entry.prediction[position])})
        exact_lower, exact_upper, exact_stability = _paired_gain(target, target, baseline_prediction, groups, int(resolved["direct_gain_bootstrap_replicates"]), int(resolved["seed"]) + member_number * 10000 + 999)
        for regime, mask in (("all", np.ones(len(rows), dtype=bool)), ("stable_or_near_floor", stable), ("unstable", ~stable)):
            completeness_rows.append({"member_id": member_id, "concept_set": "full_bottleneck_exact_head", "regime": regime, "held_out_r2": 1.0, "increment_r2_over_previous": "", "increment_ci95_lower": "", "increment_ci95_upper": "", "bootstrap_selection_stability": "", "gain_over_paper_baseline": 1.0 - baseline_r2 if regime == "all" else "", "gain_over_paper_baseline_ci95_lower": exact_lower if regime == "all" else "", "gain_over_paper_baseline_ci95_upper": exact_upper if regime == "all" else "", "gain_over_paper_baseline_selection_stability": exact_stability if regime == "all" else "", "completeness_relative_to_full_bottleneck": 1.0 if regime == "all" else "", "rows": int(mask.sum()), "outer_split_unit": "not_fitted_exact_trained_head", "inner_split_unit": "not_fitted_exact_trained_head", "bootstrap_unit": "equilibrium_files", "decoder": "trained_canonical_head_exact_ceiling", "estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION, "validity_tag": OBSERVED})
        for position in range(len(rows)):
            residual_rows.append({"member_id": member_id, "row_id": int(rows[position]), "equilibrium_file": groups[position], "stable_or_near_floor": bool(stable[position]), "concept_set": "full_bottleneck_exact_head", "target_native_prediction": float(target[position]), "decoder_prediction": float(target[position]), "signed_residual": 0.0})
        selected = {name: scores[name] for name in ("log_f_Q", "f_stab", "log_compression", "bad_curvature", "geodesic_curvature", "parallel_scale", "cross_channel_colocation", "log_FSA_grad_x")}
        for regime_number, (regime, regime_mask) in enumerate((("all", np.ones(len(rows), dtype=bool)), ("stable_or_near_floor", stable), ("unstable", ~stable))):
            selected_regime = {name: values[regime_mask] for name, values in selected.items()}
            groups_regime = groups[regime_mask]; target_regime = target[regime_mask]
            for drive_name, drive in (("a_over_LT", metadata["a_over_lt"]), ("a_over_Ln", metadata["a_over_ln"])):
                effects = stratified_directional_effects(selected_regime, drive[regime_mask], target_regime, groups_regime, bins=int(resolved["interaction_bins"]), bootstrap_replicates=int(resolved["bootstrap_replicates"]), seed=int(resolved["seed"]) + member_number * 10000 + regime_number * 2000 + (0 if drive_name == "a_over_LT" else 1000))
                for effect in effects:
                    interaction_rows.append({"member_id": member_id, "regime": regime, "drive": drive_name, **effect.__dict__, "estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION, "finite_difference_kind": "grouped_observed_directional_slope", "integrated_hessian_proxy": "quadratic_decoder_concept_by_drive_term", "perturbation_validity_tag": OBSERVED})
            hessian_values = np.column_stack((metadata["a_over_lt"][regime_mask], metadata["a_over_ln"][regime_mask], *selected_regime.values()))
            hessian = grouped_integrated_hessian(hessian_values, target_regime, groups_regime, concept_names=tuple(selected), outer_folds=int(resolved["outer_folds"]), inner_folds=int(resolved["inner_folds"]), penalties=tuple(resolved["ridge_penalties"]), bootstrap_replicates=int(resolved["bootstrap_replicates"]), seed=int(resolved["seed"]) + member_number * 10000 + regime_number * 2000 + 1500)
            for term in hessian:
                hessian_rows.append({"member_id": member_id, "regime": regime, "drive": ("a_over_LT" if term.drive_index == 0 else "a_over_Ln"), **term.__dict__, "estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION, "method": "held_out_quadratic_decoder_four_point_mixed_difference"})
        print(f"{member_id} complete", flush=True)
    artifacts.write_text("completeness.csv", _csv_text(completeness_rows))
    artifacts.write_text("concept_residuals.csv", _csv_text(residual_rows))
    artifacts.write_text("interaction_effects.csv", _csv_text(interaction_rows))
    artifacts.write_text("integrated_hessian_terms.csv", _csv_text(hessian_rows))
    interaction_summary_rows = []
    for regime, drive, concept, bin_index in sorted({(row["regime"], row["drive"], row["concept"], row["bin_index"]) for row in interaction_rows}):
        selected_rows = [row for row in interaction_rows if (row["regime"], row["drive"], row["concept"], row["bin_index"]) == (regime, drive, concept, bin_index)]
        slopes = np.asarray([float(row["slope"]) for row in selected_rows])
        interaction_summary_rows.append({"regime": regime, "drive": drive, "concept": concept, "bin_index": bin_index, "member_count": len(slopes), "median_slope": float(np.median(slopes)), "minimum_slope": float(slopes.min()), "maximum_slope": float(slopes.max()), "member_sign_agreement": float(max(np.mean(slopes > 0), np.mean(slopes < 0))), "source": "varied-gradient panel only", "validity_tag": OBSERVED})
    artifacts.write_text("interaction_summary.csv", _csv_text(interaction_summary_rows))
    figure = output_dir / "completeness_curves.png"; _plot(figure, completeness_rows, member_ids); artifacts.register_existing(figure.name)
    all_rows = [row for row in completeness_rows if row["regime"] == "all"]
    baseline = [row for row in all_rows if row["concept_set"] == "paper_baseline"]
    candidates = [row for row in all_rows if row["concept_set"] == "all_candidates"]
    full = [row for row in all_rows if row["concept_set"] == "full_bottleneck_exact_head"]
    simple_full = [row for row in all_rows if row["concept_set"] == "full_bottleneck_simple_decoder"]
    summary = {
        "step": "S09", "run_id": resolved["run_id"], "mode": resolved["mode"], "estimand": NATIVE_ESTIMAND,
        "canonical_function": CANONICAL_FUNCTION, "members": member_ids,
        "cohort": {"rows": len(rows), "unique_equilibrium_files": len(np.unique(groups)), "stable_or_near_floor": int(stable.sum()), "unstable": int((~stable).sum())},
        "median_paper_baseline_r2": float(np.median([row["held_out_r2"] for row in baseline])),
        "median_all_candidates_r2": float(np.median([row["held_out_r2"] for row in candidates])),
        "median_full_bottleneck_r2": float(np.median([row["held_out_r2"] for row in full])),
        "median_full_bottleneck_simple_decoder_r2": float(np.median([row["held_out_r2"] for row in simple_full])),
        "median_candidate_to_simple_bottleneck_decoder_ratio": float(np.median([row["held_out_r2"] for row in candidates])) / float(np.median([row["held_out_r2"] for row in simple_full])),
        "median_gain_over_paper_baseline": float(np.median([row["gain_over_paper_baseline"] for row in candidates])),
        "median_completeness_relative_to_full_bottleneck": float(np.median([row["completeness_relative_to_full_bottleneck"] for row in candidates])),
        "bootstrap": {"unit": "equilibrium_files", "replicates": int(resolved["bootstrap_replicates"])},
        "direct_gain_bootstrap": {"unit": "equilibrium_files", "replicates": int(resolved["direct_gain_bootstrap_replicates"])},
        "interaction_rows": len(interaction_rows), "integrated_hessian_rows": len(hessian_rows), "interaction_summary_rows": len(interaction_summary_rows), "interaction_source": "varied-gradient panel only",
    }
    artifacts.write_json("summary.json", summary)
    resolved["script_sha256"] = sha256_file(__file__)
    resolved["completeness_module_sha256"] = sha256_file(repository / "itg_nn/xai/completeness.py")
    artifacts.finalize(config=resolved, dataset=dataset, checkpoint=checkpoint, member_ids=member_ids, row_ids=rows, gradient_set="varied", device=ensemble.device, repository=repository, command=sys.argv, published_dir=published)
    if published is not None:
        for name in ("completeness.csv", "concept_residuals.csv", "interaction_effects.csv", "integrated_hessian_terms.csv", "interaction_summary.csv", "completeness_curves.png", "summary.json"):
            (published / name).write_bytes((output_dir / name).read_bytes())
    return output_dir


def main() -> int:
    print(run(build_parser().parse_args())); return 0


if __name__ == "__main__":
    raise SystemExit(main())
