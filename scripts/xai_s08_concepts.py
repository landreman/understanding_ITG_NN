#!/usr/bin/env python3
"""Run S08 grouped concept probes and TCAV-like layer use tests."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from itg_nn.data import load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.bottleneck import registered_invariants
from itg_nn.xai.concepts import (
    canonical_output_from_layer,
    grouped_nested_sparse_probe,
    invariant_layer_maps,
    matched_extremes,
    representation_direction_use,
)
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import CANONICAL_FUNCTION, InvariantMember


NATIVE_ESTIMAND = "native max(log Q, -2)"
OBSERVED = "observed-comparison"
OFF_MANIFOLD = "deliberately_off_manifold_diagnostic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/xai/S08_concepts.json"))
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
        value = getattr(args, name)
        if value is not None:
            resolved[name] = value
    if args.rows is not None:
        resolved["panel_varied_rows"] = args.rows
    for value, name in (
        (args.dataset, "dataset"),
        (args.checkpoint, "checkpoint"),
        (args.published_dir, "published_dir"),
    ):
        if value is not None:
            resolved[name] = str(value)
    return resolved


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def _take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(rows, return_inverse=True)
    return dataset[unique][inverse]


def _csv_text(rows: list[dict[str, Any]]) -> str:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _concept_scores(dataset: Path, rows: np.ndarray, geometry: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with h5py.File(dataset, "r") as h5_file:
        scalar_names = tuple(_decode(h5_file["scalar_features"][:]))
        scalars = _take(h5_file["scalar_feature_matrix"], rows).astype(np.float64)
        classes = _take(h5_file["equilibrium_class"], rows).astype(np.int16)
        groups = _decode(_take(h5_file["equilibrium_files"], rows))
        q_vs_z = _take(h5_file["varied_gradient_simulations/Q_avgs_vs_z"], rows).astype(np.float64)
        zonal = _take(h5_file["varied_gradient_simulations/zonal_phi2_amplitudes"], rows).astype(np.float64)
        a_lt = _take(h5_file["varied_gradient_simulations/a_over_LT"], rows).astype(np.float64)
        a_ln = _take(h5_file["varied_gradient_simulations/a_over_Ln"], rows).astype(np.float64)
    invariant = registered_invariants(geometry, scalars, scalar_names)
    bmag = geometry[:, :, 0].astype(np.float64)
    gb = geometry[:, :, 1].astype(np.float64)
    cv = geometry[:, :, 2].astype(np.float64)
    geodesic = geometry[:, :, 3].astype(np.float64)
    grad_x = np.sqrt(np.maximum(geometry[:, :, 6], 0)).astype(np.float64)
    compression = np.log(np.maximum(np.mean(grad_x**4 / bmag**6, axis=1), 1e-30))
    bad_curvature = np.mean(np.maximum(cv, 0), axis=1)
    geodesic_magnitude = np.mean(np.abs(geodesic), axis=1)
    scaled = geometry.astype(np.float64)
    scale = np.subtract(*np.quantile(scaled, [0.75, 0.25], axis=(0, 1)))
    scale[scale < 1e-12] = 1.0
    spectrum = np.abs(np.fft.rfft(scaled / scale, axis=1)) ** 2
    frequency = np.arange(spectrum.shape[1], dtype=np.float64)[None, :, None]
    parallel_scale = np.sum(spectrum * frequency, axis=(1, 2)) / np.maximum(np.sum(spectrum, axis=(1, 2)), 1e-30)
    colocation = np.mean(np.maximum(cv, 0) * grad_x**3 / bmag, axis=1)
    q_centered = q_vs_z - q_vs_z.mean(axis=1, keepdims=True)
    local_q = np.sqrt(np.mean(q_centered**2, axis=1)) / np.maximum(np.mean(np.abs(q_vs_z), axis=1), 1e-30)
    scores = {
        "log_f_Q": invariant["log_f_Q"],
        "f_stab": invariant["f_stab"],
        "log_compression": compression,
        "bad_curvature": bad_curvature,
        "geodesic_curvature": geodesic_magnitude,
        "parallel_scale": parallel_scale,
        "cross_channel_colocation": colocation,
        "log_FSA_grad_x": invariant["log_FSA_grad_x"],
        "local_Qz_concentration": local_q,
        "log10_zonal_phi2": np.log10(np.maximum(zonal, np.finfo(np.float64).tiny)),
    }
    metadata = {
        "equilibrium_class": classes,
        "equilibrium_file": groups,
        "a_over_lt": a_lt,
        "a_over_ln": a_ln,
        "geometry_scale": np.log(np.maximum(np.mean(grad_x, axis=1), 1e-30)),
    }
    return scores, metadata


def _layer_representations(member: InvariantMember, geometry: torch.Tensor, batch_size: int, device: torch.device) -> list[np.ndarray]:
    chunks: list[list[np.ndarray]] = [[] for _ in range(5)]
    with torch.inference_mode():
        for start in range(0, len(geometry), batch_size):
            maps = invariant_layer_maps(member, geometry[start : start + batch_size].to(device))
            for layer, values in enumerate(maps):
                chunks[layer].append(values.mean(-1).cpu().numpy())
    return [np.concatenate(layer_chunks).astype(np.float64) for layer_chunks in chunks]


def _bootstrap_interval(values: np.ndarray, groups: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        draws[draw] = float(np.mean(values[positions]))
    return tuple(np.quantile(draws, [0.025, 0.975]))  # type: ignore[return-value]


def _plot_matrix(path: Path, rows: list[dict[str, Any]], members: list[str], concepts: list[str]) -> None:
    figure, axes = plt.subplots(len(members), 2, figsize=(14, 3.6 * len(members)), squeeze=False)
    for member_index, member_id in enumerate(members):
        for column, metric in enumerate(("encoded_r2", "mean_directional_derivative")):
            matrix = np.full((len(concepts), 5), np.nan)
            for row in rows:
                if row["member_id"] == member_id:
                    matrix[concepts.index(str(row["concept"])), int(row["layer_index"])] = float(row[metric])
            image = axes[member_index, column].imshow(matrix, aspect="auto", cmap="coolwarm")
            axes[member_index, column].set_title(f"{member_id}: {metric}")
            axes[member_index, column].set_xticks(range(5), [f"L{i + 1}" for i in range(5)])
            axes[member_index, column].set_yticks(range(len(concepts)), concepts)
            figure.colorbar(image, ax=axes[member_index, column], shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _add_claim_gates(row: dict[str, Any]) -> dict[str, Any]:
    encoded = float(row["encoded_r2"])
    permuted = float(row["permuted_r2"])
    lower = float(row["directional_derivative_ci95_lower"])
    upper = float(row["directional_derivative_ci95_upper"])
    encoded_pass = encoded >= 0.1 and encoded - permuted >= 0.1
    stable_pass = float(row["counterexample_sign_agreement"]) >= 0.8
    interval_pass = lower * upper > 0
    intervention_pass = float(row["intervention_to_random_ratio"]) > 1.0
    balance_pass = float(row.get("counterexample_max_abs_smd", 0.0)) <= 0.25
    row.update(
        {
            "encoded_generalizes_by_equilibrium": encoded_pass,
            "tcav_stable_across_counterexamples": stable_pass,
            "tcav_ci_excludes_zero": interval_pass,
            "direction_intervention_beats_random": intervention_pass,
            "counterexample_balance_pass": balance_pass,
            "use_claim_permitted": encoded_pass
            and stable_pass
            and interval_pass
            and intervention_pass
            and balance_pass,
            "use_claim_rule": "encoded_r2>=0.1; gain_over_permutation>=0.1; counterexample_sign_agreement>=0.8; grouped_CI_excludes_0; intervention_ratio>1; matched_max_abs_SMD<=0.25",
        }
    )
    return row


def _matching_balance(match_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for concept in sorted({str(row["concept"]) for row in match_rows}):
        concept_rows = [row for row in match_rows if row["concept"] == concept]
        high = [row for row in concept_rows if row["role"] == "high"]
        low = [row for row in concept_rows if row["role"] == "low"]
        values: dict[str, float] = {}
        for field in ("a_over_lt", "a_over_ln", "geometry_scale"):
            high_values = np.asarray([float(row[field]) for row in high])
            low_values = np.asarray([float(row[field]) for row in low])
            pooled = np.sqrt((high_values.var() + low_values.var()) / 2)
            values[f"smd_{field}"] = (
                0.0 if pooled == 0 else float((high_values.mean() - low_values.mean()) / pooled)
            )
        maximum = max(abs(value) for value in values.values())
        result.append(
            {
                "concept": concept,
                "high_count": len(high),
                "low_count": len(low),
                **values,
                "max_abs_smd": maximum,
                "balance_threshold": 0.25,
                "balance_pass": maximum <= 0.25,
                "matching_method": "linear_nuisance_residual_extremes_then_within_class_nearest_neighbor",
                "validity_tag": OBSERVED,
            }
        )
    return result


def _resume_claim_gates(
    output_dir: Path,
    resolved: dict[str, Any],
    dataset: Path,
    checkpoint: Path,
    repository: Path,
    published: Path | None,
) -> Path:
    old_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if old_manifest["dataset"]["sha256"] != sha256_file(dataset):
        raise RuntimeError("resume dataset fingerprint changed")
    if old_manifest["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise RuntimeError("resume checkpoint fingerprint changed")
    match_rows = list(csv.DictReader((output_dir / "matched_examples.csv").open(newline="", encoding="utf-8")))
    balance_rows = _matching_balance(match_rows)
    balance = {row["concept"]: row for row in balance_rows}
    matrix = list(csv.DictReader((output_dir / "encoding_use_matrix.csv").open(newline="", encoding="utf-8")))
    use = list(csv.DictReader((output_dir / "tcav_use.csv").open(newline="", encoding="utf-8")))
    matrix = [
        _add_claim_gates(
            {**dict(row), "counterexample_max_abs_smd": balance[row["concept"]]["max_abs_smd"]}
        )
        for row in matrix
    ]
    probe_lookup = {
        (row["member_id"], row["layer_index"], row["concept"]): row
        for row in matrix
    }
    gated_use: list[dict[str, Any]] = []
    for row in use:
        merged = {**probe_lookup[(row["member_id"], row["layer_index"], row["concept"])], **row}
        gated = _add_claim_gates(merged)
        gated_use.append({**row, **{key: gated[key] for key in (
            "encoded_generalizes_by_equilibrium",
            "tcav_stable_across_counterexamples",
            "tcav_ci_excludes_zero",
            "direction_intervention_beats_random",
            "counterexample_balance_pass",
            "use_claim_permitted",
            "use_claim_rule",
        )}})
    (output_dir / "encoding_use_matrix.csv").write_text(_csv_text(matrix), encoding="utf-8")
    (output_dir / "tcav_use.csv").write_text(_csv_text(gated_use), encoding="utf-8")
    (output_dir / "matching_balance.csv").write_text(_csv_text(balance_rows), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["encoded_generalizes_fraction"] = float(np.mean([row["encoded_generalizes_by_equilibrium"] for row in matrix]))
    summary["use_claim_permitted_fraction"] = float(np.mean([row["use_claim_permitted"] for row in matrix]))
    summary["use_claim_rule"] = matrix[0]["use_claim_rule"]
    summary["counterexample_balance_failed_concepts"] = [
        row["concept"] for row in balance_rows if not row["balance_pass"]
    ]
    summary["counterexample_worst_max_abs_smd"] = max(
        float(row["max_abs_smd"]) for row in balance_rows
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resolved["production_compute_wall_time_seconds"] = old_manifest["wall_time_seconds"]
    artifacts = RunArtifacts(output_dir)
    for name in ("matched_examples.csv", "matching_balance.csv", "probe_scores.csv", "tcav_use.csv", "encoding_use_matrix.csv", "layer_concept_matrix.png", "summary.json"):
        artifacts.register_existing(name)
    artifacts.finalize(
        config=resolved,
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=old_manifest["member_ids"],
        row_ids=old_manifest["row_ids"],
        gradient_set="varied",
        device=old_manifest["device"],
        repository=repository,
        command=sys.argv,
        published_dir=published,
    )
    if published is not None:
        for name in ("matched_examples.csv", "matching_balance.csv", "probe_scores.csv", "tcav_use.csv", "encoding_use_matrix.csv", "layer_concept_matrix.png", "summary.json"):
            (published / name).write_bytes((output_dir / name).read_bytes())
    return output_dir


def run(args: argparse.Namespace) -> Path:
    repository = Path(__file__).resolve().parents[1]
    resolved = _resolve(json.loads(args.config.read_text(encoding="utf-8")), args)
    dataset = Path(resolved["dataset"]).resolve()
    checkpoint = (repository / resolved["checkpoint"]).resolve()
    cohorts_path = (repository / resolved["cohorts"]).resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else (repository / "output/xai/S08" / resolved["run_id"]).resolve())
    published = None if args.no_publish or resolved["mode"] == "pilot" else (repository / resolved["published_dir"]).resolve()
    set_deterministic_seed(int(resolved["seed"]))
    resolved["script_sha256"] = sha256_file(__file__)
    resolved["concept_module_sha256"] = sha256_file(repository / "itg_nn/xai/concepts.py")
    if args.resume and (output_dir / "manifest.json").is_file():
        return _resume_claim_gates(
            output_dir, resolved, dataset, checkpoint, repository, published
        )
    artifacts = RunArtifacts(output_dir)
    registry = json.loads(cohorts_path.read_text(encoding="utf-8"))
    registered = np.asarray(registry["interpretation_panel"]["varied_row_ids"], dtype=np.int64)
    row_count = int(resolved["panel_varied_rows"])
    rows = registered[:row_count]
    panel = load_hdf5_rows(dataset, rows, gradient_set="varied", include_targets=True)
    if panel.actual_log_heat_flux is None:
        raise RuntimeError("varied targets were not loaded")
    scores, metadata = _concept_scores(dataset, rows, panel.geometry.numpy())
    if len(np.unique(metadata["equilibrium_file"])) != len(rows):
        raise RuntimeError("S08 panel must contain one row per equilibrium")
    nuisance = np.column_stack((metadata["equilibrium_class"], metadata["a_over_lt"], metadata["a_over_ln"], metadata["geometry_scale"]))
    matches = {
        concept: matched_extremes(values, nuisance, metadata["equilibrium_file"], fraction=float(resolved["extreme_fraction"]), seed=int(resolved["seed"]) + index)
        for index, (concept, values) in enumerate(scores.items())
    }
    match_rows: list[dict[str, Any]] = []
    for concept, match in matches.items():
        for role, positions in (("high", match.high), ("low", match.low)):
            for position in positions:
                match_rows.append({"concept": concept, "role": role, "row_id": int(rows[position]), "equilibrium_file": metadata["equilibrium_file"][position], "equilibrium_class": int(metadata["equilibrium_class"][position]), "a_over_lt": float(metadata["a_over_lt"][position]), "a_over_ln": float(metadata["a_over_ln"][position]), "geometry_scale": float(metadata["geometry_scale"][position]), "validity_tag": match.validity_tag})
    balance_rows = _matching_balance(match_rows)
    balance_by_concept = {row["concept"]: row for row in balance_rows}
    ensemble = load_ensemble(checkpoint, device=str(resolved["device"]))
    member_ids = list(registry["member_cohorts"]["stored_validation_top_10"][: int(resolved["members"])])
    index_by_id = {member_id: index for index, member_id in enumerate(ensemble.member_ids)}
    models = {member_id: InvariantMember(ensemble.models[index_by_id[member_id]]) for member_id in member_ids}
    matrix_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    use_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(resolved["seed"]) + 991)
    use_positions = np.sort(rng.choice(len(rows), size=min(int(resolved["use_rows"]), len(rows)), replace=False))
    stable = panel.actual_log_heat_flux.numpy() <= float(resolved["stable_threshold_log_Q"])
    for member_index, member_id in enumerate(member_ids):
        model = models[member_id]
        representations = _layer_representations(model, panel.geometry, int(resolved["batch_size"]), ensemble.device)
        use_geometry = panel.geometry[use_positions].to(ensemble.device)
        use_lt = panel.a_over_lt[use_positions].to(ensemble.device)
        use_ln = panel.a_over_ln[use_positions].to(ensemble.device)
        for layer_index, representation in enumerate(representations):
            with torch.enable_grad():
                layer_map = invariant_layer_maps(model, use_geometry)[layer_index].detach()
            feature_scale = np.subtract(*np.quantile(representation, [0.75, 0.25], axis=0))
            feature_scale[feature_scale < 1e-8] = 1.0
            for concept_index, (concept, target) in enumerate(scores.items()):
                probe = grouped_nested_sparse_probe(representation, target, metadata["equilibrium_file"], outer_folds=int(resolved["outer_folds"]), inner_folds=int(resolved["inner_folds"]), penalties=tuple(resolved["probe_penalties"]), seed=int(resolved["seed"]) + 1000 * member_index + 100 * layer_index + concept_index)
                permuted = grouped_nested_sparse_probe(representation, target, metadata["equilibrium_file"], outer_folds=int(resolved["outer_folds"]), inner_folds=int(resolved["inner_folds"]), penalties=tuple(resolved["probe_penalties"]), seed=int(resolved["seed"]) + 1000 * member_index + 100 * layer_index + concept_index, permute_target=True)
                coefficient = probe.coefficients * feature_scale
                if np.linalg.norm(coefficient) < 1e-12:
                    coefficient = (representation[matches[concept].high].mean(0) - representation[matches[concept].low].mean(0)) / feature_scale
                direction = torch.as_tensor(coefficient, dtype=layer_map.dtype, device=layer_map.device)
                direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
                raw_direction = direction * torch.as_tensor(feature_scale, dtype=layer_map.dtype, device=layer_map.device)
                raw_direction = raw_direction / torch.sqrt(torch.mean(raw_direction**2)).clamp_min(1e-12)
                def output_from_mean_edit(values: torch.Tensor) -> torch.Tensor:
                    delta = values - layer_map.mean(-1)
                    changed = layer_map + delta[:, :, None]
                    return canonical_output_from_layer(model, layer_index, changed, use_lt, use_ln)
                use = representation_direction_use(output_from_mean_edit, layer_map.mean(-1), raw_direction, random_directions=int(resolved["random_directions"]), intervention_scale=float(resolved["intervention_scale"]), seed=int(resolved["seed"]) + 10000 * member_index + 100 * layer_index + concept_index)
                # Multiple matched counterexample subsets calibrate TCAV direction stability.
                counter_means: list[float] = []
                counter_positive: list[float] = []
                counter_derivatives: list[np.ndarray] = []
                base = layer_map.detach().clone().requires_grad_(True)
                output = canonical_output_from_layer(model, layer_index, base, use_lt, use_ln)
                # A direction is added uniformly at every position, so its
                # derivative is the sum (not mean) of spatial cell gradients.
                gradient = torch.autograd.grad(output.sum(), base)[0].sum(-1)
                for counter_set in range(int(resolved["counterexample_sets"])):
                    local_rng = np.random.default_rng(int(resolved["seed"]) + 100000 * member_index + 1000 * layer_index + 10 * concept_index + counter_set)
                    high = local_rng.choice(matches[concept].high, size=max(2, int(0.75 * len(matches[concept].high))), replace=False)
                    low = local_rng.choice(matches[concept].low, size=max(2, int(0.75 * len(matches[concept].low))), replace=False)
                    counter_direction = (representation[high].mean(0) - representation[low].mean(0)) / feature_scale
                    counter_raw = counter_direction * feature_scale
                    counter_tensor = torch.as_tensor(counter_raw, dtype=gradient.dtype, device=gradient.device)
                    counter_tensor /= torch.linalg.vector_norm(counter_tensor).clamp_min(1e-12)
                    derivative = gradient @ counter_tensor
                    counter_means.append(float(derivative.mean()))
                    counter_positive.append(float((derivative > 0).float().mean()))
                    counter_derivatives.append(derivative.detach().cpu().numpy())
                derivative_values = np.mean(counter_derivatives, axis=0)
                lower, upper = _bootstrap_interval(derivative_values, metadata["equilibrium_file"][use_positions], int(resolved["bootstrap_replicates"]), int(resolved["seed"]) + member_index * 1000 + layer_index * 100 + concept_index)
                use_stable = stable[use_positions]
                encoded_stable = float("nan") if np.count_nonzero(stable) < 2 else 1.0 - float(np.sum((target[stable] - probe.predictions[stable]) ** 2)) / max(float(np.sum((target[stable] - target[stable].mean()) ** 2)), 1e-30)
                encoded_unstable = float("nan") if np.count_nonzero(~stable) < 2 else 1.0 - float(np.sum((target[~stable] - probe.predictions[~stable]) ** 2)) / max(float(np.sum((target[~stable] - target[~stable].mean()) ** 2)), 1e-30)
                common = {"member_id": member_id, "layer_index": layer_index, "layer_name": f"canonical_atrous_relu_pool_{layer_index + 1}", "concept": concept, "estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION, "stable_rows": int(stable.sum()), "unstable_rows": int((~stable).sum()), "counterexample_max_abs_smd": balance_by_concept[concept]["max_abs_smd"]}
                probe_row = {**common, "encoded_r2": probe.held_out_r2, "encoded_r2_stable_or_near_floor": encoded_stable, "encoded_r2_unstable": encoded_unstable, "permuted_r2": permuted.held_out_r2, "nonzero_fraction": probe.nonzero_fraction, "outer_split_unit": "equilibrium_files", "inner_split_unit": "equilibrium_files", "encoded_column": True, "used_column": False, "validity_tag": OBSERVED}
                use_row = {**common, "mean_directional_derivative": float(np.mean(counter_means)), "mean_directional_derivative_stable_or_near_floor": float(np.mean(derivative_values[use_stable])) if np.any(use_stable) else float("nan"), "mean_directional_derivative_unstable": float(np.mean(derivative_values[~use_stable])) if np.any(~use_stable) else float("nan"), "directional_derivative_ci95_lower": lower, "directional_derivative_ci95_upper": upper, "tcav_positive_fraction": float(np.mean(counter_positive)), "counterexample_set_sd": float(np.std(counter_means)), "counterexample_sign_agreement": float(max(np.mean(np.asarray(counter_means) > 0), np.mean(np.asarray(counter_means) < 0))), "intervention_rms": use.intervention_rms, "random_intervention_rms_median": use.random_intervention_rms_median, "intervention_to_random_ratio": use.intervention_rms / max(use.random_intervention_rms_median, 1e-30), "perturbation_validity_tag": OFF_MANIFOLD, "encoded_column": False, "used_column": True}
                probe_rows.append(probe_row)
                gated_matrix = _add_claim_gates({**common, **probe_row, **use_row})
                for key in (
                    "encoded_generalizes_by_equilibrium",
                    "tcav_stable_across_counterexamples",
                    "tcav_ci_excludes_zero",
                    "direction_intervention_beats_random",
                    "counterexample_balance_pass",
                    "use_claim_permitted",
                    "use_claim_rule",
                ):
                    use_row[key] = gated_matrix[key]
                use_rows.append(use_row)
                matrix_rows.append(gated_matrix)
            random_target = np.random.default_rng(
                int(resolved["seed"]) + 900000 + member_index * 10 + layer_index
            ).normal(size=len(rows))
            random_probe = grouped_nested_sparse_probe(
                representation,
                random_target,
                metadata["equilibrium_file"],
                outer_folds=int(resolved["outer_folds"]),
                inner_folds=int(resolved["inner_folds"]),
                penalties=tuple(resolved["probe_penalties"]),
                seed=int(resolved["seed"]) + 910000 + member_index * 10 + layer_index,
            )
            probe_rows.append(
                {
                    "member_id": member_id,
                    "layer_index": layer_index,
                    "layer_name": f"canonical_atrous_relu_pool_{layer_index + 1}",
                    "concept": "random_concept_control",
                    "control_type": "equilibrium_level_random_concept",
                    "encoded_r2": random_probe.held_out_r2,
                    "outer_split_unit": "equilibrium_files",
                    "inner_split_unit": "equilibrium_files",
                    "estimand": NATIVE_ESTIMAND,
                    "validity_tag": OBSERVED,
                }
            )
            print(f"{member_id} layer {layer_index + 1}/5 complete", flush=True)
    artifacts.write_text("matched_examples.csv", _csv_text(match_rows))
    artifacts.write_text("matching_balance.csv", _csv_text(balance_rows))
    artifacts.write_text("probe_scores.csv", _csv_text(probe_rows))
    artifacts.write_text("tcav_use.csv", _csv_text(use_rows))
    artifacts.write_text("encoding_use_matrix.csv", _csv_text(matrix_rows))
    figure_path = output_dir / "layer_concept_matrix.png"
    _plot_matrix(figure_path, matrix_rows, member_ids, list(scores))
    artifacts.register_existing(figure_path.name)
    summary = {
        "step": "S08", "mode": resolved["mode"], "run_id": resolved["run_id"],
        "estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION,
        "cohort": {"panel_rows": len(rows), "unique_equilibrium_files": len(np.unique(metadata["equilibrium_file"])), "stable_or_near_floor": int(stable.sum()), "unstable": int((~stable).sum())},
        "members": member_ids, "concepts": list(scores), "layers": 5,
        "encoded_and_used_separate": True,
        "median_probe_r2": float(np.median([row["encoded_r2"] for row in probe_rows if row["concept"] != "random_concept_control"])),
        "median_permuted_r2": float(np.median([row["permuted_r2"] for row in probe_rows if "permuted_r2" in row])),
        "median_random_concept_r2": float(np.median([row["encoded_r2"] for row in probe_rows if row["concept"] == "random_concept_control"])),
        "stable_counterexample_fraction": float(np.mean([row["counterexample_sign_agreement"] >= 0.8 for row in use_rows])),
        "intervention_beats_random_fraction": float(np.mean([row["intervention_to_random_ratio"] > 1 for row in use_rows])),
        "encoded_generalizes_fraction": float(np.mean([row["encoded_generalizes_by_equilibrium"] for row in matrix_rows])),
        "use_claim_permitted_fraction": float(np.mean([row["use_claim_permitted"] for row in matrix_rows])),
        "use_claim_rule": matrix_rows[0]["use_claim_rule"],
        "counterexample_balance_failed_concepts": [row["concept"] for row in balance_rows if not row["balance_pass"]],
        "counterexample_worst_max_abs_smd": max(float(row["max_abs_smd"]) for row in balance_rows),
        "bootstrap": {"unit": "equilibrium_files", "replicates": int(resolved["bootstrap_replicates"])},
        "deferred": ["network dissection IoU/mutual-information/selectivity tables"] if resolved["mode"] == "production" else [],
    }
    artifacts.write_json("summary.json", summary)
    artifacts.finalize(config=resolved, dataset=dataset, checkpoint=checkpoint, member_ids=member_ids, row_ids=rows, gradient_set="varied", device=ensemble.device, repository=repository, command=sys.argv, published_dir=published)
    if published is not None:
        published.mkdir(parents=True, exist_ok=True)
        for name in ("matched_examples.csv", "matching_balance.csv", "probe_scores.csv", "tcav_use.csv", "encoding_use_matrix.csv", "layer_concept_matrix.png", "summary.json"):
            source = output_dir / name
            (published / name).write_bytes(source.read_bytes())
    return output_dir


def main() -> int:
    path = run(build_parser().parse_args())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
