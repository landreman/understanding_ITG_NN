#!/usr/bin/env python3
"""Run S10 cross-member representation and motif comparisons."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import shutil
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
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
from itg_nn.xai.concepts import invariant_layer_maps
from itg_nn.xai.cross_model import (
    HIDDEN_INTERVENTION_VALIDITY,
    functional_similarity,
    grouped_bootstrap_cka,
    grouped_bootstrap_match_recurrence,
    linear_cka,
    match_units,
    mean_replacement_effects,
    member_distance_matrix,
)
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import CANONICAL_FUNCTION, InvariantMember
from itg_nn.xai.unit_semantics import physics_concept_traces


NATIVE_ESTIMAND = "native max(log Q, -2)"
BOOTSTRAP_GROUP = "equilibrium_files"


def _member_cohort(rank: int) -> str:
    if not 1 <= rank <= 100:
        raise ValueError("stored-validation rank must lie in [1, 100]")
    if rank <= 10:
        return "stored_validation_top_10"
    if rank <= 50:
        return "stored_validation_ranks_11_50"
    return "stored_validation_ranks_51_100"


def _consensus_components(
    unit_ids: Sequence[str],
    edges: Sequence[dict[str, Any]],
    *,
    minimum_recurrence: float,
    minimum_causal_similarity: float,
) -> list[dict[str, Any]]:
    adjacency = {unit_id: set() for unit_id in unit_ids}
    parent = {unit_id: unit_id for unit_id in unit_ids}
    component_members = {unit_id: {unit_id.rsplit(":u", 1)[0]} for unit_id in unit_ids}

    def root(unit_id: str) -> str:
        while parent[unit_id] != unit_id:
            parent[unit_id] = parent[parent[unit_id]]
            unit_id = parent[unit_id]
        return unit_id

    accepted = []
    ordered = sorted(
        edges,
        key=lambda edge: float(edge["recurrence"]) * float(edge["causal_similarity"]),
        reverse=True,
    )
    for edge in ordered:
        if (
            float(edge["recurrence"]) < minimum_recurrence
            or float(edge["causal_similarity"]) < minimum_causal_similarity
        ):
            continue
        left, right = str(edge["left"]), str(edge["right"])
        if left not in adjacency or right not in adjacency:
            raise ValueError("consensus edge references an unknown unit")
        left_root, right_root = root(left), root(right)
        if left_root == right_root:
            accepted.append(edge)
            adjacency[left].add(right)
            adjacency[right].add(left)
            continue
        # A correspondence catalog is one-to-one within a member. Pairwise
        # assignments can form inconsistent cycles, so reject any union that
        # would put two units from the same member into one consensus motif.
        if component_members[left_root] & component_members[right_root]:
            continue
        parent[right_root] = left_root
        component_members[left_root] |= component_members[right_root]
        adjacency[left].add(right)
        adjacency[right].add(left)
        accepted.append(edge)
    motifs = []
    visited: set[str] = set()
    for start in unit_ids:
        if start in visited or not adjacency[start]:
            continue
        stack = [start]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - visited, reverse=True))
        members = {unit.rsplit(":u", 1)[0] for unit in component}
        if len(members) < 2:
            continue
        component_edges = [
            edge for edge in accepted
            if str(edge["left"]) in component and str(edge["right"]) in component
        ]
        motifs.append(
            {
                "unit_ids": sorted(component),
                "member_count": len(members),
                "edge_count": len(component_edges),
                "minimum_recurrence": min(float(edge["recurrence"]) for edge in component_edges),
                "minimum_causal_similarity": min(float(edge["causal_similarity"]) for edge in component_edges),
            }
        )
    return motifs


def _annotate_motifs(
    motif_rows: list[dict[str, Any]], *, s05_unit_motifs: Path
) -> None:
    """Attach only independently supported S05 names to consensus motifs."""

    supported: dict[str, str] = {}
    screened: set[str] = set()
    with s05_unit_motifs.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            screened.add(row["unit_id"])
            if row["motif_status"] == "supported_named_motif":
                supported[row["unit_id"]] = row["claimed_concept"]
    for row in motif_rows:
        units = str(row["unit_ids"]).split("|")
        concepts = sorted({supported[unit] for unit in units if unit in supported})
        row["definition"] = (
            "functional_signature_and_off_manifold_mean_ablation_agree_"
            "in_both_output_regimes"
        )
        row["s05_screened_unit_count"] = sum(unit in screened for unit in units)
        row["s05_supported_unit_count"] = sum(unit in supported for unit in units)
        row["s05_supported_concepts"] = "|".join(concepts) if concepts else "none"
        row["interpretive_label"] = (
            "|".join(concepts) if concepts else "unresolved_by_S05_vocabulary"
        )


def _catalog_motifs(
    unit_ids: Sequence[str],
    edges: Sequence[dict[str, Any]],
    resolved: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the registered catalog through its named config thresholds."""

    return _consensus_components(
        unit_ids,
        edges,
        minimum_recurrence=float(resolved["minimum_match_recurrence"]),
        minimum_causal_similarity=float(
            resolved["minimum_motif_regime_causal_similarity"]
        ),
    )


def _count_motif_eligible_edges(
    final_rows: Sequence[dict[str, Any]], resolved: dict[str, Any]
) -> int:
    """Count edges that pass the registered recurrence and motif-level gates."""

    return sum(
        float(row["equilibrium_bootstrap_recurrence"])
        >= float(resolved["minimum_match_recurrence"])
        and min(
            float(row["causal_effect_similarity_stable_or_near_floor"]),
            float(row["causal_effect_similarity_unstable"]),
        ) >= float(resolved["minimum_motif_regime_causal_similarity"])
        for row in final_rows
    )


def _validate_summary(summary: dict[str, Any]) -> None:
    if int(summary.get("stable_rows", 0)) < 1 or int(summary.get("unstable_rows", 0)) < 1:
        raise ValueError("stable and unstable rows must both be reported")
    if summary.get("match_bootstrap_group") != BOOTSTRAP_GROUP:
        raise ValueError("unit matching must bootstrap equilibrium_files")
    if summary.get("flux_residualized_matching") is not True:
        raise ValueError("unit matching must include the flux-residualized control")
    if summary.get("causal_effect_validity") != HIDDEN_INTERVENTION_VALIDITY:
        raise ValueError("hidden causal effects require the off-manifold validity tag")
    counts = summary.get("cohort_member_counts", {})
    if int(counts.get("stored_validation_top_10", 0)) < 1:
        raise ValueError("the registered top cohort is absent")
    if int(counts.get("stored_validation_ranks_11_50", 0)) < 1 or int(
        counts.get("stored_validation_ranks_51_100", 0)
    ) < 1:
        raise ValueError("lower-ranked registered comparisons are absent")


def _apply_regime_causal_gate(
    rows: list[dict[str, Any]],
    *,
    effects: Sequence[np.ndarray],
    stable: np.ndarray,
    member_ids: Sequence[str],
    minimum_similarity: float,
) -> None:
    """Publish and gate signed ablation agreement separately by output regime."""

    member_index = {member_id: index for index, member_id in enumerate(member_ids)}

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        return float(left @ right / denominator) if denominator > 1e-12 else 0.0

    for row in rows:
        left_member = member_index[str(row["left_member_id"])]
        right_member = member_index[str(row["right_member_id"])]
        left_unit = int(str(row["left_unit_id"]).rsplit("u", 1)[1])
        right_unit = int(str(row["right_unit_id"]).rsplit("u", 1)[1])
        stable_similarity = cosine(
            effects[left_member][stable, left_unit],
            effects[right_member][stable, right_unit],
        )
        unstable_similarity = cosine(
            effects[left_member][~stable, left_unit],
            effects[right_member][~stable, right_unit],
        )
        row["causal_effect_similarity_stable_or_near_floor"] = stable_similarity
        row["causal_effect_similarity_unstable"] = unstable_similarity
        # Match the archived float32 member-signature representation so this
        # published magnitude column is reproducible from member_signatures.h5.
        left_values = np.asarray(effects[left_member][:, left_unit], dtype=np.float32)
        right_values = np.asarray(effects[right_member][:, right_unit], dtype=np.float32)

        def rms_ratio(mask: np.ndarray) -> float:
            left_rms = float(np.sqrt(np.mean(np.square(left_values[mask]))))
            right_rms = float(np.sqrt(np.mean(np.square(right_values[mask]))))
            smaller_rms = min(left_rms, right_rms)
            if smaller_rms <= 1e-12:
                raise ValueError(
                    "matched causal effect has near-zero RMS in a regime for "
                    f"{row['left_unit_id']} and {row['right_unit_id']}"
                )
            return max(left_rms, right_rms) / smaller_rms

        row["causal_effect_rms_magnitude_ratio"] = rms_ratio(
            np.ones(len(stable), dtype=bool)
        )
        row["causal_effect_rms_magnitude_ratio_stable_or_near_floor"] = rms_ratio(stable)
        row["causal_effect_rms_magnitude_ratio_unstable"] = rms_ratio(~stable)
        row["causal_regime_gate"] = bool(
            stable_similarity >= minimum_similarity
            and unstable_similarity >= minimum_similarity
        )
        preliminary = row["consensus_gate"] is True or str(row["consensus_gate"]).lower() == "true"
        row["pre_regime_consensus_gate"] = preliminary
        row["consensus_gate"] = bool(preliminary and row["causal_regime_gate"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/xai/S10_cross_model.json"))
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
    for value, name in (
        (args.dataset, "dataset"),
        (args.checkpoint, "checkpoint"),
        (args.published_dir, "published_dir"),
    ):
        if value is not None:
            resolved[name] = str(value)
    return resolved


def _source_hashes() -> dict[str, str]:
    return {
        "runner": sha256_file(Path(__file__)),
        "library": sha256_file(Path(__file__).parents[1] / "itg_nn/xai/cross_model.py"),
        "config": sha256_file(Path(__file__).parents[1] / "configs/xai/S10_cross_model.json"),
    }


def _validate_resume_manifest(
    manifest: dict[str, Any],
    *,
    resolved: dict[str, Any],
    output_dir: Path,
    dataset: Path,
    checkpoint: Path,
) -> None:
    reproduction_config = manifest.get("postprocessing", {}).get(
        "reproduction_config", manifest.get("config")
    )
    if reproduction_config != resolved:
        raise RuntimeError("resume config or S10 source hashes differ from the completed run")
    if manifest.get("dataset", {}).get("sha256") != sha256_file(dataset):
        raise RuntimeError("resume dataset fingerprint differs from the completed run")
    if manifest.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint):
        raise RuntimeError("resume checkpoint fingerprint differs from the completed run")
    for name, expected in manifest.get("output_hashes", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"resume output hash mismatch: {name}")


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def _h5_take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(rows, return_inverse=True)
    return dataset[unique][inverse]


def _channel_scales(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["channel"]))
    values = np.asarray([float(row["iqr"]) for row in rows], dtype=np.float64)
    if values.shape != (7,) or np.any(values <= 0):
        raise ValueError("S01 must provide seven positive channel IQRs")
    return values


def _csv_text(rows: Sequence[dict[str, Any]]) -> str:
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


def _correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = left - left.mean(0, keepdims=True)
    y = right - right.mean(0, keepdims=True)
    denominator = np.linalg.norm(x, axis=0)[:, None] * np.linalg.norm(y, axis=0)[None, :]
    return np.divide(x.T @ y, denominator, out=np.zeros((x.shape[1], y.shape[1])), where=denominator > 1e-12)


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(0, keepdims=True)
    scale = centered.std(0)
    keep = scale > 1e-10
    return centered[:, keep] / scale[keep] if np.any(keep) else np.zeros((len(values), 1))


def _probe_representation(values: np.ndarray) -> np.ndarray:
    """Flatten and feature-standardize one representation on the shared probe."""

    return _standardize(np.asarray(values).reshape(len(values), -1))


def _regime_mask(target: np.ndarray, stable_threshold: float) -> np.ndarray:
    """Assign the native floor and threshold rows to the stable regime."""

    return np.asarray(target) <= stable_threshold


def _panel_covariates(
    target: np.ndarray, a_over_lt: np.ndarray, a_over_ln: np.ndarray
) -> np.ndarray:
    """Flux and both drives that the matching control must remove."""

    return np.column_stack((target, a_over_lt, a_over_ln))


def _concept_profile(selectivity: np.ndarray) -> np.ndarray:
    """Retain peak absolute and signed mean selectivity for each concept."""

    return np.concatenate(
        (np.max(np.abs(selectivity), axis=0), np.mean(selectivity, axis=0))
    )


def _attribution_profile(gradients: np.ndarray, stable: np.ndarray) -> np.ndarray:
    """Signed then absolute channel means, separately by output regime."""

    return np.concatenate(
        [gradients[mask].mean(0) for mask in (stable, ~stable)]
        + [np.abs(gradients[mask]).mean(0) for mask in (stable, ~stable)]
    )


def _outlier_trimmed_cka(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    """Recompute CKA after dropping the 5% largest joint representation norms."""

    joint_norm = np.linalg.norm(left, axis=1) + np.linalg.norm(right, axis=1)
    keep = joint_norm <= np.quantile(joint_norm, 0.95)
    return linear_cka(left[keep], right[keep]), int(np.count_nonzero(keep))


def _density_signature(density: np.ndarray) -> np.ndarray:
    centered = density - density.mean(axis=2, keepdims=True)
    power = np.square(np.abs(np.fft.rfft(centered, axis=2)))
    power /= np.maximum(power.sum(axis=2, keepdims=True), 1e-30)
    bands = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 49))
    spectral = np.stack([power[:, :, start:stop].sum(2).mean(0) for start, stop in bands], axis=1)
    maxima = density.max(2)
    support = np.mean(density >= 0.5 * maxima[:, :, None], axis=(0, 2))[:, None]
    active = np.mean(density > 1e-8, axis=(0, 2))[:, None]
    coefficient = (density.std(axis=(0, 2)) / np.maximum(density.mean(axis=(0, 2)), 1e-12))[:, None]
    return np.column_stack((spectral, support, active, coefficient))


def _effect_signature(effects: np.ndarray, stable: np.ndarray) -> np.ndarray:
    result = []
    for unit in range(effects.shape[1]):
        values = effects[:, unit]
        result.append([
            values[stable].mean(), values[~stable].mean(),
            np.sqrt(np.mean(np.square(values[stable]))),
            np.sqrt(np.mean(np.square(values[~stable]))),
        ])
    return np.asarray(result)


def _fixed_member_profile(values: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(values, [0, 0.25, 0.5, 0.75, 1], axis=0)
    return quantiles.T.ravel()


def _stratified_indices(stable: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stable_rows = np.flatnonzero(stable)
    unstable_rows = np.flatnonzero(~stable)
    stable_count = min(len(stable_rows), max(1, round(count * len(stable_rows) / len(stable))))
    unstable_count = min(len(unstable_rows), count - stable_count)
    chosen = np.concatenate((
        rng.choice(stable_rows, stable_count, replace=False),
        rng.choice(unstable_rows, unstable_count, replace=False),
    ))
    return np.sort(chosen)


def _member_attribution(
    member: InvariantMember,
    geometry: torch.Tensor,
    a_lt: torch.Tensor,
    a_ln: torch.Tensor,
    scales: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(geometry), batch_size):
        stop = min(start + batch_size, len(geometry))
        x = geometry[start:stop].to(device).detach().requires_grad_(True)
        output = member.invariant(x, a_lt[start:stop].to(device), a_ln[start:stop].to(device))
        gradient = torch.autograd.grad(output.sum(), x)[0]
        chunks.append(gradient.detach().cpu().numpy().mean(axis=1) * scales[None, :])
    return np.concatenate(chunks).astype(np.float64)


def _plot_cka(path: Path, rows: Sequence[dict[str, Any]], member_count: int) -> None:
    layers = list(dict.fromkeys(str(row["layer_name"]) for row in rows))
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), squeeze=False)
    for axis, layer in zip(axes.flat, layers):
        matrix = np.eye(member_count)
        for row in rows:
            if row["layer_name"] == layer:
                i, j = int(row["left_rank"]) - 1, int(row["right_rank"]) - 1
                matrix[i, j] = matrix[j, i] = float(row["cka"])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set_title(layer)
        axis.set_xlabel("stored-validation rank")
        axis.set_ylabel("stored-validation rank")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="linear CKA")
    figure.suptitle("Cross-member representation similarity")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolved = _resolve(config, args)
    resolved["source_hashes"] = _source_hashes()
    resolved["s05_unit_motifs_sha256"] = sha256_file(resolved["s05_unit_motifs"])
    set_deterministic_seed(int(resolved["seed"]))
    dataset, checkpoint = Path(resolved["dataset"]), Path(resolved["checkpoint"])
    output_dir = args.output_dir.resolve() if args.output_dir else (Path("output/xai/S10") / resolved["run_id"]).resolve()
    if args.resume and (output_dir / "manifest.json").is_file():
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        _validate_resume_manifest(
            manifest, resolved=resolved, output_dir=output_dir,
            dataset=dataset, checkpoint=checkpoint,
        )
        print("S10 resume validated config, sources, inputs, and output hashes", flush=True)
        return output_dir
    artifacts = RunArtifacts(output_dir)
    cohorts = json.loads(Path(resolved["cohorts"]).read_text(encoding="utf-8"))
    registered = np.asarray(cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64)
    row_ids = registered[: int(resolved["panel_varied_rows"])]
    panel = load_hdf5_rows(dataset, row_ids, gradient_set="varied", include_targets=True)
    if panel.actual_log_heat_flux is None:
        raise RuntimeError("native varied-gradient targets were not loaded")
    target = panel.actual_log_heat_flux.numpy().astype(np.float64)
    stable = _regime_mask(target, float(resolved["stable_threshold_log_Q"]))
    if not stable.any() or stable.all():
        raise RuntimeError("both output regimes are required")
    with h5py.File(dataset, "r") as handle:
        groups = _decode(_h5_take(handle["equilibrium_files"], row_ids))
    if len(np.unique(groups)) != len(groups):
        raise RuntimeError("S01 panel should contain one selected tube per equilibrium")

    scales = _channel_scales(Path(resolved["channel_scales"]))
    concepts = physics_concept_traces(panel.geometry.numpy(), channel_scales=scales).values.mean(axis=2).T
    concept_values = concepts.T
    cka_indices = _stratified_indices(stable, int(resolved["cka_rows"]), int(resolved["seed"]) + 1)
    attribution_indices = _stratified_indices(stable, int(resolved["attribution_rows"]), int(resolved["seed"]) + 2)
    ensemble = load_ensemble(checkpoint, device=resolved["device"])
    all_ids = tuple(cohorts["member_cohorts"]["all_100"])
    member_ids = all_ids[: int(resolved["members"])]
    index = {member_id: position for position, member_id in enumerate(ensemble.member_ids)}

    bottlenecks: list[np.ndarray] = []
    effects: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    effect_signatures: list[np.ndarray] = []
    auxiliary: list[np.ndarray | None] = []
    concept_profiles: list[np.ndarray] = []
    causal_profiles: list[np.ndarray] = []
    attribution_profiles: list[np.ndarray] = []
    gradient_rows: list[np.ndarray] = []
    layer_representations: list[list[np.ndarray]] = [[] for _ in range(6)]
    widths: list[int] = []
    architecture_rows: list[dict[str, Any]] = []

    for member_rank, member_id in enumerate(member_ids, start=1):
        member = InvariantMember(ensemble.models[index[member_id]])
        density_chunks, bottleneck_chunks, prediction_chunks = [], [], []
        with torch.inference_mode():
            for start in range(0, len(row_ids), int(resolved["batch_size"])):
                stop = min(start + int(resolved["batch_size"]), len(row_ids))
                x = panel.geometry[start:stop].to(ensemble.device)
                rho = member.equivariant_density(x)
                bottleneck_chunks.append(rho.mean(-1).cpu().numpy())
                prediction_chunks.append(member.head(
                    rho.mean(-1), panel.a_over_lt[start:stop].to(ensemble.device), panel.a_over_ln[start:stop].to(ensemble.device)
                ).cpu().numpy())
                if member_rank <= int(resolved["top_members"]):
                    density_chunks.append(rho.cpu().numpy())
        bottleneck = np.concatenate(bottleneck_chunks).astype(np.float64)
        prediction = np.concatenate(prediction_chunks).astype(np.float64)
        drives = np.column_stack((panel.a_over_lt.numpy(), panel.a_over_ln.numpy()))

        def head(values: np.ndarray, current=member, current_drives=drives) -> np.ndarray:
            tensor = torch.as_tensor(values, dtype=torch.float32, device=ensemble.device)
            with torch.inference_mode():
                return current.head(
                    tensor,
                    torch.as_tensor(current_drives[:, 0], dtype=torch.float32, device=ensemble.device),
                    torch.as_tensor(current_drives[:, 1], dtype=torch.float32, device=ensemble.device),
                ).cpu().numpy()

        effect = mean_replacement_effects(bottleneck, head)
        effect_signature = _effect_signature(effect, stable)
        selectivity = _correlation(bottleneck, concept_values)
        profile = _concept_profile(selectivity)
        if density_chunks:
            density = np.concatenate(density_chunks).astype(np.float64)
            aux = np.column_stack((selectivity, _density_signature(density)))
        else:
            aux = None

        cka_geometry = panel.geometry[cka_indices].to(ensemble.device)
        with torch.inference_mode():
            maps = invariant_layer_maps(member, cka_geometry)
        for layer, values in enumerate(maps):
            layer_representations[layer].append(
                _probe_representation(values.cpu().numpy())
            )
        layer_representations[5].append(
            _probe_representation(bottleneck[cka_indices])
        )
        gradients = _member_attribution(
            member,
            panel.geometry[attribution_indices], panel.a_over_lt[attribution_indices], panel.a_over_ln[attribution_indices],
            scales, int(resolved["batch_size"]), ensemble.device,
        )
        gradient_stable = stable[attribution_indices]
        attribution_profile = _attribution_profile(gradients, gradient_stable)

        width = bottleneck.shape[1]
        kernels = [int(layer.kernel_size[0]) for layer in member.model.conv_layers]
        bottlenecks.append(bottleneck)
        effects.append(effect)
        predictions.append(prediction)
        effect_signatures.append(effect_signature)
        auxiliary.append(aux)
        concept_profiles.append(profile)
        causal_profiles.append(_fixed_member_profile(effect_signature))
        attribution_profiles.append(attribution_profile)
        gradient_rows.append(gradients)
        widths.append(width)
        architecture_rows.append({
            "member_id": member_id, "stored_validation_rank": member_rank,
            "cohort": _member_cohort(member_rank), "bottleneck_width": width,
            "narrow_bottleneck_C_le_11": width <= 11, "kernel_sizes": "|".join(map(str, kernels)),
            "canonical_function": CANONICAL_FUNCTION, "estimand": NATIVE_ESTIMAND,
        })
        print(f"S10 member {member_rank}/{len(member_ids)}: {member_id}", flush=True)

    unit_match_rows: list[dict[str, Any]] = []
    accepted_edges: list[dict[str, Any]] = []
    top_count = min(int(resolved["top_members"]), len(member_ids))
    covariates = _panel_covariates(
        target, panel.a_over_lt.numpy(), panel.a_over_ln.numpy()
    )
    for left in range(top_count):
        for right in range(left + 1, top_count):
            assert auxiliary[left] is not None and auxiliary[right] is not None
            score, pieces = functional_similarity(
                bottlenecks[left], bottlenecks[right], covariates=covariates,
                left_auxiliary=auxiliary[left], right_auxiliary=auxiliary[right],
                left_effects=effect_signatures[left], right_effects=effect_signatures[right],
                component_weights=resolved["signature_component_weights"],
            )
            recurrence = grouped_bootstrap_match_recurrence(
                bottlenecks[left], bottlenecks[right], groups=groups, covariates=covariates,
                left_auxiliary=auxiliary[left], right_auxiliary=auxiliary[right],
                left_effects=effect_signatures[left], right_effects=effect_signatures[right],
                component_weights=resolved["signature_component_weights"],
                minimum_similarity=float(resolved["minimum_match_similarity"]),
                replicates=int(resolved["bootstrap_replicates"]), seed=int(resolved["seed"]) + 1009 * left + right,
            )
            for left_unit, right_unit, value in match_units(score, minimum_similarity=float(resolved["minimum_match_similarity"])):
                left_id = f"{member_ids[left]}:u{left_unit:03d}"
                right_id = f"{member_ids[right]}:u{right_unit:03d}"
                row = {
                    "left_member_id": member_ids[left], "right_member_id": member_ids[right],
                    "left_unit_id": left_id, "right_unit_id": right_id,
                    "functional_similarity": value,
                    "activation_raw_similarity": pieces["activation_raw"][left_unit, right_unit],
                    "activation_flux_residual_similarity": pieces["activation_residual"][left_unit, right_unit],
                    "concept_density_similarity": pieces["auxiliary"][left_unit, right_unit],
                    "causal_effect_similarity": pieces["causal_effect"][left_unit, right_unit],
                    "equilibrium_bootstrap_recurrence": recurrence[left_unit, right_unit],
                    "bootstrap_group": BOOTSTRAP_GROUP,
                    "causal_effect_validity": HIDDEN_INTERVENTION_VALIDITY,
                    "consensus_gate": bool(
                        recurrence[left_unit, right_unit] >= float(resolved["minimum_match_recurrence"])
                        and pieces["causal_effect"][left_unit, right_unit] >= float(resolved["minimum_causal_similarity"])
                        and pieces["activation_residual"][left_unit, right_unit] >= float(resolved["minimum_flux_residual_similarity"])
                    ),
                }
                unit_match_rows.append(row)
            print(
                f"S10 matching pair {left + 1}-{right + 1} of {top_count}",
                flush=True,
            )

    _apply_regime_causal_gate(
        unit_match_rows,
        effects=effects,
        stable=stable,
        member_ids=member_ids,
        minimum_similarity=float(resolved["minimum_regime_causal_similarity"]),
    )
    accepted_edges = [
        {
            "left": row["left_unit_id"], "right": row["right_unit_id"],
            "recurrence": row["equilibrium_bootstrap_recurrence"],
            "causal_similarity": min(
                row["causal_effect_similarity_stable_or_near_floor"],
                row["causal_effect_similarity_unstable"],
            ),
        }
        for row in unit_match_rows if row["consensus_gate"]
    ]
    preliminary_edges = [
        {
            "left": row["left_unit_id"], "right": row["right_unit_id"],
            "recurrence": row["equilibrium_bootstrap_recurrence"],
            "causal_similarity": row["causal_effect_similarity"],
        }
        for row in unit_match_rows if row["pre_regime_consensus_gate"]
    ]
    unit_ids = [f"{member_ids[m]}:u{unit:03d}" for m in range(top_count) for unit in range(widths[m])]
    preliminary_motifs = _consensus_components(
        unit_ids,
        preliminary_edges,
        minimum_recurrence=float(resolved["minimum_match_recurrence"]),
        minimum_causal_similarity=float(resolved["minimum_causal_similarity"]),
    )
    motifs = _catalog_motifs(unit_ids, accepted_edges, resolved)
    motif_rows = [
        {"motif_id": f"motif_{index:03d}", **motif, "unit_ids": "|".join(motif["unit_ids"]),
         "definition": "functional_signature_and_off_manifold_mean_ablation_agree"}
        for index, motif in enumerate(motifs, start=1)
    ]
    _annotate_motifs(
        motif_rows, s05_unit_motifs=Path(resolved["s05_unit_motifs"])
    )
    motif_sensitivity_rows = []
    for threshold in resolved["motif_regime_causal_similarity_sweep"]:
        sensitivity_motifs = _consensus_components(
            unit_ids,
            accepted_edges,
            minimum_recurrence=float(resolved["minimum_match_recurrence"]),
            minimum_causal_similarity=float(threshold),
        )
        eligible = sum(
            float(edge["recurrence"]) >= float(resolved["minimum_match_recurrence"])
            and float(edge["causal_similarity"]) >= float(threshold)
            for edge in accepted_edges
        )
        motif_sensitivity_rows.append({
            "minimum_regime_causal_similarity": float(threshold),
            "eligible_edges": eligible,
            "catalog_edges_after_one_unit_per_member": sum(
                int(motif["edge_count"]) for motif in sensitivity_motifs
            ),
            "motif_count": len(sensitivity_motifs),
            "motif_member_counts": "|".join(
                str(motif["member_count"]) for motif in sensitivity_motifs
            ),
        })

    cka_rows: list[dict[str, Any]] = []
    cka_names = [f"canonical_atrous_layer_{index}" for index in range(1, 6)] + ["invariant_bottleneck"]
    for layer, representations in enumerate(layer_representations):
        point, lower, upper = grouped_bootstrap_cka(
            representations, groups=groups[cka_indices],
            replicates=int(resolved["cka_bootstrap_replicates"]), seed=int(resolved["seed"]) + 7919 * layer,
        )
        for left in range(len(member_ids)):
            for right in range(left + 1, len(member_ids)):
                # Pairwise 5% high-norm removal checks outlier sensitivity.
                trimmed, _ = _outlier_trimmed_cka(
                    representations[left], representations[right]
                )
                cka_rows.append({
                    "layer_index": layer, "layer_name": cka_names[layer],
                    "representation_kind": "flattened_spatial_activation" if layer < 5 else "position_mean_bottleneck",
                    "left_member_id": member_ids[left], "right_member_id": member_ids[right],
                    "left_rank": left + 1, "right_rank": right + 1,
                    "cka": point[left, right], "cka_grouped_ci95_lower": lower[left, right],
                    "cka_grouped_ci95_upper": upper[left, right], "outlier_trimmed_cka": trimmed,
                    "bootstrap_group": BOOTSTRAP_GROUP, "probe_rows": len(cka_indices),
                })
        print(f"S10 CKA layer {layer + 1}/6", flush=True)

    distance = member_distance_matrix((
        np.asarray(predictions), np.asarray(attribution_profiles),
        np.asarray(causal_profiles), np.asarray(concept_profiles),
    ))
    try:
        from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
        from scipy.spatial.distance import squareform
        from scipy.stats import spearmanr
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "S10 member clustering requires the project XAI dependencies; "
            "run bash scripts/setup_xai_env.sh"
        ) from error
    condensed = squareform(distance, checks=False)
    hierarchy = linkage(condensed, method="average") if len(member_ids) > 1 else np.zeros((0, 4))
    cluster_count = min(int(resolved["member_clusters"]), len(member_ids))
    cluster = fcluster(hierarchy, cluster_count, criterion="maxclust") if len(member_ids) > 1 else np.ones(1, dtype=int)
    member_rows = []
    for rank, (member_id, label, width) in enumerate(zip(member_ids, cluster, widths), start=1):
        member_rows.append({
            "member_id": member_id, "stored_validation_rank": rank, "cohort": _member_cohort(rank),
            "cluster": int(label), "bottleneck_width": width, "narrow_bottleneck_C_le_11": width <= 11,
            "prediction_profile_source": NATIVE_ESTIMAND,
            "input_attribution": "signed robust-IQR-scaled canonical gradient",
            "causal_signature_validity": HIDDEN_INTERVENTION_VALIDITY,
        })
    distance_rows = [
        {"left_member_id": member_ids[i], "right_member_id": member_ids[j], "distance": distance[i, j]}
        for i in range(len(member_ids)) for j in range(i + 1, len(member_ids))
    ]
    cohort_rows = []
    for cohort in ("stored_validation_top_10", "stored_validation_ranks_11_50", "stored_validation_ranks_51_100"):
        selected = [i for i, row in enumerate(member_rows) if row["cohort"] == cohort]
        if not selected:
            continue
        cohort_row = {
            "cohort": cohort, "member_count": len(selected),
            "median_bottleneck_width": float(np.median(np.asarray(widths)[selected])),
            "narrow_fraction": float(np.mean(np.asarray(widths)[selected] <= 11)),
            "median_within_cohort_distance": float(np.median([
                distance[i, j] for offset, i in enumerate(selected) for j in selected[offset + 1:]
            ])) if len(selected) > 1 else 0.0,
            "clusters_present": "|".join(map(str, sorted(set(int(cluster[i]) for i in selected)))),
        }
        selected_members = {member_ids[index] for index in selected}
        for layer_name in cka_names:
            cohort_row[f"median_cka__{layer_name}"] = float(np.median([
                row["cka"] for row in cka_rows
                if row["layer_name"] == layer_name
                and row["left_member_id"] in selected_members
                and row["right_member_id"] in selected_members
            ]))
        cohort_rows.append(cohort_row)

    final_rows = [row for row in unit_match_rows if row["consensus_gate"]]
    final_effect_ratios = np.asarray(
        [row["causal_effect_rms_magnitude_ratio"] for row in final_rows],
        dtype=np.float64,
    )
    final_stable_effect_ratios = np.asarray([
        row["causal_effect_rms_magnitude_ratio_stable_or_near_floor"]
        for row in final_rows
    ], dtype=np.float64)
    final_unstable_effect_ratios = np.asarray([
        row["causal_effect_rms_magnitude_ratio_unstable"] for row in final_rows
    ], dtype=np.float64)
    medoid = int(np.argmin(distance.mean(axis=1)))
    rank_correlation, rank_p_value = spearmanr(
        np.arange(1, len(member_ids) + 1), distance[medoid]
    )
    narrow = np.asarray(widths) <= 11
    wide = ~narrow
    narrow_wide = distance[np.ix_(narrow, wide)].ravel()
    wide_wide_matrix = distance[np.ix_(wide, wide)]
    wide_wide = wide_wide_matrix[np.triu_indices(int(wide.sum()), 1)]
    cka_median_by_layer = {}
    cka_trim_change_by_layer = {}
    for layer_name in cka_names:
        layer_rows = [row for row in cka_rows if row["layer_name"] == layer_name]
        values = np.asarray([row["cka"] for row in layer_rows], dtype=np.float64)
        trimmed = np.asarray(
            [row["outlier_trimmed_cka"] for row in layer_rows], dtype=np.float64
        )
        cka_median_by_layer[layer_name] = float(np.median(values))
        cka_trim_change_by_layer[layer_name] = float(np.median(np.abs(values - trimmed)))
    motif_eligible_edges = _count_motif_eligible_edges(final_rows, resolved)

    summary = {
        "run_id": resolved["run_id"], "estimand": NATIVE_ESTIMAND,
        "canonical_function": CANONICAL_FUNCTION, "members": len(member_ids), "panel_rows": len(row_ids),
        "stable_rows": int(stable.sum()), "unstable_rows": int((~stable).sum()),
        "matched_pairs": len(unit_match_rows), "consensus_edges": len(accepted_edges),
        "consensus_motifs": len(motif_rows), "match_bootstrap_group": BOOTSTRAP_GROUP,
        "flux_residualized_matching": True, "causal_effect_validity": HIDDEN_INTERVENTION_VALIDITY,
        "cka_pair_layer_rows": len(cka_rows), "cka_bootstrap_group": BOOTSTRAP_GROUP,
        "cohort_member_counts": {cohort: sum(row["cohort"] == cohort for row in member_rows) for cohort in (
            "stored_validation_top_10", "stored_validation_ranks_11_50", "stored_validation_ranks_51_100"
        )},
        "narrow_member_count": int(np.count_nonzero(np.asarray(widths) <= 11)),
        "pre_regime_consensus_edges": len(preliminary_edges),
        "pre_regime_consensus_motifs": len(preliminary_motifs),
        "preliminary_edges_rejected_by_regime_causal_gate": len(preliminary_edges) - len(accepted_edges),
        "regime_causal_gate": (
            f"cosine_similarity>={resolved['minimum_regime_causal_similarity']} "
            f"separately on {int(stable.sum())} stable_or_near_floor and "
            f"{int((~stable).sum())} unstable signed mean-replacement effects"
        ),
        "regime_causal_gate_correction": (
            "preliminary four-summary causal gate admitted opposing within-regime "
            "effects; corrected before reporting"
        ),
        "motif_eligible_edges": motif_eligible_edges,
        "motif_minimum_recurrence": float(resolved["minimum_match_recurrence"]),
        "motif_minimum_regime_causal_similarity": float(
            resolved["minimum_motif_regime_causal_similarity"]
        ),
        "final_edge_flux_residual_similarity_median": float(np.median([
            row["activation_flux_residual_similarity"] for row in final_rows
        ])),
        "final_edge_recurrence_median": float(np.median([
            row["equilibrium_bootstrap_recurrence"] for row in final_rows
        ])),
        "final_edge_stable_causal_similarity_median": float(np.median([
            row["causal_effect_similarity_stable_or_near_floor"] for row in final_rows
        ])),
        "final_edge_unstable_causal_similarity_median": float(np.median([
            row["causal_effect_similarity_unstable"] for row in final_rows
        ])),
        "final_edge_causal_effect_rms_ratio_median": float(
            np.median(final_effect_ratios)
        ),
        "final_edge_causal_effect_rms_ratio_p90": float(
            np.quantile(final_effect_ratios, 0.9)
        ),
        "final_edge_causal_effect_rms_ratio_maximum": float(
            np.max(final_effect_ratios)
        ),
        "final_edge_stable_causal_effect_rms_ratio_median": float(
            np.median(final_stable_effect_ratios)
        ),
        "final_edge_stable_causal_effect_rms_ratio_p90": float(
            np.quantile(final_stable_effect_ratios, 0.9)
        ),
        "final_edge_stable_causal_effect_rms_ratio_maximum": float(
            np.max(final_stable_effect_ratios)
        ),
        "final_edge_unstable_causal_effect_rms_ratio_median": float(
            np.median(final_unstable_effect_ratios)
        ),
        "final_edge_unstable_causal_effect_rms_ratio_p90": float(
            np.quantile(final_unstable_effect_ratios, 0.9)
        ),
        "final_edge_unstable_causal_effect_rms_ratio_maximum": float(
            np.max(final_unstable_effect_ratios)
        ),
        "maximum_motif_member_count": max(
            (row["member_count"] for row in motif_rows), default=0
        ),
        "motifs_with_at_least_four_members": sum(
            int(row["member_count"]) >= 4 for row in motif_rows
        ),
        "s05_named_consensus_motifs": sum(
            int(row["s05_supported_unit_count"]) > 0 for row in motif_rows
        ),
        "cka_median_by_layer": cka_median_by_layer,
        "cka_median_absolute_outlier_trim_change_by_layer": cka_trim_change_by_layer,
        "member_cluster_sizes": {
            str(label): int(np.count_nonzero(cluster == label))
            for label in sorted(set(int(value) for value in cluster))
        },
        "medoid_stored_validation_rank": medoid + 1,
        "rank_vs_medoid_distance_spearman": float(rank_correlation),
        "rank_vs_medoid_distance_p_value": float(rank_p_value),
        "narrow_wide_distance_median": float(np.median(narrow_wide)),
        "wide_wide_distance_median": float(np.median(wide_wide)),
    }
    if resolved["mode"] == "production":
        _validate_summary(summary)

    artifacts.write_text("unit_matches.csv", _csv_text(unit_match_rows))
    artifacts.write_text("motif_catalog.csv", _csv_text(motif_rows))
    artifacts.write_text(
        "motif_threshold_sensitivity.csv", _csv_text(motif_sensitivity_rows)
    )
    artifacts.write_text("cka.csv", _csv_text(cka_rows))
    artifacts.write_text("member_clusters.csv", _csv_text(member_rows))
    artifacts.write_text("member_distances.csv", _csv_text(distance_rows))
    artifacts.write_text("cohort_comparison.csv", _csv_text(cohort_rows))
    artifacts.write_text("architecture.csv", _csv_text(architecture_rows))
    artifacts.write_json("summary.json", summary)

    max_width = max(widths)
    padded_bottleneck = np.full((len(member_ids), len(row_ids), max_width), np.nan, dtype=np.float32)
    padded_effect = np.full_like(padded_bottleneck, np.nan)
    unit_present = np.zeros((len(member_ids), max_width), dtype=bool)
    for member, width in enumerate(widths):
        padded_bottleneck[member, :, :width] = bottlenecks[member]
        padded_effect[member, :, :width] = effects[member]
        unit_present[member, :width] = True
    artifacts.write_hdf5(
        "member_signatures.h5",
        {
            "member_id": np.asarray(member_ids, dtype="S96"), "row_id": row_ids,
            "equilibrium_file": np.asarray(groups, dtype="S240"), "actual_log_Q": target.astype(np.float32),
            "stable_or_near_floor": stable, "prediction": np.asarray(predictions, dtype=np.float32),
            "bottleneck": padded_bottleneck, "mean_replacement_signed_effect": padded_effect,
            "unit_present": unit_present, "attribution_row_id": row_ids[attribution_indices],
            "scaled_signed_input_gradient": np.asarray(gradient_rows, dtype=np.float32),
        },
        axes={
            "member_id": ("member",), "row_id": ("sample",), "equilibrium_file": ("sample",),
            "actual_log_Q": ("sample",), "stable_or_near_floor": ("sample",),
            "prediction": ("member", "sample"), "bottleneck": ("member", "sample", "unit"),
            "mean_replacement_signed_effect": ("member", "sample", "unit"),
            "unit_present": ("member", "unit"), "attribution_row_id": ("attribution_sample",),
            "scaled_signed_input_gradient": ("member", "attribution_sample", "channel"),
        },
        attributes={
            "estimand": NATIVE_ESTIMAND, "canonical_function": CANONICAL_FUNCTION,
            "causal_effect_validity": HIDDEN_INTERVENTION_VALIDITY,
            "gradient_scaling": "S01 panel channel IQR; signed mean over position",
        }, compression="gzip",
    )

    cka_figure = output_dir / "cross_model_cka.png"
    _plot_cka(cka_figure, cka_rows, len(member_ids))
    artifacts.register_existing(cka_figure.name)
    dendrogram_figure = output_dir / "member_dendrogram.png"
    figure, axis = plt.subplots(figsize=(12, 5))
    if len(member_ids) > 1:
        dendrogram(hierarchy, labels=[str(index) for index in range(1, len(member_ids) + 1)], ax=axis, leaf_font_size=6)
    axis.set_xlabel("stored-validation rank")
    axis.set_ylabel("multi-evidence distance")
    axis.set_title("Member clusters: predictions, input gradients, causal signatures, concepts")
    figure.tight_layout()
    figure.savefig(dendrogram_figure, dpi=170)
    plt.close(figure)
    artifacts.register_existing(dendrogram_figure.name)

    published = None if args.pilot or args.no_publish else Path(resolved["published_dir"])
    manifest = artifacts.finalize(
        config=resolved, dataset=dataset, checkpoint=checkpoint, member_ids=member_ids,
        row_ids=row_ids, gradient_set="varied frozen S01 interpretation panel only",
        device=ensemble.device, repository=Path.cwd(), command=sys.argv,
        published_dir=published,
        extra_manifest={
            "postprocessing": {
                "committed_output_hashes_provenance": (
                    "output_hashes describe artifacts produced by this runner "
                    "invocation"
                ),
                "command": (
                    "regime-specific causal audit from manifest-hashed "
                    "member_signatures.h5"
                ),
                "committed_runner_reproduction": (
                    "audit, derived summary fields, cohort CKA columns, and S05 "
                    "motif annotations are folded into scripts/xai_s10_cross_model.py"
                ),
                "minimum_cosine_similarity_each_regime": float(
                    resolved["minimum_regime_causal_similarity"]
                ),
                "motif_threshold_provenance": (
                    "0.70 code literal present in first S10 commit before the "
                    "regime audit; promoted to a named config key after review"
                ),
                "reason": (
                    "preliminary four-summary causal gate admitted opposing "
                    "within-regime effects"
                ),
                "source_artifact": "member_signatures.h5",
                "reproduction_config": resolved,
                "reproduction_source_hashes": resolved["source_hashes"],
                "reproduction_artifacts_updated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "registered_wall_time_scope": (
                    "this runner invocation, including all-100 member-signature "
                    "and CKA execution"
                ),
                "stable_rows": int(stable.sum()),
                "unstable_rows": int((~stable).sum()),
            }
        },
    )
    if published is not None:
        published.mkdir(parents=True, exist_ok=True)
        for name in (
            "unit_matches.csv", "motif_catalog.csv", "cka.csv", "member_clusters.csv",
            "motif_threshold_sensitivity.csv",
            "member_distances.csv", "cohort_comparison.csv", "architecture.csv", "summary.json",
            "cross_model_cka.png", "member_dendrogram.png",
        ):
            shutil.copy2(output_dir / name, published / name)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"S10 {resolved['mode']} complete: {manifest}", flush=True)
    return output_dir


def main() -> int:
    print(run(build_parser().parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
