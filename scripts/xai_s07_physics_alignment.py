#!/usr/bin/env python3
"""Compare S05/S06 learned spatial signals with held-out GX diagnostics."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import shutil
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
from itg_nn.xai.physics_alignment import (
    CircularAlignment,
    circular_alignment,
    lag_selection_permutation_null,
    paired_native_difference,
    scalar_rank_association,
    select_balanced_case_studies,
)
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import CANONICAL_FUNCTION, InvariantMember


FUNCTIONS = ("original_f", CANONICAL_FUNCTION)
METHODS = ("ig_low_pass", "periodic_mask")
GRADIENT_SETS = ("varied", "fixed")
MODES = ("signed", "positive_contribution")
NATIVE_ESTIMAND = "native max(log Q, -2)"
OBSERVED_VALIDITY = "observed-comparison"
OFF_MANIFOLD_VALIDITY = "deliberately_off_manifold_diagnostic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xai/S07_physics_alignment.json")
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--s06b-map", type=Path)
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
    for value, key in (
        (args.dataset, "dataset"),
        (args.checkpoint, "checkpoint"),
        (args.s06b_map, "s06b_map"),
        (args.published_dir, "published_dir"),
    ):
        if value is not None:
            resolved[key] = str(value)
    return resolved


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def _strings(values: Any, width: int = 128) -> np.ndarray:
    return np.asarray([str(value).encode() for value in values], dtype=f"S{width}")


def _h5_take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return dataset[unique][inverse]


def _csv_text(rows: list[dict[str, Any]]) -> str:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _signed_lag(lag: int, grid_size: int = 96) -> int:
    return int(lag if lag <= grid_size // 2 else lag - grid_size)


def _stratified_positions(
    classes: np.ndarray, target: np.ndarray, count: int, threshold: float
) -> np.ndarray:
    if not 1 <= count <= len(target):
        raise ValueError("row count is outside the S01 panel")
    if count == len(target):
        return np.arange(len(target), dtype=np.int64)
    stable = target <= threshold
    labels = np.asarray(
        [f"{class_value}:{int(floor)}" for class_value, floor in zip(classes, stable)]
    )
    unique, sizes = np.unique(labels, return_counts=True)
    if count < len(unique):
        raise ValueError("pilot row count cannot cover class-by-stability strata")
    allocation = np.ones(len(unique), dtype=np.int64)
    remaining = count - len(unique)
    capacity = sizes - 1
    if remaining:
        ideal = remaining * capacity / capacity.sum()
        allocation += np.floor(ideal).astype(np.int64)
        order = np.argsort(-(ideal - np.floor(ideal)), kind="stable")
        allocation[order[: count - int(allocation.sum())]] += 1
    selected: list[int] = []
    for label, quota in zip(unique, allocation):
        candidates = np.flatnonzero(labels == label)
        positions = np.floor((np.arange(quota) + 0.5) * len(candidates) / quota).astype(
            int
        )
        selected.extend(candidates[positions].tolist())
    result = np.sort(np.asarray(selected, dtype=np.int64))
    if len(result) != count:
        raise RuntimeError("stratified selector returned the wrong row count")
    return result


def _load_unit_ranking(
    path: Path, member_ids: list[str], count: int
) -> tuple[list[list[int]], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_indices: list[list[int]] = []
    selected_ids: list[list[str]] = []
    for member_id in member_ids:
        member_rows = [
            row
            for row in rows
            if row["member_id"] == member_id and row["stratum"] == "overall"
        ]
        member_rows.sort(key=lambda row: (float(row["shapley_rank"]), row["unit_id"]))
        chosen = member_rows[:count]
        if len(chosen) != count:
            raise RuntimeError(f"S04 has fewer than {count} units for {member_id}")
        ids = [row["unit_id"] for row in chosen]
        indices = [int(unit_id.rsplit(":u", 1)[1]) for unit_id in ids]
        selected_indices.append(indices)
        selected_ids.append(ids)
    return selected_indices, selected_ids


def _load_panel_diagnostics(
    dataset: Path, rows: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(dataset, "r") as h5_file:
        equilibrium_files = _decode(_h5_take(h5_file["equilibrium_files"], rows))
        equilibrium_class = _h5_take(h5_file["equilibrium_class"], rows).astype(
            np.int16
        )
        for gradient_set in GRADIENT_SETS:
            group = h5_file[f"{gradient_set}_gradient_simulations"]
            result[gradient_set] = {
                "q_vs_z": _h5_take(group["Q_avgs_vs_z"], rows).astype(np.float64),
                "q": _h5_take(group["Q_avgs"], rows).astype(np.float64),
                "zonal_phi2": _h5_take(group["zonal_phi2_amplitudes"], rows).astype(
                    np.float64
                ),
                "a_over_lt": _h5_take(group["a_over_LT"], rows).astype(np.float64),
                "a_over_ln": _h5_take(group["a_over_Ln"], rows).astype(np.float64),
                "equilibrium_file": equilibrium_files,
                "equilibrium_class": equilibrium_class,
            }
    return result


def _density_and_predictions(
    models: dict[str, InvariantMember],
    member_ids: list[str],
    geometry: torch.Tensor,
    panel_data: dict[str, Any],
    selected_units: list[list[int]],
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = len(geometry)
    unit_count = len(selected_units[0])
    densities = np.empty(
        (len(member_ids), sample_count, unit_count, 96), dtype=np.float32
    )
    predictions = np.empty(
        (len(FUNCTIONS), len(member_ids), len(GRADIENT_SETS), sample_count),
        dtype=np.float32,
    )
    for member_index, member_id in enumerate(member_ids):
        model = models[member_id].to(device)
        unit_index = torch.as_tensor(selected_units[member_index], device=device)
        density_chunks: list[np.ndarray] = []
        prediction_chunks = {
            (function_index, gradient_index): []
            for function_index in range(len(FUNCTIONS))
            for gradient_index in range(len(GRADIENT_SETS))
        }
        for start in range(0, sample_count, batch_size):
            stop = min(start + batch_size, sample_count)
            x = geometry[start:stop].to(device)
            with torch.no_grad():
                rho = model.equivariant_density(x).index_select(1, unit_index)
                density_chunks.append(rho.cpu().numpy())
                for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
                    data = panel_data[gradient_set]
                    lt = data.a_over_lt[start:stop].to(device)
                    ln = data.a_over_ln[start:stop].to(device)
                    original = model.original(x, lt, ln)
                    invariant = model.invariant(x, lt, ln)
                    prediction_chunks[(0, gradient_index)].append(
                        original.cpu().numpy()
                    )
                    prediction_chunks[(1, gradient_index)].append(
                        invariant.cpu().numpy()
                    )
        densities[member_index] = np.concatenate(density_chunks, axis=0)
        for key, chunks in prediction_chunks.items():
            predictions[key[0], member_index, key[1]] = np.concatenate(chunks)
        print(f"density/prediction {member_index + 1}/{len(member_ids)} {member_id}")
    return densities, predictions


def _strata(target: np.ndarray, threshold: float) -> tuple[tuple[str, np.ndarray], ...]:
    stable = target <= threshold
    return (
        ("all", np.ones(len(target), dtype=bool)),
        ("stable_or_near_floor", stable),
        ("unstable", ~stable),
    )


def _association_bootstrap_stable(result: CircularAlignment) -> bool:
    return bool(result.rank_ci_lower > 0 or result.rank_ci_upper < 0)


def _lag_within_tolerance_recurrence(
    result: CircularAlignment, tolerance: int
) -> float:
    forward = (result.bootstrap_best_lag - result.best_lag) % 96
    backward = (result.best_lag - result.bootstrap_best_lag) % 96
    distance = np.minimum(forward, backward)
    return float(np.mean(distance <= tolerance))


def _alignment_row(
    metadata: dict[str, Any],
    result: CircularAlignment,
    *,
    sample_count: int,
    sparsity: float,
    bootstrap_replicates: int,
    minimum_recurrence: float,
    lag_tolerance: int,
) -> dict[str, Any]:
    association_stable = _association_bootstrap_stable(result)
    tolerant_recurrence = _lag_within_tolerance_recurrence(result, lag_tolerance)
    return {
        **metadata,
        "sample_count": sample_count,
        "mode": result.mode,
        "best_lag": _signed_lag(result.best_lag),
        "best_lag_index_0_to_95": result.best_lag,
        "circular_spearman": result.rank_correlation,
        "circular_spearman_ci95_lower": result.rank_ci_lower,
        "circular_spearman_ci95_upper": result.rank_ci_upper,
        "best_lag_bootstrap_recurrence": result.lag_recurrence,
        "best_lag_within_tolerance_recurrence": tolerant_recurrence,
        "lag_stability_tolerance_positions": lag_tolerance,
        "lag_bootstrap_stable": tolerant_recurrence >= minimum_recurrence,
        "overlap_at_fixed_sparsity": result.overlap,
        "overlap_chance_baseline": result.overlap_chance,
        "overlap_enrichment": result.overlap_enrichment,
        "overlap_orientation": result.overlap_orientation,
        "alignment_sparsity": sparsity,
        "association_bootstrap_stable": association_stable,
        "bootstrap_stable": association_stable,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_unit": result.bootstrap_group,
        "lag_selection": "max_abs_mean_within_tube_spearman_over_96_lags",
        "interval_scope": "fixed_full_panel_selected_lag",
        "spatial_distinction": (
            "learned prediction signal compared with held-out physical Q_avgs_vs_z; "
            "association does not make the attribution a physical heat-flux profile"
        ),
    }


def _position_source(values: np.ndarray, method: str, mode: str) -> np.ndarray:
    if mode == "signed":
        if method != "ig_low_pass":
            raise ValueError("the periodic mask is magnitude-only, not signed")
        return values.sum(axis=1)
    return np.maximum(values, 0.0).sum(axis=1)


def _top_mass_fraction(values: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    magnitude = np.abs(np.asarray(values, dtype=np.float64)).reshape(len(values), -1)
    count = max(1, int(np.ceil(magnitude.shape[1] * fraction)))
    top = np.partition(magnitude, magnitude.shape[1] - count, axis=1)[:, -count:].sum(
        axis=1
    )
    total = magnitude.sum(axis=1)
    return np.divide(top, total, out=np.zeros_like(top), where=total > 0)


def _publish(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, destination / path.name)


def _resume_if_valid(
    output_dir: Path, config: dict[str, Any], dataset: Path, checkpoint: Path
) -> Path | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_config = manifest.get("config", {})
    for key, value in config.items():
        if prior_config.get(key) != value:
            raise RuntimeError(
                f"resume config differs from the registered S07 run at {key}"
            )
    if manifest.get("dataset", {}).get("sha256") != sha256_file(dataset):
        raise RuntimeError("dataset changed before S07 resume")
    if manifest.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint):
        raise RuntimeError("checkpoint changed before S07 resume")
    for name, digest in manifest.get("output_hashes", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"S07 resume artifact changed or is missing: {name}")
    print(f"resume verified complete run: {manifest_path}")
    return manifest_path


def run(config: dict[str, Any], args: argparse.Namespace) -> Path:
    set_deterministic_seed(int(config["seed"]))
    repository = Path(__file__).resolve().parents[1]
    dataset = Path(config["dataset"]).resolve()
    checkpoint = Path(config["checkpoint"]).resolve()
    cohorts_path = Path(config["cohorts"]).resolve()
    ranking_path = Path(config["s04_ranking"]).resolve()
    motifs_path = Path(config["s05_motifs"]).resolve()
    upstream_map = Path(config["s06b_map"]).resolve()
    upstream_manifest_path = Path(config["s06b_manifest"]).resolve()
    output_dir = (
        args.output_dir or Path("output/xai/S07") / str(config["run_id"])
    ).resolve()

    upstream_manifest = json.loads(upstream_manifest_path.read_text(encoding="utf-8"))
    expected_map_hash = upstream_manifest["output_hashes"]["attribution_maps.h5"]
    actual_map_hash = sha256_file(upstream_map)
    if actual_map_hash != expected_map_hash:
        raise RuntimeError("S06b attribution map disagrees with its published manifest")
    config["script_sha256"] = sha256_file(__file__)
    config["physics_alignment_module_sha256"] = sha256_file(
        repository / "itg_nn/xai/physics_alignment.py"
    )
    config["upstream_s06b_manifest_sha256"] = sha256_file(upstream_manifest_path)
    config["upstream_s06b_map_sha256"] = actual_map_hash
    config["upstream_s05_motifs_sha256"] = sha256_file(motifs_path)
    config["upstream_s04_ranking_sha256"] = sha256_file(ranking_path)
    if args.resume:
        resumed = _resume_if_valid(output_dir, config, dataset, checkpoint)
        if resumed is not None:
            return resumed
    artifacts = RunArtifacts(output_dir)

    cohorts = json.loads(cohorts_path.read_text(encoding="utf-8"))
    registered_rows = np.asarray(
        cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64
    )
    top_registry = list(cohorts["member_cohorts"]["stored_validation_top_10"])
    member_ids = top_registry[: int(config["members"])]
    if len(member_ids) < 1:
        raise ValueError("at least one member is required")

    with h5py.File(dataset, "r") as h5_file:
        classes_all = _h5_take(h5_file["equilibrium_class"], registered_rows).astype(
            np.int16
        )
        varied_q_all = _h5_take(
            h5_file["varied_gradient_simulations/Q_avgs"], registered_rows
        )
    varied_target_all = np.maximum(np.log(varied_q_all), -2.0)
    positions = _stratified_positions(
        classes_all,
        varied_target_all,
        int(config["panel_varied_rows"]),
        float(config["stable_threshold_log_Q"]),
    )
    rows = registered_rows[positions]
    diagnostics = _load_panel_diagnostics(dataset, rows)
    panel_data = {
        gradient_set: load_hdf5_rows(
            dataset, rows, gradient_set=gradient_set, include_targets=True
        )
        for gradient_set in GRADIENT_SETS
    }
    groups = diagnostics["varied"]["equilibrium_file"]
    if len(set(groups)) != len(groups):
        raise RuntimeError("S01 S07 panel must retain one row per equilibrium")
    if not (
        np.allclose(diagnostics["fixed"]["a_over_lt"], 3.0)
        and np.allclose(diagnostics["fixed"]["a_over_ln"], 0.9)
    ):
        raise RuntimeError(
            "fixed-gradient rows do not use the registered (3, 0.9) drive"
        )

    with h5py.File(upstream_map, "r") as h5_file:
        upstream_rows = h5_file["row_id"][:]
        upstream_members = list(_decode(h5_file["member_id"][:]))
        upstream_functions = tuple(_decode(h5_file["function_name"][:]))
        upstream_methods = tuple(_decode(h5_file["method_name"][:]))
        upstream_gradient_sets = tuple(_decode(h5_file["gradient_set"][:]))
        if upstream_functions != FUNCTIONS or upstream_methods != METHODS:
            raise RuntimeError("S06b function/method axes changed")
        if upstream_gradient_sets != GRADIENT_SETS:
            raise RuntimeError("S06b gradient-set axis changed")
        row_lookup = {int(row): index for index, row in enumerate(upstream_rows)}
        member_lookup = {
            member_id: index for index, member_id in enumerate(upstream_members)
        }
        if not set(rows).issubset(row_lookup) or not set(member_ids).issubset(
            member_lookup
        ):
            raise RuntimeError("S07 cohort is absent from the S06b map")
        map_positions = np.asarray(
            [row_lookup[int(row)] for row in rows], dtype=np.int64
        )
        map_members = np.asarray(
            [member_lookup[member_id] for member_id in member_ids], dtype=np.int64
        )
        attribution = h5_file["attribution"][:, :, map_members][
            :, :, :, :, map_positions
        ].astype(np.float64)

    selected_unit_indices, selected_unit_ids = _load_unit_ranking(
        ranking_path, member_ids, int(config["selected_units_per_member"])
    )
    ensemble = load_ensemble(checkpoint, device=str(config["device"]))
    index_by_id = {
        member_id: index for index, member_id in enumerate(ensemble.member_ids)
    }
    models = {
        member_id: InvariantMember(ensemble.models[index_by_id[member_id]])
        for member_id in member_ids
    }
    densities, predictions = _density_and_predictions(
        models,
        member_ids,
        panel_data["varied"].geometry,
        panel_data,
        selected_unit_indices,
        batch_size=int(config["batch_size"]),
        device=str(config["device"]),
    )

    spatial_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    alignment_objects: dict[tuple[Any, ...], CircularAlignment] = {}
    source_profiles: dict[tuple[Any, ...], np.ndarray] = {}
    source_counter = 0
    threshold = float(config["stable_threshold_log_Q"])
    sparsity = float(config["alignment_sparsity"])
    bootstrap_replicates = int(config["bootstrap_replicates"])
    minimum_recurrence = float(config["minimum_lag_recurrence"])
    lag_tolerance = int(config["lag_stability_tolerance_positions"])
    selection_null_permutations = int(config["selection_null_permutations"])

    def add_alignment(
        key: tuple[Any, ...],
        metadata: dict[str, Any],
        source: np.ndarray,
        q_profile: np.ndarray,
        mask: np.ndarray,
        mode: str,
    ) -> None:
        nonlocal source_counter
        result = circular_alignment(
            source[mask],
            q_profile[mask],
            groups[mask],
            mode=mode,
            sparsity=sparsity,
            bootstrap_replicates=bootstrap_replicates,
            seed=int(config["seed"]) + source_counter,
        )
        source_counter += 1
        alignment_objects[key] = result
        source_profiles[key] = source
        row = _alignment_row(
            metadata,
            result,
            sample_count=int(mask.sum()),
            sparsity=sparsity,
            bootstrap_replicates=bootstrap_replicates,
            minimum_recurrence=minimum_recurrence,
            lag_tolerance=lag_tolerance,
        )
        if metadata["stratum"] == "unstable":
            selection_null = lag_selection_permutation_null(
                source[mask],
                q_profile[mask],
                groups[mask],
                mode=mode,
                permutations=selection_null_permutations,
                seed=int(config["seed"]) + 500000 + source_counter,
            )
            row.update(
                {
                    "selection_null_scope": (
                        "permuted_equilibrium_pairing_max_abs_over_96_lags"
                    ),
                    "selection_null_permutations": selection_null_permutations,
                    "selection_null_q95": selection_null.q95,
                    "selection_null_max": selection_null.maximum,
                    "observed_abs_correlation_to_selection_null_q95": (
                        abs(result.rank_correlation)
                        / max(selection_null.q95, np.finfo(np.float64).eps)
                    ),
                    "lag_search_null_resolved": (
                        abs(result.rank_correlation) > selection_null.q95
                    ),
                    "selection_null_permutation_unit": (
                        selection_null.permutation_group
                    ),
                }
            )
        else:
            row.update(
                {
                    "selection_null_scope": "not_run_non_unstable_stratum",
                    "selection_null_permutations": 0,
                    "selection_null_q95": None,
                    "selection_null_max": None,
                    "observed_abs_correlation_to_selection_null_q95": None,
                    "lag_search_null_resolved": None,
                    "selection_null_permutation_unit": "equilibrium_files",
                }
            )
        spatial_rows.append(row)
        for lag in range(96):
            lag_rows.append(
                {
                    **metadata,
                    "mode": mode,
                    "lag": _signed_lag(lag),
                    "lag_index_0_to_95": lag,
                    "mean_within_tube_spearman": result.rank_correlation_by_lag[lag],
                    "mean_within_tube_cross_correlation": result.cross_correlation_by_lag[
                        lag
                    ],
                }
            )

    for member_index, member_id in enumerate(member_ids):
        for unit_slot, unit_id in enumerate(selected_unit_ids[member_index]):
            source = densities[member_index, :, unit_slot]
            for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
                target = panel_data[gradient_set].actual_log_heat_flux
                assert target is not None
                for stratum, mask in _strata(target.numpy(), threshold):
                    for mode in MODES:
                        metadata = {
                            "source_family": "s05_density",
                            "source_id": unit_id,
                            "member_id": member_id,
                            "function": CANONICAL_FUNCTION,
                            "method": "equivariant_density_rho",
                            "gradient_set": gradient_set,
                            "stratum": stratum,
                            "source_signed": True,
                            "source_observed_nonnegative": bool(np.min(source) >= 0),
                            "validity_tag": OBSERVED_VALIDITY,
                            "source_feature_claims_permitted": True,
                            "estimator_backend": "s02_atrous_equivariant_density",
                            "feature_claims_permitted": stratum == "unstable",
                            "plasma_claims_permitted": stratum == "unstable",
                            "gx_quantity": "Q_avgs_vs_z",
                        }
                        key = (
                            "density",
                            member_id,
                            unit_id,
                            gradient_set,
                            stratum,
                            mode,
                        )
                        add_alignment(
                            key,
                            metadata,
                            source,
                            diagnostics[gradient_set]["q_vs_z"],
                            mask,
                            mode,
                        )

    for function_index, function_name in enumerate(FUNCTIONS):
        for method_index, method in enumerate(METHODS):
            modes = MODES if method == "ig_low_pass" else ("positive_contribution",)
            for member_index, member_id in enumerate(member_ids):
                for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
                    target = panel_data[gradient_set].actual_log_heat_flux
                    assert target is not None
                    maps = attribution[
                        function_index, method_index, member_index, gradient_index
                    ]
                    for mode in modes:
                        source = _position_source(maps, method, mode)
                        for stratum, mask in _strata(target.numpy(), threshold):
                            metadata = {
                                "source_family": "s06_attribution",
                                "source_id": f"{method}:channel_position_marginal",
                                "member_id": member_id,
                                "function": function_name,
                                "method": method,
                                "gradient_set": gradient_set,
                                "stratum": stratum,
                                "source_signed": method == "ig_low_pass",
                                "source_observed_nonnegative": method
                                == "periodic_mask",
                                "validity_tag": OFF_MANIFOLD_VALIDITY,
                                "source_feature_claims_permitted": method
                                == "ig_low_pass",
                                "estimator_backend": (
                                    "integrated_gradients_captum"
                                    if method == "ig_low_pass"
                                    else "periodic_extremal_mask"
                                ),
                                "feature_claims_permitted": (
                                    stratum == "unstable" and method == "ig_low_pass"
                                ),
                                "plasma_claims_permitted": False,
                                "gx_quantity": "Q_avgs_vs_z",
                            }
                            key = (
                                "attribution",
                                function_name,
                                method,
                                member_id,
                                gradient_set,
                                stratum,
                                mode,
                            )
                            add_alignment(
                                key,
                                metadata,
                                source,
                                diagnostics[gradient_set]["q_vs_z"],
                                mask,
                                mode,
                            )

    zonal_rows: list[dict[str, Any]] = []
    scalar_counter = 100000

    def add_scalar(
        metadata: dict[str, Any], values: np.ndarray, gradient_set: str
    ) -> None:
        nonlocal scalar_counter
        target = panel_data[gradient_set].actual_log_heat_flux
        assert target is not None
        zonal = diagnostics[gradient_set]["zonal_phi2"]
        log_zonal = np.log10(np.maximum(zonal, np.finfo(np.float64).tiny))
        for stratum, mask in _strata(target.numpy(), threshold):
            result = scalar_rank_association(
                values[mask],
                log_zonal[mask],
                groups[mask],
                bootstrap_replicates=bootstrap_replicates,
                seed=int(config["seed"]) + scalar_counter,
            )
            scalar_counter += 1
            zonal_rows.append(
                {
                    **metadata,
                    "gradient_set": gradient_set,
                    "stratum": stratum,
                    "sample_count": int(mask.sum()),
                    "spearman_rho": result.spearman_rho,
                    "spearman_ci95_lower": result.ci_lower,
                    "spearman_ci95_upper": result.ci_upper,
                    "bootstrap_stable": (result.ci_lower > 0 or result.ci_upper < 0),
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_unit": result.bootstrap_group,
                    "gx_quantity": "log10(zonal_phi2_amplitudes)",
                    "feature_claims_permitted": (
                        stratum == "unstable"
                        and bool(metadata.get("source_feature_claims_permitted", False))
                    ),
                    "plasma_claims_permitted": (
                        stratum == "unstable"
                        and bool(metadata.get("source_feature_claims_permitted", False))
                        and metadata.get("validity_tag") == OBSERVED_VALIDITY
                    ),
                    "association_not_causation": True,
                }
            )

    for member_index, member_id in enumerate(member_ids):
        for unit_slot, unit_id in enumerate(selected_unit_ids[member_index]):
            values = densities[member_index, :, unit_slot].mean(axis=1)
            for gradient_set in GRADIENT_SETS:
                add_scalar(
                    {
                        "source_family": "s05_density",
                        "source_id": unit_id,
                        "member_id": member_id,
                        "function": CANONICAL_FUNCTION,
                        "method": "equivariant_density_rho",
                        "summary": "mean_z_density",
                        "source_signed": True,
                        "validity_tag": OBSERVED_VALIDITY,
                        "source_feature_claims_permitted": True,
                    },
                    values,
                    gradient_set,
                )

    for function_index, function_name in enumerate(FUNCTIONS):
        for method_index, method in enumerate(METHODS):
            for member_index, member_id in enumerate(member_ids):
                for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
                    maps = attribution[
                        function_index, method_index, member_index, gradient_index
                    ]
                    summaries = {
                        "positive_cell_sum": np.maximum(maps, 0.0).sum(axis=(1, 2)),
                        "absolute_cell_sum": np.abs(maps).sum(axis=(1, 2)),
                        "top10pct_absolute_mass_fraction": _top_mass_fraction(maps),
                    }
                    if method == "ig_low_pass":
                        summaries["signed_cell_sum"] = maps.sum(axis=(1, 2))
                    for summary_name, values in summaries.items():
                        add_scalar(
                            {
                                "source_family": "s06_attribution",
                                "source_id": f"{method}:all_channels",
                                "member_id": member_id,
                                "function": function_name,
                                "method": method,
                                "summary": summary_name,
                                "source_signed": method == "ig_low_pass",
                                "validity_tag": OFF_MANIFOLD_VALIDITY,
                                "source_feature_claims_permitted": method
                                == "ig_low_pass",
                            },
                            values,
                            gradient_set,
                        )

    paired_rows: list[dict[str, Any]] = []
    paired_counter = 200000

    fixed_target = panel_data["fixed"].actual_log_heat_flux
    varied_target = panel_data["varied"].actual_log_heat_flux
    assert fixed_target is not None and varied_target is not None
    either_near_floor = (fixed_target.numpy() <= threshold) | (
        varied_target.numpy() <= threshold
    )
    pair_strata = (
        ("all", np.ones(len(groups), dtype=bool)),
        ("either_stable_or_near_floor", either_near_floor),
        ("both_unstable", ~either_near_floor),
    )

    def add_paired_difference(
        metadata: dict[str, Any],
        fixed: np.ndarray,
        varied: np.ndarray,
        estimand: str,
    ) -> None:
        nonlocal paired_counter
        for stratum, mask in pair_strata:
            result = paired_native_difference(
                fixed[mask],
                varied[mask],
                groups[mask],
                bootstrap_replicates=bootstrap_replicates,
                seed=int(config["seed"]) + paired_counter,
                estimand=estimand,
            )
            paired_counter += 1
            paired_rows.append(
                {
                    **metadata,
                    "analysis_kind": "paired_fixed_minus_varied_same_geometry",
                    "stratum": stratum,
                    "sample_count": int(mask.sum()),
                    "estimate": result.estimate,
                    "ci95_lower": result.ci_lower,
                    "ci95_upper": result.ci_upper,
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_unit": result.bootstrap_group,
                    "estimand": result.estimand,
                    "validity_tag": metadata.get("validity_tag", OBSERVED_VALIDITY),
                    "comparison_validity_tag": OBSERVED_VALIDITY,
                    "feature_claims_permitted": (
                        stratum == "both_unstable"
                        and bool(metadata.get("source_feature_claims_permitted", False))
                    ),
                    "plasma_claims_permitted": (
                        stratum == "both_unstable"
                        and bool(metadata.get("source_feature_claims_permitted", False))
                        and metadata.get("validity_tag", OBSERVED_VALIDITY)
                        == OBSERVED_VALIDITY
                    ),
                    "pair_stratum_definition": (
                        "near-floor if either observed native target is <= -1.9"
                    ),
                    "pairing_scope": (
                        "same geometry; fixed panel holds drive at (3,0.9) across "
                        "geometries, while within-pair varied drives may differ"
                    ),
                }
            )

    add_paired_difference(
        {
            "quantity": "observed_clipped_log_Q",
            "member_id": "none",
            "source_feature_claims_permitted": True,
        },
        fixed_target.numpy(),
        varied_target.numpy(),
        NATIVE_ESTIMAND,
    )
    add_paired_difference(
        {
            "quantity": "log10_zonal_phi2",
            "member_id": "none",
            "source_feature_claims_permitted": True,
        },
        np.log10(diagnostics["fixed"]["zonal_phi2"]),
        np.log10(diagnostics["varied"]["zonal_phi2"]),
        "log10 zonal_phi2_amplitudes",
    )
    for function_index, function_name in enumerate(FUNCTIONS):
        for member_index, member_id in enumerate(member_ids):
            add_paired_difference(
                {
                    "quantity": "member_prediction",
                    "member_id": member_id,
                    "function": function_name,
                },
                predictions[function_index, member_index, 1],
                predictions[function_index, member_index, 0],
                NATIVE_ESTIMAND,
            )

    for mode in MODES:
        for stratum, mask in pair_strata:
            physical_pair = circular_alignment(
                diagnostics["fixed"]["q_vs_z"][mask],
                diagnostics["varied"]["q_vs_z"][mask],
                groups[mask],
                mode=mode,
                sparsity=sparsity,
                bootstrap_replicates=bootstrap_replicates,
                seed=int(config["seed"]) + paired_counter,
            )
            paired_counter += 1
            paired_rows.append(
                {
                    "analysis_kind": "physical_Qz_fixed_vs_varied_same_geometry",
                    "quantity": "Q_avgs_vs_z",
                    "mode": mode,
                    "member_id": "none",
                    "stratum": stratum,
                    "sample_count": int(mask.sum()),
                    "best_lag": _signed_lag(physical_pair.best_lag),
                    "circular_spearman": physical_pair.rank_correlation,
                    "ci95_lower": physical_pair.rank_ci_lower,
                    "ci95_upper": physical_pair.rank_ci_upper,
                    "overlap_at_fixed_sparsity": physical_pair.overlap,
                    "overlap_orientation": physical_pair.overlap_orientation,
                    "lag_recurrence": physical_pair.lag_recurrence,
                    "bootstrap_unit": physical_pair.bootstrap_group,
                    "validity_tag": OBSERVED_VALIDITY,
                    "source_feature_claims_permitted": True,
                    "comparison_validity_tag": OBSERVED_VALIDITY,
                    "feature_claims_permitted": stratum == "both_unstable",
                    "plasma_claims_permitted": stratum == "both_unstable",
                    "pair_stratum_definition": (
                        "near-floor if either observed native target is <= -1.9"
                    ),
                    "pairing_scope": (
                        "same geometry, different registered simulation drives"
                    ),
                }
            )

    for member_index, member_id in enumerate(member_ids):
        for unit_id in selected_unit_ids[member_index]:
            for mode in MODES:
                fixed_result = alignment_objects[
                    ("density", member_id, unit_id, "fixed", "all", mode)
                ]
                varied_result = alignment_objects[
                    ("density", member_id, unit_id, "varied", "all", mode)
                ]
                add_paired_difference(
                    {
                        "quantity": "learned_Qz_spatial_spearman",
                        "source_family": "s05_density",
                        "source_id": unit_id,
                        "member_id": member_id,
                        "mode": mode,
                        "validity_tag": OBSERVED_VALIDITY,
                        "source_feature_claims_permitted": True,
                    },
                    fixed_result.per_sample_rank_correlation,
                    varied_result.per_sample_rank_correlation,
                    "fixed-minus-varied within-tube Spearman at each set's selected lag",
                )

    for function_name in FUNCTIONS:
        for method in METHODS:
            modes = MODES if method == "ig_low_pass" else ("positive_contribution",)
            for member_id in member_ids:
                for mode in modes:
                    fixed_result = alignment_objects[
                        (
                            "attribution",
                            function_name,
                            method,
                            member_id,
                            "fixed",
                            "all",
                            mode,
                        )
                    ]
                    varied_result = alignment_objects[
                        (
                            "attribution",
                            function_name,
                            method,
                            member_id,
                            "varied",
                            "all",
                            mode,
                        )
                    ]
                    add_paired_difference(
                        {
                            "quantity": "learned_Qz_spatial_spearman",
                            "source_family": "s06_attribution",
                            "source_id": f"{method}:channel_position_marginal",
                            "member_id": member_id,
                            "function": function_name,
                            "method": method,
                            "mode": mode,
                            "validity_tag": OFF_MANIFOLD_VALIDITY,
                            "source_feature_claims_permitted": method == "ig_low_pass",
                        },
                        fixed_result.per_sample_rank_correlation,
                        varied_result.per_sample_rank_correlation,
                        "fixed-minus-varied within-tube Spearman at each set's selected lag",
                    )

    case_rows: list[dict[str, Any]] = []
    case_plot_data: list[dict[str, Any]] = []
    row_to_position = {int(row): index for index, row in enumerate(rows)}
    unstable_varied = varied_target.numpy() > threshold
    for hypothesis in config["case_hypotheses"]:
        member_id = str(hypothesis["member_id"])
        unit_id = str(hypothesis["unit_id"])
        if member_id not in member_ids:
            continue
        member_index = member_ids.index(member_id)
        unit_slot = selected_unit_ids[member_index].index(unit_id)
        result = alignment_objects[
            ("density", member_id, unit_id, "varied", "unstable", "signed")
        ]
        selected = select_balanced_case_studies(
            result.per_sample_rank_correlation,
            rows[unstable_varied],
            groups[unstable_varied],
            per_direction=int(config["case_studies_per_direction"]),
            expected_sign=(1 if result.rank_correlation >= 0 else -1),
        )
        for case in selected:
            position = row_to_position[int(case["row_id"])]
            record = {
                **case,
                "hypothesis": hypothesis["hypothesis"],
                "s05_concept": hypothesis["s05_concept"],
                "member_id": member_id,
                "unit_id": unit_id,
                "gradient_set": "varied",
                "stratum": "unstable",
                "registered_lag": _signed_lag(result.best_lag),
                "q_total": diagnostics["varied"]["q"][position],
                "zonal_phi2": diagnostics["varied"]["zonal_phi2"][position],
                "a_over_lt": diagnostics["varied"]["a_over_lt"][position],
                "a_over_ln": diagnostics["varied"]["a_over_ln"][position],
                "canonical_prediction": predictions[1, member_index, 0, position],
                "original_prediction": predictions[0, member_index, 0, position],
                "density_mean": densities[member_index, position, unit_slot].mean(),
                "validity_tag": OBSERVED_VALIDITY,
                "feature_claims_permitted": True,
                "plasma_claims_permitted": True,
                "selection_rule": "extreme per-row spatial correlation at registered panel lag",
            }
            case_rows.append(record)
            case_plot_data.append(
                {
                    **record,
                    "density": densities[member_index, position, unit_slot].astype(
                        np.float64
                    ),
                    "q_vs_z": diagnostics["varied"]["q_vs_z"][position],
                    "lag_index": result.best_lag,
                }
            )

    # Exact-symmetry metric check: jointly rolling both traces cannot change any
    # lag curve or selected lag. This checks S07's comparison, while S05/S06
    # retain the model/explanation symmetry checks inherited by the sources.
    symmetry_source = densities[0, :, 0]
    symmetry_q = diagnostics["varied"]["q_vs_z"]
    symmetry_base = circular_alignment(
        symmetry_source,
        symmetry_q,
        groups,
        mode="signed",
        sparsity=sparsity,
        bootstrap_replicates=bootstrap_replicates,
        seed=int(config["seed"]) + 999001,
    )
    symmetry_shifted = circular_alignment(
        np.roll(symmetry_source, 17, axis=1),
        np.roll(symmetry_q, 17, axis=1),
        groups,
        mode="signed",
        sparsity=sparsity,
        bootstrap_replicates=bootstrap_replicates,
        seed=int(config["seed"]) + 999001,
    )
    symmetry_error = float(
        np.max(
            np.abs(
                symmetry_base.rank_correlation_by_lag
                - symmetry_shifted.rank_correlation_by_lag
            )
        )
    )

    density_score = np.empty(
        (
            len(member_ids),
            int(config["selected_units_per_member"]),
            len(GRADIENT_SETS),
            len(MODES),
            len(rows),
        ),
        dtype=np.float32,
    )
    density_overlap = np.empty_like(density_score)
    attribution_score = np.full(
        (
            len(FUNCTIONS),
            len(METHODS),
            len(member_ids),
            len(GRADIENT_SETS),
            len(MODES),
            len(rows),
        ),
        np.nan,
        dtype=np.float32,
    )
    attribution_overlap = np.full_like(attribution_score, np.nan)
    for member_index, member_id in enumerate(member_ids):
        for unit_slot, unit_id in enumerate(selected_unit_ids[member_index]):
            for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
                for mode_index, mode in enumerate(MODES):
                    result = alignment_objects[
                        ("density", member_id, unit_id, gradient_set, "all", mode)
                    ]
                    density_score[
                        member_index, unit_slot, gradient_index, mode_index
                    ] = result.per_sample_rank_correlation
                    density_overlap[
                        member_index, unit_slot, gradient_index, mode_index
                    ] = result.per_sample_overlap
    for function_index, function_name in enumerate(FUNCTIONS):
        for method_index, method in enumerate(METHODS):
            modes = MODES if method == "ig_low_pass" else ("positive_contribution",)
            for member_index, member_id in enumerate(member_ids):
                for gradient_index, gradient_set in enumerate(GRADIENT_SETS):
                    for mode in modes:
                        mode_index = MODES.index(mode)
                        result = alignment_objects[
                            (
                                "attribution",
                                function_name,
                                method,
                                member_id,
                                gradient_set,
                                "all",
                                mode,
                            )
                        ]
                        attribution_score[
                            function_index,
                            method_index,
                            member_index,
                            gradient_index,
                            mode_index,
                        ] = result.per_sample_rank_correlation
                        attribution_overlap[
                            function_index,
                            method_index,
                            member_index,
                            gradient_index,
                            mode_index,
                        ] = result.per_sample_overlap

    details_path = artifacts.write_hdf5(
        "alignment_details.h5",
        {
            "density": densities,
            "density_unit_id": np.asarray(selected_unit_ids, dtype="S128"),
            "density_spatial_spearman": density_score,
            "density_spatial_overlap": density_overlap,
            "attribution_spatial_spearman": attribution_score,
            "attribution_spatial_overlap": attribution_overlap,
            "prediction": predictions,
            "q_vs_z": np.stack(
                [diagnostics[gradient_set]["q_vs_z"] for gradient_set in GRADIENT_SETS]
            ).astype(np.float32),
            "zonal_phi2": np.stack(
                [
                    diagnostics[gradient_set]["zonal_phi2"]
                    for gradient_set in GRADIENT_SETS
                ]
            ),
            "target_native": np.stack(
                [
                    panel_data[gradient_set].actual_log_heat_flux.numpy()
                    for gradient_set in GRADIENT_SETS
                ]
            ),
            "a_over_lt": np.stack(
                [
                    diagnostics[gradient_set]["a_over_lt"]
                    for gradient_set in GRADIENT_SETS
                ]
            ),
            "a_over_ln": np.stack(
                [
                    diagnostics[gradient_set]["a_over_ln"]
                    for gradient_set in GRADIENT_SETS
                ]
            ),
            "row_id": rows,
            "equilibrium_file": _strings(groups, width=256),
            "member_id": _strings(member_ids),
            "function_name": _strings(FUNCTIONS),
            "method_name": _strings(METHODS),
            "gradient_set": _strings(GRADIENT_SETS),
            "mode": _strings(MODES),
        },
        axes={
            "density": ("member", "sample", "selected_unit", "z"),
            "density_unit_id": ("member", "selected_unit"),
            "density_spatial_spearman": (
                "member",
                "selected_unit",
                "gradient_set",
                "mode",
                "sample",
            ),
            "density_spatial_overlap": (
                "member",
                "selected_unit",
                "gradient_set",
                "mode",
                "sample",
            ),
            "attribution_spatial_spearman": (
                "function",
                "method",
                "member",
                "gradient_set",
                "mode",
                "sample",
            ),
            "attribution_spatial_overlap": (
                "function",
                "method",
                "member",
                "gradient_set",
                "mode",
                "sample",
            ),
            "prediction": ("function", "member", "gradient_set", "sample"),
            "q_vs_z": ("gradient_set", "sample", "z"),
            "zonal_phi2": ("gradient_set", "sample"),
            "target_native": ("gradient_set", "sample"),
            "a_over_lt": ("gradient_set", "sample"),
            "a_over_ln": ("gradient_set", "sample"),
            "row_id": ("sample",),
            "equilibrium_file": ("sample",),
            "member_id": ("member",),
            "function_name": ("function",),
            "method_name": ("method",),
            "gradient_set": ("gradient_set",),
            "mode": ("mode",),
        },
        attributes={
            "estimand": NATIVE_ESTIMAND,
            "member_level_signed_before_aggregation": True,
            "stable_feature_claims_permitted": False,
            "q_vs_z_role": "held-out physical diagnostic, not an attribution target",
            "upstream_s06b_map_sha256": actual_map_hash,
        },
        compression="gzip",
    )
    spatial_path = artifacts.write_text(
        "spatial_alignment.csv", _csv_text(spatial_rows)
    )
    lag_path = artifacts.write_text("lag_curves.csv", _csv_text(lag_rows))
    zonal_path = artifacts.write_text("zonal_association.csv", _csv_text(zonal_rows))
    paired_path = artifacts.write_text("paired_analysis.csv", _csv_text(paired_rows))
    cases_path = artifacts.write_text("case_studies.csv", _csv_text(case_rows))

    atlas_path = output_dir / "physics_alignment_atlas.png"
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    candidate_keys = []
    for hypothesis in config["case_hypotheses"]:
        key = (
            "density",
            str(hypothesis["member_id"]),
            str(hypothesis["unit_id"]),
            "varied",
            "unstable",
            "signed",
        )
        if key in alignment_objects:
            candidate_keys.append((str(hypothesis["hypothesis"]), key))
    for label, key in candidate_keys:
        result = alignment_objects[key]
        lag_axis = np.asarray([_signed_lag(lag) for lag in range(96)])
        order = np.argsort(lag_axis)
        axes[0, 0].plot(
            lag_axis[order], result.rank_correlation_by_lag[order], label=label
        )
    axes[0, 0].axhline(0, color="black", linewidth=0.7)
    axes[0, 0].set(
        title="S05 density vs varied unstable Q(z)",
        xlabel="lag",
        ylabel="mean Spearman",
    )
    axes[0, 0].legend(fontsize=7)

    for member_id in member_ids:
        key = (
            "attribution",
            CANONICAL_FUNCTION,
            "ig_low_pass",
            member_id,
            "varied",
            "unstable",
            "signed",
        )
        result = alignment_objects[key]
        lag_axis = np.asarray([_signed_lag(lag) for lag in range(96)])
        order = np.argsort(lag_axis)
        axes[0, 1].plot(
            lag_axis[order], result.rank_correlation_by_lag[order], label=member_id
        )
    axes[0, 1].axhline(0, color="black", linewidth=0.7)
    axes[0, 1].set(
        title="Canonical IG position marginal vs Q(z)",
        xlabel="lag",
        ylabel="mean Spearman",
    )
    axes[0, 1].legend(fontsize=7)

    density_headline = [
        row
        for row in spatial_rows
        if row["source_family"] == "s05_density"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
        and row["mode"] == "signed"
    ]
    density_headline.sort(
        key=lambda row: abs(float(row["circular_spearman"])), reverse=True
    )
    shown = density_headline[: min(9, len(density_headline))]
    axes[1, 0].barh(
        np.arange(len(shown)), [float(row["circular_spearman"]) for row in shown]
    )
    axes[1, 0].set_yticks(
        np.arange(len(shown)), [str(row["source_id"]).split(":")[-1] for row in shown]
    )
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(
        title="Top density/Q(z) associations", xlabel="lag-selected Spearman"
    )

    zonal_headline = [
        row
        for row in zonal_rows
        if row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
        and bool(row["feature_claims_permitted"])
    ]
    zonal_headline.sort(key=lambda row: abs(float(row["spearman_rho"])), reverse=True)
    shown_zonal = zonal_headline[: min(9, len(zonal_headline))]
    axes[1, 1].barh(
        np.arange(len(shown_zonal)), [float(row["spearman_rho"]) for row in shown_zonal]
    )
    axes[1, 1].set_yticks(
        np.arange(len(shown_zonal)),
        [
            f"{str(row['source_id']).split(':')[0]}:{row['summary']}"[-34:]
            for row in shown_zonal
        ],
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(title="Learned summaries vs zonal phi2", xlabel="Spearman")
    figure.savefig(atlas_path, dpi=180)
    plt.close(figure)
    artifacts.register_existing(atlas_path.name)

    case_figure_path = output_dir / "case_studies.png"
    shown_cases: list[dict[str, Any]] = []
    for hypothesis in config["case_hypotheses"]:
        matches = [
            row
            for row in case_plot_data
            if row["hypothesis"] == hypothesis["hypothesis"]
        ]
        for case_type in ("supporting", "contradicting"):
            shown_cases.extend(
                [row for row in matches if row["case_type"] == case_type][:2]
            )
    columns = 2
    rows_count = max(1, int(np.ceil(len(shown_cases) / columns)))
    figure, case_axes = plt.subplots(
        rows_count,
        columns,
        figsize=(12, 2.7 * rows_count),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, case in zip(case_axes.ravel(), shown_cases):
        density = np.asarray(case["density"])
        q_aligned = np.roll(np.asarray(case["q_vs_z"]), int(case["lag_index"]))
        density_scaled = (density - density.mean()) / max(density.std(), 1e-12)
        q_scaled = (q_aligned - q_aligned.mean()) / max(q_aligned.std(), 1e-12)
        axis.plot(density_scaled, label="unit density", linewidth=1.1)
        axis.plot(q_scaled, label="aligned GX Q(z)", linewidth=1.1)
        axis.set_title(
            f"{case['hypothesis']} | {case['case_type']} | r={case['score']:.2f}",
            fontsize=8,
        )
        axis.set_xlabel("periodic z index")
        axis.set_ylabel("within-row standardized")
    for axis in case_axes.ravel()[len(shown_cases) :]:
        axis.axis("off")
    if shown_cases:
        case_axes.ravel()[0].legend(fontsize=7)
    figure.savefig(case_figure_path, dpi=180)
    plt.close(figure)
    artifacts.register_existing(case_figure_path.name)

    strongest_density = max(
        density_headline, key=lambda row: abs(float(row["circular_spearman"]))
    )
    ig_headline = [
        row
        for row in spatial_rows
        if row["source_family"] == "s06_attribution"
        and row["method"] == "ig_low_pass"
        and row["function"] == CANONICAL_FUNCTION
        and row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
        and row["mode"] == "signed"
    ]
    strongest_ig = max(
        ig_headline, key=lambda row: abs(float(row["circular_spearman"]))
    )
    strongest_zonal = max(
        zonal_headline, key=lambda row: abs(float(row["spearman_rho"]))
    )
    original_ig = [
        row
        for row in spatial_rows
        if row["source_family"] == "s06_attribution"
        and row["method"] == "ig_low_pass"
        and row["function"] == "original_f"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "unstable"
        and row["mode"] == "signed"
    ]
    summary = {
        "step": "S07",
        "run_id": config["run_id"],
        "estimand": NATIVE_ESTIMAND,
        "cohort": {
            "rows_per_gradient_set": len(rows),
            "equilibrium_files": len(set(groups)),
            "member_ids": member_ids,
            "selected_unit_ids": selected_unit_ids,
            "stable_counts": {
                gradient_set: int(
                    np.sum(
                        panel_data[gradient_set].actual_log_heat_flux.numpy()
                        <= threshold
                    )
                )
                for gradient_set in GRADIENT_SETS
            },
        },
        "headline": {
            "strongest_varied_unstable_density_Qz": strongest_density,
            "strongest_varied_unstable_canonical_ig_Qz": strongest_ig,
            "strongest_varied_unstable_zonal_association": strongest_zonal,
            "canonical_original_ig_spearman_difference_by_member": [
                {
                    "member_id": canonical["member_id"],
                    "canonical": canonical["circular_spearman"],
                    "original": next(
                        row["circular_spearman"]
                        for row in original_ig
                        if row["member_id"] == canonical["member_id"]
                    ),
                }
                for canonical in ig_headline
            ],
        },
        "counts": {
            "spatial_alignment_rows": len(spatial_rows),
            "lag_curve_rows": len(lag_rows),
            "zonal_association_rows": len(zonal_rows),
            "paired_analysis_rows": len(paired_rows),
            "case_study_rows": len(case_rows),
            "association_bootstrap_stable_spatial_rows": sum(
                bool(row["association_bootstrap_stable"]) for row in spatial_rows
            ),
            "lag_bootstrap_stable_spatial_rows": sum(
                bool(row["lag_bootstrap_stable"]) for row in spatial_rows
            ),
        },
        "symmetry": {
            "s07_joint_shift_lag_curve_max_abs_error": symmetry_error,
            "shift": 17,
            "source_inheritance": (
                "S05 density and S06 attribution source symmetry remain in their "
                "registered artifacts; S07 checks the joint-shift invariance of its "
                "association statistic"
            ),
        },
        "interpretation_limit": (
            "Q_avgs_vs_z is a held-out GX physical diagnostic. Correlation with a "
            "network density or attribution is association, not identity or causality."
        ),
        "stable_feature_claims_permitted": False,
        "upstream": {
            "s06b_map_sha256": actual_map_hash,
            "s06b_manifest_sha256": config["upstream_s06b_manifest_sha256"],
            "s05_motifs_sha256": config["upstream_s05_motifs_sha256"],
            "s04_ranking_sha256": config["upstream_s04_ranking_sha256"],
        },
    }
    summary_path = artifacts.write_json("summary.json", summary)

    small_paths = [
        spatial_path,
        lag_path,
        zonal_path,
        paired_path,
        cases_path,
        atlas_path,
        case_figure_path,
        summary_path,
    ]
    published_dir = None
    if not args.no_publish:
        published_dir = Path(config["published_dir"]).resolve()
        _publish(small_paths, published_dir)
    config["published_artifacts"] = [path.name for path in small_paths]
    config["large_artifacts"] = [details_path.name]
    config["member_ids"] = member_ids
    config["row_ids_per_gradient_set"] = rows.tolist()
    manifest_path = artifacts.finalize(
        config=config,
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=member_ids,
        row_ids=np.concatenate((rows, rows)),
        gradient_set="fixed_and_varied_separate",
        device=str(config["device"]),
        repository=repository,
        command=sys.argv,
        published_dir=published_dir,
    )
    print(f"S07 complete: {manifest_path}")
    return manifest_path


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(_resolve(config, args), args)


if __name__ == "__main__":
    main()
