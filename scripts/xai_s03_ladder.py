#!/usr/bin/env python3
"""Execute S03's structure-destroying counterfactual ladder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from itg_nn.data import InferenceData, load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.perturbations import (
    PerturbationSpec,
    ReferenceBackgrounds,
    ScaledPCASupport,
    ValidityTag,
    attenuate_fourier_band,
    block_permutation,
    independent_channel_shifts,
    interpolate_geometry,
    joint_permutation,
    member_window_lengths,
    phase_scramble,
    random_joint_shift,
    replace_channel,
    scale_non_dc_amplitude,
    wrapped_window_mask,
)
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import (
    InvariantMember,
    receptive_field_blocks,
    reverse_parallel,
    stellarator_parity,
)
from itg_nn.xai.toys import ColocationToy, FourierBandToy, PeriodicPermutationToy


FUNCTION_NAMES = ("original_f", "invariant_tilde_f")
CHANNEL_NAMES = (
    "bmag",
    "gbdrift",
    "cvdrift",
    "gbdrift0_over_shat",
    "gds2",
    "gds21_over_shat",
    "gds22_over_shat_squared",
)
FOURIER_BANDS = {"low": (1, 4), "mid": (5, 16), "high": (17, 48)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/xai/S03_ladder.json"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--published-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--members", type=int, help="Cap the all-member cohort")
    parser.add_argument("--rows", type=int, help="Number of varied rows plus paired fixed rows")
    parser.add_argument(
        "--families",
        help="Comma-separated perturbation families for a targeted reproducible rerun",
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    return parser


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def _h5_take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    return dataset[unique][inverse]


def _resolve(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = dict(config)
    pilot = bool(args.pilot)
    if pilot:
        resolved.update(config["pilot"])
    resolved["mode"] = "pilot" if pilot else "production"
    for name in ("device", "batch_size", "seed"):
        value = getattr(args, name)
        if value is not None:
            resolved[name] = value
    if args.rows is not None:
        resolved["panel_varied_rows"] = args.rows
    if args.members is not None:
        resolved["members"] = args.members
    if args.families:
        resolved["families"] = [
            family.strip() for family in args.families.split(",") if family.strip()
        ]
    for name in ("dataset", "checkpoint", "cohorts", "published_dir"):
        value = getattr(args, name)
        if value is not None:
            resolved[name] = str(value)
    return resolved


def _load_panel(
    dataset: Path,
    cohorts: dict[str, Any],
    varied_count: int,
    stable_threshold: float,
) -> tuple[InferenceData, dict[str, np.ndarray]]:
    registered = np.asarray(cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64)
    if not 1 <= varied_count <= len(registered):
        raise ValueError("panel_varied_rows exceeds the frozen S01 panel")
    if varied_count == len(registered):
        rows = registered
    else:
        # A pilot must rehearse the registered panel strata rather than take the
        # lowest sorted dataset indices.  Allocate proportionally across
        # equilibrium class x stable/unstable strata, then sample evenly within
        # each stratum without introducing a new random selection.
        registered_data = load_hdf5_rows(
            dataset, registered, gradient_set="varied", include_targets=True
        )
        with h5py.File(dataset, "r") as h5_file:
            registered_class = _h5_take(
                h5_file["equilibrium_class"], registered
            ).astype(np.int16)
        registered_stable = (
            registered_data.actual_log_heat_flux.numpy() <= float(stable_threshold)
        ).astype(np.int8)
        labels = np.asarray(
            [f"{class_value}:{stable}" for class_value, stable in zip(
                registered_class, registered_stable
            )]
        )
        unique, counts = np.unique(labels, return_counts=True)
        if varied_count < len(unique):
            raise ValueError("pilot rows are insufficient to cover every panel stratum")
        allocation = np.ones(len(unique), dtype=np.int64)
        remaining = varied_count - len(unique)
        capacity = counts - 1
        if remaining:
            ideal = remaining * capacity / capacity.sum()
            allocation += np.floor(ideal).astype(np.int64)
            leftover = varied_count - int(allocation.sum())
            order = np.argsort(-(ideal - np.floor(ideal)), kind="stable")
            allocation[order[:leftover]] += 1
        if np.any(allocation > counts) or int(allocation.sum()) != varied_count:
            raise RuntimeError("stratified pilot allocation exceeded stratum capacity")
        selected: list[int] = []
        for label, quota in zip(unique, allocation):
            candidates = np.flatnonzero(labels == label)
            positions = np.floor(
                (np.arange(quota, dtype=np.float64) + 0.5) * len(candidates) / quota
            ).astype(np.int64)
            selected.extend(candidates[positions].tolist())
        rows = np.sort(registered[np.asarray(selected, dtype=np.int64)])
    varied = load_hdf5_rows(dataset, rows, gradient_set="varied", include_targets=True)
    fixed = load_hdf5_rows(dataset, rows, gradient_set="fixed", include_targets=True)
    target = torch.cat((varied.actual_log_heat_flux, fixed.actual_log_heat_flux))
    panel = InferenceData(
        geometry=torch.cat((varied.geometry, fixed.geometry)),
        a_over_lt=torch.cat((varied.a_over_lt, fixed.a_over_lt)),
        a_over_ln=torch.cat((varied.a_over_ln, fixed.a_over_ln)),
        row_indices=np.concatenate((rows, rows)),
        actual_log_heat_flux=target,
    )
    with h5py.File(dataset, "r") as h5_file:
        equilibrium_class = _h5_take(h5_file["equilibrium_class"], rows).astype(np.int16)
        equilibrium_file = _decode(_h5_take(h5_file["equilibrium_files"], rows))
    metadata = {
        "equilibrium_class": np.concatenate((equilibrium_class, equilibrium_class)),
        "equilibrium_file": np.concatenate((equilibrium_file, equilibrium_file)),
        "gradient_set": np.asarray(["varied"] * varied_count + ["fixed"] * varied_count),
        "panel_selection": np.asarray(
            ["full_registered_panel" if varied_count == len(registered) else "proportional_class_x_stability"]
            * (2 * varied_count)
        ),
    }
    return panel, metadata


def _load_support_reference(
    dataset: Path,
    cohorts: dict[str, Any],
    panel_rows: np.ndarray,
    count: int,
    seed: int,
) -> tuple[InferenceData, InferenceData, dict[str, np.ndarray]]:
    registered = np.asarray(cohorts["reference_varied"]["row_ids"], dtype=np.int64)
    with h5py.File(dataset, "r") as h5_file:
        equilibrium_file = _decode(_h5_take(h5_file["equilibrium_files"], registered))
        equilibrium_class = _h5_take(h5_file["equilibrium_class"], registered).astype(np.int16)
        panel_equilibrium_file = _decode(
            _h5_take(h5_file["equilibrium_files"], np.asarray(panel_rows, dtype=np.int64))
        )
    generator = np.random.default_rng(seed)
    candidates = generator.permutation(len(registered))
    selected: list[int] = []
    seen: set[str] = set()
    panel_groups = set(panel_equilibrium_file.tolist())
    for index in candidates:
        row = int(registered[index])
        group = str(equilibrium_file[index])
        if group not in panel_groups and group not in seen:
            selected.append(int(index))
            seen.add(group)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError("not enough equilibrium-unique support rows")
    selected_array = np.asarray(selected)
    rows = registered[selected_array]
    varied = load_hdf5_rows(dataset, rows, gradient_set="varied", include_targets=False)
    fixed = load_hdf5_rows(dataset, rows, gradient_set="fixed", include_targets=False)
    metadata = {
        "equilibrium_class": equilibrium_class[selected_array],
        "equilibrium_file": equilibrium_file[selected_array],
    }
    return varied, fixed, metadata


def _specs(config: dict[str, Any]) -> list[PerturbationSpec]:
    specs = [
        PerturbationSpec(
            "joint_shift_32", "joint_shift", ValidityTag.EXACT_SYMMETRY,
            parameter="shift=32", member_scope="all100", matched_control="identity"
        ),
        PerturbationSpec(
            "stellarator_parity", "parity", ValidityTag.EXACT_SYMMETRY,
            member_scope="all100", matched_control="wrong_parity"
        ),
        PerturbationSpec(
            "wrong_parity", "wrong_parity", ValidityTag.OFF_MANIFOLD,
            member_scope="all100", matched_control="stellarator_parity"
        ),
    ]
    replicates = int(config["random_replicates"])
    for replicate in range(replicates):
        scope = "all100" if replicate == 0 else "top10"
        specs.extend(
            (
                PerturbationSpec(
                    f"random_joint_shift_r{replicate}", "random_joint_shift",
                    ValidityTag.EXACT_SYMMETRY, replicate=replicate, member_scope=scope,
                    matched_control=f"independent_channel_shift_r{replicate}",
                ),
                PerturbationSpec(
                    f"joint_permutation_r{replicate}", "joint_permutation",
                    ValidityTag.OFF_MANIFOLD, replicate=replicate, member_scope=scope,
                    matched_control=f"random_joint_shift_r{replicate}",
                ),
                PerturbationSpec(
                    f"independent_channel_shift_r{replicate}", "independent_shift",
                    ValidityTag.OFF_MANIFOLD, replicate=replicate, member_scope=scope,
                    matched_control=f"random_joint_shift_r{replicate}",
                ),
                PerturbationSpec(
                    f"common_phase_scramble_r{replicate}", "common_phase_scramble",
                    ValidityTag.OFF_MANIFOLD, replicate=replicate, member_scope=scope,
                    matched_control=f"channel_phase_scramble_r{replicate}",
                ),
                PerturbationSpec(
                    f"channel_phase_scramble_r{replicate}", "channel_phase_scramble",
                    ValidityTag.OFF_MANIFOLD, replicate=replicate, member_scope=scope,
                    matched_control=f"common_phase_scramble_r{replicate}",
                ),
            )
        )
        for length in (2, 4, 8, 16, 32):
            specs.append(
                PerturbationSpec(
                    f"block_permutation_L{length}_r{replicate}", "block_permutation",
                    ValidityTag.OFF_MANIFOLD, replicate=replicate,
                    parameter=f"block_length={length}", member_scope=scope,
                    matched_control=f"random_joint_shift_r{replicate}",
                )
            )
    for band, (minimum, maximum) in FOURIER_BANDS.items():
        for dose in config["fourier_doses"]:
            specs.append(
                PerturbationSpec(
                    f"attenuate_{band}_d{dose:g}", "band_attenuation",
                    ValidityTag.OFF_MANIFOLD, dose=float(dose),
                    parameter=f"band={band};frequencies={minimum}-{maximum}",
                    matched_control="other_equal-dose Fourier bands",
                )
            )
    for dose in config["fourier_doses"]:
        specs.append(
            PerturbationSpec(
                f"scale_non_dc_d{dose:g}", "amplitude_scaling",
                ValidityTag.OFF_MANIFOLD, dose=float(dose),
                parameter=f"non_dc_factor={1-float(dose):g}",
                matched_control="band attenuation at equal dose",
            )
        )
    for channel, name in enumerate(CHANNEL_NAMES):
        specs.append(
            PerturbationSpec(
                f"replace_channel_{channel}_{name}", "channel_replacement",
                ValidityTag.OFF_MANIFOLD, parameter=f"channel={channel};name={name}",
                matched_control="same class/gradient matching for every channel",
            )
        )
    return specs


def _seed(config: dict[str, Any], spec: PerturbationSpec) -> int:
    # Block lengths intentionally share a common random-number stream within a
    # replicate, reducing realization noise in the length-scale comparison.
    # Phase variants also share a stream, and phase_scramble maps column zero of
    # the full tensor to the common endpoint for a genuinely matched contrast.
    family = (
        "matched_phase_scramble"
        if spec.family in ("common_phase_scramble", "channel_phase_scramble")
        else spec.family
    )
    family_offset = sum(
        (index + 1) * ord(value) for index, value in enumerate(family)
    )
    return int(config["seed"]) + family_offset + 1009 * spec.replicate


def _transform(
    spec: PerturbationSpec,
    geometry: torch.Tensor,
    panel: InferenceData,
    metadata: dict[str, np.ndarray],
    varied_backgrounds: ReferenceBackgrounds,
    fixed_backgrounds: ReferenceBackgrounds,
    config: dict[str, Any],
) -> torch.Tensor:
    seed = _seed(config, spec)
    if spec.family == "joint_shift":
        return torch.roll(geometry, shifts=32, dims=1)
    if spec.family == "random_joint_shift":
        return random_joint_shift(geometry, seed=seed, paired_halves=True)
    if spec.family == "parity":
        return stellarator_parity(geometry)
    if spec.family == "wrong_parity":
        return reverse_parallel(geometry)
    if spec.family == "joint_permutation":
        return joint_permutation(geometry, seed=seed, paired_halves=True)
    if spec.family == "block_permutation":
        length = int(spec.parameter.split("=")[1])
        return block_permutation(
            geometry, length, seed=seed, paired_halves=True
        )
    if spec.family == "independent_shift":
        return independent_channel_shifts(
            geometry, seed=seed, paired_halves=True
        )
    if spec.family == "common_phase_scramble":
        return phase_scramble(
            geometry,
            seed=seed,
            independent_channels=False,
            paired_halves=True,
        )
    if spec.family == "channel_phase_scramble":
        return phase_scramble(
            geometry,
            seed=seed,
            independent_channels=True,
            paired_halves=True,
        )
    if spec.family == "band_attenuation":
        band = spec.parameter.split(";")[0].split("=")[1]
        minimum, maximum = FOURIER_BANDS[band]
        return attenuate_fourier_band(
            geometry, minimum_frequency=minimum, maximum_frequency=maximum, dose=spec.dose
        )
    if spec.family == "amplitude_scaling":
        return scale_non_dc_amplitude(geometry, factor=1.0 - spec.dose)
    if spec.family == "channel_replacement":
        channel = int(spec.parameter.split(";")[0].split("=")[1])
        count = len(geometry) // 2
        gradients = np.column_stack((panel.a_over_lt.numpy(), panel.a_over_ln.numpy()))
        profiles = []
        for start, stop, backgrounds in (
            (0, count, varied_backgrounds), (count, 2 * count, fixed_backgrounds)
        ):
            profiles.append(
                backgrounds.conditional_channel_profile(
                    channel,
                    gradients[start:stop],
                    metadata["equilibrium_class"][start:stop],
                    neighbours=int(config["conditional_profile_neighbours"]),
                    source_row_ids=panel.row_indices[start:stop],
                )
            )
        return replace_channel(geometry, channel, torch.cat(profiles))
    raise ValueError(f"unknown perturbation family: {spec.family}")


def _predict(
    member: InvariantMember,
    geometry: torch.Tensor,
    panel: InferenceData,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output = np.empty((2, len(geometry)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(geometry), batch_size):
            stop = min(start + batch_size, len(geometry))
            values = geometry[start:stop].to(device)
            a_over_lt = panel.a_over_lt[start:stop].to(device)
            a_over_ln = panel.a_over_ln[start:stop].to(device)
            output[0, start:stop] = member.original(values, a_over_lt, a_over_ln).cpu().numpy()
            output[1, start:stop] = member.invariant(values, a_over_lt, a_over_ln).cpu().numpy()
    return output


def _initialize_predictions(
    path: Path,
    member_ids: tuple[str, ...],
    specs: list[PerturbationSpec],
    panel: InferenceData,
    hashes: dict[str, str],
) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(temporary, "w") as h5_file:
        h5_file.attrs["member_ids"] = json.dumps(member_ids)
        h5_file.attrs["spec_names"] = json.dumps([spec.name for spec in specs])
        h5_file.attrs["dataset_sha256"] = hashes["dataset"]
        h5_file.attrs["checkpoint_sha256"] = hashes["checkpoint"]
        h5_file.attrs["experiment_sha256"] = hashes["experiment"]
        h5_file.attrs["estimand"] = "native max(log Q, -2)"
        h5_file.create_dataset("member_id", data=np.asarray(member_ids, dtype="S"))
        h5_file.create_dataset("function", data=np.asarray(FUNCTION_NAMES, dtype="S"))
        h5_file.create_dataset("perturbation", data=np.asarray(["reference", *[s.name for s in specs]], dtype="S"))
        h5_file.create_dataset("row_id", data=panel.row_indices)
        h5_file.create_dataset("baseline_complete", data=np.zeros(len(member_ids), dtype=bool))
        h5_file.create_dataset("complete", data=np.zeros((len(member_ids), len(specs)), dtype=bool))
        prediction = h5_file.create_dataset(
            "prediction",
            shape=(len(member_ids), 2, len(specs) + 1, len(panel.row_indices)),
            dtype="f4",
            fillvalue=np.nan,
            chunks=(1, 1, 1, min(256, len(panel.row_indices))),
            compression="gzip",
            compression_opts=1,
        )
        prediction.attrs["axes"] = json.dumps(["member", "function", "perturbation", "sample"])
    temporary.replace(path)
    return h5py.File(path, "r+")


def _open_predictions(
    path: Path,
    member_ids: tuple[str, ...],
    specs: list[PerturbationSpec],
    panel: InferenceData,
    hashes: dict[str, str],
    resume: bool,
) -> h5py.File:
    if not path.exists():
        return _initialize_predictions(path, member_ids, specs, panel, hashes)
    if not resume:
        raise FileExistsError(f"{path} exists; pass --resume or select a new output directory")
    h5_file = h5py.File(path, "r+")
    checks = (
        (json.loads(h5_file.attrs["member_ids"]) == list(member_ids), "member IDs"),
        (json.loads(h5_file.attrs["spec_names"]) == [spec.name for spec in specs], "specs"),
        (np.array_equal(h5_file["row_id"][:], panel.row_indices), "panel rows"),
        (h5_file.attrs["dataset_sha256"] == hashes["dataset"], "dataset hash"),
        (h5_file.attrs["checkpoint_sha256"] == hashes["checkpoint"], "checkpoint hash"),
        (h5_file.attrs["experiment_sha256"] == hashes["experiment"], "experiment hash"),
    )
    failed = [name for passed, name in checks if not passed]
    if failed:
        h5_file.close()
        raise RuntimeError(f"resume artifact mismatch: {', '.join(failed)}")
    return h5_file


def _residual_scales(path: Path) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["entity"] != "ensemble_mean":
                result[(row["entity"], row["function"], row["stratum"])] = float(
                    row["residual_std"]
                )
    return result


def _channel_robust_scales(path: Path) -> np.ndarray:
    """Load S01's registered IQR/1.349 scales in network-channel order."""

    rows: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["channel"]), float(row["robust_sigma_iqr"])))
    rows.sort()
    if [channel for channel, _ in rows] != list(range(len(CHANNEL_NAMES))):
        raise ValueError("S01 channel scale registry does not cover channels 0..6")
    scales = np.asarray([scale for _, scale in rows], dtype=np.float64)
    if not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("S01 channel robust scales must be finite and positive")
    return scales


def _analysis_masks(
    panel: InferenceData,
    metadata: dict[str, np.ndarray],
    stable_threshold: float,
) -> dict[tuple[str, str], np.ndarray]:
    actual = panel.actual_log_heat_flux.numpy()
    masks: dict[tuple[str, str], np.ndarray] = {}
    for gradient_set in ("varied", "fixed"):
        gradient_mask = metadata["gradient_set"] == gradient_set
        masks[(gradient_set, "all")] = gradient_mask
        masks[(gradient_set, "stable_near_floor")] = gradient_mask & (
            actual <= stable_threshold
        )
        masks[(gradient_set, "unstable")] = gradient_mask & (
            actual > stable_threshold
        )
    return masks


def _robust_input_displacements(
    original: torch.Tensor,
    endpoint: torch.Tensor,
    spec: PerturbationSpec,
    masks: dict[tuple[str, str], np.ndarray],
    channel_scales: np.ndarray,
) -> dict[tuple[str, str, str], dict[str, float]]:
    """RMS and median-absolute doses after robust per-channel scaling."""

    if spec.family == "channel_replacement":
        channels = [int(spec.parameter.split(";")[0].split("=")[1])]
    else:
        channels = list(range(original.shape[2]))
    difference = endpoint[:, :, channels].numpy().astype(np.float64) - original[
        :, :, channels
    ].numpy().astype(np.float64)
    difference /= channel_scales[np.asarray(channels)][None, None, :]
    output: dict[tuple[str, str, str], dict[str, float]] = {}
    for (gradient_set, stratum), mask in masks.items():
        selected = difference[mask]
        output[(spec.name, gradient_set, stratum)] = {
            "rms": float(np.sqrt(np.mean(np.square(selected)))),
            "median_abs": float(np.median(np.abs(selected))),
        }
    return output


def _bootstrap_draws(mask: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return np.empty((replicates, 0), dtype=np.int64)
    generator = np.random.default_rng(seed)
    return indices[generator.integers(0, len(indices), size=(replicates, len(indices)))]


def _ladder_rows(
    predictions: np.ndarray,
    member_ids: tuple[str, ...],
    top_ids: set[str],
    specs: list[PerturbationSpec],
    panel: InferenceData,
    metadata: dict[str, np.ndarray],
    reference_residual_scales: dict[tuple[str, str, str], float],
    input_displacements: dict[tuple[str, str, str], dict[str, float]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    actual = panel.actual_log_heat_flux.numpy()
    masks = _analysis_masks(
        panel, metadata, float(config["stable_threshold_log_Q"])
    )
    for mask in masks.values():
        groups = metadata["equilibrium_file"][mask]
        if len(np.unique(groups)) != len(groups):
            raise RuntimeError("S03 bootstrap masks must contain one row per equilibrium_file")
    draws = {
        key: _bootstrap_draws(mask, int(config["bootstrap_replicates"]), int(config["seed"]) + index)
        for index, (key, mask) in enumerate(masks.items())
    }
    rows: list[dict[str, Any]] = []
    reference = predictions[:, :, 0]
    for spec_index, spec in enumerate(specs, start=1):
        for member_index, member_id in enumerate(member_ids):
            if spec.member_scope == "top10" and member_id not in top_ids:
                continue
            for function_index, function_name in enumerate(FUNCTION_NAMES):
                difference = predictions[member_index, function_index, spec_index].astype(np.float64) - reference[
                    member_index, function_index
                ].astype(np.float64)
                for (gradient_set, stratum), mask in masks.items():
                    selected = difference[mask]
                    rms = float(np.sqrt(np.mean(np.square(selected))))
                    normalization: float | str = ""
                    reference_normalization: float | str = ""
                    ratio: float | str = ""
                    doses = input_displacements[(spec.name, gradient_set, stratum)]
                    rms_dose = doses["rms"]
                    median_abs_dose = doses["median_abs"]
                    ratio_per_dose: float | str = ""
                    ratio_per_median_abs_dose: float | str = ""
                    if gradient_set == "varied":
                        varied_residual = reference[member_index, function_index].astype(
                            np.float64
                        ) - actual
                        normalization = float(np.std(varied_residual[mask], ddof=1))
                        reference_normalization = reference_residual_scales[
                            (member_id, function_name, stratum)
                        ]
                        ratio = rms / float(normalization)
                        ratio_per_dose = float(ratio) / rms_dose if rms_dose > 0 else ""
                        ratio_per_median_abs_dose = (
                            float(ratio) / median_abs_dose
                            if median_abs_dose > 0
                            else ""
                        )
                    ci_lower: float | str = ""
                    ci_upper: float | str = ""
                    if member_id in top_ids and len(selected):
                        samples = np.sqrt(np.mean(np.square(difference[draws[(gradient_set, stratum)]]), axis=1))
                        if gradient_set == "varied":
                            sampled_residual = varied_residual[
                                draws[(gradient_set, stratum)]
                            ]
                            sampled_denominator = np.std(sampled_residual, axis=1, ddof=1)
                            samples /= sampled_denominator
                        ci_lower, ci_upper = [float(value) for value in np.quantile(samples, (0.025, 0.975))]
                    rows.append(
                        {
                            "member_id": member_id,
                            "validation_cohort": "stored_validation_top10" if member_id in top_ids else "all100_non_top10",
                            "function": function_name,
                            "perturbation": spec.name,
                            "family": spec.family,
                            "parameter": spec.parameter,
                            "dose": spec.dose,
                            "replicate": spec.replicate,
                            "validity_tag": spec.validity.value,
                            "matched_control": spec.matched_control,
                            "gradient_set": gradient_set,
                            "stratum": stratum,
                            "n": int(mask.sum()),
                            "mean_signed_change": float(np.mean(selected)),
                            "mean_absolute_change": float(np.mean(np.abs(selected))),
                            "rms_change": rms,
                            "panel_residual_std": normalization,
                            "s02_reference_residual_std": reference_normalization,
                            "rms_change_over_residual_std": ratio,
                            "robust_input_displacement_rms": rms_dose,
                            "effect_per_robust_input_rms": ratio_per_dose,
                            "robust_input_displacement_median_abs": median_abs_dose,
                            "effect_per_robust_input_median_abs": ratio_per_median_abs_dose,
                            "bootstrap_ci95_lower": ci_lower,
                            "bootstrap_ci95_upper": ci_upper,
                            "bootstrap_unit": "equilibrium_files",
                        }
                    )
    return rows


def _support_rows(
    support: ScaledPCASupport,
    specs: list[PerturbationSpec],
    panel: InferenceData,
    metadata: dict[str, np.ndarray],
    varied_backgrounds: ReferenceBackgrounds,
    fixed_backgrounds: ReferenceBackgrounds,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    varied_count = len(panel.geometry) // 2
    original = panel.geometry[:varied_count]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        endpoint = _transform(
            spec, panel.geometry, panel, metadata, varied_backgrounds, fixed_backgrounds, config
        )[:varied_count]
        # Report a common interpolation grid for every endpoint.  For an
        # explicitly dosed Fourier operator the endpoint itself is already at
        # that registered strength; path_dose remains the interpolation to it.
        for path_dose in config["support_path_doses"]:
            values = interpolate_geometry(original, endpoint, float(path_dose)).numpy()
            scores = support.score(values)
            rows.append(
                {
                    "perturbation": spec.name,
                    "family": spec.family,
                    "registered_dose": spec.dose,
                    "path_dose": float(path_dose),
                    "replicate": spec.replicate,
                    "validity_tag": spec.validity.value,
                    "n": len(values),
                    "reconstruction_rms_median": float(np.median(scores["reconstruction_rms"])),
                    "nearest_distance_median": float(np.median(scores["nearest_distance"])),
                    "warning_score_median": float(np.median(scores["warning_score"])),
                    "warning_score_q90": float(np.quantile(scores["warning_score"], 0.9)),
                    "fraction_outside_heldout_central_95pct": float(
                        np.mean(scores["warning_score"] > 0.95)
                    ),
                    "warning_tail_definition": "two_sided_p_below_0.025_or_above_0.975",
                    "cyclic_anchor": "argmax_joint_robust_standardized_energy",
                    "interpretation": "warning_only_not_physical_validity",
                }
            )
    return rows


def _toy_checks(seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    geometry = torch.randn(32, 96, 7, generator=generator)
    gradients = torch.zeros(len(geometry))
    joint = joint_permutation(geometry, seed=seed + 1)
    blocked = block_permutation(geometry, 8, seed=seed + 1)
    shifted = random_joint_shift(geometry, seed=seed + 1)
    independent = independent_channel_shifts(geometry, seed=seed + 1)
    common_phase = phase_scramble(
        geometry, seed=seed + 1, independent_channels=False
    )
    channel_phase = phase_scramble(
        geometry, seed=seed + 1, independent_channels=True
    )
    permutation_toy = PeriodicPermutationToy()
    reference = permutation_toy(geometry, gradients, gradients)
    joint_change = float(torch.sqrt(torch.mean(torch.square(permutation_toy(joint, gradients, gradients) - reference))))
    block_change = float(torch.sqrt(torch.mean(torch.square(permutation_toy(blocked, gradients, gradients) - reference))))
    shift_change = float(torch.sqrt(torch.mean(torch.square(permutation_toy(shifted, gradients, gradients) - reference))))
    independent_change = float(torch.sqrt(torch.mean(torch.square(permutation_toy(independent, gradients, gradients) - reference))))
    colocation = ColocationToy()
    coloc_reference = colocation(geometry, gradients, gradients)
    coloc_joint = float(torch.sqrt(torch.mean(torch.square(colocation(joint, gradients, gradients) - coloc_reference))))
    coloc_block = float(torch.sqrt(torch.mean(torch.square(colocation(blocked, gradients, gradients) - coloc_reference))))
    coloc_shift = float(torch.sqrt(torch.mean(torch.square(colocation(shifted, gradients, gradients) - coloc_reference))))
    coloc_independent = float(torch.sqrt(torch.mean(torch.square(colocation(independent, gradients, gradients) - coloc_reference))))
    coloc_common_phase = float(torch.sqrt(torch.mean(torch.square(colocation(common_phase, gradients, gradients) - coloc_reference))))
    coloc_channel_phase = float(torch.sqrt(torch.mean(torch.square(colocation(channel_phase, gradients, gradients) - coloc_reference))))
    signal_geometry = torch.zeros(4, 96, 7)
    signal_geometry[:, :, 4] = torch.sin(2 * torch.pi * 3 * torch.arange(96) / 96)
    fourier = FourierBandToy(channel=4, band=3)
    band_reference = fourier(signal_geometry, torch.zeros(4), torch.zeros(4))
    band_relevant = fourier(
        attenuate_fourier_band(signal_geometry, minimum_frequency=1, maximum_frequency=4, dose=1),
        torch.zeros(4), torch.zeros(4)
    )
    band_control = fourier(
        attenuate_fourier_band(signal_geometry, minimum_frequency=17, maximum_frequency=48, dose=1),
        torch.zeros(4), torch.zeros(4)
    )
    relevant_change = float(torch.mean(torch.abs(band_relevant - band_reference)))
    control_change = float(torch.mean(torch.abs(band_control - band_reference)))
    amplitude_scaled = fourier(
        scale_non_dc_amplitude(signal_geometry, factor=0.5),
        torch.zeros(4),
        torch.zeros(4),
    )
    amplitude_change = float(torch.mean(torch.abs(amplitude_scaled - band_reference)))
    phase_magnitude_error = float(
        torch.max(
            torch.abs(
                torch.fft.rfft(channel_phase, dim=1).abs()
                - torch.fft.rfft(geometry, dim=1).abs()
            )
        )
    )
    replacement_profile = geometry[:, :, 4].roll(shifts=13, dims=1)
    replaced = replace_channel(geometry, 4, replacement_profile)
    replacement_untouched_error = float(
        torch.max(torch.abs(replaced[:, :, [0, 1, 2, 3, 5, 6]] - geometry[:, :, [0, 1, 2, 3, 5, 6]]))
    )
    parity_roundtrip_error = float(
        torch.max(torch.abs(stellarator_parity(stellarator_parity(geometry)) - geometry))
    )
    reversal_roundtrip_error = float(
        torch.max(torch.abs(reverse_parallel(reverse_parallel(geometry)) - geometry))
    )
    first_mask = wrapped_window_mask(96, start=92, length=11)
    shifted_mask = wrapped_window_mask(96, start=3, length=11)
    checks = {
        "permutation_toy_joint_rms": joint_change,
        "permutation_toy_block_rms": block_change,
        "permutation_toy_joint_shift_rms": shift_change,
        "permutation_toy_independent_rms": independent_change,
        "colocation_toy_joint_rms": coloc_joint,
        "colocation_toy_block_rms": coloc_block,
        "colocation_toy_joint_shift_rms": coloc_shift,
        "colocation_toy_independent_rms": coloc_independent,
        "colocation_toy_common_phase_rms": coloc_common_phase,
        "colocation_toy_channel_phase_rms": coloc_channel_phase,
        "fourier_toy_relevant_change": relevant_change,
        "fourier_toy_control_change": control_change,
        "fourier_toy_amplitude_scaling_change": amplitude_change,
        "phase_scramble_amplitude_max_error": phase_magnitude_error,
        "replacement_untouched_channel_max_error": replacement_untouched_error,
        "stellarator_parity_roundtrip_max_error": parity_roundtrip_error,
        "wrong_parity_roundtrip_max_error": reversal_roundtrip_error,
        "wrapped_window_no_boundary_artifact": bool(
            first_mask.sum() == shifted_mask.sum() == 11
            and torch.equal(torch.roll(first_mask, shifts=7), shifted_mask)
        ),
    }
    checks["passed"] = bool(
        joint_change < 1e-5
        and block_change < 1e-5
        and shift_change < 1e-5
        and independent_change > 0.1
        and coloc_joint < 1e-5
        and coloc_block < 1e-5
        and coloc_shift < 1e-5
        and coloc_independent > 0.1
        and coloc_common_phase < 1e-5
        and coloc_channel_phase > 0.05
        and relevant_change > control_change + 1
        and amplitude_change > control_change + 1
        and phase_magnitude_error < 2e-4
        and replacement_untouched_error == 0
        and parity_roundtrip_error == 0
        and reversal_roundtrip_error == 0
        and checks["wrapped_window_no_boundary_artifact"]
    )
    return checks


def _compact_ladder_summary(
    rows: list[dict[str, Any]], top_ids: set[str]
) -> list[dict[str, Any]]:
    """Generate the committed headline table directly from manifest-hashed rows."""

    canonical = [
        row
        for row in rows
        if row["function"] == "invariant_tilde_f"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "all"
        and row["replicate"] == 0
    ]

    def summarize(
        section: str,
        item: str,
        parameter: str,
        selected: list[dict[str, Any]],
    ) -> dict[str, Any]:
        top = [row for row in selected if row["member_id"] in top_ids]
        selected_member_count = len({row["member_id"] for row in selected})
        expanded_members = (
            selected if selected_member_count > len({row["member_id"] for row in top}) else []
        )

        def quantile(values: list[float], probability: float) -> float | str:
            return float(np.quantile(values, probability)) if values else ""

        top_effect = [float(row["rms_change_over_residual_std"]) for row in top]
        expanded_effect = [
            float(row["rms_change_over_residual_std"]) for row in expanded_members
        ]
        dose_comparable = section in (
            "fourier_full_attenuation",
            "channel_replacement",
        )
        normalized = (
            [float(row["effect_per_robust_input_rms"]) for row in top]
            if dose_comparable
            else []
        )
        median_normalized = (
            [float(row["effect_per_robust_input_median_abs"]) for row in top]
            if dose_comparable
            else []
        )
        bootstrap_lower = [float(row["bootstrap_ci95_lower"]) for row in top]
        bootstrap_upper = [float(row["bootstrap_ci95_upper"]) for row in top]
        return {
            "section": section,
            "item": item,
            "parameter": parameter,
            "robust_input_displacement_rms": (
                float(
                    np.median(
                        [float(row["robust_input_displacement_rms"]) for row in top]
                    )
                )
                if dose_comparable and top
                else ""
            ),
            "robust_dose_comparison_scope": (
                "within_section_sensitivity_analysis"
                if dose_comparable and top
                else "not_cross_family_comparable"
            ),
            "robust_input_displacement_median_abs": (
                float(
                    np.median(
                        [
                            float(row["robust_input_displacement_median_abs"])
                            for row in top
                        ]
                    )
                )
                if dose_comparable and top
                else ""
            ),
            "top10_q10": quantile(top_effect, 0.1),
            "top10_median": quantile(top_effect, 0.5),
            "top10_q90": quantile(top_effect, 0.9),
            "top10_range_kind": "member_spread_not_sampling_uncertainty",
            "top10_member_bootstrap_ci95_lower_median": quantile(
                bootstrap_lower, 0.5
            ),
            "top10_member_bootstrap_ci95_upper_median": quantile(
                bootstrap_upper, 0.5
            ),
            "top10_median_effect_per_robust_input_rms": quantile(normalized, 0.5),
            "top10_median_effect_per_robust_input_median_abs": quantile(
                median_normalized, 0.5
            ),
            "expanded_member_count": selected_member_count if expanded_members else "",
            "expanded_cohort_q10": quantile(expanded_effect, 0.1),
            "expanded_cohort_median": quantile(expanded_effect, 0.5),
            "expanded_cohort_q90": quantile(expanded_effect, 0.9),
        }

    output: list[dict[str, Any]] = []
    for family in (
        "joint_shift",
        "random_joint_shift",
        "parity",
        "wrong_parity",
        "joint_permutation",
        "independent_shift",
        "common_phase_scramble",
        "channel_phase_scramble",
    ):
        output.append(
            summarize(
                "ladder",
                family,
                "replicate=0",
                [row for row in canonical if row["family"] == family],
            )
        )
    for length in (2, 4, 8, 16, 32):
        output.append(
            summarize(
                (
                    "three_block_reversal_control"
                    if length == 32
                    else "block_length_spectrum"
                ),
                f"L{length}",
                (
                    "replicate=0;three_blocks;reversal_rotation_class"
                    if length == 32
                    else "replicate=0;non_cyclic_orders_only"
                ),
                [
                    row
                    for row in canonical
                    if row["family"] == "block_permutation"
                    and row["parameter"] == f"block_length={length}"
                ],
            )
        )
    for band in FOURIER_BANDS:
        output.append(
            summarize(
                "fourier_full_attenuation",
                band,
                "dose=1",
                [
                    row
                    for row in canonical
                    if row["family"] == "band_attenuation"
                    and str(row["parameter"]).startswith(f"band={band};")
                    and float(row["dose"]) == 1.0
                ],
            )
        )
    for channel, name in enumerate(CHANNEL_NAMES):
        output.append(
            summarize(
                "channel_replacement",
                name,
                f"channel={channel}",
                [
                    row
                    for row in canonical
                    if row["family"] == "channel_replacement"
                    and str(row["parameter"]).startswith(f"channel={channel};")
                ],
            )
        )
    return output


def _contrast_summary_rows(
    rows: list[dict[str, Any]], top_ids: set[str]
) -> list[dict[str, Any]]:
    """Summarize signed paired member contrasts between registered controls."""

    canonical = [
        row
        for row in rows
        if row["member_id"] in top_ids
        and row["function"] == "invariant_tilde_f"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "all"
    ]

    def by_member(
        family: str, replicate: int, parameter: str | None = None
    ) -> dict[str, float]:
        return {
            str(row["member_id"]): float(row["rms_change_over_residual_std"])
            for row in canonical
            if row["family"] == family
            and int(row["replicate"]) == replicate
            and (parameter is None or row["parameter"] == parameter)
        }

    registered = (
        (
            "joint_permutation_minus_random_joint_shift",
            "joint_permutation",
            None,
            "random_joint_shift",
            "exact_symmetry_diagnostic_not_effect_size_control",
        ),
        (
            "independent_shift_minus_random_joint_shift",
            "independent_shift",
            None,
            "random_joint_shift",
            "exact_symmetry_diagnostic_not_effect_size_control",
        ),
        (
            "channel_phase_minus_common_phase",
            "channel_phase_scramble",
            None,
            "common_phase_scramble",
            "registered_matched_control",
        ),
        (
            "joint_permutation_minus_independent_shift",
            "joint_permutation",
            None,
            "independent_shift",
            "ordering_vs_colocation_comparison",
        ),
        *(
            (
                f"block_permutation_L{length}_minus_random_joint_shift",
                "block_permutation",
                f"block_length={length}",
                "random_joint_shift",
                "exact_symmetry_diagnostic_not_effect_size_control",
            )
            for length in (2, 4, 8, 16, 32)
        ),
        (
            "wrong_parity_minus_stellarator_parity",
            "wrong_parity",
            None,
            "parity",
            "registered_matched_control",
        ),
    )
    output: list[dict[str, Any]] = []
    for replicate in sorted({int(row["replicate"]) for row in canonical}):
        for (
            name,
            treatment_family,
            treatment_parameter,
            control_family,
            contrast_type,
        ) in registered:
            treatment = by_member(
                treatment_family, replicate, treatment_parameter
            )
            control = by_member(control_family, replicate)
            members = sorted(set(treatment).intersection(control))
            if not members:
                continue
            differences = np.asarray(
                [treatment[member] - control[member] for member in members]
            )
            output.append(
                {
                    "contrast": name,
                    "contrast_type": contrast_type,
                    "replicate": replicate,
                    "members": len(members),
                    "treatment_family": treatment_family,
                    "treatment_parameter": treatment_parameter or "",
                    "control_family": control_family,
                    "treatment_member_median": float(
                        np.median([treatment[member] for member in members])
                    ),
                    "control_member_median": float(
                        np.median([control[member] for member in members])
                    ),
                    "paired_difference_q10": float(np.quantile(differences, 0.1)),
                    "paired_difference_median": float(np.median(differences)),
                    "paired_difference_q90": float(np.quantile(differences, 0.9)),
                    "members_positive": int(np.sum(differences > 0)),
                    "units": "panel_member_residual_standard_deviations",
                }
            )
    return output


def _plot_dose(path: Path, rows: list[dict[str, Any]], top_ids: set[str]) -> None:
    selected = [
        row for row in rows
        if row["member_id"] in top_ids
        and row["function"] == "invariant_tilde_f"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "all"
        and row["family"] in ("band_attenuation", "amplitude_scaling")
    ]
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    for family in ("band_attenuation", "amplitude_scaling"):
        family_rows = [row for row in selected if row["family"] == family]
        parameters = (
            sorted({str(row["parameter"]).split(";")[0] for row in family_rows})
            if family == "band_attenuation"
            else ["all_non_dc"]
        )
        for parameter in parameters:
            values = (
                [row for row in family_rows if str(row["parameter"]).split(";")[0] == parameter]
                if family == "band_attenuation"
                else family_rows
            )
            doses = sorted({float(row["dose"]) for row in values})
            medians = [np.median([float(row["rms_change_over_residual_std"]) for row in values if float(row["dose"]) == dose]) for dose in doses]
            axis.plot(doses, medians, marker="o", label=parameter)
    axis.set_xlabel("Registered attenuation dose")
    axis.set_ylabel("Median top-10 RMS change / panel residual std")
    axis.set_title("Phase-preserving Fourier dose response")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _publish(paths: list[Path], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for source in paths:
        temporary = directory / f"{source.name}.tmp"
        shutil.copy2(source, temporary)
        temporary.replace(directory / source.name)


def run(config: dict[str, Any], args: argparse.Namespace) -> Path:
    resolved = _resolve(config, args)
    set_deterministic_seed(int(resolved["seed"]))
    dataset = Path(resolved["dataset"]).resolve()
    checkpoint = Path(resolved["checkpoint"]).resolve()
    cohorts_path = Path(resolved["cohorts"]).resolve()
    cohorts = json.loads(cohorts_path.read_text(encoding="utf-8"))
    output_dir = (args.output_dir or Path("output/xai/S03") / str(resolved["run_id"])).resolve()
    artifacts = RunArtifacts(output_dir)
    panel, metadata = _load_panel(
        dataset,
        cohorts,
        int(resolved["panel_varied_rows"]),
        float(resolved["stable_threshold_log_Q"]),
    )
    support_varied, support_fixed, support_metadata = _load_support_reference(
        dataset,
        cohorts,
        panel.row_indices[: int(resolved["panel_varied_rows"])],
        int(resolved["support_reference_rows"]),
        int(resolved["seed"]) + 41,
    )
    panel_groups = set(metadata["equilibrium_file"])
    support_groups = set(support_metadata["equilibrium_file"])
    if panel_groups.intersection(support_groups):
        raise RuntimeError("support/background equilibria overlap the analysis panel")
    split = int(len(support_varied.geometry) * float(resolved["support_fit_fraction"]))
    if not 2 <= split < len(support_varied.geometry):
        raise ValueError("support_fit_fraction leaves an empty fit or held-out set")
    support = ScaledPCASupport.fit(
        support_varied.geometry[:split].numpy(),
        support_varied.geometry[split:].numpy(),
        components=int(resolved["support_components"]),
    )
    gradients_varied = np.column_stack((support_varied.a_over_lt.numpy(), support_varied.a_over_ln.numpy()))
    gradients_fixed = np.column_stack((support_fixed.a_over_lt.numpy(), support_fixed.a_over_ln.numpy()))
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

    # PLAN S03 requires analytic controls before any real-member inference.
    toy_checks = _toy_checks(int(resolved["seed"]) + 99)
    if not toy_checks["passed"]:
        raise RuntimeError(f"pre-inference toy gate failed: {toy_checks}")

    ensemble = load_ensemble(checkpoint, device=str(resolved["device"]))
    registered_all = tuple(cohorts["member_cohorts"]["all_100"])
    registered_top = tuple(cohorts["member_cohorts"]["stored_validation_top_10"])
    cap = int(resolved.get("members", len(registered_all)))
    member_ids = tuple(registered_all[:cap])
    top_ids = set(registered_top).intersection(member_ids)
    if not member_ids or not top_ids:
        raise ValueError("member cap removed the registered top-validation cohort")
    index_by_id = {member_id: index for index, member_id in enumerate(ensemble.member_ids)}
    if set(member_ids).difference(index_by_id):
        raise RuntimeError("registered member IDs are absent from checkpoint")
    specs = _specs(resolved)
    if "families" in resolved:
        requested_families = set(resolved["families"])
        available_families = {spec.family for spec in specs}
        unknown = requested_families.difference(available_families)
        if unknown:
            raise ValueError(f"unknown perturbation families: {sorted(unknown)}")
        specs = [spec for spec in specs if spec.family in requested_families]
    masks = _analysis_masks(
        panel, metadata, float(resolved["stable_threshold_log_Q"])
    )
    channel_scales = _channel_robust_scales(Path(resolved["s01_channel_scales"]))
    input_displacements: dict[tuple[str, str, str], dict[str, float]] = {}
    endpoint_hashes: dict[str, str] = {}
    deterministic_specs: dict[str, bool] = {}
    for spec in specs:
        endpoint = _transform(
            spec,
            panel.geometry,
            panel,
            metadata,
            varied_backgrounds,
            fixed_backgrounds,
            resolved,
        )
        repeated = _transform(
            spec,
            panel.geometry,
            panel,
            metadata,
            varied_backgrounds,
            fixed_backgrounds,
            resolved,
        )
        deterministic_specs[spec.name] = bool(torch.equal(endpoint, repeated))
        if not deterministic_specs[spec.name]:
            raise RuntimeError(f"full-batch fixed-seed repeat failed: {spec.name}")
        endpoint_hashes[spec.name] = hashlib.sha256(
            np.ascontiguousarray(endpoint.numpy()).tobytes()
        ).hexdigest()
        input_displacements.update(
            _robust_input_displacements(
                panel.geometry, endpoint, spec, masks, channel_scales
            )
        )
    experiment_payload = {
        "config": resolved,
        "specs": [
            {
                "name": spec.name,
                "family": spec.family,
                "validity": spec.validity.value,
                "dose": spec.dose,
                "replicate": spec.replicate,
                "parameter": spec.parameter,
                "member_scope": spec.member_scope,
                "matched_control": spec.matched_control,
            }
            for spec in specs
        ],
        "script_sha256": sha256_file(__file__),
        "perturbations_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "itg_nn/xai/perturbations.py"
        ),
        "cohorts_sha256": sha256_file(cohorts_path),
        "channel_scales_sha256": sha256_file(Path(resolved["s01_channel_scales"])),
    }
    hashes = {
        "dataset": sha256_file(dataset),
        "checkpoint": sha256_file(checkpoint),
        "experiment": hashlib.sha256(
            json.dumps(experiment_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    prediction_path = output_dir / "predictions.h5"
    with _open_predictions(
        prediction_path, member_ids, specs, panel, hashes, bool(args.resume)
    ) as prediction_file:
        for member_index, member_id in enumerate(member_ids):
            if not bool(prediction_file["baseline_complete"][member_index]):
                member = InvariantMember(ensemble.models[index_by_id[member_id]])
                started = time.monotonic()
                prediction_file["prediction"][member_index, :, 0] = _predict(
                    member, panel.geometry, panel, int(resolved["batch_size"]), ensemble.device
                )
                prediction_file["baseline_complete"][member_index] = True
                prediction_file.flush()
                print(
                    f"baseline {member_index + 1}/{len(member_ids)} {member_id} "
                    f"{time.monotonic() - started:.1f}s",
                    flush=True,
                )
        for spec_index, spec in enumerate(specs):
            geometry = _transform(
                spec, panel.geometry, panel, metadata, varied_backgrounds, fixed_backgrounds, resolved
            )
            applicable = [
                index for index, member_id in enumerate(member_ids)
                if spec.member_scope == "all100" or member_id in top_ids
            ]
            pending = [index for index in applicable if not bool(prediction_file["complete"][index, spec_index])]
            if not pending:
                continue
            started = time.monotonic()
            for member_index in pending:
                member_id = member_ids[member_index]
                member = InvariantMember(ensemble.models[index_by_id[member_id]])
                prediction_file["prediction"][member_index, :, spec_index + 1] = _predict(
                    member, geometry, panel, int(resolved["batch_size"]), ensemble.device
                )
                prediction_file["complete"][member_index, spec_index] = True
                prediction_file.flush()
            print(
                f"perturbation {spec_index + 1}/{len(specs)} {spec.name}: "
                f"{len(pending)} members in {time.monotonic() - started:.1f}s",
                flush=True,
            )
        predictions = prediction_file["prediction"][:]
        complete = prediction_file["complete"][:]

    support_rows = _support_rows(
        support, specs, panel, metadata, varied_backgrounds, fixed_backgrounds, resolved
    )
    reference_residual_scales = _residual_scales(Path(resolved["s02_accuracy"]))
    ladder_rows = _ladder_rows(
        predictions,
        member_ids,
        top_ids,
        specs,
        panel,
        metadata,
        reference_residual_scales,
        input_displacements,
        resolved,
    )
    spec_index_by_name = {spec.name: index + 1 for index, spec in enumerate(specs)}
    exact_names = (["joint_shift_32"] if "joint_shift_32" in spec_index_by_name else []) + [
        spec.name for spec in specs if spec.family == "random_joint_shift"
    ]
    exact_errors: dict[str, float] = {}
    exact_allclose: dict[str, bool] = {}
    exact_atol = float(resolved["exact_atol"])
    exact_rtol = float(resolved["exact_rtol"])
    for name in exact_names:
        index = spec_index_by_name[name]
        applicable = np.isfinite(predictions[:, 1, index]).all(axis=1)
        key = f"invariant_tilde_f:{name}"
        endpoint_values = predictions[applicable, 1, index]
        reference_values = predictions[applicable, 1, 0]
        exact_errors[key] = float(
            np.max(np.abs(endpoint_values - reference_values))
        )
        exact_allclose[key] = bool(
            np.allclose(
                endpoint_values,
                reference_values,
                atol=exact_atol,
                rtol=exact_rtol,
            )
        )
    if "joint_shift_32" in spec_index_by_name:
        index = spec_index_by_name["joint_shift_32"]
        key = "original_f:joint_shift_32"
        endpoint_values = predictions[:, 0, index]
        reference_values = predictions[:, 0, 0]
        exact_errors[key] = float(
            np.max(np.abs(endpoint_values - reference_values))
        )
        exact_allclose[key] = bool(
            np.allclose(
                endpoint_values,
                reference_values,
                atol=exact_atol,
                rtol=exact_rtol,
            )
        )
    exact_passed = all(exact_allclose.values())
    if not exact_passed:
        raise RuntimeError(
            f"registered exact-symmetry checks failed: {exact_errors}"
        )

    # Receptive-field tied lengths are published for every selected member.
    receptive_rows: list[dict[str, Any]] = []
    for member_id in member_ids:
        model = ensemble.models[index_by_id[member_id]]
        blocks = receptive_field_blocks(tuple(layer.kernel_size[0] for layer in model.conv_layers))
        receptive_rows.append(
            {
                "member_id": member_id,
                "window_lengths": ";".join(
                    str(value) for value in member_window_lengths(
                        [block.unique_periodic_positions for block in blocks]
                    )
                ),
                "receptive_field_positions_by_block": ";".join(
                    str(block.unique_periodic_positions) for block in blocks
                ),
            }
        )

    baseline_registry = {
        "all_zero_default_forbidden": True,
        "estimand": "native max(log Q, -2)",
        "available": {
            "robust_constant": "per-channel reference median expanded over z",
            "matched_observed": "observed robust-gradient nearest neighbour within equilibrium class; source row excluded",
            "medoid": "observed geometry nearest the robust profile center, optionally within class",
            "low_pass": "rFFT truncation of the supplied analysed or background input, retaining DC through the registered cutoff",
            "conditional_channel_profile": "pointwise median profile of nearest class/gradient-matched observed rows; tied gradients reduce explicitly to class-only matching",
        },
        "production_real_data_usage": {
            "conditional_channel_profile": "used for S03 channel replacement",
            "robust_constant_matched_observed_medoid_low_pass": "API-only in S03; first real consumers are downstream steps",
        },
        "support_warning": "robust per-channel scaling + ordinary SVD PCA + held-out nearest-neighbour distance; PCA components remain outlier-sensitive; two-sided tails; not proof of physical validity",
        "cyclic_anchor": "argmax joint robust-standardized energy across all channels",
        "panel_equilibrium_overlap": 0,
        "support_fit_rows": split,
        "support_heldout_rows": len(support_varied.geometry) - split,
        "support_components": int(resolved["support_components"]),
    }
    top_canonical = [
        row for row in ladder_rows
        if row["member_id"] in top_ids
        and row["function"] == "invariant_tilde_f"
        and row["gradient_set"] == "varied"
        and row["stratum"] == "all"
        and row["replicate"] == 0
    ]
    family_summary = {
        family: {
            "median_rms_over_residual_std": float(
                np.median([
                    float(row["rms_change_over_residual_std"])
                    for row in top_canonical if row["family"] == family
                ])
            ),
            "members_x_entries": int(sum(row["family"] == family for row in top_canonical)),
        }
        for family in sorted({str(row["family"]) for row in top_canonical})
    }
    summary = {
        "mode": resolved["mode"],
        "estimand": "native max(log Q, -2)",
        "canonical_function": "invariant_tilde_f",
        "cohort": {
            "panel_varied": int(resolved["panel_varied_rows"]),
            "panel_fixed_paired": int(resolved["panel_varied_rows"]),
            "equilibrium_unique_per_gradient_set": int(resolved["panel_varied_rows"]),
            "members": len(member_ids),
            "top_validation_members": len(top_ids),
        },
        "spec_count": len(specs),
        "all100_spec_count": int(sum(spec.member_scope == "all100" for spec in specs)),
        "checks": {
            "toy_controls": toy_checks,
            "toy_gate_timing": "before_real_member_inference",
            "exact_symmetry_max_absolute_errors": exact_errors,
            "exact_symmetry_allclose_atol_rtol": exact_allclose,
            "exact_symmetry_passed": exact_passed,
            "deterministic_seeded_operators": bool(all(deterministic_specs.values())),
            "full_batch_repeat_by_spec": deterministic_specs,
            "every_spec_has_validity_tag": all(bool(spec.validity.value) for spec in specs),
            "support_at_every_spec_and_path_dose": len(support_rows)
            == len(specs) * len(resolved["support_path_doses"]),
            "prediction_completeness": bool(
                all(
                    complete[member_index, spec_index]
                    for spec_index, spec in enumerate(specs)
                    for member_index, member_id in enumerate(member_ids)
                    if spec.member_scope == "all100" or member_id in top_ids
                )
            ),
        },
        "top10_canonical_varied_all_family_summary": family_summary,
        "bootstrap": {
            "unit": "equilibrium_files",
            "replicates": int(resolved["bootstrap_replicates"]),
            "scope": "member-level intervals for registered stored-validation top cohort; numerator and panel residual denominator recomputed together",
        },
        "normalization": {
            "point_estimate": "same frozen S01 panel and matching member/function/flux stratum",
            "comparison_only": "S02 full-reference residual std retained as a separate ladder.csv column",
            "input_dose": "S01 IQR/1.349 channel scales followed by both RMS and median-absolute aggregation; replacement dose uses only the edited channel; normalized rankings are sensitivity analyses",
        },
        "support": baseline_registry,
    }

    ladder_path = artifacts.write_text("ladder.csv", _csv_text(ladder_rows))
    ladder_summary_path = artifacts.write_text(
        "ladder_summary.csv", _csv_text(_compact_ladder_summary(ladder_rows, top_ids))
    )
    contrast_rows = _contrast_summary_rows(ladder_rows, top_ids)
    contrast_path = artifacts.write_text("contrasts.csv", _csv_text(contrast_rows))
    summary["checks"]["registered_control_contrast_rows"] = len(contrast_rows)
    support_path = artifacts.write_text("support.csv", _csv_text(support_rows))
    receptive_path = artifacts.write_text("window_registry.csv", _csv_text(receptive_rows))
    baseline_path = artifacts.write_json("baseline_registry.json", baseline_registry)
    summary_path = artifacts.write_json("summary.json", summary)
    endpoint_hash_path = artifacts.write_json(
        "operator_endpoint_hashes.json", endpoint_hashes
    )
    artifacts.register_existing("predictions.h5")
    support_model_path = output_dir / "support_model.npz"
    np.savez_compressed(
        support_model_path,
        channel_center=support.channel_center,
        channel_scale=support.channel_scale,
        feature_center=support.feature_center,
        components=support.components,
        fit_scores=support.fit_scores,
        calibration_reconstruction=support.calibration_reconstruction,
        calibration_nearest=support.calibration_nearest,
    )
    artifacts.register_existing(support_model_path.name)
    dose_path = output_dir / "dose_response.png"
    _plot_dose(dose_path, ladder_rows, top_ids)
    artifacts.register_existing(dose_path.name)
    failure_path = output_dir / "failure.json"
    if failure_path.exists():
        failure_path.unlink()
    manifest = artifacts.finalize(
        config=resolved,
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=member_ids,
        row_ids=panel.row_indices,
        gradient_set="varied frozen panel + paired fixed panel",
        device=ensemble.device,
        repository=Path(__file__).resolve().parents[1],
    )
    if not args.pilot and not args.no_publish:
        _publish(
            [
                ladder_summary_path,
                contrast_path,
                support_path,
                receptive_path,
                baseline_path,
                summary_path,
                endpoint_hash_path,
                dose_path,
            ],
            Path(resolved["published_dir"]),
        )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"S03 {resolved['mode']} completed; manifest: {manifest}", flush=True)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        run(config, args)
    except Exception as error:
        run_id = config.get("pilot", {}).get("run_id") if args.pilot else config.get("run_id")
        output = (args.output_dir or Path("output/xai/S03") / str(run_id)).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.json").write_text(
            json.dumps({"exception": repr(error), "traceback": traceback.format_exc()}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
