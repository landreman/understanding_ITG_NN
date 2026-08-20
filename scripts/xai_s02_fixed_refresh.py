#!/usr/bin/env python3
"""Recompute S02's fixed-gradient strata under the corrected input convention.

S02 measured shift and parity sensitivity on the panel's 1,000 fixed-gradient
twins while the loader was driving them to `a/L_T = -3`, which pins the ensemble
mean at the clipped-log floor and flattens every member against it. Those rows
measured saturation, not the response of the trained function, so the numbers
were withdrawn. This run replaces them.

Only the fixed half of the panel is recomputed. The varied half is read back
from the registered S02 run's `predictions.h5` and carried through unchanged,
and the run first re-predicts a sample of varied members and asserts they
reproduce the stored values bit-for-bit — otherwise a difference in the fixed
rows could be this script rather than the correction.

The committed CSVs are edited row by row: every varied row is copied verbatim
and only rows tagged `fixed` are replaced, so the diff shows exactly what the
convention changed and nothing else.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from itg_nn.data import load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, file_fingerprint, sha256_file
from itg_nn.xai.runtime import set_deterministic_seed
from itg_nn.xai.symmetry import (
    InvariantMember,
    reverse_parallel,
    stellarator_parity,
)


REPOSITORY = Path(__file__).resolve().parents[1]

# The three committed artifacts that carry per-stratum rows for fixed rows.
REFRESHED_CSVS = (
    "shift_symmetry_summary.csv",
    "phase_average_exactness.csv",
    "parity_symmetry.csv",
)

# Columns that identify a row rather than report a measurement. Everything else
# in a fixed row is replaced from the recomputed values.
KEY_COLUMNS = {
    "shift_symmetry_summary.csv": ("entity", "gradient_set", "stratum", "shift"),
    "phase_average_exactness.csv": ("entity", "gradient_set", "stratum"),
    "parity_symmetry.csv": ("entity", "gradient_set", "stratum", "transform"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xai/S03fix_s02_refresh.json")
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Compute and report without rewriting the committed S02 artifacts.",
    )
    return parser


def _load_s02_module():
    """Reuse S02's own statistic definitions rather than restating them."""

    path = REPOSITORY / "scripts" / "xai_s02_symmetry.py"
    spec = importlib.util.spec_from_file_location("xai_s02_symmetry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["xai_s02_symmetry"] = module
    spec.loader.exec_module(module)
    return module


def _csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _format(value: Any) -> str:
    """Match how csv.DictWriter rendered the original run's values."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _predict_fixed_panel(
    s02,
    ensemble,
    member_ids: tuple[str, ...],
    fixed_data,
    resolved: dict[str, Any],
    path: Path,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-member 96-shift and parity predictions on the fixed panel rows."""

    row_count = len(fixed_data.row_indices)
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; pass --resume or a new --output-dir")
    if not path.exists():
        with h5py.File(path, "w") as handle:
            handle.attrs["member_ids"] = json.dumps(list(member_ids))
            handle.attrs["estimand"] = "native max(log Q, -2)"
            handle.attrs["fixed_gradient_convention"] = "training_positive_a_over_LT"
            handle.create_dataset("panel_row_id", data=fixed_data.row_indices)
            handle.create_dataset("complete", data=np.zeros(len(member_ids), dtype=bool))
            handle.create_dataset(
                "panel_shift_prediction",
                shape=(len(member_ids), row_count, 96),
                dtype="f4",
            )
            handle.create_dataset(
                "parity_prediction", shape=(len(member_ids), 3, row_count), dtype="f4"
            )

    with h5py.File(path, "r+") as handle:
        if json.loads(handle.attrs["member_ids"]) != list(member_ids):
            raise RuntimeError("resume artifact member IDs do not match")
        if not np.array_equal(handle["panel_row_id"][:], fixed_data.row_indices):
            raise RuntimeError("resume artifact panel rows do not match")
        index_by_id = {mid: i for i, mid in enumerate(ensemble.member_ids)}
        batch_size = int(resolved["batch_size"])
        phase_chunk = int(resolved["phase_chunk"])
        for position, member_id in enumerate(member_ids):
            if bool(handle["complete"][position]):
                continue
            started = time.perf_counter()
            wrapped = InvariantMember(ensemble.models[index_by_id[member_id]])
            shifts = s02._predict_phases(
                wrapped, fixed_data, tuple(range(96)), batch_size, phase_chunk,
                ensemble.device,
            )
            parity = s02._predict_original(
                wrapped,
                s02._transformed_data(fixed_data, stellarator_parity),
                batch_size,
                ensemble.device,
            )
            wrong = s02._predict_original(
                wrapped,
                s02._transformed_data(fixed_data, reverse_parallel),
                batch_size,
                ensemble.device,
            )
            handle["panel_shift_prediction"][position] = shifts
            handle["parity_prediction"][position] = np.stack(
                (shifts[:, 0], parity, wrong)
            )
            handle["complete"][position] = True
            handle.flush()
            print(
                f"member {position + 1}/{len(member_ids)} {member_id}: "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
        if not np.all(handle["complete"][:]):
            raise RuntimeError("fixed-panel prediction artifact is incomplete")
        return (
            handle["panel_shift_prediction"][:],
            handle["parity_prediction"][:],
        )


def _verify_varied_reproduction(
    s02, ensemble, member_ids, varied_data, stored, resolved, count: int
) -> dict[str, Any]:
    """Re-predict some varied members and require the stored values back.

    If this fails, a change in the fixed rows cannot be attributed to the
    convention, because the pipeline itself no longer matches the S02 run.
    """

    index_by_id = {mid: i for i, mid in enumerate(ensemble.member_ids)}
    checked = []
    worst = 0.0
    for position in range(min(count, len(member_ids))):
        member_id = member_ids[position]
        wrapped = InvariantMember(ensemble.models[index_by_id[member_id]])
        shifts = s02._predict_phases(
            wrapped, varied_data, tuple(range(96)), int(resolved["batch_size"]),
            int(resolved["phase_chunk"]), ensemble.device,
        )
        difference = float(np.max(np.abs(shifts - stored[position])))
        worst = max(worst, difference)
        checked.append({"member_id": member_id, "max_absolute_difference": difference})
    if worst != 0.0:
        raise RuntimeError(
            f"varied-row re-prediction differs from the registered S02 run by "
            f"{worst}; the refresh pipeline is not the S02 pipeline"
        )
    return {"members_checked": checked, "max_absolute_difference": worst}


def _rebuild_rows(
    s02,
    panel_shift_prediction: np.ndarray,
    parity_prediction: np.ndarray,
    panel_data,
    varied_count: int,
    member_ids: tuple[str, ...],
    residual_std: dict[tuple[str, str], float],
    stable_threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    """Run S02's own row builders over the recombined panel."""

    strata = s02._panel_strata(panel_data, varied_count, stable_threshold)
    entities: list[tuple[str, np.ndarray, str]] = [
        (member_id, panel_shift_prediction[index], "member")
        for index, member_id in enumerate(member_ids)
    ]
    entities.extend(
        (
            ("ensemble_mean", panel_shift_prediction.mean(axis=0), "ensemble"),
            ("ensemble_spread", panel_shift_prediction.std(axis=0), "ensemble"),
        )
    )

    symmetry_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for entity, values, entity_type in entities:
        for (gradient_set, stratum), mask in strata.items():
            own_residual = (
                residual_std.get((entity, stratum)) if gradient_set == "varied" else None
            )
            for shift in range(96):
                metrics = s02._change_metrics(values[mask, 0], values[mask, shift])
                symmetry_rows.append(
                    {
                        "entity": entity,
                        "entity_type": entity_type,
                        "gradient_set": gradient_set,
                        "stratum": stratum,
                        "n": int(np.sum(mask)),
                        "shift": shift,
                        "exact_pooling_subgroup": shift in (0, 32, 64),
                        **metrics,
                        "original_reference_residual_std": own_residual,
                        "rms_change_over_residual_std": (
                            metrics["rms_change"] / own_residual if own_residual else None
                        ),
                    }
                )
            metrics = s02._change_metrics(
                values[mask, :32].mean(axis=1), values[mask].mean(axis=1)
            )
            phase_rows.append(
                {
                    "entity": entity,
                    "entity_type": entity_type,
                    "gradient_set": gradient_set,
                    "stratum": stratum,
                    "n": int(np.sum(mask)),
                    **metrics,
                }
            )

    parity_entities: list[tuple[str, np.ndarray, str]] = [
        (member_id, parity_prediction[index], "member")
        for index, member_id in enumerate(member_ids)
    ]
    parity_entities.extend(
        (
            ("ensemble_mean", parity_prediction.mean(axis=0), "ensemble"),
            ("ensemble_spread", parity_prediction.std(axis=0), "ensemble"),
        )
    )
    parity_rows: list[dict[str, Any]] = []
    for entity, values, entity_type in parity_entities:
        for (gradient_set, stratum), mask in strata.items():
            own_residual = (
                residual_std.get((entity, stratum)) if gradient_set == "varied" else None
            )
            for transform_index, transform_name in (
                (1, "stellarator_parity"),
                (2, "plain_reversal_control"),
            ):
                metrics = s02._change_metrics(
                    values[0, mask], values[transform_index, mask]
                )
                parity_rows.append(
                    {
                        "entity": entity,
                        "entity_type": entity_type,
                        "gradient_set": gradient_set,
                        "stratum": stratum,
                        "n": int(np.sum(mask)),
                        "transform": transform_name,
                        **metrics,
                        "original_reference_residual_std": own_residual,
                        "rms_change_over_residual_std": (
                            metrics["rms_change"] / own_residual if own_residual else None
                        ),
                    }
                )

    return {
        "shift_symmetry_summary.csv": s02._shift_summary_rows(symmetry_rows),
        "phase_average_exactness.csv": phase_rows,
        "parity_symmetry.csv": parity_rows,
        "_symmetry_rows": symmetry_rows,
    }


def _verify_varied_rows_reproduced(
    published: Path, rebuilt: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Check the rebuild reproduces S02's *varied* rows, which must not move.

    The rebuilt varied rows are discarded by the merge, so this costs nothing
    and buys the strongest available statement: the fixed numbers that do get
    published came out of a code path that reproduces the registered run's
    published numbers on the half of the panel where the answer is already
    known.
    """

    report: dict[str, Any] = {}
    for name, rows in rebuilt.items():
        keys = KEY_COLUMNS[name]
        with (published / name).open(newline="") as handle:
            committed = {
                tuple(row[key] for key in keys): row
                for row in csv.DictReader(handle)
                if row["gradient_set"] == "varied"
            }
        compared = 0
        worst = 0.0
        worst_field = None
        for row in rows:
            if row["gradient_set"] != "varied":
                continue
            key = tuple(_format(row[column]) for column in keys)
            if key not in committed:
                raise RuntimeError(f"{name}: rebuilt varied row {key} is not committed")
            for column, value in committed[key].items():
                if column in keys or column in ("entity_type", "n"):
                    continue
                rebuilt_text = _format(row[column])
                if rebuilt_text == value:
                    continue
                try:
                    difference = abs(float(rebuilt_text) - float(value))
                except ValueError as error:
                    raise RuntimeError(
                        f"{name}: varied row {key} column {column} changed "
                        f"{value!r} -> {rebuilt_text!r}"
                    ) from error
                if difference > worst:
                    worst, worst_field = difference, f"{name}:{key}:{column}"
            compared += 1
        report[name] = {
            "varied_rows_compared": compared,
            "max_absolute_difference": worst,
            "worst_field": worst_field,
        }
        if worst > 1e-12:
            raise RuntimeError(
                f"{name}: rebuilt varied rows differ from the registered S02 run "
                f"by {worst} at {worst_field}; the fixed rows cannot be trusted"
            )
    return report


def _merge_csv(committed: Path, rebuilt: list[dict[str, Any]], name: str) -> tuple[str, int]:
    """Replace only `fixed` rows, preserving order and every varied byte."""

    keys = KEY_COLUMNS[name]
    lookup = {
        tuple(_format(row[key]) for key in keys): row
        for row in rebuilt
        if row["gradient_set"] == "fixed"
    }
    with committed.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        original = list(reader)

    merged: list[dict[str, Any]] = []
    replaced = 0
    for row in original:
        if row["gradient_set"] != "fixed":
            merged.append(row)
            continue
        key = tuple(row[column] for column in keys)
        if key not in lookup:
            raise RuntimeError(f"{name}: no recomputed row for {key}")
        source = lookup[key]
        updated = {
            column: (
                row[column]
                if column in keys or column == "n"
                else _format(source[column])
            )
            for column in fieldnames
        }
        if updated["n"] != _format(source["n"]):
            raise RuntimeError(
                f"{name}: stratum size changed for {key}; strata come from targets "
                "and must not move"
            )
        merged.append(updated)
        replaced += 1
    if replaced != len(lookup):
        raise RuntimeError(f"{name}: replaced {replaced} rows, rebuilt {len(lookup)}")

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(merged)
    return stream.getvalue(), replaced


def run(config: dict[str, Any], args: argparse.Namespace) -> Path:
    resolved = dict(config)
    if args.pilot:
        resolved.update(resolved.get("pilot", {}))
    resolved.pop("pilot", None)
    for key, value in (("dataset", args.dataset), ("checkpoint", args.checkpoint)):
        if value is not None:
            resolved[key] = str(value)

    dataset = Path(resolved["dataset"])
    checkpoint = Path(resolved["checkpoint"])
    published = Path(resolved["published_dir"])
    s02_run = Path(resolved["s02_run_dir"])
    output_dir = (
        args.output_dir or Path("output/xai/S03fix") / str(resolved["run_id"])
    ).resolve()
    artifacts = RunArtifacts(output_dir)
    set_deterministic_seed(int(resolved["seed"]))
    s02 = _load_s02_module()

    cohorts = json.loads(Path(resolved["cohorts"]).read_text(encoding="utf-8"))
    panel_rows = np.asarray(
        cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64
    )
    if "panel_rows" in resolved:
        panel_rows = panel_rows[: int(resolved["panel_rows"])]

    stored_path = s02_run / "predictions.h5"
    with h5py.File(stored_path, "r") as stored:
        stored_member_ids = tuple(json.loads(stored.attrs["member_ids"]))
        stored_panel_rows = stored["panel_row_id"][:]
        registered_count = len(stored_panel_rows) // 2
        keep = len(panel_rows)
        if not np.array_equal(stored_panel_rows[:keep], panel_rows):
            raise RuntimeError("registered S02 panel rows do not match cohorts.json")
        if not np.array_equal(
            stored_panel_rows[registered_count : registered_count + keep], panel_rows
        ):
            raise RuntimeError("registered S02 fixed twins are not the same rows")
        stored_varied_shift = stored["panel_shift_prediction"][:, :keep, :]
        stored_varied_parity = stored["parity_prediction"][:, :, :keep]

    member_ids = stored_member_ids
    if "members" in resolved:
        member_ids = member_ids[: int(resolved["members"])]
        stored_varied_shift = stored_varied_shift[: len(member_ids)]
        stored_varied_parity = stored_varied_parity[: len(member_ids)]

    ensemble = load_ensemble(checkpoint, device=resolved["device"])
    varied_data = load_hdf5_rows(
        dataset, panel_rows, gradient_set="varied", include_targets=True
    )
    fixed_data = load_hdf5_rows(
        dataset, panel_rows, gradient_set="fixed", include_targets=True
    )
    if not np.allclose(fixed_data.a_over_lt.numpy(), 3.0):
        raise RuntimeError("fixed rows are not being supplied at the training +3")
    panel_data = s02._concatenate_data(varied_data, fixed_data)

    reproduction = _verify_varied_reproduction(
        s02, ensemble, member_ids, varied_data, stored_varied_shift, resolved,
        int(resolved["varied_check_members"]),
    )
    print(f"varied reproduction check passed: {reproduction}", flush=True)

    fixed_shift, fixed_parity = _predict_fixed_panel(
        s02, ensemble, member_ids, fixed_data, resolved,
        output_dir / "fixed_panel_predictions.h5", args.resume,
    )
    artifacts.register_existing("fixed_panel_predictions.h5")

    panel_shift_prediction = np.concatenate(
        (stored_varied_shift, fixed_shift), axis=1
    )
    parity_prediction = np.concatenate(
        (stored_varied_parity, fixed_parity), axis=2
    )

    # The varied normalisation constants are S02's, read back from its committed
    # accuracy table rather than recomputed, because no varied number moves here.
    residual_std: dict[tuple[str, str], float] = {}
    with (published / "accuracy.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["function"] == "original_f":
                residual_std[(row["entity"], row["stratum"])] = float(
                    row["residual_std"]
                )

    rebuilt = _rebuild_rows(
        s02, panel_shift_prediction, parity_prediction, panel_data, len(panel_rows),
        member_ids, residual_std, float(resolved["stable_threshold_log_Q"]),
    )
    symmetry_rows = rebuilt.pop("_symmetry_rows")

    exact_tolerance = float(resolved["exact_atol"])
    relative_tolerance = float(resolved["exact_rtol"])
    # The subgroup maximum is a max over the whole panel of a float32 roundoff
    # difference, so the single headline number is platform-specific and cannot
    # be cross-checked against any other committed table. Publish the rows it is
    # taken over, for both gradient sets and every entity, so a reviewer on a
    # different machine can compare distributions instead of one scalar.
    subgroup_rows = [
        {
            key: row[key]
            for key in (
                "entity", "entity_type", "gradient_set", "stratum", "n", "shift",
                "mean_absolute_change", "rms_change", "max_absolute_change",
            )
        }
        for row in symmetry_rows
        if row["shift"] in (32, 64)
    ]
    exact_rows = [
        row for row in symmetry_rows
        if row["entity_type"] == "member" and row["shift"] in (32, 64)
    ]
    checks = {
        "exact_subgroup_max_abs": float(
            max(row["max_absolute_change"] for row in exact_rows)
        ),
        "exact_subgroup_pass": bool(
            np.allclose(
                panel_shift_prediction[:, :, (0,)],
                panel_shift_prediction[:, :, (32, 64)],
                atol=exact_tolerance,
                rtol=relative_tolerance,
            )
        ),
        "phase_32_vs_96_max_abs": float(
            max(
                row["max_absolute_change"]
                for row in rebuilt["phase_average_exactness.csv"]
                if row["entity_type"] == "member"
            )
        ),
        "phase_32_vs_96_pass": bool(
            np.allclose(
                panel_shift_prediction[:, :, :32].mean(axis=2),
                panel_shift_prediction.mean(axis=2),
                atol=exact_tolerance,
                rtol=relative_tolerance,
            )
        ),
    }

    # A pilot runs a subset of rows, so its strata are a different size than the
    # committed table's and the merge guard would correctly refuse it. The merge
    # itself is covered by tests/xai/test_fixed_gradient_refresh.py.
    varied_rebuild_check = (
        _verify_varied_rows_reproduced(published, rebuilt) if not args.pilot else None
    )

    merged_counts: dict[str, int] = {}
    written: list[Path] = []
    if not args.pilot:
        for name in REFRESHED_CSVS:
            text, replaced = _merge_csv(published / name, rebuilt[name], name)
            written.append(artifacts.write_text(name, text))
            merged_counts[name] = replaced

    def _quantiles(gradient_set: str, stratum: str) -> dict[str, float]:
        values = np.asarray(
            [
                row["rms_change"]
                for row in symmetry_rows
                if row["entity_type"] == "member"
                and row["gradient_set"] == gradient_set
                and row["stratum"] == stratum
                and row["shift"] not in (0, 32, 64)
            ]
        )
        return {
            "q10": float(np.quantile(values, 0.1)),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.9)),
        }

    def _parity_quantiles(gradient_set: str) -> dict[str, float]:
        values = np.asarray(
            [
                row["rms_change"]
                for row in rebuilt["parity_symmetry.csv"]
                if row["entity_type"] == "member"
                and row["gradient_set"] == gradient_set
                and row["stratum"] == "all"
                and row["transform"] == "stellarator_parity"
            ]
        )
        return {
            "q10": float(np.quantile(values, 0.1)),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.9)),
        }

    summary = {
        "estimand": "native max(log Q, -2); unnormalised RMS change on fixed rows",
        "why": (
            "S02's fixed-gradient strata were computed with the loader driving "
            "a/L_T to -3, which pins the ensemble mean at the clipped-log floor "
            "and flattens every member against it."
        ),
        "convention": "training_positive_a_over_LT",
        "members": len(member_ids),
        "panel_varied_rows": int(len(panel_rows)),
        "panel_fixed_rows": int(len(panel_rows)),
        "varied_reproduction_check": reproduction,
        "varied_rebuild_check": varied_rebuild_check,
        "rows_replaced": merged_counts,
        "arbitrary_shift_member_rms_change": {
            "fixed_all": _quantiles("fixed", "all"),
            "fixed_stable_near_floor": _quantiles("fixed", "stable_near_floor"),
            "fixed_unstable": _quantiles("fixed", "unstable"),
            "varied_all": _quantiles("varied", "all"),
        },
        "parity_member_rms_change": {
            "fixed_all": _parity_quantiles("fixed"),
            "varied_all": _parity_quantiles("varied"),
        },
        "checks": checks,
        "subgroup_max_abs_by_gradient_set": {
            gradient_set: float(
                max(
                    row["max_absolute_change"]
                    for row in subgroup_rows
                    if row["entity_type"] == "member"
                    and row["gradient_set"] == gradient_set
                    and row["stratum"] == "all"
                )
            )
            for gradient_set in ("varied", "fixed")
        },
        "subgroup_max_abs_note": (
            "These maxima are float32 roundoff differences an order of magnitude "
            "below the registered atol/rtol of 2e-5. Their exact values depend on "
            "the machine and batching, so a reviewer on another platform should "
            "expect the pass verdict to reproduce and the digits not to. See "
            "s02_subgroup_exactness.csv for the rows they are taken over."
        ),
        "exact_tolerance": {"atol": exact_tolerance, "rtol": relative_tolerance},
        "s02_run": {
            "directory": str(s02_run),
            "predictions_sha256": sha256_file(stored_path),
        },
    }
    subgroup_path = artifacts.write_text(
        "s02_subgroup_exactness.csv", _csv_text(subgroup_rows)
    )
    summary_path = artifacts.write_json("summary.json", summary)

    if not args.no_publish and not args.pilot:
        for path in written:
            shutil.copy2(path, published / path.name)
        decision = Path(resolved["decision_artifacts"])
        shutil.copy2(summary_path, decision / "s02_fixed_refresh_summary.json")
        shutil.copy2(subgroup_path, decision / subgroup_path.name)
        s02_summary_path = published / "summary.json"
        s02_summary = json.loads(s02_summary_path.read_text(encoding="utf-8"))
        s02_summary["checks"].update(checks)
        s02_summary["fixed_gradient_convention"] = {
            "convention": "training_positive_a_over_LT",
            "refreshed_by": str(Path("scripts") / Path(__file__).name),
            "run_id": str(resolved["run_id"]),
            "note": (
                "Fixed-gradient strata in shift_symmetry_summary.csv, "
                "phase_average_exactness.csv and parity_symmetry.csv were "
                "recomputed after the loader correction. Varied-row results, "
                "accuracy, density and bottleneck artifacts are the original "
                "S02 run's and are unaffected."
            ),
        }
        s02_summary_path.write_text(
            json.dumps(s02_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    manifest = artifacts.finalize(
        config=resolved,
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=member_ids,
        row_ids=panel_rows,
        gradient_set="fixed",
        device=ensemble.device,
        repository=REPOSITORY,
        published_dir=None,
    )
    if not args.no_publish and not args.pilot:
        shutil.copy2(
            manifest,
            Path(resolved["decision_artifacts"]) / "s02_fixed_refresh_manifest.json",
        )
    print(json.dumps(summary["arbitrary_shift_member_rms_change"], indent=2))
    print(json.dumps(checks, indent=2))
    print(f"manifest: {manifest}")
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(config, args)


if __name__ == "__main__":
    main()
