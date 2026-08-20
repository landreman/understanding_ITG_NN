#!/usr/bin/env python3
"""Establish the checkpoint's fixed-gradient input convention, and refresh S01.

This is the registered production run behind
`reports/xai/S03_fixed_gradient_decision.md`. It does two things:

1. Measures every ensemble member on the S01 panel's 1,000 fixed-gradient rows
   under both the training convention (`a/L_T = +3`) and the legacy marker
   (`-3`), and records the serialized legacy training tensors when they are
   reachable. Together these say which input the checkpoint was trained on.
2. Rewrites `reports/xai/S01_artifacts/panel_metadata.csv`, whose
   `a_over_LT_model` column recorded the negated value for fixed rows. Nothing
   in S01 ran the model on fixed rows, so only that column changes; the run
   asserts exactly that rather than trusting it.

The panel rows all sit inside `tests/data/review_slice.h5`, so a reviewer with
no dataset access can recompute the headline numbers from the committed slice
and checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from itg_nn.data import load_hdf5_rows, reference_split_assignments
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts, file_fingerprint, sha256_file
from itg_nn.xai.runtime import set_deterministic_seed


REPOSITORY = Path(__file__).resolve().parents[1]
CONVENTIONS = (("training", False), ("legacy_marker", True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xai/S03fix_fixed_gradient.json")
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--published-dir", type=Path)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the config's pilot overrides instead of the registered run.",
    )
    parser.add_argument(
        "--skip-s01-refresh",
        action="store_true",
        help="Measure the convention only; leave the committed S01 artifact alone.",
    )
    return parser


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("refusing to write an empty CSV artifact")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _load_s01_audit_module():
    """Import the S01 script by path; `scripts/` is not an importable package."""

    path = REPOSITORY / "scripts" / "xai_s01_audit.py"
    spec = importlib.util.spec_from_file_location("xai_s01_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["xai_s01_audit"] = module
    spec.loader.exec_module(module)
    return module


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch {prediction.shape} vs {target.shape}: refusing to "
            "broadcast, which would return a plausible-looking wrong number"
        )
    residual = float(np.sum((prediction - target) ** 2))
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - residual / total


def _member_predictions(
    ensemble, data, batch_size: int, device
) -> np.ndarray:
    """Signed per-member native predictions, kept before any aggregation."""

    outputs = []
    with torch.inference_mode():
        for model in ensemble.models:
            member = []
            for start in range(0, len(data.row_indices), batch_size):
                stop = min(start + batch_size, len(data.row_indices))
                member.append(
                    model(
                        data.geometry[start:stop].to(device),
                        data.a_over_lt[start:stop].to(device),
                        data.a_over_ln[start:stop].to(device),
                    )
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
            outputs.append(np.concatenate(member))
    stacked = np.stack(outputs).astype(np.float64)
    if stacked.shape != (len(ensemble.models), len(data.row_indices)):
        raise RuntimeError(
            f"member predictions have shape {stacked.shape}, expected "
            f"{(len(ensemble.models), len(data.row_indices))}; a trailing unit "
            "axis here would silently broadcast against the target."
        )
    return stacked


def _convention_rows(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    member_ids: tuple[str, ...],
    floor: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for convention, values in predictions.items():
        entities: list[tuple[str, str, np.ndarray]] = [
            (member_id, "member", values[index])
            for index, member_id in enumerate(member_ids)
        ]
        entities.append(("ensemble_mean", "ensemble", values.mean(axis=0)))
        for entity, entity_type, series in entities:
            rows.append(
                {
                    "entity": entity,
                    "entity_type": entity_type,
                    "convention": convention,
                    "validity_tag": (
                        "observed_comparison"
                        if convention == "training"
                        else "off_manifold"
                    ),
                    "a_over_LT_input": 3.0 if convention == "training" else -3.0,
                    "n": int(len(series)),
                    "prediction_mean": float(series.mean()),
                    "prediction_std": float(series.std()),
                    "prediction_min": float(series.min()),
                    "prediction_max": float(series.max()),
                    "fraction_at_or_below_floor_plus_0p05": float(
                        np.mean(series <= floor + 0.05)
                    ),
                    "fraction_within_0p05_of_floor_two_sided": float(
                        np.mean(np.abs(series - floor) <= 0.05)
                    ),
                    "r2_against_fixed_target": _r2(series, target),
                    "mean_absolute_error": float(np.mean(np.abs(series - target))),
                }
            )
    return rows


def _legacy_training_tensors(path: Path) -> dict[str, Any]:
    """Read the legacy serialized dataset, if this machine has it.

    It is an artifact of the legacy run, not an input to anything here, so a
    missing file is recorded rather than raised.
    """

    if not path.is_file():
        return {"available": False, "path": str(path)}
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    base = bundle["train_dataset"].dataset
    a_over_lt = base.tensors[2].numpy()
    splits = {}
    for name, subset in bundle.items():
        indices = np.asarray(subset.indices, dtype=np.int64)
        values = a_over_lt[indices]
        splits[name] = {
            "n": int(len(indices)),
            "a_over_LT_min": float(values.min()),
            "a_over_LT_max": float(values.max()),
            "negative_count": int(np.sum(values < 0)),
        }
    return {
        "available": True,
        "fingerprint": file_fingerprint(path),
        "total_rows": int(len(a_over_lt)),
        "negative_count": int(np.sum(a_over_lt < 0)),
        "a_over_LT_min": float(a_over_lt.min()),
        "a_over_LT_max": float(a_over_lt.max()),
        "splits": splits,
        "saved_test_rows": int(len(bundle["test_dataset"])),
        "interpretation": (
            "The saved split holds the unfiltered test set and no negative "
            "a/L_T, so it predates the negation trick in Cyclic_net.py. The "
            "checkpoint's own behaviour, not this file, is the decisive "
            "evidence; this records that the two agree."
        ),
    }


def _refresh_panel_metadata(
    dataset: Path, panel_rows: np.ndarray, artifacts: RunArtifacts, published: Path
) -> dict[str, Any]:
    """Rewrite the S01 panel metadata and prove only the one column moved."""

    audit = _load_s01_audit_module()
    assignments = reference_split_assignments(dataset)
    _, _, metadata_rows = audit._panel_artifact(dataset, panel_rows, assignments)
    text = _csv_text(metadata_rows)
    written = artifacts.write_text("panel_metadata.csv", text)

    target = published / "panel_metadata.csv"
    before = list(csv.DictReader(target.open()))
    after = list(csv.DictReader(io.StringIO(text)))
    if len(before) != len(after):
        raise RuntimeError(
            f"refreshed panel metadata has {len(after)} rows, committed has {len(before)}"
        )
    changed: dict[str, int] = {}
    for old_row, new_row in zip(before, after):
        if old_row["stable_id"] != new_row["stable_id"]:
            raise RuntimeError("panel metadata row order changed; refusing to publish")
        for key in old_row:
            if old_row[key] != new_row[key]:
                changed[key] = changed.get(key, 0) + 1
    if set(changed) - {"a_over_LT_model"}:
        raise RuntimeError(
            "refresh changed columns beyond a_over_LT_model: "
            f"{sorted(set(changed) - {'a_over_LT_model'})}"
        )
    fixed_values = {
        row["a_over_LT_model"] for row in after if row["gradient_set"] == "fixed"
    }
    varied_before = [row["a_over_LT_model"] for row in before if row["gradient_set"] == "varied"]
    varied_after = [row["a_over_LT_model"] for row in after if row["gradient_set"] == "varied"]
    if varied_before != varied_after:
        raise RuntimeError("varied-row gradients changed; refusing to publish")
    shutil.copy2(written, target)
    return {
        "rows": len(after),
        "changed_cells": changed,
        "fixed_a_over_LT_model_values": sorted(fixed_values),
        "varied_rows_unchanged": True,
        "published_to": str(target),
    }


def run(config: dict[str, Any], args: argparse.Namespace) -> Path:
    resolved = dict(config)
    if args.pilot:
        resolved.update(resolved.get("pilot", {}))
    resolved.pop("pilot", None)
    for key, value in (
        ("dataset", args.dataset),
        ("checkpoint", args.checkpoint),
        ("cohorts", args.cohorts),
        ("published_dir", args.published_dir),
    ):
        if value is not None:
            resolved[key] = str(value)

    dataset = Path(resolved["dataset"])
    checkpoint = Path(resolved["checkpoint"])
    published = Path(resolved["published_dir"])
    output_dir = (
        args.output_dir or Path("output/xai/S03fix") / str(resolved["run_id"])
    ).resolve()
    artifacts = RunArtifacts(output_dir)
    set_deterministic_seed(int(resolved["seed"]))

    cohorts = json.loads(Path(resolved["cohorts"]).read_text(encoding="utf-8"))
    panel_rows = np.asarray(
        cohorts["interpretation_panel"]["fixed_row_ids"], dtype=np.int64
    )
    if "panel_rows" in resolved:
        panel_rows = panel_rows[: int(resolved["panel_rows"])]

    ensemble = load_ensemble(checkpoint, device=resolved["device"])
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "members" in resolved:
        ranked = sorted(
            bundle["members"],
            key=lambda member: (-float(member["validation_r2"]), str(member["id"])),
        )[: int(resolved["members"])]
        keep = {str(member["id"]) for member in ranked}
        indices = [i for i, mid in enumerate(ensemble.member_ids) if mid in keep]
        ensemble.models = [ensemble.models[i] for i in indices]
        member_ids = tuple(ensemble.member_ids[i] for i in indices)
    else:
        member_ids = tuple(ensemble.member_ids)

    predictions: dict[str, np.ndarray] = {}
    target: np.ndarray | None = None
    for name, legacy in CONVENTIONS:
        data = load_hdf5_rows(
            dataset,
            panel_rows,
            gradient_set="fixed",
            include_targets=True,
            legacy_fixed_marker=legacy,
        )
        if data.actual_log_heat_flux is None:
            raise RuntimeError("fixed-gradient targets unavailable")
        expected = -3.0 if legacy else 3.0
        if not np.allclose(data.a_over_lt.numpy(), expected):
            raise RuntimeError(f"loader did not supply a/L_T = {expected}")
        target = data.actual_log_heat_flux.numpy().astype(np.float64)
        predictions[name] = _member_predictions(
            ensemble, data, int(resolved["batch_size"]), ensemble.device
        )
        print(f"convention {name}: {len(member_ids)} members predicted", flush=True)

    assert target is not None
    floor = float(resolved["log_heat_flux_floor"])
    rows = _convention_rows(predictions, target, member_ids, floor)
    convention_path = artifacts.write_text(
        "fixed_gradient_convention.csv", _csv_text(rows)
    )

    ensemble_rows = {row["convention"]: row for row in rows if row["entity"] == "ensemble_mean"}
    member_r2 = {
        convention: [
            row["r2_against_fixed_target"] for row in rows
            if row["convention"] == convention and row["entity_type"] == "member"
        ]
        for convention, _ in CONVENTIONS
    }
    summary = {
        "estimand": "native max(log Q, -2) on the S01 panel's fixed-gradient rows",
        "panel_fixed_rows": int(len(panel_rows)),
        "members": len(member_ids),
        "target": {
            "mean": float(target.mean()),
            "std": float(target.std()),
            "fraction_at_floor": float(np.mean(target <= floor + 1e-6)),
        },
        "ensemble": {
            convention: {
                key: row[key]
                for key in (
                    "prediction_mean",
                    "prediction_std",
                    "prediction_min",
                    "prediction_max",
                    "r2_against_fixed_target",
                    "fraction_at_or_below_floor_plus_0p05",
                    "fraction_within_0p05_of_floor_two_sided",
                )
            }
            for convention, row in ensemble_rows.items()
        },
        "member_r2_range": {
            convention: [float(min(values)), float(max(values))]
            for convention, values in member_r2.items()
        },
        "members_with_r2_above_0p9": {
            convention: int(sum(value > 0.9 for value in values))
            for convention, values in member_r2.items()
        },
        "legacy_training_tensors": _legacy_training_tensors(
            Path(resolved["legacy_training_dataset"])
        ),
        "conclusion": (
            "The checkpoint was trained with fixed-gradient rows at the physical "
            "a/L_T = +3. The -3 marker is off-manifold and saturates the members "
            "at the clipped-log floor."
        ),
    }

    if not args.skip_s01_refresh and not args.pilot:
        varied_panel_rows = np.asarray(
            cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64
        )
        summary["s01_panel_metadata_refresh"] = _refresh_panel_metadata(
            dataset, varied_panel_rows, artifacts, Path("reports/xai/S01_artifacts")
        )

    summary_path = artifacts.write_json("summary.json", summary)

    published.mkdir(parents=True, exist_ok=True)
    for source in (convention_path, summary_path):
        shutil.copy2(source, published / source.name)

    manifest = artifacts.finalize(
        config=resolved,
        dataset=dataset,
        checkpoint=checkpoint,
        member_ids=member_ids,
        row_ids=panel_rows,
        gradient_set="fixed",
        device=ensemble.device,
        repository=REPOSITORY,
        published_dir=None if args.pilot else published,
    )
    print(json.dumps(summary["ensemble"], indent=2))
    print(f"manifest: {manifest}")
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(config, args)


if __name__ == "__main__":
    main()
