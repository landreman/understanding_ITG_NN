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

import numpy as np

from xai_s03_ladder import (
    _compact_ladder_summary,
    _contrast_summary_rows,
    _csv_text,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--published-dir", type=Path, default=Path("reports/xai/S03_artifacts")
    )
    args = parser.parse_args()

    manifest_path = args.source_run / "manifest.json"
    ladder_path = args.source_run / "ladder.csv"
    support_path = args.source_run / "support.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (ladder_path, support_path):
        expected = manifest["output_hashes"][path.name]
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{path} hash {actual} does not match manifest {expected}")

    ladder = _read_csv(ladder_path)
    support = _read_csv(support_path)
    top_ids = {
        row["member_id"]
        for row in ladder
        if row["validation_cohort"] == "stored_validation_top10"
    }
    contrast_rows = _contrast_summary_rows(ladder, top_ids)
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
        amplitude.append(
            {
                "dose": dose,
                "member_median_effect": float(np.median(effects)),
                "member_median_robust_input_rms": float(np.median(inputs)),
                "member_median_effect_per_robust_input_rms": float(
                    np.median(efficiencies)
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
        },
        "scope": "varied-gradient rows only; fixed-gradient interpretations withdrawn",
        "support_warning_structural_null": {
            "median": math.sqrt(0.5),
            "fraction_above_0.95": 1 - 0.95**2,
            "observed_unperturbed_median": float(support_reference["warning_score_median"]),
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
    _atomic_text(ladder_summary_path, _csv_text(compact_rows))
    _atomic_text(contrasts_path, _csv_text(contrast_rows))
    summary["generated_artifacts"] = {
        "ladder_summary.csv": _sha256(ladder_summary_path),
        "contrasts.csv": _sha256(contrasts_path),
        "generator_sha256": _sha256(Path(__file__)),
    }
    _atomic_text(review_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(review_path)


if __name__ == "__main__":
    main()
