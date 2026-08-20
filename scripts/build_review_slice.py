#!/usr/bin/env python3
"""Build the committed review slice from the external HDF5 dataset.

The automated pull-request review (`.claude/commands/review-step.md`) runs on a
GitHub Actions runner, which cannot see the 647 MB dataset registered in
`PLAN.md`. Without real rows a reviewer can only read code and re-read the
report's own summary tables, so it cannot catch a number that is simply wrong.

This script writes `tests/data/review_slice.h5`: 2,000 real rows, about 7 MB
gzipped, small enough for plain git. Geometry is stored as float32, which is
lossless with respect to what the model sees because `itg_nn.data.load_hdf5_rows`
casts to float32 on the way in.

Selection rule, in order, deterministic given the seed:

1. All 1,000 rows of the S01 frozen interpretation panel, so the reviewer can
   recompute the numbers a report actually claims. The panel's varied and fixed
   stable IDs share one `raw_feature_tensor` row, so 2,000 panel sample IDs are
   1,000 HDF5 rows.
2. Every sibling flux tube of a panel equilibrium that is also in the S01 varied
   reference cohort. The panel takes at most one tube per equilibrium, so on the
   panel alone a bootstrap grouped by `equilibrium_files` is indistinguishable
   from one grouped by flux tube. These rows restore that distinction inside the
   registered cohort.
3. Remaining sibling tubes of panel equilibria, taken round-robin so multiplicity
   spreads across many equilibria rather than piling onto a few, until the row
   budget is full.

Rows whose `Q_avgs` is not positive in either gradient set are skipped in steps 2
and 3, so every slice row can be loaded with `include_targets=True`.

The slice also carries ensemble predictions computed here, from the **parent**
file, for every slice row in both gradient sets. `tests/xai/test_review_slice.py`
checks the slice reproduces them, which is what makes a silently corrupted or
mis-indexed slice a test failure rather than a wrong scientific result.

Usage, from the repository root, in an environment that can see the dataset:

    .venv-xai/bin/python scripts/build_review_slice.py

Regenerating is a deliberate act: the slice is a frozen verification artifact,
and changing it invalidates every review that used the old one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

from itg_nn.data import load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.config import DEFAULT_CHECKPOINT, DEFAULT_DATASET


SLICE_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path("tests/data/review_slice.h5")
COHORTS = Path("reports/xai/S01_artifacts/cohorts.json")
DEFAULT_ROW_BUDGET = 2000
DEFAULT_SEED = 20260820

# Copied verbatim from the parent file; small and needed to interpret the rest.
SHARED_METADATA = (
    "equilibrium_class_descriptions",
    "n_scalar_features",
    "n_tubes",
    "n_z",
    "n_z_functions",
    "scalar_features",
    "scalar_features_long",
    "z",
    "z_functions_GX",
    "z_functions_pretty",
)
PER_ROW = (
    "FSA_grad_xs",
    "QUASR_IDs",
    "equilibrium_class",
    "equilibrium_files",
    "tube_files",
)
GRADIENT_GROUPS = ("fixed_gradient_simulations", "varied_gradient_simulations")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cohorts", type=Path, default=COHORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROW_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def select_rows(
    dataset: Path, cohorts: dict, budget: int, seed: int
) -> tuple[np.ndarray, dict]:
    """Return the sorted source row IDs and a record of how they were chosen."""

    panel = np.asarray(cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64)
    fixed = np.asarray(cohorts["interpretation_panel"]["fixed_row_ids"], dtype=np.int64)
    if not np.array_equal(np.sort(panel), np.sort(fixed)):
        raise ValueError(
            "The panel's varied and fixed row IDs differ; the pairing assumption "
            "in cohorts.json no longer holds and this selection rule is invalid."
        )
    reference = set(int(r) for r in cohorts["reference_varied"]["row_ids"])

    if budget < len(panel):
        raise ValueError(f"row budget {budget} cannot hold the {len(panel)}-row panel")

    with h5py.File(dataset, "r") as source:
        equilibria = source["equilibrium_files"][:]
        positive = np.ones(len(equilibria), dtype=bool)
        for group in GRADIENT_GROUPS:
            positive &= source[group]["Q_avgs"][:] > 0.0

    # Sibling tubes of the panel equilibria, keyed by the panel row they join.
    order = np.argsort(equilibria, kind="stable")
    sorted_equilibria = equilibria[order]
    panel_equilibria = equilibria[panel]
    starts = np.searchsorted(sorted_equilibria, panel_equilibria, "left")
    ends = np.searchsorted(sorted_equilibria, panel_equilibria, "right")

    rng = np.random.default_rng(seed)
    chosen = set(int(r) for r in panel)
    cohort_siblings: list[int] = []
    other_siblings: list[list[int]] = []

    for index in range(len(panel)):
        siblings = [
            int(r)
            for r in order[starts[index] : ends[index]]
            if int(r) not in chosen and positive[int(r)]
        ]
        rng.shuffle(siblings)
        in_cohort = [r for r in siblings if r in reference]
        cohort_siblings.extend(in_cohort)
        other_siblings.append([r for r in siblings if r not in reference])

    for row in cohort_siblings:
        if len(chosen) >= budget:
            break
        chosen.add(row)
    cohort_sibling_count = len(chosen) - len(panel)

    # Round-robin, so multiplicity spreads over equilibria instead of piling up.
    depth = 0
    deepest = max((len(s) for s in other_siblings), default=0)
    while len(chosen) < budget and depth < deepest:
        for siblings in other_siblings:
            if len(chosen) >= budget:
                break
            if depth < len(siblings):
                chosen.add(siblings[depth])
        depth += 1

    rows = np.array(sorted(chosen), dtype=np.int64)
    record = {
        "panel_rows": int(len(panel)),
        "cohort_sibling_rows": int(cohort_sibling_count),
        "other_sibling_rows": int(len(rows) - len(panel) - cohort_sibling_count),
        "row_budget": int(budget),
        "seed": int(seed),
        "round_robin_depth": int(depth),
    }
    return rows, record


def ensemble_reference(
    dataset: Path, checkpoint: Path, rows: np.ndarray
) -> dict[str, np.ndarray]:
    """Predictions from the parent file, so the slice can be checked against it."""

    ensemble = load_ensemble(checkpoint, device="cpu")
    reference: dict[str, np.ndarray] = {}
    for gradient_set in ("varied", "fixed"):
        data = load_hdf5_rows(dataset, rows, gradient_set=gradient_set)
        prediction = ensemble.predict(data.geometry, data.a_over_lt, data.a_over_ln)
        reference[f"{gradient_set}_mean_log_heat_flux"] = np.asarray(
            prediction.mean_log_heat_flux, dtype=np.float32
        )
        reference[f"{gradient_set}_std_log_heat_flux"] = np.asarray(
            prediction.std_log_heat_flux, dtype=np.float32
        )
    return reference


def write_slice(
    dataset: Path,
    checkpoint: Path,
    output: Path,
    rows: np.ndarray,
    record: dict,
    cohorts: dict,
) -> None:
    reference_rows = set(int(r) for r in cohorts["reference_varied"]["row_ids"])
    panel_rows = set(int(r) for r in cohorts["interpretation_panel"]["varied_row_ids"])
    compression = dict(compression="gzip", compression_opts=4, shuffle=True)

    predictions = ensemble_reference(dataset, checkpoint, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset, "r") as source, h5py.File(output, "w") as out:
        out.create_dataset(
            "raw_feature_tensor",
            data=source["raw_feature_tensor"][rows].astype(np.float32),
            **compression,
        )
        out.create_dataset(
            "scalar_feature_matrix",
            data=source["scalar_feature_matrix"][rows].astype(np.float32),
            **compression,
        )
        for name in PER_ROW:
            out.create_dataset(name, data=source[name][rows], **compression)
        for group_name in GRADIENT_GROUPS:
            group = out.create_group(group_name)
            for name in source[group_name]:
                group.create_dataset(
                    name, data=source[group_name][name][rows], **compression
                )
        for name in SHARED_METADATA:
            out.create_dataset(name, data=source[name][()])

        index = out.create_group("review_slice")
        index.create_dataset("source_row_ids", data=rows)
        index.create_dataset(
            "is_panel_row",
            data=np.array([int(r) in panel_rows for r in rows], dtype=bool),
        )
        index.create_dataset(
            "is_reference_cohort_row",
            data=np.array([int(r) in reference_rows for r in rows], dtype=bool),
        )
        for name, values in predictions.items():
            index.create_dataset(f"reference_{name}", data=values)

        index.attrs["format_version"] = SLICE_FORMAT_VERSION
        index.attrs["source_dataset"] = str(dataset)
        index.attrs["source_sha256"] = file_sha256(dataset)
        index.attrs["source_row_count"] = len(source["raw_feature_tensor"])
        index.attrs["checkpoint"] = str(checkpoint)
        index.attrs["checkpoint_sha256"] = file_sha256(checkpoint)
        index.attrs["cohorts_schema_version"] = cohorts["schema_version"]
        index.attrs["git_commit"] = git_commit()
        index.attrs["generator"] = "scripts/build_review_slice.py"
        index.attrs["selection"] = json.dumps(record, sort_keys=True)
        index.attrs["geometry_dtype_note"] = (
            "raw_feature_tensor is float32 here; load_hdf5_rows casts the float64 "
            "parent to float32, so the model sees identical values."
        )
        index.attrs["purpose"] = (
            "Verification only. Implementers must not develop, tune, or select "
            "against this slice; see AGENTS.md."
        )


def main() -> None:
    args = build_parser().parse_args()
    cohorts = json.loads(args.cohorts.read_text())
    rows, record = select_rows(args.dataset, cohorts, args.rows, args.seed)
    write_slice(args.dataset, args.checkpoint, args.output, rows, record, cohorts)
    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} — {len(rows)} rows, {size_mb:.2f} MB")
    print(f"  selection: {json.dumps(record, sort_keys=True)}")


if __name__ == "__main__":
    main()
