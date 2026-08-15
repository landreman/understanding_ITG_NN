"""Consolidate selected legacy state dictionaries into one inference bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch


ARCHITECTURE_COLUMNS = {
    "kernel_sizes": [f"p:kernel_size{index}" for index in range(1, 6)],
    "convolution_channels": [
        f"p:conv_channels{index}" for index in range(1, 6)
    ],
    "dense_dimensions": [f"p:fc_dim{index}" for index in range(1, 3)],
}


def selected_rows(results_path: Path, member_count: int) -> list[dict[str, str]]:
    with results_path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["objective"] != "F"]
    rows.sort(key=lambda row: float(row["objective"]), reverse=True)
    selected = rows[:member_count]
    if len(selected) != member_count:
        raise ValueError(
            f"Requested {member_count} models but found only {len(selected)}"
        )
    if {int(row["p:pre_method"]) for row in selected} != {2}:
        raise ValueError("Selected models do not all use preprocessing method 2")
    if {row["p:use_batch_norm"].lower() for row in selected} != {"false"}:
        raise ValueError("Selected models unexpectedly include batch normalization")
    return selected


def architecture_from_row(row: dict[str, str]) -> dict[str, list[int]]:
    return {
        name: [int(row[column]) for column in columns]
        for name, columns in ARCHITECTURE_COLUMNS.items()
    }


def build_bundle(
    results_path: Path, model_directory: Path, member_count: int
) -> dict[str, Any]:
    members = []
    for row in selected_rows(results_path, member_count):
        member_id = row["m:task_id"]
        state_path = model_directory / f"model_{member_id}.pth"
        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        members.append(
            {
                "id": member_id,
                "validation_r2": float(row["objective"]),
                "architecture": architecture_from_row(row),
                "state_dict": state_dict,
            }
        )
    return {
        "format_version": 1,
        "target_transform": "log_clip_min_-2",
        "members": members,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--members", type=int, default=100)
    args = parser.parse_args()

    bundle = build_bundle(args.results_csv, args.model_directory, args.members)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output_path)
    print(f"Wrote {len(bundle['members'])} models to {args.output_path}")


if __name__ == "__main__":
    main()
