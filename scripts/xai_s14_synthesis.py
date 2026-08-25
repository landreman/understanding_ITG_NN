#!/usr/bin/env python3
"""Build the S14 evidence matrix, claims, and reproducibility index."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.synthesis import (
    NATIVE_ESTIMAND,
    validate_claim_register,
    validate_evidence_ledger,
    validate_evidence_matrix,
    validate_reproducibility_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xai/S14_synthesis.json")
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--published-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--members", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    return parser


def _resolve(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if args.pilot:
        resolved.update(config["pilot"])
    resolved["mode"] = "pilot" if args.pilot else "production"
    for name in ("device", "seed", "members", "rows"):
        value = getattr(args, name)
        if value is not None:
            resolved[name] = value
    for value, name in (
        (args.dataset, "dataset"),
        (args.checkpoint, "checkpoint"),
        (args.published_dir, "published_dir"),
    ):
        if value is not None:
            resolved[name] = str(value)
    resolved["resume"] = bool(args.resume)
    return resolved


def _csv_text(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
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


def _repository_path(repository: Path, value: str) -> Path:
    path = (repository / value).resolve()
    if path != repository and repository not in path.parents:
        raise ValueError(f"configured path escapes repository: {value}")
    return path


def _select_csv_rows(
    path: Path,
    selector: dict[str, Any],
    fields: list[str],
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        available = set(reader.fieldnames or ())
        missing_selector = sorted(set(selector) - available)
        missing_fields = sorted(set(fields) - available)
        if missing_selector or missing_fields:
            raise ValueError(
                f"{path} lacks selector fields {missing_selector} or source fields {missing_fields}"
            )
        selected = [
            row
            for row in reader
            if all(str(row[key]) == str(value) for key, value in selector.items())
        ]
    if not selected:
        raise ValueError(f"selector {selector} matched no rows in {path}")
    return [{field: row[field] for field in fields} for row in selected]


def _evidence_rows(
    specs: list[dict[str, Any]], repository: Path
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = _repository_path(repository, str(spec["source_artifact"]))
        if path.suffix.lower() != ".csv":
            raise ValueError(f"S14 evidence source must be CSV: {path}")
        selector = dict(spec.get("selector", {}))
        fields = [str(field) for field in spec["source_fields"]]
        values = _select_csv_rows(path, selector, fields)
        rows.append(
            {
                "evidence_id": spec["evidence_id"],
                "candidate_id": spec["candidate_id"],
                "matrix_column": spec["matrix_column"],
                "source_step": spec["source_step"],
                "source_artifact": spec["source_artifact"],
                "source_selector": json.dumps(selector, sort_keys=True, separators=(",", ":")),
                "source_fields": ";".join(fields),
                "source_record_count": len(values),
                "source_values_json": json.dumps(
                    values, sort_keys=True, separators=(",", ":")
                ),
                "method_family": spec["method_family"],
                "direction": spec["direction"],
                "estimand": NATIVE_ESTIMAND,
                "function_scope": spec["function_scope"],
                "cohort": spec["cohort"],
                "regime": spec["regime"],
                "validity_tag": spec["validity_tag"],
                "intervention": spec["intervention"],
                "intervention_executed": bool(spec["intervention_executed"]),
                "machine_readable": True,
                "summary": spec["summary"],
            }
        )
    return validate_evidence_ledger(rows)


def _pilot_specs(
    resolved: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matrix = list(resolved["matrix"])
    claims = list(resolved["claims"])
    evidence = list(resolved["evidence"])
    if resolved["mode"] == "pilot":
        matrix = matrix[: int(resolved["candidate_limit"])]
        claims = claims[: int(resolved["claim_limit"])]
        required_ids: set[str] = set()
        for row in matrix:
            for value in row.values():
                required_ids.update(
                    item for item in str(value).split(";") if item.startswith("E")
                )
        for claim in claims:
            required_ids.update(str(claim["evidence_ids"]).split(";"))
        evidence = [row for row in evidence if row["evidence_id"] in required_ids]
    return evidence, matrix, claims


def _manifest_rows(
    sources: list[dict[str, Any]],
    *,
    repository: Path,
    artifacts: RunArtifacts,
    publish: bool,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        source_path = _repository_path(repository, str(source["source_path"]))
        published_path = _repository_path(repository, str(source["published_path"]))
        available = source_path if source_path.is_file() else published_path
        if not available.is_file():
            raise FileNotFoundError(
                f"neither source nor published manifest exists for {source['run_key']}"
            )
        index_path = available
        if publish and bool(source["copy_to_s14"]):
            content = available.read_bytes()
            output_name = f"upstream_manifest_{source['run_key']}.json"
            artifacts.write_text(output_name, content.decode("utf-8"))
            published_path.parent.mkdir(parents=True, exist_ok=True)
            published_path.write_bytes(content)
            index_path = published_path
        elif publish:
            index_path = published_path
        rows.append(
            {
                "run_key": source["run_key"],
                "step": source["step"],
                "run_id": source["run_id"],
                "manifest_path": str(index_path.relative_to(repository)),
                "manifest_sha256": sha256_file(index_path),
                "role": source["role"],
                "recreates_claims": bool(source["recreates_claims"]),
                "caveat": source["caveat"],
            }
        )
    return validate_reproducibility_index(rows, repository=repository)


def _source_hashes(
    evidence: tuple[dict[str, Any], ...],
    manifests: tuple[dict[str, Any], ...],
    repository: Path,
) -> dict[str, Any]:
    evidence_paths = sorted({str(row["source_artifact"]) for row in evidence})
    return {
        "evidence_artifacts": {
            path: sha256_file(_repository_path(repository, path))
            for path in evidence_paths
        },
        "manifests": {
            str(row["run_key"]): {
                "path": row["manifest_path"],
                "sha256": row["manifest_sha256"],
            }
            for row in manifests
        },
    }


def run(args: argparse.Namespace) -> Path:
    repository = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else repository / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = _resolve(config, args)
    publish = not args.no_publish and resolved["mode"] == "production"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repository / "output" / "xai" / "S14" / str(resolved["run_id"])
    )
    published_dir = (
        None
        if not publish
        else _repository_path(repository, str(resolved["published_dir"]))
    )
    artifacts = RunArtifacts(output_dir)

    evidence_specs, matrix_specs, claim_specs = _pilot_specs(resolved)
    evidence = _evidence_rows(evidence_specs, repository)
    matrix = validate_evidence_matrix(matrix_specs, evidence)
    claims = validate_claim_register(claim_specs, evidence)
    manifest_sources = list(resolved["manifest_sources"])
    if resolved["mode"] == "pilot":
        manifest_sources = manifest_sources[: int(resolved["manifest_limit"])]
    manifests = _manifest_rows(
        manifest_sources,
        repository=repository,
        artifacts=artifacts,
        publish=publish,
    )
    next_experiments = [dict(row) for row in resolved["next_experiments"]]
    if [int(row["priority"]) for row in next_experiments] != list(
        range(1, len(next_experiments) + 1)
    ):
        raise ValueError("next-experiment priorities must be consecutive from 1")

    source_hashes = _source_hashes(evidence, manifests, repository)
    artifacts.write_text("evidence_ledger.csv", _csv_text(evidence))
    artifacts.write_text("evidence_matrix.csv", _csv_text(matrix))
    artifacts.write_text("claim_register.csv", _csv_text(claims))
    artifacts.write_text("reproducibility_index.csv", _csv_text(manifests))
    artifacts.write_text("next_experiments.csv", _csv_text(next_experiments))
    artifacts.write_json("source_hashes.json", source_hashes)
    status_counts: dict[str, int] = {}
    for row in matrix:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    summary = {
        "step": "S14",
        "run_id": resolved["run_id"],
        "synthesis_kind": "registered committed evidence only; no new model or GX computation",
        "estimand": NATIVE_ESTIMAND,
        "canonical_function": "invariant_tilde_f",
        "function_scopes_separated": [
            "original_f",
            "ensemble_mean",
            "phase_average_bar_f",
            "invariant_tilde_f",
            "observed_GX",
        ],
        "varied_panel_rows": 1000,
        "varied_stable_or_near_floor_rows": 240,
        "varied_unstable_rows": 760,
        "fixed_panel_rows": 1000,
        "fixed_stable_or_near_floor_rows": 23,
        "fixed_unstable_rows": 977,
        "candidate_count": len(matrix),
        "evidence_record_count": len(evidence),
        "headline_claim_count": sum(bool(row["headline"]) for row in claims),
        "all_headlines_have_two_independent_method_families": all(
            not bool(row["headline"])
            or int(row["independent_method_family_count"]) >= 2
            for row in claims
        ),
        "causal_statement_count": sum(bool(row["causal_statement"]) for row in claims),
        "all_causal_statements_name_executed_interventions": all(
            not bool(row["causal_statement"])
            or str(row["intervention"]) != "not_causal"
            for row in claims
        ),
        "indexed_upstream_run_count": len(manifests),
        "all_indexed_runs_recreatable_from_manifests": all(
            bool(row["recreates_claims"]) for row in manifests
        ),
        "matrix_status_counts": status_counts,
        "smallest_next_calculation": next_experiments[0],
        "model_outputs_computed": False,
        "gx_outputs_computed": False,
        "review_slice_used_for_development_or_reporting": False,
    }
    artifacts.write_json("summary.json", summary)

    manifest_path = artifacts.finalize(
        config=resolved,
        dataset=_repository_path(repository, str(resolved["dataset"]))
        if not Path(str(resolved["dataset"])).is_absolute()
        else Path(str(resolved["dataset"])),
        checkpoint=_repository_path(repository, str(resolved["checkpoint"])),
        member_ids=(),
        row_ids=(),
        gradient_set="synthesis of S01 registered fixed and varied panels",
        device=str(resolved["device"]),
        repository=repository,
        command=sys.argv,
        published_dir=published_dir,
        extra_manifest={
            "model_outputs_computed": False,
            "gx_outputs_computed": False,
            "source_manifest_count": len(manifests),
            "source_evidence_artifact_count": len(source_hashes["evidence_artifacts"]),
            "synthesis_source_hashes": source_hashes,
            "review_slice_used": False,
            "pre_existing_user_changes_preserved": [
                "reports/xai/S13_executive_summary.md",
                "output/",
                "scratch/",
                "reports/xai/XAI_most_interesting_results.md",
                "reports/xai/ideas_to_try.md",
            ],
        },
    )

    if published_dir is not None:
        published_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "evidence_ledger.csv",
            "evidence_matrix.csv",
            "claim_register.csv",
            "reproducibility_index.csv",
            "next_experiments.csv",
            "source_hashes.json",
            "summary.json",
        ):
            (published_dir / name).write_bytes((output_dir / name).read_bytes())
    return manifest_path


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
