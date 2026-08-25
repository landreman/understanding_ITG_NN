#!/usr/bin/env python3
"""Build the S14 evidence matrix, claims, and reproducibility index."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from itg_nn.xai.artifacts import RunArtifacts, sha256_file
from itg_nn.xai.synthesis import (
    attach_evidence_manifest_pins,
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


def _selected_source_outcome(
    path: Path,
    selector: dict[str, Any],
    *,
    default_outcome: str,
) -> tuple[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if all(str(row[key]) == str(value) for key, value in selector.items())
        ]
    for field in ("outcome", "estimand"):
        values = {str(row[field]).strip() for row in rows if field in row and row[field]}
        if len(values) > 1:
            raise ValueError(f"selector {selector} has multiple {field} values in {path}")
        if values:
            return values.pop(), f"{path.name}:{field}"
    return default_outcome, "config.estimand"


def _json_path_value(path: Path, keys: list[str]) -> Any:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        value = value[key]
    return value


def _uncertainty_unit(
    spec: dict[str, Any],
    fields: list[str],
    values: list[dict[str, str]],
    *,
    repository: Path,
) -> tuple[str, str]:
    unit_fields = (
        "bootstrap_unit",
        "resampling_unit",
        "split_unit",
        "outer_split_unit",
    )
    for field in unit_fields:
        if field in fields:
            units = {row[field] for row in values if row[field]}
            if len(units) != 1:
                raise ValueError(
                    f"evidence {spec['evidence_id']} has ambiguous {field}: {units}"
                )
            return units.pop(), f"{spec['source_artifact']}:{field}"
    has_interval = any("ci95" in field or "interval_" in field for field in fields)
    sidecar = spec.get("uncertainty_unit_source")
    if sidecar is not None:
        sidecar_path = _repository_path(repository, str(sidecar["artifact"]))
        sidecar_unit = str(_json_path_value(sidecar_path, list(sidecar["json_path"])))
        configured = str(spec["uncertainty_unit"])
        if sidecar_unit != configured:
            raise ValueError(
                f"evidence {spec['evidence_id']} uncertainty unit does not match sidecar"
            )
        return configured, (
            f"{sidecar['artifact']}:" + ".".join(str(key) for key in sidecar["json_path"])
        )
    if has_interval:
        raise ValueError(
            f"evidence {spec['evidence_id']} selects an interval without its grouping unit"
        )
    return "not_applicable_no_interval_selected", "not_applicable"


def _derive_direction(
    spec: dict[str, Any], values: list[dict[str, str]]
) -> tuple[str, str]:
    """Derive candidate-level evidence direction from a declared, auditable rule."""

    rule = spec.get("direction_rule")
    if rule is None:
        direction = str(spec["direction"])
        return direction, "config.direction:interpretive_not_used_for_claim_gate"
    if not isinstance(rule, dict):
        raise ValueError(f"evidence {spec['evidence_id']} direction_rule must be an object")
    kind = str(rule.get("kind", ""))

    def numbers(field: str) -> list[float]:
        try:
            return [float(row[field]) for row in values]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"evidence {spec['evidence_id']} cannot derive direction from {field}"
            ) from error

    statistic = ""
    if kind == "tcav_pass_fraction":
        field = str(rule["field"])
        verdicts = [str(row[field]).lower() == "true" for row in values]
        passed = sum(verdicts)
        direction = (
            "supports"
            if passed == len(verdicts)
            else "regime-dependent"
            if passed
            else "contradicts"
        )
        statistic = f"{field}={passed}/{len(verdicts)}"
    elif kind == "probe_vs_permutation":
        minimum_gain = float(rule["minimum_gain"])
        control = statistics.median(numbers(str(rule["control_field"])))
        overall_gain = statistics.median(numbers(str(rule["encoded_field"]))) - control
        stable_gain = statistics.median(numbers(str(rule["stable_field"]))) - control
        unstable_gain = statistics.median(numbers(str(rule["unstable_field"]))) - control
        direction = (
            "contradicts"
            if overall_gain < minimum_gain
            else "supports"
            if stable_gain >= minimum_gain and unstable_gain >= minimum_gain
            else "mixed"
        )
        statistic = (
            f"median_gains=overall:{overall_gain:.6g},stable:{stable_gain:.6g},"
            f"unstable:{unstable_gain:.6g};minimum_gain={minimum_gain:.6g}"
        )
    elif kind == "resolved_fold_count":
        field = str(rule["field"])
        counts = {int(float(row[field])) for row in values}
        if len(counts) != 1:
            raise ValueError(
                f"evidence {spec['evidence_id']} has ambiguous resolved-fold counts"
            )
        count = counts.pop()
        total = int(rule["total"])
        direction = (
            "supports"
            if count == total
            else "regime-dependent"
            if count > 0
            else "unresolved"
        )
        statistic = f"{field}={count}/{total}"
    elif kind == "recurrence_across_members":
        recurrence = numbers(str(rule["field"]))
        threshold = float(rule["support_threshold"])
        direction = (
            "supports"
            if min(recurrence) >= threshold
            else "regime-dependent"
            if max(recurrence) >= threshold
            else "mixed"
            if max(recurrence) > 0
            else "contradicts"
        )
        statistic = (
            f"range={min(recurrence):.6g}-{max(recurrence):.6g};"
            f"support_threshold={threshold:.6g}"
        )
    elif kind == "signed_spatial_association":
        correlations = numbers(str(rule["correlation_field"]))
        lags = numbers(str(rule["lag_field"]))
        direction = (
            "supports"
            if statistics.median(correlations) > 0 and statistics.median(lags) == 0
            else "mixed"
        )
        statistic = (
            f"median_correlation={statistics.median(correlations):.6g};"
            f"median_lag={statistics.median(lags):.6g}"
        )
    elif kind == "positive_association_with_disjoint_regime_intervals":
        field = str(rule["field"])
        lower_field = str(rule["lower_field"])
        upper_field = str(rule["upper_field"])
        regime_field = str(rule["regime_field"])
        regime_values = [str(item) for item in rule["regime_values"]]
        by_regime = {
            str(row[regime_field]): (
                float(row[field]),
                float(row[lower_field]),
                float(row[upper_field]),
            )
            for row in values
            if str(row[regime_field]) in regime_values
        }
        if set(by_regime) != set(regime_values):
            raise ValueError(
                f"evidence {spec['evidence_id']} lacks declared regime intervals"
            )
        intervals = [by_regime[item] for item in regime_values]
        all_positive = all(lower > 0 for _, lower, _ in intervals)
        disjoint = intervals[0][1] > intervals[1][2] or intervals[1][1] > intervals[0][2]
        direction = (
            "contradicts"
            if not all_positive
            else "regime-dependent"
            if disjoint
            else "supports"
        )
        statistic = ";".join(
            f"{regime}={by_regime[regime][0]:.6g}"
            f"[{by_regime[regime][1]:.6g},{by_regime[regime][2]:.6g}]"
            for regime in regime_values
        )
    elif kind == "nonzero_failure_count":
        selected = values
        if "scope_field" in rule:
            selected = [
                row
                for row in values
                if str(row[str(rule["scope_field"])]) == str(rule["scope_value"])
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"evidence {spec['evidence_id']} lacks one declared failure-count scope"
                )
        counts = [float(row[str(rule["field"])]) for row in selected]
        direction = "contradicts" if sum(counts) > 0 else "null_control"
        statistic = f"total_failures={sum(counts):.6g}"
    else:
        raise ValueError(
            f"evidence {spec['evidence_id']} has unknown direction rule {kind!r}"
        )
    configured = spec.get("direction")
    if configured is not None and str(configured) != direction:
        raise ValueError(
            f"evidence {spec['evidence_id']} configured direction {configured!r} "
            f"does not match derived direction {direction!r}"
        )
    return direction, f"config.direction_rule:{kind};{statistic}"


def _evidence_rows(
    specs: list[dict[str, Any]], repository: Path, *, estimand: str
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = _repository_path(repository, str(spec["source_artifact"]))
        if path.suffix.lower() != ".csv":
            raise ValueError(f"S14 evidence source must be CSV: {path}")
        selector = dict(spec.get("selector", {}))
        fields = [str(field) for field in spec["source_fields"]]
        values = _select_csv_rows(path, selector, fields)
        outcome, outcome_source = _selected_source_outcome(
            path, selector, default_outcome=estimand
        )
        configured_outcome = spec.get("outcome")
        if configured_outcome is not None and str(configured_outcome) != outcome:
            raise ValueError(
                f"evidence {spec['evidence_id']} outcome {configured_outcome!r} "
                f"does not match source {outcome!r}"
            )
        uncertainty_unit, uncertainty_unit_source = _uncertainty_unit(
            spec, fields, values, repository=repository
        )
        direction, direction_source = _derive_direction(spec, values)
        direction_rule = spec.get(
            "direction_rule",
            {
                "kind": "interpretive",
                "value": direction,
                "basis": "upstream result interpretation; not used by the headline gate",
            },
        )
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
                "direction": direction,
                "direction_rule": json.dumps(
                    direction_rule, sort_keys=True, separators=(",", ":")
                ),
                "direction_source": direction_source,
                "estimand": estimand,
                "outcome": outcome,
                "outcome_source": outcome_source,
                "function_scope": spec["function_scope"],
                "cohort": spec["cohort"],
                "regime": spec["regime"],
                "validity_tag": spec["validity_tag"],
                "intervention": spec["intervention"],
                "intervention_executed": bool(spec["intervention_executed"]),
                "machine_readable": True,
                "uncertainty_unit": uncertainty_unit,
                "uncertainty_unit_source": uncertainty_unit_source,
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
    evidence = _evidence_rows(
        evidence_specs, repository, estimand=str(resolved["estimand"])
    )
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
    manifests = attach_evidence_manifest_pins(
        manifests,
        evidence,
        repository=repository,
        require_all=resolved["mode"] == "production",
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
        "estimand": resolved["estimand"],
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
            or int(row["corroborating_method_family_count"]) >= 2
            for row in claims
        ),
        "physical_causal_statement_count": sum(
            bool(row["physical_causal_statement"]) for row in claims
        ),
        "all_physical_causal_statements_name_executed_interventions": all(
            not bool(row["physical_causal_statement"])
            or str(row["physical_intervention"]) != "not_causal"
            for row in claims
        ),
        "indexed_upstream_run_count": sum(
            bool(row["is_run_manifest"]) for row in manifests
        ),
        "indexed_provenance_record_count": len(manifests),
        "all_indexed_runs_recreatable_from_manifests": all(
            bool(row["recreates_claims"])
            for row in manifests
            if bool(row["is_run_manifest"])
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
            "source_run_manifest_count": sum(
                bool(row["is_run_manifest"]) for row in manifests
            ),
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
