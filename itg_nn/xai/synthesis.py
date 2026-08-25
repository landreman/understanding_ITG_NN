"""Validation and assembly helpers for the S14 evidence synthesis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


NATIVE_ESTIMAND = "native max(log Q, -2)"

MATRIX_EVIDENCE_COLUMNS = (
    "bottleneck_shapley",
    "unit_semantics",
    "input_attribution",
    "supported_perturbation",
    "hidden_encoding",
    "hidden_intervention",
    "cross_model_consensus",
    "distillation",
    "Q_z",
    "zonal_flow",
    "natural_experiment",
)

_EVIDENCE_REQUIRED = {
    "evidence_id",
    "candidate_id",
    "matrix_column",
    "source_step",
    "source_artifact",
    "source_selector",
    "source_fields",
    "source_record_count",
    "method_family",
    "direction",
    "estimand",
    "function_scope",
    "cohort",
    "regime",
    "validity_tag",
    "intervention",
    "intervention_executed",
    "machine_readable",
    "summary",
}
_MATRIX_REQUIRED = {
    "candidate_id",
    "hypothesis",
    "status",
    "claim_grade",
    "function_scope",
    "regime",
    "uncertainty",
    "negative_results",
    *MATRIX_EVIDENCE_COLUMNS,
}
_CLAIM_REQUIRED = {
    "claim_id",
    "headline",
    "claim_text",
    "status",
    "scope",
    "causal_statement",
    "intervention",
    "evidence_ids",
    "limitations",
}
_MANIFEST_REQUIRED = {
    "command",
    "config",
    "checkpoint",
    "dataset",
    "device",
    "git_commit",
    "git_dirty",
    "gradient_set",
    "member_ids",
    "output_hashes",
    "package_versions",
    "python",
    "row_ids",
    "seed",
    "wall_time_seconds",
}
_VALIDITY_TAGS = {
    "exact-symmetry",
    "observed-comparison",
    "plausibly-local",
    "deliberately_off_manifold_diagnostic",
}
_DIRECTIONS = {
    "supports",
    "contradicts",
    "mixed",
    "regime-dependent",
    "null_control",
    "descriptive",
    "unresolved",
}
_MATRIX_STATUSES = {"supported", "contradicted", "regime-dependent", "unresolved"}


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field} must be boolean")


def _missing(row: Mapping[str, Any], required: set[str]) -> list[str]:
    return sorted(required - set(row))


def _ids(value: Any) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(";") if item.strip())


def validate_evidence_ledger(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if not rows:
        raise ValueError("evidence ledger must not be empty")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(rows):
        missing = _missing(source, _EVIDENCE_REQUIRED)
        if missing:
            raise ValueError(f"evidence row {index} missing fields: {missing}")
        row = dict(source)
        evidence_id = str(row["evidence_id"]).strip()
        if not evidence_id or evidence_id in seen:
            raise ValueError(f"evidence_id must be nonempty and unique: {evidence_id!r}")
        seen.add(evidence_id)
        if row["matrix_column"] not in MATRIX_EVIDENCE_COLUMNS:
            raise ValueError(f"unknown evidence-matrix column {row['matrix_column']!r}")
        if row["estimand"] != NATIVE_ESTIMAND:
            raise ValueError(
                f"evidence {evidence_id} must retain the native max(log Q, -2) estimand"
            )
        if row["validity_tag"] not in _VALIDITY_TAGS:
            raise ValueError(f"evidence {evidence_id} has invalid validity tag")
        if row["direction"] not in _DIRECTIONS:
            raise ValueError(f"evidence {evidence_id} has invalid direction")
        if int(row["source_record_count"]) < 1:
            raise ValueError(f"evidence {evidence_id} selects no source records")
        if not _as_bool(row["machine_readable"], field="machine_readable"):
            raise ValueError(f"evidence {evidence_id} must have a machine-readable source")
        row["intervention_executed"] = _as_bool(
            row["intervention_executed"], field="intervention_executed"
        )
        row["machine_readable"] = True
        for field in (
            "candidate_id",
            "source_step",
            "source_artifact",
            "source_selector",
            "source_fields",
            "method_family",
            "function_scope",
            "cohort",
            "regime",
            "intervention",
            "summary",
        ):
            if not str(row[field]).strip():
                raise ValueError(f"evidence {evidence_id} has empty {field}")
        result.append(row)
    return tuple(result)


def validate_evidence_matrix(
    matrix_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    evidence = {str(row["evidence_id"]): dict(row) for row in evidence_rows}
    if not matrix_rows:
        raise ValueError("evidence matrix must not be empty")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(matrix_rows):
        missing = _missing(source, _MATRIX_REQUIRED)
        if missing:
            raise ValueError(f"matrix row {index} missing fields: {missing}")
        row = dict(source)
        candidate_id = str(row["candidate_id"]).strip()
        if not candidate_id or candidate_id in seen:
            raise ValueError(f"candidate_id must be nonempty and unique: {candidate_id!r}")
        seen.add(candidate_id)
        if row["status"] not in _MATRIX_STATUSES:
            raise ValueError(f"candidate {candidate_id} has invalid status")
        linked: list[str] = []
        for column in MATRIX_EVIDENCE_COLUMNS:
            value = str(row[column]).strip()
            if not value:
                raise ValueError(
                    f"candidate {candidate_id} must make absence explicit in {column}"
                )
            if value.startswith("not_"):
                continue
            for evidence_id in _ids(value):
                if evidence_id not in evidence:
                    raise ValueError(
                        f"candidate {candidate_id} references unknown evidence {evidence_id}"
                    )
                linked_row = evidence[evidence_id]
                if linked_row["candidate_id"] != candidate_id:
                    raise ValueError(
                        f"evidence {evidence_id} belongs to {linked_row['candidate_id']}, "
                        f"not {candidate_id}"
                    )
                if linked_row["matrix_column"] != column:
                    raise ValueError(
                        f"evidence {evidence_id} belongs in {linked_row['matrix_column']}, "
                        f"not {column}"
                    )
                linked.append(evidence_id)
        if not linked:
            raise ValueError(f"candidate {candidate_id} has no linked evidence")
        row["linked_evidence_count"] = len(set(linked))
        row["linked_method_family_count"] = len(
            {str(evidence[item]["method_family"]) for item in linked}
        )
        result.append(row)
    return tuple(result)


def validate_claim_register(
    claims: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    evidence = {str(row["evidence_id"]): dict(row) for row in evidence_rows}
    if not claims:
        raise ValueError("claim register must not be empty")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(claims):
        missing = _missing(source, _CLAIM_REQUIRED)
        if missing:
            raise ValueError(f"claim row {index} missing fields: {missing}")
        row = dict(source)
        claim_id = str(row["claim_id"]).strip()
        if not claim_id or claim_id in seen:
            raise ValueError(f"claim_id must be nonempty and unique: {claim_id!r}")
        seen.add(claim_id)
        ids = _ids(row["evidence_ids"])
        if not ids:
            raise ValueError(f"claim {claim_id} has no evidence")
        unknown = sorted(set(ids) - set(evidence))
        if unknown:
            raise ValueError(f"claim {claim_id} references unknown evidence: {unknown}")
        families = sorted({str(evidence[item]["method_family"]) for item in ids})
        headline = _as_bool(row["headline"], field="headline")
        if headline and len(families) < 2:
            raise ValueError(
                f"headline claim {claim_id} needs at least two independent method families"
            )
        causal = _as_bool(row["causal_statement"], field="causal_statement")
        intervention = str(row["intervention"]).strip()
        if causal:
            if not intervention or intervention == "not_causal":
                raise ValueError(f"causal claim {claim_id} must identify its intervention")
            if not any(
                evidence[item]["intervention_executed"]
                and evidence[item]["intervention"] == intervention
                for item in ids
            ):
                raise ValueError(
                    f"causal claim {claim_id} lacks evidence from the named intervention"
                )
        elif intervention != "not_causal":
            raise ValueError(
                f"non-causal claim {claim_id} must use intervention='not_causal'"
            )
        row["headline"] = headline
        row["causal_statement"] = causal
        row["evidence_source_count"] = len(set(ids))
        row["independent_method_family_count"] = len(families)
        row["independent_method_families"] = ";".join(families)
        row["machine_readable_sources"] = ";".join(
            sorted({str(evidence[item]["source_artifact"]) for item in ids})
        )
        result.append(row)
    return tuple(result)


def validate_reproducibility_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    repository: str | Path,
) -> tuple[dict[str, Any], ...]:
    if not rows:
        raise ValueError("reproducibility index must not be empty")
    root = Path(repository).resolve()
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(rows):
        required = {
            "run_key",
            "step",
            "run_id",
            "manifest_path",
            "manifest_sha256",
            "role",
            "recreates_claims",
            "caveat",
        }
        missing = _missing(source, required)
        if missing:
            raise ValueError(f"reproducibility row {index} missing fields: {missing}")
        row = dict(source)
        run_key = str(row["run_key"]).strip()
        if not run_key or run_key in seen:
            raise ValueError(f"run_key must be nonempty and unique: {run_key!r}")
        seen.add(run_key)
        path = (root / str(row["manifest_path"])).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"manifest for {run_key} escapes repository")
        if not path.is_file():
            raise ValueError(f"manifest for {run_key} does not exist: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["manifest_sha256"]:
            raise ValueError(f"manifest hash mismatch for {run_key}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"manifest for {run_key} is not valid JSON") from error
        missing_manifest = sorted(_MANIFEST_REQUIRED - set(manifest))
        if missing_manifest:
            raise ValueError(
                f"manifest for {run_key} missing required provenance: {missing_manifest}"
            )
        if not manifest["command"] or not isinstance(manifest["command"], list):
            raise ValueError(f"manifest for {run_key} has no recreation command")
        manifest_run_id = manifest["config"].get("run_id")
        if manifest_run_id != row["run_id"]:
            raise ValueError(
                f"run_id mismatch for {run_key}: {manifest_run_id!r} != {row['run_id']!r}"
            )
        row["recreates_claims"] = _as_bool(
            row["recreates_claims"], field="recreates_claims"
        )
        row["output_count"] = len(manifest["output_hashes"])
        row["member_count"] = len(manifest["member_ids"])
        row["row_count"] = len(manifest["row_ids"])
        row["git_commit"] = manifest["git_commit"]
        row["git_tracked_dirty"] = manifest.get("git_tracked_dirty", "not_recorded")
        row["dataset_sha256"] = manifest["dataset"].get("sha256", "")
        row["checkpoint_sha256"] = manifest["checkpoint"].get("sha256", "")
        row["command"] = json.dumps(manifest["command"], separators=(",", ":"))
        row["config_path_or_inline"] = str(row.get("config_path_or_inline", "manifest.config"))
        result.append(row)
    return tuple(result)
