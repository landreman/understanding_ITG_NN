from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[2] / "scripts/xai_s14_synthesis.py"
    spec = importlib.util.spec_from_file_location("xai_s14_synthesis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_exposes_registered_step_interface() -> None:
    module = _load_script()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/xai/S14_synthesis.json",
            "--members",
            "2",
            "--rows",
            "64",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--resume",
        ]
    )
    assert args.members == 2
    assert args.rows == 64
    assert args.device == "cpu"
    assert args.seed == 7
    assert args.resume


def test_csv_selector_returns_every_exact_row_and_rejects_empty_selection(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source = tmp_path / "source.csv"
    source.write_text("kind,value\nkept,1\nkept,2\ndropped,3\n", encoding="utf-8")
    selected = module._select_csv_rows(source, {"kind": "kept"}, ["value"])
    assert selected == [{"value": "1"}, {"value": "2"}]
    with pytest.raises(ValueError, match="matched no rows"):
        module._select_csv_rows(source, {"kind": "missing"}, ["value"])


def test_pilot_run_finalizes_outputs_without_publishing(tmp_path: Path) -> None:
    module = _load_script()
    repository = Path(__file__).parents[2]
    output_dir = tmp_path / "pilot-output"
    published_dir = tmp_path / "must-not-publish"
    args = module.build_parser().parse_args(
        [
            "--config",
            "configs/xai/S14_synthesis.json",
            "--pilot",
            "--no-publish",
            "--dataset",
            str(repository / "models/cyclic_ensemble_pre2.pt"),
            "--output-dir",
            str(output_dir),
            "--published-dir",
            str(published_dir),
        ]
    )
    manifest_path = module.run(args)
    assert manifest_path == output_dir / "manifest.json"
    assert not published_dir.exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_count"] == 4
    assert summary["review_slice_used_for_development_or_reporting"] is False
    ledger = (output_dir / "evidence_ledger.csv").read_text(encoding="utf-8")
    assert "outcome_source" in ledger
    assert "uncertainty_unit" in ledger


def test_interval_without_grouping_unit_is_rejected(tmp_path: Path) -> None:
    module = _load_script()
    spec = {
        "evidence_id": "interval-without-unit",
        "source_artifact": "source.csv",
    }
    with pytest.raises(ValueError, match="without its grouping unit"):
        module._uncertainty_unit(
            spec,
            ["effect", "effect_ci95_lower", "effect_ci95_upper"],
            [{"effect": "1", "effect_ci95_lower": "0", "effect_ci95_upper": "2"}],
            repository=tmp_path,
        )


@pytest.mark.parametrize(
    ("rule", "values", "expected"),
    [
        (
            {"kind": "tcav_pass_fraction", "field": "use_claim_permitted"},
            [{"use_claim_permitted": "True"}] * 10
            + [{"use_claim_permitted": "False"}] * 5,
            "regime-dependent",
        ),
        (
            {
                "kind": "probe_vs_permutation",
                "encoded_field": "encoded_r2",
                "stable_field": "encoded_r2_stable_or_near_floor",
                "unstable_field": "encoded_r2_unstable",
                "control_field": "permuted_r2",
                "minimum_gain": 0.1,
            },
            [
                {
                    "encoded_r2": "0.3",
                    "encoded_r2_stable_or_near_floor": "-0.1",
                    "encoded_r2_unstable": "0.5",
                    "permuted_r2": "0.0",
                }
            ],
            "mixed",
        ),
        (
            {
                "kind": "probe_vs_permutation",
                "encoded_field": "encoded_r2",
                "stable_field": "encoded_r2_stable_or_near_floor",
                "unstable_field": "encoded_r2_unstable",
                "control_field": "permuted_r2",
                "minimum_gain": 0.1,
            },
            [
                {
                    "encoded_r2": "0.05",
                    "encoded_r2_stable_or_near_floor": "-0.2",
                    "encoded_r2_unstable": "-0.3",
                    "permuted_r2": "0.0",
                }
            ],
            "contradicts",
        ),
        (
            {
                "kind": "resolved_fold_count",
                "field": "aipw_resolved_fold_count",
                "total": 7,
            },
            [{"aipw_resolved_fold_count": "0"}],
            "unresolved",
        ),
    ],
)
def test_direction_is_derived_from_declared_source_rule(
    rule: dict[str, object], values: list[dict[str, str]], expected: str
) -> None:
    module = _load_script()
    direction, source = module._derive_direction(
        {"evidence_id": "derived", "direction_rule": rule}, values
    )
    assert direction == expected
    assert "direction_rule" in source
