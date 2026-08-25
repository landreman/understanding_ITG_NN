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
