from __future__ import annotations

import importlib.util
from pathlib import Path


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

