import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "xai_s11_disagreement.py"
CONFIG = ROOT / "configs" / "xai" / "S11_disagreement.json"


def _module():
    spec = importlib.util.spec_from_file_location("xai_s11_disagreement", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s11_cli_exposes_required_reproducibility_controls():
    parser = _module().build_parser()
    destinations = {action.dest for action in parser._actions}
    assert {"config", "members", "rows", "device", "seed", "resume", "output_dir"} <= destinations


def test_s11_config_freezes_native_thresholds_features_and_grouping():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["canonical_function"] == "invariant_tilde_f"
    assert config["estimand"] == "native max(log Q, -2)"
    assert config["resampling_unit"] == "equilibrium_files"
    assert config["high_error_threshold_native"] > 0
    assert config["high_spread_threshold_native"] > 0
    assert config["diagnostic_features"] == [
        "support_warning_score",
        "equilibrium_class",
        "a_over_lt",
        "a_over_ln",
        "q_stds",
        "symmetry_error",
        "motif_activation_dispersion",
        "concept_activation_dispersion",
        "nfp",
        "iota",
        "shat",
        "d_pressure_d_s",
        "aspect",
        "rho",
        "aspect_over_rho",
    ]
    assert config["feature_selection"] == "none_frozen_before_residual_analysis"


def test_crossfit_table_accepts_a_single_equilibrium_class_pilot():
    rows = _module()._crossfit_rows(
        {"support_warning_score": np.arange(8.0), "q_stds": np.arange(8.0) ** 2},
        np.zeros(8, dtype=int),
        {"ensemble_spread": np.linspace(0.1, 0.8, 8)},
        np.asarray([f"eq{index}" for index in range(8)]),
        folds=4,
        alpha=1.0,
        seed=3,
    )
    assert len(rows) == 1
    assert rows[0]["split_unit"] == "equilibrium_files"
