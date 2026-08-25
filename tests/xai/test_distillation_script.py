from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts import xai_s12_distillation as script


ROOT = Path(__file__).resolve().parents[2]


def test_s12_config_freezes_estimand_grouping_features_and_interactions() -> None:
    config = json.loads((ROOT / "configs/xai/S12_distillation.json").read_text())
    assert config["feature_table_version"] == "S12-v1"
    assert config["split_unit"] == "equilibrium_files"
    assert config["stable_threshold_log_Q"] == -1.9
    assert config["members"] == 3
    assert len(config["registered_interactions"]) == 5
    assert config["fidelity_bootstrap_replicates"] == 2000
    assert [row["name"] for row in config["nested_feature_sets"]] == [
        "drives_only",
        "baseline_trio",
        "paper_five",
        "all_17_main_effects",
        "all_17_registered_interactions",
        "baseline_trio_aLT_logfQ_interaction",
    ]
    assert ["a_over_LT", "bad_curvature_compression"] in config[
        "registered_interactions"
    ]
    assert script.NATIVE_ESTIMAND == "native max(log Q, -2)"


def test_nested_subset_spec_maps_names_and_interactions() -> None:
    names = ("a", "b", "c")
    positions, subset_names, interactions = script._nested_subset_spec(
        {"features": ["c", "a"], "interactions": [["a", "c"]]},
        names,
        [("a", "b")],
    )
    np.testing.assert_array_equal(positions, [2, 0])
    assert subset_names == ("c", "a")
    assert interactions == ((1, 0),)


def test_s12_cli_exposes_required_production_controls() -> None:
    options = {action.dest for action in script.build_parser()._actions}
    assert {
        "config",
        "members",
        "rows",
        "device",
        "seed",
        "resume",
        "output_dir",
        "batch_size",
    }.issubset(options)


def test_interaction_effect_export_trims_edge_names_to_score_cells() -> None:
    class Explanation:
        @staticmethod
        def data(index):
            assert index == 0
            return {
                "type": "interaction",
                "left_names": [0.0, 1.0, 2.0],
                "right_names": [0.0, 1.0, 2.0, 3.0],
                "scores": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            }

    class Estimator:
        term_names_ = ["a & b"]

        @staticmethod
        def explain_global():
            return Explanation()

    rows = script._global_effect_rows("member_output", "m0", Estimator())
    assert len(rows) == 6
    assert rows[-1]["signed_effect_native_units"] == 6.0


def test_member_value_wiring_keeps_signed_native_output() -> None:
    class Panel:
        geometry = torch.zeros((3, 96, 7))
        a_over_lt = torch.ones(3)
        a_over_ln = torch.zeros(3)

    class Member:
        @staticmethod
        def invariant_bottleneck(geometry):
            return torch.column_stack((geometry[:, 0, 0], torch.ones(len(geometry))))

        @staticmethod
        def __call__(geometry, a_over_lt, a_over_ln):
            return torch.asarray([-2.0, -0.5, 1.0])

    bottleneck, prediction = script._member_values(
        Member(), Panel(), batch_size=3, device=torch.device("cpu")
    )
    np.testing.assert_array_equal(bottleneck[:, 1], np.ones(3))
    np.testing.assert_array_equal(prediction, np.asarray([-2.0, -0.5, 1.0]))
