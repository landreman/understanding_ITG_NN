from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).parents[2] / "scripts/xai_s13_physical_validation.py"
    spec = importlib.util.spec_from_file_location("xai_s13_physical_validation", path)
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
            "configs/xai/S13_physical_validation.json",
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


def test_gx_spec_is_proposal_only_and_budget_arithmetic_is_explicit() -> None:
    module = _load_script()
    ranking = [
        {"candidate": "localized", "matched_native_effect": 0.4},
        {"candidate": "geodesic", "matched_native_effect": -0.2},
    ]
    design = {
        "anchor_equilibria": 3,
        "candidate_directions": 2,
        "signed_steps_per_direction": 2,
        "drive_points": [[3.0, 0.9], [4.5, 0.9]],
        "standard_run_node_hours": 0.5,
        "convergence_cases": 6,
        "convergence_run_node_hours": 2.0,
        "vmec_search_node_hours": 2.0,
        "contingency_fraction": 0.25,
    }
    spec = module._gx_spec(ranking, design)
    assert spec["status"] == "proposal_only_researcher_approval_required"
    assert spec["compute_estimate"]["standard_runs"] == 24
    assert spec["compute_estimate"]["total_perlmutter_node_hours"] == 32.5
    assert [row["expected_native_log_Q_sign"] for row in spec["interventions"]] == [
        1,
        -1,
    ]
    assert all("VMEC" in row["construction"] for row in spec["interventions"])


def test_candidate_ranking_uses_paper_residual_when_available() -> None:
    module = _load_script()
    candidates = ["candidate"]
    effects = [
        {
            "candidate": "candidate",
            "outcome": "target_native",
            "regime": "all",
            "mean_high_minus_low": 1.0,
            "ci95_lower": 0.5,
            "ci95_upper": 1.5,
            "max_abs_nuisance_smd_after": 0.2,
        }
    ]
    sensitivity = [
        {
            "candidate": "candidate",
            "outcome": "target_native",
            "regime": "all",
            "aipw_high_minus_low": 0.8,
            "ci95_lower": 0.4,
            "ci95_upper": 1.2,
            "overlap_fraction": 0.9,
        }
    ]
    residuals = [
        {
            "candidate": "candidate",
            "gradient_set": "fixed",
            "regime": "all",
            "baseline": "f_Q_baseline",
            "mse_improvement": 0.2,
            "mse_improvement_ci95_lower": 0.1,
            "mse_improvement_ci95_upper": 0.3,
        },
        {
            "candidate": "candidate",
            "gradient_set": "fixed",
            "regime": "all",
            "baseline": "paper_selected",
            "mse_improvement": 0.01,
            "mse_improvement_ci95_lower": -0.02,
            "mse_improvement_ci95_upper": 0.03,
        },
    ]
    ranking = module._candidate_ranking(candidates, effects, sensitivity, residuals)[0]
    assert ranking["ranking_residual_baseline"] == "paper_selected"
    assert not ranking["ranking_residual_gain_resolved"]
    assert ranking["claim_grade"] == "observational-physical"
