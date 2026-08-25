from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


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
        "minimum_candidate_separation_panel_iqr": 0.5,
        "maximum_constraint_drift_panel_iqr": 0.1,
        "minimum_native_log_Q_response": 0.2,
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
    assert {row["validity_tag"] for row in spec["interventions"]} == {
        "plausibly-local"
    }
    assert spec["pre_budget_feasibility_gate"][
        "required_before_researcher_budget_approval"
    ]
    assert spec["decisive_effect_threshold"][
        "minimum_absolute_native_log_Q_response"
    ] == 0.2


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
    ranking = module._candidate_ranking(
        candidates,
        effects,
        sensitivity,
        residuals,
        balance_threshold=0.5,
        overlap_threshold=0.8,
        paper_baseline_features={"paper_feature"},
    )[0]
    assert ranking["ranking_residual_baseline"] == "paper_selected"
    assert not ranking["ranking_residual_gain_resolved"]
    assert ranking["claim_grade"] == "observational-physical"


def test_candidate_ranking_uses_registered_balance_and_overlap_gates() -> None:
    module = _load_script()

    def ranking(*, balance: float, overlap: float):
        return module._candidate_ranking(
            ["candidate"],
            [
                {
                    "candidate": "candidate",
                    "outcome": "target_native",
                    "regime": "all",
                    "mean_high_minus_low": 1.0,
                    "ci95_lower": 0.5,
                    "ci95_upper": 1.5,
                    "max_abs_nuisance_smd_after": balance,
                }
            ],
            [
                {
                    "candidate": "candidate",
                    "outcome": "target_native",
                    "regime": "all",
                    "aipw_high_minus_low": 0.8,
                    "ci95_lower": 0.4,
                    "ci95_upper": 1.2,
                    "overlap_fraction": overlap,
                }
            ],
            [
                {
                    "candidate": "candidate",
                    "gradient_set": "fixed",
                    "regime": "all",
                    "baseline": "paper_selected",
                    "mse_improvement": 0.2,
                    "mse_improvement_ci95_lower": 0.1,
                    "mse_improvement_ci95_upper": 0.3,
                }
            ],
            balance_threshold=0.5,
            overlap_threshold=0.8,
            paper_baseline_features={"paper_feature"},
        )[0]

    assert ranking(balance=0.5, overlap=0.8)["claim_grade"] == "intervention-ready"
    assert not ranking(balance=0.5001, overlap=0.8)["balance_acceptable"]
    assert not ranking(balance=0.5, overlap=0.7999)["overlap_acceptable"]


def test_candidate_inside_paper_baseline_gets_no_incomparable_residual_point() -> None:
    module = _load_script()
    effects = [
        {
            "candidate": "candidate",
            "outcome": "target_native",
            "regime": "all",
            "mean_high_minus_low": 1.0,
            "ci95_lower": 0.5,
            "ci95_upper": 1.5,
            "max_abs_nuisance_smd_after": 1.0,
        }
    ]
    sensitivity = [
        {
            "candidate": "candidate",
            "outcome": "target_native",
            "regime": "all",
            "aipw_high_minus_low": 0.8,
            "ci95_lower": -0.1,
            "ci95_upper": 1.2,
            "overlap_fraction": 0.5,
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
        }
    ]
    row = module._candidate_ranking(
        ["candidate"],
        effects,
        sensitivity,
        residuals,
        balance_threshold=0.5,
        overlap_threshold=0.8,
        paper_baseline_features={"candidate"},
    )[0]
    assert row["ranking_residual_baseline"] == "not_applicable_candidate_in_baseline"
    assert not row["ranking_residual_comparable"]
    assert not row["ranking_residual_gain_resolved"]
    assert row["rank_score"] == 1
    assert row["matched_aipw_point_same_sign"]
    assert not row["resolved_effect_sign_agreement"]


def test_physical_outcomes_keep_native_clipping_and_panel_identity(tmp_path: Path) -> None:
    module = _load_script()
    dataset = tmp_path / "physical-outcomes.h5"
    fixed_q = np.asarray([0.01, np.exp(-1.0), np.exp(0.5)])
    varied_q = np.asarray([np.exp(-1.8), np.exp(-0.5), np.exp(1.0)])
    with h5py.File(dataset, "w") as handle:
        for name, q, drive in (
            ("fixed", fixed_q, 3.0),
            ("varied", varied_q, 4.5),
        ):
            group = handle.create_group(f"{name}_gradient_simulations")
            group.create_dataset("Q_avgs", data=q)
            group.create_dataset("Q_stds", data=np.ones(3))
            group.create_dataset("zonal_phi2_amplitudes", data=np.ones(3))
            group.create_dataset("Q_avgs_vs_z", data=np.ones((3, 4)))
            group.create_dataset("a_over_LT", data=np.full(3, drive))
            group.create_dataset("a_over_Ln", data=np.full(3, 0.9))

    outcomes = module._physical_outcomes(dataset, np.arange(3))
    np.testing.assert_allclose(
        outcomes["fixed"]["target_native"], np.maximum(np.log(fixed_q), -2.0)
    )
    np.testing.assert_allclose(
        outcomes["varied"]["target_native"], np.maximum(np.log(varied_q), -2.0)
    )
    assert np.all(outcomes["fixed"]["a_over_LT"] == 3.0)
    assert np.all(outcomes["varied"]["a_over_LT"] == 4.5)
