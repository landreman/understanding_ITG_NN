from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from itg_nn.xai.concepts import MatchedExtremes


@pytest.fixture(scope="module")
def s08_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "xai_s08_concepts.py"
    spec = importlib.util.spec_from_file_location("xai_s08_concepts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["xai_s08_concepts"] = module
    spec.loader.exec_module(module)
    return module


def test_counterexample_subsets_keep_matched_pairs_together(s08_script) -> None:
    match = MatchedExtremes(
        high=np.arange(10), low=np.arange(10) + 100, validity_tag="observed-comparison"
    )
    subsets = s08_script._paired_counterexample_subsets(
        match, sets=5, fraction=0.6, seed=17
    )
    assert len(subsets) == 5
    for high, low in subsets:
        assert len(high) == 6
        np.testing.assert_array_equal(low - high, np.full(6, 100))


def test_aggregate_cav_uses_all_normalized_counterexample_directions(s08_script) -> None:
    directions = [torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0])]
    aggregate = s08_script._aggregate_cav_direction(directions)
    torch.testing.assert_close(aggregate, torch.tensor([1.0, 1.0]) / np.sqrt(2))


def test_published_use_fields_keep_stable_and_unstable_results_distinct(
    s08_script,
) -> None:
    use = SimpleNamespace(
        mean_directional_derivative=7.0,
        intervention_rms=9.0,
        intervention_rms_stable_or_near_floor=2.0,
        intervention_rms_unstable=6.0,
        random_intervention_rms_median=3.0,
        random_intervention_rms_median_stable_or_near_floor=0.5,
        random_intervention_rms_median_unstable=2.0,
        scale_matched_random_rms_median=1.5,
        scale_matched_random_rms_median_stable_or_near_floor=0.25,
        scale_matched_random_rms_median_unstable=3.0,
        orthogonal_complement_ablation_rms=11.0,
    )
    fields = s08_script._published_use_fields(
        use,
        derivative_values=np.array([1.0, 10.0, 3.0, 14.0]),
        stable_mask=np.array([True, False, True, False]),
    )

    assert fields["mean_directional_derivative_stable_or_near_floor"] == 2.0
    assert fields["mean_directional_derivative_unstable"] == 12.0
    assert fields["intervention_rms_stable_or_near_floor"] == 2.0
    assert fields["intervention_rms_unstable"] == 6.0
    assert fields["intervention_to_random_ratio_stable_or_near_floor"] == 4.0
    assert fields["intervention_to_random_ratio_unstable"] == 3.0
    assert fields["scale_matched_random_rms_median_stable_or_near_floor"] == 0.25
    assert fields["scale_matched_random_rms_median_unstable"] == 3.0
    assert (
        fields["intervention_to_scale_matched_random_ratio_stable_or_near_floor"]
        == 8.0
    )
    assert fields["intervention_to_scale_matched_random_ratio_unstable"] == 2.0


def test_bootstrap_and_complete_gate_pin_production_statistics(s08_script) -> None:
    values = np.array([2.0, 2.0, -1.0, -1.0, 3.0, 3.0])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    assert s08_script._bootstrap_inference(values, groups, 200, 9) == pytest.approx(
        (-1.0, 3.0, 0.25870646766169153)
    )

    base = {
        "encoded_r2": 0.8,
        "permuted_r2": 0.0,
        "counterexample_sign_agreement": 1.0,
        "directional_derivative_ci95_lower": 0.1,
        "directional_derivative_ci95_upper": 0.4,
        "bootstrap_p_value": 0.01,
        "intervention_to_random_ratio": 2.0,
        "intervention_to_scale_matched_random_ratio": 0.8,
        "counterexample_max_abs_smd": 0.1,
        "counterexample_subset_max_abs_smd": 0.1,
    }
    other = {**base, "bootstrap_p_value": 0.8}
    gated = s08_script._finalize_claim_rows([base, other])
    assert gated[0]["bootstrap_fdr_q_value"] == pytest.approx(0.02)
    assert gated[0]["tcav_ci_excludes_zero"] is True
    assert gated[0]["tcav_interval_and_fdr_pass"] is True
    # The stronger scale-matched control, not the isotropic ratio, gates use.
    assert gated[0]["direction_intervention_beats_random"] is False
    assert gated[0]["use_claim_permitted"] is False

    fdr_failure = s08_script._finalize_claim_rows(
        [{**base, "bootstrap_p_value": 0.8,
          "intervention_to_scale_matched_random_ratio": 1.2}]
    )[0]
    assert fdr_failure["tcav_ci_excludes_zero"] is True
    assert fdr_failure["tcav_fdr_q_le_0_05"] is False
    assert fdr_failure["use_claim_permitted"] is False

    balanced_failure = s08_script._finalize_claim_rows(
        [{**base, "intervention_to_scale_matched_random_ratio": 1.2,
          "counterexample_subset_max_abs_smd": 0.3}]
    )[0]
    assert balanced_failure["counterexample_balance_pass"] is False
    assert balanced_failure["use_claim_permitted"] is False
