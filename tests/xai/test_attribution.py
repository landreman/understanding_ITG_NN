from __future__ import annotations

import numpy as np
import pytest
import torch

from itg_nn.xai.attribution import (
    AttributionMap,
    absolute_rank_correlation,
    attribution_equivariance_error,
    attribution_sensitivity,
    completeness_residual,
    curve_area,
    cyclic_grouped_occlusion,
    deletion_insertion_curves,
    expected_gradients,
    grouped_bootstrap_mean,
    integrated_gradients,
    native_scaled_gradients,
    periodic_extremal_mask,
    perturbation_infidelity,
    temporal_saliency_rescale,
    toy_recovery,
    vargrad,
)
from itg_nn.xai.perturbations import ValidityTag
from scripts.xai_s06a_attribution import (
    METHOD_NAMES,
    _curve_margin_components,
    _forward,
    _grouped_curve_margin_interval,
    _resolve,
    _select_methods,
    build_parser,
)


def _linear_periodic_forward(values: torch.Tensor) -> torch.Tensor:
    positions = torch.as_tensor((94, 95, 0, 1), device=values.device)
    return values[:, positions, 2].sum(dim=1) / 4


def _invariant_quadratic_forward(values: torch.Tensor) -> torch.Tensor:
    return (values[:, :, 1].square() + 0.5 * values[:, :, 4]).mean(dim=1)


def test_integrated_gradients_recovers_wrapped_toy_and_null_control() -> None:
    geometry = torch.zeros((3, 96, 7), dtype=torch.float32)
    geometry[:, (94, 95, 0, 1), 2] = torch.as_tensor((1.0, 2.0, 3.0, 4.0))
    geometry[:, :, 6] = 20.0  # large irrelevant channel must remain null
    baseline = torch.zeros_like(geometry)

    result = integrated_gradients(
        _linear_periodic_forward, geometry, baseline, steps=16, backend="fallback"
    )

    expected = torch.zeros_like(geometry)
    expected[:, (94, 95, 0, 1), 2] = geometry[:, (94, 95, 0, 1), 2] / 4
    torch.testing.assert_close(result.values, expected, atol=1e-7, rtol=0)
    torch.testing.assert_close(
        result.values.sum(dim=(1, 2)), _linear_periodic_forward(geometry), atol=1e-7, rtol=0
    )
    assert torch.count_nonzero(result.values[:, :, 6]) == 0
    assert result.signed
    assert result.validity == ValidityTag.OFF_MANIFOLD
    assert result.method == "integrated_gradients_fallback"


def test_fallback_integrated_gradients_matches_nonlinear_analytic_path() -> None:
    geometry = torch.randn((2, 96, 7), generator=torch.Generator().manual_seed(21))
    result = integrated_gradients(
        _invariant_quadratic_forward,
        geometry,
        torch.zeros_like(geometry),
        steps=9,
        backend="fallback",
    )
    expected = torch.zeros_like(geometry)
    expected[:, :, 1] = geometry[:, :, 1].square() / 96
    expected[:, :, 4] = 0.5 * geometry[:, :, 4] / 96
    torch.testing.assert_close(result.values, expected, atol=2e-7, rtol=0)


def test_native_scaled_gradients_explain_clipped_log_units_not_exp_output() -> None:
    geometry = torch.zeros((2, 96, 7), dtype=torch.float32)
    scales = torch.as_tensor((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))

    def native_output(values: torch.Tensor) -> torch.Tensor:
        # A native max(log Q, -2)-unit value. Exponentiating it would multiply
        # this derivative by exp(output), which this exact check rejects.
        return 2.0 + values[:, :, 4].mean(dim=1)

    result = native_scaled_gradients(native_output, geometry, scales)
    expected = torch.zeros_like(geometry)
    expected[:, :, 4] = scales[4] / 96
    torch.testing.assert_close(result.values, expected, atol=1e-8, rtol=0)
    assert result.metadata["estimand"] == "native max(log Q, -2)"
    assert result.metadata["scale"] == "robust_per_channel"


def test_expected_gradients_and_vargrad_are_seed_deterministic() -> None:
    generator = torch.Generator().manual_seed(8)
    geometry = torch.randn((4, 96, 7), generator=generator)
    baselines = torch.stack((torch.zeros_like(geometry[0]), geometry.mean(dim=0)))
    first = expected_gradients(
        _linear_periodic_forward,
        geometry,
        baselines,
        samples=24,
        seed=19,
        backend="fallback",
    )
    second = expected_gradients(
        _linear_periodic_forward,
        geometry,
        baselines,
        samples=24,
        seed=19,
        backend="fallback",
    )
    torch.testing.assert_close(first.values, second.values, atol=0, rtol=0)

    def nonlinear(values: torch.Tensor) -> torch.Tensor:
        return values[:, :, 3].square().mean(dim=1)

    var_first = vargrad(
        nonlinear,
        geometry,
        robust_scales=torch.ones(7),
        samples=20,
        noise_fraction=0.1,
        seed=22,
    )
    var_second = vargrad(
        nonlinear,
        geometry,
        robust_scales=torch.ones(7),
        samples=20,
        noise_fraction=0.1,
        seed=22,
    )
    torch.testing.assert_close(var_first.values, var_second.values, atol=0, rtol=0)
    assert float(var_first.values[:, :, 3].mean()) > 0
    assert torch.count_nonzero(var_first.values[:, :, 0]) == 0
    assert not var_first.signed


def test_completeness_and_cyclic_equivariance_are_measured_in_map_coordinates() -> None:
    generator = torch.Generator().manual_seed(12)
    geometry = torch.randn((3, 96, 7), generator=generator)
    baseline = torch.zeros_like(geometry)
    result = integrated_gradients(
        _invariant_quadratic_forward, geometry, baseline, steps=65, backend="fallback"
    )
    residual = completeness_residual(
        _invariant_quadratic_forward, geometry, baseline, result.values
    )
    assert float(residual.abs().max()) < 2e-5

    def attributor(values: torch.Tensor) -> AttributionMap:
        return integrated_gradients(
            _invariant_quadratic_forward,
            values,
            torch.zeros_like(values),
            steps=33,
            backend="fallback",
        )

    error = attribution_equivariance_error(attributor, geometry, shift=17)
    assert error < 2e-6


def test_equivariance_distinguishes_fixed_and_co_shifted_baselines() -> None:
    geometry = torch.randn((2, 96, 7), generator=torch.Generator().manual_seed(31))
    baseline = torch.randn((2, 96, 7), generator=torch.Generator().manual_seed(32))
    shift = 17

    def with_baseline(reference: torch.Tensor):
        return lambda values: integrated_gradients(
            _invariant_quadratic_forward,
            values,
            reference,
            steps=17,
            backend="fallback",
        )

    fixed = attribution_equivariance_error(
        with_baseline(baseline), geometry, shift=shift
    )
    co_shifted = attribution_equivariance_error(
        with_baseline(baseline),
        geometry,
        shift=shift,
        shifted_attributor=with_baseline(torch.roll(baseline, shift, dims=1)),
    )
    assert co_shifted < 2e-6
    assert fixed > 0.5


def test_cyclic_occlusion_recovers_wrapped_window_and_ignores_null_channel() -> None:
    geometry = torch.zeros((2, 96, 7), dtype=torch.float32)
    geometry[:, (94, 95, 0, 1), 2] = 2.0
    geometry[:, :, 5] = 50.0
    result = cyclic_grouped_occlusion(
        _linear_periodic_forward,
        geometry,
        torch.zeros_like(geometry),
        window=4,
        stride=2,
    )
    recovery = toy_recovery(
        result.values,
        relevant_channels=(2,),
        relevant_positions=(94, 95, 0, 1),
    )
    assert recovery["channel_top1"] == 1.0
    assert recovery["position_average_precision"] > 0.9
    assert torch.count_nonzero(result.values[:, :, 5]) == 0
    assert result.validity == ValidityTag.PLAUSIBLY_LOCAL


def test_toy_recovery_counts_negative_contributions_as_relevant() -> None:
    attribution = torch.zeros((2, 96, 7))
    attribution[:, 95, 2] = -4.0
    attribution[:, 0, 2] = 3.0
    attribution[:, 20, 6] = 1.0
    recovery = toy_recovery(
        attribution,
        relevant_channels=(2,),
        relevant_positions=(95, 0),
    )
    assert recovery == {"channel_top1": 1.0, "position_average_precision": 1.0}


def test_temporal_rescaling_retains_signs_and_separates_marginals() -> None:
    values = torch.zeros((2, 8, 3))
    values[:, 6:, 1] = torch.as_tensor(((1.0, -3.0), (2.0, -4.0)))
    source = AttributionMap(
        values=values,
        method="source",
        validity=ValidityTag.OFF_MANIFOLD,
        signed=True,
        runtime_seconds=0.0,
        metadata={},
    )
    result = temporal_saliency_rescale(source)
    assert result.values.shape == values.shape
    assert torch.equal(torch.sign(result.values), torch.sign(values))
    torch.testing.assert_close(
        result.values.abs().sum(dim=(1, 2)), values.abs().sum(dim=(1, 2))
    )
    assert torch.count_nonzero(result.values[:, :, 0]) == 0
    assert result.metadata["channel_marginal"] == "mean_absolute_over_z"
    assert result.metadata["position_marginal"] == "mean_absolute_over_channel"


def test_periodic_mask_selects_natural_signal_without_boundary_penalty() -> None:
    geometry = torch.zeros((1, 96, 7), dtype=torch.float32)
    geometry[:, (94, 95, 0, 1), 2] = 3.0
    result = periodic_extremal_mask(
        _linear_periodic_forward,
        geometry,
        torch.zeros_like(geometry),
        area_fraction=4 / (96 * 7),
        steps=80,
        learning_rate=0.2,
        seed=3,
    )
    relevant = result.values[0, (94, 95, 0, 1), 2]
    irrelevant = result.values[0].clone()
    irrelevant[(94, 95, 0, 1), 2] = 0
    assert float(relevant.mean()) > float(irrelevant.max())
    assert result.metadata["periodic_total_variation"] is True
    assert result.validity == ValidityTag.OFF_MANIFOLD


def test_deletion_insertion_reports_random_controls_and_support_drift_every_dose() -> None:
    geometry = torch.zeros((5, 12, 3), dtype=torch.float32)
    geometry[:, :3, 1] = 2.0
    geometry[:, 3:, 2] = 0.25
    baseline = torch.zeros_like(geometry)

    def forward(values: torch.Tensor) -> torch.Tensor:
        return values[:, :3, 1].sum(dim=1) + 0.01 * values[:, :, 2].sum(dim=1)

    attribution = geometry.clone()
    curves = deletion_insertion_curves(
        forward,
        geometry,
        baseline,
        attribution,
        fractions=(0.0, 0.25, 0.5, 1.0),
        robust_scales=torch.ones(3),
        seed=10,
        support_scorer=lambda values: {
            "warning_score": np.sqrt(np.mean(np.square(values), axis=(1, 2)))
        },
    )
    assert len(curves) == 4
    assert [row["fraction"] for row in curves] == [0.0, 0.25, 0.5, 1.0]
    assert all("deletion_support_drift_rms" in row for row in curves)
    assert all("insertion_support_drift_rms" in row for row in curves)
    assert all("random_deletion_output" in row for row in curves)
    assert all(np.isfinite(float(row["deletion_support_warning"])) for row in curves)
    assert curves[1]["deletion_output"] < curves[1]["random_deletion_output"]
    assert curves[-1]["deletion_output"] == pytest.approx(
        curves[-1]["baseline_output"], abs=1e-7
    )
    assert curves[-1]["insertion_output"] == pytest.approx(
        curves[-1]["original_output"], abs=1e-7
    )


def test_deletion_insertion_ranks_mixed_sign_maps_by_absolute_magnitude() -> None:
    geometry = torch.as_tensor((1.0, -10.0, 0.0, 0.0)).reshape(1, 4, 1)
    rows = deletion_insertion_curves(
        lambda values: values.sum(dim=(1, 2)),
        geometry,
        torch.zeros_like(geometry),
        geometry.clone(),
        fractions=(0.0, 0.25, 1.0),
        robust_scales=torch.ones(1),
        seed=3,
    )
    assert float(rows[1]["deletion_output"]) == 1.0
    assert float(rows[1]["insertion_output"]) == -10.0


def test_curve_area_orients_outputs_when_original_is_below_baseline() -> None:
    rows = [
        {
            "fraction": 0.0,
            "original_output": -2.0,
            "baseline_output": 1.0,
            "deletion_output": -2.0,
            "random_deletion_output": -2.0,
        },
        {
            "fraction": 0.5,
            "original_output": -2.0,
            "baseline_output": 1.0,
            "deletion_output": 1.0,
            "random_deletion_output": -0.5,
        },
        {
            "fraction": 1.0,
            "original_output": -2.0,
            "baseline_output": 1.0,
            "deletion_output": 1.0,
            "random_deletion_output": 1.0,
        },
    ]
    # The targeted curve reaches the baseline earlier, so its normalized area
    # is lower regardless of the signed direction from original to baseline.
    assert curve_area(rows, "deletion_output") < curve_area(
        rows, "random_deletion_output"
    )


def test_grouped_bootstrap_resamples_equilibrium_files_and_is_deterministic() -> None:
    values = np.asarray((1.0, 1.0, 3.0, 3.0, 8.0, 8.0))
    groups = np.asarray(("eq0", "eq0", "eq1", "eq1", "eq2", "eq2"))
    first = grouped_bootstrap_mean(values, groups, replicates=200, seed=42)
    second = grouped_bootstrap_mean(values, groups, replicates=200, seed=42)
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.resampling_unit == "equilibrium_files"
    assert first.estimate == pytest.approx(values.mean())
    # Every resample is a mean of three whole, equally sized equilibrium groups.
    allowed = {
        (left + middle + right) / 3
        for left in (1.0, 3.0, 8.0)
        for middle in (1.0, 3.0, 8.0)
        for right in (1.0, 3.0, 8.0)
    }
    assert set(np.round(first.samples, 12)).issubset(set(np.round(tuple(allowed), 12)))


def test_curve_margin_interval_resamples_sibling_tubes_as_equilibrium_groups() -> None:
    fractions = (0.0, 0.5, 1.0)
    original = np.asarray((1.0, 1.0, 1.0, 2.0, 3.0, 3.0))
    baseline = np.zeros(6)
    midpoint_gap = np.asarray((0.0, 0.0, 0.0, 3.0, 8.0, 8.0))
    curves = []
    for fraction in fractions:
        if fraction == 0.0:
            method = original
            random = original
        elif fraction == 0.5:
            method = np.zeros(6)
            random = midpoint_gap
        else:
            method = baseline
            random = baseline
        curves.append(
            {
                "fraction": fraction,
                "original_output_per_sample": original,
                "baseline_output_per_sample": baseline,
                "deletion_output_per_sample": method,
                "random_deletion_output_per_sample": random,
                "insertion_output_per_sample": method,
                "random_insertion_output_per_sample": random,
            }
        )

    groups = np.asarray(("eq0", "eq0", "eq0", "eq1", "eq2", "eq2"))
    rows = np.arange(6)
    actual = _grouped_curve_margin_interval(
        curves,
        groups,
        rows,
        direction="deletion",
        replicates=200,
        seed=42,
    )

    group_rows = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    grouped_rng = np.random.default_rng(42)
    grouped_gaps = []
    for _ in range(200):
        chosen = grouped_rng.integers(0, len(group_rows), size=len(group_rows))
        sampled = np.concatenate([group_rows[index] for index in chosen])
        grouped_gaps.append(
            _curve_margin_components(curves, sampled, "deletion")["native_gap"]
        )
    grouped_interval = np.quantile(grouped_gaps, (0.025, 0.975))

    row_rng = np.random.default_rng(42)
    row_gaps = [
        _curve_margin_components(
            curves, row_rng.integers(0, len(rows), size=len(rows)), "deletion"
        )["native_gap"]
        for _ in range(200)
    ]
    row_interval = np.quantile(row_gaps, (0.025, 0.975))

    np.testing.assert_allclose(
        (actual["native_gap_ci_lower"], actual["native_gap_ci_upper"]),
        grouped_interval,
    )
    assert actual["oriented_native_gap_estimate"] > 0
    assert not np.allclose(grouped_interval, row_interval)
    assert actual["resampling_unit"] == "equilibrium_files"


def test_s06a_selection_requires_positive_faithfulness_in_each_floor_stratum() -> None:
    rows = []
    toy = {}
    for method in METHOD_NAMES:
        toy[method] = {
            "channel_top1": 1.0,
            "position_average_precision": 1.0,
        }
        for stratum in ("all", "stable_or_near_floor", "unstable"):
            rows.append(
                {
                    "function": "invariant_tilde_f",
                    "method": method,
                    "stratum": stratum,
                    "deletion_margin_vs_random": 1.0,
                    "insertion_margin_vs_random": 1.0,
                    "parameter_randomization_correlation": 0.0,
                    "normalized_infidelity": 1.0,
                    "runtime_seconds": 1.0,
                }
            )
    for row in rows:
        if row["method"] == "ig_low_pass" and row["stratum"] == "stable_or_near_floor":
            row["deletion_margin_vs_random"] = -0.1

    rule = {
        "path_candidates": ["ig_low_pass", "ig_medoid"],
        "perturbation_candidates": ["periodic_mask"],
        "faithfulness_strata": ["stable_or_near_floor", "unstable"],
        "minimum_toy_channel_top1": 1.0,
        "minimum_toy_position_average_precision": 0.75,
        "minimum_deletion_margin": 0.0,
        "minimum_insertion_margin": 0.0,
        "maximum_parameter_randomization_correlation": 0.95,
        "tie_break": "lowest_normalized_infidelity_then_runtime",
    }
    selected = _select_methods(rows, toy, rule)
    assert selected["eligible"]["ig_low_pass"] is False
    assert selected["primary_path_gradient"] == "ig_medoid"


def test_infidelity_sensitivity_and_randomized_map_agreement_have_controls() -> None:
    generator = torch.Generator().manual_seed(44)
    geometry = torch.randn((3, 12, 3), generator=generator)
    baseline = torch.zeros_like(geometry)

    def forward(values: torch.Tensor) -> torch.Tensor:
        return (values[:, :, 0] + 2 * values[:, :, 1]).sum(dim=1)

    exact = integrated_gradients(forward, geometry, baseline, steps=8, backend="fallback")
    infidelity = perturbation_infidelity(
        forward,
        geometry,
        baseline,
        exact.values,
        trials=20,
        removal_fraction=0.25,
        seed=4,
    )
    assert infidelity < 1e-12

    def attributor(values: torch.Tensor) -> AttributionMap:
        return integrated_gradients(
            forward, values, torch.zeros_like(values), steps=8, backend="fallback"
        )

    sensitivity = attribution_sensitivity(
        attributor,
        geometry,
        robust_scales=torch.ones(3),
        trials=6,
        noise_fraction=0.01,
        seed=5,
    )
    assert 0 < sensitivity < 0.1
    assert absolute_rank_correlation(exact.values, exact.values) == pytest.approx(1.0)
    randomized = exact.values.roll(shifts=1, dims=2)
    assert absolute_rank_correlation(exact.values, randomized) < 0.4


def test_captum_and_fallback_integrated_gradients_match_on_linear_toy() -> None:
    pytest.importorskip("captum")
    geometry = torch.randn((2, 96, 7), generator=torch.Generator().manual_seed(2))
    baseline = torch.zeros_like(geometry)
    fallback = integrated_gradients(
        _linear_periodic_forward, geometry, baseline, steps=16, backend="fallback"
    )
    captum = integrated_gradients(
        _linear_periodic_forward, geometry, baseline, steps=16, backend="captum"
    )
    torch.testing.assert_close(captum.values, fallback.values, atol=2e-7, rtol=0)
    assert captum.method == "integrated_gradients_captum"


def test_captum_expected_gradients_seeds_numpy_and_is_shift_equivariant() -> None:
    pytest.importorskip("captum")
    geometry = torch.randn((3, 96, 7), generator=torch.Generator().manual_seed(7))
    baselines = torch.randn((4, 96, 7), generator=torch.Generator().manual_seed(8))
    first = expected_gradients(
        _linear_periodic_forward,
        geometry,
        baselines,
        samples=20,
        seed=9,
        backend="captum",
    )
    second = expected_gradients(
        _linear_periodic_forward,
        geometry,
        baselines,
        samples=20,
        seed=9,
        backend="captum",
    )
    torch.testing.assert_close(first.values, second.values, atol=0, rtol=0)

    shift = 13
    shifted = expected_gradients(
        lambda values: _linear_periodic_forward(torch.roll(values, -shift, dims=1)),
        torch.roll(geometry, shift, dims=1),
        torch.roll(baselines, shift, dims=1),
        samples=20,
        seed=9,
        backend="captum",
    )
    torch.testing.assert_close(
        shifted.values, torch.roll(first.values, shift, dims=1), atol=2e-7, rtol=0
    )


def test_captum_expected_gradients_preserves_panel_drive_alignment() -> None:
    pytest.importorskip("captum")

    class DriveMember:
        def original(
            self,
            geometry: torch.Tensor,
            a_over_lt: torch.Tensor,
            a_over_ln: torch.Tensor,
        ) -> torch.Tensor:
            del a_over_ln
            return geometry[:, 0, 0] * a_over_lt

    geometry = torch.zeros((2, 96, 7))
    geometry[:, 0, 0] = torch.as_tensor((2.0, 3.0))
    baselines = torch.zeros((3, 96, 7))
    forward = _forward(
        DriveMember(),  # type: ignore[arg-type]
        "original_f",
        torch.as_tensor((5.0, 11.0)),
        torch.zeros(2),
    )

    captum = expected_gradients(
        forward, geometry, baselines, samples=4, seed=5, backend="captum"
    )
    fallback = expected_gradients(
        forward, geometry, baselines, samples=4, seed=5, backend="fallback"
    )
    torch.testing.assert_close(captum.values, fallback.values, atol=1e-7, rtol=0)


def test_s06a_cli_exposes_required_reproducibility_overrides() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert {
        "config",
        "members",
        "rows",
        "device",
        "seed",
        "resume",
        "output_dir",
    }.issubset(destinations)


def test_s06a_pilot_replays_historical_pooled_gate() -> None:
    parser = build_parser()
    args = parser.parse_args(["--pilot"])
    config = {
        "pilot": {"run_id": "pilot"},
        "selection_rule": {"faithfulness_strata": ["stable_or_near_floor", "unstable"]},
    }
    resolved = _resolve(config, args)
    assert resolved["selection_rule"]["faithfulness_strata"] == ["all"]
    assert config["selection_rule"]["faithfulness_strata"] == [
        "stable_or_near_floor",
        "unstable",
    ]


def test_s06a_forward_repeats_drives_for_captum_step_major_batches() -> None:
    class DummyMember:
        def original(
            self,
            geometry: torch.Tensor,
            a_over_lt: torch.Tensor,
            a_over_ln: torch.Tensor,
        ) -> torch.Tensor:
            del geometry
            return a_over_lt + a_over_ln

    forward = _forward(
        DummyMember(),  # type: ignore[arg-type]
        "original_f",
        torch.as_tensor((3.0, 4.0)),
        torch.as_tensor((1.0, 2.0)),
    )
    output = forward(torch.zeros((6, 96, 7)))
    torch.testing.assert_close(output, torch.as_tensor((4.0, 6.0, 4.0, 6.0, 4.0, 6.0)))
    with pytest.raises(ValueError, match="multiple of panel rows"):
        forward(torch.zeros((5, 96, 7)))
