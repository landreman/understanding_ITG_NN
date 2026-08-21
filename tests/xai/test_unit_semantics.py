from __future__ import annotations

import numpy as np
import pytest

from itg_nn.xai.unit_semantics import (
    NATURAL_EXEMPLAR_VALIDITY,
    cluster_natural_exemplars,
    extract_wrapped_patches,
    first_layer_transfer,
    grouped_bootstrap_weights,
    native_output_comparison,
    physics_concept_traces,
    select_natural_exemplars,
    shift_consistency_error,
    row_permutation_selection_null,
    unit_concept_alignment,
)


def _valid_geometry(samples: int = 4) -> np.ndarray:
    geometry = np.zeros((samples, 96, 7), dtype=np.float64)
    z = np.arange(96, dtype=np.float64)
    geometry[:, :, 0] = 2.0
    geometry[:, :, 1] = np.cos(2 * np.pi * z / 96)
    geometry[:, :, 2] = np.where((z % 8) < 4, 1.0, -1.0)
    geometry[:, :, 3] = np.sin(4 * np.pi * z / 96)
    geometry[:, :, 4] = 1.0 + 0.2 * np.cos(6 * np.pi * z / 96)
    geometry[:, :, 5] = np.sin(2 * np.pi * z / 96)
    geometry[:, :, 6] = 4.0
    return geometry


def test_paper_concept_traces_have_registered_pointwise_formulas() -> None:
    geometry = _valid_geometry(2)
    traces = physics_concept_traces(
        geometry,
        channel_scales=np.ones(7),
        window_widths=(1, 5),
    )
    by_name = {name: traces.values[:, index] for index, name in enumerate(traces.names)}

    np.testing.assert_array_equal(
        by_name["bad_curvature"], (geometry[:, :, 2] > 0).astype(np.float64)
    )
    np.testing.assert_allclose(by_name["compression_abs_grad_x"], 2.0)
    np.testing.assert_allclose(
        by_name["f_Q_integrand_p3_Bm1"],
        ((geometry[:, :, 2] > 0) + 0.2) * 4.0,
    )
    expected_shear = 96 * (
        np.roll(geometry[:, :, 5] / geometry[:, :, 6], -1, axis=1)
        - np.roll(geometry[:, :, 5] / geometry[:, :, 6], 1, axis=1)
    ) / 2
    np.testing.assert_allclose(by_name["local_shear_dz_gds21_over_gds22"], expected_shear)
    assert "bad_curvature__mean_w5" in by_name
    assert traces.validity_tag == "observed-comparison"


def test_parallel_scale_requires_robust_channel_scales() -> None:
    geometry = _valid_geometry(1)
    with pytest.raises(ValueError, match="seven positive robust channel scales"):
        physics_concept_traces(geometry, channel_scales=np.ones(6))
    first = physics_concept_traces(geometry, channel_scales=np.ones(7))
    second = physics_concept_traces(
        geometry, channel_scales=np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=float)
    )
    first_index = first.names.index("parallel_scale_local_dimensionless")
    second_index = second.names.index("parallel_scale_local_dimensionless")
    assert not np.allclose(first.values[:, first_index], second.values[:, second_index])


def test_circular_windowed_fourier_scale_recovers_known_parallel_mode() -> None:
    geometry = _valid_geometry(1)
    z = np.arange(96, dtype=np.float64)
    geometry[:, :, 0] = 2.0 + 0.1 * np.cos(2 * np.pi * 3 * z / 96)
    traces = physics_concept_traces(
        geometry, channel_scales=np.ones(7), window_widths=(1, 96)
    )
    index = traces.names.index("parallel_fourier_expected_k_bmag_w96")
    np.testing.assert_allclose(traces.values[:, index], 3.0, atol=1e-10)


def test_lagged_alignment_recovers_analytic_cyclic_concept_and_null() -> None:
    rng = np.random.default_rng(23)
    samples = 18
    signal = rng.normal(size=(samples, 96))
    null = rng.normal(size=(samples, 96))
    traces = np.stack((signal, null), axis=1)
    density = np.roll(signal, 7, axis=1)
    controls = rng.normal(size=(samples, 7, 96))

    result = unit_concept_alignment(
        density,
        traces,
        concept_names=("known_signal", "null_control"),
        channel_magnitude_controls=controls,
        sparsity=0.1,
    )
    signal_row = result.rows[0]
    null_row = result.rows[1]
    assert signal_row["concept"] == "known_signal"
    assert signal_row["best_lag"] == 7
    assert signal_row["lag_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert signal_row["overlap_at_fixed_sparsity_tie_inclusive"] == pytest.approx(1.0)
    assert abs(float(null_row["lag_correlation"])) < 0.15
    assert (
        abs(float(signal_row["partial_rank_correlation_density_local_controls"]))
        > 0.9
    )
    direct_zero_lag = np.mean(
        [np.corrcoef(density[row], signal[row])[0, 1] for row in range(samples)]
    )
    assert signal_row["lag_correlation_zero_lag"] == pytest.approx(direct_zero_lag)


def test_partial_rank_control_handles_noncontiguous_channel_view() -> None:
    rng = np.random.default_rng(231)
    samples = 24
    confound = rng.normal(size=(samples, 96))
    density = confound + 0.25 * rng.normal(size=(samples, 96))
    concept = confound + 0.25 * rng.normal(size=(samples, 96))
    raw_controls = rng.normal(size=(samples, 96, 7))
    raw_controls[:, :, 0] = confound
    controls = np.moveaxis(raw_controls, 2, 1)
    assert not controls.flags.c_contiguous
    controlled = unit_concept_alignment(
        density,
        concept[:, None, :],
        concept_names=("confounded",),
        channel_magnitude_controls=controls,
        sparsity=0.05,
    ).rows[0]
    zeroed = unit_concept_alignment(
        density,
        concept[:, None, :],
        concept_names=("confounded",),
        channel_magnitude_controls=np.zeros_like(controls),
        sparsity=0.05,
    ).rows[0]
    assert float(controlled["lag_correlation"]) > 0.8
    assert abs(float(controlled["partial_rank_correlation_density_local_controls"])) < 0.15
    assert float(zeroed["partial_rank_correlation_density_local_controls"]) > 0.8


def test_overlap_includes_ties_and_reports_its_chance_baseline() -> None:
    density = np.zeros((3, 96), dtype=np.float64)
    density[:, 0] = 2.0
    density[:, 1:11] = 1.0
    controls = np.zeros((3, 7, 96), dtype=np.float64)
    row = unit_concept_alignment(
        density,
        density[:, None, :],
        concept_names=("tied_signal",),
        channel_magnitude_controls=controls,
        sparsity=0.05,
    ).rows[0]
    assert row["density_mask_mean_count"] == pytest.approx(11.0)
    assert row["concept_mask_mean_count"] == pytest.approx(11.0)
    assert row["overlap_at_fixed_sparsity_tie_inclusive"] == pytest.approx(1.0)
    assert row["overlap_chance_baseline"] == pytest.approx(11 / 96)
    assert row["overlap_enrichment"] == pytest.approx(96 / 11)


def test_flat_density_rows_are_counted_and_zero_weighted_by_convention() -> None:
    rng = np.random.default_rng(92)
    signal = rng.normal(size=(4, 96))
    density = signal.copy()
    density[:2] = 0.0
    row = unit_concept_alignment(
        density,
        signal[:, None, :],
        concept_names=("signal",),
        channel_magnitude_controls=np.zeros((4, 7, 96)),
        sparsity=0.05,
    ).rows[0]
    assert row["n_rows_with_defined_correlation"] == 2
    assert row["lag_correlation"] == pytest.approx(0.5)
    assert row["lag_correlation_defined_rows"] == pytest.approx(1.0)


def test_row_permutation_selection_null_is_deterministic_and_breaks_signal() -> None:
    rng = np.random.default_rng(77)
    signal = rng.normal(size=(32, 96))
    density = np.roll(signal, 5, axis=1)
    traces = np.stack((signal, rng.normal(size=(32, 96))), axis=1)
    first = row_permutation_selection_null(
        density, traces, permutations=7, seed=13
    )
    second = row_permutation_selection_null(
        density, traces, permutations=7, seed=13
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (7,)
    assert np.max(first) < 0.2


def test_alignment_and_bootstrap_are_shift_invariant_and_grouped() -> None:
    rng = np.random.default_rng(41)
    geometry = rng.normal(size=(12, 96, 7))
    geometry[:, :, 0] = np.abs(geometry[:, :, 0]) + 0.5
    geometry[:, :, 6] = np.abs(geometry[:, :, 6]) + 0.2
    traces = physics_concept_traces(geometry, channel_scales=np.ones(7))
    density = traces.values[:, traces.names.index("compression_abs_grad_x")]
    shifted_traces = physics_concept_traces(
        np.roll(geometry, 19, axis=1), channel_scales=np.ones(7)
    )
    assert shift_consistency_error(
        traces.values, shifted_traces.values, shift=19, position_axis=2
    ) < 1e-12

    groups = np.repeat(np.asarray(["eq0", "eq1", "eq2", "eq3", "eq4", "eq5"]), 2)
    first, count = grouped_bootstrap_weights(groups, replicates=30, seed=8)
    second, second_count = grouped_bootstrap_weights(groups, replicates=30, seed=8)
    np.testing.assert_array_equal(first, second)
    assert count == second_count == 6
    for start in range(0, len(groups), 2):
        np.testing.assert_array_equal(first[:, start], first[:, start + 1])

    aligned = unit_concept_alignment(
        density,
        traces.values,
        concept_names=traces.names,
        channel_magnitude_controls=np.moveaxis(np.abs(geometry), 2, 1),
        sparsity=0.1,
        groups=groups,
        bootstrap_replicates=30,
        seed=8,
    )
    repeated = unit_concept_alignment(
        density,
        traces.values,
        concept_names=traces.names,
        channel_magnitude_controls=np.moveaxis(np.abs(geometry), 2, 1),
        sparsity=0.1,
        groups=groups,
        bootstrap_replicates=30,
        seed=8,
    )
    assert aligned.recurrence == repeated.recurrence
    assert aligned.bootstrap_group == "equilibrium_files"


def test_wrapped_natural_exemplars_preserve_recorded_circular_coordinates() -> None:
    geometry = np.arange(3 * 96 * 7, dtype=np.float64).reshape(3, 96, 7)
    samples = np.asarray([0, 2])
    centers = np.asarray([1, 95])
    offsets = np.asarray([-3, -1, 0, 2])
    patches = extract_wrapped_patches(geometry, samples, centers, offsets)
    expected_positions = (centers[:, None] + offsets[None, :]) % 96
    for exemplar in range(2):
        np.testing.assert_array_equal(
            patches.values[exemplar],
            geometry[samples[exemplar], expected_positions[exemplar]].T,
        )
    np.testing.assert_array_equal(patches.source_positions, expected_positions)
    assert patches.alignment_operation == "joint_circular_roll_to_activation_center"
    assert patches.validity_tag == NATURAL_EXEMPLAR_VALIDITY == "observed-comparison"

    shifted = np.roll(geometry, 11, axis=1)
    shifted_patches = extract_wrapped_patches(
        shifted, samples, (centers + 11) % 96, offsets
    )
    np.testing.assert_array_equal(patches.values, shifted_patches.values)


def test_natural_exemplars_are_maximal_and_recur_across_equilibria() -> None:
    density = np.zeros((6, 96), dtype=np.float64)
    density[0, 3] = 10
    density[1, 4] = 9  # Same equilibrium as row 0: must not displace recurrence.
    density[2, 5] = 8
    density[3, 6] = 7
    density[4, 7] = 6
    density[5, 8] = 5
    groups = np.asarray(["eq0", "eq0", "eq1", "eq2", "eq3", "eq4"])
    selected = select_natural_exemplars(density, groups, count=4)
    np.testing.assert_array_equal(selected.sample_indices, [0, 2, 3, 4])
    np.testing.assert_array_equal(selected.centers, [3, 5, 6, 7])
    np.testing.assert_allclose(selected.activations, [10, 8, 7, 6])
    assert len(np.unique(groups[selected.sample_indices])) == 4
    assert selected.selection_unit == "equilibrium_files"


def test_natural_exemplar_clusters_are_deterministic_and_keep_dispersion() -> None:
    low = np.zeros((5, 7, 9), dtype=np.float64)
    high = np.ones((5, 7, 9), dtype=np.float64) * 5
    patches = np.concatenate((low, high), axis=0)
    first = cluster_natural_exemplars(patches, clusters=2, seed=17)
    second = cluster_natural_exemplars(patches, clusters=2, seed=17)
    np.testing.assert_array_equal(first.assignment, second.assignment)
    np.testing.assert_allclose(first.centers, second.centers)
    assert sorted(np.bincount(first.assignment).tolist()) == [5, 5]
    assert first.dispersion.shape == first.centers.shape == (2, 7, 9)
    assert np.max(first.dispersion) == pytest.approx(0.0)


def test_first_layer_fourier_transfer_keeps_filter_channel_axes() -> None:
    weights = np.zeros((2, 7, 5), dtype=np.float64)
    weights[0, 3, 2] = 2.0
    weights[1, 4] = np.asarray([1, -1, 0, 1, -1])
    result = first_layer_transfer(weights, grid_size=96)
    assert result.amplitude.shape == (2, 7, 49)
    np.testing.assert_allclose(result.amplitude[0, 3], 2.0)
    np.testing.assert_array_equal(result.kernels, weights)
    assert result.frequency_index.tolist() == list(range(49))


def test_native_output_comparison_retains_member_signed_clipped_log_units() -> None:
    original = np.asarray([-2.0, -1.7, 0.4, 1.2])
    invariant = np.asarray([-1.9, -1.8, 0.1, 1.4])
    rows = native_output_comparison(
        original,
        invariant,
        stable_or_near_floor=np.asarray([True, True, False, False]),
    )
    overall = next(row for row in rows if row["stratum"] == "overall")
    assert overall["signed_delta_mean"] == pytest.approx(-0.025)
    assert overall["signed_delta_min"] == pytest.approx(-0.3)
    assert overall["signed_delta_max"] == pytest.approx(0.2)
    assert overall["estimand"] == "native max(log Q, -2)"
    assert overall["delta_sign"] == "invariant_tilde_f minus original_f"
