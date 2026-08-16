from __future__ import annotations

import numpy as np
import pytest

from itg_nn.xai.audit import (
    channel_correlation_statistics,
    flux_regimes,
    grouped_bootstrap,
    median_normalized_power_spectrum,
    regression_metrics,
    robust_channel_statistics,
    select_panel_rows,
    spearman_correlation,
    top_k_members,
)


def test_native_regression_metrics_and_rank_correlation() -> None:
    actual = np.array([-2.0, -1.0, 0.0, 2.0])
    perfect = regression_metrics(actual, actual)
    assert perfect["r2"] == pytest.approx(1.0)
    assert perfect["mse"] == pytest.approx(0.0)
    assert perfect["bias"] == pytest.approx(0.0)
    assert spearman_correlation(np.array([1, 2, 3]), np.array([3, 2, 1])) == pytest.approx(-1)


def test_flux_regimes_keep_floor_and_threshold_separate() -> None:
    actual = np.array([-2.0, -1.95, -1.8, -1.0, 0.0, 1.0])
    labels, definition = flux_regimes(actual)
    assert labels[0] == "stable_floor"
    assert labels[1] == "near_threshold"
    assert set(labels[2:]) == {"low_flux", "medium_flux", "high_flux"}
    assert len(definition["unstable_flux_tertile_cuts"]) == 2


def test_grouped_bootstrap_is_deterministic_and_preserves_member_axis() -> None:
    actual = np.linspace(-2, 2, 12)
    predictions = np.vstack((actual + 0.1, actual + np.linspace(-0.2, 0.2, 12)))
    groups = np.repeat(np.arange(4), 3)
    first = grouped_bootstrap(actual, predictions, groups, replicates=20, seed=8)
    second = grouped_bootstrap(actual, predictions, groups, replicates=20, seed=8)
    np.testing.assert_array_equal(first.r2, second.r2)
    assert first.r2.shape == (20, 2)
    assert first.ranks.shape == (20, 2)
    assert first.group_count == 4


def test_top_k_members_has_exact_cardinality_under_ties() -> None:
    values = np.asarray([0.9, 0.8, 0.8, 0.8, 0.7])
    np.testing.assert_array_equal(top_k_members(values, 3), np.asarray([0, 1, 2]))


def test_panel_selection_uses_unique_equilibrium_files_and_covers_diagnostics() -> None:
    count = 60
    rows = np.arange(count)
    equilibrium = np.asarray([f"eq-{index // 2}" for index in range(count)])
    classes = np.arange(count) % 5
    flux = np.asarray((["stable_floor", "near_threshold", "low_flux"] * 20))
    bins = np.arange(count) % 3
    error = np.arange(count, dtype=float)
    disagreement = error[::-1].copy()
    selected, metadata = select_panel_rows(
        rows,
        equilibrium,
        classes,
        flux,
        bins,
        bins[::-1],
        error,
        disagreement,
        panel_size=20,
        seed=3,
    )
    selected_groups = equilibrium[selected]
    assert len(selected) == 20
    assert len(np.unique(selected_groups)) == 20
    assert metadata["sampling_unit"] == "equilibrium_files"
    assert metadata["diagnostic_quota_per_stage"] == 2


def test_robust_channel_scales_do_not_use_standard_deviation() -> None:
    geometry = np.zeros((2, 96, 7))
    geometry[:, :, 4] = np.arange(96)
    statistics = robust_channel_statistics(geometry)
    assert statistics[4]["iqr"] > 0
    assert statistics[4]["robust_sigma_iqr"] == pytest.approx(
        statistics[4]["iqr"] / 1.349
    )


def test_channel_correlations_separate_within_and_between_tube_effects() -> None:
    geometry = np.zeros((4, 8, 2), dtype=float)
    local = np.linspace(-1, 1, 8)
    for sample in range(4):
        geometry[sample, :, 0] = sample + local
        geometry[sample, :, 1] = -sample + local
    correlations = channel_correlation_statistics(geometry)
    assert correlations["within_tube_channel_correlation"][0, 1] == pytest.approx(1.0)
    assert correlations["between_tube_mean_channel_correlation"][0, 1] == pytest.approx(-1.0)


def test_median_normalized_spectrum_drops_dc_and_resists_sample_scale() -> None:
    position = np.arange(96)
    wave = np.sin(2 * np.pi * 3 * position / 96)
    geometry = np.zeros((3, 96, 1), dtype=float)
    geometry[0, :, 0] = wave + 1000.0
    geometry[1, :, 0] = 2.0 * wave - 500.0
    geometry[2, :, 0] = 1e6 * wave
    spectrum = median_normalized_power_spectrum(geometry)
    assert spectrum.shape == (1, 48)
    assert int(np.argmax(spectrum[0])) + 1 == 3
    assert spectrum[0, 2] == pytest.approx(1.0)
