from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from itg_nn.data import (
    clipped_log_heat_flux,
    load_hdf5_rows,
    reference_split_assignments,
    reference_test_rows,
)
from itg_nn.ensemble import EnsemblePrediction
from itg_nn.model import Architecture, CyclicInvariantNet
from itg_nn.plotting import r2_score


def test_clipped_log_heat_flux() -> None:
    result = clipped_log_heat_flux(np.array([np.exp(-3), 1.0, np.e]))
    np.testing.assert_allclose(result, [-2.0, 0.0, 1.0])


def test_clipped_log_rejects_nonpositive_flux() -> None:
    with pytest.raises(ValueError, match="positive"):
        clipped_log_heat_flux(np.array([0.0]))


@pytest.fixture
def fixed_gradient_file(tmp_path):
    hdf5_path = tmp_path / "sample.h5"
    with h5py.File(hdf5_path, "w") as h5_file:
        h5_file.create_dataset("raw_feature_tensor", data=np.zeros((2, 96, 7)))
        for name in ("fixed", "varied"):
            group = h5_file.create_group(f"{name}_gradient_simulations")
            group.create_dataset("a_over_LT", data=np.array([3.0, 3.0]))
            group.create_dataset("a_over_Ln", data=np.array([0.9, 0.9]))
            group.create_dataset("Q_avgs", data=np.array([1.0, np.e]))
    return hdf5_path


def test_fixed_gradient_rows_use_the_training_convention(fixed_gradient_file) -> None:
    """Fixed rows reach the network at the physical +3, as the checkpoint saw them.

    The legacy loader negated this. `reports/xai/S03_fixed_gradient_decision.md`
    records why that was wrong and how it was established.
    """

    data = load_hdf5_rows(
        fixed_gradient_file, [1], gradient_set="fixed", include_targets=True
    )
    assert data.a_over_lt.item() == pytest.approx(3.0)
    assert data.a_over_ln.item() == pytest.approx(0.9)
    assert data.actual_log_heat_flux is not None
    assert data.actual_log_heat_flux.item() == pytest.approx(1.0)


def test_legacy_fixed_marker_is_opt_in(fixed_gradient_file) -> None:
    """The off-manifold `-3` input stays reachable, but only when asked for."""

    data = load_hdf5_rows(
        fixed_gradient_file,
        [1],
        gradient_set="fixed",
        legacy_fixed_marker=True,
    )
    assert data.a_over_lt.item() == pytest.approx(-3.0)
    assert data.a_over_ln.item() == pytest.approx(0.9)


def test_legacy_fixed_marker_does_not_touch_varied_rows(fixed_gradient_file) -> None:
    """The null control: the flag names a fixed-set convention and nothing else."""

    default = load_hdf5_rows(fixed_gradient_file, [0, 1], gradient_set="varied")
    flagged = load_hdf5_rows(
        fixed_gradient_file, [0, 1], gradient_set="varied", legacy_fixed_marker=True
    )
    assert torch.equal(default.a_over_lt, flagged.a_over_lt)
    assert torch.equal(default.a_over_ln, flagged.a_over_ln)
    assert float(default.a_over_lt.min()) > 0.0


def test_reference_split_assignments_match_test_row_reconstruction(tmp_path) -> None:
    hdf5_path = tmp_path / "split.h5"
    fixed_q = np.ones(10)
    varied_q = np.ones(10)
    varied_q[3] = 0.0
    with h5py.File(hdf5_path, "w") as h5_file:
        fixed = h5_file.create_group("fixed_gradient_simulations")
        varied = h5_file.create_group("varied_gradient_simulations")
        fixed.create_dataset("Q_avgs", data=fixed_q)
        varied.create_dataset("Q_avgs", data=varied_q)
    assignments = reference_split_assignments(hdf5_path, seed=42)
    assert assignments["fixed"].shape == (10,)
    assert assignments["varied"][3] == -1
    expected = np.flatnonzero(assignments["varied"] == 2)
    np.testing.assert_array_equal(reference_test_rows(hdf5_path, seed=42), expected)


def test_inference_model_output_shape() -> None:
    architecture = Architecture(
        kernel_sizes=(3, 3, 3, 3, 3),
        convolution_channels=(4, 5, 6, 7, 8),
        dense_dimensions=(9, 10),
    )
    model = CyclicInvariantNet(architecture).eval()
    with torch.inference_mode():
        result = model(torch.zeros(2, 96, 7), torch.ones(2), torch.ones(2))
    assert result.shape == (2, 1)


def test_r2_score() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    assert r2_score(actual, actual) == pytest.approx(1.0)


def test_prediction_inverse_transform() -> None:
    prediction = EnsemblePrediction(
        mean_log_heat_flux=np.array([0.0], dtype=np.float32),
        std_log_heat_flux=np.array([np.log(2)], dtype=np.float32),
        member_count=3,
    )
    assert prediction.mean_heat_flux[0] == pytest.approx(1.0)
    assert prediction.lower_heat_flux[0] == pytest.approx(0.5)
    assert prediction.upper_heat_flux[0] == pytest.approx(2.0)
