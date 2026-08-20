"""Tests for the committed review slice of real dataset rows.

These run without the external HDF5 dataset, which is the point: they are what
tells the automated review, on a runner that cannot see the parent file, that the
slice it is about to compute with is the slice the generator wrote.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from itg_nn.data import load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.config import DEFAULT_CHECKPOINT
from itg_nn.xai.review_slice import (
    FIXED_GRADIENT_CONVENTION,
    REVIEW_SLICE_PATH,
    load_review_slice_index,
)


COHORTS = "reports/xai/S01_artifacts/cohorts.json"

# The reference predictions were computed on macOS/arm64 from the parent file.
# CI runs x86-64 Linux, where the same weights and inputs can differ in the last
# bits, so this is a tolerance rather than a bit-for-bit comparison.
PREDICTION_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def index():
    return load_review_slice_index(REVIEW_SLICE_PATH)


@pytest.fixture(scope="module")
def cohorts():
    with open(COHORTS) as handle:
        return json.load(handle)


def test_slice_is_present_and_indexed(index):
    assert len(index) == 2000
    assert index.source_row_ids.dtype == np.int64
    assert np.all(np.diff(index.source_row_ids) > 0)
    assert index.provenance["source_sha256"]
    assert index.provenance["checkpoint_sha256"]


def test_slice_contains_the_whole_registered_panel(index, cohorts):
    panel = np.asarray(cohorts["interpretation_panel"]["varied_row_ids"], dtype=np.int64)
    fixed = np.asarray(cohorts["interpretation_panel"]["fixed_row_ids"], dtype=np.int64)

    # The panel is what reports quote, so a partial panel would let a wrong
    # headline number survive review by being unverifiable.
    assert index.contains(panel).all()
    assert index.contains(fixed).all()
    assert len(index.panel_slice_rows()) == len(panel)

    mapped = index.slice_rows(panel)
    assert np.array_equal(index.source_row_ids[mapped], panel)


def test_parent_row_ids_absent_from_the_slice_raise(index):
    present = int(index.source_row_ids[0])
    absent = int(index.provenance["source_row_count"]) - 1
    assert not index.contains([absent])[0]

    with pytest.raises(KeyError, match="not in the review slice"):
        index.slice_rows([present, absent])


def test_equilibrium_grouping_is_not_degenerate(index):
    """A grouped bootstrap must be distinguishable from a tube-level one here.

    The S01 panel takes at most one tube per equilibrium. On the panel alone,
    grouping by `equilibrium_files` and grouping by flux tube coincide, so the
    slice deliberately carries sibling tubes; without them the review cannot
    tell a correctly grouped resample from an incorrectly grouped one.
    """

    import h5py

    with h5py.File(REVIEW_SLICE_PATH, "r") as handle:
        equilibria = handle["equilibrium_files"][:]

    _, counts = np.unique(equilibria, return_counts=True)
    assert counts.max() > 1
    assert (counts > 1).sum() >= 500, "too few equilibria carry a sibling tube"

    cohort_rows = index.reference_cohort_slice_rows()
    _, cohort_counts = np.unique(equilibria[cohort_rows], return_counts=True)
    assert (cohort_counts > 1).sum() >= 100, (
        "the reference-cohort subset must also carry repeated equilibria, or a "
        "cohort-restricted grouped bootstrap cannot be checked"
    )


def test_slice_rows_reproduce_parent_file_predictions(index):
    """The slice's geometry still drives the ensemble to the parent's answer.

    The expected values were computed by the generator against the full dataset.
    Any silent corruption, dtype change, or row-order mistake in the slice breaks
    this, which is the guarantee the review needs before it computes anything.
    """

    ensemble = load_ensemble(DEFAULT_CHECKPOINT, device="cpu")
    rows = np.arange(64, dtype=np.int64)

    for gradient_set in ("varied", "fixed"):
        data = load_hdf5_rows(
            REVIEW_SLICE_PATH, rows, gradient_set=gradient_set, include_targets=True
        )
        prediction = ensemble.predict(data.geometry, data.a_over_lt, data.a_over_ln)
        expected_mean, expected_std = index.reference_prediction(gradient_set)

        assert np.allclose(
            np.asarray(prediction.mean_log_heat_flux),
            expected_mean[rows],
            atol=PREDICTION_TOLERANCE,
        )
        assert np.allclose(
            np.asarray(prediction.std_log_heat_flux),
            expected_std[rows],
            atol=PREDICTION_TOLERANCE,
        )
        assert data.actual_log_heat_flux is not None
        assert float(data.actual_log_heat_flux.min()) >= -2.0


def test_slice_preserves_the_clipped_floor_population(index):
    """A third of varied rows sit at the floor; the slice must still show that.

    AGENTS.md requires near-floor rows to be reported separately. A slice that
    accidentally dropped them would let a report that pools them pass review.
    """

    import h5py

    with h5py.File(REVIEW_SLICE_PATH, "r") as handle:
        q = handle["varied_gradient_simulations/Q_avgs"][:]

    at_floor = np.mean(np.maximum(np.log(q), -2.0) <= -2.0 + 1e-9)
    assert 0.05 < at_floor < 0.6, f"floor fraction {at_floor:.3f} is not representative"


def test_slice_records_the_fixed_gradient_convention(index):
    """The stored fixed-row baseline says which a/L_T produced it."""

    assert index.provenance["fixed_gradient_convention"] == FIXED_GRADIENT_CONVENTION


def test_a_slice_without_the_convention_attribute_is_refused(tmp_path):
    """A pre-correction slice must raise, not silently supply floor baselines.

    Its fixed-row reference predictions were made at the off-manifold -3 and
    would have a review confirm numbers against saturation.
    """

    import shutil

    import h5py

    copied = tmp_path / "stale_slice.h5"
    shutil.copy(REVIEW_SLICE_PATH, copied)
    with h5py.File(copied, "r+") as handle:
        del handle["review_slice"].attrs["fixed_gradient_convention"]

    with pytest.raises(ValueError, match="fixed-gradient convention"):
        load_review_slice_index(copied)
