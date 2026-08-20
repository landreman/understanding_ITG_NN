"""The checkpoint's own answer to which fixed-gradient input convention it saw.

These run on the committed review slice and the committed checkpoint, with no
external dataset, so the automated review recomputes the claim that decided the
correction rather than reading it in a report. The evidence and provenance are
in `reports/xai/S03_fixed_gradient_decision.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

from itg_nn.data import load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.config import DEFAULT_CHECKPOINT
from itg_nn.xai.review_slice import REVIEW_SLICE_PATH


# Enough rows for R^2 to mean something, few enough to stay a fast unit test.
ROWS = np.arange(128, dtype=np.int64)

# The clipped-log floor the members collapse onto when driven off-manifold.
LOG_HEAT_FLUX_FLOOR = -2.0


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = float(np.sum((prediction - target) ** 2))
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - residual / total


@pytest.fixture(scope="module")
def ensemble():
    return load_ensemble(DEFAULT_CHECKPOINT, device="cpu")


@pytest.fixture(scope="module")
def fixed_predictions(ensemble):
    """Ensemble mean and target for both conventions on the same fixed rows."""

    result = {}
    for name, legacy in (("training", False), ("legacy_marker", True)):
        data = load_hdf5_rows(
            REVIEW_SLICE_PATH,
            ROWS,
            gradient_set="fixed",
            include_targets=True,
            legacy_fixed_marker=legacy,
        )
        assert data.actual_log_heat_flux is not None
        prediction = ensemble.predict(
            data.geometry, data.a_over_lt, data.a_over_ln
        )
        result[name] = (
            np.asarray(prediction.mean_log_heat_flux, dtype=np.float64),
            data.actual_log_heat_flux.numpy().astype(np.float64),
            float(data.a_over_lt[0]),
        )
    return result


def test_training_convention_predicts_fixed_rows_accurately(fixed_predictions):
    """At +3 the members reproduce fixed-row targets as well as varied ones."""

    prediction, target, a_over_lt = fixed_predictions["training"]
    assert a_over_lt == pytest.approx(3.0)
    assert target.std() > 0.5, "the slice's fixed targets must carry real spread"
    assert _r2(prediction, target) > 0.9
    assert prediction.std() > 0.5


def test_legacy_marker_saturates_the_members_at_the_floor(fixed_predictions):
    """At -3 the members leave the training manifold and stop varying at all.

    A checkpoint trained with the negative marker on half its rows would fit
    those rows at -3. This is what says it never saw that input.
    """

    prediction, target, a_over_lt = fixed_predictions["legacy_marker"]
    assert a_over_lt == pytest.approx(-3.0)
    assert prediction.max() - prediction.min() < 0.5
    assert prediction.mean() < LOG_HEAT_FLUX_FLOOR + 0.2
    assert _r2(prediction, target) < -1.0


def test_the_two_conventions_are_not_a_small_perturbation(fixed_predictions):
    """Guard against a future edit that makes the flag a no-op."""

    training = fixed_predictions["training"][0]
    legacy = fixed_predictions["legacy_marker"][0]
    assert np.mean(np.abs(training - legacy)) > 1.0
