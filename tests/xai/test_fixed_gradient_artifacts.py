"""Schema and self-consistency of the committed fixed-gradient correction artifacts.

The automated review pointed out that the columns these tables are read for —
the `AGENTS.md`-required validity tag, and the floor fraction whose one-sided
definition was misread once already — were pinned by nothing. A later edit could
drop the tag or rename the column back and the suite would stay green. That is
how the original misreading survived, so it is pinned here.

Everything runs off committed files, so the automated review checks it too.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
DECISION_ARTIFACTS = REPOSITORY / "reports" / "xai" / "S03_fixed_gradient_artifacts"
S02_ARTIFACTS = REPOSITORY / "reports" / "xai" / "S02_artifacts"

# The registered perturbation-validity vocabulary, as used by
# reports/xai/S03_artifacts/support.csv.
VALIDITY_BY_CONVENTION = {
    "training": "observed_comparison",
    "legacy_marker": "off_manifold",
}
A_OVER_LT_BY_CONVENTION = {"training": 3.0, "legacy_marker": -3.0}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def convention_rows():
    return _rows(DECISION_ARTIFACTS / "fixed_gradient_convention.csv")


@pytest.fixture(scope="module")
def subgroup_rows():
    return _rows(DECISION_ARTIFACTS / "s02_subgroup_exactness.csv")


def test_convention_artifact_carries_the_registered_validity_tag(convention_rows):
    """`AGENTS.md` requires the tag in the artifact, not only in the report."""

    assert convention_rows, "the convention artifact must not be empty"
    for row in convention_rows:
        assert row["validity_tag"] == VALIDITY_BY_CONVENTION[row["convention"]]
    tags = {row["validity_tag"] for row in convention_rows}
    assert tags == {"observed_comparison", "off_manifold"}


def test_convention_artifact_records_the_input_it_measured(convention_rows):
    for row in convention_rows:
        assert float(row["a_over_LT_input"]) == pytest.approx(
            A_OVER_LT_BY_CONVENTION[row["convention"]]
        )


def test_both_floor_fractions_are_published_and_differ(convention_rows):
    """The one-sided column reads as two-sided; publishing only it misled once."""

    legacy = [row for row in convention_rows if row["convention"] == "legacy_marker"]
    assert legacy
    one_sided = np.array(
        [float(row["fraction_at_or_below_floor_plus_0p05"]) for row in legacy]
    )
    two_sided = np.array(
        [float(row["fraction_within_0p05_of_floor_two_sided"]) for row in legacy]
    )
    assert np.all(two_sided <= one_sided)
    assert np.any(two_sided < one_sided), (
        "predictions below the floor make the two definitions differ; if they "
        "ever coincide the second column has stopped being a distinct check"
    )


def test_the_hundred_percent_floor_claim_is_an_ensemble_statement(convention_rows):
    """Scope guard. The ensemble mean is pinned; individual members are not.

    The reports quote "100% at or below the floor plus 0.05" for the legacy
    marker. That holds for the ensemble mean and not for every member — the
    least-pinned member has 0% of its rows there while still being essentially
    flat. Pin the distinction so the claim cannot silently widen.
    """

    legacy = [row for row in convention_rows if row["convention"] == "legacy_marker"]
    ensemble = [row for row in legacy if row["entity_type"] == "ensemble"]
    assert len(ensemble) == 1
    assert float(ensemble[0]["fraction_at_or_below_floor_plus_0p05"]) == 1.0

    members = [row for row in legacy if row["entity_type"] == "member"]
    fractions = np.array(
        [float(row["fraction_at_or_below_floor_plus_0p05"]) for row in members]
    )
    assert fractions.min() < 1.0, (
        "if every member were pinned too, the reports could make the stronger "
        "claim; they deliberately do not"
    )


def test_the_legacy_marker_flattens_every_member_relative_to_the_training_input(
    convention_rows,
):
    """The scale-free form of the collapse, which does hold member by member.

    Distance to the floor is not the right statistic for a member sitting in a
    narrow band slightly off it. Spread is: at -3 no member varies as much as
    the least-varying member does at +3.
    """

    spread = {
        convention: np.array(
            [
                float(row["prediction_std"])
                for row in convention_rows
                if row["convention"] == convention and row["entity_type"] == "member"
            ]
        )
        for convention in VALIDITY_BY_CONVENTION
    }
    assert spread["legacy_marker"].max() < spread["training"].min()
    assert np.median(spread["training"]) / np.median(spread["legacy_marker"]) > 10


def test_convention_artifact_covers_every_member_under_both_conventions(convention_rows):
    members = {
        convention: sorted(
            row["entity"]
            for row in convention_rows
            if row["convention"] == convention and row["entity_type"] == "member"
        )
        for convention in VALIDITY_BY_CONVENTION
    }
    assert len(members["training"]) == 100
    assert members["training"] == members["legacy_marker"]
    ensembles = [row for row in convention_rows if row["entity_type"] == "ensemble"]
    assert len(ensembles) == 2


def test_subgroup_artifact_backs_the_headline_exactness_maximum(subgroup_rows):
    """The published maximum must be reproducible from a committed table.

    It is a float32 roundoff maximum and does not reproduce across machines, so
    the rows it is taken over are what make the number checkable at all.
    """

    assert subgroup_rows
    assert {int(row["shift"]) for row in subgroup_rows} == {32, 64}
    assert {row["gradient_set"] for row in subgroup_rows} == {"varied", "fixed"}

    member_max = max(
        float(row["max_absolute_change"])
        for row in subgroup_rows
        if row["entity_type"] == "member"
    )
    summary = json.loads((S02_ARTIFACTS / "summary.json").read_text())
    assert member_max == pytest.approx(
        summary["checks"]["exact_subgroup_max_abs"], rel=0, abs=0
    )
    assert member_max < 2e-5, "the registered atol/rtol the check is graded against"


@pytest.fixture(scope="module")
def convention_module():
    path = REPOSITORY / "scripts" / "xai_fixed_gradient_convention.py"
    spec = importlib.util.spec_from_file_location("xai_fixed_gradient_convention", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["xai_fixed_gradient_convention"] = module
    spec.loader.exec_module(module)
    return module


def test_row_builder_tags_and_measures_synthetic_input(convention_module):
    """Pin the generator too, so a regression is caught before republishing."""

    target = np.array([0.0, 1.0, 2.0, 3.0])
    predictions = {
        "training": np.array([[0.1, 1.1, 2.1, 3.1]]),
        # Saturated at the floor, as the off-manifold input actually behaves.
        "legacy_marker": np.array([[-2.02, -2.01, -2.03, -2.06]]),
    }
    rows = convention_module._convention_rows(
        predictions, target, ("m1",), floor=-2.0
    )
    by_key = {(row["entity"], row["convention"]): row for row in rows}

    training = by_key[("m1", "training")]
    assert training["validity_tag"] == "observed_comparison"
    assert training["r2_against_fixed_target"] > 0.9
    assert training["fraction_at_or_below_floor_plus_0p05"] == 0.0

    legacy = by_key[("m1", "legacy_marker")]
    assert legacy["validity_tag"] == "off_manifold"
    assert legacy["r2_against_fixed_target"] < 0
    # All four are at or below -1.95; only three are within 0.05 either side.
    assert legacy["fraction_at_or_below_floor_plus_0p05"] == 1.0
    assert legacy["fraction_within_0p05_of_floor_two_sided"] == pytest.approx(0.75)
