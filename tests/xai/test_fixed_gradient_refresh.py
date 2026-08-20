"""The row-replacement rule used to refresh S02's fixed-gradient strata.

The refresh recomputes only the fixed half of the panel and edits the committed
CSVs in place. These tests pin the two properties that make that safe to read as
a diff: varied rows come through byte-identical, and a stratum whose size moved
is refused rather than overwritten.

The refresh script itself needs the external dataset and the registered S02 run
directory, so only its merge logic is exercised here — which is the part a
reviewer can neither rerun nor eyeball across 1,700 rows.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]

HEADER = "entity,entity_type,gradient_set,stratum,n,rms_change,max_absolute_change\n"
COMMITTED = (
    HEADER
    + "m1,member,varied,all,7,0.125,0.44\n"
    + "m1,member,fixed,all,5,0.0117,0.03\n"
    + "m1,member,fixed,unstable,4,0.0118,0.031\n"
    + "ensemble_mean,ensemble,varied,all,7,0.11,0.4\n"
    + "ensemble_mean,ensemble,fixed,all,5,0.012,0.032\n"
)


@pytest.fixture(scope="module")
def refresh():
    path = REPOSITORY / "scripts" / "xai_s02_fixed_refresh.py"
    spec = importlib.util.spec_from_file_location("xai_s02_fixed_refresh", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["xai_s02_fixed_refresh"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def committed(tmp_path):
    path = tmp_path / "phase_average_exactness.csv"
    path.write_text(COMMITTED)
    return path


def _rebuilt(overrides=None):
    rows = [
        {
            "entity": "m1", "entity_type": "member", "gradient_set": "fixed",
            "stratum": "all", "n": 5, "rms_change": 0.0688,
            "max_absolute_change": 0.21,
        },
        {
            "entity": "m1", "entity_type": "member", "gradient_set": "fixed",
            "stratum": "unstable", "n": 4, "rms_change": 0.0693,
            "max_absolute_change": 0.22,
        },
        {
            "entity": "ensemble_mean", "entity_type": "ensemble",
            "gradient_set": "fixed", "stratum": "all", "n": 5,
            "rms_change": 0.0651, "max_absolute_change": 0.2,
        },
        # A varied row is present in the rebuild and must be ignored by the merge.
        {
            "entity": "m1", "entity_type": "member", "gradient_set": "varied",
            "stratum": "all", "n": 7, "rms_change": 9.99,
            "max_absolute_change": 9.99,
        },
    ]
    for row in rows:
        key = (row["gradient_set"], row["stratum"], row["entity"])
        row.update((overrides or {}).get(key, {}))
    return rows


def test_merge_replaces_every_fixed_row(refresh, committed):
    text, replaced = refresh._merge_csv(
        committed, _rebuilt(), "phase_average_exactness.csv"
    )
    assert replaced == 3
    rows = list(csv.DictReader(text.splitlines()))
    fixed = {(r["entity"], r["stratum"]): r for r in rows if r["gradient_set"] == "fixed"}
    assert fixed[("m1", "all")]["rms_change"] == "0.0688"
    assert fixed[("m1", "unstable")]["rms_change"] == "0.0693"
    assert fixed[("ensemble_mean", "all")]["rms_change"] == "0.0651"


def test_merge_leaves_varied_rows_byte_identical(refresh, committed):
    """The rebuild's varied rows are deliberately wrong here; none may land."""

    text, _ = refresh._merge_csv(
        committed, _rebuilt(), "phase_average_exactness.csv"
    )
    original_varied = [
        line for line in COMMITTED.splitlines() if ",varied," in line
    ]
    merged_varied = [line for line in text.splitlines() if ",varied," in line]
    assert merged_varied == original_varied
    assert "9.99" not in text


def test_merge_preserves_row_order_and_header(refresh, committed):
    text, _ = refresh._merge_csv(
        committed, _rebuilt(), "phase_average_exactness.csv"
    )
    assert text.splitlines()[0] == HEADER.strip()
    identity = [
        tuple(row[key] for key in ("entity", "gradient_set", "stratum"))
        for row in csv.DictReader(text.splitlines())
    ]
    expected = [
        tuple(row[key] for key in ("entity", "gradient_set", "stratum"))
        for row in csv.DictReader(COMMITTED.splitlines())
    ]
    assert identity == expected


def test_merge_refuses_a_changed_stratum_size(refresh, committed):
    """A moved stratum means the cohort changed, which is not a refresh."""

    rebuilt = _rebuilt(overrides={("fixed", "all", "m1"): {"n": 6}})
    with pytest.raises(RuntimeError, match="stratum size changed"):
        refresh._merge_csv(committed, rebuilt, "phase_average_exactness.csv")


def test_merge_refuses_a_missing_recomputed_row(refresh, committed):
    """Silently leaving a stale fixed row behind is the failure that matters."""

    rebuilt = [row for row in _rebuilt() if row["stratum"] != "unstable"]
    with pytest.raises(RuntimeError, match="no recomputed row"):
        refresh._merge_csv(committed, rebuilt, "phase_average_exactness.csv")
