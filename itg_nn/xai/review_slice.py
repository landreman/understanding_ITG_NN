"""Access to the committed review slice of real dataset rows.

`tests/data/review_slice.h5` holds 2,000 real rows drawn from the external HDF5
dataset so that the automated pull-request review, which runs on a GitHub Actions
runner without that dataset, can recompute numbers instead of only reading code.
`scripts/build_review_slice.py` documents how the rows were chosen.

Slice row IDs are **not** parent row IDs. The registered cohorts in
`reports/xai/S01_artifacts/cohorts.json` are written in parent row IDs, so
passing one straight to a reader pointed at the slice would silently return the
wrong flux tube. Everything here exists to make that mistake raise instead.

The slice is a verification artifact. Implementers must not develop against it,
tune on it, or select results with it; see `AGENTS.md`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from itg_nn.data import GradientSet


REVIEW_SLICE_PATH = Path("tests/data/review_slice.h5")
SLICE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ReviewSliceIndex:
    """The mapping and provenance stored alongside the slice's rows."""

    path: Path
    source_row_ids: np.ndarray
    is_panel_row: np.ndarray
    is_reference_cohort_row: np.ndarray
    provenance: dict[str, Any]

    def __len__(self) -> int:
        return len(self.source_row_ids)

    @property
    def selection(self) -> dict[str, Any]:
        """How the rows were chosen, as recorded by the generator."""

        return json.loads(self.provenance["selection"])

    def slice_rows(self, source_rows: Sequence[int] | np.ndarray) -> np.ndarray:
        """Translate parent-dataset row IDs into slice row IDs.

        Raises if any requested row is absent, because a review that silently
        analysed a different flux tube than the report did would be worse than
        one that could not run at all.
        """

        requested = np.asarray(source_rows, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("source_rows must be one-dimensional")
        positions = np.searchsorted(self.source_row_ids, requested)
        clipped = np.clip(positions, 0, len(self.source_row_ids) - 1)
        missing = self.source_row_ids[clipped] != requested
        if missing.any():
            absent = np.unique(requested[missing])
            preview = ", ".join(str(int(row)) for row in absent[:10])
            suffix = ", ..." if len(absent) > 10 else ""
            raise KeyError(
                f"{len(absent)} parent row ID(s) are not in the review slice "
                f"({preview}{suffix}). The slice holds the S01 panel and sibling "
                "tubes only; a claim about other rows cannot be checked here."
            )
        return positions.astype(np.int64)

    def contains(self, source_rows: Sequence[int] | np.ndarray) -> np.ndarray:
        """Elementwise membership test, for deciding what a review can check."""

        requested = np.asarray(source_rows, dtype=np.int64)
        positions = np.clip(
            np.searchsorted(self.source_row_ids, requested),
            0,
            len(self.source_row_ids) - 1,
        )
        return self.source_row_ids[positions] == requested

    def panel_slice_rows(self) -> np.ndarray:
        """Slice row IDs of the S01 frozen interpretation panel."""

        return np.flatnonzero(self.is_panel_row).astype(np.int64)

    def reference_cohort_slice_rows(self) -> np.ndarray:
        """Slice row IDs that belong to the S01 varied reference cohort."""

        return np.flatnonzero(self.is_reference_cohort_row).astype(np.int64)

    def reference_prediction(
        self, gradient_set: GradientSet = "varied"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble mean and std computed from the parent file, per slice row.

        These were produced by the generator against the full dataset, so
        comparing a fresh prediction to them checks the slice itself.
        """

        if gradient_set not in ("fixed", "varied"):
            raise ValueError("gradient_set must be 'fixed' or 'varied'")
        with h5py.File(self.path, "r") as handle:
            group = handle["review_slice"]
            mean = group[f"reference_{gradient_set}_mean_log_heat_flux"][:]
            std = group[f"reference_{gradient_set}_std_log_heat_flux"][:]
        return mean, std


def load_review_slice_index(
    path: str | Path = REVIEW_SLICE_PATH,
) -> ReviewSliceIndex:
    """Read the slice's row mapping, membership flags, and provenance."""

    resolved = Path(path)
    with h5py.File(resolved, "r") as handle:
        group = handle["review_slice"]
        version = int(group.attrs["format_version"])
        if version != SLICE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported review slice format version {version}; "
                f"this code expects {SLICE_FORMAT_VERSION}"
            )
        source_row_ids = group["source_row_ids"][:].astype(np.int64)
        index = ReviewSliceIndex(
            path=resolved,
            source_row_ids=source_row_ids,
            is_panel_row=group["is_panel_row"][:].astype(bool),
            is_reference_cohort_row=group["is_reference_cohort_row"][:].astype(bool),
            provenance={key: group.attrs[key] for key in group.attrs},
        )
    if np.any(np.diff(source_row_ids) <= 0):
        raise ValueError("review slice source row IDs must be sorted and unique")
    return index
