"""HDF5 readers for ensemble inference and reference-data reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import random_split


GradientSet = Literal["fixed", "varied"]
HDF5_GROUPS = {
    "fixed": "fixed_gradient_simulations",
    "varied": "varied_gradient_simulations",
}
LOG_HEAT_FLUX_FLOOR = -2.0
REFERENCE_SPLIT_SEED = 42


@dataclass(frozen=True)
class InferenceData:
    """Tensors and source-row identifiers needed for inference."""

    geometry: torch.Tensor
    a_over_lt: torch.Tensor
    a_over_ln: torch.Tensor
    row_indices: np.ndarray
    actual_log_heat_flux: torch.Tensor | None = None

    def __post_init__(self) -> None:
        sample_count = len(self.row_indices)
        lengths = (len(self.geometry), len(self.a_over_lt), len(self.a_over_ln))
        if lengths != (sample_count, sample_count, sample_count):
            raise ValueError("All inference arrays must have the same first dimension")
        if self.actual_log_heat_flux is not None:
            if len(self.actual_log_heat_flux) != sample_count:
                raise ValueError("Target and feature counts do not match")


def clipped_log_heat_flux(heat_flux: np.ndarray) -> np.ndarray:
    """Apply the transform used to train ensemble members tagged ``pre2``."""

    heat_flux_64 = np.asarray(heat_flux, dtype=np.float64)
    if np.any(heat_flux_64 <= 0):
        raise ValueError("Heat flux must be positive before taking its logarithm")
    return np.maximum(np.log(heat_flux_64), LOG_HEAT_FLUX_FLOOR).astype(
        np.float32
    )


def _take_rows(dataset: h5py.Dataset, row_indices: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows while satisfying h5py's sorted-index rule."""

    unique_rows, inverse = np.unique(row_indices, return_inverse=True)
    return dataset[unique_rows][inverse]


def _validated_rows(
    row_indices: Sequence[int] | np.ndarray, row_count: int
) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("row_indices must be one-dimensional")
    if len(rows) and (rows.min() < 0 or rows.max() >= row_count):
        raise IndexError(f"HDF5 row indices must be in [0, {row_count})")
    return rows


def load_hdf5_rows(
    hdf5_path: str | Path,
    row_indices: Sequence[int] | np.ndarray,
    *,
    gradient_set: GradientSet = "varied",
    include_targets: bool = False,
) -> InferenceData:
    """Load selected flux tubes from the metadata-rich HDF5 dataset.

    The trained target called ``Q_avgs_without_FSA_grad_x`` in the legacy
    pickle files is byte-for-byte equal to ``Q_avgs`` in the HDF5 file. The
    newer HDF5 name and metadata are authoritative here.
    """

    try:
        simulation_group = HDF5_GROUPS[gradient_set]
    except KeyError as error:
        raise ValueError(f"Unknown gradient_set: {gradient_set!r}") from error

    with h5py.File(hdf5_path, "r") as h5_file:
        geometry_dataset = h5_file["raw_feature_tensor"]
        rows = _validated_rows(row_indices, len(geometry_dataset))
        geometry = _take_rows(geometry_dataset, rows).astype(np.float32)
        group = h5_file[simulation_group]
        a_over_lt = _take_rows(group["a_over_LT"], rows).astype(np.float32)
        if gradient_set == "fixed":
            # The trained models saw a negative a/L_T as the fixed-gradient
            # group marker. Preserve that learned input convention while the
            # HDF5 file retains the physical, positive value in its metadata.
            a_over_lt = -a_over_lt
        a_over_ln = _take_rows(group["a_over_Ln"], rows).astype(np.float32)
        target = None
        if include_targets:
            heat_flux = _take_rows(group["Q_avgs"], rows)
            target = torch.from_numpy(clipped_log_heat_flux(heat_flux))

    return InferenceData(
        geometry=torch.from_numpy(geometry),
        a_over_lt=torch.from_numpy(a_over_lt),
        a_over_ln=torch.from_numpy(a_over_ln),
        row_indices=rows,
        actual_log_heat_flux=target,
    )


def reference_test_rows(
    hdf5_path: str | Path, *, seed: int = REFERENCE_SPLIT_SEED
) -> np.ndarray:
    """Reconstruct the legacy varied-gradient test rows without its 518 MB cache."""

    with h5py.File(hdf5_path, "r") as h5_file:
        fixed_heat_flux = h5_file["fixed_gradient_simulations/Q_avgs"][:]
        varied_heat_flux = h5_file["varied_gradient_simulations/Q_avgs"][:]

    tube_count = len(fixed_heat_flux)
    if len(varied_heat_flux) != tube_count:
        raise ValueError("Fixed- and varied-gradient groups have different row counts")

    # The legacy loader concatenated fixed then varied samples, dropped Q <= 0,
    # and called torch.random_split with an 80/10/remainder ratio.
    positive_combined_rows = np.flatnonzero(
        np.concatenate((fixed_heat_flux, varied_heat_flux)) > 0
    )
    sample_count = len(positive_combined_rows)
    train_count = int(0.8 * sample_count)
    validation_count = int(0.1 * sample_count)
    test_count = sample_count - train_count - validation_count
    generator = torch.Generator().manual_seed(seed)
    _, _, test_subset = random_split(
        range(sample_count),
        (train_count, validation_count, test_count),
        generator=generator,
    )

    combined_test_rows = positive_combined_rows[np.asarray(test_subset.indices)]
    varied_test_rows = combined_test_rows[combined_test_rows >= tube_count]
    return (varied_test_rows - tube_count).astype(np.int64)


def load_reference_test_data(
    hdf5_path: str | Path, *, seed: int = REFERENCE_SPLIT_SEED
) -> InferenceData:
    """Load the exact test cohort used to make the legacy reference figure."""

    rows = reference_test_rows(hdf5_path, seed=seed)
    return load_hdf5_rows(
        hdf5_path, rows, gradient_set="varied", include_targets=True
    )
