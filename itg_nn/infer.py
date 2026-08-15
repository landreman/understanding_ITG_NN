"""Command-line interface for HDF5 ensemble inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from .data import HDF5_GROUPS, load_hdf5_rows
from .ensemble import load_ensemble


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "models" / "cyclic_ensemble_pre2.pt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the cyclic heat-flux neural-network ensemble on HDF5 rows."
    )
    parser.add_argument("hdf5_path", type=Path)
    parser.add_argument("output_path", type=Path, help="Output .npz file")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--gradient-set", choices=tuple(HDF5_GROUPS), default="varied"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with h5py.File(args.hdf5_path, "r") as h5_file:
        row_count = len(h5_file["raw_feature_tensor"])
    stop = row_count if args.stop is None else args.stop
    if not 0 <= args.start <= stop <= row_count:
        raise ValueError(f"Require 0 <= start <= stop <= {row_count}")

    data = load_hdf5_rows(
        args.hdf5_path,
        np.arange(args.start, stop, dtype=np.int64),
        gradient_set=args.gradient_set,
    )
    ensemble = load_ensemble(args.checkpoint, device=args.device)
    prediction = ensemble.predict(
        data.geometry,
        data.a_over_lt,
        data.a_over_ln,
        batch_size=args.batch_size,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_path,
        row_index=data.row_indices,
        predicted_log_Q=prediction.mean_log_heat_flux,
        ensemble_std_log_Q=prediction.std_log_heat_flux,
        predicted_Q=prediction.mean_heat_flux,
        predicted_Q_lower=prediction.lower_heat_flux,
        predicted_Q_upper=prediction.upper_heat_flux,
        model_a_over_LT=data.a_over_lt.numpy(),
        model_a_over_Ln=data.a_over_ln.numpy(),
        member_ids=np.asarray(ensemble.member_ids),
    )
    print(
        f"Wrote {len(data.row_indices)} predictions from "
        f"{prediction.member_count} models to {args.output_path}"
    )


if __name__ == "__main__":
    main()
