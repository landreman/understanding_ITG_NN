"""Reproduce the prediction-versus-actual validation figure from our JPP paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_reference_test_data
from .ensemble import load_ensemble
from .infer import DEFAULT_CHECKPOINT
from .plotting import save_prediction_comparison


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "pdf"
    / "pred_vs_actual_plot_pre2.pdf"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the pre2 ensemble validation figure from HDF5."
    )
    parser.add_argument(
        "hdf5_path",
        type=Path,
        nargs="?",
        default=Path(
            "/Users/mattland/20260523-01-files_for_Kosmos_interpreting_neural_networks/"
            "20250102-01_GX_stellarator_dataset.h5"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_reference_test_data(args.hdf5_path)
    ensemble = load_ensemble(args.checkpoint, device=args.device)
    prediction = ensemble.predict(
        data.geometry,
        data.a_over_lt,
        data.a_over_ln,
        batch_size=args.batch_size,
    )
    assert data.actual_log_heat_flux is not None
    score = save_prediction_comparison(
        data.actual_log_heat_flux.numpy(), prediction, args.output
    )

    if args.predictions is not None:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions,
            row_index=data.row_indices,
            actual_log_Q=data.actual_log_heat_flux.numpy(),
            predicted_log_Q=prediction.mean_log_heat_flux,
            ensemble_std_log_Q=prediction.std_log_heat_flux,
            member_ids=np.asarray(ensemble.member_ids),
        )

    print(f"device: {ensemble.device}")
    print(f"models: {prediction.member_count}")
    print(f"test samples: {len(data.row_indices)}")
    print(f"R2 in clipped-log space: {score:.12f}")
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
