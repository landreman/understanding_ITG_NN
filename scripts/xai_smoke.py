#!/usr/bin/env python3
"""Run the registered S00 CPU smoke calculation and write a provenance manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from itg_nn.data import load_hdf5_rows
from itg_nn.ensemble import load_ensemble
from itg_nn.xai.artifacts import RunArtifacts
from itg_nn.xai.config import DEFAULT_DATASET, XAIConfig, load_config
from itg_nn.xai.members import MemberPredictor, select_member_ids
from itg_nn.xai.module_model import ModuleCyclicInvariantNet
from itg_nn.xai.runtime import iter_inference_batches, set_deterministic_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/xai/S00_smoke.json")
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser


def _override(config: XAIConfig, args: argparse.Namespace) -> XAIConfig:
    values = config.as_manifest_dict()
    if args.dataset is not None:
        values["dataset"] = str(args.dataset)
    if args.device is not None:
        values["device"] = args.device
    return XAIConfig(
        step=values["step"],
        run_id=values["run_id"],
        rows=tuple(values["rows"]),
        gradient_set=values["gradient_set"],
        seed=values["seed"],
        device=values["device"],
        batch_size=values["batch_size"],
        member_selection=config.member_selection,
        dataset=Path(values["dataset"]),
        checkpoint=Path(values["checkpoint"]),
    )


def verify_module_equivalence(
    ensemble, geometry: torch.Tensor, a_over_lt: torch.Tensor, a_over_ln: torch.Tensor
) -> dict[str, object]:
    """Prove the module-form conversion has identical CPU outputs for all members."""

    mismatches: list[str] = []
    with torch.inference_mode():
        for member_id, original in zip(ensemble.member_ids, ensemble.models):
            wrapped = ModuleCyclicInvariantNet.from_inference_model(original)
            expected = original(geometry, a_over_lt, a_over_ln)
            actual = wrapped(geometry, a_over_lt, a_over_ln)
            if not torch.equal(expected, actual):
                mismatches.append(member_id)
    if mismatches:
        raise RuntimeError(f"module-form outputs differ for members: {mismatches}")
    return {
        "comparison": "torch.equal on native clipped-log output",
        "fixture_samples": len(geometry),
        "member_count": len(ensemble.models),
        "passed": True,
    }


def run(config: XAIConfig, output_dir: Path) -> Path:
    """Execute the pilot and return its complete manifest path."""

    if not config.dataset.exists():
        raise FileNotFoundError(
            f"dataset not found at {config.dataset}; pass --dataset to override"
        )
    set_deterministic_seed(config.seed)
    data = load_hdf5_rows(
        config.dataset, config.rows, gradient_set=config.gradient_set, include_targets=True
    )
    ensemble = load_ensemble(config.checkpoint, device=config.device)
    checkpoint_members = torch.load(
        config.checkpoint, map_location="cpu", weights_only=True
    )["members"]
    member_ids = select_member_ids(checkpoint_members, config.member_selection)
    predictor = MemberPredictor.from_ensemble(ensemble, member_ids).eval()

    predictions: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in iter_inference_batches(data, config.batch_size):
            predictions.append(
                predictor(
                    batch.geometry.to(ensemble.device),
                    batch.a_over_lt.to(ensemble.device),
                    batch.a_over_ln.to(ensemble.device),
                ).cpu()
            )
    member_predictions = torch.cat(predictions, dim=1).numpy()
    equivalence = verify_module_equivalence(
        ensemble,
        data.geometry.to(ensemble.device),
        data.a_over_lt.to(ensemble.device),
        data.a_over_ln.to(ensemble.device),
    )

    artifacts = RunArtifacts(output_dir)
    artifacts.write_hdf5(
        "predictions.h5",
        {
            "member_prediction_log_Q": member_predictions,
            "row_id": data.row_indices,
            "actual_log_Q": data.actual_log_heat_flux.numpy(),
        },
        axes={
            "member_prediction_log_Q": ("member", "sample"),
            "row_id": ("sample",),
            "actual_log_Q": ("sample",),
        },
        attributes={"member_ids": list(member_ids), "target_transform": "log_clip_min_-2"},
    )
    artifacts.write_json("module_equivalence.json", equivalence)
    return artifacts.finalize(
        config=config.as_manifest_dict(),
        dataset=config.dataset,
        checkpoint=config.checkpoint,
        member_ids=member_ids,
        row_ids=data.row_indices,
        gradient_set=config.gradient_set,
        device=ensemble.device,
        repository=Path(__file__).resolve().parents[1],
    )


def main() -> None:
    args = build_parser().parse_args()
    config = _override(load_config(args.config), args)
    output_dir = args.output_dir or Path("output/xai") / config.step / config.run_id
    manifest = run(config, output_dir)
    print(f"S00 smoke run completed; manifest: {manifest}")


if __name__ == "__main__":
    main()
