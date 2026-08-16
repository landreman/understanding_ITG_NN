"""Configuration shared by reproducible XAI commands.

JSON is deliberately used for the first scaffold so that configs have no parser
dependency beyond the Python standard library.  Later steps can add fields while
keeping the complete resolved config in each run manifest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from itg_nn.data import GradientSet
from itg_nn.infer import DEFAULT_CHECKPOINT as INFERENCE_CHECKPOINT


DEFAULT_DATASET = Path(
    "/Users/mattland/20260523-01-files_for_Kosmos_interpreting_neural_networks/"
    "20250102-01_GX_stellarator_dataset.h5"
)
DEFAULT_CHECKPOINT = INFERENCE_CHECKPOINT


@dataclass(frozen=True)
class MemberSelectionConfig:
    """A stable selection rule based only on checkpoint validation metadata."""

    kind: Literal["top_validation", "ids", "all"] = "top_validation"
    count: int | None = 1
    ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "top_validation" and (self.count is None or self.count < 1):
            raise ValueError("top_validation member selection requires count >= 1")
        if self.kind == "ids" and not self.ids:
            raise ValueError("ids member selection requires at least one member ID")


@dataclass(frozen=True)
class XAIConfig:
    """Minimum common configuration needed by every XAI run."""

    step: str
    run_id: str
    rows: tuple[int, ...]
    gradient_set: GradientSet = "varied"
    seed: int = 0
    device: str = "cpu"
    batch_size: int = 32
    member_selection: MemberSelectionConfig = field(default_factory=MemberSelectionConfig)
    dataset: Path = DEFAULT_DATASET
    checkpoint: Path = DEFAULT_CHECKPOINT

    def __post_init__(self) -> None:
        if not self.step.startswith("S"):
            raise ValueError("step must use the registered SNN identifier")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.rows:
            raise ValueError("at least one HDF5 row is required")
        if any(row < 0 for row in self.rows):
            raise ValueError("row IDs must be non-negative")
        if self.gradient_set not in ("fixed", "varied"):
            raise ValueError("gradient_set must be 'fixed' or 'varied'")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def as_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-safe fully resolved configuration."""

        result = asdict(self)
        result["dataset"] = str(self.dataset)
        result["checkpoint"] = str(self.checkpoint)
        result["rows"] = list(self.rows)
        result["member_selection"]["ids"] = list(self.member_selection.ids)
        return result


def load_config(path: str | Path) -> XAIConfig:
    """Read and validate one small JSON XAI configuration file."""

    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    selection = MemberSelectionConfig(
        kind=raw.get("member_selection", {}).get("kind", "top_validation"),
        count=raw.get("member_selection", {}).get("count", 1),
        ids=tuple(raw.get("member_selection", {}).get("ids", ())),
    )
    return XAIConfig(
        step=raw["step"],
        run_id=raw["run_id"],
        rows=tuple(int(row) for row in raw["rows"]),
        gradient_set=raw.get("gradient_set", "varied"),
        seed=int(raw.get("seed", 0)),
        device=raw.get("device", "cpu"),
        batch_size=int(raw.get("batch_size", 32)),
        member_selection=selection,
        dataset=Path(raw.get("dataset", DEFAULT_DATASET)),
        checkpoint=Path(raw.get("checkpoint", DEFAULT_CHECKPOINT)),
    )
