"""Validation-ranked member selection and individual-member prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from itg_nn.ensemble import Ensemble
from itg_nn.model import CyclicInvariantNet

from .config import MemberSelectionConfig


def select_member_ids(
    members: Sequence[dict[str, Any]], selection: MemberSelectionConfig
) -> tuple[str, ...]:
    """Select checkpoint members without consulting test-set performance."""

    by_id = {str(member["id"]): member for member in members}
    if len(by_id) != len(members):
        raise ValueError("checkpoint member IDs must be unique")
    if selection.kind == "all":
        return tuple(str(member["id"]) for member in members)
    if selection.kind == "ids":
        unknown = sorted(set(selection.ids).difference(by_id))
        if unknown:
            raise ValueError(f"unknown checkpoint member IDs: {unknown}")
        return selection.ids
    ranked = sorted(
        members,
        key=lambda member: (-float(member["validation_r2"]), str(member["id"])),
    )
    return tuple(str(member["id"]) for member in ranked[: selection.count])


class MemberPredictor(nn.Module):
    """Return native clipped-log predictions for each selected ensemble member.

    The leading output axis is member and the second is sample.  This wrapper is
    intentionally free of aggregation so member-level signed attributions can be
    retained by all later steps.
    """

    def __init__(self, models: Sequence[CyclicInvariantNet]) -> None:
        super().__init__()
        if not models:
            raise ValueError("at least one model is required")
        self.models = nn.ModuleList(models)

    @classmethod
    def from_ensemble(cls, ensemble: Ensemble, member_ids: Sequence[str]) -> "MemberPredictor":
        index_by_id = {member_id: index for index, member_id in enumerate(ensemble.member_ids)}
        missing = sorted(set(member_ids).difference(index_by_id))
        if missing:
            raise ValueError(f"selected members are not in the loaded ensemble: {missing}")
        return cls([ensemble.models[index_by_id[member_id]] for member_id in member_ids])

    def forward(
        self,
        geometry: torch.Tensor,
        a_over_lt: torch.Tensor,
        a_over_ln: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [model(geometry, a_over_lt, a_over_ln).squeeze(-1) for model in self.models],
            dim=0,
        )
