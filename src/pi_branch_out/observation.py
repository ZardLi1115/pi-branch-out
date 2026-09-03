from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetObservation:
    requested_ratio: float
    applied_ratio: float
    feasible_budget_tokens: int
    budget_tokens: int
    injected_tokens: int
    candidate_tokens: int = 0
    snapshot_id: str = ""

    @classmethod
    def parse(cls, raw: str) -> "BudgetObservation":
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("kind") != "memory_budget_ratio":
            raise ValueError("invalid budget observation")
        return cls(
            requested_ratio=float(value["requested_ratio"]),
            applied_ratio=float(value["applied_ratio"]),
            feasible_budget_tokens=int(value["feasible_budget_tokens"]),
            budget_tokens=int(value["budget_tokens"]),
            injected_tokens=int(value["injected_tokens"]),
            candidate_tokens=int(value.get("candidate_tokens", 0)),
            snapshot_id=str(value.get("snapshot_id", "")),
        )

    def verify(self, expected_ratio: float, *, tolerance: float = 1e-9) -> None:
        if abs(self.requested_ratio - expected_ratio) > tolerance:
            raise ValueError(
                f"observation requested_ratio={self.requested_ratio} != expected={expected_ratio}"
            )
        if abs(self.applied_ratio - expected_ratio) > tolerance:
            raise ValueError(
                f"observation applied_ratio={self.applied_ratio} != expected={expected_ratio}"
            )
        if min(
            self.feasible_budget_tokens,
            self.budget_tokens,
            self.injected_tokens,
            self.candidate_tokens,
        ) < 0:
            raise ValueError("observation contains negative token counts")
        if self.budget_tokens > self.feasible_budget_tokens:
            raise ValueError("applied budget exceeds feasible budget")
        if self.injected_tokens > self.budget_tokens:
            raise ValueError("injected tokens exceed applied budget")
        if self.candidate_tokens and self.feasible_budget_tokens > self.candidate_tokens:
            raise ValueError("feasible budget exceeds frozen candidate pool")
