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

    @classmethod
    def parse(cls, raw: str) -> "BudgetObservation":
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("kind") != "memory_budget_ratio":
            raise ValueError("invalid TDAI budget observation")
        return cls(
            requested_ratio=float(value["requested_ratio"]),
            applied_ratio=float(value["applied_ratio"]),
            feasible_budget_tokens=int(value["feasible_budget_tokens"]),
            budget_tokens=int(value["budget_tokens"]),
            injected_tokens=int(value["injected_tokens"]),
        )

    def verify(self, expected_ratio: float, *, tolerance: float = 1e-9) -> None:
        if abs(self.requested_ratio - expected_ratio) > tolerance:
            raise ValueError(
                f"TDAI observation requested_ratio={self.requested_ratio} != expected={expected_ratio}"
            )
        if abs(self.applied_ratio - expected_ratio) > tolerance:
            raise ValueError(
                f"TDAI observation applied_ratio={self.applied_ratio} != expected={expected_ratio}"
            )
        if self.feasible_budget_tokens < 0 or self.budget_tokens < 0 or self.injected_tokens < 0:
            raise ValueError("TDAI observation contains negative token counts")
        if self.budget_tokens > self.feasible_budget_tokens:
            raise ValueError("TDAI applied budget exceeds feasible budget")
        if self.injected_tokens > self.budget_tokens:
            raise ValueError("TDAI injected tokens exceed applied budget")
