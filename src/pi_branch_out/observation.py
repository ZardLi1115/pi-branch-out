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
    injected_content_sha256: str = ""
    effective_action_id: str = ""
    tokenizer_version: str = ""
    context_tokens_before_injection: int = 0
    context_tokens_after_injection: int = 0

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
            injected_content_sha256=str(value.get("injected_content_sha256", "")),
            effective_action_id=str(value.get("effective_action_id", "")),
            tokenizer_version=str(value.get("tokenizer_version", "")),
            context_tokens_before_injection=int(value.get("context_tokens_before_injection", 0)),
            context_tokens_after_injection=int(value.get("context_tokens_after_injection", 0)),
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
            self.context_tokens_before_injection,
            self.context_tokens_after_injection,
        ) < 0:
            raise ValueError("observation contains negative token counts")
        if self.budget_tokens > self.feasible_budget_tokens:
            raise ValueError("applied budget exceeds feasible budget")
        if self.injected_tokens > self.budget_tokens:
            raise ValueError("injected tokens exceed applied budget")
        if self.candidate_tokens and self.feasible_budget_tokens > self.candidate_tokens:
            raise ValueError("feasible budget exceeds frozen candidate pool")
        if self.context_tokens_after_injection:
            delta = self.context_tokens_after_injection - self.context_tokens_before_injection
            if delta != self.injected_tokens:
                raise ValueError("context token delta does not equal rendered injection tokens")
        if self.injected_content_sha256:
            if len(self.injected_content_sha256) != 64:
                raise ValueError("invalid injected content SHA-256")
            expected_action = f"sha256:{self.injected_content_sha256}"
            if self.effective_action_id and self.effective_action_id != expected_action:
                raise ValueError("effective action id does not match injected content")
