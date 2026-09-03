from __future__ import annotations

import json

import pytest

from pi_branch_out.observation import BudgetObservation


def test_budget_observation_verifies_realized_action() -> None:
    raw = json.dumps(
        {
            "kind": "memory_budget_ratio",
            "requested_ratio": 0.8,
            "applied_ratio": 0.8,
            "granularity": "detailed",
            "feasible_budget_tokens": 20000,
            "budget_tokens": 16000,
            "injected_tokens": 12000,
        }
    )
    observation = BudgetObservation.parse(raw)
    observation.verify(0.8, "detailed")


def test_budget_observation_rejects_wrong_action() -> None:
    observation = BudgetObservation(
        requested_ratio=0.8,
        applied_ratio=0.6,
        feasible_budget_tokens=20000,
        budget_tokens=12000,
        injected_tokens=10000,
    )
    with pytest.raises(ValueError):
        observation.verify(0.8)


def test_budget_observation_rejects_wrong_granularity() -> None:
    observation = BudgetObservation(
        requested_ratio=0.8,
        applied_ratio=0.8,
        granularity="compact",
        feasible_budget_tokens=20000,
        budget_tokens=16000,
        injected_tokens=10000,
    )
    with pytest.raises(ValueError):
        observation.verify(0.8, "standard")


def test_budget_observation_rejects_budget_overflow() -> None:
    observation = BudgetObservation(
        requested_ratio=1.0,
        applied_ratio=1.0,
        feasible_budget_tokens=10000,
        budget_tokens=12000,
        injected_tokens=9000,
    )
    with pytest.raises(ValueError):
        observation.verify(1.0)
