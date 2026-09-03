from __future__ import annotations

import pytest

from pi_branch_out.checkpoint import BranchAction


def test_branch_action_contains_ratio_and_granularity() -> None:
    action = BranchAction(0.6, "detailed")
    assert action.action_id == "budget-0.600-detailed"
    assert action.as_runtime_payload() == {
        "kind": "memory_budget_ratio",
        "budget_ratio": 0.6,
        "granularity": "detailed",
        "one_shot": True,
    }


def test_branch_action_defaults_to_standard() -> None:
    assert BranchAction(0.2).granularity == "standard"


def test_branch_action_rejects_invalid_granularity() -> None:
    with pytest.raises(ValueError):
        BranchAction(0.5, "huge")  # type: ignore[arg-type]
