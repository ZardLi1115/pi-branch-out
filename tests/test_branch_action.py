from __future__ import annotations

import pytest

from pi_branch_out.checkpoint import BranchAction


def test_branch_action_contains_ratio_only() -> None:
    action = BranchAction(0.6)
    assert action.action_id == "budget-0.600"
    assert action.as_runtime_payload() == {
        "kind": "memory_budget_ratio",
        "budget_ratio": 0.6,
        "one_step": True,
    }


def test_branch_action_rejects_out_of_range_ratio() -> None:
    with pytest.raises(ValueError):
        BranchAction(-0.01)
    with pytest.raises(ValueError):
        BranchAction(1.01)
