from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointManifest:
    task_name: str
    step_index: int
    step_name: str
    workspace_archive: str
    pi_checkpoint_session: str
    pi_source_session: str
    pi_leaf_id: str
    tdai_state_archive: str | None = None
    tdai_state_mode: str = "none"
    source_trial_dir: str | None = None
    source_reward: float | None = None

    @classmethod
    def load(cls, path: Path) -> "CheckpointManifest":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint manifest must be an object: {path}")
        return cls(**value)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class BranchAction:
    """One-shot Memory budget action.

    ``budget_ratio`` is relative to the feasible dynamic-memory budget computed
    by the TDAI-side allocator. 0 means no automatic L1/L0 injection for this
    decision; 1 means the allocator may use the entire feasible budget.
    """

    budget_ratio: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.budget_ratio <= 1.0:
            raise ValueError("budget_ratio must be in [0, 1]")

    @property
    def action_id(self) -> str:
        return f"budget-{self.budget_ratio:.3f}"

    def as_runtime_payload(self) -> dict[str, Any]:
        return {
            "kind": "memory_budget_ratio",
            "budget_ratio": self.budget_ratio,
            "one_shot": True,
        }
