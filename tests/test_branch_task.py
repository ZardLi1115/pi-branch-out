from __future__ import annotations

from pathlib import Path

import pytest
import tomlkit

from pi_branch_out.branch_task import build_branch_task
from pi_branch_out.checkpoint import BranchAction, CheckpointManifest


def test_budget_action_bounds() -> None:
    assert BranchAction(0).as_runtime_payload()["budget_ratio"] == 0
    assert BranchAction(1).action_id == "budget-1.000-standard"
    with pytest.raises(ValueError):
        BranchAction(-0.01)
    with pytest.raises(ValueError):
        BranchAction(1.01)


def test_branch_task_keeps_current_and_later_steps(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "environment").mkdir()
    for name in ("round-1", "round-2", "round-3"):
        step = task / "steps" / name
        step.mkdir(parents=True)
        (step / "instruction.md").write_text(name, encoding="utf-8")

    (task / "task.toml").write_text(
        """schema_version = \"1.2\"\n\n[metadata]\nname = \"demo\"\n\n[[steps]]\nname = \"round-1\"\n\n[[steps]]\nname = \"round-2\"\n\n[[steps]]\nname = \"round-3\"\n""",
        encoding="utf-8",
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    CheckpointManifest(
        task_name="demo",
        step_index=2,
        step_name="round-2",
        workspace_archive="workspace.tar.gz",
        pi_checkpoint_session="checkpoint-session.jsonl",
        pi_source_session="source.jsonl",
        pi_leaf_id="leaf",
    ).dump(checkpoint / "checkpoint.json")

    output = build_branch_task(task, checkpoint, tmp_path / "branch-task")
    doc = tomlkit.parse((output / "task.toml").read_text(encoding="utf-8"))
    assert [step["name"] for step in doc["steps"]] == ["round-2", "round-3"]
    assert doc["metadata"]["branch_out_source_step"] == 2
