from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from pi_branch_out.checkpoint import CheckpointManifest
from pi_branch_out.harbor_agent import PiTdaiBranchAgent


class FakeEnvironment:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploads: list[tuple[Path, str]] = []

    async def exec(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        if command == "pwd":
            return SimpleNamespace(stdout="/app\n", stderr="", return_code=0)
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    async def upload_file(self, source: Path, destination: str) -> None:
        self.uploads.append((source, destination))


def test_restore_empty_git_patch_still_restores_untracked_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "workspace.patch").write_bytes(b"")
    (checkpoint / "workspace-untracked.tar.gz").write_bytes(b"archive")
    CheckpointManifest(
        task_name="demo",
        step_index=1,
        step_name="model-call-18",
        workspace_archive="",
        pi_checkpoint_session="session.jsonl",
        pi_source_session="session.jsonl",
        pi_leaf_id="leaf",
        workspace_mode="git-delta-v1",
        workspace_base_commit="abc123",
        workspace_patch="workspace.patch",
        workspace_untracked_archive="workspace-untracked.tar.gz",
    ).dump(checkpoint / "checkpoint.json")

    environment = FakeEnvironment()
    agent = object.__new__(PiTdaiBranchAgent)
    asyncio.run(agent._restore_checkpoint(environment, checkpoint))

    assert all("git -C /app apply" not in command for command in environment.commands)
    assert any("git -C /app reset --hard abc123" in command for command in environment.commands)
    assert any("tar -xzf /tmp/pi-branch-untracked.tar.gz" in command for command in environment.commands)
    assert environment.uploads == [
        (checkpoint / "workspace-untracked.tar.gz", "/tmp/pi-branch-untracked.tar.gz")
    ]
