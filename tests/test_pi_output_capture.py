from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from pi_branch_out.harbor_agent import PiTdaiBranchAgent


class OutputEnvironment:
    environment_name = "demo"

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, **kwargs) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(stdout="pi-exit=0\n", stderr="", return_code=0)

    async def download_file(self, source: str, target: Path) -> None:
        content = '{"type":"result","value":"ok"}\n' if source.endswith("stdout.jsonl") else ""
        Path(target).write_text(content, encoding="utf-8")


def test_pi_output_is_redirected_before_harbor_exec_buffers_it(tmp_path: Path) -> None:
    agent = PiTdaiBranchAgent(logs_dir=tmp_path, pi_executable="pi")
    agent._step_index = 1
    environment = OutputEnvironment()

    asyncio.run(
        agent._run_pi(
            "do the task",
            environment,
            resume=False,
            fork_session=None,
            budget_ratio=None,
            snapshot=None,
            continue_from_checkpoint=False,
        )
    )

    pi_command = environment.commands[0]
    assert ">/tmp/pi-branch-out-step-1.stdout.jsonl" in pi_command
    assert "2>/tmp/pi-branch-out-step-1.stderr.txt" in pi_command
    assert (tmp_path / "pi-step-1.stdout.jsonl").read_text(encoding="utf-8").startswith("{")
    assert (tmp_path / "pi-step-1.stderr.txt").read_text(encoding="utf-8") == ""
