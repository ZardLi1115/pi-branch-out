from __future__ import annotations

from pathlib import Path

from pi_branch_out.cli import _harbor_command, build_parser


def test_natural_passes_offline_runtime_and_1200_second_timeout(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "natural",
            "--task",
            str(tmp_path / "task"),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--model",
            "tdai/gpt-5.6-luna",
            "--pi-runtime-archive",
            str(tmp_path / "pi-runtime.tar.gz"),
            "--include-task-name",
            "plr-*",
            "--n-tasks",
            "5",
        ]
    )

    command = _harbor_command(
        args,
        task=tmp_path / "task",
        jobs_dir=tmp_path / "jobs",
    )

    assert command[command.index("--environment-build-timeout-multiplier") + 1] == "2.0"
    runtime_kwarg = next(item for item in command if item.startswith("pi_runtime_archive="))
    assert runtime_kwarg == f"pi_runtime_archive={tmp_path / 'pi-runtime.tar.gz'}"
    assert command[command.index("--include-task-name") + 1] == "plr-*"
    assert command[command.index("--n-tasks") + 1] == "5"
