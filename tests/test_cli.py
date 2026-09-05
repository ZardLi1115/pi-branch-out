from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pi_branch_out.cli as cli
from pi_branch_out.checkpoint import CheckpointManifest
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


def test_branch_grid_skips_equivalent_rendered_actions(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    snapshot = {
        "version": 1,
        "query": "q",
        "action_plans": [
            {"ratio": 0.0, "injected_content_sha256": "empty"},
            {"ratio": 0.2, "injected_content_sha256": "empty"},
            {"ratio": 0.4, "injected_content_sha256": "different"},
        ],
    }
    snapshot_bytes = (json.dumps(snapshot) + "\n").encode("utf-8")
    (checkpoint / "recall-snapshot.json").write_bytes(snapshot_bytes)
    CheckpointManifest(
        task_name="task", step_index=1, step_name="call-2", workspace_archive="",
        pi_checkpoint_session="session.jsonl", pi_source_session="session.jsonl", pi_leaf_id="leaf",
        recall_snapshot="recall-snapshot.json", recall_snapshot_status="ready",
        snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
    ).dump(checkpoint / "checkpoint.json")
    calls: list[float] = []
    monkeypatch.setattr(cli, "_run_one_branch", lambda args, cp, ratio, root, run_id: calls.append(ratio) or 0)
    args = build_parser().parse_args([
        "branch-grid", "--task", str(tmp_path / "task"), "--checkpoint", str(checkpoint),
        "--output-root", str(tmp_path / "out"), "--model", "tdai/model",
        "--ratios", "0,0.2,0.4", "--tdai-isolation-mode", "isolated-instance",
        "--backend-instance-id", "test-{run_id}", "--backend-proxy-url", "http://{run_id}.isolated.test:8096",
    ])
    assert cli.run_branch_grid(args) == 0
    assert calls == [0.0, 0.4]
