from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .branch_task import build_branch_task
from .checkpoint import BranchAction, CheckpointManifest


AGENT_IMPORT = "pi_branch_out.harbor_agent:PiTdaiBranchAgent"


def _agent_kwargs(
    args: argparse.Namespace,
    *,
    checkpoint: Path | None = None,
    action: BranchAction | None = None,
) -> list[str]:
    pairs: list[tuple[str, str]] = [
        ("pi_executable", args.pi_executable),
        ("pi_thinking", args.pi_thinking),
        ("pi_extensions", ",".join(args.pi_extension or [])),
        ("tdai_state_dir", args.tdai_state_dir or ""),
    ]
    if args.branch_control_extension:
        pairs.append(("branch_control_extension", str(Path(args.branch_control_extension).resolve())))
    if checkpoint is not None:
        pairs.append(("checkpoint_dir", str(checkpoint.resolve())))
    if action is not None:
        pairs.append(("budget_ratio", str(action.budget_ratio)))
        pairs.append(
            (
                "require_budget_observation",
                "false" if getattr(args, "allow_unverified_budget", False) else "true",
            )
        )

    result: list[str] = []
    for key, value in pairs:
        result.extend(["--agent-kwarg", f"{key}={value}"])
    return result


def _harbor_command(
    args: argparse.Namespace,
    *,
    task: Path,
    jobs_dir: Path,
    checkpoint: Path | None = None,
    action: BranchAction | None = None,
) -> list[str]:
    command = shlex.split(args.harbor_bin)
    command.extend(
        [
            "run",
            "--path",
            str(task.resolve()),
            "--agent",
            AGENT_IMPORT,
            "--model",
            args.model,
            "--n-attempts",
            "1",
            "--n-concurrent",
            "1",
            "--resume-trajectory",
            "--jobs-dir",
            str(jobs_dir.resolve()),
        ]
    )
    command.extend(_agent_kwargs(args, checkpoint=checkpoint, action=action))
    return command


def _run(command: list[str], *, cwd: Path | None = None) -> int:
    print("[pi-branch-out]", shlex.join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=os.environ.copy(), check=False).returncode


def run_natural(args: argparse.Namespace) -> int:
    task = Path(args.task).resolve()
    jobs_dir = Path(args.jobs_dir).resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return _run(_harbor_command(args, task=task, jobs_dir=jobs_dir))


def run_branch(args: argparse.Namespace) -> int:
    source_task = Path(args.task).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    manifest = CheckpointManifest.load(checkpoint / "checkpoint.json")
    action = BranchAction(float(args.budget_ratio))

    run_id = args.run_id or datetime.now(timezone.utc).strftime("branch-%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root).resolve() / run_id
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"branch output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    branch_task = build_branch_task(source_task, checkpoint, output_root / "task")
    jobs_dir = output_root / "jobs"
    metadata = {
        "run_id": run_id,
        "source_task": str(source_task),
        "checkpoint": str(checkpoint),
        "source_step_index": manifest.step_index,
        "source_step_name": manifest.step_name,
        "has_native_pi_history": bool(manifest.pi_checkpoint_session),
        "action": action.as_runtime_payload(),
        "require_budget_observation": not args.allow_unverified_budget,
        "branch_task": str(branch_task),
        "jobs_dir": str(jobs_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / "branch_request.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rc = _run(
        _harbor_command(
            args,
            task=branch_task,
            jobs_dir=jobs_dir,
            checkpoint=checkpoint,
            action=action,
        )
    )
    metadata["return_code"] = rc
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output_root / "branch_result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rc


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--harbor-bin", default="harbor")
    parser.add_argument("--pi-executable", default="pi")
    parser.add_argument("--pi-thinking", default="off")
    parser.add_argument(
        "--pi-extension",
        action="append",
        default=[],
        help=(
            "Pi extension path visible inside the Harbor agent environment; repeatable. "
            "Put the TDAI Pi adapter here."
        ),
    )
    parser.add_argument(
        "--tdai-state-dir",
        default="",
        help=(
            "Optional TDAI local state directory inside the Harbor environment. "
            "Leave empty when TDAI uses an external Gateway; external state needs "
            "its own snapshot/namespace adapter."
        ),
    )
    parser.add_argument(
        "--branch-control-extension",
        default=str(Path(__file__).resolve().parents[2] / "extensions" / "tdai-budget-override.ts"),
        help="Host path to the one-shot TDAI budget bridge extension.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pi + TDAI + Harbor structured branch-out runner")
    sub = parser.add_subparsers(dest="command", required=True)

    natural = sub.add_parser(
        "natural",
        help="Run a natural multi-step task and capture pre-action checkpoints",
    )
    _common(natural)
    natural.add_argument("--task", required=True)
    natural.add_argument("--jobs-dir", required=True)
    natural.set_defaults(func=run_natural)

    branch = sub.add_parser(
        "branch",
        help="Restore one checkpoint and force one Memory budget action",
    )
    _common(branch)
    branch.add_argument("--task", required=True, help="Original EvoCodeBench task directory")
    branch.add_argument("--checkpoint", required=True)
    branch.add_argument("--budget-ratio", required=True, type=float)
    branch.add_argument("--output-root", default="branch_runs")
    branch.add_argument("--run-id")
    branch.add_argument(
        "--allow-unverified-budget",
        action="store_true",
        help=(
            "Do not fail when TDAI does not emit a realized budget observation. "
            "Use only while wiring the integration, never for training-data collection."
        ),
    )
    branch.set_defaults(func=run_branch)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
