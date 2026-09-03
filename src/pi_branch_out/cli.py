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
DEFAULT_ACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _agent_kwargs(args: argparse.Namespace, *, checkpoint: Path | None = None, action: BranchAction | None = None) -> list[str]:
    pairs: list[tuple[str, str]] = [
        ("pi_executable", args.pi_executable),
        ("pi_thinking", args.pi_thinking),
        ("pi_extensions", ",".join(args.pi_extension or [])),
    ]
    if args.branch_control_extension:
        pairs.append(("branch_control_extension", str(Path(args.branch_control_extension).resolve())))
    if checkpoint is not None:
        pairs.append(("checkpoint_dir", str(checkpoint.resolve())))
    if action is not None:
        pairs.append(("budget_ratio", str(action.budget_ratio)))
        pairs.append(("require_budget_observation", "false" if args.allow_unverified_budget else "true"))

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
            "run", "--path", str(task.resolve()),
            "--agent", AGENT_IMPORT,
            "--model", args.model,
            "--n-attempts", "1",
            "--n-concurrent", "1",
            "--resume-trajectory",
            "--jobs-dir", str(jobs_dir.resolve()),
        ]
    )
    command.extend(_agent_kwargs(args, checkpoint=checkpoint, action=action))
    return command


def _run(command: list[str]) -> int:
    print("[pi-branch-out]", shlex.join(command), flush=True)
    return subprocess.run(command, env=os.environ.copy(), check=False).returncode


def run_natural(args: argparse.Namespace) -> int:
    task = Path(args.task).resolve()
    jobs_dir = Path(args.jobs_dir).resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return _run(_harbor_command(args, task=task, jobs_dir=jobs_dir))


def _run_one_branch(args: argparse.Namespace, checkpoint: Path, ratio: float, output_root: Path, run_id: str) -> int:
    source_task = Path(args.task).resolve()
    manifest = CheckpointManifest.load(checkpoint / "checkpoint.json")
    if manifest.recall_snapshot_status != "ready" or not manifest.recall_snapshot:
        raise ValueError(
            f"checkpoint step {manifest.step_index} has no frozen recall snapshot; "
            "step 1 is baseline-only and cannot be used for strict branch-out"
        )
    action = BranchAction(ratio)
    run_root = output_root / run_id
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"branch output is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    branch_task = build_branch_task(source_task, checkpoint, run_root / "task")
    jobs_dir = run_root / "jobs"
    metadata = {
        "run_id": run_id,
        "source_task": str(source_task),
        "checkpoint": str(checkpoint),
        "source_step_index": manifest.step_index,
        "source_step_name": manifest.step_name,
        "recall_snapshot": manifest.recall_snapshot,
        "baseline_budget_ratio": manifest.baseline_budget_ratio,
        "action": action.as_runtime_payload(),
        "branch_task": str(branch_task),
        "jobs_dir": str(jobs_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_root / "branch_request.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rc = _run(_harbor_command(args, task=branch_task, jobs_dir=jobs_dir, checkpoint=checkpoint, action=action))
    metadata["return_code"] = rc
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    (run_root / "branch_result.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rc


def run_branch(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("branch-%Y%m%dT%H%M%SZ")
    return _run_one_branch(args, checkpoint, float(args.budget_ratio), Path(args.output_root).resolve(), run_id)


def _parse_ratios(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        values = list(DEFAULT_ACTIONS)
    for value in values:
        BranchAction(value)
    return values


def run_branch_grid(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).resolve()
    ratios = _parse_ratios(args.ratios)
    root = Path(args.output_root).resolve()
    stamp = args.run_id_prefix or datetime.now(timezone.utc).strftime("grid-%Y%m%dT%H%M%SZ")
    failures = 0
    for ratio in ratios:
        run_id = f"{stamp}-budget-{ratio:.3f}"
        rc = _run_one_branch(args, checkpoint, ratio, root, run_id)
        if rc != 0:
            failures += 1
            if args.fail_fast:
                return rc
    return 1 if failures else 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--harbor-bin", default="harbor")
    parser.add_argument("--pi-executable", default="pi")
    parser.add_argument("--pi-thinking", default="off")
    parser.add_argument(
        "--pi-extension", action="append", default=[],
        help="Pi extension path visible inside Harbor; repeatable. Add the official TDAI Pi plugin here.",
    )
    parser.add_argument(
        "--branch-control-extension",
        default=str(Path(__file__).resolve().parents[2] / "extensions" / "tdai-budget-override.ts"),
        help="Host path to the external budget adapter. It never patches TDAI.",
    )


def _branch_common(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default="branch_runs")
    parser.add_argument(
        "--allow-unverified-budget", action="store_true",
        help="Keep a branch if the external adapter does not emit an observation. Wiring-only; do not use for training data.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pi + TDAI + Harbor structured branch-out runner")
    sub = parser.add_subparsers(dest="command", required=True)

    natural = sub.add_parser("natural", help="Run the untouched Pi+TDAI baseline and freeze pre-action recall snapshots")
    _common(natural)
    natural.add_argument("--task", required=True)
    natural.add_argument("--jobs-dir", required=True)
    natural.set_defaults(func=run_natural)

    branch = sub.add_parser("branch", help="Restore one checkpoint and force one Memory Budget ratio")
    _branch_common(branch)
    branch.add_argument("--budget-ratio", required=True, type=float)
    branch.add_argument("--run-id")
    branch.set_defaults(func=run_branch)

    grid = sub.add_parser("branch-grid", help="Run several budget counterfactuals from the same frozen checkpoint")
    _branch_common(grid)
    grid.add_argument("--ratios", default=",".join(str(v) for v in DEFAULT_ACTIONS))
    grid.add_argument("--run-id-prefix")
    grid.add_argument("--fail-fast", action="store_true")
    grid.set_defaults(func=run_branch_grid)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
