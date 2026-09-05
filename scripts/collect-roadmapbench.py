#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REQUIRED_ENV = (
    "TDAI_PROXY_URL", "TDAI_SPACE_ID", "TDAI_TEAM_ID", "TDAI_AGENT_ID",
    "TDAI_USER_KEY", "TDAI_MODEL", "CUSTOM_API_KEY",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run(command: list[str]) -> None:
    print("[roadmapbench-collector]", shlex.join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with rc={result.returncode}: {shlex.join(command)}")


def run_validated(
    command: list[str], validator: Callable[[], Any], *, max_attempts: int,
    retry_delay_seconds: float,
) -> Any:
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            try:
                return validator()
            except (FileNotFoundError, RuntimeError, json.JSONDecodeError):
                run(command)
                return validator()
        except Exception as error:
            errors.append(f"attempt {attempt}: {error}")
            if attempt < max_attempts:
                print(
                    f"[roadmapbench-collector] retrying after attempt {attempt}/{max_attempts}: {error}",
                    flush=True,
                )
                time.sleep(retry_delay_seconds)
    raise RuntimeError("; ".join(errors))


def trial_result_paths(jobs_dir: Path, task_name: str) -> list[Path]:
    return sorted(
        jobs_dir.glob(f"*/{task_name}__*/result.json"),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )


def valid_trial(
    jobs_dir: Path, task_name: str,
    validator: Callable[[Path, dict[str, Any]], Any],
) -> Any:
    paths = trial_result_paths(jobs_dir, task_name)
    if not paths:
        raise RuntimeError(f"no trial result found for {task_name} under {jobs_dir}")
    path = paths[0]
    result = json.loads(path.read_text(encoding="utf-8"))
    return validator(path, result)


def validate_prebuild(jobs_dir: Path, task_name: str) -> None:
    def validate(path: Path, result: dict[str, Any]) -> None:
        if error := result.get("exception_info"):
            raise RuntimeError(f"prebuild failed: {error}")
        print(f"[roadmapbench-collector] prebuild ready: {path}", flush=True)

    valid_trial(jobs_dir, task_name, validate)


def validate_natural(jobs_dir: Path, task_name: str, max_checkpoints: int) -> dict[str, Any]:
    def validate(path: Path, result: dict[str, Any]) -> dict[str, Any]:
        error = result.get("exception_info")
        verifier_compat_error = False
        if error:
            message = str(error.get("exception_message", ""))
            verifier_compat_error = (
                error.get("exception_type") == "ValidationError"
                and "rewards.phase_scores.float" in message
                and "rewards.phase_scores.int" in message
            )
            if not verifier_compat_error:
                raise RuntimeError(f"Natural failed: {error}")

        checkpoints_root = path.parent / "agent/model-call-checkpoints"
        states_path = checkpoints_root / "model-call-states.jsonl"
        usage_path = checkpoints_root / "model-call-usage.jsonl"
        collector_summary_path = checkpoints_root / "collector-summary.json"
        for required in (states_path, usage_path, collector_summary_path):
            if not required.is_file():
                raise RuntimeError(f"required collector artifact missing: {required}")
        states = [json.loads(line) for line in states_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        usage = [json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        collector_summary = json.loads(collector_summary_path.read_text(encoding="utf-8"))
        if not states:
            raise RuntimeError("no model calls recorded")
        if len(usage) != len(states):
            raise RuntimeError(f"usage/state row mismatch: {len(usage)} != {len(states)}")
        expected_calls = list(range(1, len(states) + 1))
        state_calls = [row.get("model_call_index") for row in states]
        usage_calls = [row.get("model_call_index") for row in usage]
        if state_calls != expected_calls or usage_calls != expected_calls:
            raise RuntimeError("state/usage ledgers must contain each model call exactly once in order")
        if collector_summary.get("calls_observed") != len(states):
            raise RuntimeError("collector summary call count does not match the state ledger")
        probe_errors = [row for row in states if row.get("checkpoint_status") == "probe-error"]
        if probe_errors:
            raise RuntimeError(f"{len(probe_errors)} failed candidate probes")
        saved = [row for row in states if row.get("checkpoint_status") == "ready"]
        if len(saved) > max_checkpoints:
            raise RuntimeError(f"checkpoint quota exceeded: {len(saved)} > {max_checkpoints}")
        if collector_summary.get("checkpoints_saved") != len(saved):
            raise RuntimeError("collector summary checkpoint count does not match the state ledger")
        if any(row.get("recall_snapshot_status") != "ready" for row in saved):
            raise RuntimeError("saved checkpoint without a ready recall snapshot")
        stderr_bytes = sum(item.stat().st_size for item in (path.parent / "agent").glob("pi-step-*.stderr.txt"))
        if stderr_bytes:
            raise RuntimeError(f"Pi stderr is not empty ({stderr_bytes} bytes)")

        if verifier_compat_error:
            reward_path = path.parent / "verifier/reward.json"
            if not reward_path.is_file():
                raise RuntimeError("Harbor verifier compatibility error without reward.json")
            rewards = json.loads(reward_path.read_text(encoding="utf-8"))
        else:
            rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        if not isinstance(rewards.get("reward"), (int, float)):
            raise RuntimeError(f"official reward is not numeric: {rewards}")

        usage_totals = {
            field: sum(int(row.get(field) or 0) for row in usage)
            for field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
        }
        cache_denominator = usage_totals["input_tokens"] + usage_totals["cache_read_tokens"]
        summary = {
            "task_name": task_name,
            "trial_result": str(path),
            "reward": rewards["reward"],
            "model_calls": len(states),
            "ready_checkpoints": len(saved),
            "candidate_probes": collector_summary.get("candidate_probes"),
            "checkpoint_bytes": collector_summary.get("cumulative_checkpoint_bytes"),
            **usage_totals,
            "cache_rate": usage_totals["cache_read_tokens"] / cache_denominator if cache_denominator else 0.0,
            "verifier_compat_error": verifier_compat_error,
        }
        for checkpoint_manifest in checkpoints_root.glob("call-*/checkpoint.json"):
            value = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
            value["source_reward"] = rewards["reward"]
            value["source_trial_dir"] = str(path.parent)
            atomic_write_json(checkpoint_manifest, value)
        print(f"[roadmapbench-collector] Natural ready: {json.dumps(summary)}", flush=True)
        return summary

    return valid_trial(jobs_dir, task_name, validate)


def remove_task_images(task_name: str, configured_image: str) -> None:
    try:
        listing = subprocess.run(
            ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, check=False,
        )
        if listing.returncode != 0:
            raise RuntimeError(listing.stderr.strip() or f"docker image ls rc={listing.returncode}")
        local_prefix = "hb__" + task_name.lower().replace(".", "-")
        targets = {
            image for image in listing.stdout.splitlines()
            if image in (configured_image, configured_image + ":latest") or image.startswith(local_prefix)
        }
        if targets and subprocess.run(["docker", "image", "rm", *sorted(targets)], check=False).returncode != 0:
            raise RuntimeError("docker image rm failed")
    except Exception as error:
        print(f"[roadmapbench-collector] warning: image cleanup failed for {task_name}: {error}", flush=True)


def load_status(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def committed_summaries(status_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(status_root.glob("*/status.json")):
        value = load_status(path)
        if value and value.get("state") == "committed" and isinstance(value.get("summary"), dict):
            summaries.append(value["summary"])
    return sorted(summaries, key=lambda item: item["task_name"])


def rebuild_indexes(output_root: Path, completed_file: Path) -> list[dict[str, Any]]:
    summaries = committed_summaries(output_root / "status")
    atomic_write_text(
        output_root / "natural-summary.jsonl",
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in summaries),
    )
    atomic_write_text(completed_file, "".join(item["task_name"] + "\n" for item in summaries))
    return summaries


def total_usage(summaries: list[dict[str, Any]]) -> dict[str, int]:
    fields = ("model_calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    totals = {field: sum(int(item.get(field) or 0) for item in summaries) for field in fields}
    totals["token_units"] = sum(totals[field] for field in fields if field != "model_calls")
    return totals


def observed_usage(output_root: Path) -> dict[str, int]:
    fields = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    totals = {"model_calls": 0, **{field: 0 for field in fields}}
    for usage_path in (output_root / "tasks").glob(
        "*/*/*/agent/model-call-checkpoints/model-call-usage.jsonl"
    ):
        for line in usage_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            totals["model_calls"] += 1
            for field in fields:
                totals[field] += int(row.get(field) or 0)
    totals["token_units"] = sum(totals[field] for field in fields)
    return totals


def limit_reached(args: argparse.Namespace, totals: dict[str, int], started: float) -> str | None:
    if args.max_wall_seconds and time.monotonic() - started >= args.max_wall_seconds:
        return "max_wall_seconds"
    if args.max_total_model_calls and totals["model_calls"] >= args.max_total_model_calls:
        return "max_total_model_calls"
    if args.max_total_token_units and totals["token_units"] >= args.max_total_token_units:
        return "max_total_token_units"
    return None


def docker_daemon_ready() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--pi-extension", type=Path, required=True)
    parser.add_argument("--completed-file", type=Path)
    parser.add_argument("--harbor-bin", default="harbor")
    parser.add_argument("--pi-branch-out-bin", default="pi-branch-out")
    parser.add_argument("--model", default="tdai/gpt-5.6-luna")
    parser.add_argument("--thinking", default="medium")
    parser.add_argument("--include-task-name", action="append", default=[])
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--max-wall-seconds", type=float, default=0)
    parser.add_argument("--max-total-model-calls", type=int, default=0)
    parser.add_argument("--max-total-token-units", type=int, default=0)
    parser.add_argument("--max-checkpoints", type=int, default=2)
    parser.add_argument("--min-checkpoint-gap", type=int, default=10)
    parser.add_argument("--sample-probability", type=float, default=0.1)
    parser.add_argument("--max-candidate-probes", type=int, default=8)
    parser.add_argument("--sampling-batch", default="default-v1")
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list-pending", action="store_true")
    args = parser.parse_args()

    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.max_tasks < 0 or args.max_checkpoints < 0:
        parser.error("task and checkpoint limits cannot be negative")
    if not 0 <= args.sample_probability <= 1:
        parser.error("--sample-probability must be in [0, 1]")

    include = set(args.include_task_name)
    tasks: list[tuple[str, Path, str]] = []
    for task_file in sorted(args.dataset.glob("*/task.toml")):
        task_name = task_file.parent.name
        if include and task_name not in include:
            continue
        config = tomllib.loads(task_file.read_text(encoding="utf-8"))
        image = config.get("environment", {}).get("docker_image")
        if not image:
            raise RuntimeError(f"docker_image missing: {task_file}")
        tasks.append((task_name, task_file.parent, image))
    if include:
        missing = sorted(include - {task[0] for task in tasks})
        if missing:
            raise RuntimeError(f"requested tasks are missing from the dataset: {missing}")
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    if not tasks:
        raise RuntimeError("no RoadmapBench tasks selected")

    args.output_root.mkdir(parents=True, exist_ok=True)
    completed_file = args.completed_file or args.output_root / "completed.txt"
    status_root = args.output_root / "status"
    status_root.mkdir(parents=True, exist_ok=True)
    summaries = rebuild_indexes(args.output_root, completed_file)
    completed = {item["task_name"] for item in summaries}
    pending = [task for task in tasks if task[0] not in completed]
    print(f"[roadmapbench-collector] completed={len(completed)} pending={len(pending)}", flush=True)
    if args.list_pending:
        for task_name, _, _ in pending:
            print(task_name)
        return 0

    missing_env = [key for key in REQUIRED_ENV if not os.environ.get(key)]
    if missing_env:
        raise RuntimeError("missing required environment keys: " + ", ".join(missing_env))
    if not args.runtime_archive.is_file():
        raise FileNotFoundError(args.runtime_archive)
    if not args.pi_extension.is_file():
        raise FileNotFoundError(args.pi_extension)
    for executable in (args.harbor_bin, args.pi_branch_out_bin):
        if not Path(executable).is_file() and shutil.which(executable) is None:
            raise FileNotFoundError(f"executable not found: {executable}")

    batch_manifest = {
        "version": 1,
        "dataset": str(args.dataset.resolve()),
        "output_root": str(args.output_root.resolve()),
        "model": args.model,
        "thinking": args.thinking,
        "tasks": [task[0] for task in tasks],
        "sampling": {
            "max_checkpoints": args.max_checkpoints,
            "min_checkpoint_gap": args.min_checkpoint_gap,
            "sample_probability": args.sample_probability,
            "max_candidate_probes": args.max_candidate_probes,
            "sampling_batch": args.sampling_batch,
        },
        "limits": {
            "max_tasks": args.max_tasks,
            "max_attempts": args.max_attempts,
            "min_free_gib": args.min_free_gib,
            "max_wall_seconds": args.max_wall_seconds,
            "max_total_model_calls": args.max_total_model_calls,
            "max_total_token_units": args.max_total_token_units,
        },
        "created_at": utc_now(),
    }
    batch_path = args.output_root / "batch.json"
    if batch_path.is_file():
        existing = json.loads(batch_path.read_text(encoding="utf-8"))
        for key in ("dataset", "model", "thinking", "tasks", "sampling", "limits"):
            if existing.get(key) != batch_manifest.get(key):
                raise RuntimeError(f"batch configuration changed for {key}; use a new output root")
        batch_manifest["created_at"] = existing.get("created_at", batch_manifest["created_at"])
    atomic_write_json(batch_path, batch_manifest)

    started = time.monotonic()
    failures: list[str] = []
    for task_name, task_dir, image in pending:
        summaries = rebuild_indexes(args.output_root, completed_file)
        totals = observed_usage(args.output_root)
        if not docker_daemon_ready():
            atomic_write_json(
                args.output_root / "batch-status.json",
                {
                    "state": "infrastructure-unavailable",
                    "reason": "docker-daemon",
                    "next_task": task_name,
                    "usage": totals,
                    "updated_at": utc_now(),
                },
            )
            print(f"[roadmapbench-collector] stopping before {task_name}: Docker daemon unavailable", flush=True)
            return 3
        if reason := limit_reached(args, totals, started):
            atomic_write_json(
                args.output_root / "batch-status.json",
                {"state": "budget-exhausted", "reason": reason, "usage": totals, "updated_at": utc_now()},
            )
            print(f"[roadmapbench-collector] stopping before {task_name}: {reason}", flush=True)
            return 0

        free_gib = shutil.disk_usage(args.output_root).free / (1024 ** 3)
        if free_gib < args.min_free_gib:
            raise RuntimeError(f"only {free_gib:.1f} GiB free before {task_name}; refusing to fill the disk")

        status_path = status_root / task_name / "status.json"
        status = {
            "version": 1, "task_name": task_name, "state": "running", "phase": "prebuild",
            "started_at": utc_now(), "updated_at": utc_now(),
        }
        atomic_write_json(status_path, status)
        prebuild_jobs = args.output_root / "prebuild" / task_name
        natural_jobs = args.output_root / "tasks" / task_name
        try:
            run_validated(
                [args.harbor_bin, "run", "--path", str(task_dir), "--agent", "oracle", "--install-only",
                 "--n-concurrent", "1", "--environment-build-timeout-multiplier", "2",
                 "--jobs-dir", str(prebuild_jobs), "--yes"],
                lambda: validate_prebuild(prebuild_jobs, task_name),
                max_attempts=args.max_attempts, retry_delay_seconds=args.retry_delay_seconds,
            )

            status.update(phase="initial-score", updated_at=utc_now())
            atomic_write_json(status_path, status)
            initial_score = args.output_root / "initial-scores" / f"{task_name}.json"
            if not initial_score.is_file():
                run(
                    [args.pi_branch_out_bin, "score-initial", "--task", str(task_dir), "--output-root",
                     str(args.output_root / "initial-score-jobs" / task_name / str(time.time_ns())),
                     "--score-output", str(initial_score), "--model", args.model,
                     "--harbor-bin", args.harbor_bin]
                )
            score = json.loads(initial_score.read_text(encoding="utf-8"))
            if (
                not isinstance(score, dict)
                or score.get("task_id") != task_name
                or not isinstance(score.get("reward"), (int, float))
                or score.get("official_verifier") is not True
                or score.get("isolated_copy") is not True
            ):
                raise RuntimeError(f"invalid initial score artifact: {initial_score}")

            status.update(phase="natural", updated_at=utc_now())
            atomic_write_json(status_path, status)
            summary = run_validated(
                [args.pi_branch_out_bin, "natural", "--task", str(task_dir), "--jobs-dir", str(natural_jobs),
                 "--model", args.model, "--harbor-bin", args.harbor_bin, "--pi-thinking", args.thinking,
                 "--checkpoint-boundary", "model-call", "--max-checkpoints", str(args.max_checkpoints),
                 "--min-checkpoint-gap", str(args.min_checkpoint_gap), "--sample-probability",
                 str(args.sample_probability), "--max-candidate-probes", str(args.max_candidate_probes),
                 "--sampling-batch", args.sampling_batch, "--pi-runtime-archive", str(args.runtime_archive),
                 "--pi-extension", str(args.pi_extension)],
                lambda: validate_natural(natural_jobs, task_name, args.max_checkpoints),
                max_attempts=1, retry_delay_seconds=args.retry_delay_seconds,
            )
            status.update(
                state="committed", phase="complete", summary=summary, finished_at=utc_now(), updated_at=utc_now(),
            )
            atomic_write_json(status_path, status)
            rebuild_indexes(args.output_root, completed_file)
            if not args.keep_images:
                remove_task_images(task_name, image)
        except Exception as error:
            status.update(state="failed", error=str(error), finished_at=utc_now(), updated_at=utc_now())
            atomic_write_json(status_path, status)
            failures.append(task_name)
            print(f"[roadmapbench-collector] task failed: {task_name}: {error}", flush=True)
            if args.fail_fast:
                raise

    summaries = rebuild_indexes(args.output_root, completed_file)
    totals = observed_usage(args.output_root)
    atomic_write_json(
        args.output_root / "batch-status.json",
        {"state": "complete" if not failures else "complete-with-failures", "committed_tasks": len(summaries),
         "failed_tasks": failures, "usage": totals, "updated_at": utc_now()},
    )
    print(f"[roadmapbench-collector] batch finished: committed={len(summaries)} failures={len(failures)}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
