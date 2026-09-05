#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path


REQUIRED_ENV = (
    "TDAI_PROXY_URL",
    "TDAI_SPACE_ID",
    "TDAI_TEAM_ID",
    "TDAI_AGENT_ID",
    "TDAI_USER_KEY",
    "TDAI_MODEL",
    "CUSTOM_API_KEY",
)


def run(command: list[str]) -> None:
    print("[roadmapbench-collector]", shlex.join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with rc={result.returncode}: {shlex.join(command)}")


def latest_trial_result(jobs_dir: Path, task_name: str) -> tuple[Path, dict]:
    candidates = list(jobs_dir.glob(f"*/{task_name}__*/result.json"))
    if not candidates:
        raise RuntimeError(f"no trial result found for {task_name} under {jobs_dir}")
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    return path, json.loads(path.read_text(encoding="utf-8"))


def validate_prebuild(jobs_dir: Path, task_name: str) -> None:
    path, result = latest_trial_result(jobs_dir, task_name)
    if error := result.get("exception_info"):
        raise RuntimeError(f"prebuild failed for {task_name}: {error}")
    print(f"[roadmapbench-collector] prebuild ready: {path}", flush=True)


def validate_natural(jobs_dir: Path, task_name: str) -> dict:
    path, result = latest_trial_result(jobs_dir, task_name)
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
            raise RuntimeError(f"Natural failed for {task_name}: {error}")

    states_path = path.parent / "agent/model-call-checkpoints/model-call-states.jsonl"
    if not states_path.is_file():
        raise RuntimeError(f"model-call states missing for {task_name}: {states_path}")
    states = [
        json.loads(line)
        for line in states_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not states:
        raise RuntimeError(f"no model calls recorded for {task_name}")
    not_ready = [row for row in states[1:] if row.get("recall_snapshot_status") != "ready"]
    if not_ready:
        raise RuntimeError(f"{task_name} has {len(not_ready)} non-ready checkpoints")

    stderr_bytes = sum(
        item.stat().st_size for item in (path.parent / "agent").glob("pi-step-*.stderr.txt")
    )
    if stderr_bytes:
        raise RuntimeError(f"{task_name} Pi stderr is not empty ({stderr_bytes} bytes)")

    if verifier_compat_error:
        reward_path = path.parent / "verifier/reward.json"
        if not reward_path.is_file():
            raise RuntimeError(
                f"Harbor verifier compatibility error without reward.json for {task_name}"
            )
        rewards = json.loads(reward_path.read_text(encoding="utf-8"))
        if not isinstance(rewards.get("reward"), (int, float)):
            raise RuntimeError(f"official reward is not numeric for {task_name}: {rewards}")
    else:
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    summary = {
        "task_name": task_name,
        "trial_result": str(path),
        "reward": rewards.get("reward"),
        "model_calls": len(states),
        "ready_checkpoints": len(states) - 1,
        "verifier_compat_error": verifier_compat_error,
    }
    print(f"[roadmapbench-collector] Natural ready: {json.dumps(summary)}", flush=True)
    return summary


def remove_task_images(task_name: str, configured_image: str) -> None:
    listing = subprocess.run(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    local_prefix = "hb__" + task_name.lower().replace(".", "-")
    targets = {
        image
        for image in listing.stdout.splitlines()
        if image == configured_image
        or image == configured_image + ":latest"
        or image.startswith(local_prefix)
    }
    if not targets:
        return
    result = subprocess.run(["docker", "image", "rm", *sorted(targets)], check=False)
    if result.returncode != 0:
        print(f"[roadmapbench-collector] warning: could not remove all images for {task_name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--pi-extension", type=Path, required=True)
    parser.add_argument("--completed-file", type=Path, required=True)
    parser.add_argument("--model", default="tdai/gpt-5.6-luna")
    parser.add_argument("--thinking", default="medium")
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--list-pending", action="store_true")
    args = parser.parse_args()

    tasks: list[tuple[str, Path, str]] = []
    for task_file in sorted(args.dataset.glob("*/task.toml")):
        config = tomllib.loads(task_file.read_text(encoding="utf-8"))
        image = config.get("environment", {}).get("docker_image")
        if not image:
            raise RuntimeError(f"docker_image missing: {task_file}")
        tasks.append((task_file.parent.name, task_file.parent, image))

    completed = set()
    if args.completed_file.is_file():
        completed = {
            line.strip()
            for line in args.completed_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
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

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.completed_file.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "natural-summary.jsonl"

    # Reclaim Roadmap images whose Natural runs were already accepted.
    task_by_name = {name: (path, image) for name, path, image in tasks}
    for task_name in sorted(completed):
        if task_name in task_by_name:
            remove_task_images(task_name, task_by_name[task_name][1])

    for task_name, task_dir, image in pending:
        free_gib = shutil.disk_usage(args.output_root).free / (1024**3)
        if free_gib < args.min_free_gib:
            raise RuntimeError(
                f"only {free_gib:.1f} GiB free before {task_name}; refusing to fill the disk"
            )

        prebuild_jobs = args.output_root / "prebuild" / task_name
        natural_jobs = args.output_root / "tasks" / task_name
        run(
            [
                "/opt/anaconda3/bin/harbor",
                "run",
                "--path",
                str(task_dir),
                "--agent",
                "oracle",
                "--install-only",
                "--n-concurrent",
                "1",
                "--environment-build-timeout-multiplier",
                "2",
                "--jobs-dir",
                str(prebuild_jobs),
                "--yes",
            ]
        )
        validate_prebuild(prebuild_jobs, task_name)

        try:
            summary = validate_natural(natural_jobs, task_name)
            print(f"[roadmapbench-collector] reusing accepted Natural: {task_name}", flush=True)
        except RuntimeError:
            run(
                [
                    "/opt/anaconda3/bin/pi-branch-out",
                    "natural",
                    "--task",
                    str(task_dir),
                    "--jobs-dir",
                    str(natural_jobs),
                    "--model",
                    args.model,
                    "--harbor-bin",
                    "/opt/anaconda3/bin/harbor",
                    "--pi-thinking",
                    args.thinking,
                    "--checkpoint-boundary",
                    "model-call",
                    "--pi-runtime-archive",
                    str(args.runtime_archive),
                    "--pi-extension",
                    str(args.pi_extension),
                ]
            )
            summary = validate_natural(natural_jobs, task_name)
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        with args.completed_file.open("a", encoding="utf-8") as handle:
            handle.write(task_name + "\n")
        remove_task_images(task_name, image)

    print("[roadmapbench-collector] all tasks complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
