from __future__ import annotations

import shutil
from pathlib import Path

import tomlkit

from .checkpoint import CheckpointManifest


def build_branch_task(source_task: Path, checkpoint_dir: Path, output_dir: Path) -> Path:
    """Create a Harbor task that starts at the checkpoint's current step.

    The workspace itself is restored by ``PiTdaiBranchAgent.setup``. This helper
    only trims the Harbor step list so the branch receives the same user request
    that was about to be processed at the checkpoint, then continues through the
    remaining EvoCodeBench rounds normally.
    """
    source_task = source_task.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    output_dir = output_dir.resolve()
    manifest = CheckpointManifest.load(checkpoint_dir / "checkpoint.json")

    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"branch task output is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)

    shutil.copytree(
        source_task,
        output_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("harbor_jobs", ".git", "__pycache__"),
    )

    task_toml = output_dir / "task.toml"
    doc = tomlkit.parse(task_toml.read_text(encoding="utf-8"))
    raw_steps = doc.get("steps")
    if raw_steps is None:
        raise ValueError(f"task has no [[steps]] array: {task_toml}")
    steps = list(raw_steps)
    if not steps:
        raise ValueError(f"task has an empty [[steps]] array: {task_toml}")

    # step_index is 1-based and points to the user request that has not yet run.
    start = manifest.step_index - 1
    if start < 0 or start >= len(steps):
        raise ValueError(
            f"checkpoint step_index={manifest.step_index} outside task step range 1..{len(steps)}"
        )
    kept = steps[start:]
    new_steps = tomlkit.aot()
    for step in kept:
        new_steps.append(step)
    doc["steps"] = new_steps

    metadata = doc.get("metadata")
    if metadata is not None:
        metadata["branch_out_source_step"] = manifest.step_index
        metadata["branch_out_checkpoint"] = str(checkpoint_dir)

    task_toml.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return output_dir
