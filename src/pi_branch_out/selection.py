from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def select_checkpoints(
    natural_root: Path,
    branch_root: Path,
    output_path: Path,
    *,
    per_task: int = 2,
    seed: str = "tdai-budget-selection-v1",
) -> dict[str, Any]:
    if per_task not in (1, 2):
        raise ValueError("the first collection pass supports one or two checkpoints per task")
    coverage: dict[str, int] = {}
    for request in branch_root.rglob("branch_request.json"):
        value = json.loads(request.read_text(encoding="utf-8"))
        checkpoint = str(Path(value.get("checkpoint", "")).resolve())
        coverage[checkpoint] = coverage.get(checkpoint, 0) + 1

    by_task: dict[str, list[dict[str, Any]]] = {}
    for state_log in natural_root.rglob("model-call-states.jsonl"):
        for line in state_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            state = json.loads(line)
            checkpoint_name = state.get("checkpoint")
            plans = state.get("action_plans") or []
            if state.get("checkpoint_status") != "ready" or not checkpoint_name or not plans:
                continue
            hashes = {plan.get("injected_content_sha256") for plan in plans if isinstance(plan, dict)}
            if len(hashes) <= 1:
                continue
            checkpoint = (state_log.parent / str(checkpoint_name)).resolve()
            key = str(checkpoint)
            jitter = int.from_bytes(hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()[:8], "big") / 2**64
            row = {
                "task_id": state.get("task"),
                "checkpoint": key,
                "model_call_index": state.get("model_call_index"),
                "unique_effective_actions": len(hashes),
                "existing_branches": coverage.get(key, 0),
                "candidate_memory_tokens": state.get("candidate_memory_tokens", 0),
                "selection_score": len(hashes) * 100 - coverage.get(key, 0) * 20 + jitter,
            }
            by_task.setdefault(str(state.get("task")), []).append(row)

    selected = []
    for task, rows in sorted(by_task.items()):
        selected.extend(sorted(rows, key=lambda row: (-row["selection_score"], row["model_call_index"]))[:per_task])
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "per_task": per_task,
        "selection_basis": "effective-content-diversity, undercoverage, deterministic-random-tiebreak",
        "selected": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
