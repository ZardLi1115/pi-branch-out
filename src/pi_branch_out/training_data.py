from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_roadmapbench_split(
    overview_path: Path,
    output_path: Path,
    *,
    seed: str = "tdai-budget-v1",
    train_fraction: float = 0.7,
    dev_fraction: float = 0.15,
) -> dict[str, Any]:
    if train_fraction <= 0 or dev_fraction <= 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("split fractions must be positive and leave a non-empty test fraction")
    rows = _read_jsonl(overview_path)
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_task_ids: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("every RoadmapBench row must contain task_id")
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate RoadmapBench task_id: {task_id}")
        seen_task_ids.add(task_id)
        key = (str(row.get("language", "unknown")), str(row.get("domain", "unknown")))
        by_stratum.setdefault(key, []).append(row)

    assignments: dict[str, str] = {}
    for key, members in sorted(by_stratum.items()):
        ordered = sorted(
            members,
            key=lambda row: hashlib.sha256(
                f"{seed}\0{row['task_id']}".encode("utf-8")
            ).hexdigest(),
        )
        n = len(ordered)
        n_train = max(1, round(n * train_fraction)) if n >= 3 else max(1, n - 1)
        n_dev = max(1, round(n * dev_fraction)) if n >= 3 else (1 if n == 2 else 0)
        if n_train + n_dev >= n and n >= 3:
            n_train = max(1, n - n_dev - 1)
        for index, row in enumerate(ordered):
            split = "train" if index < n_train else "dev" if index < n_train + n_dev else "test"
            assignments[str(row["task_id"])] = split

    manifest = {
        "schema_version": 1,
        "benchmark": "RoadmapBench",
        "seed": seed,
        "method": "language-domain-stratified-sha256",
        "fractions": {
            "train": train_fraction,
            "dev": dev_fraction,
            "test": 1.0 - train_fraction - dev_fraction,
        },
        "source_sha256": hashlib.sha256(overview_path.read_bytes()).hexdigest(),
        "assignments": dict(sorted(assignments.items())),
        "counts": {
            split: sum(value == split for value in assignments.values())
            for split in ("train", "dev", "test")
        },
    }
    manifest["manifest_sha256"] = _fingerprint(manifest)
    _write_json(output_path, manifest)
    return manifest


@dataclass(frozen=True)
class PriceTable:
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float
    cache_write_per_million: float

    def cost(self, usage: dict[str, Any]) -> float:
        return (
            float(usage.get("input_tokens", 0)) * self.input_per_million
            + float(usage.get("output_tokens", 0)) * self.output_per_million
            + float(usage.get("cache_read_tokens", 0)) * self.cache_read_per_million
            + float(usage.get("cache_write_tokens", 0)) * self.cache_write_per_million
        ) / 1_000_000.0


def _task_reward(trial_dir: Path) -> tuple[float | None, bool]:
    result_path = trial_dir / "result.json"
    if result_path.is_file():
        result = _read_json(result_path)
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        if isinstance(reward, (int, float)):
            return float(reward), bool(result.get("exception_info"))
    reward_path = trial_dir / "verifier" / "reward.json"
    if reward_path.is_file():
        reward = _read_json(reward_path).get("reward")
        if isinstance(reward, (int, float)):
            return float(reward), False
    return None, True


def _usage_by_call(state_path: Path) -> dict[int, dict[str, Any]]:
    usage_path = state_path.with_name("model-call-usage.jsonl")
    if not usage_path.is_file():
        return {}
    result: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(usage_path):
        index = int(row.get("model_call_index", 0))
        current = result.setdefault(index, {
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
            "usage_schema": "",
        })
        schema = str(row.get("usage_schema", ""))
        if current["usage_schema"] and schema != current["usage_schema"]:
            raise ValueError(f"mixed usage schemas for model call {index}: {usage_path}")
        current["usage_schema"] = schema
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            current[key] += max(0, int(row.get(key, 0)))
    return result


def _score_at_checkpoint(state_path: Path, call_index: int) -> float | None:
    path = state_path.parent / f"call-{call_index:03d}" / "checkpoint-score.json"
    if not path.is_file():
        return None
    value = _read_json(path)
    if value.get("official_verifier") is not True or value.get("isolated_copy") is not True:
        raise ValueError(f"untrusted checkpoint score: {path}")
    score = value.get("reward")
    return float(score) if isinstance(score, (int, float)) else None


def _visible_state(row: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "task", "model_call_index", "context_tokens", "context_window_tokens",
        "reserve_tokens", "remaining_call_budget", "remaining_cost_budget_usd",
        "remaining_time_seconds", "query", "recent_tool_result",
        "candidate_memory_tokens", "candidate_count", "l1_count", "l0_count",
        "l1_lengths", "l0_lengths", "l1_scores", "l0_scores",
        "default_actual_memory_tokens", "default_mapped_action",
        "previous_actual_memory_tokens", "previous_mapped_action", "previous_budget_tokens",
        "snapshot_sha256",
    )
    return {key: row.get(key) for key in allowed}


def _trajectory_records(
    state_path: Path,
    *,
    trajectory_id: str,
    split: str,
    action_default: float,
    final_reward: float,
    truncated: bool,
    prices: PriceTable,
    cost_coefficient: float,
    cost_normalizer_usd: float,
    policy_version: str,
    isolation_mode: str,
    training_eligible: bool,
    fork_id: str,
    initial_score: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    states = sorted(_read_jsonl(state_path), key=lambda row: int(row.get("model_call_index", 0)))
    usage = _usage_by_call(state_path)
    prefixes: list[dict[str, Any]] = []
    ids: list[str] = []
    for row in states:
        visible = _visible_state(row)
        state_id = _fingerprint({"task": row.get("task"), "visible": visible})
        ids.append(state_id)
        prefixes.append({"state_id": state_id, "task_id": row.get("task"), "split": split, "state": visible})

    observed = initial_score
    scores: list[float] = []
    for row in states:
        score = _score_at_checkpoint(state_path, int(row.get("model_call_index", 0)))
        if score is not None:
            observed = score
        scores.append(observed)

    transitions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for index, row in enumerate(states):
        call = int(row.get("model_call_index", 0))
        action = float(row.get("mapped_action", row.get("default_mapped_action", action_default)))
        call_usage = usage.get(call)
        cost_missing = call_usage is None or call_usage.get("usage_schema") != "pi-exclusive-input-cache-v1"
        billed = prices.cost(call_usage or {})
        next_score = scores[index + 1] if index + 1 < len(scores) else final_reward
        quality_delta = next_score - scores[index]
        normalized_cost = billed / cost_normalizer_usd
        done = index + 1 == len(states)
        transitions.append({
            "schema_version": 1,
            "trajectory_id": trajectory_id,
            "task_id": row.get("task"),
            "split": split,
            "state_id": ids[index],
            "next_state_id": None if done else ids[index + 1],
            "action": action,
            "reward": quality_delta - cost_coefficient * normalized_cost,
            "quality_delta": quality_delta,
            "cost_usd": billed,
            "normalized_cost": normalized_cost,
            "usage": call_usage,
            "cost_missing": cost_missing,
            "done": done and not truncated,
            "truncated": done and truncated,
            "policy_version": policy_version,
            "isolation_mode": isolation_mode,
            "training_eligible": training_eligible and not cost_missing and not (done and truncated),
            "fork_id": fork_id,
            "injected_content_sha256": str(row.get("actual_injected_content_sha256", EMPTY_SHA256)),
            "effective_action_id": row.get("effective_action_id"),
        })
        if isolation_mode == "natural":
            content_hash = str(row.get("actual_injected_content_sha256", EMPTY_SHA256))
            labels.append({
                "schema_version": 1,
                "task_id": row.get("task"),
                "trajectory_id": trajectory_id,
                "split": split,
                "state_id": ids[index],
                "default_action": float(row.get("default_mapped_action", 0)),
                "actual_memory_tokens": int(row.get("default_actual_memory_tokens", 0)),
                "actual_injected_content_sha256": content_hash,
                "allocator_content_match": content_hash == EMPTY_SHA256,
                "match_basis": "current-pi-has-no-extra-dynamic-l1-l0-injection",
            })
    return prefixes, transitions, labels


def export_training_data(
    natural_root: Path,
    branch_root: Path,
    output_dir: Path,
    split_manifest_path: Path,
    *,
    prices: PriceTable,
    cost_coefficient: float,
    cost_normalizer_usd: float = 1.0,
    actions: tuple[float, ...] = DEFAULT_ACTIONS,
) -> dict[str, Any]:
    if cost_coefficient < 0 or cost_normalizer_usd <= 0:
        raise ValueError("invalid cost normalization")
    split_manifest = _read_json(split_manifest_path)
    assignments = split_manifest.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("split manifest has no assignments")

    all_prefixes: dict[str, dict[str, Any]] = {}
    transitions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    trajectories = 0

    def ingest(state_path: Path, metadata: dict[str, Any], *, natural: bool) -> None:
        nonlocal trajectories
        state_rows = _read_jsonl(state_path)
        if not state_rows:
            return
        task = str(state_rows[0].get("task", ""))
        if task not in assignments:
            raise ValueError(f"task {task!r} is absent from split manifest")
        trial_dir = state_path.parents[2]
        reward, truncated = _task_reward(trial_dir)
        reward_missing = reward is None
        if reward_missing:
            reward = 0.0
            truncated = True
        trajectory_id = _fingerprint({"state_path": str(state_path.resolve()), "metadata": metadata})
        if natural:
            score_path = natural_root / "initial-scores" / f"{task}.json"
        else:
            score_path = Path(str(metadata.get("checkpoint", ""))) / "checkpoint-score.json"
        initial_score = 0.0
        initial_score_missing = not score_path.is_file()
        if score_path.is_file():
            score_record = _read_json(score_path)
            if score_record.get("official_verifier") is not True or score_record.get("isolated_copy") is not True:
                raise ValueError(f"untrusted initial score: {score_path}")
            initial_score = float(score_record["reward"])
        prefixes, rows, weak_labels = _trajectory_records(
            state_path,
            trajectory_id=trajectory_id,
            split=str(assignments[task]),
            action_default=0.0,
            final_reward=reward,
            truncated=truncated,
            prices=prices,
            cost_coefficient=cost_coefficient,
            cost_normalizer_usd=cost_normalizer_usd,
            policy_version=str(metadata.get("policy_version", "baseline-v1")),
            isolation_mode="natural" if natural else str(metadata.get("tdai_isolation_mode", "shared")),
            training_eligible=(
                (natural or bool(metadata.get("training_eligible_long_horizon", False)))
                and not reward_missing
                and not initial_score_missing
            ),
            fork_id=(
                trajectory_id if natural else _fingerprint({
                    "checkpoint": metadata.get("checkpoint"),
                    "snapshot_sha256": metadata.get("snapshot_sha256"),
                })
            ),
            initial_score=initial_score,
        )
        for prefix in prefixes:
            all_prefixes.setdefault(str(prefix["state_id"]), prefix)
        transitions.extend(rows)
        labels.extend(weak_labels)
        trajectories += 1

    for state_path in sorted(natural_root.rglob("model-call-states.jsonl")):
        ingest(state_path, {"policy_version": "baseline-v1"}, natural=True)

    for request_path in sorted(branch_root.rglob("branch_request.json")):
        metadata = _read_json(request_path)
        run_root = request_path.parent
        state_paths = sorted((run_root / "jobs").rglob("model-call-states.jsonl"))
        if len(state_paths) != 1:
            raise ValueError(f"expected one branch trajectory under {run_root}, found {len(state_paths)}")
        ingest(state_paths[0], metadata, natural=False)

    # Same state + rendered content is one effective action. Keep the first
    # executed record and save the other nominal ratios as aliases.
    effective: dict[tuple[str, str, str], dict[str, Any]] = {}
    deduplicated: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for row in transitions:
        key = (str(row["fork_id"]), str(row["state_id"]), str(row["injected_content_sha256"]))
        if key in effective:
            aliases.append({"kept_trajectory_id": effective[key]["trajectory_id"], "alias_trajectory_id": row["trajectory_id"]})
        else:
            effective[key] = row
            deduplicated.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "state-prefixes.jsonl", all_prefixes.values())
    _write_jsonl(output_dir / "default-labels.jsonl", labels)
    _write_jsonl(output_dir / "transitions.jsonl", deduplicated)
    _write_jsonl(output_dir / "equivalent-action-aliases.jsonl", aliases)
    manifest = {
        "schema_version": 1,
        "benchmark": "RoadmapBench",
        "split_manifest_sha256": split_manifest.get("manifest_sha256"),
        "action_table": list(actions),
        "action_table_version": "budget-ratios-v1",
        "allocator_version": "complete-render-v2",
        "tokenizer_version": "tdai-estimator-v2-complete-render",
        "cost_coefficient": cost_coefficient,
        "cost_normalizer_usd": cost_normalizer_usd,
        "prices_per_million": prices.__dict__,
        "trajectories": trajectories,
        "unique_states": len(all_prefixes),
        "default_labels": len(labels),
        "transitions": len(deduplicated),
        "training_eligible_transitions": sum(bool(row["training_eligible"]) for row in deduplicated),
        "truncated_transitions": sum(bool(row["truncated"]) for row in deduplicated),
    }
    manifest["dataset_sha256"] = _fingerprint({
        "manifest": manifest,
        "states": list(all_prefixes),
        "transitions": [_fingerprint(row) for row in deduplicated],
    })
    _write_json(output_dir / "dataset-manifest.json", manifest)
    return manifest
