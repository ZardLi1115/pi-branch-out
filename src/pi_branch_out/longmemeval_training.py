from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def split_for_question(question_id: str, *, seed: str = "longmemeval-v1") -> str:
    bucket = int.from_bytes(hashlib.sha256(f"{seed}\0{question_id}".encode()).digest()[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def export_longmemeval_training(
    collection_root: Path,
    output_dir: Path,
    *,
    cost_coefficient: float = 0.0,
    cost_normalizer_tokens: float = 10_000.0,
    split_seed: str = "longmemeval-v1",
) -> dict[str, Any]:
    if cost_coefficient < 0:
        raise ValueError("cost_coefficient must be non-negative")
    if cost_normalizer_tokens <= 0:
        raise ValueError("cost_normalizer_tokens must be positive")
    source_manifest = _read_json(collection_root / "dataset-manifest.json")
    actions = [float(value) for value in source_manifest["action_ratios"]]
    states: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []

    for item_dir in sorted((collection_root / "items").iterdir()):
        if not (item_dir / "complete.json").is_file():
            continue
        state_payload = _read_json(item_dir / "state.json")
        state_id = str(state_payload.pop("state_id"))
        question_id = str(state_payload["question_id"])
        split = split_for_question(question_id, seed=split_seed)
        states.append({"state_id": state_id, "task_id": question_id, "split": split, "state": state_payload})
        labels.append({
            "state_id": state_id,
            "task_id": question_id,
            "split": split,
            "default_action": 1.0,
            "allocator_content_match": True,
            "label_source": "beta1-openclaw-full-top5",
            "label_semantics": "behavior-imitation-not-optimal-action",
        })
        for sample in _read_jsonl(item_dir / "samples.jsonl"):
            usage = sample.get("answer_usage") or {}
            billable_tokens = (
                float(usage.get("input_tokens") or 0)
                - float(usage.get("cache_read_tokens") or 0)
                + float(usage.get("output_tokens") or 0)
            )
            normalized_cost = max(0.0, billable_tokens) / cost_normalizer_tokens
            quality_reward = float(sample["reward"])
            reward = quality_reward - cost_coefficient * normalized_cost
            action_aliases = [float(value) for value in sample.get("action_aliases", [sample["action"]])]
            sample_weight = 1.0 / len(action_aliases)
            for action in action_aliases:
                transitions.append({
                    "trajectory_id": f"{question_id}:{sample['effective_action_id']}:{action:g}",
                    "state_id": state_id,
                    "next_state_id": None,
                    "task_id": question_id,
                    "split": split,
                    "action": action,
                    "reward": reward,
                    "quality_reward": quality_reward,
                    "normalized_cost": normalized_cost,
                    "billable_token_proxy": billable_tokens,
                    "usage": usage,
                    "done": True,
                    "truncated": False,
                    "training_eligible": bool(sample.get("training_eligible", True)),
                    "fork_id": question_id,
                    "injected_content_sha256": str(sample["effective_action_id"]).removeprefix("sha256:"),
                    "effective_action_id": sample["effective_action_id"],
                    "sample_weight": sample_weight,
                    "observation_reused": action != float(sample["action"]),
                })
            for alias in action_aliases:
                if alias == float(sample["action"]):
                    continue
                aliases.append({
                    "state_id": state_id,
                    "task_id": question_id,
                    "kept_action": float(sample["action"]),
                    "alias_action": alias,
                    "effective_action_id": sample["effective_action_id"],
                })

    if not transitions:
        raise ValueError("no completed LongMemEval samples found")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "state-prefixes.jsonl", states)
    _write_jsonl(output_dir / "default-labels.jsonl", labels)
    _write_jsonl(output_dir / "transitions.jsonl", transitions)
    _write_jsonl(output_dir / "equivalent-action-aliases.jsonl", aliases)
    split_counts = {split: sum(row["split"] == split for row in states) for split in ("train", "dev", "test")}
    manifest = {
        "schema_version": 1,
        "benchmark": "LongMemEval",
        "source_collection": str(collection_root.resolve()),
        "source_schema_version": source_manifest.get("schema_version"),
        "action_table": actions,
        "action_table_version": "budget-ratios-v1",
        "allocator_version": source_manifest.get("allocator_version"),
        "tokenizer_version": "tdai-estimator-v2-complete-render",
        "tdai_version": source_manifest.get("tdai_version"),
        "cost_coefficient": cost_coefficient,
        "cost_normalizer_tokens": cost_normalizer_tokens,
        "cost_measure": "answer-input-minus-cache-read-plus-output-token-proxy",
        "judge_model": source_manifest.get("judge_model"),
        "judge_protocol": source_manifest.get("judge_protocol"),
        "split_seed": split_seed,
        "split_counts": split_counts,
        "unique_states": len(states),
        "default_labels": len(labels),
        "transitions": len(transitions),
        "training_eligible_transitions": sum(bool(row["training_eligible"]) for row in transitions),
        "effective_observations": sum(not bool(row["observation_reused"]) for row in transitions),
        "training_weight_sum": sum(
            float(row["sample_weight"]) for row in transitions if row["training_eligible"]
        ),
        "equivalent_action_aliases": len(aliases),
    }
    fingerprint_payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    manifest["dataset_sha256"] = hashlib.sha256(fingerprint_payload.encode()).hexdigest()
    _write_json(output_dir / "dataset-manifest.json", manifest)
    return manifest
