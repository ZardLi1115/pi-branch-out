from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "dataset-manifest.json")
    statuses = read_jsonl(root / "status.jsonl")
    latest_status = {str(row.get("question_id")): row for row in statuses}
    nominal_rewards: dict[float, list[float]] = defaultdict(list)
    total_input = total_output = total_cache_read = total_judge_input = 0.0
    completed = diverse = effective_actions = aliases = 0
    l1_counts: list[int] = []
    consistency_errors: list[str] = []
    for item_dir in sorted((root / "items").iterdir()):
        if not (item_dir / "complete.json").is_file():
            continue
        completed += 1
        state_payload = read_json(item_dir / "state.json")
        state_id = state_payload["state_id"]
        plans = read_json(item_dir / "action-plan.json")["plans"]
        samples = read_jsonl(item_dir / "samples.jsonl")
        sample_by_id = {row["effective_action_id"]: row for row in samples}
        l1_counts.append(int(state_payload.get("l1_count") or 0))
        effective_actions += len(samples)
        aliases += len(plans) - len(samples)
        diverse += len(samples) >= 2
        for sample in samples:
            if sample.get("state_id") != state_id:
                consistency_errors.append(f"{item_dir.name}: state_id mismatch")
            usage = sample.get("answer_usage") or {}
            judge_usage = sample.get("judge_usage") or {}
            total_input += float(usage.get("input_tokens") or 0)
            total_output += float(usage.get("output_tokens") or 0)
            total_cache_read += float(usage.get("cache_read_tokens") or 0)
            total_judge_input += float(judge_usage.get("input_tokens") or 0)
        for plan in plans:
            sample = sample_by_id.get(plan["effectiveActionId"])
            if sample is None:
                consistency_errors.append(f"{item_dir.name}: missing effective action {plan['effectiveActionId']}")
                continue
            nominal_rewards[float(plan["action"])].append(float(sample["reward"]))
    return {
        "benchmark": manifest.get("benchmark"),
        "source_file": manifest.get("data_file"),
        "tdai_version": manifest.get("tdai_version"),
        "tdai_prompt_mode": manifest.get("tdai_prompt_mode"),
        "answer_model": manifest.get("answer_model"),
        "answer_prompt_version": manifest.get("answer_prompt_version"),
        "judge_model": manifest.get("judge_model"),
        "status_rows": len(statuses),
        "unique_status_items": len(latest_status),
        "completed_items": completed,
        "failed_items": sum(row.get("status") == "failed" for row in latest_status.values()),
        "diverse_items": diverse,
        "nonempty_l1_items": sum(value > 0 for value in l1_counts),
        "mean_l1_count": sum(l1_counts) / len(l1_counts) if l1_counts else 0.0,
        "effective_actions": effective_actions,
        "equivalent_action_aliases": aliases,
        "nominal_action_accuracy": {
            str(action): sum(values) / len(values) for action, values in sorted(nominal_rewards.items())
        },
        "answer_usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_read_tokens": total_cache_read,
            "cache_rate": total_cache_read / total_input if total_input else 0.0,
        },
        "judge_input_tokens_research_cost": total_judge_input,
        "consistency_errors": consistency_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize a LongMemEval budget collection")
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.collection_root)
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")


if __name__ == "__main__":
    main()
