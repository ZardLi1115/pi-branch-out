from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .training_data import PriceTable


def _reward(result_path: Path) -> float | None:
    value = json.loads(result_path.read_text(encoding="utf-8"))
    rewards = (value.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward)
    fallback = result_path.parent / "verifier" / "reward.json"
    if fallback.is_file():
        reward = json.loads(fallback.read_text(encoding="utf-8")).get("reward")
        if isinstance(reward, (int, float)):
            return float(reward)
    return None


def _task_id(result_path: Path) -> str:
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if value.get("task_name"):
        return str(value["task_name"])
    name = result_path.parent.name
    return name.split("__", 1)[0]


def _cost_and_latency(trial_dir: Path, prices: PriceTable) -> tuple[float | None, list[float]]:
    usage_files = list(trial_dir.rglob("model-call-usage.jsonl"))
    if not usage_files:
        return None, []
    total = 0.0
    for path in usage_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("usage_schema") != "pi-exclusive-input-cache-v1":
                    return None, []
                total += prices.cost(row)
    latencies: list[float] = []
    for path in trial_dir.rglob("policy-observations.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line).get("latency_ms")
                if isinstance(value, (int, float)):
                    latencies.append(float(value))
    return total, latencies


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize_evaluation(
    variants: dict[str, Path],
    output_path: Path,
    *,
    prices: PriceTable,
    quality_tolerance: float,
) -> dict[str, Any]:
    if len(variants) < 2:
        raise ValueError("at least two variants are required")
    by_variant: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, root in variants.items():
        tasks: dict[str, list[dict[str, Any]]] = {}
        for result_path in root.rglob("result.json"):
            reward = _reward(result_path)
            if reward is None:
                continue
            cost, latencies = _cost_and_latency(result_path.parent, prices)
            tasks.setdefault(_task_id(result_path), []).append({
                "reward": reward, "cost_usd": cost, "latencies": latencies,
            })
        by_variant[name] = tasks

    common_tasks = sorted(set.intersection(*(set(tasks) for tasks in by_variant.values())))
    if not common_tasks:
        raise ValueError("variants have no common complete tasks")
    summaries: dict[str, Any] = {}
    baseline_name = next(iter(variants))
    for name, tasks in by_variant.items():
        task_rewards = [sum(run["reward"] for run in tasks[task]) / len(tasks[task]) for task in common_tasks]
        task_costs = [
            sum(run["cost_usd"] for run in tasks[task]) / len(tasks[task])
            for task in common_tasks if all(run["cost_usd"] is not None for run in tasks[task])
        ]
        latencies = [latency for task in common_tasks for run in tasks[task] for latency in run["latencies"]]
        summaries[name] = {
            "tasks": len(common_tasks),
            "mean_completion_score": sum(task_rewards) / len(task_rewards),
            "complete_task_success_rate": sum(score >= 1.0 for score in task_rewards) / len(task_rewards),
            "mean_api_cost_usd": sum(task_costs) / len(task_costs) if len(task_costs) == len(common_tasks) else None,
            "plugin_p95_latency_ms": _percentile(latencies, 0.95),
        }
    baseline_score = summaries[baseline_name]["mean_completion_score"]
    for name, summary in summaries.items():
        summary["quality_within_tolerance"] = summary["mean_completion_score"] >= baseline_score - quality_tolerance
    result = {
        "schema_version": 1,
        "paired_by_complete_task": True,
        "baseline": baseline_name,
        "quality_tolerance": quality_tolerance,
        "common_tasks": common_tasks,
        "variants": summaries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
