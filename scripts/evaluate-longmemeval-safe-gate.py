from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pi_branch_out.policy_training import state_features


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def policy_probabilities(policy: dict[str, Any], state: dict[str, Any]) -> np.ndarray:
    x = np.asarray(state_features(state, feature_version=str(policy["feature_version"])), dtype="float32")
    w1 = np.asarray(policy["w1"], dtype="float32")
    b1 = np.asarray(policy["b1"], dtype="float32")
    w2 = np.asarray(policy["w2"], dtype="float32")
    b2 = np.asarray(policy["b2"], dtype="float32")
    q = np.maximum(0, x @ w1 + b1) @ w2 + b2
    shifted = q - q.max()
    values = np.exp(shifted)
    return values / values.sum()


def parse_group(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("policy group must be NAME=GLOB")
    name, pattern = value.split("=", 1)
    if not name or not pattern:
        raise argparse.ArgumentTypeError("policy group must be NAME=GLOB")
    return name, pattern


def expand_group(pattern: str) -> list[Path]:
    path = Path(pattern)
    parent = path.parent if path.parent != Path("") else Path(".")
    matches = sorted(candidate for candidate in parent.glob(path.name) if (candidate / "policy.json").is_file())
    if not matches:
        raise ValueError(f"policy group matched no policies: {pattern}")
    return matches


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    baseline_quality = sum(row["baseline_quality"] for row in rows) / count if count else 0.0
    policy_quality = sum(row["policy_quality"] for row in rows) / count if count else 0.0
    baseline_tokens = sum(row["baseline_tokens"] for row in rows) / count if count else 0.0
    policy_tokens = sum(row["policy_tokens"] for row in rows) / count if count else 0.0
    both_correct = sum(row["baseline_quality"] >= 1 and row["policy_quality"] >= 1 for row in rows)
    policy_only = sum(row["baseline_quality"] < 1 and row["policy_quality"] >= 1 for row in rows)
    fixed1_only = sum(row["baseline_quality"] >= 1 and row["policy_quality"] < 1 for row in rows)
    return {
        "states": count,
        "baseline_quality": baseline_quality,
        "policy_quality": policy_quality,
        "quality_delta": policy_quality - baseline_quality,
        "baseline_mean_injected_tokens": baseline_tokens,
        "policy_mean_injected_tokens": policy_tokens,
        "mean_token_saving": baseline_tokens - policy_tokens,
        "token_saving_fraction": (baseline_tokens - policy_tokens) / baseline_tokens if baseline_tokens else 0.0,
        "replacement_coverage": sum(row["action"] < 1.0 for row in rows) / count if count else 0.0,
        "both_correct": both_correct,
        "policy_only_correct": policy_only,
        "fixed1_only_correct": fixed1_only,
        "both_wrong": count - both_correct - policy_only - fixed1_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate an ensemble gate that falls back to fixed top-5")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--policy-group", action="append", type=parse_group, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.dataset_dir / "dataset-manifest.json")
    actions = [float(value) for value in manifest["action_table"]]
    default_index = actions.index(1.0)
    states = {row["state_id"]: row for row in read_jsonl(args.dataset_dir / "state-prefixes.jsonl")}
    transitions: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(args.dataset_dir / "transitions.jsonl"):
        transitions[str(row["state_id"])][float(row["action"])] = row

    groups: dict[str, list[dict[str, Any]]] = {}
    group_sources: dict[str, list[str]] = {}
    for name, pattern in args.policy_group:
        paths = expand_group(pattern)
        policies = []
        for path in paths:
            policy_manifest = read_json(path / "policy-manifest.json")
            if policy_manifest.get("dataset_sha256") != manifest.get("dataset_sha256"):
                raise ValueError(f"dataset fingerprint mismatch: {path}")
            policies.append(read_json(path / "policy.json"))
        groups[name] = policies
        group_sources[name] = [str(path.resolve()) for path in paths]

    def state_risks(policies: list[dict[str, Any]], state: dict[str, Any]) -> list[float]:
        probabilities = [policy_probabilities(policy, state) for policy in policies]
        return [
            max(float(values[default_index] - values[index]) for values in probabilities)
            for index in range(len(actions))
        ]

    def choose_rows(
        policies: list[dict[str, Any]], state_rows: list[dict[str, Any]], threshold: float | None
    ) -> list[dict[str, Any]]:
        result = []
        for state_row in state_rows:
            available = transitions[state_row["state_id"]]
            risks = state_risks(policies, state_row["state"])
            candidates = [default_index]
            if threshold is not None:
                candidates.extend(index for index in range(default_index) if risks[index] <= threshold)
            selected = min(candidates, key=lambda index: (float(available[actions[index]]["injected_tokens"]), actions[index]))
            observed = available[actions[selected]]
            baseline = available[1.0]
            result.append({
                "question_id": state_row["task_id"],
                "state_id": state_row["state_id"],
                "action": actions[selected],
                "risk_score": risks[selected],
                "policy_quality": float(observed["quality_reward"]),
                "baseline_quality": float(baseline["quality_reward"]),
                "policy_tokens": float(observed["injected_tokens"]),
                "baseline_tokens": float(baseline["injected_tokens"]),
            })
        return result

    dev_states = [row for row in states.values() if row["split"] == "dev"]
    test_states = [row for row in states.values() if row["split"] == "test"]
    calibration: dict[str, Any] = {}
    feasible: list[tuple[float, float, str, float | None, list[dict[str, Any]], dict[str, Any]]] = []
    for name, policies in groups.items():
        risks = sorted({
            risk
            for state_row in dev_states
            for risk in state_risks(policies, state_row["state"])[:default_index]
        })
        candidates: list[dict[str, Any]] = []
        for threshold in [None, *risks]:
            rows = choose_rows(policies, dev_states, threshold)
            summary = metrics(rows)
            safe = summary["fixed1_only_correct"] == 0
            candidates.append({"threshold": threshold, "safe_on_dev": safe, **summary})
            if safe:
                feasible.append((
                    summary["policy_mean_injected_tokens"],
                    -summary["policy_quality"],
                    name,
                    threshold,
                    rows,
                    summary,
                ))
        calibration[name] = {
            "policies": group_sources[name],
            "candidate_count": len(candidates),
            "safe_candidate_count": sum(bool(row["safe_on_dev"]) for row in candidates),
            "best_safe_candidate": min(
                (row for row in candidates if row["safe_on_dev"]),
                key=lambda row: (row["policy_mean_injected_tokens"], -row["policy_quality"]),
            ),
        }
    if not feasible:
        raise ValueError("no dev-safe gate candidate; fixed1 fallback should always be feasible")
    _, _, selected_name, selected_threshold, selected_dev_rows, selected_dev_metrics = min(feasible)
    selected_test_rows = choose_rows(groups[selected_name], test_states, selected_threshold)
    output = {
        "schema_version": 1,
        "dataset_sha256": manifest.get("dataset_sha256"),
        "selection_rule": "minimum-dev-tokens-with-zero-fixed1-only-errors",
        "risk_score": "max-ensemble-softmax-default-minus-action",
        "selected_group": selected_name,
        "selected_threshold": selected_threshold,
        "calibration": calibration,
        "dev": selected_dev_metrics,
        "test": metrics(selected_test_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    pairs_path = args.output.with_name(f"{args.output.stem}-pairs.jsonl")
    pairs_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected_test_rows),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
