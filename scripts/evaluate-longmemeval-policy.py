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


def choose_action(policy: dict[str, Any], state: dict[str, Any]) -> float:
    x = np.asarray(state_features(state), dtype="float32")
    w1 = np.asarray(policy["w1"], dtype="float32")
    b1 = np.asarray(policy["b1"], dtype="float32")
    w2 = np.asarray(policy["w2"], dtype="float32")
    b2 = np.asarray(policy["b2"], dtype="float32")
    q = np.maximum(0, x @ w1 + b1) @ w2 + b2
    return float(policy["actions"][int(np.argmax(q))])


def mean(rows: list[float]) -> float:
    return sum(rows) / len(rows) if rows else 0.0


def evaluate(dataset: Path, policy_dir: Path) -> dict[str, Any]:
    manifest = read_json(dataset / "dataset-manifest.json")
    policy_manifest = read_json(policy_dir / "policy-manifest.json")
    policy = read_json(policy_dir / "policy.json")
    if policy_manifest.get("dataset_sha256") != manifest.get("dataset_sha256"):
        raise ValueError("policy was not trained on this dataset fingerprint")
    states = {row["state_id"]: row for row in read_jsonl(dataset / "state-prefixes.jsonl")}
    transitions: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(dataset / "transitions.jsonl"):
        transitions[row["state_id"]][float(row["action"])] = row

    splits: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        state_rows = [row for row in states.values() if row["split"] == split]
        policy_rows: list[dict[str, Any]] = []
        fixed: dict[float, list[float]] = defaultdict(list)
        for state_row in state_rows:
            state_id = state_row["state_id"]
            available = transitions[state_id]
            action = choose_action(policy, state_row["state"])
            observed = available[action]
            policy_rows.append(observed)
            for fixed_action, row in available.items():
                fixed[fixed_action].append(float(row["quality_reward"]))
        splits[split] = {
            "states": len(state_rows),
            "policy_quality": mean([float(row["quality_reward"]) for row in policy_rows]),
            "policy_reward": mean([float(row["reward"]) for row in policy_rows]),
            "policy_mean_action": mean([float(row["action"]) for row in policy_rows]),
            "policy_mean_injected_tokens": mean([float(row["injected_tokens"]) for row in policy_rows]),
            "fixed_quality": {str(action): mean(values) for action, values in sorted(fixed.items())},
            "oracle_quality": mean([
                max(float(row["quality_reward"]) for row in transitions[state_row["state_id"]].values())
                for state_row in state_rows
            ]),
            "oracle_min_action_at_best_quality": mean([
                min(
                    float(row["action"])
                    for row in transitions[state_row["state_id"]].values()
                    if float(row["quality_reward"]) == max(
                        float(candidate["quality_reward"])
                        for candidate in transitions[state_row["state_id"]].values()
                    )
                )
                for state_row in state_rows
            ]),
        }
    return {
        "schema_version": 1,
        "dataset_sha256": manifest.get("dataset_sha256"),
        "policy_version": policy_manifest.get("policy_version"),
        "feature_version": policy.get("feature_version"),
        "splits": splits,
        "warning": "Small pilot metrics are wiring evidence, not replacement-policy evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline paired-action evaluation for a LongMemEval budget policy")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.dataset_dir, args.policy_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
