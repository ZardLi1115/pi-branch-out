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
    x = np.asarray(
        state_features(state, feature_version=str(policy.get("feature_version", "visible-state-hash-v4-memory-text"))),
        dtype="float32",
    )
    w1 = np.asarray(policy["w1"], dtype="float32")
    b1 = np.asarray(policy["b1"], dtype="float32")
    w2 = np.asarray(policy["w2"], dtype="float32")
    b2 = np.asarray(policy["b2"], dtype="float32")
    q = np.maximum(0, x @ w1 + b1) @ w2 + b2
    return float(policy["actions"][int(np.argmax(q))])


def mean(rows: list[float]) -> float:
    return sum(rows) / len(rows) if rows else 0.0


def bootstrap_mean_ci(values: list[float], *, seed: int, resamples: int = 2000) -> dict[str, Any] | None:
    if not values:
        return None
    samples = np.asarray(values, dtype="float64")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(samples), size=(resamples, len(samples)))
    estimates = samples[indices].mean(axis=1)
    return {
        "estimate": float(samples.mean()),
        "low": float(np.percentile(estimates, 2.5)),
        "high": float(np.percentile(estimates, 97.5)),
        "resamples": resamples,
        "seed": seed,
        "paired_by": "question_id",
    }


def token_summary(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [float(row["injected_tokens"]) for row in rows]
    baseline_tokens = [float(row["injected_tokens"]) for row in baseline_rows]
    token_delta = [value - baseline for value, baseline in zip(tokens, baseline_tokens, strict=True)]
    baseline_mean = mean(baseline_tokens)
    saving = baseline_mean - mean(tokens)
    return {
        "questions": len(rows),
        "mean_injected_tokens": mean(tokens),
        "mean_fixed1_tokens": baseline_mean,
        "mean_token_delta": mean(token_delta),
        "mean_token_saving": saving,
        "token_saving_fraction": saving / baseline_mean if baseline_mean > 0 else 0.0,
    }


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

    paired_rows_by_split: dict[str, list[dict[str, Any]]] = {}

    def metrics(state_rows: list[dict[str, Any]], *, bootstrap_seed: int) -> dict[str, Any]:
        policy_rows: list[dict[str, Any]] = []
        fixed: dict[float, list[float]] = defaultdict(list)
        fixed_tokens: dict[float, list[float]] = defaultdict(list)
        baseline_rows: list[dict[str, Any]] = []
        posthoc_rows: list[dict[str, Any]] = []
        paired_rows: list[dict[str, Any]] = []
        for state_row in state_rows:
            state_id = state_row["state_id"]
            available = transitions[state_id]
            if 1.0 not in available:
                raise ValueError(f"state {state_id} has no fixed action 1.0 baseline")
            action = choose_action(policy, state_row["state"])
            if action not in available:
                raise ValueError(f"policy action {action} is absent for state {state_id}")
            observed = available[action]
            baseline = available[1.0]
            eligible = [
                row for row in available.values()
                if float(row["quality_reward"]) >= float(baseline["quality_reward"])
            ]
            posthoc = min(eligible, key=lambda row: (float(row["injected_tokens"]), float(row["action"])))
            policy_rows.append(observed)
            baseline_rows.append(baseline)
            posthoc_rows.append(posthoc)
            for fixed_action, row in available.items():
                fixed[fixed_action].append(float(row["quality_reward"]))
                fixed_tokens[fixed_action].append(float(row["injected_tokens"]))
            paired_rows.append({
                "question_id": state_row["task_id"],
                "state_id": state_id,
                "split": state_row["split"],
                "fixed1": {
                    "quality": float(baseline["quality_reward"]),
                    "injected_tokens": float(baseline["injected_tokens"]),
                },
                "policy": {
                    "action": action,
                    "quality": float(observed["quality_reward"]),
                    "injected_tokens": float(observed["injected_tokens"]),
                },
                "posthoc_min_token": {
                    "action": float(posthoc["action"]),
                    "quality": float(posthoc["quality_reward"]),
                    "injected_tokens": float(posthoc["injected_tokens"]),
                },
            })

        fixed1_correct_indices = [
            index for index, row in enumerate(baseline_rows) if float(row["quality_reward"]) >= 1.0
        ]
        fixed1_correct_posthoc = [posthoc_rows[index] for index in fixed1_correct_indices]
        fixed1_correct_baseline = [baseline_rows[index] for index in fixed1_correct_indices]
        quadrants: dict[str, list[int]] = {
            "both_correct": [], "policy_only_correct": [],
            "fixed1_only_correct": [], "both_wrong": [],
        }
        for index, (observed, baseline) in enumerate(zip(policy_rows, baseline_rows, strict=True)):
            policy_correct = float(observed["quality_reward"]) >= 1.0
            baseline_correct = float(baseline["quality_reward"]) >= 1.0
            key = (
                "both_correct" if policy_correct and baseline_correct else
                "policy_only_correct" if policy_correct else
                "fixed1_only_correct" if baseline_correct else
                "both_wrong"
            )
            quadrants[key].append(index)

        def quadrant_summary(indices: list[int]) -> dict[str, Any]:
            selected_policy = [policy_rows[index] for index in indices]
            selected_baseline = [baseline_rows[index] for index in indices]
            return token_summary(selected_policy, selected_baseline)

        quality_delta = [
            float(observed["quality_reward"]) - float(baseline["quality_reward"])
            for observed, baseline in zip(policy_rows, baseline_rows, strict=True)
        ]
        token_delta = [
            float(observed["injected_tokens"]) - float(baseline["injected_tokens"])
            for observed, baseline in zip(policy_rows, baseline_rows, strict=True)
        ]
        posthoc_token_delta = [
            float(observed["injected_tokens"]) - float(baseline["injected_tokens"])
            for observed, baseline in zip(posthoc_rows, baseline_rows, strict=True)
        ]
        return {
            "states": len(state_rows),
            "policy_quality": mean([float(row["quality_reward"]) for row in policy_rows]),
            "policy_reward": mean([float(row["reward"]) for row in policy_rows]),
            "policy_mean_action": mean([float(row["action"]) for row in policy_rows]),
            "policy_mean_injected_tokens": mean([float(row["injected_tokens"]) for row in policy_rows]),
            "fixed_quality": {str(action): mean(values) for action, values in sorted(fixed.items())},
            "fixed_mean_injected_tokens": {
                str(action): mean(fixed_tokens[action]) for action in sorted(fixed)
            },
            "fixed_actions": {
                str(action): {
                    "questions": len(fixed[action]),
                    "quality": mean(fixed[action]),
                    "mean_injected_tokens": mean(fixed_tokens[action]),
                }
                for action in sorted(fixed)
            },
            "posthoc_min_token_preserving_fixed1": {
                "all": token_summary(posthoc_rows, baseline_rows),
                "fixed1_correct_only": token_summary(fixed1_correct_posthoc, fixed1_correct_baseline),
            },
            "policy_vs_fixed1": {
                key: quadrant_summary(indices) for key, indices in quadrants.items()
            },
            "bootstrap_ci": {
                "quality_delta": bootstrap_mean_ci(quality_delta, seed=bootstrap_seed),
                "token_delta": bootstrap_mean_ci(token_delta, seed=bootstrap_seed + 1),
                "posthoc_token_delta": bootstrap_mean_ci(posthoc_token_delta, seed=bootstrap_seed + 2),
            },
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

    splits: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        state_rows = [row for row in states.values() if row["split"] == split]
        informative = [
            row for row in state_rows
            if len({float(item["quality_reward"]) for item in transitions[row["state_id"]].values()}) > 1
        ]
        split_index = ("train", "dev", "test").index(split)
        splits[split] = {
            "all": metrics(state_rows, bootstrap_seed=700 + split_index * 10),
            "informative": metrics(informative, bootstrap_seed=705 + split_index * 10),
        }
        paired_rows_by_split[split] = [
            {
                "question_id": row["task_id"],
                "state_id": row["state_id"],
                "split": split,
                "available_actions": {
                    str(action): {
                        "quality": float(value["quality_reward"]),
                        "injected_tokens": float(value["injected_tokens"]),
                    }
                    for action, value in sorted(transitions[row["state_id"]].items())
                },
            }
            for row in state_rows
        ]
    return {
        "schema_version": 2,
        "dataset_sha256": manifest.get("dataset_sha256"),
        "policy_version": policy_manifest.get("policy_version"),
        "feature_version": policy.get("feature_version"),
        "splits": splits,
        "paired_rows": paired_rows_by_split,
        "warning": "Small pilot metrics are wiring evidence, not replacement-policy evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline paired-action evaluation for a LongMemEval budget policy")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.dataset_dir, args.policy_dir)
    paired_rows = result.pop("paired_rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    paired_output = args.output.with_name(f"{args.output.stem}-pairs.jsonl")
    paired_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for split in ("train", "dev", "test")
            for row in paired_rows[split]
        ),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
