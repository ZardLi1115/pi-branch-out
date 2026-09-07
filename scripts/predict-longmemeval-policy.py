from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pi_branch_out.policy_training import state_features


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def predict(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    feature_version = str(policy["feature_version"])
    x = np.asarray(state_features(state, feature_version=feature_version), dtype="float32")
    w1 = np.asarray(policy["w1"], dtype="float32")
    b1 = np.asarray(policy["b1"], dtype="float32")
    w2 = np.asarray(policy["w2"], dtype="float32")
    b2 = np.asarray(policy["b2"], dtype="float32")
    if x.shape != (w1.shape[0],):
        raise ValueError(f"feature dimension mismatch: state={x.shape[0]} policy={w1.shape[0]}")
    q_values = np.maximum(0, x @ w1 + b1) @ w2 + b2
    actions = [float(value) for value in policy["actions"]]
    selected_index = int(np.argmax(q_values))
    return {
        "feature_version": feature_version,
        "selected_action": actions[selected_index],
        "selected_index": selected_index,
        "actions": actions,
        "q_values": [float(value) for value in q_values],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one offline LongMemEval MLP budget decision")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    state_payload = read_json(args.state)
    state = state_payload.get("state", state_payload)
    if not isinstance(state, dict):
        raise ValueError("state payload must be an object or contain an object-valued state field")
    result = predict(read_json(args.policy), state)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
