from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HASH_DIM = 128
NUMERIC_KEYS = (
    "context_tokens", "context_window_tokens", "reserve_tokens",
    "remaining_call_budget", "remaining_cost_budget_usd", "remaining_time_seconds",
    "candidate_memory_tokens", "candidate_count", "l1_count", "l0_count",
    "default_actual_memory_tokens", "previous_actual_memory_tokens",
    "previous_mapped_action", "previous_budget_tokens",
)
FEATURE_VERSION = "visible-state-hash-v3-history"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _text_features(text: str) -> list[float]:
    result = [0.0] * HASH_DIM
    tokens = re.findall(r"[A-Za-z0-9_-]+|[^\x00-\x7f]", text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % HASH_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        result[index] += sign
    norm = math.sqrt(sum(value * value for value in result)) or 1.0
    return [value / norm for value in result]


def state_features(state: dict[str, Any]) -> list[float]:
    numeric: list[float] = []
    for key in NUMERIC_KEYS:
        raw = state.get(key)
        value = float(raw) if isinstance(raw, (int, float)) and math.isfinite(float(raw)) else 0.0
        numeric.append(math.copysign(math.log1p(abs(value)), value))
    for key in ("l1_lengths", "l0_lengths", "l1_scores", "l0_scores"):
        values = [float(item) for item in (state.get(key) or []) if isinstance(item, (int, float))]
        if values:
            numeric.extend((len(values), sum(values) / len(values), min(values), max(values)))
        else:
            numeric.extend((0.0, 0.0, 0.0, 0.0))
    text = f"{state.get('query', '')}\n{state.get('recent_tool_result', '')}"
    return numeric + _text_features(text)


@dataclass
class Adam:
    params: list[Any]
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        import numpy as np
        self.step = 0
        self.m = [np.zeros_like(param) for param in self.params]
        self.v = [np.zeros_like(param) for param in self.params]

    def update(self, grads: list[Any]) -> None:
        import numpy as np
        self.step += 1
        for index, (param, grad) in enumerate(zip(self.params, grads, strict=True)):
            self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * grad
            self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * np.square(grad)
            m_hat = self.m[index] / (1 - self.beta1**self.step)
            v_hat = self.v[index] / (1 - self.beta2**self.step)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int) -> None:
        import numpy as np
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, math.sqrt(2 / input_dim), (input_dim, hidden_dim)).astype("float32")
        self.b1 = np.zeros(hidden_dim, dtype="float32")
        self.w2 = rng.normal(0, math.sqrt(2 / hidden_dim), (hidden_dim, output_dim)).astype("float32")
        self.b2 = np.zeros(output_dim, dtype="float32")

    @property
    def params(self) -> list[Any]:
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, x: Any) -> tuple[Any, Any]:
        import numpy as np
        hidden = np.maximum(0, x @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2, hidden

    def gradients(self, x: Any, hidden: Any, grad_output: Any) -> list[Any]:
        grad_w2 = hidden.T @ grad_output
        grad_b2 = grad_output.sum(axis=0)
        grad_hidden = grad_output @ self.w2.T
        grad_hidden[hidden <= 0] = 0
        return [x.T @ grad_hidden, grad_hidden.sum(axis=0), grad_w2, grad_b2]

    def copy_from(self, other: "MLP") -> None:
        for target, source in zip(self.params, other.params, strict=True):
            target[...] = source


def _softmax(logits: Any) -> Any:
    import numpy as np
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _action_index(value: float, actions: tuple[float, ...]) -> int:
    distances = [abs(value - action) for action in actions]
    index = min(range(len(actions)), key=distances.__getitem__)
    if distances[index] > 1e-9:
        raise ValueError(f"action {value} is absent from fixed action table")
    return index


def _batch_indices(size: int, batch_size: int, rng: Any) -> list[Any]:
    order = rng.permutation(size)
    return [order[start:start + batch_size] for start in range(0, size, batch_size)]


def train_policy(
    dataset_dir: Path,
    output_dir: Path,
    *,
    hidden_dim: int = 64,
    seed: int = 7,
    pretrain_epochs: int = 20,
    cql_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    cql_alpha: float = 1.0,
    gamma: float = 1.0,
) -> dict[str, Any]:
    if hidden_dim <= 0 or batch_size <= 0 or pretrain_epochs < 0 or cql_epochs <= 0:
        raise ValueError("hidden_dim, batch_size and cql_epochs must be positive; pretrain_epochs may be zero")
    if learning_rate <= 0 or cql_alpha < 0 or not 0 <= gamma <= 1:
        raise ValueError("learning_rate must be positive, cql_alpha non-negative, and gamma within [0, 1]")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("training requires the 'training' extra: pip install -e .[training]") from exc

    manifest = _read_json(dataset_dir / "dataset-manifest.json")
    actions = tuple(float(value) for value in manifest["action_table"])
    prefixes = {row["state_id"]: row for row in _read_jsonl(dataset_dir / "state-prefixes.jsonl")}
    labels = [
        row for row in _read_jsonl(dataset_dir / "default-labels.jsonl")
        if row.get("allocator_content_match") and row.get("split") == "train"
    ]
    transitions = [
        row for row in _read_jsonl(dataset_dir / "transitions.jsonl")
        if row.get("training_eligible") and row.get("split") == "train"
    ]
    if not transitions:
        raise ValueError("no training-eligible real-action transitions")

    first = next(iter(prefixes.values()))["state"]
    input_dim = len(state_features(first))
    online = MLP(input_dim, hidden_dim, len(actions), seed)
    target = MLP(input_dim, hidden_dim, len(actions), seed + 1)
    target.copy_from(online)
    optimizer = Adam(online.params, learning_rate)
    rng = np.random.default_rng(seed)

    distinct_labels = {float(row["default_action"]) for row in labels}
    pretrain_status = "skipped-all-default-zero" if distinct_labels in (set(), {0.0}) else "trained"
    if pretrain_status == "skipped-all-default-zero":
        online.b2[_action_index(0.0, actions)] = 1.0
        target.copy_from(online)
    if pretrain_status == "trained":
        x = np.asarray([state_features(prefixes[row["state_id"]]["state"]) for row in labels], dtype="float32")
        y = np.asarray([_action_index(float(row["default_action"]), actions) for row in labels], dtype="int64")
        for _ in range(pretrain_epochs):
            for indices in _batch_indices(len(x), batch_size, rng):
                logits, hidden = online.forward(x[indices])
                probabilities = _softmax(logits)
                grad = probabilities
                grad[np.arange(len(indices)), y[indices]] -= 1
                grad /= len(indices)
                optimizer.update(online.gradients(x[indices], hidden, grad))
        target.copy_from(online)

    x = np.asarray([state_features(prefixes[row["state_id"]]["state"]) for row in transitions], dtype="float32")
    next_x = np.asarray([
        state_features(prefixes[row["next_state_id"]]["state"]) if row.get("next_state_id") else [0.0] * input_dim
        for row in transitions
    ], dtype="float32")
    action_index = np.asarray([_action_index(float(row["action"]), actions) for row in transitions], dtype="int64")
    rewards = np.asarray([float(row["reward"]) for row in transitions], dtype="float32")
    terminal = np.asarray([bool(row.get("done")) for row in transitions], dtype="float32")
    losses: list[float] = []

    for epoch in range(cql_epochs):
        for indices in _batch_indices(len(x), batch_size, rng):
            q, hidden = online.forward(x[indices])
            next_q, _ = target.forward(next_x[indices])
            targets = rewards[indices] + gamma * (1 - terminal[indices]) * next_q.max(axis=1)
            chosen = q[np.arange(len(indices)), action_index[indices]]
            td = chosen - targets
            probabilities = _softmax(q)
            grad = cql_alpha * probabilities / len(indices)
            grad[np.arange(len(indices)), action_index[indices]] -= cql_alpha / len(indices)
            grad[np.arange(len(indices)), action_index[indices]] += 2 * td / len(indices)
            optimizer.update(online.gradients(x[indices], hidden, grad))
            conservative = np.log(np.exp(q - q.max(axis=1, keepdims=True)).sum(axis=1)) + q.max(axis=1) - chosen
            losses.append(float(np.mean(np.square(td)) + cql_alpha * np.mean(conservative)))
        if (epoch + 1) % 10 == 0:
            target.copy_from(online)

    output_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        "schema_version": 1,
        "model_type": "one-hidden-layer-relu-q-network",
        "feature_version": FEATURE_VERSION,
        "numeric_keys": list(NUMERIC_KEYS),
        "hash_dim": HASH_DIM,
        "actions": list(actions),
        "w1": online.w1.tolist(), "b1": online.b1.tolist(),
        "w2": online.w2.tolist(), "b2": online.b2.tolist(),
    }
    weights_text = json.dumps(weights, ensure_ascii=False, separators=(",", ":")) + "\n"
    (output_dir / "policy.json").write_text(weights_text, encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "policy_version": f"cql-{hashlib.sha256(weights_text.encode('utf-8')).hexdigest()[:12]}",
        "policy_sha256": hashlib.sha256(weights_text.encode("utf-8")).hexdigest(),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "action_table_version": manifest.get("action_table_version"),
        "allocator_version": manifest.get("allocator_version"),
        "tokenizer_version": manifest.get("tokenizer_version"),
        "feature_version": FEATURE_VERSION,
        "pretrain_status": pretrain_status,
        "training_transitions": len(transitions),
        "hyperparameters": {
            "hidden_dim": hidden_dim, "seed": seed, "pretrain_epochs": pretrain_epochs,
            "cql_epochs": cql_epochs, "batch_size": batch_size, "learning_rate": learning_rate,
            "cql_alpha": cql_alpha, "gamma": gamma,
        },
        "final_mean_loss": sum(losses[-max(1, min(len(losses), 20)):]) / max(1, min(len(losses), 20)),
    }
    (output_dir / "policy-manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result
