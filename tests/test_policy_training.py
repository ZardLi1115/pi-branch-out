from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_branch_out.policy_training import train_policy


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cql_trains_only_real_transitions_and_skips_all_zero_pretrain(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    state = {
        "context_tokens": 100,
        "context_window_tokens": 1000,
        "candidate_memory_tokens": 50,
        "candidate_count": 2,
        "l1_count": 1,
        "l0_count": 1,
        "query": "fix parser",
        "recent_tool_result": "test failed",
    }
    _jsonl(dataset / "state-prefixes.jsonl", [{"state_id": "s1", "task_id": "task", "split": "train", "state": state}])
    _jsonl(dataset / "default-labels.jsonl", [{"state_id": "s1", "default_action": 0, "allocator_content_match": True}])
    _jsonl(dataset / "transitions.jsonl", [{
        "state_id": "s1", "next_state_id": None, "action": 0, "reward": 1,
        "done": True, "truncated": False, "split": "train", "training_eligible": True,
    }])
    (dataset / "dataset-manifest.json").write_text(json.dumps({
        "action_table": [0, 0.5, 1], "dataset_sha256": "dataset",
        "action_table_version": "actions", "allocator_version": "allocator",
        "tokenizer_version": "tokens",
    }), encoding="utf-8")
    result = train_policy(dataset, tmp_path / "policy", cql_epochs=2, batch_size=1)
    assert result["pretrain_status"] == "skipped-all-default-zero"
    assert result["training_transitions"] == 1
    assert result["training_weight_sum"] == 1.0
    assert (tmp_path / "policy" / "policy.json").is_file()


def test_visible_candidate_text_changes_policy_features() -> None:
    from pi_branch_out.policy_training import POSITIONAL_FEATURE_VERSION, state_features

    base = {"query": "where did I go?", "l1_contents": ["Cafe A"]}
    changed = {"query": "where did I go?", "l1_contents": ["Cafe B"]}
    assert state_features(base) != state_features(changed)
    assert len(state_features(base)) == 158
    positional = state_features(base, feature_version=POSITIONAL_FEATURE_VERSION)
    assert len(positional) == 356


def test_positional_features_preserve_l1_order() -> None:
    from pi_branch_out.policy_training import POSITIONAL_FEATURE_VERSION, state_features

    first = {"query": "where?", "l1_contents": ["Cafe A", "Cafe B"], "l1_lengths": [6, 6]}
    swapped = {"query": "where?", "l1_contents": ["Cafe B", "Cafe A"], "l1_lengths": [6, 6]}
    assert state_features(first, feature_version=POSITIONAL_FEATURE_VERSION) != state_features(
        swapped, feature_version=POSITIONAL_FEATURE_VERSION
    )


def test_policy_can_select_best_epoch_on_dev(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    states = []
    transitions = []
    labels = []
    for split, state_id in (("train", "train"), ("dev", "dev")):
        states.append({"state_id": state_id, "task_id": state_id, "split": split, "state": {"query": state_id}})
        labels.append({"state_id": state_id, "default_action": 1, "allocator_content_match": True, "split": split})
        for action, reward in ((0, 0), (1, 1)):
            transitions.append({
                "state_id": state_id, "next_state_id": None, "action": action,
                "reward": reward, "quality_reward": reward, "injected_tokens": action * 10,
                "done": True, "split": split, "training_eligible": True,
            })
    _jsonl(dataset / "state-prefixes.jsonl", states)
    _jsonl(dataset / "default-labels.jsonl", labels)
    _jsonl(dataset / "transitions.jsonl", transitions)
    (dataset / "dataset-manifest.json").write_text(json.dumps({
        "action_table": [0, 1], "dataset_sha256": "dataset",
        "action_table_version": "actions", "allocator_version": "allocator",
        "tokenizer_version": "tokens",
    }), encoding="utf-8")

    result = train_policy(
        dataset, tmp_path / "policy", cql_epochs=3, batch_size=2,
        cql_alpha=0, select_best_dev=True,
    )
    assert result["dev_states"] == 1
    assert 1 <= result["selected_epoch"] <= 3
    assert result["selection_metric"] == "dev-policy-mean-reward-then-min-injected-tokens"
    assert result["best_dev_policy_reward"] is not None
