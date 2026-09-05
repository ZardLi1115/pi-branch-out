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
    assert (tmp_path / "policy" / "policy.json").is_file()
