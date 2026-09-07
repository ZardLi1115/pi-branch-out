from __future__ import annotations

import json
from pathlib import Path

from pi_branch_out.longmemeval_training import export_longmemeval_training, split_for_question


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_longmemeval_export_preserves_paired_actions_and_aliases(tmp_path: Path) -> None:
    root = tmp_path / "collection"
    item = root / "items" / "q1"
    item.mkdir(parents=True)
    _json(root / "dataset-manifest.json", {
        "schema_version": "collection-v1",
        "action_ratios": [0, 0.5, 1],
        "allocator_version": "allocator",
        "tdai_version": "v2.0.0-beta.1",
        "judge_model": "judge",
        "judge_protocol": "protocol",
    })
    _json(item / "complete.json", {"status": "complete"})
    _json(item / "state.json", {"state_id": "s1", "question_id": "q1", "query": "where?"})
    (item / "samples.jsonl").write_text("\n".join([
        json.dumps({
            "effective_action_id": "sha256:empty", "action": 0,
            "action_aliases": [0, 0.5], "reward": 0,
            "injected_tokens": 0,
            "answer_usage": {"input_tokens": 100, "cache_read_tokens": 80, "output_tokens": 10},
        }),
        json.dumps({
            "effective_action_id": "sha256:full", "action": 1,
            "action_aliases": [1], "reward": 1,
            "injected_tokens": 40,
            "answer_usage": {"input_tokens": 150, "cache_read_tokens": 100, "output_tokens": 10},
        }),
    ]) + "\n", encoding="utf-8")
    output = tmp_path / "training"
    manifest = export_longmemeval_training(root, output, cost_coefficient=0.1, cost_normalizer_tokens=100)
    transitions = [json.loads(line) for line in (output / "transitions.jsonl").read_text().splitlines()]
    aliases = [json.loads(line) for line in (output / "equivalent-action-aliases.jsonl").read_text().splitlines()]
    labels = [json.loads(line) for line in (output / "default-labels.jsonl").read_text().splitlines()]
    assert len(transitions) == 3
    assert transitions[0]["sample_weight"] == 0.5
    assert transitions[1]["observation_reused"] is True
    assert transitions[2]["reward"] == 0.94
    assert aliases[0]["alias_action"] == 0.5
    assert labels[0]["default_action"] == 1.0
    assert labels[0]["allocator_content_match"] is True
    assert manifest["unique_states"] == 1


def test_longmemeval_export_can_use_injected_l1_cost_and_stable_subset(tmp_path: Path) -> None:
    root = tmp_path / "collection"
    for question_id, injected_tokens in (("q1", 25), ("q2", 75)):
        item = root / "items" / question_id
        item.mkdir(parents=True)
        _json(item / "complete.json", {"status": "complete"})
        _json(item / "state.json", {"state_id": f"s-{question_id}", "question_id": question_id})
        (item / "samples.jsonl").write_text(json.dumps({
            "effective_action_id": f"sha256:{question_id}",
            "action": 1,
            "action_aliases": [1],
            "reward": 1,
            "injected_tokens": injected_tokens,
            "answer_usage": {"input_tokens": 1000, "output_tokens": 100},
        }) + "\n", encoding="utf-8")
    _json(root / "dataset-manifest.json", {
        "schema_version": "collection-v1",
        "action_ratios": [1],
    })
    selection = tmp_path / "selection.json"
    _json(selection, {"question_ids": ["q2", "q1"]})

    output = tmp_path / "training"
    manifest = export_longmemeval_training(
        root,
        output,
        cost_coefficient=0.5,
        cost_normalizer_tokens=100,
        cost_measure="injected-l1-tokens",
        question_ids_file=selection,
        limit=1,
    )
    transitions = [json.loads(line) for line in (output / "transitions.jsonl").read_text().splitlines()]
    assert len(transitions) == 1
    assert transitions[0]["task_id"] == "q2"
    assert transitions[0]["normalized_cost"] == 0.75
    assert transitions[0]["reward"] == 0.625
    assert manifest["unique_states"] == 1
    assert manifest["cost_measure"] == "injected-l1-tokens"
    assert manifest["question_ids_limit"] == 1


def test_longmemeval_split_is_stable() -> None:
    assert split_for_question("abc") == split_for_question("abc")
