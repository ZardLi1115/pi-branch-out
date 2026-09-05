from __future__ import annotations

import json
from pathlib import Path

from pi_branch_out.training_data import PriceTable, build_roadmapbench_split, export_training_data


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_roadmapbench_split_is_stable_and_task_level(tmp_path: Path) -> None:
    overview = tmp_path / "tasks_overview.jsonl"
    _jsonl(
        overview,
        [
            {"task_id": f"task-{index}", "language": "python", "domain": "library"}
            for index in range(12)
        ],
    )
    first = build_roadmapbench_split(overview, tmp_path / "split-a.json", seed="fixed")
    second = build_roadmapbench_split(overview, tmp_path / "split-b.json", seed="fixed")
    assert first["assignments"] == second["assignments"]
    assert set(first["assignments"].values()) == {"train", "dev", "test"}


def test_export_keeps_labels_and_real_transitions_separate(tmp_path: Path) -> None:
    natural = tmp_path / "natural"
    branch = tmp_path / "branches"
    output = tmp_path / "dataset"
    states = natural / "trial" / "agent" / "model-call-checkpoints" / "model-call-states.jsonl"
    base = {
        "task": "task-1",
        "context_tokens": 100,
        "context_window_tokens": 1000,
        "candidate_memory_tokens": 20,
        "default_actual_memory_tokens": 0,
        "default_mapped_action": 0,
        "actual_injected_content_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    _jsonl(states, [{**base, "model_call_index": 1}, {**base, "model_call_index": 2}])
    _jsonl(
        states.with_name("model-call-usage.jsonl"),
        [
            {"model_call_index": 1, "input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 0, "cache_write_tokens": 0, "usage_schema": "pi-exclusive-input-cache-v1"},
            {"model_call_index": 2, "input_tokens": 110, "output_tokens": 10, "cache_read_tokens": 0, "cache_write_tokens": 0, "usage_schema": "pi-exclusive-input-cache-v1"},
        ],
    )
    result = {"verifier_result": {"rewards": {"reward": 1.0}}}
    (natural / "trial" / "result.json").write_text(json.dumps(result), encoding="utf-8")
    initial_score = natural / "initial-scores" / "task-1.json"
    initial_score.parent.mkdir(parents=True)
    initial_score.write_text(json.dumps({
        "reward": 0.0, "official_verifier": True, "isolated_copy": True,
    }), encoding="utf-8")
    branch.mkdir()
    split = {
        "assignments": {"task-1": "train"},
        "manifest_sha256": "split",
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")

    manifest = export_training_data(
        natural,
        branch,
        output,
        split_path,
        prices=PriceTable(1.0, 2.0, 0.1, 1.0),
        cost_coefficient=0.5,
    )
    labels = [json.loads(line) for line in (output / "default-labels.jsonl").read_text(encoding="utf-8").splitlines()]
    transitions = [json.loads(line) for line in (output / "transitions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(labels) == 2
    assert len(transitions) == 2
    assert sum(row["quality_delta"] for row in transitions) == 1.0
    assert all(row["training_eligible"] for row in transitions)
    assert manifest["default_labels"] == 2
    assert manifest["transitions"] == 2
