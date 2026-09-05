from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "collect-roadmapbench.py"
SPEC = importlib.util.spec_from_file_location("collect_roadmapbench", SCRIPT)
assert SPEC and SPEC.loader
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def test_indexes_are_rebuilt_from_committed_statuses(tmp_path: Path) -> None:
    output = tmp_path / "batch"
    committed = output / "status" / "task-a" / "status.json"
    failed = output / "status" / "task-b" / "status.json"
    collector.atomic_write_json(
        committed,
        {"state": "committed", "summary": {"task_name": "task-a", "reward": 1.0}},
    )
    collector.atomic_write_json(
        failed,
        {"state": "failed", "summary": {"task_name": "task-b", "reward": 0.0}},
    )

    summaries = collector.rebuild_indexes(output, output / "completed.txt")

    assert summaries == [{"task_name": "task-a", "reward": 1.0}]
    assert (output / "completed.txt").read_text(encoding="utf-8") == "task-a\n"
    rows = (output / "natural-summary.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(row)["task_name"] for row in rows] == ["task-a"]


def test_valid_trial_does_not_hide_new_failure_with_old_success(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    older = jobs / "001" / "demo__old" / "result.json"
    newer = jobs / "002" / "demo__new" / "result.json"
    collector.atomic_write_json(older, {"accepted": True})
    collector.atomic_write_json(newer, {"accepted": False})
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    def validate(path: Path, value: dict) -> str:
        if not value["accepted"]:
            raise RuntimeError("rejected")
        return path.parent.name

    try:
        collector.valid_trial(jobs, "demo", validate)
    except RuntimeError as error:
        assert str(error) == "rejected"
    else:
        raise AssertionError("new failed trial must remain authoritative")


def test_total_usage_includes_all_billing_token_classes() -> None:
    totals = collector.total_usage(
        [{
            "model_calls": 2,
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_read_tokens": 20,
            "cache_write_tokens": 4,
        }]
    )
    assert totals["model_calls"] == 2
    assert totals["token_units"] == 37


def test_observed_usage_counts_failed_and_uncommitted_trials(tmp_path: Path) -> None:
    usage = (
        tmp_path / "tasks" / "demo" / "run" / "demo__trial" / "agent"
        / "model-call-checkpoints" / "model-call-usage.jsonl"
    )
    usage.parent.mkdir(parents=True)
    usage.write_text(
        json.dumps({
            "model_call_index": 1,
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_tokens": 30,
            "cache_write_tokens": 4,
        }) + "\n",
        encoding="utf-8",
    )

    totals = collector.observed_usage(tmp_path)

    assert totals["model_calls"] == 1
    assert totals["token_units"] == 46


def test_docker_daemon_preflight_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        collector.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 1, "stdout": ""})(),
    )
    assert collector.docker_daemon_ready() is False
