from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "select-longmemeval.py"
    spec = importlib.util.spec_from_file_location("select_longmemeval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_round_robins_strata_and_is_stable() -> None:
    data = [
        {"question_id": "a1", "question_type": "a"},
        {"question_id": "a2", "question_type": "a"},
        {"question_id": "b1", "question_type": "b"},
        {"question_id": "x_abs", "question_type": "a"},
    ]
    first = _module().build_selection(data, "seed")
    second = _module().build_selection(data, "seed")
    assert first == second
    assert set(first["question_ids"][:3]) == {"a1", "b1", "x_abs"} or set(first["question_ids"][:3]) == {"a2", "b1", "x_abs"}
    assert len(first["question_ids"]) == len(data)
