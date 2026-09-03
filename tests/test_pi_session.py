from __future__ import annotations

import json
from pathlib import Path

from pi_branch_out.pi_session import create_checkpoint_session, read_pi_session, terminal_entry_id


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_checkpoint_follows_parent_chain_only(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write(
        source,
        [
            {"type": "session", "version": 3, "id": "s1", "cwd": "/app"},
            {"type": "message", "id": "u1", "parentId": None, "message": {"role": "user"}},
            {"type": "message", "id": "a1", "parentId": "u1", "message": {"role": "assistant"}},
            {"type": "message", "id": "t1", "parentId": "a1", "message": {"role": "toolResult"}},
            {"type": "message", "id": "side", "parentId": "u1", "message": {"role": "assistant"}},
        ],
    )

    output = create_checkpoint_session(source, "t1", tmp_path / "checkpoint.jsonl")
    rows = read_pi_session(output)
    assert [row.get("id") for row in rows[1:]] == ["u1", "a1", "t1"]
    assert rows[0]["parentSession"] == str(source.resolve())


def test_terminal_entry_id_reads_last_native_entry(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write(
        source,
        [
            {"type": "session", "version": 3, "id": "s1"},
            {"type": "message", "id": "m1", "parentId": None},
            {"type": "message", "id": "m2", "parentId": "m1"},
        ],
    )
    assert terminal_entry_id(source) == "m2"
