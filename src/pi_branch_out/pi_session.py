from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def read_pi_session(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    if not rows or rows[0].get("type") != "session":
        raise ValueError(f"invalid Pi session: {path}")
    return rows


def only_session_file(session_dir: Path) -> Path:
    files = sorted(session_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one Pi session in {session_dir}, found {len(files)}")
    return files[0].resolve()


def terminal_entry_id(path: Path) -> str:
    rows = read_pi_session(path)
    for row in reversed(rows[1:]):
        entry_id = row.get("id")
        if isinstance(entry_id, str) and entry_id:
            return entry_id
    raise ValueError(f"Pi session has no entries: {path}")


def create_checkpoint_session(source_path: Path, leaf_id: str, output_path: Path) -> Path:
    """Copy only the native parentId ancestry ending at ``leaf_id``.

    This intentionally mirrors Pi's own tree semantics. We do not flatten the
    dialogue into a synthetic prompt, so tool calls/results and structured
    messages remain native Pi entries.
    """
    rows = read_pi_session(source_path)
    header = rows[0]
    entries = {
        str(row["id"]): row
        for row in rows[1:]
        if isinstance(row.get("id"), str)
    }
    if leaf_id not in entries:
        raise ValueError(f"Pi entry {leaf_id} is not present in {source_path}")

    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in seen:
            raise ValueError(f"Pi session parent cycle at {current}")
        seen.add(current)
        entry = entries.get(current)
        if entry is None:
            raise ValueError(f"Pi session parent entry {current} is missing")
        path.append(entry)
        parent = entry.get("parentId")
        current = str(parent) if parent is not None else None
    path.reverse()

    checkpoint_header = {
        "type": "session",
        "version": int(header.get("version", 3)),
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": str(header.get("cwd", "")),
        "parentSession": str(source_path.resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(checkpoint_header, ensure_ascii=False) + "\n")
        for entry in path:
            if entry.get("type") != "label":
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return output_path
