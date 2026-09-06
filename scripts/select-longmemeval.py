from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_selection(data: list[dict[str, Any]], seed: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in data:
        stratum = "abstention" if "_abs" in str(entry["question_id"]) else str(entry["question_type"])
        groups[stratum].append(entry)
    for rows in groups.values():
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}\0{row['question_id']}".encode()).digest())
    ordered: list[str] = []
    positions = {name: 0 for name in groups}
    names = sorted(groups)
    while len(ordered) < len(data):
        progressed = False
        for name in names:
            index = positions[name]
            if index >= len(groups[name]):
                continue
            ordered.append(str(groups[name][index]["question_id"]))
            positions[name] += 1
            progressed = True
        if not progressed:
            break
    return {
        "schema_version": "longmemeval-stratified-selection-v1",
        "seed": seed,
        "question_ids": ordered,
        "strata_counts": {name: len(groups[name]) for name in names},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stable stratified LongMemEval question order")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="longmemeval-selection-v1")
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    selection = build_selection(data, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(args.output), **selection["strata_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
