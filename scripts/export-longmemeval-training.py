from __future__ import annotations

import argparse
import json
from pathlib import Path

from pi_branch_out.longmemeval_training import export_longmemeval_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paired LongMemEval budget actions for the existing CQL trainer")
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-coefficient", type=float, default=0.0)
    parser.add_argument("--cost-normalizer-tokens", type=float, default=10_000.0)
    parser.add_argument(
        "--cost-measure",
        choices=("answer-billable-token-proxy", "injected-l1-tokens"),
        default="answer-billable-token-proxy",
    )
    parser.add_argument("--split-seed", default="longmemeval-v1")
    parser.add_argument("--question-ids-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-action-features", action="store_true")
    args = parser.parse_args()
    result = export_longmemeval_training(
        args.collection_root,
        args.output_dir,
        cost_coefficient=args.cost_coefficient,
        cost_normalizer_tokens=args.cost_normalizer_tokens,
        cost_measure=args.cost_measure,
        split_seed=args.split_seed,
        question_ids_file=args.question_ids_file,
        limit=args.limit,
        include_action_features=args.include_action_features,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
