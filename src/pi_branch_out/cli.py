from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .branch_task import build_branch_task
from .checkpoint import BranchAction, CheckpointManifest
from .observation import BudgetObservation
from .training_data import PriceTable, build_roadmapbench_split, export_training_data
from .policy_training import train_policy
from .selection import select_checkpoints
from .evaluation import summarize_evaluation


AGENT_IMPORT = "pi_branch_out.harbor_agent:PiTdaiBranchAgent"
SCORE_AGENT_IMPORT = "pi_branch_out.harbor_agent:CheckpointScoreAgent"
INITIAL_SCORE_AGENT_IMPORT = "pi_branch_out.harbor_agent:InitialScoreAgent"
DEFAULT_ACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _agent_kwargs(args: argparse.Namespace, *, checkpoint: Path | None = None, action: BranchAction | None = None) -> list[str]:
    pairs: list[tuple[str, str]] = [
        ("pi_executable", args.pi_executable),
        ("pi_runtime_archive", args.pi_runtime_archive),
        ("pi_thinking", args.pi_thinking),
        ("pi_extensions", ",".join(args.pi_extension or [])),
        ("checkpoint_boundary", args.checkpoint_boundary),
        ("policy_file", args.policy_file),
        ("policy_version", args.policy_version),
        ("max_checkpoints", str(args.max_checkpoints)),
        ("min_checkpoint_gap", str(args.min_checkpoint_gap)),
        ("sample_probability", str(args.sample_probability)),
        ("max_candidate_probes", str(args.max_candidate_probes)),
        ("sampling_batch", args.sampling_batch),
    ]
    if args.branch_control_extension:
        pairs.append(("branch_control_extension", str(Path(args.branch_control_extension).resolve())))
    if checkpoint is not None:
        pairs.append(("checkpoint_dir", str(checkpoint.resolve())))
    if action is not None:
        pairs.append(("budget_ratio", str(action.budget_ratio)))
        pairs.append(("require_budget_observation", "false" if args.allow_unverified_budget else "true"))

    result: list[str] = []
    for key, value in pairs:
        result.extend(["--agent-kwarg", f"{key}={value}"])
    return result


def _harbor_command(
    args: argparse.Namespace,
    *,
    task: Path,
    jobs_dir: Path,
    checkpoint: Path | None = None,
    action: BranchAction | None = None,
) -> list[str]:
    command = shlex.split(args.harbor_bin)
    command.extend(
        [
            "run", "--path", str(task.resolve()),
            "--agent", AGENT_IMPORT,
            "--model", args.model,
            "--n-attempts", "1",
            "--n-concurrent", "1",
            "--environment-build-timeout-multiplier", str(args.environment_build_timeout_multiplier),
            "--resume-trajectory",
            "--jobs-dir", str(jobs_dir.resolve()),
        ]
    )
    for task_name in args.include_task_name:
        command.extend(["--include-task-name", task_name])
    if args.n_tasks is not None:
        command.extend(["--n-tasks", str(args.n_tasks)])
    command.extend(_agent_kwargs(args, checkpoint=checkpoint, action=action))
    return command


def _run(command: list[str], *, env_overrides: dict[str, str] | None = None) -> int:
    print("[pi-branch-out]", shlex.join(command), flush=True)
    env = os.environ.copy()
    env.update(env_overrides or {})
    return subprocess.run(command, env=env, check=False).returncode


def run_natural(args: argparse.Namespace) -> int:
    task = Path(args.task).resolve()
    jobs_dir = Path(args.jobs_dir).resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return _run(_harbor_command(args, task=task, jobs_dir=jobs_dir))


def _run_one_branch(args: argparse.Namespace, checkpoint: Path, ratio: float, output_root: Path, run_id: str) -> int:
    source_task = Path(args.task).resolve()
    manifest = CheckpointManifest.load(checkpoint / "checkpoint.json")
    if manifest.recall_snapshot_status != "ready" or not manifest.recall_snapshot:
        raise ValueError(
            f"checkpoint {manifest.step_name} has no frozen recall snapshot and cannot be used for strict branch-out"
        )
    snapshot_path = checkpoint / manifest.recall_snapshot
    if manifest.snapshot_sha256 and hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != manifest.snapshot_sha256:
        raise ValueError("frozen recall snapshot fingerprint does not match checkpoint manifest")
    isolation_mode = args.tdai_isolation_mode
    if isolation_mode == "shared" and not args.allow_shared_backend_long_branch:
        raise ValueError(
            "long-horizon branch-out requires an isolated TDAI instance or a restored backend snapshot; "
            "frozen recall alone only makes the first injection reproducible. Use --tdai-isolation-mode "
            "isolated-instance|snapshot-restore, or explicitly opt into a non-training local test with "
            "--allow-shared-backend-long-branch"
        )
    backend_instance_id = args.backend_instance_id.replace("{run_id}", run_id)
    if isolation_mode != "shared" and not backend_instance_id:
        raise ValueError("--backend-instance-id is required for an isolated backend branch")
    backend_proxy_url = args.backend_proxy_url.strip().replace("{run_id}", run_id)
    if isolation_mode == "isolated-instance":
        if not backend_proxy_url:
            raise ValueError("--backend-proxy-url is required and must route to the isolated TDAI instance")
        source_proxy = os.environ.get("TDAI_PROXY_URL", "").rstrip("/").lower()
        if source_proxy and backend_proxy_url.rstrip("/").lower() == source_proxy:
            raise ValueError("isolated backend proxy URL must differ from the source/shared TDAI_PROXY_URL")
    if not manifest.snapshot_sha256 or len(manifest.snapshot_sha256) != 64:
        raise ValueError("strict branch-out requires a checkpoint with a SHA-256 snapshot fingerprint")
    action = BranchAction(ratio)
    run_root = output_root / run_id
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"branch output is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    branch_task = build_branch_task(source_task, checkpoint, run_root / "task")
    jobs_dir = run_root / "jobs"
    metadata = {
        "run_id": run_id,
        "source_task": str(source_task),
        "checkpoint": str(checkpoint),
        "source_step_index": manifest.step_index,
        "source_step_name": manifest.step_name,
        "recall_snapshot": manifest.recall_snapshot,
        "baseline_budget_ratio": manifest.baseline_budget_ratio,
        "snapshot_sha256": manifest.snapshot_sha256,
        "allocator_version": manifest.allocator_version,
        "tokenizer_version": manifest.tokenizer_version,
        "action_table_version": manifest.action_table_version,
        "policy_version": manifest.policy_version,
        "tdai_isolation_mode": isolation_mode,
        "backend_instance_id": backend_instance_id or None,
        "backend_proxy_sha256": hashlib.sha256(backend_proxy_url.encode("utf-8")).hexdigest() if backend_proxy_url else None,
        "training_eligible_long_horizon": isolation_mode != "shared",
        "action": action.as_runtime_payload(),
        "branch_task": str(branch_task),
        "jobs_dir": str(jobs_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_root / "branch_request.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env_overrides = {}
    if isolation_mode == "isolated-instance":
        env_overrides = {
            "TDAI_PROXY_URL": backend_proxy_url,
            "PI_BRANCH_OUT_BACKEND_INSTANCE_ID": backend_instance_id,
        }
    rc = _run(
        _harbor_command(args, task=branch_task, jobs_dir=jobs_dir, checkpoint=checkpoint, action=action),
        env_overrides=env_overrides,
    )
    metadata["return_code"] = rc
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    (run_root / "branch_result.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rc


def run_branch(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("branch-%Y%m%dT%H%M%SZ")
    return _run_one_branch(args, checkpoint, float(args.budget_ratio), Path(args.output_root).resolve(), run_id)


def _parse_ratios(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        values = list(DEFAULT_ACTIONS)
    for value in values:
        BranchAction(value)
    return values


def run_branch_grid(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).resolve()
    ratios = list(dict.fromkeys(_parse_ratios(args.ratios)))
    root = Path(args.output_root).resolve()
    stamp = args.run_id_prefix or datetime.now(timezone.utc).strftime("grid-%Y%m%dT%H%M%SZ")
    manifest = CheckpointManifest.load(checkpoint / "checkpoint.json")
    plans_by_ratio: dict[float, dict] = {}
    if manifest.recall_snapshot:
        snapshot_path = checkpoint / manifest.recall_snapshot
        if snapshot_path.is_file():
            snapshot_bytes = snapshot_path.read_bytes()
            actual_sha = hashlib.sha256(snapshot_bytes).hexdigest()
            if manifest.snapshot_sha256 and actual_sha != manifest.snapshot_sha256:
                raise ValueError("frozen recall snapshot fingerprint does not match checkpoint manifest")
            snapshot = json.loads(snapshot_bytes.decode("utf-8"))
            for plan in snapshot.get("action_plans", []):
                if isinstance(plan, dict) and isinstance(plan.get("ratio"), (int, float)):
                    plans_by_ratio[float(plan["ratio"])] = plan

    groups: dict[str, list[float]] = {}
    for ratio in ratios:
        plan = next((value for key, value in plans_by_ratio.items() if abs(key - ratio) <= 1e-9), None)
        group_key = str(plan.get("injected_content_sha256")) if plan else f"unplanned:{ratio:.12g}"
        groups.setdefault(group_key, []).append(ratio)
    equivalence = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "snapshot_sha256": manifest.snapshot_sha256,
        "groups": [
            {"injected_content_sha256": key, "executed_ratio": values[0], "alias_ratios": values[1:]}
            for key, values in groups.items()
        ],
    }
    if len(groups) > 1 and args.tdai_isolation_mode == "isolated-instance":
        if "{run_id}" not in args.backend_instance_id or "{run_id}" not in args.backend_proxy_url:
            raise ValueError(
                "branch-grid with multiple effective actions requires {run_id} in both "
                "--backend-instance-id and --backend-proxy-url so every branch uses a fresh backend"
            )
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{stamp}-equivalent-actions.json").write_text(
        json.dumps(equivalence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failures = 0
    for group_ratios in groups.values():
        ratio = group_ratios[0]
        run_id = f"{stamp}-budget-{ratio:.3f}"
        rc = _run_one_branch(args, checkpoint, ratio, root, run_id)
        if rc != 0:
            failures += 1
            if args.fail_fast:
                return rc
    return 1 if failures else 0


def run_split_roadmapbench(args: argparse.Namespace) -> int:
    manifest = build_roadmapbench_split(
        Path(args.overview).resolve(),
        Path(args.output).resolve(),
        seed=args.seed,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False), flush=True)
    return 0


def run_export_training(args: argparse.Namespace) -> int:
    manifest = export_training_data(
        Path(args.natural_root).resolve(),
        Path(args.branch_root).resolve(),
        Path(args.output_dir).resolve(),
        Path(args.split_manifest).resolve(),
        prices=PriceTable(
            input_per_million=args.input_per_million,
            output_per_million=args.output_per_million,
            cache_read_per_million=args.cache_read_per_million,
            cache_write_per_million=args.cache_write_per_million,
        ),
        cost_coefficient=args.cost_coefficient,
        cost_normalizer_usd=args.cost_normalizer_usd,
        actions=tuple(_parse_ratios(args.ratios)),
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


def run_train_policy(args: argparse.Namespace) -> int:
    result = train_policy(
        Path(args.dataset_dir).resolve(),
        Path(args.output_dir).resolve(),
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        pretrain_epochs=args.pretrain_epochs,
        cql_epochs=args.cql_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        cql_alpha=args.cql_alpha,
        gamma=args.gamma,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def run_score_checkpoint(args: argparse.Namespace) -> int:
    source_task = Path(args.task).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"score output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    branch_task = build_branch_task(source_task, checkpoint, output_root / "task")
    jobs_dir = output_root / "jobs"
    command = shlex.split(args.harbor_bin) + [
        "run", "--path", str(branch_task), "--agent", SCORE_AGENT_IMPORT,
        "--model", args.model, "--n-attempts", "1", "--n-concurrent", "1",
        "--environment-build-timeout-multiplier", str(args.environment_build_timeout_multiplier),
        "--jobs-dir", str(jobs_dir), "--agent-kwarg", f"checkpoint_dir={checkpoint}",
    ]
    rc = _run(command)
    results = sorted(jobs_dir.rglob("result.json"), key=lambda path: path.stat().st_mtime_ns)
    if rc != 0 or not results:
        return rc or 1
    result = json.loads(results[-1].read_text(encoding="utf-8"))
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    if not isinstance(reward, (int, float)):
        fallback = results[-1].parent / "verifier" / "reward.json"
        if fallback.is_file():
            reward = json.loads(fallback.read_text(encoding="utf-8")).get("reward")
    if not isinstance(reward, (int, float)):
        raise RuntimeError("official RoadmapBench verifier did not emit a numeric reward")
    manifest = CheckpointManifest.load(checkpoint / "checkpoint.json")
    score = {
        "schema_version": 1,
        "reward": float(reward),
        "official_verifier": True,
        "isolated_copy": True,
        "checkpoint": str(checkpoint),
        "snapshot_sha256": manifest.snapshot_sha256,
        "result": str(results[-1]),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    (checkpoint / "checkpoint-score.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def run_score_initial(args: argparse.Namespace) -> int:
    task = Path(args.task).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"initial-score output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    jobs_dir = output_root / "jobs"
    command = shlex.split(args.harbor_bin) + [
        "run", "--path", str(task), "--agent", INITIAL_SCORE_AGENT_IMPORT,
        "--model", args.model, "--n-attempts", "1", "--n-concurrent", "1",
        "--environment-build-timeout-multiplier", str(args.environment_build_timeout_multiplier),
        "--jobs-dir", str(jobs_dir),
    ]
    rc = _run(command)
    results = sorted(jobs_dir.rglob("result.json"), key=lambda path: path.stat().st_mtime_ns)
    if rc != 0 or not results:
        return rc or 1
    result = json.loads(results[-1].read_text(encoding="utf-8"))
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    if not isinstance(reward, (int, float)):
        fallback = results[-1].parent / "verifier" / "reward.json"
        if fallback.is_file():
            reward = json.loads(fallback.read_text(encoding="utf-8")).get("reward")
    if not isinstance(reward, (int, float)):
        raise RuntimeError("official RoadmapBench verifier did not emit a numeric initial reward")
    score = {
        "schema_version": 1,
        "task_id": task.name,
        "reward": float(reward),
        "official_verifier": True,
        "isolated_copy": True,
        "result": str(results[-1]),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    score_output = Path(args.score_output).resolve()
    score_output.parent.mkdir(parents=True, exist_ok=True)
    score_output.write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def run_select_checkpoints(args: argparse.Namespace) -> int:
    result = select_checkpoints(
        Path(args.natural_root).resolve(),
        Path(args.branch_root).resolve(),
        Path(args.output).resolve(),
        per_task=args.per_task,
        seed=args.seed,
    )
    print(json.dumps({"selected": len(result["selected"])}, ensure_ascii=False), flush=True)
    return 0


def run_verify_recovery(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint).resolve()
    branch_root = Path(args.branch_run).resolve()
    manifest = CheckpointManifest.load(checkpoint / "checkpoint.json")
    if manifest.source_reward is None:
        raise ValueError("checkpoint has no Natural source reward; run collection finalization first")
    observations = list((branch_root / "jobs").rglob("budget-observation-step-*.json"))
    results = list((branch_root / "jobs").rglob("result.json"))
    if len(observations) != 1 or len(results) != 1:
        raise ValueError("recovery control must contain exactly one budget observation and one trial result")
    observation = BudgetObservation.parse(observations[0].read_text(encoding="utf-8"))
    observation.verify(0.0)
    result = json.loads(results[0].read_text(encoding="utf-8"))
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    branch_reward = rewards.get("reward")
    if not isinstance(branch_reward, (int, float)):
        fallback = results[0].parent / "verifier" / "reward.json"
        if fallback.is_file():
            branch_reward = json.loads(fallback.read_text(encoding="utf-8")).get("reward")
    if not isinstance(branch_reward, (int, float)):
        raise ValueError("recovery branch has no numeric official reward")
    passed = abs(float(branch_reward) - manifest.source_reward) <= args.reward_tolerance
    audit = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "branch_run": str(branch_root),
        "snapshot_sha256": manifest.snapshot_sha256,
        "natural_reward": manifest.source_reward,
        "replay_reward": float(branch_reward),
        "reward_tolerance": args.reward_tolerance,
        "budget_zero_verified": observation.injected_tokens == 0,
        "passed": passed and observation.injected_tokens == 0,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if audit["passed"] else 1


def run_summarize_evaluation(args: argparse.Namespace) -> int:
    variants: dict[str, Path] = {}
    for raw in args.variant:
        name, separator, path = raw.partition("=")
        if not separator or not name or name in variants:
            raise ValueError("--variant must be a unique NAME=PATH pair")
        variants[name] = Path(path).resolve()
    result = summarize_evaluation(
        variants,
        Path(args.output).resolve(),
        prices=PriceTable(
            args.input_per_million, args.output_per_million,
            args.cache_read_per_million, args.cache_write_per_million,
        ),
        quality_tolerance=args.quality_tolerance,
    )
    print(json.dumps(result["variants"], ensure_ascii=False), flush=True)
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--harbor-bin", default="harbor")
    parser.add_argument("--pi-executable", default="pi")
    parser.add_argument(
        "--pi-runtime-archive",
        default="",
        help="Host path to a prebuilt Linux Pi runtime archive. When set, upload it instead of installing Pi with NVM/npm.",
    )
    parser.add_argument("--pi-thinking", default="off")
    parser.add_argument(
        "--environment-build-timeout-multiplier",
        type=float,
        default=2.0,
        help="Harbor environment build timeout multiplier. The RoadmapBench default 600 seconds becomes 1200 seconds.",
    )
    parser.add_argument(
        "--include-task-name",
        action="append",
        default=[],
        help="Harbor dataset task-name filter; repeatable and supports glob patterns.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="Maximum number of dataset tasks after filtering.",
    )
    parser.add_argument(
        "--checkpoint-boundary",
        choices=("harbor-step", "model-call"),
        default="harbor-step",
        help="Freeze state before each Harbor step or each internal Pi model call.",
    )
    parser.add_argument(
        "--pi-extension", action="append", default=[],
        help="Pi extension path visible inside Harbor; repeatable. Add the official TDAI Pi plugin here.",
    )
    parser.add_argument(
        "--branch-control-extension",
        default=str(Path(__file__).resolve().parents[2] / "extensions" / "tdai-budget-override.ts"),
        help="Host path to the external budget adapter. It never patches TDAI.",
    )
    parser.add_argument(
        "--policy-file",
        default="",
        help="Frozen policy.json emitted by train-policy. When set, it controls every non-overridden model call.",
    )
    parser.add_argument(
        "--policy-version",
        default="",
        help="Frozen policy version recorded in collection logs.",
    )
    parser.add_argument("--max-checkpoints", type=int, default=2)
    parser.add_argument("--min-checkpoint-gap", type=int, default=10)
    parser.add_argument("--sample-probability", type=float, default=0.1)
    parser.add_argument("--max-candidate-probes", type=int, default=8)
    parser.add_argument("--sampling-batch", default="default-v1")


def _branch_common(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default="branch_runs")
    parser.add_argument(
        "--allow-unverified-budget", action="store_true",
        help="Keep a branch if the external adapter does not emit an observation. Wiring-only; do not use for training data.",
    )
    parser.add_argument(
        "--tdai-isolation-mode",
        choices=("shared", "isolated-instance"),
        default="shared",
        help="TDAI backend isolation used by this branch. Shared mode is rejected for long-horizon runs by default.",
    )
    parser.add_argument(
        "--backend-instance-id",
        default="",
        help="Stable isolated instance identifier; branch-grid requires a {run_id} template.",
    )
    parser.add_argument(
        "--backend-proxy-url",
        default="",
        help="Proxy URL of a genuinely independent TDAI deployment; branch-grid requires a {run_id} template.",
    )
    parser.add_argument(
        "--allow-shared-backend-long-branch",
        action="store_true",
        help="Run a non-training local injection test against a shared backend. Frozen recall does not isolate later queries/writes.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pi + TDAI + Harbor structured branch-out runner")
    sub = parser.add_subparsers(dest="command", required=True)

    natural = sub.add_parser("natural", help="Run the untouched Pi+TDAI baseline and freeze pre-action recall snapshots")
    _common(natural)
    natural.add_argument("--task", required=True)
    natural.add_argument("--jobs-dir", required=True)
    natural.set_defaults(func=run_natural)

    branch = sub.add_parser("branch", help="Restore one checkpoint and force one Memory Budget ratio")
    _branch_common(branch)
    branch.add_argument("--budget-ratio", required=True, type=float)
    branch.add_argument("--run-id")
    branch.set_defaults(func=run_branch)

    grid = sub.add_parser("branch-grid", help="Run several budget counterfactuals from the same frozen checkpoint")
    _branch_common(grid)
    grid.add_argument("--ratios", default=",".join(str(v) for v in DEFAULT_ACTIONS))
    grid.add_argument("--run-id-prefix")
    grid.add_argument("--fail-fast", action="store_true")
    grid.set_defaults(func=run_branch_grid)

    split = sub.add_parser("split-roadmapbench", help="Create a stable task-level train/dev/test split")
    split.add_argument("--overview", required=True, help="RoadmapBench data/tasks_overview.jsonl")
    split.add_argument("--output", required=True)
    split.add_argument("--seed", default="tdai-budget-v1")
    split.add_argument("--train-fraction", type=float, default=0.7)
    split.add_argument("--dev-fraction", type=float, default=0.15)
    split.set_defaults(func=run_split_roadmapbench)

    export = sub.add_parser("export-training", help="Export weak labels and real-action transitions separately")
    export.add_argument("--natural-root", required=True)
    export.add_argument("--branch-root", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--split-manifest", required=True)
    export.add_argument("--ratios", default=",".join(str(v) for v in DEFAULT_ACTIONS))
    export.add_argument("--cost-coefficient", type=float, required=True)
    export.add_argument("--cost-normalizer-usd", type=float, default=1.0)
    export.add_argument("--input-per-million", type=float, required=True)
    export.add_argument("--output-per-million", type=float, required=True)
    export.add_argument("--cache-read-per-million", type=float, required=True)
    export.add_argument("--cache-write-per-million", type=float, required=True)
    export.set_defaults(func=run_export_training)

    train = sub.add_parser("train-policy", help="Train the small MLP Q-network with CQL on real actions")
    train.add_argument("--dataset-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--pretrain-epochs", type=int, default=20)
    train.add_argument("--cql-epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--cql-alpha", type=float, default=1.0)
    train.add_argument("--gamma", type=float, default=1.0)
    train.set_defaults(func=run_train_policy)

    score = sub.add_parser("score-checkpoint", help="Restore one checkpoint in an isolated Harbor task and run the official verifier")
    score.add_argument("--task", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--output-root", required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--harbor-bin", default="harbor")
    score.add_argument("--environment-build-timeout-multiplier", type=float, default=2.0)
    score.set_defaults(func=run_score_checkpoint)

    initial = sub.add_parser("score-initial", help="Run the official verifier on a pristine RoadmapBench task without a model")
    initial.add_argument("--task", required=True)
    initial.add_argument("--output-root", required=True)
    initial.add_argument("--score-output", required=True)
    initial.add_argument("--model", required=True)
    initial.add_argument("--harbor-bin", default="harbor")
    initial.add_argument("--environment-build-timeout-multiplier", type=float, default=2.0)
    initial.set_defaults(func=run_score_initial)

    select = sub.add_parser("select-checkpoints", help="Choose one or two informative recoverable nodes per RoadmapBench task")
    select.add_argument("--natural-root", required=True)
    select.add_argument("--branch-root", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--per-task", type=int, choices=(1, 2), default=2)
    select.add_argument("--seed", default="tdai-budget-selection-v1")
    select.set_defaults(func=run_select_checkpoints)

    recovery = sub.add_parser("verify-recovery", help="Compare a budget-0 replay with its Natural source trajectory")
    recovery.add_argument("--checkpoint", required=True)
    recovery.add_argument("--branch-run", required=True)
    recovery.add_argument("--output", required=True)
    recovery.add_argument("--reward-tolerance", type=float, default=0.0)
    recovery.set_defaults(func=run_verify_recovery)

    evaluation = sub.add_parser("summarize-evaluation", help="Compute task-paired quality, success, cost and policy p95 latency")
    evaluation.add_argument("--variant", action="append", required=True, help="NAME=PATH; first variant is the baseline")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--quality-tolerance", type=float, required=True)
    evaluation.add_argument("--input-per-million", type=float, required=True)
    evaluation.add_argument("--output-per-million", type=float, required=True)
    evaluation.add_argument("--cache-read-per-million", type=float, required=True)
    evaluation.add_argument("--cache-write-per-million", type=float, required=True)
    evaluation.set_defaults(func=run_summarize_evaluation)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
