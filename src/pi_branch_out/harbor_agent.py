from __future__ import annotations

import json
import os
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .checkpoint import CheckpointManifest
from .observation import BudgetObservation
from .pi_session import create_checkpoint_session, only_session_file, terminal_entry_id


_TDAI_ENV_KEYS = (
    "TDAI_PROXY_URL",
    "TDAI_SPACE_ID",
    "TDAI_AGENT_SOURCE",
    "TDAI_TEAM_ID",
    "TDAI_AGENT_ID",
    "TDAI_TASK_ID",
    "TDAI_USER_KEY",
    "TDAI_MODEL",
    "TDAI_MEMORY_HARD_CAP_TOKENS",
    "TDAI_MEMORY_RESERVE_TOKENS",
)


class PiTdaiBranchAgent(BaseAgent):
    """Pi + TDAI + Harbor agent with pre-action checkpoints and frozen recall."""

    capabilities = AgentCapabilities(resume=True)

    @staticmethod
    def name() -> str:
        return "pi-tdai-branch"

    def version(self) -> str:
        return "0.3.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        pi_executable: str = "pi",
        pi_thinking: str = "off",
        pi_extensions: str = "",
        checkpoint_dir: str = "",
        budget_ratio: float | str | None = None,
        branch_control_extension: str = "",
        require_budget_observation: bool | str = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.pi_executable = pi_executable
        self.pi_thinking = pi_thinking
        self.pi_extensions = [p for p in pi_extensions.split(",") if p]
        self.branch_checkpoint = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self.branch_budget_ratio = float(budget_ratio) if budget_ratio not in (None, "") else None
        self.branch_control_extension = branch_control_extension.strip()
        self.require_budget_observation = str(require_budget_observation).lower() not in {"0", "false", "no", "off"}
        self._remote_branch_control_extension = ""
        self._step_index = 0
        self._branch_restored = False

    @property
    def _remote_session_dir(self) -> PurePosixPath:
        return self.environment_logs_dir / "pi-session"

    @property
    def _remote_action_file(self) -> PurePosixPath:
        return self.environment_logs_dir / "branch-action.json"

    @property
    def _remote_snapshot_file(self) -> PurePosixPath:
        return self.environment_logs_dir / "recall-snapshot.json"

    @property
    def _remote_observation_file(self) -> PurePosixPath:
        return self.environment_logs_dir / "budget-observation.json"

    def _tdai_env(self) -> dict[str, str]:
        return {key: value for key in _TDAI_ENV_KEYS if (value := os.environ.get(key))}

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(f"mkdir -p {shlex.quote(str(self._remote_session_dir))}")
        if not self.branch_control_extension:
            return
        local_extension = Path(self.branch_control_extension).resolve()
        if not local_extension.is_file():
            self._remote_branch_control_extension = self.branch_control_extension
            return

        # Preserve the repository layout so the extension can import the local
        # controller/allocator without bundling or modifying TDAI.
        repo_root = Path(__file__).resolve().parents[2]
        remote_root = self.environment_logs_dir / "pi-branch-plugin"
        await environment.exec(
            f"mkdir -p {shlex.quote(str(remote_root / 'extensions'))} {shlex.quote(str(remote_root / 'tdai'))}"
        )
        remote_extension = remote_root / "extensions" / "tdai-budget-override.ts"
        await environment.upload_file(local_extension, str(remote_extension))
        for name in ("memory-budget-controller.ts", "progressive-memory-allocator.ts"):
            await environment.upload_file(repo_root / "tdai" / name, str(remote_root / "tdai" / name))
        self._remote_branch_control_extension = str(remote_extension)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self._step_index += 1
        if self.branch_checkpoint is None:
            await self._capture_pre_action_checkpoint(environment, instruction, step_name=f"step-{self._step_index}")
            await self._run_pi(instruction, environment, resume=False, fork_session=None, budget_ratio=None, snapshot=None)
            return

        if not self._branch_restored:
            await self._restore_checkpoint(environment, self.branch_checkpoint)
            self._branch_restored = True

        manifest = CheckpointManifest.load(self.branch_checkpoint / "checkpoint.json")
        if not manifest.recall_snapshot or manifest.recall_snapshot_status != "ready":
            raise RuntimeError(
                "checkpoint has no frozen recall snapshot; branch step 1 is intentionally unsupported. "
                "Use a natural checkpoint from Harbor step 2 or later."
            )
        checkpoint_session = (
            self.branch_checkpoint / manifest.pi_checkpoint_session if manifest.pi_checkpoint_session else None
        )
        snapshot = self.branch_checkpoint / manifest.recall_snapshot
        await self._run_pi(
            instruction,
            environment,
            resume=False,
            fork_session=checkpoint_session,
            budget_ratio=self.branch_budget_ratio,
            snapshot=snapshot,
        )

    async def resume(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self._step_index += 1
        if self.branch_checkpoint is None:
            await self._capture_pre_action_checkpoint(environment, instruction, step_name=f"step-{self._step_index}")
        await self._run_pi(instruction, environment, resume=True, fork_session=None, budget_ratio=None, snapshot=None)

    async def _run_pi(
        self,
        instruction: str,
        environment: BaseEnvironment,
        *,
        resume: bool,
        fork_session: Path | None,
        budget_ratio: float | None,
        snapshot: Path | None,
    ) -> None:
        argv = [
            self.pi_executable,
            "--print",
            "--mode",
            "json",
            "--session-dir",
            str(self._remote_session_dir),
            "--thinking",
            self.pi_thinking,
        ]
        if self.model_name:
            argv.extend(["--model", self.model_name])
        if resume:
            argv.append("--continue")
        if fork_session is not None:
            remote_fork = self.environment_logs_dir / "checkpoint-session.jsonl"
            await environment.upload_file(fork_session, str(remote_fork))
            argv.extend(["--fork", str(remote_fork)])

        extensions = list(self.pi_extensions)
        exec_env = self._tdai_env()
        if budget_ratio is not None:
            if snapshot is None or not snapshot.is_file():
                raise FileNotFoundError(f"frozen recall snapshot missing: {snapshot}")
            if not self._remote_branch_control_extension:
                raise RuntimeError("branch control extension is not available")
            extensions.insert(0, self._remote_branch_control_extension)
            await environment.upload_file(snapshot, str(self._remote_snapshot_file))
            payload = {"kind": "memory_budget_ratio", "budget_ratio": budget_ratio, "one_step": True}
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
                json.dump(payload, handle)
                local_action = Path(handle.name)
            try:
                await environment.upload_file(local_action, str(self._remote_action_file))
            finally:
                local_action.unlink(missing_ok=True)
            await environment.exec(f"rm -f {shlex.quote(str(self._remote_observation_file))}")
            exec_env.update(
                {
                    "PI_BRANCH_OUT_ACTION_FILE": str(self._remote_action_file),
                    "PI_BRANCH_OUT_RECALL_SNAPSHOT": str(self._remote_snapshot_file),
                    "PI_BRANCH_OUT_OBSERVATION_FILE": str(self._remote_observation_file),
                }
            )

        for extension in extensions:
            argv.extend(["--extension", extension])
        argv.append(instruction)

        result = await environment.exec(" ".join(shlex.quote(arg) for arg in argv), env=exec_env or None)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / f"pi-step-{self._step_index}.stdout.jsonl").write_text(result.stdout or "", encoding="utf-8")
        (self.logs_dir / f"pi-step-{self._step_index}.stderr.txt").write_text(result.stderr or "", encoding="utf-8")
        if result.return_code != 0:
            raise RuntimeError(f"Pi exited with {result.return_code}: {(result.stderr or '')[-2000:]}")

        if budget_ratio is not None:
            raw = await environment.exec(f"cat {shlex.quote(str(self._remote_observation_file))}")
            if raw.return_code != 0 or not (raw.stdout or "").strip():
                if self.require_budget_observation:
                    raise RuntimeError("branch adapter did not emit a budget observation")
                return
            observation = BudgetObservation.parse(raw.stdout or "")
            observation.verify(budget_ratio)
            (self.logs_dir / f"budget-observation-step-{self._step_index}.json").write_text(
                (raw.stdout or "").rstrip() + "\n", encoding="utf-8"
            )

    async def _current_pi_session_id(self, environment: BaseEnvironment) -> str:
        command = (
            "python - <<'PY'\n"
            "import glob,json,os\n"
            f"files=glob.glob({str(self._remote_session_dir)!r}+'/*.jsonl')\n"
            "if not files: raise SystemExit(2)\n"
            "p=max(files,key=os.path.getmtime)\n"
            "with open(p,encoding='utf-8') as f: h=json.loads(f.readline())\n"
            "print(h.get('id',''))\n"
            "PY"
        )
        result = await environment.exec(command)
        return (result.stdout or "").strip() if result.return_code == 0 else ""

    async def _bridge_search(
        self,
        environment: BaseEnvironment,
        *,
        conversation_id: str,
        kind: str,
        query: str,
        limit: int,
    ) -> dict[str, Any]:
        proxy = os.environ.get("TDAI_PROXY_URL", "http://127.0.0.1:8096").rstrip("/")
        space = os.environ.get("TDAI_SPACE_ID", "default")
        url = f"{proxy}/memory-bridge/v3/{kind}"
        body = json.dumps({"query": query[:2048], "limit": limit}, ensure_ascii=False)
        command = " ".join(
            [
                "curl", "-fsS", "--max-time", "20", "-X", "POST", shlex.quote(url),
                "-H", shlex.quote("Content-Type: application/json"),
                "-H", shlex.quote(f"x-conversation-id: {conversation_id}"),
                "-H", shlex.quote(f"x-tdai-service-id: {space}"),
                "-d", shlex.quote(body),
            ]
        )
        result = await environment.exec(command, env=self._tdai_env() or None)
        if result.return_code != 0:
            raise RuntimeError(f"memory bridge {kind} failed: {(result.stderr or '')[-1000:]}")
        value = json.loads(result.stdout or "{}")
        return value if isinstance(value, dict) else {}

    async def _capture_recall_snapshot(
        self,
        environment: BaseEnvironment,
        instruction: str,
        checkpoint_root: Path,
    ) -> tuple[str | None, str]:
        # The Memory Bridge requires an initialized Proxy session. Harbor step 1
        # is therefore baseline-only; from step 2 onward the previous Pi request
        # has initialized the session and we can freeze a true pre-action recall.
        if self._step_index <= 1:
            return None, "session-not-initialized"
        session_id = await self._current_pi_session_id(environment)
        if not session_id:
            return None, "pi-session-missing"
        conversation_id = f"pi-{session_id}"
        try:
            l1 = await self._bridge_search(
                environment,
                conversation_id=conversation_id,
                kind="atomic/search",
                query=instruction,
                limit=36,
            )
            l0 = await self._bridge_search(
                environment,
                conversation_id=conversation_id,
                kind="conversation/search",
                query=instruction,
                limit=72,
            )
        except Exception as exc:
            (checkpoint_root / "recall-snapshot-error.txt").write_text(str(exc) + "\n", encoding="utf-8")
            return None, "bridge-error"

        snapshot = {
            "version": 1,
            "task": environment.environment_name,
            "step_index": self._step_index,
            "query": instruction[:2048],
            "conversation_id": conversation_id,
            "atomic_search": l1,
            "conversation_search": l0,
            "baseline": {"mode": "current-pi", "budget_ratio": 0.0},
        }
        path = checkpoint_root / "recall-snapshot.json"
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path.name, "ready"

    async def _capture_pre_action_checkpoint(
        self,
        environment: BaseEnvironment,
        instruction: str,
        *,
        step_name: str,
    ) -> None:
        checkpoint_root = self.logs_dir / "branch-checkpoints" / f"step-{self._step_index:03d}"
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        cwd_result = await environment.exec("pwd")
        cwd = (cwd_result.stdout or "/app").strip() or "/app"
        remote_workspace = f"/tmp/pi-branch-workspace-{self._step_index}.tar.gz"
        await environment.exec(f"tar -czf {shlex.quote(remote_workspace)} -C {shlex.quote(cwd)} .")
        workspace_archive = checkpoint_root / "workspace.tar.gz"
        await environment.download_file(remote_workspace, workspace_archive)
        await environment.exec(f"rm -f {shlex.quote(remote_workspace)}")

        session_download = checkpoint_root / "pi-session-full"
        try:
            await environment.download_dir(str(self._remote_session_dir), session_download)
            source_session = only_session_file(session_download)
            leaf_id = terminal_entry_id(source_session)
            checkpoint_session = create_checkpoint_session(
                source_session, leaf_id, checkpoint_root / "checkpoint-session.jsonl"
            )
            pi_source = str(source_session.relative_to(checkpoint_root))
            pi_checkpoint = str(checkpoint_session.relative_to(checkpoint_root))
        except Exception:
            if self._step_index > 1:
                raise
            leaf_id = ""
            pi_source = ""
            pi_checkpoint = ""

        snapshot_name, snapshot_status = await self._capture_recall_snapshot(
            environment, instruction, checkpoint_root
        )
        CheckpointManifest(
            task_name=environment.environment_name,
            step_index=self._step_index,
            step_name=step_name,
            workspace_archive=workspace_archive.name,
            pi_checkpoint_session=pi_checkpoint,
            pi_source_session=pi_source,
            pi_leaf_id=leaf_id,
            recall_snapshot=snapshot_name,
            recall_snapshot_status=snapshot_status,
            baseline_budget_ratio=0.0,
            baseline_action=0.0,
        ).dump(checkpoint_root / "checkpoint.json")

    async def _restore_checkpoint(self, environment: BaseEnvironment, checkpoint_dir: Path) -> None:
        manifest = CheckpointManifest.load(checkpoint_dir / "checkpoint.json")
        workspace_archive = checkpoint_dir / manifest.workspace_archive
        if not workspace_archive.is_file():
            raise FileNotFoundError(workspace_archive)
        cwd_result = await environment.exec("pwd")
        cwd = (cwd_result.stdout or "/app").strip() or "/app"
        remote_workspace = "/tmp/pi-branch-workspace-restore.tar.gz"
        await environment.upload_file(workspace_archive, remote_workspace)
        await environment.exec(
            f"find {shlex.quote(cwd)} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + && "
            f"tar -xzf {shlex.quote(remote_workspace)} -C {shlex.quote(cwd)} && rm -f {shlex.quote(remote_workspace)}"
        )
