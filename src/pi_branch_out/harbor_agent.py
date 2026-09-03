from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .checkpoint import CheckpointManifest
from .observation import BudgetObservation
from .pi_session import create_checkpoint_session, only_session_file, terminal_entry_id


def _parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _parse_granularity(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"compact", "standard", "detailed"}:
        raise ValueError(f"invalid memory granularity: {value}")
    return normalized


class PiTdaiBranchAgent(BaseAgent):
    """Harbor agent that keeps one native Pi session across multi-step tasks.

    Natural mode captures a pre-action checkpoint at the beginning of every
    Harbor step. A checkpoint contains the current workspace, the Pi session
    ending before the new user instruction, and optional local TDAI state.

    Branch mode restores one checkpoint at the same pre-action boundary, forks
    the native Pi session when one exists, and applies one adaptive Memory action
    to the first resumed step only.
    """

    capabilities = AgentCapabilities(resume=True)

    @staticmethod
    def name() -> str:
        return "pi-tdai-branch"

    def version(self) -> str:
        return "0.2.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        pi_executable: str = "pi",
        pi_thinking: str = "off",
        pi_extensions: str = "",
        tdai_state_dir: str = "",
        checkpoint_dir: str = "",
        budget_ratio: float | str | None = None,
        memory_granularity: str = "standard",
        branch_control_extension: str = "",
        require_budget_observation: bool | str = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.pi_executable = pi_executable
        self.pi_thinking = pi_thinking
        self.pi_extensions = [p for p in pi_extensions.split(",") if p]
        self.tdai_state_dir = tdai_state_dir.strip()
        self.branch_checkpoint = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self.branch_budget_ratio = float(budget_ratio) if budget_ratio not in (None, "") else None
        self.memory_granularity = _parse_granularity(memory_granularity)
        self.branch_control_extension = branch_control_extension.strip()
        self.require_budget_observation = _parse_bool(require_budget_observation)
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
    def _remote_observation_file(self) -> PurePosixPath:
        return self.environment_logs_dir / "tdai-budget-observation.json"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(f"mkdir -p {shlex.quote(str(self._remote_session_dir))}")
        if self.branch_control_extension:
            local_bridge = Path(self.branch_control_extension)
            if local_bridge.is_file():
                remote_bridge = self.environment_logs_dir / "tdai-budget-override.ts"
                await environment.upload_file(local_bridge, str(remote_bridge))
                self._remote_branch_control_extension = str(remote_bridge)
            else:
                self._remote_branch_control_extension = self.branch_control_extension

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._step_index += 1
        if self.branch_checkpoint is None:
            await self._capture_pre_action_checkpoint(environment, step_name=f"step-{self._step_index}")
            await self._run_pi(
                instruction,
                environment,
                context,
                resume=False,
                fork_session=None,
                budget_ratio=None,
                granularity="standard",
            )
            return

        if not self._branch_restored:
            await self._restore_checkpoint(environment, self.branch_checkpoint)
            self._branch_restored = True

        manifest = CheckpointManifest.load(self.branch_checkpoint / "checkpoint.json")
        checkpoint_session = (
            self.branch_checkpoint / manifest.pi_checkpoint_session
            if manifest.pi_checkpoint_session
            else None
        )
        await self._run_pi(
            instruction,
            environment,
            context,
            resume=False,
            fork_session=checkpoint_session,
            budget_ratio=self.branch_budget_ratio,
            granularity=self.memory_granularity,
        )

    async def resume(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._step_index += 1
        if self.branch_checkpoint is None:
            await self._capture_pre_action_checkpoint(environment, step_name=f"step-{self._step_index}")
        await self._run_pi(
            instruction,
            environment,
            context,
            resume=True,
            fork_session=None,
            budget_ratio=None,
            granularity="standard",
        )

    async def _run_pi(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
        *,
        resume: bool,
        fork_session: Path | None,
        budget_ratio: float | None,
        granularity: str,
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
        if budget_ratio is not None and self._remote_branch_control_extension:
            extensions.insert(0, self._remote_branch_control_extension)
        for extension in extensions:
            argv.extend(["--extension", extension])

        exec_env: dict[str, str] = {}
        observation_id: str | None = None
        if budget_ratio is not None:
            observation_id = uuid4().hex
            payload = {
                "kind": "memory_budget_ratio",
                "budget_ratio": budget_ratio,
                "granularity": granularity,
                "one_shot": True,
            }
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
                json.dump(payload, handle)
                local_action = Path(handle.name)
            try:
                await environment.upload_file(local_action, str(self._remote_action_file))
            finally:
                local_action.unlink(missing_ok=True)

            await environment.exec(f"rm -f {shlex.quote(str(self._remote_observation_file))}")
            exec_env["PI_BRANCH_OUT_ACTION_FILE"] = str(self._remote_action_file)
            exec_env["TDAI_MEMORY_BUDGET_RATIO_OVERRIDE"] = str(budget_ratio)
            exec_env["TDAI_MEMORY_GRANULARITY_OVERRIDE"] = granularity
            exec_env["TDAI_MEMORY_BUDGET_OVERRIDE_ONE_SHOT"] = "1"
            exec_env["TDAI_BRANCH_OUT_OBSERVATION_FILE"] = str(self._remote_observation_file)
            exec_env["TDAI_BRANCH_OUT_OBSERVATION_ID"] = observation_id

        argv.append(instruction)
        command = " ".join(shlex.quote(arg) for arg in argv)
        result = await environment.exec(command, env=exec_env or None)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / f"pi-step-{self._step_index}.stdout.jsonl").write_text(
            result.stdout or "", encoding="utf-8"
        )
        (self.logs_dir / f"pi-step-{self._step_index}.stderr.txt").write_text(
            result.stderr or "", encoding="utf-8"
        )
        if result.return_code != 0:
            raise RuntimeError(f"Pi exited with {result.return_code}: {(result.stderr or '')[-2000:]}")

        if budget_ratio is not None and observation_id is not None:
            await self._verify_budget_observation(
                environment,
                expected_ratio=budget_ratio,
                expected_granularity=granularity,
                observation_id=observation_id,
            )

    async def _read_proxy_observation(
        self,
        environment: BaseEnvironment,
        observation_id: str,
    ) -> str:
        proxy = await environment.exec("printenv TDAI_PROXY_URL")
        proxy_url = (proxy.stdout or "").strip().rstrip("/")
        if proxy.return_code != 0 or not proxy_url:
            return ""
        url = f"{proxy_url}/__branch_out/observations/{observation_id}"
        fetched = await environment.exec(
            f"curl -fsS --max-time 10 {shlex.quote(url)}"
        )
        if fetched.return_code != 0:
            return ""
        return fetched.stdout or ""

    async def _verify_budget_observation(
        self,
        environment: BaseEnvironment,
        *,
        expected_ratio: float,
        expected_granularity: str,
        observation_id: str,
    ) -> None:
        # In-process TDAI writes directly to the shared agent log path.
        local = await environment.exec(f"cat {shlex.quote(str(self._remote_observation_file))}")
        raw = (local.stdout or "") if local.return_code == 0 else ""

        # Pi's current official TDAI adapter uses MemoryProxy. In that mode the
        # proxy is a different process, so retrieve the observation through the
        # experiment-only rendezvous endpoint instead of assuming a shared FS.
        if not raw.strip():
            raw = await self._read_proxy_observation(environment, observation_id)

        if not raw.strip():
            if self.require_budget_observation:
                raise RuntimeError(
                    "TDAI did not emit a branch budget observation; refusing to keep an unverified branch. "
                    "Apply the MemoryProxy branch-out patch and set TDAI_BRANCH_OUT_ENABLED=1, or pass "
                    "require_budget_observation=false for wiring-only smoke tests."
                )
            return

        observation = BudgetObservation.parse(raw)
        observation.verify(expected_ratio, expected_granularity)
        (self.logs_dir / f"tdai-budget-observation-step-{self._step_index}.json").write_text(
            raw if raw.endswith("\n") else raw + "\n",
            encoding="utf-8",
        )

    async def _capture_pre_action_checkpoint(self, environment: BaseEnvironment, *, step_name: str) -> None:
        checkpoint_root = self.logs_dir / "branch-checkpoints" / f"step-{self._step_index:03d}"
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        cwd_result = await environment.exec("pwd")
        cwd = (cwd_result.stdout or "/app").strip() or "/app"
        remote_workspace = f"/tmp/pi-branch-workspace-{self._step_index}.tar.gz"
        await environment.exec(
            f"tar -czf {shlex.quote(remote_workspace)} -C {shlex.quote(cwd)} ."
        )
        workspace_archive = checkpoint_root / "workspace.tar.gz"
        await environment.download_file(remote_workspace, workspace_archive)
        await environment.exec(f"rm -f {shlex.quote(remote_workspace)}")

        session_download = checkpoint_root / "pi-session-full"
        try:
            await environment.download_dir(str(self._remote_session_dir), session_download)
            source_session = only_session_file(session_download)
            leaf_id = terminal_entry_id(source_session)
            checkpoint_session = create_checkpoint_session(
                source_session,
                leaf_id,
                checkpoint_root / "checkpoint-session.jsonl",
            )
            pi_source = str(source_session.relative_to(checkpoint_root))
            pi_checkpoint = str(checkpoint_session.relative_to(checkpoint_root))
        except Exception:
            if self._step_index > 1:
                raise
            leaf_id = ""
            pi_source = ""
            pi_checkpoint = ""

        tdai_archive: str | None = None
        tdai_mode = "none"
        if self.tdai_state_dir:
            remote_tdai = f"/tmp/pi-branch-tdai-{self._step_index}.tar.gz"
            check = await environment.exec(f"test -d {shlex.quote(self.tdai_state_dir)}")
            if check.return_code == 0:
                await environment.exec(
                    f"tar -czf {shlex.quote(remote_tdai)} -C {shlex.quote(self.tdai_state_dir)} ."
                )
                local_tdai = checkpoint_root / "tdai-state.tar.gz"
                await environment.download_file(remote_tdai, local_tdai)
                await environment.exec(f"rm -f {shlex.quote(remote_tdai)}")
                tdai_archive = local_tdai.name
                tdai_mode = "directory"

        CheckpointManifest(
            task_name=environment.environment_name,
            step_index=self._step_index,
            step_name=step_name,
            workspace_archive=workspace_archive.name,
            pi_checkpoint_session=pi_checkpoint,
            pi_source_session=pi_source,
            pi_leaf_id=leaf_id,
            tdai_state_archive=tdai_archive,
            tdai_state_mode=tdai_mode,
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
            f"tar -xzf {shlex.quote(remote_workspace)} -C {shlex.quote(cwd)} && "
            f"rm -f {shlex.quote(remote_workspace)}"
        )

        if (
            manifest.tdai_state_mode == "directory"
            and manifest.tdai_state_archive
            and self.tdai_state_dir
        ):
            local_tdai = checkpoint_dir / manifest.tdai_state_archive
            remote_tdai = "/tmp/pi-branch-tdai-restore.tar.gz"
            await environment.upload_file(local_tdai, remote_tdai)
            await environment.exec(
                f"mkdir -p {shlex.quote(self.tdai_state_dir)} && "
                f"find {shlex.quote(self.tdai_state_dir)} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + && "
                f"tar -xzf {shlex.quote(remote_tdai)} -C {shlex.quote(self.tdai_state_dir)} && "
                f"rm -f {shlex.quote(remote_tdai)}"
            )
