from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .checkpoint import CheckpointManifest
from .pi_session import create_checkpoint_session, only_session_file, terminal_entry_id


class PiTdaiBranchAgent(BaseAgent):
    """Harbor agent that keeps one native Pi session across multi-step tasks.

    Natural mode captures a *pre-action* checkpoint at the beginning of every
    Harbor step. That checkpoint contains the current workspace, the Pi session
    ending before the new user instruction, and optional TDAI local state.

    Branch mode restores one such checkpoint, forks the native Pi session, and
    applies a Memory budget override to the first resumed step only.
    """

    capabilities = AgentCapabilities(resume=True)

    @staticmethod
    def name() -> str:
        return "pi-tdai-branch"

    def version(self) -> str:
        return "0.1.0"

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
        branch_control_extension: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.pi_executable = pi_executable
        self.pi_thinking = pi_thinking
        self.pi_extensions = [p for p in pi_extensions.split(",") if p]
        self.tdai_state_dir = tdai_state_dir.strip()
        self.branch_checkpoint = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self.branch_budget_ratio = float(budget_ratio) if budget_ratio not in (None, "") else None
        self.branch_control_extension = branch_control_extension.strip()
        self._step_index = 0
        self._branch_started = False

    @property
    def _remote_session_dir(self) -> PurePosixPath:
        return self.environment_logs_dir / "pi-session"

    @property
    def _remote_action_file(self) -> PurePosixPath:
        return self.environment_logs_dir / "branch-action.json"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(f"mkdir -p {shlex.quote(str(self._remote_session_dir))}")
        if self.branch_checkpoint is not None:
            await self._restore_checkpoint(environment, self.branch_checkpoint)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._step_index += 1
        if self.branch_checkpoint is None:
            await self._capture_pre_action_checkpoint(environment, step_name=f"step-{self._step_index}")
            await self._run_pi(instruction, environment, context, resume=False, fork_session=None, budget_ratio=None)
            return

        manifest = CheckpointManifest.load(self.branch_checkpoint / "checkpoint.json")
        checkpoint_session = self.branch_checkpoint / manifest.pi_checkpoint_session
        await self._run_pi(
            instruction,
            environment,
            context,
            resume=False,
            fork_session=checkpoint_session,
            budget_ratio=self.branch_budget_ratio,
        )
        self._branch_started = True

    async def resume(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._step_index += 1
        if self.branch_checkpoint is None:
            await self._capture_pre_action_checkpoint(environment, step_name=f"step-{self._step_index}")
        # The branch intervention is deliberately one-shot. Every later Harbor
        # step continues the branch session with no forced budget action.
        await self._run_pi(instruction, environment, context, resume=True, fork_session=None, budget_ratio=None)

    async def _run_pi(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
        *,
        resume: bool,
        fork_session: Path | None,
        budget_ratio: float | None,
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
        if budget_ratio is not None and self.branch_control_extension:
            extensions.insert(0, self.branch_control_extension)
        for extension in extensions:
            argv.extend(["--extension", extension])

        exec_env: dict[str, str] = {}
        if budget_ratio is not None:
            payload = {
                "kind": "memory_budget_ratio",
                "budget_ratio": budget_ratio,
                "one_shot": True,
            }
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
                json.dump(payload, handle)
                local_action = Path(handle.name)
            try:
                await environment.upload_file(local_action, str(self._remote_action_file))
            finally:
                local_action.unlink(missing_ok=True)
            exec_env["PI_BRANCH_OUT_ACTION_FILE"] = str(self._remote_action_file)
            # This direct variable is also supplied for a TDAI adapter that does
            # not need the small Pi bridge extension.
            exec_env["TDAI_MEMORY_BUDGET_RATIO_OVERRIDE"] = str(budget_ratio)

        command = " ".join(shlex.quote(arg) for arg in argv)
        result = await environment.exec(command, env=exec_env or None, stdin=instruction)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / f"pi-step-{self._step_index}.stdout.jsonl").write_text(
            result.stdout or "", encoding="utf-8"
        )
        (self.logs_dir / f"pi-step-{self._step_index}.stderr.txt").write_text(
            result.stderr or "", encoding="utf-8"
        )
        if result.return_code != 0:
            raise RuntimeError(f"Pi exited with {result.return_code}: {(result.stderr or '')[-2000:]}")

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
            # Step 1 may legitimately have no prior Pi messages. Keep an empty
            # marker and let branch planning skip this checkpoint.
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

        manifest = CheckpointManifest(
            task_name=environment.environment_name,
            step_index=self._step_index,
            step_name=step_name,
            workspace_archive=workspace_archive.name,
            pi_checkpoint_session=pi_checkpoint,
            pi_source_session=pi_source,
            pi_leaf_id=leaf_id,
            tdai_state_archive=tdai_archive,
            tdai_state_mode=tdai_mode,
        )
        manifest.dump(checkpoint_root / "checkpoint.json")

    async def _restore_checkpoint(self, environment: BaseEnvironment, checkpoint_dir: Path) -> None:
        manifest = CheckpointManifest.load(checkpoint_dir / "checkpoint.json")
        workspace_archive = checkpoint_dir / manifest.workspace_archive
        if not workspace_archive.is_file():
            raise FileNotFoundError(workspace_archive)

        cwd_result = await environment.exec("pwd")
        cwd = (cwd_result.stdout or "/app").strip() or "/app"
        remote_workspace = "/tmp/pi-branch-workspace-restore.tar.gz"
        await environment.upload_file(workspace_archive, remote_workspace)
        # Preserve the working directory itself and replace its contents.
        await environment.exec(
            f"find {shlex.quote(cwd)} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + && "
            f"tar -xzf {shlex.quote(remote_workspace)} -C {shlex.quote(cwd)} && "
            f"rm -f {shlex.quote(remote_workspace)}"
        )

        if manifest.tdai_state_mode == "directory" and manifest.tdai_state_archive and self.tdai_state_dir:
            local_tdai = checkpoint_dir / manifest.tdai_state_archive
            remote_tdai = "/tmp/pi-branch-tdai-restore.tar.gz"
            await environment.upload_file(local_tdai, remote_tdai)
            await environment.exec(
                f"mkdir -p {shlex.quote(self.tdai_state_dir)} && "
                f"find {shlex.quote(self.tdai_state_dir)} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + && "
                f"tar -xzf {shlex.quote(remote_tdai)} -C {shlex.quote(self.tdai_state_dir)} && "
                f"rm -f {shlex.quote(remote_tdai)}"
            )
