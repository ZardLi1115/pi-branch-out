from __future__ import annotations

import json
import hashlib
import os
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.base import BaseAgent
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
    "TDAI_WIRE_API",
    "TDAI_MEMORY_HARD_CAP_TOKENS",
    "TDAI_MEMORY_RESERVE_TOKENS",
    "PI_BRANCH_OUT_BACKEND_INSTANCE_ID",
)

_PI_ENV_KEYS = (
    "CUSTOM_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "PI_CODING_AGENT_DIR",
)

_NVM_INSTALL = (
    'curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env -u NODE_VERSION bash && '
    'export NVM_DIR="$HOME/.nvm" && '
    '. "$NVM_DIR/nvm.sh" || true && '
    "command -v nvm >/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
    "nvm install 22 && nvm alias default 22 && npm -v"
)


class PiTdaiBranchAgent(BaseAgent):
    """Pi + TDAI + Harbor agent with pre-action checkpoints and frozen recall."""

    SUPPORTS_RESUME = True

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
        pi_runtime_archive: str = "",
        pi_thinking: str = "off",
        pi_extensions: str = "",
        checkpoint_dir: str = "",
        budget_ratio: float | str | None = None,
        branch_control_extension: str = "",
        require_budget_observation: bool | str = True,
        checkpoint_boundary: str = "harbor-step",
        policy_file: str = "",
        policy_version: str = "",
        max_checkpoints: int | str = 2,
        min_checkpoint_gap: int | str = 10,
        sample_probability: float | str = 0.1,
        max_candidate_probes: int | str = 8,
        sampling_batch: str = "default-v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.pi_executable = pi_executable
        self.pi_runtime_archive = Path(pi_runtime_archive).expanduser().resolve() if pi_runtime_archive else None
        self._remote_pi_runtime_root = ""
        self.pi_thinking = pi_thinking
        self.pi_extensions = [p for p in pi_extensions.split(",") if p]
        self.branch_checkpoint = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        self.branch_budget_ratio = float(budget_ratio) if budget_ratio not in (None, "") else None
        self.branch_control_extension = branch_control_extension.strip()
        self.require_budget_observation = str(require_budget_observation).lower() not in {"0", "false", "no", "off"}
        if checkpoint_boundary not in {"harbor-step", "model-call"}:
            raise ValueError("checkpoint_boundary must be 'harbor-step' or 'model-call'")
        self.checkpoint_boundary = checkpoint_boundary
        self.policy_file = Path(policy_file).expanduser().resolve() if policy_file else None
        self.policy_version = policy_version.strip()
        self.max_checkpoints = int(max_checkpoints)
        self.min_checkpoint_gap = int(min_checkpoint_gap)
        self.sample_probability = float(sample_probability)
        self.max_candidate_probes = int(max_candidate_probes)
        self.sampling_batch = sampling_batch.strip() or "default-v1"
        if self.max_checkpoints < 0 or self.min_checkpoint_gap < 0 or self.max_candidate_probes < 0:
            raise ValueError("sampling counts and gap must be non-negative")
        if not 0 <= self.sample_probability <= 1:
            raise ValueError("sample_probability must be within [0, 1]")
        self._remote_policy_file = ""
        if self.branch_checkpoint is not None:
            manifest = CheckpointManifest.load(self.branch_checkpoint / "checkpoint.json")
            if manifest.checkpoint_boundary == "model-call":
                self.checkpoint_boundary = "model-call"
        self._remote_branch_control_extension = ""
        self._remote_continue_runner = ""
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

    @property
    def _remote_model_call_dir(self) -> PurePosixPath:
        return self.environment_logs_dir / "model-call-checkpoints"

    def _tdai_env(self) -> dict[str, str]:
        env = {key: value for key in _TDAI_ENV_KEYS if (value := os.environ.get(key))}
        proxy = env.get("TDAI_PROXY_URL") or "http://host.docker.internal:8096"
        if "127.0.0.1" in proxy or "localhost" in proxy:
            proxy = proxy.replace("127.0.0.1", "host.docker.internal").replace(
                "localhost", "host.docker.internal"
            )
        env["TDAI_PROXY_URL"] = proxy
        return env

    def _container_url(self, url: str, *, loopback: str) -> str:
        if "127.0.0.1" in url or "localhost" in url:
            return url.replace("127.0.0.1", loopback).replace("localhost", loopback)
        return url

    def _pi_env(self) -> dict[str, str]:
        env = {key: value for key in _PI_ENV_KEYS if (value := os.environ.get(key))}
        if "CUSTOM_API_KEY" in env and "OPENAI_API_KEY" not in env:
            env["OPENAI_API_KEY"] = env["CUSTOM_API_KEY"]
        if "OPENAI_BASE_URL" not in env and os.environ.get("OPENAI_API_BASE"):
            env["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]
        if "OPENAI_BASE_URL" in env:
            env["OPENAI_BASE_URL"] = self._container_url(
                env["OPENAI_BASE_URL"], loopback="host.docker.internal"
            )
        if "OPENAI_API_BASE" in env:
            env["OPENAI_API_BASE"] = self._container_url(
                env["OPENAI_API_BASE"], loopback="host.docker.internal"
            )
        return env

    def _runtime_env(self) -> dict[str, str]:
        return {**self._pi_env(), **self._tdai_env()}

    async def _ensure_pi(self, environment: BaseEnvironment) -> None:
        if self.pi_runtime_archive is not None:
            if not self.pi_runtime_archive.is_file():
                raise FileNotFoundError(f"Pi runtime archive missing: {self.pi_runtime_archive}")
            remote_archive = self.environment_logs_dir / "pi-runtime.tar.gz"
            remote_root = PurePosixPath("/tmp/pi-branch-out-pi-runtime")
            remote_cli = remote_root / "lib/node_modules/@mariozechner/pi-coding-agent/dist/cli.js"
            await environment.upload_file(self.pi_runtime_archive, str(remote_archive))
            install = await environment.exec(
                "bash -lc "
                + shlex.quote(
                    f"set -euo pipefail; rm -rf {remote_root}; mkdir -p {remote_root}; "
                    f"tar -xzf {shlex.quote(str(remote_archive))} -C {remote_root} --strip-components=1; "
                    f"rm -f {shlex.quote(str(remote_archive))}; "
                    f"test -x {remote_root}/bin/node; test -f {remote_cli}; "
                    f"sed -i '1c #!{remote_root}/bin/node' {remote_cli}; chmod +x {remote_cli}; "
                    f"{remote_cli} --version"
                )
            )
            if install.return_code != 0:
                raise RuntimeError(
                    "failed to unpack offline Pi runtime: "
                    + ((install.stderr or install.stdout or "")[-2000:])
                )
            self._remote_pi_runtime_root = str(remote_root)
            self.pi_executable = str(remote_cli)
            return

        check = await environment.exec(
            'bash -lc \'export NVM_DIR="$HOME/.nvm"; '
            '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"; '
            "command -v pi >/dev/null'"
        )
        if check.return_code == 0:
            return
        install = await environment.exec(
            "bash -lc "
            + shlex.quote(
                "set -euo pipefail; "
                + _NVM_INSTALL
                + " && npm install -g --ignore-scripts @mariozechner/pi-coding-agent@0.73.1 && pi --version"
            )
        )
        if install.return_code != 0:
            raise RuntimeError(
                "failed to install Pi in Harbor environment: "
                + ((install.stderr or install.stdout or "")[-2000:])
            )

    async def _write_pi_models(self, environment: BaseEnvironment) -> None:
        # Optional smoke-test provider (`--model cpa/<id>`). The TDAI memory
        # path uses `--model tdai/<id>` from the Pi plugin and does not need
        # this file. Skip rather than bake a machine-local proxy URL.
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        model_id = os.environ.get("CPA_MODEL") or os.environ.get("TDAI_MODEL")
        if not base_url or not model_id:
            return
        if "127.0.0.1" in base_url or "localhost" in base_url:
            base_url = (
                base_url.replace("127.0.0.1", "host.docker.internal").replace(
                    "localhost", "host.docker.internal"
                )
            )
        models = {
            "providers": {
                "cpa": {
                    "baseUrl": base_url,
                    "api": "openai-responses",
                    # Pi resolves this via process.env[value], so it must be a bare
                    # env var name -- "$CUSTOM_API_KEY" would fall through as a literal.
                    "apiKey": "CUSTOM_API_KEY",
                    "authHeader": True,
                    "models": [
                        {
                            "id": model_id,
                            "name": model_id,
                            "reasoning": True,
                            "thinkingLevelMap": {
                                "off": "none",
                                "minimal": "minimal",
                                "low": "low",
                                "medium": "medium",
                                "high": "high",
                            },
                        }
                    ],
                }
            }
        }
        remote_dir = "/root/.pi/agent"
        remote_path = f"{remote_dir}/models.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(models, handle, indent=2)
            handle.write("\n")
            local_path = Path(handle.name)
        try:
            await environment.exec(f"mkdir -p {shlex.quote(remote_dir)}")
            await environment.upload_file(local_path, remote_path)
        finally:
            local_path.unlink(missing_ok=True)

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(f"mkdir -p {shlex.quote(str(self._remote_session_dir))}")
        await self._ensure_pi(environment)
        await self._write_pi_models(environment)

        # If a Pi extension path exists on the host, upload it. This makes a
        # local checkout of the official TDAI Pi plugin usable without asking
        # the caller to separately mount that file into every Harbor container.
        uploaded_extensions: list[str] = []
        remote_user_ext_dir = self.environment_logs_dir / "pi-user-extensions"
        await environment.exec(f"mkdir -p {shlex.quote(str(remote_user_ext_dir))}")
        for index, extension in enumerate(self.pi_extensions):
            local = Path(extension).expanduser().resolve()
            if local.is_file():
                remote = remote_user_ext_dir / f"{index:02d}-{local.name}"
                await environment.upload_file(local, str(remote))
                uploaded_extensions.append(str(remote))
            else:
                uploaded_extensions.append(extension)
        self.pi_extensions = uploaded_extensions

        # Pi 0.73.1 never emits before_provider_headers, so the official TDAI
        # plugin cannot set x-conversation-id. Re-register the provider with
        # that header once the session id exists. Do not patch TDAI source.
        local_conversation_id = Path(__file__).resolve().parents[2] / "extensions" / "tdai-conversation-id.ts"
        if local_conversation_id.is_file():
            remote_conversation_id = remote_user_ext_dir / "99-tdai-conversation-id.ts"
            await environment.upload_file(local_conversation_id, str(remote_conversation_id))
            remote_conversation_id_s = str(remote_conversation_id)
            if remote_conversation_id_s not in self.pi_extensions:
                self.pi_extensions.append(remote_conversation_id_s)

        if self.checkpoint_boundary == "model-call":
            repo_root = Path(__file__).resolve().parents[2]
            local_collector = repo_root / "extensions" / "tdai-model-call-collector.ts"
            remote_collector_root = self.environment_logs_dir / "pi-collector-plugin"
            await environment.exec(
                f"mkdir -p {shlex.quote(str(remote_collector_root / 'extensions'))} {shlex.quote(str(remote_collector_root / 'tdai'))}"
            )
            remote_collector = remote_collector_root / "extensions" / "tdai-model-call-collector.ts"
            await environment.upload_file(local_collector, str(remote_collector))
            await environment.upload_file(
                repo_root / "extensions" / "tdai-budget-override.ts",
                str(remote_collector_root / "extensions" / "tdai-budget-override.ts"),
            )
            for name in ("memory-budget-controller.ts", "progressive-memory-allocator.ts"):
                await environment.upload_file(repo_root / "tdai" / name, str(remote_collector_root / "tdai" / name))
            self.pi_extensions.append(str(remote_collector))
            local_runner = repo_root / "scripts" / "pi-continue.mjs"
            remote_runner = remote_user_ext_dir / "pi-continue.mjs"
            await environment.upload_file(local_runner, str(remote_runner))
            self._remote_continue_runner = str(remote_runner)
            await environment.exec(f"mkdir -p {shlex.quote(str(self._remote_model_call_dir))}")

        if self.policy_file is not None:
            if not self.policy_file.is_file():
                raise FileNotFoundError(f"policy file missing: {self.policy_file}")
            repo_root = Path(__file__).resolve().parents[2]
            remote_policy_root = self.environment_logs_dir / "pi-policy-plugin"
            await environment.exec(
                f"mkdir -p {shlex.quote(str(remote_policy_root / 'extensions'))} {shlex.quote(str(remote_policy_root / 'tdai'))}"
            )
            for extension_name in ("tdai-budget-policy.ts", "tdai-budget-override.ts"):
                await environment.upload_file(
                    repo_root / "extensions" / extension_name,
                    str(remote_policy_root / "extensions" / extension_name),
                )
            for name in ("memory-budget-controller.ts", "progressive-memory-allocator.ts"):
                await environment.upload_file(repo_root / "tdai" / name, str(remote_policy_root / "tdai" / name))
            remote_policy = remote_policy_root / "policy.json"
            await environment.upload_file(self.policy_file, str(remote_policy))
            self._remote_policy_file = str(remote_policy)
            self.pi_extensions.insert(0, str(remote_policy_root / "extensions" / "tdai-budget-policy.ts"))

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
            if self.checkpoint_boundary == "harbor-step":
                await self._capture_pre_action_checkpoint(environment, instruction, step_name=f"step-{self._step_index}")
            await self._run_pi(
                instruction, environment, resume=False, fork_session=None,
                budget_ratio=None, snapshot=None, continue_from_checkpoint=False,
            )
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
            continue_from_checkpoint=manifest.checkpoint_boundary == "model-call",
        )

    async def resume(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self._step_index += 1
        if self.branch_checkpoint is None and self.checkpoint_boundary == "harbor-step":
            await self._capture_pre_action_checkpoint(environment, instruction, step_name=f"step-{self._step_index}")
        await self._run_pi(
            instruction, environment, resume=True, fork_session=None,
            budget_ratio=None, snapshot=None, continue_from_checkpoint=False,
        )

    async def _run_pi(
        self,
        instruction: str,
        environment: BaseEnvironment,
        *,
        resume: bool,
        fork_session: Path | None,
        budget_ratio: float | None,
        snapshot: Path | None,
        continue_from_checkpoint: bool,
    ) -> None:
        pi_args = [
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
            pi_args.extend(["--model", self.model_name])
        if resume:
            pi_args.append("--continue")
        if fork_session is not None:
            remote_fork = self.environment_logs_dir / "checkpoint-session.jsonl"
            await environment.upload_file(fork_session, str(remote_fork))
            pi_args.extend(["--fork", str(remote_fork)])

        extensions = list(self.pi_extensions)
        exec_env = self._runtime_env()
        if self._remote_policy_file:
            exec_env["PI_BRANCH_OUT_POLICY_FILE"] = self._remote_policy_file
            exec_env["PI_BRANCH_OUT_POLICY_VERSION"] = self.policy_version or "unversioned-policy"
        if self.checkpoint_boundary == "model-call":
            call_offset = 0
            if self.branch_checkpoint is not None:
                manifest = CheckpointManifest.load(self.branch_checkpoint / "checkpoint.json")
                call_offset = max(0, (manifest.model_call_index or 1) - 1)
            exec_env.update(
                {
                    "PI_BRANCH_OUT_MODEL_CALL_DIR": str(self._remote_model_call_dir),
                    "PI_BRANCH_OUT_CALL_OFFSET": str(call_offset),
                    "PI_BRANCH_OUT_TASK_NAME": environment.environment_name,
                    "PI_BRANCH_OUT_MAX_CHECKPOINTS": str(self.max_checkpoints),
                    "PI_BRANCH_OUT_MIN_CHECKPOINT_GAP": str(self.min_checkpoint_gap),
                    "PI_BRANCH_OUT_SAMPLE_PROBABILITY": str(self.sample_probability),
                    "PI_BRANCH_OUT_MAX_CANDIDATE_PROBES": str(self.max_candidate_probes),
                    "PI_BRANCH_OUT_SAMPLING_BATCH": self.sampling_batch,
                }
            )
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
            pi_args.extend(["--extension", extension])
        if continue_from_checkpoint:
            if fork_session is None:
                raise RuntimeError("model-call continuation requires a checkpoint session")
            if not self._remote_continue_runner:
                raise RuntimeError("Pi continuation runner is not available")
            exec_env.update(
                {
                    "PI_BRANCH_OUT_SESSION": str(remote_fork),
                    "PI_BRANCH_OUT_LEAF_ID": manifest.pi_leaf_id,
                    "PI_BRANCH_OUT_MODEL": self.model_name or "",
                    "PI_BRANCH_OUT_THINKING": self.pi_thinking,
                    "PI_BRANCH_OUT_EXTENSIONS": json.dumps(extensions),
                }
            )
            if self._remote_pi_runtime_root:
                inner = (
                    f"export PATH={shlex.quote(self._remote_pi_runtime_root + '/bin')}:$PATH; "
                    f"export PI_CODING_AGENT_ROOT={shlex.quote(self._remote_pi_runtime_root + '/lib/node_modules/@mariozechner/pi-coding-agent')}; "
                    f"{shlex.quote(self._remote_pi_runtime_root + '/bin/node')} "
                    f"{shlex.quote(self._remote_continue_runner)} </dev/null"
                )
            else:
                inner = (
                    'export NVM_DIR="$HOME/.nvm"; '
                    '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"; '
                    'export PI_CODING_AGENT_ROOT="$(npm root -g)/@mariozechner/pi-coding-agent"; '
                    f"node {shlex.quote(self._remote_continue_runner)} </dev/null"
                )
        else:
            pi_args.append(instruction)
            inner = (
                (
                    f"export PATH={shlex.quote(self._remote_pi_runtime_root + '/bin')}:$PATH; "
                    if self._remote_pi_runtime_root
                    else ""
                )
                +
                'export NVM_DIR="$HOME/.nvm"; '
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"; '
                + " ".join(shlex.quote(arg) for arg in pi_args)
                # Non-interactive: with an open stdin Pi can block waiting on input
                # instead of exiting after --print.
                + " </dev/null"
            )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        local_stdout = self.logs_dir / f"pi-step-{self._step_index}.stdout.jsonl"
        local_stderr = self.logs_dir / f"pi-step-{self._step_index}.stderr.txt"
        remote_stdout = f"/tmp/pi-branch-out-step-{self._step_index}.stdout.jsonl"
        remote_stderr = f"/tmp/pi-branch-out-step-{self._step_index}.stderr.txt"
        redirected = (
            f"set +e; ( {inner} ) >{shlex.quote(remote_stdout)} 2>{shlex.quote(remote_stderr)}; "
            'status=$?; printf "pi-exit=%s\\n" "$status"; exit "$status"'
        )
        argv = ["bash", "-lc", redirected]
        result = await environment.exec(" ".join(shlex.quote(arg) for arg in argv), env=exec_env or None)
        try:
            await environment.download_file(remote_stdout, local_stdout)
            await environment.download_file(remote_stderr, local_stderr)
        finally:
            await environment.exec(
                f"rm -f {shlex.quote(remote_stdout)} {shlex.quote(remote_stderr)}"
            )
        if result.return_code != 0:
            stderr_tail = ""
            if local_stderr.is_file():
                with local_stderr.open("rb") as handle:
                    handle.seek(0, 2)
                    handle.seek(max(0, handle.tell() - 8192))
                    stderr_tail = handle.read().decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"Pi exited with {result.return_code}: {stderr_tail}")

        if budget_ratio is not None:
            raw = await environment.exec(f"cat {shlex.quote(str(self._remote_observation_file))}")
            if raw.return_code != 0 or not (raw.stdout or "").strip():
                if self.require_budget_observation:
                    raise RuntimeError("branch adapter did not emit a budget observation")
                return
            observation = BudgetObservation.parse(raw.stdout or "")
            observation.verify(budget_ratio)
            if snapshot is not None:
                expected_snapshot_id = f"sha256:{hashlib.sha256(snapshot.read_bytes()).hexdigest()}"
                if observation.snapshot_id != expected_snapshot_id:
                    raise RuntimeError("branch adapter used a different recall snapshot than the checkpoint")
            (self.logs_dir / f"budget-observation-step-{self._step_index}.json").write_text(
                (raw.stdout or "").rstrip() + "\n", encoding="utf-8"
            )

    async def _current_pi_session_id(self, environment: BaseEnvironment) -> str:
        # python3, not python: Ubuntu ships no unversioned `python`, and the
        # heredoc swallows the resulting "command not found" as rc=0, so this
        # failed silently and every recall snapshot came back as
        # "pi-session-missing".
        command = (
            "python3 - <<'PY'\n"
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
        proxy = os.environ.get("TDAI_PROXY_URL", "http://host.docker.internal:8096").rstrip("/")
        if "127.0.0.1" in proxy or "localhost" in proxy:
            proxy = proxy.replace("127.0.0.1", "host.docker.internal").replace(
                "localhost", "host.docker.internal"
            )
        space = os.environ.get("TDAI_SPACE_ID", "default")
        url = f"{proxy}/memory-bridge/v3/{kind}"
        body = json.dumps({"query": query[:2048], "limit": limit}, ensure_ascii=False)
        # Do not use curl -f: it discards the HTTP body on 4xx, which is the
        # only place proxy puts "session not initialized" / auth errors.
        remote_body = f"/tmp/tdai-bridge-{kind.replace('/', '-')}.json"
        bridge_conversation_id = (
            f"codex:{conversation_id}"
            if os.environ.get("TDAI_WIRE_API") == "responses" and ":" not in conversation_id
            else conversation_id
        )
        candidate_keys = [bridge_conversation_id]
        if bridge_conversation_id != conversation_id:
            candidate_keys.append(conversation_id)
        for index, candidate_key in enumerate(candidate_keys):
            command = " ".join(
                [
                    "curl", "-sS", "--max-time", "20", "-o", shlex.quote(remote_body),
                    "-w", shlex.quote("%{http_code}"),
                    "-X", "POST", shlex.quote(url),
                    "-H", shlex.quote("Content-Type: application/json"),
                    "-H", shlex.quote(f"x-conversation-id: {candidate_key}"),
                    "-H", shlex.quote(f"x-tdai-service-id: {space}"),
                    "-d", shlex.quote(body),
                ]
            )
            result = await environment.exec(command, env=self._runtime_env() or None)
            status = (result.stdout or "").strip()
            raw = await environment.exec(f"cat {shlex.quote(remote_body)} 2>/dev/null || true")
            payload = (raw.stdout or "").strip()
            if result.return_code == 0 and status.startswith("2"):
                value = json.loads(payload or "{}")
                return value if isinstance(value, dict) else {}
            cold_miss = status == "401" and "session not initialized" in payload
            if not cold_miss or index + 1 == len(candidate_keys):
                detail = payload or (result.stderr or "").strip() or status
                raise RuntimeError(f"memory bridge {kind} failed: HTTP {status or '?'} {detail[-1000:]}")
        raise RuntimeError(f"memory bridge {kind} failed without a response")

    async def _capture_recall_snapshot(
        self,
        environment: BaseEnvironment,
        instruction: str,
        checkpoint_root: Path,
    ) -> tuple[str | None, str, str | None]:
        if self._step_index <= 1:
            return None, "session-not-initialized", None
        session_id = await self._current_pi_session_id(environment)
        if not session_id:
            return None, "pi-session-missing", None
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
            return None, "bridge-error", None

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
        snapshot_text = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
        path.write_text(snapshot_text, encoding="utf-8")
        return path.name, "ready", hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()

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

        snapshot_name, snapshot_status, snapshot_sha256 = await self._capture_recall_snapshot(
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
            snapshot_sha256=snapshot_sha256,
            backend_instance_id=os.environ.get("PI_BRANCH_OUT_BACKEND_INSTANCE_ID"),
            backend_proxy_sha256=hashlib.sha256(
                os.environ.get("TDAI_PROXY_URL", "").encode("utf-8")
            ).hexdigest(),
        ).dump(checkpoint_root / "checkpoint.json")

    async def _restore_checkpoint(self, environment: BaseEnvironment, checkpoint_dir: Path) -> None:
        manifest = CheckpointManifest.load(checkpoint_dir / "checkpoint.json")
        if manifest.workspace_mode == "git-delta-v1":
            if not manifest.workspace_base_commit or not manifest.workspace_patch:
                raise ValueError("git-delta-v1 checkpoint is missing base commit or patch")
            patch = checkpoint_dir / manifest.workspace_patch
            if not patch.is_file():
                raise FileNotFoundError(patch)
            cwd_result = await environment.exec("pwd")
            cwd = (cwd_result.stdout or "/app").strip() or "/app"
            command = (
                f"git -C {shlex.quote(cwd)} reset --hard {shlex.quote(manifest.workspace_base_commit)} && "
                f"git -C {shlex.quote(cwd)} clean -fd"
            )
            if patch.stat().st_size > 0:
                remote_patch = "/tmp/pi-branch-workspace.patch"
                await environment.upload_file(patch, remote_patch)
                command += (
                    f" && git -C {shlex.quote(cwd)} apply --binary {shlex.quote(remote_patch)}"
                    f" && rm -f {shlex.quote(remote_patch)}"
                )
            result = await environment.exec(command)
            if result.return_code != 0:
                raise RuntimeError(f"failed to restore git workspace delta: {(result.stderr or result.stdout or '')[-2000:]}")
            if manifest.workspace_untracked_archive:
                untracked = checkpoint_dir / manifest.workspace_untracked_archive
                if not untracked.is_file():
                    raise FileNotFoundError(untracked)
                remote_untracked = "/tmp/pi-branch-untracked.tar.gz"
                await environment.upload_file(untracked, remote_untracked)
                result = await environment.exec(
                    f"tar -xzf {shlex.quote(remote_untracked)} -C {shlex.quote(cwd)} && "
                    f"rm -f {shlex.quote(remote_untracked)}"
                )
                if result.return_code != 0:
                    raise RuntimeError(f"failed to restore untracked files: {(result.stderr or result.stdout or '')[-2000:]}")
            return
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


class CheckpointScoreAgent(PiTdaiBranchAgent):
    """Restore a checkpoint and let Harbor run the official verifier.

    No Pi/model request is made. Harbor creates a fresh task environment, so
    verifier logs and test files never enter the source agent trajectory.
    """

    @staticmethod
    def name() -> str:
        return "pi-branch-checkpoint-score"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if self.branch_checkpoint is None:
            raise ValueError("checkpoint_dir is required")
        await self._restore_checkpoint(environment, self.branch_checkpoint)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        return None

    async def resume(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        return None


class InitialScoreAgent(BaseAgent):
    """Make no workspace changes; Harbor verifies the pristine task state."""

    @staticmethod
    def name() -> str:
        return "pi-branch-initial-score"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        return None

    async def resume(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        return None
