from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MARKER = "pi-branch-out adaptive memory"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected patch anchor not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def copy_runtime_files(repo_root: Path, patch_root: Path) -> None:
    target = repo_root / "MemoryProxy" / "src" / "branch-out"
    target.mkdir(parents=True, exist_ok=True)
    files = {
        patch_root / "tdai" / "memory-budget-controller.ts": target / "memory-budget-controller.ts",
        patch_root / "tdai" / "progressive-memory-allocator.ts": target / "progressive-memory-allocator.ts",
        patch_root / "tdai" / "memory-proxy" / "token-estimator.ts": target / "token-estimator.ts",
        patch_root / "tdai" / "memory-proxy" / "branch-observation-store.ts": target / "branch-observation-store.ts",
        patch_root / "tdai" / "memory-proxy" / "tdai-adaptive-memory-injector.ts": target / "tdai-adaptive-memory-injector.ts",
    }
    for source, destination in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)


def patch_types(repo_root: Path) -> None:
    path = repo_root / "MemoryProxy" / "src" / "tdai" / "types.ts"
    anchor = "export interface TdaiL2Entry {\n"
    addition = """/** pi-branch-out adaptive memory: query-conditioned raw L0 hit. */
export interface TdaiL0Memory {
  id: string;
  role?: string;
  content: string;
  score?: number;
  sessionId?: string;
  recordedAt?: string;
}

export interface TdaiL2Entry {
"""
    replace_once(path, anchor, addition)


def patch_client(repo_root: Path) -> None:
    path = repo_root / "MemoryProxy" / "src" / "tdai" / "client.ts"
    replace_once(
        path,
        "  TdaiL1Memory,\n  TdaiL2Entry,",
        "  TdaiL1Memory,\n  TdaiL0Memory,\n  TdaiL2Entry,",
    )
    anchor = "  async listL2(identity: TdaiIdentity): Promise<TdaiL2Entry[]> {\n"
    method = """  /** pi-branch-out adaptive memory: one broad L0 search per source agent. */
  async searchL0ForCtx(
    ctx: TdaiAgentCtx,
    query: string,
    sessionId: string,
    taskId?: string,
    limit = 24,
  ): Promise<TdaiL0Memory[]> {
    if (!this.isEnabled() || !query.trim()) return [];
    const data = await this.postForCtx<{ messages?: Array<Record<string, unknown>> }>(
      "/v3/conversation/search",
      ctx,
      {
        team_id: ctx.teamId,
        user_id: ctx.userId,
        agent_id: ctx.agentId,
        task_id: taskId,
        query: query.slice(0, 2048),
        limit,
      },
      sessionId,
      taskId,
      { includeSession: true, includeTask: true },
    );
    return (data.messages ?? [])
      .map((item) => ({
        id: String(item.id ?? ""),
        role: typeof item.role === "string" ? item.role : undefined,
        content: typeof item.content === "string" ? item.content : "",
        score: typeof item.score === "number" ? item.score : undefined,
        sessionId: typeof item.session_id === "string" ? item.session_id : undefined,
        recordedAt:
          typeof item.recorded_at === "string"
            ? item.recorded_at
            : typeof item.timestamp === "string"
              ? item.timestamp
              : undefined,
      }))
      .filter((item) => item.id && item.content);
  }

  async listL2(identity: TdaiIdentity): Promise<TdaiL2Entry[]> {
"""
    replace_once(path, anchor, method)


def patch_injection_registry(repo_root: Path) -> None:
    path = repo_root / "MemoryProxy" / "src" / "injection" / "index.ts"
    replace_once(
        path,
        'import { TdaiProfileMemoryInjector } from "./injectors/tdai-profile-memory-injector.js";\n',
        'import { TdaiProfileMemoryInjector } from "./injectors/tdai-profile-memory-injector.js";\n'
        'import { TdaiClient } from "../tdai/client.js";\n'
        'import { TdaiAdaptiveMemoryInjector } from "../branch-out/tdai-adaptive-memory-injector.js";\n',
    )
    anchor = """    if (config.tdai.memory.injectL2L3) {
      registry.register(new TdaiProfileMemoryInjector(tdaiBaseConfig, config.coreSkill));
    }
"""
    replacement = """    if (config.tdai.memory.injectL2L3) {
      registry.register(new TdaiProfileMemoryInjector(tdaiBaseConfig, config.coreSkill));
    }
    // pi-branch-out adaptive memory: this hook is inert unless the request
    // carries an explicit branch/policy action in metadata.custom.branchMemory.
    // Therefore today's normal Proxy behavior remains unchanged.
    if (config.tdai.memory.recallL1) {
      registry.register(
        new TdaiAdaptiveMemoryInjector(new TdaiClient(tdaiBaseConfig), config.coreSkill),
      );
    }
"""
    replace_once(path, anchor, replacement)


def patch_handler(repo_root: Path) -> None:
    path = repo_root / "MemoryProxy" / "src" / "handler.ts"
    anchor = """      const injectionTurnSeq = countHumanTurns(messages, "openai");
      const { getInjectionPipeline } = await import("./injection/index.js");
"""
    replacement = """      const injectionTurnSeq = countHumanTurns(messages, "openai");
      // pi-branch-out adaptive memory: explicit request-scoped action forwarded
      // by the Pi extension. Missing headers preserve the baseline exactly.
      const branchRatioHeader = c.req.header("x-tdai-memory-budget-ratio");
      const branchObservationId = c.req.header("x-tdai-branch-observation-id") ?? "";
      const branchContextWindow = c.req.header("x-tdai-context-window-tokens");
      const branchGranularity = c.req.header("x-tdai-memory-granularity") ?? "standard";
      const branchMemory = branchRatioHeader != null
        ? {
            ratio: Number(branchRatioHeader),
            granularity: branchGranularity,
            observationId: branchObservationId,
            contextWindowTokens: Number(branchContextWindow ?? 0),
          }
        : null;
      const { getInjectionPipeline } = await import("./injection/index.js");
"""
    replace_once(path, anchor, replacement)

    old_custom = """        custom: sessionInfo
          ? {
              session: sessionInfo,
              assetCapabilities,
              userKey: apiKey || undefined,
            }
          : undefined,
"""
    new_custom = """        custom: sessionInfo
          ? {
              session: sessionInfo,
              assetCapabilities,
              userKey: apiKey || undefined,
              ...(branchMemory ? { branchMemory } : {}),
            }
          : undefined,
"""
    replace_once(path, old_custom, new_custom)


def patch_server(repo_root: Path) -> None:
    path = repo_root / "MemoryProxy" / "src" / "server.ts"
    replace_once(
        path,
        'import { getEffectiveBackend } from "./storage/factory.js";\n',
        'import { getEffectiveBackend } from "./storage/factory.js";\n'
        'import { branchOutEnabled, takeBranchMemoryObservation } from "./branch-out/branch-observation-store.js";\n',
    )
    anchor = """  // Whoami: resolve API key → key ID (plain text, easy to use with curl)
"""
    route = """  // pi-branch-out adaptive memory: experiment-only realized-action rendezvous.
  // Disabled by default. Observation ids are random per Harbor branch step and
  // entries are consumed on read.
  app.get("/__branch_out/observations/:id", (c) => {
    if (!branchOutEnabled()) return c.json({ error: "branch_out_disabled" }, 404);
    const observation = takeBranchMemoryObservation(c.req.param("id"));
    if (!observation) return c.json({ error: "observation_not_found" }, 404);
    return c.json(observation);
  });

  // Whoami: resolve API key → key ID (plain text, easy to use with curl)
"""
    replace_once(path, anchor, route)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply pi-branch-out adaptive MemoryProxy integration")
    parser.add_argument("tdai_repo", type=Path, help="Checkout of TencentDB-Agent-Memory (feat/server_team)")
    args = parser.parse_args()

    repo_root = args.tdai_repo.resolve()
    patch_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "MemoryProxy" / "src" / "handler.ts").is_file():
        raise SystemExit(f"not a TencentDB-Agent-Memory checkout: {repo_root}")

    copy_runtime_files(repo_root, patch_root)
    patch_types(repo_root)
    patch_client(repo_root)
    patch_injection_registry(repo_root)
    patch_handler(repo_root)
    patch_server(repo_root)
    print(f"Applied {MARKER} patch to {repo_root}")


if __name__ == "__main__":
    main()
