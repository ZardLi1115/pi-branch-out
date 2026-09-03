import type { AgentContext, ContextBlock, InjectionHook, HookPriority } from "../injection/types.js";
import { HOOK_PRIORITY } from "../injection/types.js";
import { getLastUserMessage, getMessageText } from "../injection/context.js";
import { extractUserQueryText } from "../tdai/recorder.js";
import type { TdaiClient } from "../tdai/client.js";
import type { TdaiAgentCtx, TdaiL0Memory } from "../tdai/types.js";
import { getTdaiIdentity } from "../tdai/identity.js";
import { getMetadataClient } from "../meta/client.js";
import { resolveFixedAssetCtxs } from "../injection/injectors/tdai-fixed-asset.js";
import type { CoreSkillConfig } from "../types.js";
import { decideMemoryBudget } from "./memory-budget-controller.js";
import {
  allocateProgressiveMemory,
  renderProgressiveMemory,
  type L1Candidate,
  type MemoryGranularity,
} from "./progressive-memory-allocator.js";
import { estimateContextTokens, estimateTextTokens } from "./token-estimator.js";
import { putBranchMemoryObservation } from "./branch-observation-store.js";

interface BranchMemoryRequest {
  ratio: number;
  granularity: MemoryGranularity;
  observationId: string;
  contextWindowTokens: number;
}

interface L1Hit {
  id: string;
  type?: string;
  content: string;
  score?: number;
  fromAgentId: string;
  fromAgentName?: string;
  sourceCtx: TdaiAgentCtx;
}

interface L0SearchClient {
  searchL0ForCtx(
    ctx: TdaiAgentCtx,
    query: string,
    sessionId: string,
    taskId?: string,
    limit?: number,
  ): Promise<TdaiL0Memory[]>;
}

function readBranchRequest(ctx: AgentContext): BranchMemoryRequest | null {
  const custom = ctx.metadata.custom as Record<string, unknown> | undefined;
  const raw = custom?.branchMemory as Record<string, unknown> | undefined;
  if (!raw) return null;

  const ratio = Number(raw.ratio);
  const contextWindowTokens = Number(raw.contextWindowTokens);
  const observationId = typeof raw.observationId === "string" ? raw.observationId : "";
  const granularityRaw = String(raw.granularity ?? "standard");
  if (!Number.isFinite(ratio) || ratio < 0 || ratio > 1) {
    throw new Error(`invalid branch memory ratio: ${raw.ratio}`);
  }
  if (!Number.isFinite(contextWindowTokens) || contextWindowTokens <= 0) {
    throw new Error(`invalid branch context window: ${raw.contextWindowTokens}`);
  }
  if (!observationId) throw new Error("branch observation id is required");
  if (!["compact", "standard", "detailed"].includes(granularityRaw)) {
    throw new Error(`invalid branch memory granularity: ${granularityRaw}`);
  }

  return {
    ratio,
    contextWindowTokens: Math.floor(contextWindowTokens),
    observationId,
    granularity: granularityRaw as MemoryGranularity,
  };
}

function currentContextTokens(ctx: AgentContext): number {
  const messages = ctx.messages.map((message) => ({
    role: message.role,
    content: message.blocks.map((block) => block.content).join("\n"),
  }));
  let tokens = estimateContextTokens(messages);
  if (ctx.tools?.length) tokens += estimateTextTokens(JSON.stringify(ctx.tools));
  return tokens;
}

function distributeL0Pool(
  l1s: L1Candidate[],
  sourceAgentId: string,
  pool: TdaiL0Memory[],
): void {
  const targets = l1s.filter((item) => item.fromAgentId === sourceAgentId);
  if (targets.length === 0) return;

  // Until atomic/search exposes source_message_ids, this is deliberately a
  // deterministic distribution of one query-conditioned L0 pool. It does not
  // pretend to be provenance. When provenance is exposed, only this function
  // needs to change.
  pool.forEach((item, index) => {
    const target = targets[index % targets.length];
    target.l0.push({
      id: item.id,
      content: item.content,
      tokenCount: estimateTextTokens(item.content),
      score: item.score,
      sessionId: item.sessionId,
      role: item.role,
    });
  });
}

export class TdaiAdaptiveMemoryInjector implements InjectionHook {
  id = "tdai-adaptive-memory-injector";
  point = "user.before" as const;
  priority: HookPriority = HOOK_PRIORITY.MEMORY;
  description = "One-shot adaptive L1/L0 injection for branch-out and future policy actions";

  constructor(
    private client: TdaiClient,
    private coreSkillCfg: Pick<CoreSkillConfig, "endpoint" | "serviceToken" | "serviceId" | "timeoutMs"> | null = null,
    private perAgentL1Limit = 12,
    private perAgentL0Limit = 24,
  ) {}

  async execute(ctx: AgentContext): Promise<ContextBlock[]> {
    const branch = readBranchRequest(ctx);
    // Preserve today's Proxy baseline exactly when branch-out is not active.
    if (!branch) return [];

    const identity = getTdaiIdentity(ctx.metadata.custom);
    if (!identity) throw new Error("branch memory request has no TDAI identity");

    const lastUser = getLastUserMessage(ctx);
    if (!lastUser) return [];
    const query = extractUserQueryText(getMessageText(lastUser)).trim().slice(0, 2048);
    if (!query) return [];

    const session = (ctx.metadata.custom as any)?.session as {
      user_key?: string;
      space_id?: string;
    } | undefined;
    const userKey = session?.user_key;
    const spaceId = session?.space_id ?? "";
    const metadataClient = this.coreSkillCfg && userKey
      ? getMetadataClient(this.coreSkillCfg, spaceId, userKey)
      : null;
    const sourceCtxs = await resolveFixedAssetCtxs(ctx, identity, metadataClient);

    const l1Groups = await Promise.all(
      sourceCtxs.map(async (sourceCtx) => {
        const items = await this.client.searchL1ForCtx(
          sourceCtx,
          query,
          identity.sessionId,
          identity.taskId,
          this.perAgentL1Limit,
        );
        return items.map((item): L1Hit => ({
          ...item,
          fromAgentId: sourceCtx.agentId,
          fromAgentName: sourceCtx.agentName,
          sourceCtx,
        }));
      }),
    );

    const hits = ([] as L1Hit[])
      .concat(...l1Groups)
      .sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));

    const candidates: L1Candidate[] = hits.map((hit) => ({
      id: hit.id,
      type: hit.type,
      content: hit.content,
      score: hit.score,
      fromAgentId: hit.fromAgentId,
      fromAgentName: hit.fromAgentName,
      tokenCount: estimateTextTokens(hit.content),
      l0: [],
    }));

    const currentTokens = currentContextTokens(ctx);
    const maxOutput = Number(ctx.requestParams.max_tokens ?? ctx.requestParams.max_completion_tokens ?? 0);
    const safetyReserve = Number(process.env.TDAI_MEMORY_SAFETY_RESERVE_TOKENS ?? 1024);
    const reserveTokens = Math.max(0, Number.isFinite(maxOutput) ? maxOutput : 0)
      + Math.max(0, Number.isFinite(safetyReserve) ? safetyReserve : 1024);
    const hardCapRaw = Number(process.env.TDAI_MEMORY_HARD_CAP_TOKENS ?? 0);

    const decision = decideMemoryBudget({
      branchRatio: branch.ratio,
      contextWindowTokens: branch.contextWindowTokens,
      currentContextTokens: currentTokens,
      reserveTokens,
      hardCapTokens: Number.isFinite(hardCapRaw) && hardCapRaw > 0 ? hardCapRaw : null,
    });

    // Phase 1 admits complete L1 only. This tells us which source agents are
    // actually relevant to L0 expansion, so we do not perform needless L0 calls.
    const l1Only = allocateProgressiveMemory({
      candidates,
      budgetTokens: decision.budgetTokens,
      granularity: "compact",
    });
    const admittedIds = new Set(l1Only.selected.map((item) => item.id));
    const admitted = candidates.filter((item) => admittedIds.has(item.id));

    if (branch.granularity !== "compact" && admitted.length > 0) {
      const ctxByAgent = new Map(sourceCtxs.map((item) => [item.agentId, item]));
      const sourceAgentIds = [...new Set(admitted.map((item) => item.fromAgentId).filter(Boolean))] as string[];
      const l0Client = this.client as TdaiClient & L0SearchClient;

      await Promise.all(
        sourceAgentIds.map(async (agentId) => {
          const sourceCtx = ctxByAgent.get(agentId);
          if (!sourceCtx) return;
          const pool = await l0Client.searchL0ForCtx(
            sourceCtx,
            query,
            identity.sessionId,
            identity.taskId,
            this.perAgentL0Limit,
          );
          distributeL0Pool(admitted, agentId, pool);
        }),
      );
    }

    const allocation = allocateProgressiveMemory({
      candidates,
      budgetTokens: decision.budgetTokens,
      granularity: branch.granularity,
    });
    const rendered = renderProgressiveMemory(allocation);

    putBranchMemoryObservation({
      kind: "memory_budget_ratio",
      observation_id: branch.observationId,
      source: decision.source,
      requested_ratio: decision.requestedRatio,
      applied_ratio: decision.appliedRatio,
      granularity: branch.granularity,
      context_window_tokens: decision.contextWindowTokens,
      current_context_tokens: decision.currentContextTokens,
      reserve_tokens: decision.reserveTokens,
      feasible_budget_tokens: decision.feasibleBudgetTokens,
      budget_tokens: decision.budgetTokens,
      injected_tokens: allocation.injectedTokens,
      l1_ids: allocation.selected.map((item) => item.id),
      l0_ids: allocation.selected.flatMap((item) => item.selectedL0.map((chunk) => chunk.id)),
      created_at: new Date().toISOString(),
    });

    if (!rendered) return [];
    return [{
      type: "text",
      content: rendered,
      metadata: {
        source: this.id,
        budgetRatio: branch.ratio,
        granularity: branch.granularity,
        budgetTokens: decision.budgetTokens,
        injectedTokens: allocation.injectedTokens,
        l1Count: allocation.selected.length,
        l0Count: allocation.selected.reduce((sum, item) => sum + item.selectedL0.length, 0),
      },
    }];
  }
}
