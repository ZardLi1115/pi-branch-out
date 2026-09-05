import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { decideMemoryBudget } from "../tdai/memory-budget-controller.js";
import { allocateProgressiveMemory, renderProgressiveMemory, type L1Candidate } from "../tdai/progressive-memory-allocator.js";

type ExtensionAPI = { on(event: string, callback: (event: any, ctx: any) => unknown): void };
export type SearchHit = {
  id?: unknown; type?: unknown; content?: unknown; score?: unknown; role?: unknown;
  session_id?: unknown; source_agent_id?: unknown; source_agent_name?: unknown;
  parent_id?: unknown; parent_l1_id?: unknown; atomic_id?: unknown; memory_id?: unknown;
  source_memory_id?: unknown;
};
export type Snapshot = {
  version: number;
  query: string;
  model_call_index?: number;
  atomic_search: unknown;
  conversation_search: unknown;
  baseline?: Record<string, unknown>;
};

export function textOfContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content == null ? "" : JSON.stringify(content);
  return content.map((part) => {
    if (typeof part === "string") return part;
    if (!part || typeof part !== "object") return "";
    const p = part as Record<string, unknown>;
    return typeof p.text === "string" ? p.text : typeof p.content === "string" ? p.content : "";
  }).filter(Boolean).join("\n");
}

export function estimateTextTokens(text: string): number {
  if (!text) return 0;
  let cjk = 0;
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if ((cp >= 0x3400 && cp <= 0x9fff) || (cp >= 0xf900 && cp <= 0xfaff)) cjk += 1;
  }
  return Math.max(1, Math.ceil(cjk / 1.7 + Math.max(0, text.length - cjk) / 4));
}

export function estimateMessagesTokens(messages: any[]): number {
  return messages.reduce((sum, m) => sum + estimateTextTokens(`${String(m?.role ?? "")}\n${textOfContent(m?.content)}`) + 4, 0);
}

export function estimateContextTokens(event: any): number {
  const messages = Array.isArray(event?.messages) ? event.messages : [];
  const system = textOfContent(event?.systemPrompt ?? event?.system ?? "");
  const tools = Array.isArray(event?.tools) ? JSON.stringify(event.tools) : "";
  return estimateMessagesTokens(messages) + estimateTextTokens(system) + estimateTextTokens(tools) + 8;
}

function rows(value: unknown, key: "items" | "messages"): SearchHit[] {
  if (!value || typeof value !== "object") return [];
  const list = (value as any)?.data?.[key];
  return Array.isArray(list) ? list as SearchHit[] : [];
}

function explicitParentId(row: SearchHit): string | undefined {
  for (const value of [row.parent_l1_id, row.atomic_id, row.memory_id, row.source_memory_id, row.parent_id]) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return undefined;
}

export function toCandidates(snapshot: Snapshot): { candidates: L1Candidate[]; independentL0: import("../tdai/progressive-memory-allocator.js").L0Candidate[] } {
  const l1 = rows(snapshot.atomic_search, "items").flatMap((row) => {
    const id = typeof row.id === "string" ? row.id : "";
    const content = typeof row.content === "string" ? row.content : "";
    if (!id || !content) return [];
    const type = typeof row.type === "string" ? row.type : undefined;
    const score = typeof row.score === "number" ? row.score : undefined;
    const fromAgentId = typeof row.source_agent_id === "string" ? row.source_agent_id : undefined;
    const fromAgentName = typeof row.source_agent_name === "string" ? row.source_agent_name : undefined;
    const rendered = `[L1:${type ?? "memory"}] ${content}`;
    return [{ id, content, type, score, fromAgentId, fromAgentName, tokenCount: estimateTextTokens(rendered), l0: [] }];
  });

  const byId = new Map(l1.map((candidate) => [candidate.id, candidate]));
  const independentL0: import("../tdai/progressive-memory-allocator.js").L0Candidate[] = [];
  rows(snapshot.conversation_search, "messages").forEach((row, retrievalIndex) => {
    const id = typeof row.id === "string" ? row.id : "";
    const content = typeof row.content === "string" ? row.content : "";
    if (!id || !content) return;
    const parentL1Id = explicitParentId(row);
    const chunk = {
      id,
      content,
      score: typeof row.score === "number" ? row.score : undefined,
      role: typeof row.role === "string" ? row.role : undefined,
      sessionId: typeof row.session_id === "string" ? row.session_id : undefined,
      tokenCount: estimateTextTokens(`[L0] ${content}`),
      retrievalIndex,
      parentL1Id,
    };
    const parent = parentL1Id ? byId.get(parentL1Id) : undefined;
    if (parent) parent.l0.push(chunk);
    else independentL0.push(chunk);
  });
  return { candidates: l1, independentL0 };
}

export function prependToLatestUser(messages: any[], block: string): any[] {
  if (!block) return messages;
  const cloned = [...messages];
  for (let i = cloned.length - 1; i >= 0; i -= 1) {
    if (cloned[i]?.role !== "user") continue;
    const copy = { ...cloned[i] };
    if (typeof copy.content === "string") copy.content = `${block}\n\n${copy.content}`;
    else if (Array.isArray(copy.content)) copy.content = [{ type: "text", text: block }, ...copy.content];
    else copy.content = `${block}\n\n${textOfContent(copy.content)}`;
    cloned[i] = copy;
    break;
  }
  return cloned;
}

export default function tdaiBudgetOverride(pi: ExtensionAPI): void {
  const actionPath = process.env.PI_BRANCH_OUT_ACTION_FILE;
  const snapshotPath = process.env.PI_BRANCH_OUT_RECALL_SNAPSHOT;
  if (!actionPath || !snapshotPath || !existsSync(actionPath) || !existsSync(snapshotPath)) return;

  const action = JSON.parse(readFileSync(actionPath, "utf8")) as Record<string, unknown>;
  const ratio = Number(action.budget_ratio);
  if (action.kind !== "memory_budget_ratio" || !Number.isFinite(ratio) || ratio < 0 || ratio > 1) {
    throw new Error("invalid branch budget action");
  }
  const snapshotText = readFileSync(snapshotPath, "utf8");
  const snapshot = JSON.parse(snapshotText) as Snapshot;
  const { candidates, independentL0 } = toCandidates(snapshot);
  const snapshotFingerprint = createHash("sha256").update(snapshotText).digest("hex");
  const observationPath = process.env.PI_BRANCH_OUT_OBSERVATION_FILE;
  let applied = false;

  pi.on("context", async (event: any, ctx: any) => {
    if (applied || ctx?.model?.provider !== "tdai") return;
    const messages = Array.isArray(event?.messages) ? event.messages : [];
    const currentContextTokens = estimateContextTokens(event);
    const countRenderedTokens = (block: string): number => {
      if (!block) return 0;
      return Math.max(0, estimateMessagesTokens(prependToLatestUser(messages, block)) - estimateMessagesTokens(messages));
    };
    const fullAllocation = allocateProgressiveMemory({
      candidates,
      independentL0,
      budgetTokens: Number.MAX_SAFE_INTEGER,
      countRenderedTokens,
    });
    const candidateTokens = fullAllocation.injectedTokens;
    const contextWindow = Number(ctx?.model?.contextWindow ?? 524288);
    const reserveRaw = Number(process.env.TDAI_MEMORY_RESERVE_TOKENS ?? 16384);
    const hardCapRaw = Number(process.env.TDAI_MEMORY_HARD_CAP_TOKENS ?? 0);
    const decision = decideMemoryBudget({
      branchRatio: ratio,
      contextWindowTokens: Number.isFinite(contextWindow) && contextWindow > 0 ? contextWindow : 524288,
      currentContextTokens,
      reserveTokens: Number.isFinite(reserveRaw) && reserveRaw >= 0 ? reserveRaw : 16384,
      candidateTokens,
      hardCapTokens: Number.isFinite(hardCapRaw) && hardCapRaw > 0 ? hardCapRaw : null,
    });
    const allocation = allocateProgressiveMemory({
      candidates,
      independentL0,
      budgetTokens: decision.budgetTokens,
      countRenderedTokens,
    });
    const block = renderProgressiveMemory(allocation);
    const injectedTokens = allocation.injectedTokens;
    if (injectedTokens > decision.budgetTokens) throw new Error("rendered injection exceeded the selected budget");
    const contentSha256 = createHash("sha256").update(block).digest("hex");
    const observation = {
      kind: "memory_budget_ratio",
      requested_ratio: decision.requestedRatio,
      applied_ratio: decision.appliedRatio,
      candidate_tokens: candidateTokens,
      feasible_budget_tokens: decision.feasibleBudgetTokens,
      budget_tokens: decision.budgetTokens,
      injected_tokens: injectedTokens,
      injected_content_sha256: contentSha256,
      effective_action_id: `sha256:${contentSha256}`,
      l1_ids: allocation.selected.map((item) => item.id),
      l0_ids: [
        ...allocation.selected.flatMap((item) => item.selectedL0.map((chunk) => chunk.id)),
        ...allocation.selectedIndependentL0.map((chunk) => chunk.id),
      ],
      independent_l0_ids: allocation.selectedIndependentL0.map((chunk) => chunk.id),
      snapshot_id: `sha256:${snapshotFingerprint}`,
      tokenizer_version: "tdai-estimator-v2-complete-render",
      context_tokens_before_injection: currentContextTokens,
      context_tokens_after_injection: currentContextTokens + injectedTokens,
      adapter: "pi-frozen-recall-context-hook",
      model_call_index: snapshot.model_call_index ?? null,
    };
    if (observationPath) writeFileSync(observationPath, `${JSON.stringify(observation, null, 2)}\n`, "utf8");
    applied = true;
    if (!block) return;
    return { messages: prependToLatestUser(messages, block) };
  });
}
