import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { decideMemoryBudget } from "../tdai/memory-budget-controller.js";
import { allocateProgressiveMemory, renderProgressiveMemory, type L1Candidate } from "../tdai/progressive-memory-allocator.js";

type ExtensionAPI = { on(event: string, callback: (event: any, ctx: any) => unknown): void };
type SearchHit = {
  id?: unknown; type?: unknown; content?: unknown; score?: unknown; role?: unknown;
  session_id?: unknown; source_agent_id?: unknown; source_agent_name?: unknown;
};
type Snapshot = {
  version: number;
  query: string;
  atomic_search: unknown;
  conversation_search: unknown;
  baseline?: Record<string, unknown>;
};

function textOfContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content == null ? "" : JSON.stringify(content);
  return content.map((part) => {
    if (typeof part === "string") return part;
    if (!part || typeof part !== "object") return "";
    const p = part as Record<string, unknown>;
    return typeof p.text === "string" ? p.text : typeof p.content === "string" ? p.content : "";
  }).filter(Boolean).join("\n");
}

function estimateTextTokens(text: string): number {
  if (!text) return 0;
  let cjk = 0;
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if ((cp >= 0x3400 && cp <= 0x9fff) || (cp >= 0xf900 && cp <= 0xfaff)) cjk += 1;
  }
  return Math.max(1, Math.ceil(cjk / 1.7 + Math.max(0, text.length - cjk) / 4));
}

function estimateMessagesTokens(messages: any[]): number {
  return messages.reduce((sum, m) => sum + estimateTextTokens(`${String(m?.role ?? "")}\n${textOfContent(m?.content)}`) + 4, 0);
}

function rows(value: unknown, key: "items" | "messages"): SearchHit[] {
  if (!value || typeof value !== "object") return [];
  const list = (value as any)?.data?.[key];
  return Array.isArray(list) ? list as SearchHit[] : [];
}

function toCandidates(snapshot: Snapshot): L1Candidate[] {
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

  const byAgent = new Map<string, L1Candidate[]>();
  for (const candidate of l1) {
    const key = candidate.fromAgentId ?? "__unknown__";
    const list = byAgent.get(key) ?? [];
    list.push(candidate);
    byAgent.set(key, list);
  }
  const next = new Map<string, number>();
  for (const row of rows(snapshot.conversation_search, "messages")) {
    const id = typeof row.id === "string" ? row.id : "";
    const content = typeof row.content === "string" ? row.content : "";
    if (!id || !content) continue;
    const key = typeof row.source_agent_id === "string" ? row.source_agent_id : "__unknown__";
    const targets = byAgent.get(key);
    if (!targets?.length) continue;
    const idx = next.get(key) ?? 0;
    const target = targets[idx % targets.length];
    next.set(key, idx + 1);
    target.l0.push({
      id,
      content,
      score: typeof row.score === "number" ? row.score : undefined,
      role: typeof row.role === "string" ? row.role : undefined,
      sessionId: typeof row.session_id === "string" ? row.session_id : undefined,
      tokenCount: estimateTextTokens(`[L0] ${content}`),
    });
  }
  return l1;
}

function prependToLatestUser(messages: any[], block: string): any[] {
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
  const snapshot = JSON.parse(readFileSync(snapshotPath, "utf8")) as Snapshot;
  const candidates = toCandidates(snapshot);
  const candidateTokens = candidates.reduce(
    (sum, item) => sum + item.tokenCount + item.l0.reduce((s, chunk) => s + chunk.tokenCount, 0),
    0,
  );
  const observationPath = process.env.PI_BRANCH_OUT_OBSERVATION_FILE;
  let applied = false;

  pi.on("context", async (event: any, ctx: any) => {
    if (applied || ctx?.model?.provider !== "tdai") return;
    const messages = Array.isArray(event?.messages) ? event.messages : [];
    const contextWindow = Number(ctx?.model?.contextWindow ?? 524288);
    const reserveRaw = Number(process.env.TDAI_MEMORY_RESERVE_TOKENS ?? 16384);
    const hardCapRaw = Number(process.env.TDAI_MEMORY_HARD_CAP_TOKENS ?? 0);
    const decision = decideMemoryBudget({
      branchRatio: ratio,
      contextWindowTokens: Number.isFinite(contextWindow) && contextWindow > 0 ? contextWindow : 524288,
      currentContextTokens: estimateMessagesTokens(messages),
      reserveTokens: Number.isFinite(reserveRaw) && reserveRaw >= 0 ? reserveRaw : 16384,
      candidateTokens,
      hardCapTokens: Number.isFinite(hardCapRaw) && hardCapRaw > 0 ? hardCapRaw : null,
    });
    const wrapperReserve = Math.min(64, decision.budgetTokens);
    const allocation = allocateProgressiveMemory({
      candidates,
      budgetTokens: Math.max(0, decision.budgetTokens - wrapperReserve),
    });
    const block = renderProgressiveMemory(allocation);
    const injectedTokens = block ? Math.min(decision.budgetTokens, estimateTextTokens(block)) : 0;
    const observation = {
      kind: "memory_budget_ratio",
      requested_ratio: decision.requestedRatio,
      applied_ratio: decision.appliedRatio,
      candidate_tokens: candidateTokens,
      feasible_budget_tokens: decision.feasibleBudgetTokens,
      budget_tokens: decision.budgetTokens,
      injected_tokens: injectedTokens,
      injected_content_sha256: createHash("sha256").update(block).digest("hex"),
      l1_ids: allocation.selected.map((item) => item.id),
      l0_ids: allocation.selected.flatMap((item) => item.selectedL0.map((chunk) => chunk.id)),
      snapshot_id: `${snapshot.version}:${snapshot.query}`,
      adapter: "pi-frozen-recall-context-hook",
    };
    if (observationPath) writeFileSync(observationPath, `${JSON.stringify(observation, null, 2)}\n`, "utf8");
    applied = true;
    if (!block) return;
    return { messages: prependToLatestUser(messages, block) };
  });
}
