import { createHash } from "node:crypto";
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import {
  estimateContextTokens,
  estimateMessagesTokens,
  prependToLatestUser,
  textOfContent,
  toCandidates,
  type Snapshot,
} from "./tdai-budget-override.js";
import { decideMemoryBudget } from "../tdai/memory-budget-controller.js";
import { allocateProgressiveMemory, renderProgressiveMemory } from "../tdai/progressive-memory-allocator.js";

type ExtensionAPI = { on(event: string, callback: (event: any, ctx: any) => unknown): void };
export type Policy = {
  schema_version: number;
  feature_version: string;
  hash_dim: number;
  actions: number[];
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
};

const NUMERIC_KEYS = [
  "context_tokens", "context_window_tokens", "reserve_tokens",
  "remaining_call_budget", "remaining_cost_budget_usd", "remaining_time_seconds",
  "candidate_memory_tokens", "candidate_count", "l1_count", "l0_count",
  "default_actual_memory_tokens", "previous_actual_memory_tokens",
  "previous_mapped_action", "previous_budget_tokens",
] as const;

function rows(value: unknown, key: "items" | "messages"): Record<string, unknown>[] {
  const list = value && typeof value === "object" ? (value as any)?.data?.[key] : null;
  return Array.isArray(list) ? list : [];
}

function queryFromMessages(messages: any[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role !== "user" && messages[index]?.role !== "toolResult") continue;
    const text = textOfContent(messages[index]?.content).trim();
    if (text) return text.slice(0, 2048);
  }
  return "";
}

function recentToolResult(messages: any[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "toolResult") return textOfContent(messages[index]?.content).trim().slice(0, 8192);
  }
  return "";
}

async function bridgeSearch(conversationId: string, kind: string, query: string, limit: number): Promise<Record<string, unknown>> {
  const proxy = (process.env.TDAI_PROXY_URL ?? "http://127.0.0.1:8096").replace(/\/$/, "");
  const space = process.env.TDAI_SPACE_ID ?? "default";
  const keys = bridgeSessionKey(conversationId) === conversationId
    ? [conversationId]
    : [bridgeSessionKey(conversationId), conversationId];
  for (let index = 0; index < keys.length; index += 1) {
    const response = await fetch(`${proxy}/memory-bridge/v3/${kind}`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-conversation-id": keys[index], "x-tdai-service-id": space },
      body: JSON.stringify({ query, limit }),
    });
    const text = await response.text();
    if (response.ok) {
      const value = JSON.parse(text || "{}") as unknown;
      return value && typeof value === "object" ? value as Record<string, unknown> : {};
    }
    const coldMiss = response.status === 401 && text.includes("session not initialized");
    if (!coldMiss || index + 1 === keys.length) {
      throw new Error(`memory bridge ${kind} failed: HTTP ${response.status} ${text.slice(-1000)}`);
    }
  }
  throw new Error(`memory bridge ${kind} failed without a response`);
}

function bridgeSessionKey(conversationId: string): string {
  if (conversationId.includes(":")) return conversationId;
  if (process.env.TDAI_WIRE_API === "responses") return `codex:${conversationId}`;
  return conversationId;
}

function optionalFiniteEnv(name: string): number | null {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function hashedText(text: string, dimension: number): number[] {
  const result = Array(dimension).fill(0) as number[];
  const tokens = text.toLowerCase().match(/[A-Za-z0-9_-]+|[^\x00-\x7f]/g) ?? [];
  for (const token of tokens) {
    const digest = createHash("sha256").update(token).digest();
    const index = digest.readUInt32BE(0) % dimension;
    result[index] += digest[4] & 1 ? 1 : -1;
  }
  const norm = Math.sqrt(result.reduce((sum, value) => sum + value * value, 0)) || 1;
  return result.map((value) => value / norm);
}

export function policyFeatures(state: Record<string, unknown>, policy: Policy): number[] {
  const numeric = NUMERIC_KEYS.map((key) => {
    const raw = state[key];
    const value = typeof raw === "number" && Number.isFinite(raw) ? raw : 0;
    return Math.sign(value) * Math.log1p(Math.abs(value));
  });
  for (const key of ["l1_lengths", "l0_lengths", "l1_scores", "l0_scores"]) {
    const values = (Array.isArray(state[key]) ? state[key] : []).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    if (values.length) numeric.push(values.length, values.reduce((a, b) => a + b, 0) / values.length, Math.min(...values), Math.max(...values));
    else numeric.push(0, 0, 0, 0);
  }
  return numeric.concat(hashedText(`${String(state.query ?? "")}\n${String(state.recent_tool_result ?? "")}`, policy.hash_dim));
}

export function chooseRatio(policy: Policy, state: Record<string, unknown>): { ratio: number; qValues: number[] } {
  if (policy.feature_version !== "visible-state-hash-v3-history") throw new Error(`unsupported feature version ${policy.feature_version}`);
  const x = policyFeatures(state, policy);
  if (x.length !== policy.w1.length) throw new Error(`policy input mismatch: ${x.length} != ${policy.w1.length}`);
  const hidden = policy.b1.map((bias, column) => Math.max(0, bias + x.reduce((sum, value, row) => sum + value * policy.w1[row][column], 0)));
  const qValues = policy.b2.map((bias, column) => bias + hidden.reduce((sum, value, row) => sum + value * policy.w2[row][column], 0));
  let best = 0;
  for (let index = 1; index < qValues.length; index += 1) if (qValues[index] > qValues[best]) best = index;
  return { ratio: policy.actions[best], qValues };
}

function validatePolicy(policy: Policy): void {
  const finite = (values: number[]) => values.every((value) => Number.isFinite(value));
  if (policy.schema_version !== 1 || policy.hash_dim !== 128) throw new Error("unsupported policy schema");
  if (!policy.actions.length || !finite(policy.actions) || policy.actions.some((value) => value < 0 || value > 1)) {
    throw new Error("policy actions must be finite ratios within [0, 1]");
  }
  const hidden = policy.b1.length;
  const output = policy.actions.length;
  if (!hidden || policy.w1.length !== 14 + 16 + policy.hash_dim) throw new Error("invalid policy input matrix");
  if (policy.w1.some((row) => row.length !== hidden || !finite(row)) || !finite(policy.b1)) throw new Error("invalid policy hidden matrix");
  if (policy.w2.length !== hidden || policy.w2.some((row) => row.length !== output || !finite(row))) throw new Error("invalid policy output matrix");
  if (policy.b2.length !== output || !finite(policy.b2)) throw new Error("invalid policy output bias");
}

export default function tdaiBudgetPolicy(pi: ExtensionAPI): void {
  const policyPath = process.env.PI_BRANCH_OUT_POLICY_FILE;
  if (!policyPath || !existsSync(policyPath)) return;
  const policyText = readFileSync(policyPath, "utf8");
  const policy = JSON.parse(policyText) as Policy;
  validatePolicy(policy);
  const policySha256 = createHash("sha256").update(policyText).digest("hex");
  let skipBranchDecision = Boolean(process.env.PI_BRANCH_OUT_ACTION_FILE);
  let policyCallIndex = Number(process.env.PI_BRANCH_OUT_CALL_OFFSET ?? 0);
  let lastActualMemoryTokens = 0;
  let lastMappedAction = 0;
  let lastBudgetTokens = 0;

  pi.on("context", async (event: any, ctx: any) => {
    if (ctx?.model?.provider !== "tdai") return;
    policyCallIndex += 1;
    if (skipBranchDecision) {
      const observationPath = process.env.PI_BRANCH_OUT_OBSERVATION_FILE;
      if (observationPath && existsSync(observationPath)) {
        const observation = JSON.parse(readFileSync(observationPath, "utf8")) as Record<string, unknown>;
        lastActualMemoryTokens = Number(observation.injected_tokens ?? 0);
        lastMappedAction = Number(observation.applied_ratio ?? observation.requested_ratio ?? 0);
        lastBudgetTokens = Number(observation.budget_tokens ?? 0);
      }
      skipBranchDecision = false;
      return;
    }
    if (policyCallIndex === 1) return;
    const started = Date.now();
    const messages = Array.isArray(event?.messages) ? event.messages : [];
    const sessionId = ctx?.sessionManager?.getSessionId?.();
    if (typeof sessionId !== "string" || !sessionId) throw new Error("policy recall requires a Pi session id");
    const query = queryFromMessages(messages);
    const [atomic, conversation] = await Promise.all([
      bridgeSearch(`pi-${sessionId}`, "atomic/search", query, 36),
      bridgeSearch(`pi-${sessionId}`, "conversation/search", query, 72),
    ]);
    const snapshot: Snapshot = { version: 1, query, atomic_search: atomic, conversation_search: conversation };
    const { candidates, independentL0 } = toCandidates(snapshot);
    const countRenderedTokens = (block: string) => block
      ? Math.max(0, estimateMessagesTokens(prependToLatestUser(messages, block)) - estimateMessagesTokens(messages))
      : 0;
    const full = allocateProgressiveMemory({ candidates, independentL0, budgetTokens: Number.MAX_SAFE_INTEGER, countRenderedTokens });
    const l1 = rows(atomic, "items");
    const l0 = rows(conversation, "messages");
    const contextTokens = estimateContextTokens(event);
    const contextWindow = Number(ctx?.model?.contextWindow ?? 524288);
    const reserveTokens = Number(process.env.TDAI_MEMORY_RESERVE_TOKENS ?? 16384);
    const state: Record<string, unknown> = {
      context_tokens: contextTokens,
      context_window_tokens: contextWindow,
      reserve_tokens: reserveTokens,
      remaining_call_budget: optionalFiniteEnv("PI_BRANCH_OUT_REMAINING_CALLS"),
      remaining_cost_budget_usd: optionalFiniteEnv("PI_BRANCH_OUT_REMAINING_COST_USD"),
      remaining_time_seconds: optionalFiniteEnv("PI_BRANCH_OUT_REMAINING_TIME_SECONDS"),
      candidate_memory_tokens: full.injectedTokens,
      candidate_count: l1.length + l0.length,
      l1_count: l1.length,
      l0_count: l0.length,
      default_actual_memory_tokens: 0,
      previous_actual_memory_tokens: lastActualMemoryTokens,
      previous_mapped_action: lastMappedAction,
      previous_budget_tokens: lastBudgetTokens,
      l1_lengths: l1.map((row) => [...String(row.content ?? "")].length),
      l0_lengths: l0.map((row) => [...String(row.content ?? "")].length),
      l1_scores: l1.map((row) => typeof row.score === "number" ? row.score : null),
      l0_scores: l0.map((row) => typeof row.score === "number" ? row.score : null),
      query,
      recent_tool_result: recentToolResult(messages),
    };
    const selected = chooseRatio(policy, state);
    const hardCap = Number(process.env.TDAI_MEMORY_HARD_CAP_TOKENS ?? 0);
    const decision = decideMemoryBudget({
      policyRatio: selected.ratio,
      contextWindowTokens: contextWindow,
      currentContextTokens: contextTokens,
      reserveTokens,
      candidateTokens: full.injectedTokens,
      hardCapTokens: Number.isFinite(hardCap) && hardCap > 0 ? hardCap : null,
    });
    const allocation = allocateProgressiveMemory({
      candidates, independentL0, budgetTokens: decision.budgetTokens, countRenderedTokens,
    });
    const block = renderProgressiveMemory(allocation);
    lastActualMemoryTokens = allocation.injectedTokens;
    lastMappedAction = selected.ratio;
    lastBudgetTokens = decision.budgetTokens;
    const observation = {
      schema_version: 1,
      timestamp: new Date().toISOString(),
      source: "policy",
      policy_version: process.env.PI_BRANCH_OUT_POLICY_VERSION ?? `sha256:${policySha256}`,
      requested_ratio: selected.ratio,
      q_values: selected.qValues,
      feasible_budget_tokens: decision.feasibleBudgetTokens,
      budget_tokens: decision.budgetTokens,
      injected_tokens: allocation.injectedTokens,
      injected_content_sha256: createHash("sha256").update(block).digest("hex"),
      latency_ms: Date.now() - started,
      model_call_index: policyCallIndex,
    };
    const outputRoot = process.env.PI_BRANCH_OUT_MODEL_CALL_DIR;
    if (outputRoot) {
      appendFileSync(join(outputRoot, "policy-observations.jsonl"), `${JSON.stringify(observation)}\n`, "utf8");
      writeFileSync(join(outputRoot, "latest-policy-observation.json"), `${JSON.stringify(observation)}\n`, "utf8");
    }
    if (!block) return;
    return { messages: prependToLatestUser(messages, block) };
  });
}
