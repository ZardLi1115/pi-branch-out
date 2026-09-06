import { createHash } from "node:crypto";
import {
  allocateProgressiveMemory,
  type AllocationResult,
  type L1Candidate,
} from "./progressive-memory-allocator.js";
import { decideMemoryBudget } from "./memory-budget-controller.js";

export const LONGMEMEVAL_ACTION_RATIOS = [0, 0.2, 0.4, 0.6, 0.8, 1] as const;

export interface LongMemEvalL1Item {
  id: string;
  type?: string;
  content: string;
  score?: number;
}

export interface LongMemEvalSceneEntry {
  path: string;
  created_at?: string;
  updated_at?: string;
}

export interface LongMemEvalActionPlan {
  action: number;
  budgetTokens: number;
  injectedTokens: number;
  selectedL1Ids: string[];
  renderedMemory: string;
  contentSha256: string;
  effectiveActionId: string;
}

export function estimateLongMemEvalTokens(text: string): number {
  if (!text) return 0;
  let cjk = 0;
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if ((cp >= 0x3400 && cp <= 0x9fff) || (cp >= 0xf900 && cp <= 0xfaff)) cjk += 1;
  }
  return Math.max(1, Math.ceil(cjk / 1.7 + Math.max(0, text.length - cjk) / 4));
}

/** Mirrors v2.0.0-beta.1 MemoryCore/openclaw-plugin/src/format.ts. */
export function renderOpenClawL1(result: AllocationResult): string {
  if (result.selected.length === 0) return "";
  const lines = ["<relevant-memories>", ""];
  for (const item of result.selected) {
    const typeTag = item.type ? `[${item.type}]` : "";
    lines.push(`- ${typeTag} ${item.content}`);
  }
  lines.push("", "</relevant-memories>");
  return lines.join("\n");
}

/** Persona and scene navigation are fixed state, never controlled by the L1 action. */
export function renderOpenClawStableContext(
  persona: string | null,
  scenes: LongMemEvalSceneEntry[],
): string {
  const parts: string[] = [];
  if (persona) parts.push(`<user-persona>\n${persona}\n</user-persona>`);
  if (scenes.length > 0 && (!persona || !persona.includes("Scene Navigation"))) {
    parts.push([
      "## 🗺️ Scene Navigation",
      "*以下是当前场景记忆索引。*",
      "",
      ...scenes.map((scene) => `- \`${scene.path}\``),
    ].join("\n"));
  }
  return parts.join("\n\n");
}

export function planLongMemEvalActions(args: {
  items: LongMemEvalL1Item[];
  ratios?: readonly number[];
  hardCapTokens?: number | null;
  contextWindowTokens?: number;
  currentContextTokens?: number;
  reserveTokens?: number;
}): { candidateTokens: number; feasibleBudgetTokens: number; plans: LongMemEvalActionPlan[] } {
  const candidates: L1Candidate[] = args.items.map((item) => ({
    ...item,
    tokenCount: estimateLongMemEvalTokens(`- ${item.type ? `[${item.type}] ` : ""}${item.content}`),
    l0: [],
  }));
  const countRenderedTokens = (rendered: string) => estimateLongMemEvalTokens(rendered);
  const full = allocateProgressiveMemory({
    candidates,
    budgetTokens: Number.MAX_SAFE_INTEGER,
    countRenderedTokens,
    renderResult: renderOpenClawL1,
  });
  const candidateTokens = full.injectedTokens;
  let feasibleBudgetTokens = 0;
  const plans = (args.ratios ?? LONGMEMEVAL_ACTION_RATIOS).map((action) => {
    const decision = decideMemoryBudget({
      branchRatio: action,
      contextWindowTokens: args.contextWindowTokens ?? 524288,
      currentContextTokens: args.currentContextTokens ?? 0,
      reserveTokens: args.reserveTokens ?? 16384,
      candidateTokens,
      hardCapTokens: args.hardCapTokens ?? null,
    });
    feasibleBudgetTokens = decision.feasibleBudgetTokens;
    const allocation = allocateProgressiveMemory({
      candidates,
      budgetTokens: decision.budgetTokens,
      countRenderedTokens,
      renderResult: renderOpenClawL1,
    });
    const renderedMemory = renderOpenClawL1(allocation);
    const contentSha256 = createHash("sha256").update(renderedMemory).digest("hex");
    return {
      action,
      budgetTokens: decision.budgetTokens,
      injectedTokens: allocation.injectedTokens,
      selectedL1Ids: allocation.selected.map((item) => item.id),
      renderedMemory,
      contentSha256,
      effectiveActionId: `sha256:${contentSha256}`,
    };
  });
  return { candidateTokens, feasibleBudgetTokens, plans };
}
