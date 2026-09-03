export type BudgetRatioSource = "branch" | "policy" | "baseline";

export interface MemoryBudgetInput {
  branchRatio?: number | null;
  policyRatio?: number | null;
  baselineRatio?: number;
  contextWindowTokens: number;
  currentContextTokens: number;
  reserveTokens: number;
  /** Total token cost of the frozen L1/L0 candidate pool. */
  candidateTokens: number;
  /** Optional operator safety cap. */
  hardCapTokens?: number | null;
}

export interface MemoryBudgetDecision {
  source: BudgetRatioSource;
  requestedRatio: number;
  appliedRatio: number;
  contextWindowTokens: number;
  currentContextTokens: number;
  reserveTokens: number;
  candidateTokens: number;
  headroomTokens: number;
  hardCapTokens: number | null;
  feasibleBudgetTokens: number;
  budgetTokens: number;
}

function nonNegativeInt(name: string, value: number): number {
  if (!Number.isFinite(value) || value < 0) throw new Error(`${name} must be finite and >= 0, got ${value}`);
  return Math.floor(value);
}

export function validateBudgetRatio(name: string, value: number): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${name} must be within [0, 1], got ${value}`);
  }
  return value;
}

export function resolveBudgetRatio(input: Pick<MemoryBudgetInput, "branchRatio" | "policyRatio" | "baselineRatio">): {
  source: BudgetRatioSource;
  ratio: number;
} {
  if (input.branchRatio != null) return { source: "branch", ratio: validateBudgetRatio("branchRatio", input.branchRatio) };
  if (input.policyRatio != null) return { source: "policy", ratio: validateBudgetRatio("policyRatio", input.policyRatio) };
  return { source: "baseline", ratio: validateBudgetRatio("baselineRatio", input.baselineRatio ?? 0) };
}

/**
 * Convert a budget-ratio action into a per-turn token cap.
 *
 * The feasible budget is candidate-aware. This prevents a large model context
 * window from collapsing multiple ratio actions into the exact same injected
 * memory when the actual recall pool is much smaller.
 */
export function decideMemoryBudget(input: MemoryBudgetInput): MemoryBudgetDecision {
  const contextWindowTokens = nonNegativeInt("contextWindowTokens", input.contextWindowTokens);
  const currentContextTokens = nonNegativeInt("currentContextTokens", input.currentContextTokens);
  const reserveTokens = nonNegativeInt("reserveTokens", input.reserveTokens);
  const candidateTokens = nonNegativeInt("candidateTokens", input.candidateTokens);
  if (contextWindowTokens === 0) throw new Error("contextWindowTokens must be > 0");

  const { source, ratio } = resolveBudgetRatio(input);
  const headroomTokens = Math.max(0, contextWindowTokens - currentContextTokens - reserveTokens);
  let hardCapTokens: number | null = null;
  if (input.hardCapTokens != null && input.hardCapTokens > 0) hardCapTokens = nonNegativeInt("hardCapTokens", input.hardCapTokens);

  let feasibleBudgetTokens = Math.min(headroomTokens, candidateTokens);
  if (hardCapTokens != null) feasibleBudgetTokens = Math.min(feasibleBudgetTokens, hardCapTokens);
  const budgetTokens = Math.floor(feasibleBudgetTokens * ratio);

  return {
    source,
    requestedRatio: ratio,
    appliedRatio: ratio,
    contextWindowTokens,
    currentContextTokens,
    reserveTokens,
    candidateTokens,
    headroomTokens,
    hardCapTokens,
    feasibleBudgetTokens,
    budgetTokens,
  };
}
