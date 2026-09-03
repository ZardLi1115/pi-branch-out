export type BudgetRatioSource = "branch" | "policy" | "default";

export interface MemoryBudgetInput {
  /** Counterfactual override. Highest priority when present. */
  branchRatio?: number | null;
  /** Future CQL/Base Policy output. Not required for branch-out collection. */
  policyRatio?: number | null;
  /** Stable fallback when neither branch nor policy supplies a ratio. */
  defaultRatio?: number;

  /** Model context window for this request. */
  contextWindowTokens: number;
  /** Tokens already occupied before dynamic L1/L0 injection. */
  currentContextTokens: number;
  /** Reserved space for output, tool growth, and safety margin. */
  reserveTokens: number;
  /** Optional experiment/operator cap. Omit or <=0 for no additional cap. */
  hardCapTokens?: number | null;
}

export interface MemoryBudgetDecision {
  source: BudgetRatioSource;
  requestedRatio: number;
  appliedRatio: number;
  contextWindowTokens: number;
  currentContextTokens: number;
  reserveTokens: number;
  headroomTokens: number;
  hardCapTokens: number | null;
  feasibleBudgetTokens: number;
  budgetTokens: number;
}

function finiteNonNegativeInt(name: string, value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be a finite non-negative number, got ${value}`);
  }
  return Math.floor(value);
}

export function validateBudgetRatio(name: string, value: number): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${name} must be within [0, 1], got ${value}`);
  }
  return value;
}

export function resolveBudgetRatio(input: Pick<MemoryBudgetInput, "branchRatio" | "policyRatio" | "defaultRatio">): {
  source: BudgetRatioSource;
  ratio: number;
} {
  if (input.branchRatio != null) {
    return { source: "branch", ratio: validateBudgetRatio("branchRatio", input.branchRatio) };
  }
  if (input.policyRatio != null) {
    return { source: "policy", ratio: validateBudgetRatio("policyRatio", input.policyRatio) };
  }
  const fallback = input.defaultRatio ?? 1;
  return { source: "default", ratio: validateBudgetRatio("defaultRatio", fallback) };
}

/**
 * Convert a ratio action into a token cap for this turn.
 *
 * The controller deliberately does not try to "fill" the budget. It only
 * computes a maximum. The allocator may inject fewer tokens when there are not
 * enough useful candidates or when the next complete L1 atom does not fit.
 */
export function decideMemoryBudget(input: MemoryBudgetInput): MemoryBudgetDecision {
  const contextWindowTokens = finiteNonNegativeInt("contextWindowTokens", input.contextWindowTokens);
  const currentContextTokens = finiteNonNegativeInt("currentContextTokens", input.currentContextTokens);
  const reserveTokens = finiteNonNegativeInt("reserveTokens", input.reserveTokens);

  if (contextWindowTokens === 0) {
    throw new Error("contextWindowTokens must be greater than zero");
  }

  const { source, ratio } = resolveBudgetRatio(input);
  const headroomTokens = Math.max(0, contextWindowTokens - currentContextTokens - reserveTokens);

  let hardCapTokens: number | null = null;
  if (input.hardCapTokens != null && input.hardCapTokens > 0) {
    hardCapTokens = finiteNonNegativeInt("hardCapTokens", input.hardCapTokens);
  }

  const feasibleBudgetTokens = hardCapTokens == null
    ? headroomTokens
    : Math.min(headroomTokens, hardCapTokens);
  const budgetTokens = Math.floor(feasibleBudgetTokens * ratio);

  return {
    source,
    requestedRatio: ratio,
    appliedRatio: ratio,
    contextWindowTokens,
    currentContextTokens,
    reserveTokens,
    headroomTokens,
    hardCapTokens,
    feasibleBudgetTokens,
    budgetTokens,
  };
}
