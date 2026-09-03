import { writeFileSync } from "node:fs";

export interface BranchBudgetOverride {
	requestedRatio: number;
	oneShot: boolean;
}

export interface BranchBudgetObservation {
	requestedRatio: number;
	appliedRatio: number;
	feasibleBudgetTokens: number;
	budgetTokens: number;
	injectedTokens: number;
	l1Ids?: string[];
	l0Ids?: string[];
}

let consumed = false;

/**
 * Read the one-shot counterfactual Memory budget requested by pi-branch-out.
 *
 * Copy/import this helper from the TDAI recall path before the normal learned
 * budget policy is evaluated. When it returns a value, that value must override
 * the policy for this recall only.
 */
export function consumeBranchBudgetOverride(): BranchBudgetOverride | null {
	if (consumed) return null;
	const raw = process.env.TDAI_MEMORY_BUDGET_RATIO_OVERRIDE;
	if (raw == null || raw.trim() === "") return null;
	const ratio = Number(raw);
	if (!Number.isFinite(ratio) || ratio < 0 || ratio > 1) {
		throw new Error(`Invalid TDAI_MEMORY_BUDGET_RATIO_OVERRIDE=${raw}`);
	}
	const oneShot = process.env.TDAI_MEMORY_BUDGET_OVERRIDE_ONE_SHOT !== "0";
	if (oneShot) consumed = true;
	return { requestedRatio: ratio, oneShot };
}

/**
 * Emit proof that TDAI actually applied the requested branch action.
 * The Harbor branch runner reads this file after Pi exits and rejects the
 * branch when the observation is missing or mismatched.
 */
export function writeBranchBudgetObservation(observation: BranchBudgetObservation): void {
	const path = process.env.TDAI_BRANCH_OUT_OBSERVATION_FILE;
	if (!path) return;
	const payload = {
		kind: "memory_budget_ratio",
		requested_ratio: observation.requestedRatio,
		applied_ratio: observation.appliedRatio,
		feasible_budget_tokens: Math.max(0, Math.floor(observation.feasibleBudgetTokens)),
		budget_tokens: Math.max(0, Math.floor(observation.budgetTokens)),
		injected_tokens: Math.max(0, Math.floor(observation.injectedTokens)),
		l1_ids: observation.l1Ids ?? [],
		l0_ids: observation.l0Ids ?? [],
	};
	writeFileSync(path, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}
