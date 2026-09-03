import { writeFileSync } from "node:fs";

export type MemoryGranularity = "compact" | "standard" | "detailed";

export interface BranchBudgetOverride {
	requestedRatio: number;
	granularity: MemoryGranularity;
	oneShot: boolean;
}

export interface BranchBudgetObservation {
	requestedRatio: number;
	appliedRatio: number;
	granularity: MemoryGranularity;
	feasibleBudgetTokens: number;
	budgetTokens: number;
	injectedTokens: number;
	l1Ids?: string[];
	l0Ids?: string[];
}

let consumed = false;

/**
 * In-process TDAI fallback for deployments that do not use MemoryProxy.
 *
 * The current official Pi integration goes through MemoryProxy and therefore
 * uses explicit HTTP headers instead. Both paths expose the same ratio +
 * granularity action semantics.
 */
export function consumeBranchBudgetOverride(): BranchBudgetOverride | null {
	if (consumed) return null;
	const raw = process.env.TDAI_MEMORY_BUDGET_RATIO_OVERRIDE;
	if (raw == null || raw.trim() === "") return null;
	const ratio = Number(raw);
	if (!Number.isFinite(ratio) || ratio < 0 || ratio > 1) {
		throw new Error(`Invalid TDAI_MEMORY_BUDGET_RATIO_OVERRIDE=${raw}`);
	}
	const granularityRaw = process.env.TDAI_MEMORY_GRANULARITY_OVERRIDE ?? "standard";
	if (!["compact", "standard", "detailed"].includes(granularityRaw)) {
		throw new Error(`Invalid TDAI_MEMORY_GRANULARITY_OVERRIDE=${granularityRaw}`);
	}
	const oneShot = process.env.TDAI_MEMORY_BUDGET_OVERRIDE_ONE_SHOT !== "0";
	if (oneShot) consumed = true;
	return {
		requestedRatio: ratio,
		granularity: granularityRaw as MemoryGranularity,
		oneShot,
	};
}

/**
 * Emit proof that in-process TDAI actually applied the requested branch action.
 */
export function writeBranchBudgetObservation(observation: BranchBudgetObservation): void {
	const path = process.env.TDAI_BRANCH_OUT_OBSERVATION_FILE;
	if (!path) return;
	const payload = {
		kind: "memory_budget_ratio",
		requested_ratio: observation.requestedRatio,
		applied_ratio: observation.appliedRatio,
		granularity: observation.granularity,
		feasible_budget_tokens: Math.max(0, Math.floor(observation.feasibleBudgetTokens)),
		budget_tokens: Math.max(0, Math.floor(observation.budgetTokens)),
		injected_tokens: Math.max(0, Math.floor(observation.injectedTokens)),
		l1_ids: observation.l1Ids ?? [],
		l0_ids: observation.l0Ids ?? [],
	};
	writeFileSync(path, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}
