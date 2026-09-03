import { existsSync, readFileSync } from "node:fs";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

type BranchAction = {
	kind?: string;
	budget_ratio?: number;
	one_shot?: boolean;
};

/**
 * Tiny bridge between the branch runner and a TDAI Pi adapter.
 *
 * The branch runner starts only the first counterfactual Harbor step with
 * PI_BRANCH_OUT_ACTION_FILE. Loading this extension copies the requested budget
 * ratio into TDAI_MEMORY_BUDGET_RATIO_OVERRIDE before the agent starts. The TDAI
 * adaptive-recall adapter should treat that variable as a one-step override of
 * its normal policy and log the realized budget/context for verification.
 */
export default function tdaiBudgetOverride(_pi: ExtensionAPI): void {
	const actionPath = process.env.PI_BRANCH_OUT_ACTION_FILE;
	if (!actionPath || !existsSync(actionPath)) return;

	const payload = JSON.parse(readFileSync(actionPath, "utf8")) as BranchAction;
	if (payload.kind !== "memory_budget_ratio") return;
	const ratio = Number(payload.budget_ratio);
	if (!Number.isFinite(ratio) || ratio < 0 || ratio > 1) {
		throw new Error(`Invalid branch memory budget ratio: ${payload.budget_ratio}`);
	}
	process.env.TDAI_MEMORY_BUDGET_RATIO_OVERRIDE = String(ratio);
	process.env.TDAI_MEMORY_BUDGET_OVERRIDE_ONE_SHOT = payload.one_shot === false ? "0" : "1";
}
