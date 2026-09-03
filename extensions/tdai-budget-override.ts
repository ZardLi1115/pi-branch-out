import { existsSync, readFileSync } from "node:fs";

type ExtensionAPI = {
	on(event: string, callback: (event: any, ctx: any) => unknown): void;
};

type BranchAction = {
	kind?: string;
	budget_ratio?: number;
	granularity?: "compact" | "standard" | "detailed";
	one_shot?: boolean;
};

/**
 * Bridge between the Harbor branch runner and both TDAI deployment modes:
 *
 * 1. in-process TDAI can read the exported environment variables;
 * 2. Pi -> MemoryProxy forwards the same action through request headers.
 *
 * The one-shot flag is enforced here for Proxy traffic, so only the first
 * provider request of the counterfactual Harbor step receives dynamic memory.
 */
export default function tdaiBudgetOverride(pi: ExtensionAPI): void {
	const actionPath = process.env.PI_BRANCH_OUT_ACTION_FILE;
	if (!actionPath || !existsSync(actionPath)) return;

	const payload = JSON.parse(readFileSync(actionPath, "utf8")) as BranchAction;
	if (payload.kind !== "memory_budget_ratio") return;
	const ratio = Number(payload.budget_ratio);
	if (!Number.isFinite(ratio) || ratio < 0 || ratio > 1) {
		throw new Error(`Invalid branch memory budget ratio: ${payload.budget_ratio}`);
	}
	const granularity = payload.granularity ?? "standard";
	if (!["compact", "standard", "detailed"].includes(granularity)) {
		throw new Error(`Invalid branch memory granularity: ${payload.granularity}`);
	}

	const oneShot = payload.one_shot !== false;
	const observationId = process.env.TDAI_BRANCH_OUT_OBSERVATION_ID ?? "";
	process.env.TDAI_MEMORY_BUDGET_RATIO_OVERRIDE = String(ratio);
	process.env.TDAI_MEMORY_GRANULARITY_OVERRIDE = granularity;
	process.env.TDAI_MEMORY_BUDGET_OVERRIDE_ONE_SHOT = oneShot ? "1" : "0";

	let sent = false;
	pi.on("before_provider_headers", (event: any, ctx: any) => {
		if (ctx?.model?.provider !== "tdai") return;
		if (oneShot && sent) return;
		event.headers ??= {};
		event.headers["x-tdai-memory-budget-ratio"] = String(ratio);
		event.headers["x-tdai-memory-granularity"] = granularity;
		event.headers["x-tdai-memory-budget-one-shot"] = oneShot ? "1" : "0";
		if (observationId) {
			event.headers["x-tdai-branch-observation-id"] = observationId;
		}
		const contextWindow = Number(ctx?.model?.contextWindow ?? 0);
		if (Number.isFinite(contextWindow) && contextWindow > 0) {
			event.headers["x-tdai-context-window-tokens"] = String(Math.floor(contextWindow));
		}
		sent = true;
	});
}
