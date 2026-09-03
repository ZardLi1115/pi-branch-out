import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { decideMemoryBudget } from "../tdai/memory-budget-controller.js";
import {
	allocateProgressiveMemory,
	renderProgressiveMemory,
	type L1Candidate,
	type MemoryGranularity,
} from "../tdai/progressive-memory-allocator.js";

type ExtensionAPI = {
	on(event: string, callback: (event: any, ctx: any) => unknown): void;
};

type BranchAction = {
	kind?: string;
	budget_ratio?: number;
	granularity?: MemoryGranularity;
	one_shot?: boolean;
};

type SearchHit = {
	id?: unknown;
	type?: unknown;
	content?: unknown;
	score?: unknown;
	role?: unknown;
	session_id?: unknown;
	source_agent_id?: unknown;
	source_agent_name?: unknown;
};

type CachedRecall = {
	query: string;
	block: string;
	observation: Record<string, unknown>;
};

function textOfContent(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return content == null ? "" : JSON.stringify(content);
	return content.map((part) => {
		if (typeof part === "string") return part;
		if (!part || typeof part !== "object") return "";
		const p = part as Record<string, unknown>;
		if (typeof p.text === "string") return p.text;
		if (typeof p.content === "string") return p.content;
		return "";
	}).filter(Boolean).join("\n");
}

function lastUserQuery(messages: any[]): string {
	for (let i = messages.length - 1; i >= 0; i -= 1) {
		const m = messages[i];
		if (m?.role !== "user") continue;
		return textOfContent(m.content).trim().slice(0, 2048);
	}
	return "";
}

function estimateTextTokens(text: string): number {
	if (!text) return 0;
	let cjk = 0;
	for (const ch of text) {
		const cp = ch.codePointAt(0) ?? 0;
		if ((cp >= 0x3400 && cp <= 0x9fff) || (cp >= 0xf900 && cp <= 0xfaff)) cjk += 1;
	}
	const rest = Math.max(0, text.length - cjk);
	return Math.max(1, Math.ceil(cjk / 1.7 + rest / 4));
}

function estimateMessagesTokens(messages: any[]): number {
	return messages.reduce((sum, message) => {
		return sum + estimateTextTokens(`${String(message?.role ?? "")}\n${textOfContent(message?.content)}`) + 4;
	}, 0);
}

function parseEnvelopeRows(value: unknown, key: "items" | "messages"): SearchHit[] {
	if (!value || typeof value !== "object") return [];
	const data = (value as any).data;
	const rows = data?.[key];
	return Array.isArray(rows) ? rows as SearchHit[] : [];
}

async function memoryBridgeSearch(
	proxyBase: string,
	spaceId: string,
	conversationId: string,
	kind: "atomic/search" | "conversation/search",
	query: string,
	limit: number,
): Promise<SearchHit[]> {
	const response = await fetch(`${proxyBase.replace(/\/$/, "")}/memory-bridge/v3/${kind}`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"x-conversation-id": conversationId,
			"x-tdai-service-id": spaceId,
		},
		body: JSON.stringify({ query, limit }),
	});
	if (!response.ok) {
		throw new Error(`TDAI memory bridge ${kind} failed: HTTP ${response.status}`);
	}
	const json = await response.json();
	return parseEnvelopeRows(json, kind === "atomic/search" ? "items" : "messages");
}

function toL1Candidates(l1Rows: SearchHit[]): L1Candidate[] {
	return l1Rows.flatMap((row) => {
		const id = typeof row.id === "string" ? row.id : "";
		const content = typeof row.content === "string" ? row.content : "";
		if (!id || !content) return [];
		const type = typeof row.type === "string" ? row.type : undefined;
		const fromAgentId = typeof row.source_agent_id === "string" ? row.source_agent_id : undefined;
		const fromAgentName = typeof row.source_agent_name === "string" ? row.source_agent_name : undefined;
		const score = typeof row.score === "number" ? row.score : undefined;
		const rendered = `[L1:${type ?? "memory"}${score == null ? "" : ` score=${score.toFixed(3)}`}] ${content}`;
		return [{
			id,
			content,
			type,
			score,
			fromAgentId,
			fromAgentName,
			tokenCount: estimateTextTokens(rendered),
			l0: [],
		}];
	});
}

function attachL0Pool(candidates: L1Candidate[], l0Rows: SearchHit[]): void {
	const byAgent = new Map<string, L1Candidate[]>();
	for (const candidate of candidates) {
		const key = candidate.fromAgentId ?? "__unknown__";
		const list = byAgent.get(key) ?? [];
		list.push(candidate);
		byAgent.set(key, list);
	}
	const nextIndex = new Map<string, number>();
	for (const row of l0Rows) {
		const id = typeof row.id === "string" ? row.id : "";
		const content = typeof row.content === "string" ? row.content : "";
		if (!id || !content) continue;
		const agentKey = typeof row.source_agent_id === "string" ? row.source_agent_id : "__unknown__";
		const targets = byAgent.get(agentKey);
		if (!targets?.length) continue;
		const idx = nextIndex.get(agentKey) ?? 0;
		const target = targets[idx % targets.length];
		nextIndex.set(agentKey, idx + 1);
		const score = typeof row.score === "number" ? row.score : undefined;
		const role = typeof row.role === "string" ? row.role : undefined;
		const rendered = `[L0${role ? ` role=${role}` : ""}${score == null ? "" : ` score=${score.toFixed(3)}`}] ${content}`;
		target.l0.push({
			id,
			content,
			score,
			role,
			sessionId: typeof row.session_id === "string" ? row.session_id : undefined,
			tokenCount: estimateTextTokens(rendered),
		});
	}
}

function prependMemoryToLatestUser(messages: any[], block: string): any[] {
	if (!block) return messages;
	const cloned = [...messages];
	for (let i = cloned.length - 1; i >= 0; i -= 1) {
		const message = cloned[i];
		if (message?.role !== "user") continue;
		const copy = { ...message };
		if (typeof copy.content === "string") {
			copy.content = `${block}\n\n${copy.content}`;
		} else if (Array.isArray(copy.content)) {
			copy.content = [{ type: "text", text: block }, ...copy.content];
		} else {
			copy.content = `${block}\n\n${textOfContent(copy.content)}`;
		}
		cloned[i] = copy;
		break;
	}
	return cloned;
}

function writeObservation(path: string | undefined, observation: Record<string, unknown>): void {
	if (!path) return;
	writeFileSync(path, `${JSON.stringify(observation, null, 2)}\n`, "utf8");
}

/**
 * External-only TDAI adapter for branch-out.
 *
 * IMPORTANT: this extension does not patch MemoryCore or MemoryProxy. It calls
 * the existing read-only memory-bridge APIs, applies the Budget Controller and
 * progressive L1/L0 allocator locally, then modifies Pi's transient context.
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

	const proxyBase = process.env.TDAI_PROXY_URL ?? "http://127.0.0.1:8096";
	const spaceId = process.env.TDAI_SPACE_ID ?? "default";
	const observationPath = process.env.TDAI_BRANCH_OUT_OBSERVATION_FILE;
	const hardCapRaw = Number(process.env.TDAI_MEMORY_HARD_CAP_TOKENS ?? 0);
	const reserveRaw = Number(process.env.TDAI_MEMORY_RESERVE_TOKENS ?? 16384);
	let cached: CachedRecall | null = null;

	pi.on("context", async (event: any, ctx: any) => {
		if (ctx?.model?.provider !== "tdai") return;
		const messages = Array.isArray(event?.messages) ? event.messages : [];
		const query = lastUserQuery(messages);
		if (!query) return;

		if (!cached || cached.query !== query) {
			const sessionId = ctx?.sessionManager?.getSessionId?.();
			if (!sessionId) throw new Error("Pi session id unavailable for TDAI memory bridge recall");
			const conversationId = `pi-${sessionId}`;
			const contextWindow = Number(ctx?.model?.contextWindow ?? process.env.TDAI_CONTEXT_WINDOW_TOKENS ?? 524288);
			const currentContextTokens = estimateMessagesTokens(messages);
			const reserveTokens = Number.isFinite(reserveRaw) && reserveRaw >= 0 ? Math.floor(reserveRaw) : 16384;
			const decision = decideMemoryBudget({
				branchRatio: ratio,
				contextWindowTokens: contextWindow,
				currentContextTokens,
				reserveTokens,
				hardCapTokens: Number.isFinite(hardCapRaw) && hardCapRaw > 0 ? hardCapRaw : null,
			});

			const l1Rows = await memoryBridgeSearch(proxyBase, spaceId, conversationId, "atomic/search", query, 36);
			const candidates = toL1Candidates(l1Rows);
			const wrapperReserve = Math.min(decision.budgetTokens, 64);
			const allocBudget = Math.max(0, decision.budgetTokens - wrapperReserve);
			const l1Only = allocateProgressiveMemory({ candidates, budgetTokens: allocBudget, granularity: "compact" });
			const admittedIds = new Set(l1Only.selected.map((item) => item.id));
			const admitted = candidates.filter((item) => admittedIds.has(item.id));

			if (granularity !== "compact" && admitted.length > 0) {
				const l0Rows = await memoryBridgeSearch(proxyBase, spaceId, conversationId, "conversation/search", query, 72);
				attachL0Pool(admitted, l0Rows);
			}

			const allocation = allocateProgressiveMemory({ candidates, budgetTokens: allocBudget, granularity });
			const block = renderProgressiveMemory(allocation);
			const renderedTokens = block ? estimateTextTokens(block) : 0;
			const injectedTokens = Math.min(decision.budgetTokens, renderedTokens);
			const observation = {
				kind: "memory_budget_ratio",
				requested_ratio: decision.requestedRatio,
				applied_ratio: decision.appliedRatio,
				granularity,
				feasible_budget_tokens: decision.feasibleBudgetTokens,
				budget_tokens: decision.budgetTokens,
				injected_tokens: injectedTokens,
				l1_ids: allocation.selected.map((item) => item.id),
				l0_ids: allocation.selected.flatMap((item) => item.selectedL0.map((chunk) => chunk.id)),
				adapter: "pi-context-hook",
			};
			writeObservation(observationPath, observation);
			cached = { query, block, observation };
		}

		if (!cached.block) return;
		return { messages: prependMemoryToLatestUser(messages, cached.block) };
	});
}
