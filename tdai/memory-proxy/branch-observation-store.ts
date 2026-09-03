export interface BranchMemoryObservation {
  kind: "memory_budget_ratio";
  observation_id: string;
  source: "branch" | "policy" | "default";
  requested_ratio: number;
  applied_ratio: number;
  granularity: "compact" | "standard" | "detailed";
  context_window_tokens: number;
  current_context_tokens: number;
  reserve_tokens: number;
  feasible_budget_tokens: number;
  budget_tokens: number;
  injected_tokens: number;
  l1_ids: string[];
  l0_ids: string[];
  created_at: string;
}

/**
 * Small process-local rendezvous store used only by branch-out experiments.
 *
 * The branch runner sends a cryptographically random observation id with the
 * counterfactual request. The proxy writes the realized action under that id,
 * and the Harbor agent immediately fetches it after Pi finishes the turn.
 * Entries are short-lived to avoid turning this into another persistence layer.
 */
const observations = new Map<string, { value: BranchMemoryObservation; expiresAt: number }>();
const DEFAULT_TTL_MS = 10 * 60_000;
const MAX_ENTRIES = 4096;

function sweep(now = Date.now()): void {
  for (const [key, entry] of observations) {
    if (entry.expiresAt <= now) observations.delete(key);
  }
  if (observations.size <= MAX_ENTRIES) return;
  const overflow = observations.size - MAX_ENTRIES;
  let removed = 0;
  for (const key of observations.keys()) {
    observations.delete(key);
    removed += 1;
    if (removed >= overflow) break;
  }
}

export function putBranchMemoryObservation(
  observation: BranchMemoryObservation,
  ttlMs = DEFAULT_TTL_MS,
): void {
  if (!observation.observation_id) throw new Error("observation_id is required");
  sweep();
  observations.set(observation.observation_id, {
    value: observation,
    expiresAt: Date.now() + Math.max(1_000, ttlMs),
  });
}

export function takeBranchMemoryObservation(id: string): BranchMemoryObservation | null {
  sweep();
  const entry = observations.get(id);
  if (!entry) return null;
  observations.delete(id);
  return entry.value;
}

export function peekBranchMemoryObservation(id: string): BranchMemoryObservation | null {
  sweep();
  return observations.get(id)?.value ?? null;
}

export function branchOutEnabled(): boolean {
  return process.env.TDAI_BRANCH_OUT_ENABLED === "1";
}
