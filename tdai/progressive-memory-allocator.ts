export interface L0Candidate {
  id: string;
  content: string;
  tokenCount: number;
  score?: number;
  sessionId?: string;
  role?: string;
}

export interface L1Candidate {
  id: string;
  content: string;
  tokenCount: number;
  score?: number;
  type?: string;
  fromAgentId?: string;
  fromAgentName?: string;
  /** L0 candidates are already ordered by the retrieval layer. */
  l0: L0Candidate[];
}

export interface AllocationInput {
  candidates: L1Candidate[];
  budgetTokens: number;
}

export interface SelectedL1 extends L1Candidate {
  selectedL0: L0Candidate[];
}

export interface AllocationResult {
  selected: SelectedL1[];
  l1Tokens: number;
  l0Tokens: number;
  injectedTokens: number;
  droppedL1Ids: string[];
  skippedOversizeL0Ids: string[];
  skippedDuplicateL0Ids: string[];
}

function assertCandidate(candidate: L1Candidate): void {
  if (!candidate.id) throw new Error("L1 candidate id is required");
  if (!Number.isFinite(candidate.tokenCount) || candidate.tokenCount < 0) {
    throw new Error(`invalid L1 tokenCount for ${candidate.id}: ${candidate.tokenCount}`);
  }
  for (const chunk of candidate.l0) {
    if (!chunk.id) throw new Error(`L0 candidate id is required for L1 ${candidate.id}`);
    if (!Number.isFinite(chunk.tokenCount) || chunk.tokenCount < 0) {
      throw new Error(`invalid L0 tokenCount for ${chunk.id}: ${chunk.tokenCount}`);
    }
  }
}

/**
 * Deterministic budget-driven allocator.
 *
 * 1. Admit complete L1 atoms in retrieval order. Never truncate an L1.
 * 2. If the next L1 does not fit, stop admitting lower-ranked L1s.
 * 3. Spend all remaining budget on L0 in round-robin depth order:
 *    every admitted L1 gets a chance at depth 0, then depth 1, etc.
 * 4. Oversized L0 chunks are skipped, never truncated.
 * 5. The same L0 id is injected at most once globally.
 *
 * There is intentionally no learned or configured "granularity". A larger
 * budget naturally creates deeper L0 expansion.
 */
export function allocateProgressiveMemory(input: AllocationInput): AllocationResult {
  const budgetTokens = Math.max(0, Math.floor(input.budgetTokens));
  const selected: SelectedL1[] = [];
  const droppedL1Ids: string[] = [];
  const skippedOversizeL0Ids: string[] = [];
  const skippedDuplicateL0Ids: string[] = [];

  let used = 0;
  for (let index = 0; index < input.candidates.length; index += 1) {
    const candidate = input.candidates[index];
    assertCandidate(candidate);
    const cost = Math.floor(candidate.tokenCount);
    if (used + cost > budgetTokens) {
      for (let j = index; j < input.candidates.length; j += 1) droppedL1Ids.push(input.candidates[j].id);
      break;
    }
    selected.push({ ...candidate, selectedL0: [] });
    used += cost;
  }

  const l1Tokens = used;
  let l0Tokens = 0;
  const selectedL0Ids = new Set<string>();
  let depth = 0;
  let anyCandidateAtDepth = true;

  while (selected.length > 0 && used < budgetTokens && anyCandidateAtDepth) {
    anyCandidateAtDepth = false;
    for (const memory of selected) {
      const chunk = memory.l0[depth];
      if (!chunk) continue;
      anyCandidateAtDepth = true;
      if (selectedL0Ids.has(chunk.id)) {
        skippedDuplicateL0Ids.push(chunk.id);
        continue;
      }
      const cost = Math.floor(chunk.tokenCount);
      if (used + cost > budgetTokens) {
        skippedOversizeL0Ids.push(chunk.id);
        continue;
      }
      memory.selectedL0.push(chunk);
      selectedL0Ids.add(chunk.id);
      used += cost;
      l0Tokens += cost;
    }
    depth += 1;
  }

  return {
    selected,
    l1Tokens,
    l0Tokens,
    injectedTokens: used,
    droppedL1Ids,
    skippedOversizeL0Ids,
    skippedDuplicateL0Ids,
  };
}

export function renderProgressiveMemory(result: AllocationResult): string {
  if (result.selected.length === 0) return "";
  const lines: string[] = [
    "<tdai_recalled_memories>",
    "以下为当前轮在 Memory Budget 内选择的记忆。L1 始终完整，剩余预算用于逐层展开 L0：",
  ];

  result.selected.forEach((memory, index) => {
    const fromTag = memory.fromAgentName
      ? ` from=${memory.fromAgentName}`
      : memory.fromAgentId
        ? ` from=${memory.fromAgentId}`
        : "";
    const score = typeof memory.score === "number" ? ` score=${memory.score.toFixed(3)}` : "";
    lines.push(`${index + 1}. [L1:${memory.type ?? "memory"}${score}${fromTag}] ${memory.content}`);
    memory.selectedL0.forEach((chunk, l0Index) => {
      const role = chunk.role ? ` role=${chunk.role}` : "";
      const l0Score = typeof chunk.score === "number" ? ` score=${chunk.score.toFixed(3)}` : "";
      lines.push(`   - [L0:${l0Index + 1}${role}${l0Score}] ${chunk.content}`);
    });
  });

  lines.push("</tdai_recalled_memories>");
  return lines.join("\n");
}
