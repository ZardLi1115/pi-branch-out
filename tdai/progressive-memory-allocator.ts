export type MemoryGranularity = "compact" | "standard" | "detailed";

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
  /**
   * Associated L0 candidates, already ordered by the retrieval layer.
   * v1 may be query-conditioned same-session L0. A future provenance-aware
   * retriever can fill this array from source_message_ids without changing the
   * allocator.
   */
  l0: L0Candidate[];
}

export interface AllocationInput {
  candidates: L1Candidate[];
  budgetTokens: number;
  granularity: MemoryGranularity;
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

export function maxL0PerL1(granularity: MemoryGranularity): number {
  if (granularity === "compact") return 0;
  if (granularity === "standard") return 1;
  return 3;
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
 * Deterministic allocator used by both natural policy execution and branch-out.
 *
 * Invariants:
 * 1. A selected L1 atom is always complete. No truncation is allowed.
 * 2. L1 admission is ordered by the retrieval layer. If the next complete atom
 *    does not fit, lower-ranked L1 atoms are not admitted either.
 * 3. Only the remaining budget can be used by L0.
 * 4. L0 expansion is round-robin over admitted L1 atoms. This prevents the
 *    first L1 from consuming the entire expansion budget while preserving the
 *    per-L1 Top-0/Top-1/Top-3 semantics.
 * 5. An oversized L0 chunk is skipped, never truncated. The allocator may still
 *    consider later, smaller chunks for other admitted L1 atoms.
 * 6. The same L0 message id is injected at most once globally.
 */
export function allocateProgressiveMemory(input: AllocationInput): AllocationResult {
  const budgetTokens = Math.max(0, Math.floor(input.budgetTokens));
  const maxL0 = maxL0PerL1(input.granularity);
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
      for (let j = index; j < input.candidates.length; j += 1) {
        droppedL1Ids.push(input.candidates[j].id);
      }
      break;
    }
    selected.push({ ...candidate, selectedL0: [] });
    used += cost;
  }

  const l1Tokens = used;
  let l0Tokens = 0;
  const selectedL0Ids = new Set<string>();

  if (maxL0 > 0 && selected.length > 0 && used < budgetTokens) {
    for (let depth = 0; depth < maxL0; depth += 1) {
      for (const memory of selected) {
        const chunk = memory.l0[depth];
        if (!chunk) continue;
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
    }
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
    "以下为当前轮按 Memory Budget 选择的结构化记忆。L1 为完整原子记忆，L0 为按预算渐进展开的原始对话片段：",
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
