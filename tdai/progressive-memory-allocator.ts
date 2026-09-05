export interface L0Candidate {
  id: string;
  content: string;
  tokenCount: number;
  score?: number;
  sessionId?: string;
  role?: string;
  /** Global retrieval rank from the frozen conversation-search response. */
  retrievalIndex?: number;
  /** Present only when the backend returned an explicit L1 relationship. */
  parentL1Id?: string;
}

export interface L1Candidate {
  id: string;
  content: string;
  tokenCount: number;
  score?: number;
  type?: string;
  fromAgentId?: string;
  fromAgentName?: string;
  /** Only explicitly related L0 candidates may be nested here. */
  l0: L0Candidate[];
}

export interface AllocationInput {
  candidates: L1Candidate[];
  /** L0 rows without an explicit source relation remain standalone evidence. */
  independentL0?: L0Candidate[];
  budgetTokens: number;
  /** Counts the complete rendered injection, including wrappers and metadata. */
  countRenderedTokens?: (rendered: string) => number;
}

export interface SelectedL1 extends L1Candidate {
  selectedL0: L0Candidate[];
}

export interface AllocationResult {
  selected: SelectedL1[];
  selectedIndependentL0: L0Candidate[];
  l1Tokens: number;
  l0Tokens: number;
  injectedTokens: number;
  droppedL1Ids: string[];
  skippedOversizeL0Ids: string[];
  skippedDuplicateL0Ids: string[];
}

function assertL0(chunk: L0Candidate, owner: string): void {
  if (!chunk.id) throw new Error(`L0 candidate id is required for ${owner}`);
  if (!Number.isFinite(chunk.tokenCount) || chunk.tokenCount < 0) {
    throw new Error(`invalid L0 tokenCount for ${chunk.id}: ${chunk.tokenCount}`);
  }
}

function assertCandidate(candidate: L1Candidate): void {
  if (!candidate.id) throw new Error("L1 candidate id is required");
  if (!Number.isFinite(candidate.tokenCount) || candidate.tokenCount < 0) {
    throw new Error(`invalid L1 tokenCount for ${candidate.id}: ${candidate.tokenCount}`);
  }
  for (const chunk of candidate.l0) assertL0(chunk, `L1 ${candidate.id}`);
}

function emptyResult(): AllocationResult {
  return {
    selected: [], selectedIndependentL0: [], l1Tokens: 0, l0Tokens: 0,
    injectedTokens: 0, droppedL1Ids: [], skippedOversizeL0Ids: [],
    skippedDuplicateL0Ids: [],
  };
}

function measuredTokens(result: AllocationResult, counter?: (rendered: string) => number): number {
  if (counter) return Math.max(0, Math.floor(counter(renderProgressiveMemory(result))));
  return result.l1Tokens + result.l0Tokens;
}

function l0InRetrievalOrder(selected: SelectedL1[], independent: L0Candidate[]): Array<{
  chunk: L0Candidate;
  owner: SelectedL1 | null;
}> {
  const rows = independent.map((chunk, stable) => ({ chunk, owner: null as SelectedL1 | null, stable }));
  let stable = rows.length;
  for (const owner of selected) {
    for (const chunk of owner.l0) rows.push({ chunk, owner, stable: stable++ });
  }
  rows.sort((a, b) => {
    const ai = Number.isFinite(a.chunk.retrievalIndex) ? Number(a.chunk.retrievalIndex) : Number.MAX_SAFE_INTEGER;
    const bi = Number.isFinite(b.chunk.retrievalIndex) ? Number(b.chunk.retrievalIndex) : Number.MAX_SAFE_INTEGER;
    return ai - bi || a.stable - b.stable;
  });
  return rows.map(({ chunk, owner }) => ({ chunk, owner }));
}

/**
 * Deterministic budget-driven allocator.
 *
 * L1 atoms are admitted whole and in retrieval order. Remaining space is
 * offered to L0 rows in their original global retrieval order. An L0 is
 * nested under an L1 only when the backend supplied an explicit relationship;
 * otherwise it is rendered as standalone historical evidence. Every trial is
 * measured as the complete rendered block, so wrapper and metadata overhead
 * can never be hidden by clamping an observation counter.
 */
export function allocateProgressiveMemory(input: AllocationInput): AllocationResult {
  const budgetTokens = Math.max(0, Math.floor(input.budgetTokens));
  const result = emptyResult();
  const independent = input.independentL0 ?? [];
  for (const candidate of input.candidates) assertCandidate(candidate);
  for (const chunk of independent) assertL0(chunk, "independent history");
  if (budgetTokens === 0) return result;

  for (let index = 0; index < input.candidates.length; index += 1) {
    const candidate = input.candidates[index];
    const selected: SelectedL1 = { ...candidate, selectedL0: [] };
    result.selected.push(selected);
    result.l1Tokens += Math.floor(candidate.tokenCount);
    const trialTokens = measuredTokens(result, input.countRenderedTokens);
    if (trialTokens > budgetTokens) {
      result.selected.pop();
      result.l1Tokens -= Math.floor(candidate.tokenCount);
      for (let j = index; j < input.candidates.length; j += 1) result.droppedL1Ids.push(input.candidates[j].id);
      break;
    }
    result.injectedTokens = trialTokens;
  }

  const selectedIds = new Set(result.selected.map((item) => item.id));
  const selectedL0Ids = new Set<string>();
  for (const { chunk, owner } of l0InRetrievalOrder(result.selected, independent)) {
    if (owner && !selectedIds.has(owner.id)) continue;
    if (selectedL0Ids.has(chunk.id)) {
      result.skippedDuplicateL0Ids.push(chunk.id);
      continue;
    }
    if (owner) owner.selectedL0.push(chunk);
    else result.selectedIndependentL0.push(chunk);
    result.l0Tokens += Math.floor(chunk.tokenCount);
    const trialTokens = measuredTokens(result, input.countRenderedTokens);
    if (trialTokens > budgetTokens) {
      if (owner) owner.selectedL0.pop();
      else result.selectedIndependentL0.pop();
      result.l0Tokens -= Math.floor(chunk.tokenCount);
      result.skippedOversizeL0Ids.push(chunk.id);
      continue;
    }
    selectedL0Ids.add(chunk.id);
    result.injectedTokens = trialTokens;
  }

  result.injectedTokens = measuredTokens(result, input.countRenderedTokens);
  if (result.injectedTokens > budgetTokens) throw new Error("rendered memory exceeds budget");
  return result;
}

export function renderProgressiveMemory(result: AllocationResult): string {
  if (result.selected.length === 0 && result.selectedIndependentL0.length === 0) return "";
  const lines: string[] = [
    "<tdai_recalled_memories>",
    "以下为当前轮在 Memory Budget 内选择的记忆。L1 保持完整；只有具备明确来源关系的 L0 才会嵌套展示：",
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

  if (result.selectedIndependentL0.length > 0) {
    lines.push("独立历史证据（后端未提供与 L1 的明确来源关系）：");
    result.selectedIndependentL0.forEach((chunk, index) => {
      const role = chunk.role ? ` role=${chunk.role}` : "";
      const score = typeof chunk.score === "number" ? ` score=${chunk.score.toFixed(3)}` : "";
      lines.push(`${index + 1}. [L0${role}${score}] ${chunk.content}`);
    });
  }

  lines.push("</tdai_recalled_memories>");
  return lines.join("\n");
}
