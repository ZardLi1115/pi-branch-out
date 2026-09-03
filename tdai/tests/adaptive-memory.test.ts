import assert from "node:assert/strict";
import test from "node:test";

import { decideMemoryBudget } from "../memory-budget-controller.ts";
import { allocateProgressiveMemory } from "../progressive-memory-allocator.ts";

test("branch ratio scales candidate-aware feasible budget", () => {
  const decision = decideMemoryBudget({
    branchRatio: 0.5,
    contextWindowTokens: 100_000,
    currentContextTokens: 30_000,
    reserveTokens: 10_000,
    candidateTokens: 30_000,
    hardCapTokens: 40_000,
  });
  assert.equal(decision.source, "branch");
  assert.equal(decision.headroomTokens, 60_000);
  assert.equal(decision.feasibleBudgetTokens, 30_000);
  assert.equal(decision.budgetTokens, 15_000);
});

test("allocator never truncates a complete L1", () => {
  const result = allocateProgressiveMemory({
    budgetTokens: 100,
    candidates: [
      { id: "a", content: "A", tokenCount: 80, l0: [] },
      { id: "b", content: "B", tokenCount: 30, l0: [] },
      { id: "c", content: "C", tokenCount: 10, l0: [] },
    ],
  });
  assert.deepEqual(result.selected.map((item) => item.id), ["a"]);
  assert.deepEqual(result.droppedL1Ids, ["b", "c"]);
});

test("remaining budget progressively expands L0 round-robin", () => {
  const result = allocateProgressiveMemory({
    budgetTokens: 360,
    candidates: [
      {
        id: "a", content: "A", tokenCount: 100,
        l0: [
          { id: "a1", content: "a1", tokenCount: 40 },
          { id: "a2", content: "a2", tokenCount: 40 },
        ],
      },
      {
        id: "b", content: "B", tokenCount: 100,
        l0: [
          { id: "b1", content: "b1", tokenCount: 40 },
          { id: "b2", content: "b2", tokenCount: 40 },
        ],
      },
    ],
  });
  assert.deepEqual(result.selected[0].selectedL0.map((item) => item.id), ["a1", "a2"]);
  assert.deepEqual(result.selected[1].selectedL0.map((item) => item.id), ["b1", "b2"]);
  assert.equal(result.injectedTokens, 360);
});

test("duplicate L0 ids are injected once", () => {
  const shared = { id: "shared", content: "shared", tokenCount: 20 };
  const result = allocateProgressiveMemory({
    budgetTokens: 500,
    candidates: [
      { id: "a", content: "A", tokenCount: 100, l0: [shared] },
      { id: "b", content: "B", tokenCount: 100, l0: [shared] },
    ],
  });
  const ids = result.selected.flatMap((item) => item.selectedL0.map((chunk) => chunk.id));
  assert.equal(ids.filter((id) => id === "shared").length, 1);
});
