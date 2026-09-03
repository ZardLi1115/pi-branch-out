import assert from "node:assert/strict";
import test from "node:test";

import { decideMemoryBudget } from "../memory-budget-controller.ts";
import { allocateProgressiveMemory } from "../progressive-memory-allocator.ts";

test("branch ratio has priority and scales feasible headroom", () => {
  const decision = decideMemoryBudget({
    branchRatio: 0.5,
    policyRatio: 1,
    contextWindowTokens: 100_000,
    currentContextTokens: 30_000,
    reserveTokens: 10_000,
    hardCapTokens: 40_000,
  });
  assert.equal(decision.source, "branch");
  assert.equal(decision.headroomTokens, 60_000);
  assert.equal(decision.feasibleBudgetTokens, 40_000);
  assert.equal(decision.budgetTokens, 20_000);
});

test("allocator never truncates a complete L1", () => {
  const result = allocateProgressiveMemory({
    budgetTokens: 100,
    granularity: "detailed",
    candidates: [
      { id: "a", content: "A", tokenCount: 80, l0: [] },
      { id: "b", content: "B", tokenCount: 30, l0: [] },
      { id: "c", content: "C", tokenCount: 10, l0: [] },
    ],
  });
  assert.deepEqual(result.selected.map((item) => item.id), ["a"]);
  assert.deepEqual(result.droppedL1Ids, ["b", "c"]);
  assert.equal(result.injectedTokens, 80);
});

test("standard adds at most one L0 per admitted L1", () => {
  const result = allocateProgressiveMemory({
    budgetTokens: 500,
    granularity: "standard",
    candidates: [
      {
        id: "a",
        content: "A",
        tokenCount: 100,
        l0: [
          { id: "a1", content: "a1", tokenCount: 50 },
          { id: "a2", content: "a2", tokenCount: 50 },
        ],
      },
      {
        id: "b",
        content: "B",
        tokenCount: 100,
        l0: [
          { id: "b1", content: "b1", tokenCount: 50 },
          { id: "b2", content: "b2", tokenCount: 50 },
        ],
      },
    ],
  });
  assert.deepEqual(result.selected[0].selectedL0.map((item) => item.id), ["a1"]);
  assert.deepEqual(result.selected[1].selectedL0.map((item) => item.id), ["b1"]);
});

test("detailed expands round-robin and never injects the same L0 twice", () => {
  const shared = { id: "shared", content: "shared", tokenCount: 20 };
  const result = allocateProgressiveMemory({
    budgetTokens: 500,
    granularity: "detailed",
    candidates: [
      {
        id: "a",
        content: "A",
        tokenCount: 100,
        l0: [shared, { id: "a2", content: "a2", tokenCount: 20 }],
      },
      {
        id: "b",
        content: "B",
        tokenCount: 100,
        l0: [shared, { id: "b2", content: "b2", tokenCount: 20 }],
      },
    ],
  });
  const ids = result.selected.flatMap((item) => item.selectedL0.map((chunk) => chunk.id));
  assert.equal(ids.filter((id) => id === "shared").length, 1);
  assert.ok(result.skippedDuplicateL0Ids.includes("shared"));
});
