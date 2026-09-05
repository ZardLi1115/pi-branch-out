import assert from "node:assert/strict";
import test from "node:test";

import { decideMemoryBudget } from "../memory-budget-controller.ts";
import { allocateProgressiveMemory, renderProgressiveMemory } from "../progressive-memory-allocator.ts";
import { chooseRatio, policyFeatures } from "../../extensions/tdai-budget-policy.ts";
import { deterministicSample } from "../../extensions/tdai-model-call-collector.ts";

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

test("complete rendered block is recounted and never exceeds budget", () => {
  const candidates = [
    { id: "a", content: "alpha", tokenCount: 1, l0: [] },
    { id: "b", content: "beta", tokenCount: 1, l0: [] },
  ];
  const countRenderedTokens = (value: string) => value.length;
  const full = allocateProgressiveMemory({
    budgetTokens: 10_000, candidates, countRenderedTokens,
  });
  const constrained = allocateProgressiveMemory({
    budgetTokens: full.injectedTokens - 1, candidates, countRenderedTokens,
  });
  assert.deepEqual(full.selected.map((item) => item.id), ["a", "b"]);
  assert.deepEqual(constrained.selected.map((item) => item.id), ["a"]);
  assert.equal(constrained.injectedTokens, renderProgressiveMemory(constrained).length);
  assert.ok(constrained.injectedTokens <= full.injectedTokens - 1);
});

test("unrelated L0 remains standalone evidence and survives without L1", () => {
  const result = allocateProgressiveMemory({
    budgetTokens: 10_000,
    candidates: [],
    independentL0: [{ id: "history", content: "old evidence", tokenCount: 2, retrievalIndex: 0 }],
    countRenderedTokens: (value) => value.length,
  });
  assert.deepEqual(result.selectedIndependentL0.map((item) => item.id), ["history"]);
  assert.match(renderProgressiveMemory(result), /独立历史证据/);
});

test("frozen MLP policy emits one of the fixed budget actions", () => {
  const inputDimension = 14 + 16 + 128;
  const policy = {
    schema_version: 1,
    feature_version: "visible-state-hash-v3-history",
    hash_dim: 128,
    actions: [0, 0.5, 1],
    w1: Array.from({ length: inputDimension }, () => [0]),
    b1: [0],
    w2: [[0, 0, 0]],
    b2: [0, 2, 1],
  };
  const state = { query: "fix parser", recent_tool_result: "one test failed" };
  assert.equal(policyFeatures(state, policy).length, inputDimension);
  assert.equal(chooseRatio(policy, state).ratio, 0.5);
});

test("checkpoint sampling is deterministic and bounded", () => {
  const first = deterministicSample("task", "batch", 17);
  const second = deterministicSample("task", "batch", 17);
  assert.equal(first, second);
  assert.ok(first >= 0 && first < 1);
  assert.notEqual(first, deterministicSample("task", "batch", 18));
});
