import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { basename, join } from "node:path";
import {
  estimateContextTokens,
  estimateMessagesTokens as estimateRenderedMessagesTokens,
  prependToLatestUser,
  toCandidates,
  type Snapshot,
} from "./tdai-budget-override.js";
import { decideMemoryBudget } from "../tdai/memory-budget-controller.js";
import { allocateProgressiveMemory, renderProgressiveMemory } from "../tdai/progressive-memory-allocator.js";

type ExtensionAPI = { on(event: string, callback: (event: any, ctx: any) => unknown): void };
type SearchHit = {
  id?: unknown;
  content?: unknown;
  type?: unknown;
  source_agent_id?: unknown;
};

function textOfContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content == null ? "" : JSON.stringify(content);
  return content.map((part) => {
    if (typeof part === "string") return part;
    if (!part || typeof part !== "object") return "";
    const value = part as Record<string, unknown>;
    return typeof value.text === "string" ? value.text : typeof value.content === "string" ? value.content : "";
  }).filter(Boolean).join("\n");
}

function estimateTextTokens(text: string): number {
  if (!text) return 0;
  let cjk = 0;
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if ((cp >= 0x3400 && cp <= 0x9fff) || (cp >= 0xf900 && cp <= 0xfaff)) cjk += 1;
  }
  return Math.max(1, Math.ceil(cjk / 1.7 + Math.max(0, text.length - cjk) / 4));
}

function estimateMessagesTokens(messages: any[]): number {
  return messages.reduce(
    (sum, message) => sum + estimateTextTokens(`${String(message?.role ?? "")}\n${textOfContent(message?.content)}`) + 4,
    0,
  );
}

function rows(value: unknown, key: "items" | "messages"): SearchHit[] {
  if (!value || typeof value !== "object") return [];
  const list = (value as any)?.data?.[key];
  return Array.isArray(list) ? list as SearchHit[] : [];
}

function queryFromMessages(messages: any[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const role = messages[index]?.role;
    if (role !== "user" && role !== "toolResult") continue;
    const text = textOfContent(messages[index]?.content).trim();
    if (text) return text.slice(0, 2048);
  }
  return "";
}

function recentToolResult(messages: any[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role !== "toolResult") continue;
    return textOfContent(messages[index]?.content).trim().slice(0, 8192);
  }
  return "";
}

function optionalFiniteEnv(name: string): number | null {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function usageNumber(...values: unknown[]): number {
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number) && number >= 0) return number;
  }
  return 0;
}

function boundedIntEnv(name: string, fallback: number, minimum: number): number {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= minimum ? Math.floor(value) : fallback;
}

function probabilityEnv(name: string, fallback: number): number {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= 0 && value <= 1 ? value : fallback;
}

export function deterministicSample(task: string, batch: string, call: number): number {
  const digest = createHash("sha256").update(`${task}\0${batch}\0${call}`).digest();
  const high = digest.readUInt32BE(0);
  const low = digest.readUInt32BE(4);
  return (high * 0x1_0000_0000 + low) / 0x1_0000_0000_0000_0000;
}

async function bridgeSearch(conversationId: string, kind: string, query: string, limit: number): Promise<Record<string, unknown>> {
  const proxy = (process.env.TDAI_PROXY_URL ?? "http://127.0.0.1:8096").replace(/\/$/, "");
  const space = process.env.TDAI_SPACE_ID ?? "default";
  const keys = bridgeSessionKey(conversationId) === conversationId
    ? [conversationId]
    : [bridgeSessionKey(conversationId), conversationId];
  for (let index = 0; index < keys.length; index += 1) {
    const response = await fetch(`${proxy}/memory-bridge/v3/${kind}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-conversation-id": keys[index],
        "x-tdai-service-id": space,
      },
      body: JSON.stringify({ query, limit }),
    });
    const text = await response.text();
    if (response.ok) {
      const value = JSON.parse(text || "{}") as unknown;
      return value && typeof value === "object" ? value as Record<string, unknown> : {};
    }
    const coldMiss = response.status === 401 && text.includes("session not initialized");
    if (!coldMiss || index + 1 === keys.length) {
      throw new Error(`memory bridge ${kind} failed: HTTP ${response.status} ${text.slice(-1000)}`);
    }
  }
  throw new Error(`memory bridge ${kind} failed without a response`);
}

function candidateTokens(atomic: unknown, conversation: unknown): number {
  const l1 = rows(atomic, "items");
  const l0 = rows(conversation, "messages");
  return l1.reduce((sum, row) => sum + estimateTextTokens(`[L1:${String(row.type ?? "memory")}] ${String(row.content ?? "")}`), 0)
    + l0.reduce((sum, row) => sum + estimateTextTokens(`[L0] ${String(row.content ?? "")}`), 0);
}

function git(cwd: string, args: string[], encoding?: BufferEncoding): string | Buffer {
  return execFileSync("git", args, {
    cwd,
    encoding: encoding ?? null,
    maxBuffer: 512 * 1024 * 1024,
  });
}

function captureWorkspace(cwd: string, root: string, baseCommit: string): {
  patch: string;
  untrackedArchive: string | null;
  bytes: number;
} {
  const patchName = "workspace.patch";
  const patch = git(cwd, ["diff", "--binary", "--no-ext-diff", baseCommit, "--", "."], "utf8") as string;
  writeFileSync(join(root, patchName), patch, "utf8");

  const untracked = git(cwd, ["ls-files", "--others", "--exclude-standard", "-z"]) as Buffer;
  if (untracked.length === 0) return { patch: patchName, untrackedArchive: null, bytes: Buffer.byteLength(patch) };

  const listPath = join(root, "untracked-files.list");
  const archiveName = "workspace-untracked.tar.gz";
  writeFileSync(listPath, untracked);
  execFileSync("tar", ["--null", "-T", listPath, "-czf", join(root, archiveName)], {
    cwd,
    maxBuffer: 512 * 1024 * 1024,
  });
  return {
    patch: patchName,
    untrackedArchive: archiveName,
    bytes: Buffer.byteLength(patch) + statSync(join(root, archiveName)).size,
  };
}

function appendJsonl(path: string, value: Record<string, unknown>): void {
  writeFileSync(path, `${JSON.stringify(value)}\n`, { encoding: "utf8", flag: "a" });
}

function branchObservation(): Record<string, unknown> | null {
  const path = process.env.PI_BRANCH_OUT_OBSERVATION_FILE;
  if (!path || !existsSync(path)) return null;
  try {
    const value = JSON.parse(readFileSync(path, "utf8")) as unknown;
    return value && typeof value === "object" ? value as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

function previousSamplingState(path: string): { probes: number; saved: number; lastSavedCall: number | null } {
  if (!existsSync(path)) return { probes: 0, saved: 0, lastSavedCall: null };
  try {
    const rows = readFileSync(path, "utf8").split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
    return {
      probes: rows.filter((row) => row?.sampling?.probe_attempted === true).length,
      saved: rows.filter((row) => row?.checkpoint_status === "ready").length,
      lastSavedCall: rows.reduce(
        (last, row) => row?.checkpoint_status === "ready" ? Number(row.model_call_index) : last,
        null as number | null,
      ),
    };
  } catch {
    return { probes: 0, saved: 0, lastSavedCall: null };
  }
}

function bridgeSessionKey(conversationId: string): string {
  if (conversationId.includes(":")) return conversationId;
  if (process.env.TDAI_WIRE_API === "responses") return `codex:${conversationId}`;
  return conversationId;
}

function policyObservation(outputRoot: string, callIndex: number): Record<string, unknown> | null {
  const path = join(outputRoot, "latest-policy-observation.json");
  if (!existsSync(path)) return null;
  try {
    const value = JSON.parse(readFileSync(path, "utf8")) as unknown;
    if (!value || typeof value !== "object") return null;
    const observation = value as Record<string, unknown>;
    return Number(observation.model_call_index) === callIndex ? observation : null;
  } catch {
    return null;
  }
}

export default function tdaiModelCallCollector(pi: ExtensionAPI): void {
  const outputRoot = process.env.PI_BRANCH_OUT_MODEL_CALL_DIR;
  if (!outputRoot) return;
  mkdirSync(outputRoot, { recursive: true });
  const stateLog = join(outputRoot, "model-call-states.jsonl");
  const usageLog = join(outputRoot, "model-call-usage.jsonl");
  const requestShapeLog = join(outputRoot, "provider-request-shapes.jsonl");
  const summaryPath = join(outputRoot, "collector-summary.json");
  const maxCheckpoints = boundedIntEnv("PI_BRANCH_OUT_MAX_CHECKPOINTS", 2, 0);
  const minCheckpointGap = boundedIntEnv("PI_BRANCH_OUT_MIN_CHECKPOINT_GAP", 10, 0);
  const maxCandidateProbes = boundedIntEnv("PI_BRANCH_OUT_MAX_CANDIDATE_PROBES", 8, 0);
  const sampleProbability = probabilityEnv("PI_BRANCH_OUT_SAMPLE_PROBABILITY", 0.1);
  const samplingBatch = process.env.PI_BRANCH_OUT_SAMPLING_BATCH ?? "default-v1";
  const previousSampling = previousSamplingState(stateLog);
  let callIndex = Number(process.env.PI_BRANCH_OUT_CALL_OFFSET ?? 0);
  let baseCommit = "";
  let branchObservationConsumed = false;
  let previousActualMemoryTokens = 0;
  let previousMappedAction = 0;
  let previousBudgetTokens = 0;
  let candidateProbes = previousSampling.probes;
  let savedCheckpoints = previousSampling.saved;
  let lastSavedCall = previousSampling.lastSavedCall;
  let cumulativeProbeMs = 0;
  let cumulativeCheckpointMs = 0;
  let cumulativeCheckpointBytes = 0;

  const writeSummary = (): void => {
    writeFileSync(summaryPath, `${JSON.stringify({
      version: 1,
      sampling_batch: samplingBatch,
      max_checkpoints: maxCheckpoints,
      min_checkpoint_gap: minCheckpointGap,
      sample_probability: sampleProbability,
      max_candidate_probes: maxCandidateProbes,
      calls_observed: callIndex,
      candidate_probes: candidateProbes,
      checkpoints_saved: savedCheckpoints,
      last_saved_call: lastSavedCall,
      cumulative_probe_ms: cumulativeProbeMs,
      cumulative_checkpoint_ms: cumulativeCheckpointMs,
      cumulative_checkpoint_bytes: cumulativeCheckpointBytes,
    }, null, 2)}\n`, "utf8");
  };

  pi.on("message_end", (event: any) => {
    const message = event?.message;
    if (message?.role !== "assistant" || !message?.usage) return;
    const usage = message.usage as Record<string, unknown>;
    const cost = usage.cost && typeof usage.cost === "object" ? usage.cost as Record<string, unknown> : {};
    appendJsonl(usageLog, {
      version: 1,
      model_call_index: callIndex,
      timestamp: new Date().toISOString(),
      input_tokens: usageNumber(usage.input, usage.inputTokens, usage.input_tokens),
      output_tokens: usageNumber(usage.output, usage.outputTokens, usage.output_tokens),
      cache_read_tokens: usageNumber(usage.cacheRead, usage.cache_read, usage.cache_read_tokens),
      cache_write_tokens: usageNumber(usage.cacheWrite, usage.cache_write, usage.cache_write_tokens),
      provider_reported_cost_usd: usageNumber(cost.total, usage.costUsd, usage.cost_usd),
      usage_schema: "pi-exclusive-input-cache-v1",
    });
  });

  pi.on("before_provider_request", (event: any, ctx: any) => {
    if (ctx?.model?.provider !== "tdai") return;
    const nestedKeys = Object.fromEntries(
      Object.entries(event ?? {}).flatMap(([key, value]) =>
        value && typeof value === "object" && !Array.isArray(value)
          ? [[key, Object.keys(value as Record<string, unknown>).sort()]]
          : [],
      ),
    );
    const body = event?.payload && typeof event.payload === "object"
      ? event.payload
      : event?.body && typeof event.body === "object"
      ? event.body
      : event?.request?.body && typeof event.request.body === "object"
        ? event.request.body
        : event?.request && typeof event.request === "object"
          ? event.request
          : event;
    const input = Array.isArray(body?.input) ? body.input : [];
    appendJsonl(requestShapeLog, {
      version: 1,
      model_call_index: callIndex + 1,
      event_keys: Object.keys(event ?? {}).sort(),
      nested_object_keys: nestedKeys,
      input_is_array: Array.isArray(body?.input),
      input_items: input.map((item: any) => ({
        type: typeof item?.type === "string" ? item.type : null,
        role: typeof item?.role === "string" ? item.role : null,
        content_is_array: Array.isArray(item?.content),
        content_types: Array.isArray(item?.content)
          ? item.content.map((part: any) => typeof part?.type === "string" ? part.type : null)
          : [],
      })),
      has_client_metadata: Boolean(body?.client_metadata),
      client_metadata_has_session_id: typeof body?.client_metadata?.session_id === "string",
    });
  });

  pi.on("context", async (event: any, ctx: any) => {
    callIndex += 1;
    const workspaceCwd = ctx?.cwd ?? process.cwd();
    if (!baseCommit) {
      try {
        baseCommit = String(git(workspaceCwd, ["rev-parse", "HEAD"], "utf8")).trim();
      } catch {
        baseCommit = "";
      }
    }
    const messages = Array.isArray(event?.messages) ? event.messages : [];
    const query = queryFromMessages(messages);
    const state: Record<string, unknown> = {
      version: 1,
      task: process.env.PI_BRANCH_OUT_TASK_NAME ?? basename(ctx?.cwd ?? process.cwd()),
      model_call_index: callIndex,
      timestamp: new Date().toISOString(),
      context_tokens: estimateContextTokens(event),
      context_window_tokens: Number(ctx?.model?.contextWindow ?? 524288),
      reserve_tokens: Number(process.env.TDAI_MEMORY_RESERVE_TOKENS ?? 16384),
      remaining_call_budget: optionalFiniteEnv("PI_BRANCH_OUT_REMAINING_CALLS"),
      remaining_cost_budget_usd: optionalFiniteEnv("PI_BRANCH_OUT_REMAINING_COST_USD"),
      remaining_time_seconds: optionalFiniteEnv("PI_BRANCH_OUT_REMAINING_TIME_SECONDS"),
      query,
      recent_tool_result: recentToolResult(messages),
      candidate_memory_tokens: null,
      candidate_count: null,
      l1_count: null,
      l0_count: null,
      default_actual_memory_tokens: 0,
      default_mapped_action: 0,
      previous_actual_memory_tokens: previousActualMemoryTokens,
      previous_mapped_action: previousMappedAction,
      previous_budget_tokens: previousBudgetTokens,
      actual_injected_content_sha256: createHash("sha256").update("").digest("hex"),
      checkpoint_status: callIndex === 1 ? "cold-start" : "not-selected",
      tokenizer_version: "tdai-estimator-v2-complete-render",
      allocator_version: "complete-render-v2",
      action_table_version: "budget-ratios-v1",
      policy_version: process.env.PI_BRANCH_OUT_POLICY_VERSION ?? "baseline-v1",
    };

    if (callIndex === 1) {
      state.candidate_observation = "unobserved";
      state.pi_leaf_id = ctx?.sessionManager?.getLeafId?.() ?? null;
      state.sampling = { eligible: false, reason: "cold-start", probe_attempted: false };
      appendJsonl(stateLog, state);
      writeSummary();
      return;
    }


    const taskName = String(state.task);
    const randomDraw = deterministicSample(taskName, samplingBatch, callIndex);
    const hasCheckpointQuota = savedCheckpoints < maxCheckpoints;
    const hasProbeQuota = candidateProbes < maxCandidateProbes;
    const gapSatisfied = lastSavedCall == null || callIndex - lastSavedCall >= minCheckpointGap;
    const sampled = randomDraw < sampleProbability;
    const shouldProbe = hasCheckpointQuota && hasProbeQuota && gapSatisfied && sampled;
    state.pi_leaf_id = ctx?.sessionManager?.getLeafId?.() ?? null;
    state.candidate_observation = "unobserved";
    state.sampling = {
      batch: samplingBatch,
      random_draw: randomDraw,
      probability: sampleProbability,
      eligible: hasCheckpointQuota && hasProbeQuota && gapSatisfied,
      sampled,
      probe_attempted: shouldProbe,
      checkpoints_saved_before: savedCheckpoints,
      candidate_probes_before: candidateProbes,
      calls_since_checkpoint: lastSavedCall == null ? null : callIndex - lastSavedCall,
      reason: !hasCheckpointQuota
        ? "checkpoint-quota-exhausted"
        : !hasProbeQuota
          ? "probe-quota-exhausted"
          : !gapSatisfied
            ? "minimum-gap"
            : !sampled
              ? "not-sampled"
              : "probe",
    };
    if (!shouldProbe) {
      appendJsonl(stateLog, state);
      writeSummary();
      return;
    }

    candidateProbes += 1;
    const probeStarted = Date.now();

    const sessionId = ctx?.sessionManager?.getSessionId?.();
    const conversationId = typeof sessionId === "string" && sessionId ? `pi-${sessionId}` : "";
    let atomic: Record<string, unknown> = {};
    let conversation: Record<string, unknown> = {};
    let recallStatus = "ready";

    try {
      if (!conversationId) throw new Error("pi session id is unavailable");
      [atomic, conversation] = await Promise.all([
        bridgeSearch(conversationId, "atomic/search", query, 36),
        bridgeSearch(conversationId, "conversation/search", query, 72),
      ]);
    } catch (error) {
      recallStatus = "bridge-error";
      state.recall_snapshot_error = String(error).slice(-1000);
    }

    const probeMs = Date.now() - probeStarted;
    cumulativeProbeMs += probeMs;

    const l1 = rows(atomic, "items");
    const l0 = rows(conversation, "messages");
    state.candidate_count = l1.length + l0.length;
    state.l1_count = l1.length;
    state.l0_count = l0.length;
    state.l1_lengths = l1.map((row) => [...String(row.content ?? "")].length);
    state.l0_lengths = l0.map((row) => [...String(row.content ?? "")].length);
    state.l1_scores = l1.map((row) => typeof (row as any).score === "number" ? (row as any).score : null);
    state.l0_scores = l0.map((row) => typeof (row as any).score === "number" ? (row as any).score : null);
    state.recall_snapshot_status = recallStatus;
    state.candidate_observation = recallStatus === "ready" ? "observed" : "probe-error";
    state.candidate_probe_ms = probeMs;

    const branchCandidate = branchObservationConsumed ? null : branchObservation();
    const branchForCall = branchCandidate && Number(branchCandidate.model_call_index) === callIndex ? branchCandidate : null;
    const observation = branchForCall ?? policyObservation(outputRoot, callIndex);
    if (observation) {
      state.actual_memory_tokens = observation.injected_tokens ?? 0;
      state.mapped_action = observation.applied_ratio ?? observation.requested_ratio;
      state.actual_injected_content_sha256 = observation.injected_content_sha256 ?? state.actual_injected_content_sha256;
      state.effective_action_id = observation.effective_action_id ?? null;
      if (branchForCall) branchObservationConsumed = true;
    }
    previousActualMemoryTokens = Number(state.actual_memory_tokens ?? 0);
    previousMappedAction = Number(state.mapped_action ?? state.default_mapped_action ?? 0);
    previousBudgetTokens = Number(observation?.budget_tokens ?? 0);

    const snapshot: Snapshot & Record<string, unknown> = {
      version: 1,
      task: state.task,
      model_call_index: callIndex,
      query,
      conversation_id: conversationId,
      atomic_search: atomic,
      conversation_search: conversation,
      baseline: {
        mode: "tdai-default-dynamic-l1-l0",
        budget_ratio: 0,
        actual_memory_tokens: 0,
        injected_content_sha256: createHash("sha256").update("").digest("hex"),
      },
    };
    const { candidates, independentL0 } = toCandidates(snapshot);
    const countRenderedTokens = (block: string): number => block
      ? Math.max(0, estimateRenderedMessagesTokens(prependToLatestUser(messages, block)) - estimateRenderedMessagesTokens(messages))
      : 0;
    const fullAllocation = allocateProgressiveMemory({
      candidates, independentL0, budgetTokens: Number.MAX_SAFE_INTEGER, countRenderedTokens,
    });
    state.candidate_memory_tokens = fullAllocation.injectedTokens;
    const contextWindow = Number(ctx?.model?.contextWindow ?? 524288);
    const reserve = Number(process.env.TDAI_MEMORY_RESERVE_TOKENS ?? 16384);
    const hardCap = Number(process.env.TDAI_MEMORY_HARD_CAP_TOKENS ?? 0);
    const ratios = (process.env.PI_BRANCH_OUT_ACTION_TABLE ?? "0,0.2,0.4,0.6,0.8,1")
      .split(",").map((value) => Number(value.trim())).filter((value) => Number.isFinite(value) && value >= 0 && value <= 1);
    const actionPlans = ratios.map((actionRatio) => {
      const decision = decideMemoryBudget({
        branchRatio: actionRatio,
        contextWindowTokens: contextWindow,
        currentContextTokens: estimateContextTokens(event),
        reserveTokens: reserve,
        candidateTokens: fullAllocation.injectedTokens,
        hardCapTokens: Number.isFinite(hardCap) && hardCap > 0 ? hardCap : null,
      });
      const allocation = allocateProgressiveMemory({
        candidates, independentL0, budgetTokens: decision.budgetTokens, countRenderedTokens,
      });
      const rendered = renderProgressiveMemory(allocation);
      return {
        ratio: actionRatio,
        feasible_budget_tokens: decision.feasibleBudgetTokens,
        budget_tokens: decision.budgetTokens,
        injected_tokens: allocation.injectedTokens,
        injected_content_sha256: createHash("sha256").update(rendered).digest("hex"),
        l1_ids: allocation.selected.map((item) => item.id),
        l0_ids: [
          ...allocation.selected.flatMap((item) => item.selectedL0.map((chunk) => chunk.id)),
          ...allocation.selectedIndependentL0.map((chunk) => chunk.id),
        ],
      };
    });
    const canonicalByHash = new Map<string, number>();
    for (const plan of actionPlans) {
      if (!canonicalByHash.has(plan.injected_content_sha256)) canonicalByHash.set(plan.injected_content_sha256, plan.ratio);
      (plan as Record<string, unknown>).canonical_ratio = canonicalByHash.get(plan.injected_content_sha256);
    }
    snapshot.action_plans = actionPlans;
    state.action_plans = actionPlans;
    const effectiveActionCount = new Set(actionPlans.map((plan) => plan.injected_content_sha256)).size;
    state.effective_action_count = effectiveActionCount;
    if (recallStatus !== "ready" || effectiveActionCount < 2) {
      state.checkpoint_status = recallStatus === "ready" ? "not-worthwhile" : "probe-error";
      (state.sampling as Record<string, unknown>).reason = recallStatus === "ready"
        ? "equivalent-actions"
        : "recall-failed";
      appendJsonl(stateLog, state);
      writeSummary();
      return;
    }

    const checkpointStarted = Date.now();
    const checkpointRoot = join(outputRoot, `call-${String(callIndex).padStart(3, "0")}`);
    mkdirSync(checkpointRoot, { recursive: true });
    const snapshotName = "recall-snapshot.json";
    const snapshotText = `${JSON.stringify(snapshot, null, 2)}\n`;
    const snapshotSha256 = createHash("sha256").update(snapshotText).digest("hex");
    state.snapshot_sha256 = snapshotSha256;
    if (recallStatus === "ready") {
      writeFileSync(join(checkpointRoot, snapshotName), snapshotText, "utf8");
    }

    try {
      if (!baseCommit) throw new Error("workspace is not a git repository");
      const sessionFile = ctx?.sessionManager?.getSessionFile?.();
      const leafId = ctx?.sessionManager?.getLeafId?.();
      if (typeof sessionFile !== "string" || !sessionFile) throw new Error("Pi session file is unavailable");
      if (typeof leafId !== "string" || !leafId) throw new Error("Pi session leaf is unavailable");
      const sessionName = `../../pi-session/${basename(sessionFile)}`;
      const workspace = captureWorkspace(workspaceCwd, checkpointRoot, baseCommit);
      const manifest = {
        task_name: state.task,
        step_index: 1,
        step_name: `model-call-${callIndex}`,
        workspace_archive: "",
        pi_checkpoint_session: sessionName,
        pi_source_session: basename(sessionFile),
        pi_leaf_id: leafId,
        tdai_state_archive: null,
        tdai_state_mode: "none",
        source_trial_dir: null,
        source_reward: null,
        recall_snapshot: recallStatus === "ready" ? snapshotName : null,
        recall_snapshot_status: recallStatus,
        baseline_budget_ratio: 0,
        baseline_action: 0,
        checkpoint_boundary: "model-call",
        model_call_index: callIndex,
        workspace_mode: "git-delta-v1",
        workspace_base_commit: baseCommit,
        workspace_patch: workspace.patch,
        workspace_untracked_archive: workspace.untrackedArchive,
        snapshot_sha256: snapshotSha256,
        allocator_version: "complete-render-v2",
        tokenizer_version: "tdai-estimator-v2-complete-render",
        action_table_version: "budget-ratios-v1",
        policy_version: process.env.PI_BRANCH_OUT_POLICY_VERSION ?? "baseline-v1",
        backend_snapshot_status: "not-captured",
        backend_isolation_mode: "shared",
        backend_instance_id: process.env.PI_BRANCH_OUT_BACKEND_INSTANCE_ID ?? null,
        backend_proxy_sha256: createHash("sha256").update(process.env.TDAI_PROXY_URL ?? "").digest("hex"),
      };
      writeFileSync(join(checkpointRoot, "checkpoint.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
      state.checkpoint_status = "ready";
      state.checkpoint = `call-${String(callIndex).padStart(3, "0")}`;
      savedCheckpoints += 1;
      lastSavedCall = callIndex;
      const checkpointBytes = Buffer.byteLength(snapshotText) + workspace.bytes;
      cumulativeCheckpointBytes += checkpointBytes;
      state.checkpoint_bytes = checkpointBytes;
      (state.sampling as Record<string, unknown>).reason = "saved";
    } catch (error) {
      state.checkpoint_status = "error";
      state.checkpoint_error = String(error);
      writeFileSync(join(checkpointRoot, "checkpoint-error.txt"), `${String(error)}\n`, "utf8");
    }
    const checkpointMs = Date.now() - checkpointStarted;
    cumulativeCheckpointMs += checkpointMs;
    state.checkpoint_save_ms = checkpointMs;
    appendJsonl(stateLog, state);
    writeSummary();
  });
}
