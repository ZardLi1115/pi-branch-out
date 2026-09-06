import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  estimateLongMemEvalTokens,
  planLongMemEvalActions,
  renderOpenClawStableContext,
  type LongMemEvalActionPlan,
  type LongMemEvalL1Item,
  type LongMemEvalSceneEntry,
} from "../tdai/longmemeval-budget.js";

type Json = Record<string, any>;
type OverlongPolicy = "error" | "skip-instance" | "truncate";

interface Options {
  data: string;
  outputRoot: string;
  runtime: string;
  coreUrl?: string;
  answerModel: string;
  judgeModel: string;
  ratios: number[];
  offset: number;
  limit?: number;
  questionId?: string;
  maxAnswerTokens: number;
  overlongPolicy: OverlongPolicy;
  pipelineSettleSeconds: number;
  pipelineTimeoutSeconds: number;
  hardCapTokens?: number;
  requireActionDiversity: boolean;
}

const ANSWER_SYSTEM = "You answer questions about a user's prior conversations. Return a concise final answer.";
const ANSWER_PROMPT_VERSION = "longmemeval-retrieved-facts-v1";

function argValue(argv: string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) return undefined;
  if (index + 1 >= argv.length) throw new Error(`${name} requires a value`);
  return argv[index + 1];
}

function requiredArg(argv: string[], name: string): string {
  const value = argValue(argv, name);
  if (!value) throw new Error(`missing required argument ${name}`);
  return value;
}

function parseNumber(raw: string | undefined, fallback: number, name: string): number {
  if (raw == null) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${name} must be numeric`);
  return value;
}

function parseOptions(argv: string[]): Options {
  const ratios = (argValue(argv, "--ratios") ?? "0,0.2,0.4,0.6,0.8,1")
    .split(",").map(Number);
  if (ratios.some((ratio) => !Number.isFinite(ratio) || ratio < 0 || ratio > 1)) {
    throw new Error("--ratios must be comma-separated values within [0,1]");
  }
  const overlongPolicy = (argValue(argv, "--overlong-policy") ?? "error") as OverlongPolicy;
  if (!["error", "skip-instance", "truncate"].includes(overlongPolicy)) {
    throw new Error("--overlong-policy must be error, skip-instance, or truncate");
  }
  const hardCap = argValue(argv, "--hard-cap-tokens");
  const limit = argValue(argv, "--limit");
  return {
    data: resolve(requiredArg(argv, "--data")),
    outputRoot: resolve(requiredArg(argv, "--output-root")),
    runtime: resolve(requiredArg(argv, "--runtime")),
    coreUrl: argValue(argv, "--core-url"),
    answerModel: argValue(argv, "--answer-model") ?? "gpt-5.6-luna",
    judgeModel: argValue(argv, "--judge-model") ?? "gpt-5.6-luna",
    ratios,
    offset: Math.floor(parseNumber(argValue(argv, "--offset"), 0, "--offset")),
    limit: limit == null ? undefined : Math.floor(parseNumber(limit, 0, "--limit")),
    questionId: argValue(argv, "--question-id"),
    maxAnswerTokens: Math.floor(parseNumber(argValue(argv, "--max-answer-tokens"), 512, "--max-answer-tokens")),
    overlongPolicy,
    pipelineSettleSeconds: parseNumber(argValue(argv, "--pipeline-settle-seconds"), 10, "--pipeline-settle-seconds"),
    pipelineTimeoutSeconds: parseNumber(argValue(argv, "--pipeline-timeout-seconds"), 1800, "--pipeline-timeout-seconds"),
    hardCapTokens: hardCap == null ? undefined : Math.floor(parseNumber(hardCap, 0, "--hard-cap-tokens")),
    requireActionDiversity: argv.includes("--require-action-diversity"),
  };
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Json).sort(([a], [b]) => a.localeCompare(b));
    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function sha256(value: string | NodeJS.ArrayBufferView): string {
  return createHash("sha256").update(value).digest("hex");
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function appendJsonl(path: string, value: unknown): void {
  appendFileSync(path, `${JSON.stringify(value)}\n`, "utf8");
}

function appendJsonlUnique(path: string, value: Json, key: (row: Json) => string): void {
  const wanted = key(value);
  const existing = _readJsonl(path);
  if (existing.some((row) => key(row) === wanted)) return;
  appendJsonl(path, value);
}

function _readJsonl(path: string): Json[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8").split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, ms));
}

async function fetchJson(url: string, init: RequestInit, attempts = 5): Promise<Json> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, { ...init, signal: AbortSignal.timeout(600_000) });
      const text = await response.text();
      let body: Json;
      try { body = text ? JSON.parse(text) : {}; } catch { body = { raw: text }; }
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${JSON.stringify(body).slice(0, 1000)}`);
      return body;
    } catch (error) {
      lastError = error;
      if (attempt < attempts) await sleep(Math.min(30_000, 1000 * 2 ** (attempt - 1)));
    }
  }
  throw lastError;
}

class CoreClient {
  constructor(
    private readonly baseUrl: string,
    private readonly userKey: string,
    private readonly teamId: string,
    private readonly userId: string,
    private readonly serviceId = "default",
  ) {}

  private headers(extra: Json = {}): Record<string, string> {
    return {
      "content-type": "application/json; charset=utf-8",
      authorization: `Bearer ${this.userKey}`,
      "x-tdai-service-id": this.serviceId,
      "x-tdai-user-key": this.userKey,
      ...extra,
    };
  }

  async post(path: string, body: Json, strictEnvelope = true, attempts = 5): Promise<Json> {
    const result = await fetchJson(`${this.baseUrl.replace(/\/$/, "")}${path}`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    }, attempts);
    if (strictEnvelope && result.code !== 0) throw new Error(`${path} failed: ${JSON.stringify(result)}`);
    return result;
  }

  isolation(agentId: string, sessionId?: string): Json {
    return {
      team_id: this.teamId,
      user_id: this.userId,
      agent_id: agentId,
      ...(sessionId ? { session_id: sessionId } : {}),
    };
  }

  async addConversation(agentId: string, sessionId: string, messages: Json[]): Promise<Json> {
    return this.post("/v3/conversation/add", {
      ...this.isolation(agentId, sessionId),
      messages,
    }, true, 1);
  }

  async recall(agentId: string, query: string): Promise<{
    l1: LongMemEvalL1Item[];
    persona: string | null;
    scenes: LongMemEvalSceneEntry[];
    raw: Json;
  }> {
    const iso = this.isolation(agentId);
    const [atomic, persona, scenarios] = await Promise.all([
      this.post("/v3/atomic/search", { ...iso, query, limit: 5 }),
      this.post("/v3/core/read", iso),
      this.post("/v3/scenario/ls", iso),
    ]);
    return {
      l1: Array.isArray(atomic?.data?.items) ? atomic.data.items : [],
      persona: typeof persona?.data?.content === "string" ? persona.data.content : null,
      scenes: Array.isArray(scenarios?.data?.entries) ? scenarios.data.entries : [],
      raw: { atomic_search: atomic, core_read: persona, scenario_ls: scenarios },
    };
  }

  async waitForPipeline(settleSeconds: number, timeoutSeconds: number): Promise<Json> {
    await sleep(Math.max(0, settleSeconds) * 1000);
    const deadline = Date.now() + timeoutSeconds * 1000;
    let stableIdle = 0;
    let last: Json = {};
    while (Date.now() < deadline) {
      last = await this.post("/v2/pipeline/status", {}, true);
      const data = last.data ?? {};
      const idle = ["l1", "l2", "l3"].every((layer) => data[layer]?.idle === true);
      stableIdle = idle ? stableIdle + 1 : 0;
      if (stableIdle >= 3) return last;
      await sleep(2000);
    }
    throw new Error(`pipeline did not settle: ${JSON.stringify(last)}`);
  }
}

function normalizeChatUrl(baseUrl: string): string {
  const clean = baseUrl.replace(/\/$/, "");
  return clean.endsWith("/chat/completions") ? clean : `${clean}/chat/completions`;
}

async function chatCompletion(args: {
  baseUrl: string;
  apiKey: string;
  model: string;
  messages: Json[];
  maxTokens: number;
}): Promise<Json> {
  return fetchJson(normalizeChatUrl(args.baseUrl), {
    method: "POST",
    headers: {
      "content-type": "application/json; charset=utf-8",
      authorization: `Bearer ${args.apiKey}`,
    },
    body: JSON.stringify({
      model: args.model,
      messages: args.messages,
      temperature: 0,
      max_tokens: args.maxTokens,
    }),
  });
}

function completionText(response: Json): string {
  const content = response?.choices?.[0]?.message?.content;
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) return content.map((part) => part?.text ?? "").join("\n").trim();
  throw new Error(`chat response has no text content: ${JSON.stringify(response).slice(0, 1000)}`);
}

function normalizedUsage(response: Json): Json {
  const usage = response?.usage ?? {};
  const prompt = Number(usage.prompt_tokens ?? usage.input_tokens ?? 0);
  const completion = Number(usage.completion_tokens ?? usage.output_tokens ?? 0);
  const cached = Number(
    usage?.prompt_tokens_details?.cached_tokens
      ?? usage?.input_tokens_details?.cached_tokens
      ?? usage.cache_read_input_tokens
      ?? 0,
  );
  return {
    input_tokens: prompt,
    output_tokens: completion,
    cache_read_tokens: cached,
    cache_write_tokens: Number(usage.cache_creation_input_tokens ?? 0),
    provider_reported_cost_usd: Number(usage.cost ?? usage.cost_usd ?? 0),
    raw: usage,
  };
}

function answerMessages(entry: Json, stableContext: string, memory: string): Json[] {
  const system = [ANSWER_SYSTEM, stableContext].filter(Boolean).join("\n\n");
  const user = [
    "I will give you several facts extracted from history chats between you and a user. Please answer the question based on the relevant facts.",
    memory,
    `Current Date: ${entry.question_date}`,
    `Question: ${entry.question}`,
    "Answer:",
  ].filter(Boolean).join("\n\n");
  return [{ role: "system", content: system }, { role: "user", content: user }];
}

export function judgePrompt(entry: Json, response: string): string {
  const question = entry.question;
  const answer = entry.answer;
  if (String(entry.question_id).includes("_abs")) {
    return `I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: ${question}\n\nExplanation: ${answer}\n\nModel Response: ${response}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.`;
  }
  if (["single-session-user", "single-session-assistant", "multi-session"].includes(entry.question_type)) {
    return `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: ${question}\n\nCorrect Answer: ${answer}\n\nModel Response: ${response}\n\nIs the model response correct? Answer yes or no only.`;
  }
  if (entry.question_type === "temporal-reasoning") {
    return `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: ${question}\n\nCorrect Answer: ${answer}\n\nModel Response: ${response}\n\nIs the model response correct? Answer yes or no only.`;
  }
  if (entry.question_type === "knowledge-update") {
    return `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: ${question}\n\nCorrect Answer: ${answer}\n\nModel Response: ${response}\n\nIs the model response correct? Answer yes or no only.`;
  }
  if (entry.question_type === "single-session-preference") {
    return `I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: ${question}\n\nRubric: ${answer}\n\nModel Response: ${response}\n\nIs the model response correct? Answer yes or no only.`;
  }
  throw new Error(`unsupported question_type: ${entry.question_type}`);
}

function parseLongMemDate(value: string, offsetMs: number): string {
  const cleaned = value.replace(/\s*\([A-Za-z]{3}\)\s*/, " ").trim().replace(/\//g, "-");
  const match = cleaned.match(/^(\d{4})-(\d{2})-(\d{2})(?:\s+(\d{2}):(\d{2}))?/);
  if (!match) throw new Error(`unsupported LongMemEval date: ${value}`);
  const [, year, month, day, hour = "00", minute = "00"] = match;
  return new Date(Date.UTC(+year, +month - 1, +day, +hour, +minute, 0, offsetMs)).toISOString();
}

export function chronologicalSessionIndices(entry: Json): number[] {
  return entry.haystack_dates
    .map((date: string, index: number) => ({ index, timestamp: Date.parse(parseLongMemDate(date, 0)) }))
    .sort((left: Json, right: Json) => left.timestamp - right.timestamp || left.index - right.index)
    .map((row: Json) => row.index);
}

function prepareMessages(entry: Json, options: Options): {
  sessions: Array<{ sourceIndex: number; sourceSessionId: string; messages: Json[] }>;
  truncations: Json[];
} {
  const sessions: Array<{ sourceIndex: number; sourceSessionId: string; messages: Json[] }> = [];
  const truncations: Json[] = [];
  for (const sessionIndex of chronologicalSessionIndices(entry)) {
    const source = entry.haystack_sessions[sessionIndex];
    const date = entry.haystack_dates[sessionIndex];
    const messages: Json[] = [];
    for (let turnIndex = 0; turnIndex < source.length; turnIndex += 1) {
      const message = source[turnIndex];
      if (!["user", "assistant"].includes(message.role)) continue;
      let content = String(message.content ?? "");
      if (content.length > 8192) {
        if (options.overlongPolicy === "error") {
          throw new Error(`message exceeds beta.1 8192-char L0 limit at session ${sessionIndex}, turn ${turnIndex}`);
        }
        if (options.overlongPolicy === "skip-instance") {
          throw new Error("SKIP_INSTANCE_OVERLONG");
        }
        truncations.push({ session_index: sessionIndex, turn_index: turnIndex, original_chars: content.length, stored_chars: 8192 });
        content = content.slice(0, 8192);
      }
      messages.push({ role: message.role, content, timestamp: parseLongMemDate(date, turnIndex) });
    }
    sessions.push({
      sourceIndex: sessionIndex,
      sourceSessionId: String(entry.haystack_session_ids[sessionIndex]),
      messages,
    });
  }
  return { sessions, truncations };
}

function actionSlug(action: number): string {
  return action.toFixed(3).replace(/0+$/, "").replace(/\.$/, "").replace(".", "p");
}

async function runItem(
  entry: Json,
  options: Options,
  runtime: Json,
  core: CoreClient,
  modelBaseUrl: string,
  modelApiKey: string,
  collectionId: string,
  sourceSha256: string,
): Promise<Json> {
  const questionId = String(entry.question_id);
  const itemRoot = join(options.outputRoot, "items", questionId);
  mkdirSync(itemRoot, { recursive: true });
  const completePath = join(itemRoot, "complete.json");
  if (existsSync(completePath)) return { question_id: questionId, status: "already-complete" };

  let prepared: ReturnType<typeof prepareMessages>;
  try {
    prepared = prepareMessages(entry, options);
  } catch (error) {
    if (String(error).includes("SKIP_INSTANCE_OVERLONG")) {
      const skipped = { question_id: questionId, status: "skipped-overlong", at: new Date().toISOString() };
      writeJson(join(itemRoot, "skipped.json"), skipped);
      return skipped;
    }
    throw error;
  }

  const agentId = `agt-lme-${sha256(`${collectionId}\0${questionId}`).slice(0, 16)}`;
  const ingestLog = join(itemRoot, "ingest.jsonl");
  const uncertainIngestPath = join(itemRoot, "ingest-uncertain.json");
  if (existsSync(uncertainIngestPath)) {
    throw new Error("previous conversation/add outcome is uncertain; use a new collection batch/namespace");
  }
  const completedBatches = new Set(
    _readJsonl(ingestLog).map((row) => `${row.source_session_id}\0${row.batch_start}`),
  );
  let acceptedMessages = 0;
  for (const session of prepared.sessions) {
    const messages = session.messages;
    const sourceSessionId = session.sourceSessionId;
    const sessionId = `lme-${sha256(`${questionId}\0${sourceSessionId}`).slice(0, 24)}`;
    for (let start = 0; start < messages.length; start += 100) {
      const batch = messages.slice(start, start + 100);
      acceptedMessages += batch.length;
      if (completedBatches.has(`${sourceSessionId}\0${start}`)) continue;
      let receipt: Json;
      try {
        // conversation/add is not idempotent in beta.1. Never retry an unknown
        // outcome: quarantine this namespace instead of silently duplicating L0.
        receipt = await core.addConversation(agentId, sessionId, batch);
      } catch (error) {
        writeJson(uncertainIngestPath, {
          source_session_id: sourceSessionId,
          memory_session_id: sessionId,
          batch_start: start,
          message_count: batch.length,
          error: error instanceof Error ? error.message : String(error),
          at: new Date().toISOString(),
        });
        throw error;
      }
      appendJsonl(ingestLog, {
        source_session_id: sourceSessionId,
        source_session_index: session.sourceIndex,
        memory_session_id: sessionId,
        batch_start: start,
        message_count: batch.length,
        accepted_ids: receipt?.data?.accepted_ids ?? [],
        request_id: receipt.request_id,
      });
    }
  }
  const snapshotPath = join(itemRoot, "candidate-snapshot.json");
  let pipelineStatus: Json;
  let recalled: Awaited<ReturnType<CoreClient["recall"]>>;
  let frozenCandidateSnapshot: Json | null = null;
  if (existsSync(snapshotPath)) {
    const frozen = JSON.parse(readFileSync(snapshotPath, "utf8"));
    frozenCandidateSnapshot = frozen;
    pipelineStatus = existsSync(join(itemRoot, "pipeline-status.json"))
      ? JSON.parse(readFileSync(join(itemRoot, "pipeline-status.json"), "utf8"))
      : {};
    recalled = {
      l1: frozen.l1 ?? [],
      persona: frozen.persona ?? null,
      scenes: frozen.scenes ?? [],
      raw: frozen.raw ?? {},
    };
  } else {
    pipelineStatus = await core.waitForPipeline(options.pipelineSettleSeconds, options.pipelineTimeoutSeconds);
    recalled = await core.recall(agentId, entry.question);
  }
  const stableContext = renderOpenClawStableContext(recalled.persona, recalled.scenes);
  const currentContextTokens = estimateLongMemEvalTokens(`${ANSWER_SYSTEM}\n${stableContext}\n${entry.question_date}\n${entry.question}`);
  const planned = planLongMemEvalActions({
    items: recalled.l1,
    ratios: options.ratios,
    hardCapTokens: options.hardCapTokens,
    currentContextTokens,
  });
  const candidateSnapshot = frozenCandidateSnapshot ?? {
    version: 1,
    tdai_version: runtime.TDAI_VERSION,
    question_id: questionId,
    query: entry.question,
    agent_id: agentId,
    l1: recalled.l1,
    persona: recalled.persona,
    scenes: recalled.scenes,
    raw: recalled.raw,
    captured_at: new Date().toISOString(),
  };
  const snapshotId = `sha256:${sha256(stableJson(candidateSnapshot))}`;
  const state = {
    benchmark: "LongMemEval",
    schema_version: "longmemeval-budget-state-v1",
    question_id: questionId,
    question_type: entry.question_type,
    question: entry.question,
    question_date: entry.question_date,
    source_file: basename(options.data),
    history_session_count: entry.haystack_sessions.length,
    history_message_count: acceptedMessages,
    history_ingest_order: "haystack_dates-ascending-stable-v1",
    candidate_snapshot_id: snapshotId,
    l1_count: recalled.l1.length,
    l1_lengths: recalled.l1.map((item) => item.content.length),
    l1_scores: recalled.l1.map((item) => item.score ?? null),
    persona_chars: recalled.persona?.length ?? 0,
    scene_count: recalled.scenes.length,
    candidate_tokens: planned.candidateTokens,
    candidate_memory_tokens: planned.candidateTokens,
    candidate_count: recalled.l1.length,
    l0_count: 0,
    feasible_budget_tokens: planned.feasibleBudgetTokens,
    context_tokens: currentContextTokens,
    context_window_tokens: 524288,
    reserve_tokens: 16384,
    context_tokens_before_injection: currentContextTokens,
    action_table_version: "budget-ratios-v1",
    allocator_version: "openclaw-beta1-complete-render-v1",
    tdai_version: runtime.TDAI_VERSION,
    answer_model: options.answerModel,
    answer_prompt_version: ANSWER_PROMPT_VERSION,
    query: entry.question,
    recent_tool_result: "",
    default_actual_memory_tokens: planned.plans.at(-1)?.injectedTokens ?? 0,
    previous_actual_memory_tokens: 0,
    previous_mapped_action: 1,
    previous_budget_tokens: planned.plans.at(-1)?.budgetTokens ?? 0,
  };
  const stateId = `sha256:${sha256(stableJson(state))}`;
  writeJson(join(itemRoot, "source-reference.json"), {
    question_id: questionId,
    source_file: options.data,
    source_sha256: sourceSha256,
    history_sha256: sha256(stableJson(entry.haystack_sessions)),
    answer_session_ids: entry.answer_session_ids,
    truncations: prepared.truncations,
  });
  writeJson(join(itemRoot, "pipeline-status.json"), pipelineStatus);
  if (!existsSync(snapshotPath)) writeJson(snapshotPath, candidateSnapshot);
  writeJson(join(itemRoot, "state.json"), { state_id: stateId, ...state });
  writeJson(join(itemRoot, "action-plan.json"), planned);

  const byEffective = new Map<string, LongMemEvalActionPlan[]>();
  for (const plan of planned.plans) {
    const group = byEffective.get(plan.effectiveActionId) ?? [];
    group.push(plan);
    byEffective.set(plan.effectiveActionId, group);
  }
  if (options.requireActionDiversity && byEffective.size < 2) {
    throw new Error("no action diversity: all budget ratios render identical L1 content");
  }
  const hasActionDiversity = byEffective.size >= 2;
  const results: Json[] = [];
  const existingSamples = _readJsonl(join(itemRoot, "samples.jsonl"));
  for (const [effectiveActionId, aliases] of byEffective) {
    const plan = aliases[0];
    const existing = existingSamples.find((item) => item.effective_action_id === effectiveActionId);
    if (existing) {
      results.push(existing);
      continue;
    }
    const messages = answerMessages(entry, stableContext, plan.renderedMemory);
    const answerResponse = await chatCompletion({
      baseUrl: modelBaseUrl,
      apiKey: modelApiKey,
      model: options.answerModel,
      messages,
      maxTokens: options.maxAnswerTokens,
    });
    const hypothesis = completionText(answerResponse);
    const scoringPrompt = judgePrompt(entry, hypothesis);
    const judgeResponse = await chatCompletion({
      baseUrl: modelBaseUrl,
      apiKey: modelApiKey,
      model: options.judgeModel,
      messages: [{ role: "user", content: scoringPrompt }],
      maxTokens: 10,
    });
    const judgeText = completionText(judgeResponse);
    const reward = judgeText.toLowerCase().includes("yes") ? 1 : 0;
    const actionDir = join(itemRoot, "actions", effectiveActionId.replace(":", "-"));
    mkdirSync(actionDir, { recursive: true });
    writeJson(join(actionDir, "request.json"), {
      model: options.answerModel,
      messages,
      max_tokens: options.maxAnswerTokens,
    });
    writeJson(join(actionDir, "response.json"), {
      hypothesis,
      usage: normalizedUsage(answerResponse),
      response_id: answerResponse.id ?? null,
    });
    writeJson(join(actionDir, "judge.json"), {
      model: options.judgeModel,
      prompt: scoringPrompt,
      response: judgeText,
      reward,
      usage: normalizedUsage(judgeResponse),
      protocol: options.judgeModel === "gpt-4o-2024-08-06" ? "official" : "official-prompt-custom-judge",
    });
    const result = {
      state_id: stateId,
      question_id: questionId,
      action: plan.action,
      action_aliases: aliases.map((item) => item.action),
      effective_action_id: effectiveActionId,
      budget_tokens: plan.budgetTokens,
      injected_tokens: plan.injectedTokens,
      selected_l1_ids: plan.selectedL1Ids,
      hypothesis,
      reward,
      answer_usage: normalizedUsage(answerResponse),
      judge_usage: normalizedUsage(judgeResponse),
      done: true,
      truncated: false,
      training_eligible: hasActionDiversity,
    };
    results.push(result);
    appendJsonl(join(itemRoot, "samples.jsonl"), result);
  }

  for (const plan of planned.plans) {
    const result = results.find((item) => item.effective_action_id === plan.effectiveActionId)!;
    appendJsonlUnique(join(options.outputRoot, "hypotheses", `action-${actionSlug(plan.action)}.jsonl`), {
      question_id: questionId,
      hypothesis: result.hypothesis,
      effective_action_id: plan.effectiveActionId,
    }, (row) => `${row.question_id}\0${row.effective_action_id}`);
  }
  const completed = {
    question_id: questionId,
    status: "complete",
    state_id: stateId,
    candidate_snapshot_id: snapshotId,
    action_count: planned.plans.length,
    effective_action_count: results.length,
    rewards: results.map((item) => ({ action_aliases: item.action_aliases, reward: item.reward })),
    completed_at: new Date().toISOString(),
  };
  writeJson(completePath, completed);
  return completed;
}

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  mkdirSync(options.outputRoot, { recursive: true });
  mkdirSync(join(options.outputRoot, "items"), { recursive: true });
  mkdirSync(join(options.outputRoot, "hypotheses"), { recursive: true });
  const runtime = JSON.parse(readFileSync(options.runtime, "utf8"));
  if (runtime.TDAI_PROMPT_MODE !== "chat") {
    throw new Error("LongMemEval requires a dedicated TDAI runtime with TDAI_PROMPT_MODE=chat");
  }
  const coreUrl = options.coreUrl ?? runtime.TDAI_CORE_URL;
  if (!coreUrl) throw new Error("MemoryCore URL missing (--core-url or runtime TDAI_CORE_URL)");
  const userKey = runtime.TDAI_USER_KEY;
  const teamId = runtime.TDAI_TEAM_ID;
  const userId = runtime.TDAI_USER_ID;
  if (!userKey || !teamId || !userId) throw new Error("runtime must contain TDAI_USER_KEY, TDAI_TEAM_ID, TDAI_USER_ID");
  const modelBaseUrl = process.env.OPENAI_BASE_URL ?? process.env.OPENAI_API_BASE;
  const modelApiKey = process.env.OPENAI_API_KEY;
  if (!modelBaseUrl || !modelApiKey) throw new Error("OPENAI_BASE_URL and OPENAI_API_KEY are required");
  const entries = JSON.parse(readFileSync(options.data, "utf8"));
  if (!Array.isArray(entries)) throw new Error("LongMemEval data must be a JSON array");
  let selected = entries as Json[];
  if (options.questionId) selected = selected.filter((entry) => entry.question_id === options.questionId);
  else selected = selected.slice(options.offset, options.limit == null ? undefined : options.offset + options.limit);
  if (selected.length === 0) throw new Error("no LongMemEval entries selected");

  const sourceSha256 = sha256(readFileSync(options.data));
  const manifestPath = join(options.outputRoot, "dataset-manifest.json");
  if (!existsSync(manifestPath)) {
    writeJson(manifestPath, {
      benchmark: "LongMemEval",
      schema_version: "longmemeval-budget-collection-v1",
      data_file: options.data,
      data_sha256: sourceSha256,
      collection_id: `lme-${sha256(`${sourceSha256}\0${Date.now()}\0${Math.random()}`).slice(0, 16)}`,
      tdai_version: runtime.TDAI_VERSION,
      tdai_prompt_mode: runtime.TDAI_PROMPT_MODE,
      answer_model: options.answerModel,
      answer_prompt_version: ANSWER_PROMPT_VERSION,
      judge_model: options.judgeModel,
      judge_protocol: options.judgeModel === "gpt-4o-2024-08-06" ? "official" : "official-prompt-custom-judge",
      action_ratios: options.ratios,
      allocator_version: "openclaw-beta1-complete-render-v1",
      openclaw_recall_max_results: 5,
      history_ingest_order: "haystack_dates-ascending-stable-v1",
      fixed_layers: ["L2", "L3"],
      controlled_layers: ["L1"],
      overlong_policy: options.overlongPolicy,
      runtime_instance: runtime.instance_name,
      created_at: new Date().toISOString(),
    });
  }
  const collectionManifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (collectionManifest.data_sha256 !== sourceSha256) throw new Error("existing batch uses a different source dataset");
  if (stableJson(collectionManifest.action_ratios) !== stableJson(options.ratios)) {
    throw new Error("existing batch uses a different action table");
  }
  if (collectionManifest.answer_prompt_version !== ANSWER_PROMPT_VERSION) {
    throw new Error("existing batch uses a different answer prompt version");
  }
  if (collectionManifest.history_ingest_order !== "haystack_dates-ascending-stable-v1") {
    throw new Error("existing batch uses a different history ingest order");
  }
  if (collectionManifest.answer_model !== options.answerModel || collectionManifest.judge_model !== options.judgeModel) {
    throw new Error("existing batch uses different answer/judge models");
  }
  const core = new CoreClient(coreUrl, userKey, teamId, userId);
  for (const entry of selected) {
    const startedAt = Date.now();
    try {
      const result = await runItem(
        entry, options, runtime, core, modelBaseUrl, modelApiKey,
        collectionManifest.collection_id, sourceSha256,
      );
      appendJsonl(join(options.outputRoot, "status.jsonl"), { ...result, elapsed_ms: Date.now() - startedAt });
      process.stdout.write(`${JSON.stringify(result)}\n`);
    } catch (error) {
      const failure = {
        question_id: entry.question_id,
        status: "failed",
        error: error instanceof Error ? error.message : String(error),
        elapsed_ms: Date.now() - startedAt,
        at: new Date().toISOString(),
      };
      appendJsonl(join(options.outputRoot, "status.jsonl"), failure);
      process.stderr.write(`${JSON.stringify(failure)}\n`);
      throw error;
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
    process.exitCode = 1;
  });
}
