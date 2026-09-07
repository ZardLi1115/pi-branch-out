import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { chronologicalSessionIndices, judgePrompt } from "./collect-longmemeval.js";

type Json = Record<string, any>;
type OverlongPolicy = "error" | "skip-instance" | "truncate";

interface Options {
  data: string;
  outputRoot: string;
  runtime: string;
  questionIdsFile?: string;
  questionId?: string;
  offset: number;
  limit?: number;
  agentModel: string;
  judgeModel: string;
  maxAgentTurns: number;
  maxAnswerTokens: number;
  searchLimit: number;
  pipelineSettleSeconds: number;
  pipelineTimeoutSeconds: number;
  overlongPolicy: OverlongPolicy;
  continueOnError: boolean;
}

const AGENT_SYSTEM = [
  "Answer questions about the user's prior conversations.",
  "No memory is automatically included. Use the available read-only memory tools when the question depends on history.",
  "Search iteratively when needed, prefer precise evidence, and return a concise final answer without describing tool usage.",
].join(" ");
const PROMPT_VERSION = "longmemeval-native-agent-search-v1";
const TOOL_PROTOCOL = "tdai-v201-memory-bridge-readonly-v1";

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

function numeric(raw: string | undefined, fallback: number, name: string): number {
  const value = raw == null ? fallback : Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${name} must be numeric`);
  return value;
}

function parseOptions(argv: string[]): Options {
  const limit = argValue(argv, "--limit");
  const overlongPolicy = (argValue(argv, "--overlong-policy") ?? "error") as OverlongPolicy;
  if (!["error", "skip-instance", "truncate"].includes(overlongPolicy)) {
    throw new Error("--overlong-policy must be error, skip-instance, or truncate");
  }
  return {
    data: resolve(requiredArg(argv, "--data")),
    outputRoot: resolve(requiredArg(argv, "--output-root")),
    runtime: resolve(requiredArg(argv, "--runtime")),
    questionIdsFile: argValue(argv, "--question-ids-file"),
    questionId: argValue(argv, "--question-id"),
    offset: Math.floor(numeric(argValue(argv, "--offset"), 0, "--offset")),
    limit: limit == null ? undefined : Math.floor(numeric(limit, 0, "--limit")),
    agentModel: argValue(argv, "--agent-model") ?? "gpt-5.6-luna",
    judgeModel: argValue(argv, "--judge-model") ?? "gpt-5.6-luna",
    maxAgentTurns: Math.floor(numeric(argValue(argv, "--max-agent-turns"), 4, "--max-agent-turns")),
    maxAnswerTokens: Math.floor(numeric(argValue(argv, "--max-answer-tokens"), 512, "--max-answer-tokens")),
    searchLimit: Math.floor(numeric(argValue(argv, "--search-limit"), 5, "--search-limit")),
    pipelineSettleSeconds: numeric(argValue(argv, "--pipeline-settle-seconds"), 10, "--pipeline-settle-seconds"),
    pipelineTimeoutSeconds: numeric(argValue(argv, "--pipeline-timeout-seconds"), 1800, "--pipeline-timeout-seconds"),
    overlongPolicy,
    continueOnError: argv.includes("--continue-on-error"),
  };
}

function sha256(value: string | NodeJS.ArrayBufferView): string {
  return createHash("sha256").update(value).digest("hex");
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Json).sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function appendJsonl(path: string, value: unknown): void {
  appendFileSync(path, `${JSON.stringify(value)}\n`, "utf8");
}

function readJsonl(path: string): Json[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8").split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, ms));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function fetchJson(url: string, init: RequestInit, attempts = 5): Promise<Json> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, { ...init, signal: AbortSignal.timeout(600_000) });
      const text = await response.text();
      let body: Json;
      try { body = text ? JSON.parse(text) : {}; } catch { body = { raw: text }; }
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${JSON.stringify(body).slice(0, 2000)}`);
      return body;
    } catch (error) {
      lastError = error;
      if (attempt < attempts) await sleep(Math.min(30_000, 1000 * 2 ** (attempt - 1)));
    }
  }
  throw lastError;
}

function normalizedUsage(response: Json): Json {
  const usage = response?.usage ?? {};
  return {
    input_tokens: Number(usage.prompt_tokens ?? usage.input_tokens ?? 0),
    output_tokens: Number(usage.completion_tokens ?? usage.output_tokens ?? 0),
    cache_read_tokens: Number(
      usage?.prompt_tokens_details?.cached_tokens
      ?? usage?.input_tokens_details?.cached_tokens
      ?? usage.cache_read_input_tokens
      ?? 0,
    ),
    cache_write_tokens: Number(usage.cache_creation_input_tokens ?? 0),
    provider_reported_cost_usd: Number(usage.cost ?? usage.cost_usd ?? 0),
    raw: usage,
  };
}

function parseLongMemDate(value: string, offsetMs: number): string {
  const cleaned = value.replace(/\s*\([A-Za-z]{3}\)\s*/, " ").trim().replace(/\//g, "-");
  const match = cleaned.match(/^(\d{4})-(\d{2})-(\d{2})(?:\s+(\d{2}):(\d{2}))?/);
  if (!match) throw new Error(`unsupported LongMemEval date: ${value}`);
  const [, year, month, day, hour = "00", minute = "00"] = match;
  return new Date(Date.UTC(+year, +month - 1, +day, +hour, +minute, 0, offsetMs)).toISOString();
}

function prepareSessions(entry: Json, policy: OverlongPolicy): { sessions: Json[]; truncations: Json[] } {
  const sessions: Json[] = [];
  const truncations: Json[] = [];
  for (const index of chronologicalSessionIndices(entry)) {
    const messages: Json[] = [];
    for (let turn = 0; turn < entry.haystack_sessions[index].length; turn += 1) {
      const source = entry.haystack_sessions[index][turn];
      if (!["user", "assistant"].includes(source.role)) continue;
      let content = String(source.content ?? "");
      if (content.length > 8192) {
        if (policy === "error") throw new Error(`message exceeds 8192 characters at session ${index}, turn ${turn}`);
        if (policy === "skip-instance") throw new Error("SKIP_INSTANCE_OVERLONG");
        truncations.push({ session_index: index, turn_index: turn, original_chars: content.length, stored_chars: 8192 });
        content = content.slice(0, 8192);
      }
      messages.push({ role: source.role, content, timestamp: parseLongMemDate(entry.haystack_dates[index], turn) });
    }
    sessions.push({ source_index: index, source_session_id: String(entry.haystack_session_ids[index]), messages });
  }
  return { sessions, truncations };
}

class CoreClient {
  constructor(
    private readonly baseUrl: string,
    private readonly userKey: string,
    private readonly teamId: string,
    private readonly userId: string,
  ) {}

  async post(path: string, body: Json, attempts = 5): Promise<Json> {
    const result = await fetchJson(`${this.baseUrl.replace(/\/$/, "")}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json; charset=utf-8",
        authorization: `Bearer ${this.userKey}`,
        "x-tdai-service-id": "default",
        "x-tdai-user-key": this.userKey,
      },
      body: JSON.stringify(body),
    }, attempts);
    if (result.code !== 0) throw new Error(`${path} failed: ${JSON.stringify(result).slice(0, 2000)}`);
    return result;
  }

  async createNamespace(questionId: string, namespaceId: string): Promise<Json> {
    const agentResult = await this.post("/v3/meta/agent/create", {
      team_id: this.teamId,
      name: `LongMemEval-${questionId}-${namespaceId.slice(-6)}`,
      description: "LongMemEval native agent-search isolated namespace",
      owner_user_id: this.userId,
      visibility: "team",
    }, 1);
    const agentId = String(agentResult?.data?.agent?.agent_id ?? agentResult?.data?.agent_id ?? "");
    if (!agentId) throw new Error("agent/create returned no agent_id");
    const taskResult = await this.post("/v3/meta/task/create", {
      team_id: this.teamId,
      creator_user_id: this.userId,
      title: `LongMemEval ${questionId}`,
      description: "Native no-injection explicit-search evaluation",
    }, 1);
    const taskId = String(taskResult?.data?.task?.task_id ?? taskResult?.data?.task_id ?? "");
    if (!taskId) throw new Error("task/create returned no task_id");
    await this.post("/v3/meta/task-agent/link", {
      task_id: taskId,
      agent_id: agentId,
      role_in_task: "primary",
    }, 1);
    return { agent_id: agentId, task_id: taskId };
  }

  async addConversation(agentId: string, taskId: string, sessionId: string, messages: Json[]): Promise<Json> {
    return this.post("/v3/conversation/add", {
      team_id: this.teamId,
      user_id: this.userId,
      agent_id: agentId,
      task_id: taskId,
      session_id: sessionId,
      messages,
    }, 1);
  }

  async waitForPipeline(settleSeconds: number, timeoutSeconds: number): Promise<Json> {
    await sleep(Math.max(0, settleSeconds) * 1000);
    const deadline = Date.now() + timeoutSeconds * 1000;
    let stable = 0;
    let last: Json = {};
    while (Date.now() < deadline) {
      last = await this.post("/v2/pipeline/status", {});
      const data = last.data ?? {};
      const idle = ["l1", "l2", "l3"].every((layer) => data[layer]?.idle === true);
      stable = idle ? stable + 1 : 0;
      if (stable >= 3) return last;
      await sleep(2000);
    }
    throw new Error(`pipeline did not settle: ${JSON.stringify(last)}`);
  }
}

const TOOLS: Json[] = [
  {
    type: "function",
    function: {
      name: "tdai_memory_search",
      description: "Search extracted long-term memories (L1) for facts relevant to the question.",
      parameters: {
        type: "object", additionalProperties: false, required: ["query"],
        properties: { query: { type: "string" }, limit: { type: "integer", minimum: 1, maximum: 20 } },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "tdai_conversation_search",
      description: "Search original conversation messages (L0) when exact wording or details are needed.",
      parameters: {
        type: "object", additionalProperties: false, required: ["query"],
        properties: { query: { type: "string" }, limit: { type: "integer", minimum: 1, maximum: 20 } },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "tdai_scenario_list",
      description: "List available memory scene paths (L2) to discover organized historical topics.",
      parameters: { type: "object", additionalProperties: false, properties: { path_prefix: { type: "string" } } },
    },
  },
  {
    type: "function",
    function: {
      name: "tdai_scenario_read",
      description: "Read one memory scene (L2) by its exact path.",
      parameters: {
        type: "object", additionalProperties: false, required: ["path"],
        properties: { path: { type: "string" } },
      },
    },
  },
];

class AgentClient {
  constructor(
    private readonly proxyUrl: string,
    private readonly userKey: string,
    private readonly teamId: string,
    private readonly agentId: string,
    private readonly taskId: string,
    private readonly conversationId: string,
    private readonly model: string,
    private readonly searchLimit: number,
  ) {}

  private headers(): Record<string, string> {
    return {
      "content-type": "application/json; charset=utf-8",
      authorization: `Bearer ${this.userKey}`,
      "x-tdai-user-key": this.userKey,
      "x-tdai-service-id": "default",
      "x-conversation-id": this.conversationId,
      "x-team-id": this.teamId,
      "x-agent-id": this.agentId,
      "x-task-id": this.taskId,
    };
  }

  async complete(messages: Json[], maxTokens: number, toolChoice: "auto" | "none"): Promise<Json> {
    return fetchJson(`${this.proxyUrl.replace(/\/$/, "")}/codebuddy/default/v1/chat/completions`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({
        model: this.model,
        messages,
        tools: TOOLS,
        tool_choice: toolChoice,
        temperature: 0,
        max_tokens: maxTokens,
      }),
    });
  }

  async executeTool(call: Json): Promise<Json> {
    const name = String(call?.function?.name ?? "");
    let args: Json;
    try { args = JSON.parse(String(call?.function?.arguments ?? "{}")); }
    catch { throw new Error(`invalid tool arguments for ${name}`); }
    let path: string;
    let body: Json;
    if (name === "tdai_memory_search" || name === "tdai_conversation_search") {
      const query = String(args.query ?? "").trim().slice(0, 2048);
      if (!query) throw new Error(`${name} requires a non-empty query`);
      const limit = Math.max(1, Math.min(20, Number(args.limit ?? this.searchLimit)));
      path = name === "tdai_memory_search" ? "atomic/search" : "conversation/search";
      body = { query, limit };
    } else if (name === "tdai_scenario_list") {
      path = "scenario/ls";
      body = typeof args.path_prefix === "string" ? { path_prefix: args.path_prefix } : {};
    } else if (name === "tdai_scenario_read") {
      const scenePath = String(args.path ?? "").trim();
      if (!scenePath) throw new Error("tdai_scenario_read requires path");
      path = "scenario/read";
      body = { path: scenePath };
    } else {
      throw new Error(`unsupported tool: ${name}`);
    }
    const response = await fetchJson(`${this.proxyUrl.replace(/\/$/, "")}/memory-bridge/v3/${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json; charset=utf-8",
        "x-conversation-id": `codebuddy:${this.conversationId}`,
        "x-tdai-service-id": "default",
      },
      body: JSON.stringify(body),
    });
    if (response.code !== 0) throw new Error(`${name} failed: ${JSON.stringify(response).slice(0, 2000)}`);
    return { tool: name, request: body, data: response.data ?? {} };
  }
}

async function directJudge(baseUrl: string, apiKey: string, model: string, prompt: string): Promise<Json> {
  const clean = baseUrl.replace(/\/$/, "");
  return fetchJson(clean.endsWith("/chat/completions") ? clean : `${clean}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json; charset=utf-8", authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }], temperature: 0, max_tokens: 10 }),
  });
}

function responseMessage(response: Json): Json {
  const message = response?.choices?.[0]?.message;
  if (!message || typeof message !== "object") {
    throw new Error(`chat response has no assistant message: ${JSON.stringify(response).slice(0, 2000)}`);
  }
  return message;
}

function messageText(message: Json): string {
  if (typeof message.content === "string") return message.content.trim();
  if (Array.isArray(message.content)) return message.content.map((part: Json) => part?.text ?? "").join("\n").trim();
  return "";
}

async function runItem(
  entry: Json,
  options: Options,
  runtime: Json,
  core: CoreClient,
  judgeBaseUrl: string,
  judgeApiKey: string,
  collectionId: string,
  sourceSha256: string,
): Promise<Json> {
  const questionId = String(entry.question_id);
  const itemRoot = join(options.outputRoot, "items", questionId);
  mkdirSync(itemRoot, { recursive: true });
  const completePath = join(itemRoot, "complete.json");
  if (existsSync(completePath)) return { question_id: questionId, status: "already-complete" };

  const namespacePath = join(itemRoot, "namespace.json");
  let namespace: Json;
  if (existsSync(namespacePath)) {
    namespace = JSON.parse(readFileSync(namespacePath, "utf8"));
  } else {
    const namespaceId = `ns-${sha256(`${collectionId}\0${questionId}\0${Date.now()}\0${Math.random()}`).slice(0, 16)}`;
    const created = await core.createNamespace(questionId, namespaceId);
    namespace = { schema_version: "longmemeval-agent-namespace-v1", namespace_id: namespaceId, ...created };
    writeJson(namespacePath, namespace);
  }

  let prepared: ReturnType<typeof prepareSessions>;
  try { prepared = prepareSessions(entry, options.overlongPolicy); }
  catch (error) {
    if (String(error).includes("SKIP_INSTANCE_OVERLONG")) {
      const skipped = { question_id: questionId, status: "skipped-overlong", at: new Date().toISOString() };
      writeJson(join(itemRoot, "skipped.json"), skipped);
      return skipped;
    }
    throw error;
  }
  const ingestPath = join(itemRoot, "ingest.jsonl");
  const uncertainPath = join(itemRoot, "ingest-uncertain.json");
  if (existsSync(uncertainPath)) throw new Error("previous conversation/add outcome is uncertain; quarantine namespace");
  const completedBatches = new Set(readJsonl(ingestPath).map((row) => `${row.source_session_id}\0${row.batch_start}`));
  let historyMessages = 0;
  for (const session of prepared.sessions) {
    const memorySessionId = `lme-${sha256(`${questionId}\0${namespace.namespace_id}\0${session.source_session_id}`).slice(0, 24)}`;
    for (let start = 0; start < session.messages.length; start += 100) {
      const batch = session.messages.slice(start, start + 100);
      historyMessages += batch.length;
      if (completedBatches.has(`${session.source_session_id}\0${start}`)) continue;
      try {
        const receipt = await core.addConversation(namespace.agent_id, namespace.task_id, memorySessionId, batch);
        appendJsonl(ingestPath, {
          source_session_id: session.source_session_id,
          source_session_index: session.source_index,
          memory_session_id: memorySessionId,
          batch_start: start,
          message_count: batch.length,
          accepted_ids: receipt?.data?.accepted_ids ?? [],
          request_id: receipt.request_id,
        });
      } catch (error) {
        writeJson(uncertainPath, {
          source_session_id: session.source_session_id,
          memory_session_id: memorySessionId,
          batch_start: start,
          message_count: batch.length,
          error: errorMessage(error),
          at: new Date().toISOString(),
        });
        throw error;
      }
    }
  }
  const pipelineStatus = await core.waitForPipeline(options.pipelineSettleSeconds, options.pipelineTimeoutSeconds);
  writeJson(join(itemRoot, "pipeline-status.json"), pipelineStatus);
  writeJson(join(itemRoot, "source-reference.json"), {
    question_id: questionId,
    source_file: options.data,
    source_sha256: sourceSha256,
    history_sha256: sha256(stableJson(entry.haystack_sessions)),
    answer_session_ids: entry.answer_session_ids,
    truncations: prepared.truncations,
  });

  if (existsSync(join(itemRoot, "agent-started.json"))) {
    throw new Error("incomplete agent attempt cannot be replayed in the same namespace; quarantine it");
  }
  const conversationId = `lme-agent-${sha256(`${questionId}\0${namespace.namespace_id}`).slice(0, 20)}`;
  writeJson(join(itemRoot, "agent-started.json"), { conversation_id: conversationId, at: new Date().toISOString() });
  const agent = new AgentClient(
    runtime.TDAI_PROXY_URL,
    runtime.TDAI_USER_KEY,
    runtime.TDAI_TEAM_ID,
    namespace.agent_id,
    namespace.task_id,
    conversationId,
    options.agentModel,
    options.searchLimit,
  );
  const messages: Json[] = [
    { role: "system", content: AGENT_SYSTEM },
    { role: "user", content: `Current Date: ${entry.question_date}\n\nQuestion: ${entry.question}` },
  ];
  const usageRows: Json[] = [];
  const toolTracePath = join(itemRoot, "tool-trace.jsonl");
  let finalAnswer = "";
  let forcedFinal = false;
  for (let turn = 0; turn < options.maxAgentTurns; turn += 1) {
    const response = await agent.complete(messages, options.maxAnswerTokens, "auto");
    const assistant = responseMessage(response);
    const usage = normalizedUsage(response);
    usageRows.push(usage);
    appendJsonl(join(itemRoot, "usage.jsonl"), { kind: "agent", turn, ...usage });
    const toolCalls = Array.isArray(assistant.tool_calls) ? assistant.tool_calls : [];
    appendJsonl(toolTracePath, { kind: "assistant", turn, message: assistant, usage, response_id: response.id ?? null });
    messages.push({ role: "assistant", content: assistant.content ?? null, ...(toolCalls.length ? { tool_calls: toolCalls } : {}) });
    if (toolCalls.length === 0) {
      finalAnswer = messageText(assistant);
      break;
    }
    for (const call of toolCalls) {
      let result: Json;
      try { result = await agent.executeTool(call); }
      catch (error) { result = { tool: call?.function?.name, error: errorMessage(error) }; }
      appendJsonl(toolTracePath, { kind: "tool", turn, tool_call_id: call.id, result });
      messages.push({ role: "tool", tool_call_id: call.id, name: call?.function?.name, content: JSON.stringify(result) });
    }
  }
  if (!finalAnswer) {
    forcedFinal = true;
    const response = await agent.complete(messages, options.maxAnswerTokens, "none");
    const assistant = responseMessage(response);
    const usage = normalizedUsage(response);
    usageRows.push(usage);
    appendJsonl(join(itemRoot, "usage.jsonl"), { kind: "agent-forced-final", turn: options.maxAgentTurns, ...usage });
    appendJsonl(toolTracePath, {
      kind: "assistant-forced-final", turn: options.maxAgentTurns,
      message: assistant, usage, response_id: response.id ?? null,
    });
    finalAnswer = messageText(assistant);
  }
  if (!finalAnswer) throw new Error("agent produced no final answer");

  const prompt = judgePrompt(entry, finalAnswer);
  const judgeResponse = await directJudge(judgeBaseUrl, judgeApiKey, options.judgeModel, prompt);
  const judgeMessage = responseMessage(judgeResponse);
  const judgeText = messageText(judgeMessage);
  const reward = judgeText.toLowerCase().includes("yes") ? 1 : 0;
  const judgeUsage = normalizedUsage(judgeResponse);
  appendJsonl(join(itemRoot, "usage.jsonl"), { kind: "judge", ...judgeUsage });
  writeJson(join(itemRoot, "answer.json"), {
    hypothesis: finalAnswer,
    model: options.agentModel,
    model_calls: usageRows.length,
    tool_calls: readJsonl(toolTracePath).filter((row) => row.kind === "tool").length,
    forced_final: forcedFinal,
    usage: usageRows,
  });
  writeJson(join(itemRoot, "judge.json"), {
    model: options.judgeModel,
    prompt,
    response: judgeText,
    reward,
    usage: judgeUsage,
    protocol: options.judgeModel === "gpt-4o-2024-08-06" ? "official" : "official-prompt-custom-judge",
  });
  const completed = {
    question_id: questionId,
    status: "complete",
    reward,
    agent_model_calls: usageRows.length,
    tool_calls: readJsonl(toolTracePath).filter((row) => row.kind === "tool").length,
    history_message_count: historyMessages,
    completed_at: new Date().toISOString(),
  };
  writeJson(completePath, completed);
  appendJsonl(join(options.outputRoot, "hypotheses.jsonl"), { question_id: questionId, hypothesis: finalAnswer, reward });
  return completed;
}

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  if (options.maxAgentTurns <= 0 || options.searchLimit <= 0) throw new Error("turn and search limits must be positive");
  mkdirSync(options.outputRoot, { recursive: true });
  mkdirSync(join(options.outputRoot, "items"), { recursive: true });
  const runtime = JSON.parse(readFileSync(options.runtime, "utf8"));
  if (runtime.TDAI_VERSION !== "v2.0.1") throw new Error("agent-search collection requires TDAI v2.0.1 runtime");
  if (runtime.TDAI_INJECTION_DISABLED !== true) throw new Error("agent-search runtime must explicitly disable injection");
  const judgeBaseUrl = process.env.OPENAI_BASE_URL ?? process.env.OPENAI_API_BASE;
  const judgeApiKey = process.env.OPENAI_API_KEY;
  if (!judgeBaseUrl || !judgeApiKey) throw new Error("OPENAI_BASE_URL and OPENAI_API_KEY are required for judge");
  const entries = JSON.parse(readFileSync(options.data, "utf8"));
  if (!Array.isArray(entries)) throw new Error("LongMemEval data must be an array");
  let selected = entries as Json[];
  let selectionSha256: string | null = null;
  if (options.questionId && options.questionIdsFile) throw new Error("use one selection mode");
  if (options.questionId) {
    selected = selected.filter((entry) => String(entry.question_id) === options.questionId);
  } else if (options.questionIdsFile) {
    const selectionText = readFileSync(resolve(options.questionIdsFile), "utf8");
    selectionSha256 = sha256(selectionText);
    const parsed = JSON.parse(selectionText);
    const ids = Array.isArray(parsed) ? parsed : parsed.question_ids;
    if (!Array.isArray(ids)) throw new Error("selection file must contain question_ids");
    const byId = new Map(selected.map((entry) => [String(entry.question_id), entry]));
    selected = ids.map((id: unknown) => byId.get(String(id))).filter(Boolean) as Json[];
  }
  selected = selected.slice(options.offset, options.limit == null ? undefined : options.offset + options.limit);
  if (!selected.length) throw new Error("no LongMemEval entries selected");

  const sourceSha256 = sha256(readFileSync(options.data));
  const manifestPath = join(options.outputRoot, "dataset-manifest.json");
  if (!existsSync(manifestPath)) {
    writeJson(manifestPath, {
      benchmark: "LongMemEval",
      schema_version: "longmemeval-agent-search-collection-v1",
      collection_mode: "native-agent-search-no-injection-v1",
      data_file: options.data,
      data_sha256: sourceSha256,
      selection_sha256: selectionSha256,
      collection_id: `lme-agent-${sha256(`${sourceSha256}\0${Date.now()}\0${Math.random()}`).slice(0, 16)}`,
      tdai_version: runtime.TDAI_VERSION,
      runtime_instance: runtime.instance_name,
      injection_disabled: runtime.TDAI_INJECTION_DISABLED,
      agent_model: options.agentModel,
      answer_prompt_version: PROMPT_VERSION,
      judge_model: options.judgeModel,
      judge_protocol: options.judgeModel === "gpt-4o-2024-08-06" ? "official" : "official-prompt-custom-judge",
      tool_protocol: TOOL_PROTOCOL,
      tools: TOOLS.map((tool) => tool.function.name),
      max_agent_turns: options.maxAgentTurns,
      search_limit: options.searchLimit,
      history_ingest_order: "haystack_dates-ascending-stable-v1",
      overlong_policy: options.overlongPolicy,
      created_at: new Date().toISOString(),
    });
  }
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (manifest.data_sha256 !== sourceSha256 || (manifest.selection_sha256 ?? null) !== selectionSha256) {
    throw new Error("existing batch source/selection mismatch");
  }
  if (manifest.tdai_version !== runtime.TDAI_VERSION || manifest.runtime_instance !== runtime.instance_name) {
    throw new Error("existing batch runtime mismatch");
  }
  if (manifest.agent_model !== options.agentModel || manifest.judge_model !== options.judgeModel) {
    throw new Error("existing batch model mismatch");
  }
  const core = new CoreClient(runtime.TDAI_CORE_URL, runtime.TDAI_USER_KEY, runtime.TDAI_TEAM_ID, runtime.TDAI_USER_ID);
  for (const entry of selected) {
    const started = Date.now();
    try {
      const result = await runItem(
        entry, options, runtime, core, judgeBaseUrl, judgeApiKey,
        manifest.collection_id, sourceSha256,
      );
      appendJsonl(join(options.outputRoot, "status.jsonl"), { ...result, elapsed_ms: Date.now() - started });
      process.stdout.write(`${JSON.stringify(result)}\n`);
    } catch (error) {
      const failure = {
        question_id: entry.question_id,
        status: "failed",
        error: errorMessage(error),
        elapsed_ms: Date.now() - started,
        at: new Date().toISOString(),
      };
      appendJsonl(join(options.outputRoot, "status.jsonl"), failure);
      process.stderr.write(`${JSON.stringify(failure)}\n`);
      if (!options.continueOnError) throw error;
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
    process.exitCode = 1;
  });
}
