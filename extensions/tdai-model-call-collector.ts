import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { basename, join } from "node:path";

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

async function bridgeSearch(conversationId: string, kind: string, query: string, limit: number): Promise<Record<string, unknown>> {
  const proxy = (process.env.TDAI_PROXY_URL ?? "http://127.0.0.1:8096").replace(/\/$/, "");
  const space = process.env.TDAI_SPACE_ID ?? "default";
  const response = await fetch(`${proxy}/memory-bridge/v3/${kind}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-conversation-id": conversationId,
      "x-tdai-service-id": space,
    },
    body: JSON.stringify({ query, limit }),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`memory bridge ${kind} failed: HTTP ${response.status} ${text.slice(-1000)}`);
  const value = JSON.parse(text || "{}") as unknown;
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
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
} {
  const patchName = "workspace.patch";
  const patch = git(cwd, ["diff", "--binary", "--no-ext-diff", baseCommit, "--", "."], "utf8") as string;
  writeFileSync(join(root, patchName), patch, "utf8");

  const untracked = git(cwd, ["ls-files", "--others", "--exclude-standard", "-z"]) as Buffer;
  if (untracked.length === 0) return { patch: patchName, untrackedArchive: null };

  const listPath = join(root, "untracked-files.list");
  const archiveName = "workspace-untracked.tar.gz";
  writeFileSync(listPath, untracked);
  execFileSync("tar", ["--null", "-T", listPath, "-czf", join(root, archiveName)], {
    cwd,
    maxBuffer: 512 * 1024 * 1024,
  });
  return { patch: patchName, untrackedArchive: archiveName };
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

export default function tdaiModelCallCollector(pi: ExtensionAPI): void {
  const outputRoot = process.env.PI_BRANCH_OUT_MODEL_CALL_DIR;
  if (!outputRoot) return;
  mkdirSync(outputRoot, { recursive: true });
  const stateLog = join(outputRoot, "model-call-states.jsonl");
  let callIndex = Number(process.env.PI_BRANCH_OUT_CALL_OFFSET ?? 0);
  let baseCommit = "";

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
      context_tokens: estimateMessagesTokens(messages),
      query,
      candidate_memory_tokens: 0,
      candidate_count: 0,
      l1_count: 0,
      l0_count: 0,
      default_actual_memory_tokens: 0,
      default_mapped_action: 0,
      actual_injected_content_sha256: createHash("sha256").update("").digest("hex"),
      checkpoint_status: callIndex === 1 ? "cold-start" : "pending",
    };

    if (callIndex === 1) {
      appendJsonl(stateLog, state);
      return;
    }

    const checkpointRoot = join(outputRoot, `call-${String(callIndex).padStart(3, "0")}`);
    mkdirSync(checkpointRoot, { recursive: true });
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
      writeFileSync(join(checkpointRoot, "recall-snapshot-error.txt"), `${String(error)}\n`, "utf8");
    }

    const l1 = rows(atomic, "items");
    const l0 = rows(conversation, "messages");
    state.candidate_memory_tokens = candidateTokens(atomic, conversation);
    state.candidate_count = l1.length + l0.length;
    state.l1_count = l1.length;
    state.l0_count = l0.length;
    state.recall_snapshot_status = recallStatus;

    const observation = branchObservation();
    if (observation) {
      state.actual_memory_tokens = observation.injected_tokens ?? 0;
      state.mapped_action = observation.applied_ratio ?? observation.requested_ratio;
      state.actual_injected_content_sha256 = observation.injected_content_sha256 ?? state.actual_injected_content_sha256;
    }

    const snapshot = {
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
    const snapshotName = "recall-snapshot.json";
    if (recallStatus === "ready") {
      writeFileSync(join(checkpointRoot, snapshotName), `${JSON.stringify(snapshot, null, 2)}\n`, "utf8");
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
      };
      writeFileSync(join(checkpointRoot, "checkpoint.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
      state.checkpoint_status = "ready";
      state.checkpoint = `call-${String(callIndex).padStart(3, "0")}`;
    } catch (error) {
      state.checkpoint_status = "error";
      state.checkpoint_error = String(error);
      writeFileSync(join(checkpointRoot, "checkpoint-error.txt"), `${String(error)}\n`, "utf8");
    }
    appendJsonl(stateLog, state);
  });
}
