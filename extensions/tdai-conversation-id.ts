/**
 * Pi 0.73.1 has no `before_provider_headers` event. The official TDAI plugin
 * listens for that event to set `x-conversation-id`, so the proxy never binds
 * the LLM session to `pi-${sessionId}` and memory-bridge 401s.
 *
 * Re-register the tdai provider with a static conversation-id header once the
 * session id is known. registerProvider is live after load; headers are read
 * per request by getApiKeyAndHeaders, which runs after session_start /
 * before_agent_start. The full provider config is sent on every re-register
 * because a headers-only call would wipe the stored apiKey.
 */
type ExtensionAPI = {
  on(event: string, callback: (event: any, ctx: any) => unknown): void;
  registerProvider(name: string, config: Record<string, unknown>): void;
};

function identityHeaders(conversationId?: string): Record<string, string> | null {
  const teamId = process.env.TDAI_TEAM_ID ?? "";
  const agentId = process.env.TDAI_AGENT_ID ?? "";
  const userKey = process.env.TDAI_USER_KEY ?? "";
  if (!teamId || !agentId || !userKey) return null;
  const headers: Record<string, string> = {
    "x-team-id": teamId,
    "x-agent-id": agentId,
  };
  const taskId = process.env.TDAI_TASK_ID ?? "";
  if (taskId) headers["x-task-id"] = taskId;
  if (conversationId) {
    headers["x-conversation-id"] = conversationId;
    // MemoryProxy's Responses/Codex handler currently triggers sessionInit
    // from `session-id`, while memory-bridge resolves `x-conversation-id`.
    // Send the same stable Pi id under both protocol-specific names.
    headers["session-id"] = conversationId;
  }
  return headers;
}

function registerTdai(pi: ExtensionAPI, conversationId?: string): void {
  const headers = identityHeaders(conversationId);
  if (!headers) return;
  const proxyBase = (process.env.TDAI_PROXY_URL ?? "http://127.0.0.1:8096").replace(/\/$/, "");
  const spaceId = process.env.TDAI_SPACE_ID ?? "default";
  const agentSource = process.env.TDAI_AGENT_SOURCE ?? "pi";
  const wireApi = process.env.TDAI_WIRE_API === "responses" ? "responses" : "chat-completions";
  const model = process.env.TDAI_MODEL ?? "glm-5.2-vision";
  const userKey = process.env.TDAI_USER_KEY ?? "";
  pi.registerProvider("tdai", {
    name: "TDAI Memory Proxy",
    baseUrl: wireApi === "responses"
      ? `${proxyBase}/codex/${spaceId}/v1`
      : `${proxyBase}/${agentSource}/${spaceId}/v1`,
    api: wireApi === "responses" ? "openai-responses" : "openai-completions",
    apiKey: userKey,
    headers,
    models: [
      {
        id: model,
        name: model,
        input: ["text", "image"],
        reasoning: true,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 524288,
        maxTokens: 16384,
        thinkingLevelMap: {
          off: "none",
          minimal: "minimal",
          low: "low",
          medium: "medium",
          high: "high",
          xhigh: "xhigh",
          max: "max",
        },
      },
    ],
  });
}

export default function tdaiConversationId(pi: ExtensionAPI): void {
  const bind = (_event: any, ctx: any) => {
    const sid = ctx?.sessionManager?.getSessionId?.();
    if (typeof sid !== "string" || !sid) return;
    registerTdai(pi, `pi-${sid}`);
  };
  pi.on("session_start", bind);
  pi.on("before_agent_start", bind);
  pi.on("before_provider_request", (event: any, ctx: any) => {
    if (ctx?.model?.provider !== "tdai" || process.env.TDAI_WIRE_API !== "responses") return;
    const payload = event?.payload;
    if (!payload || !Array.isArray(payload.input)) return;
    payload.input = payload.input.map((item: any) => {
      if (!item || typeof item !== "object" || typeof item.role !== "string" || item.type) return item;
      return { ...item, type: "message" };
    });
  });
}
