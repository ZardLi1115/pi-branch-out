/**
 * Proxy-side deterministic token estimator.
 *
 * MemoryCore already has a precise o200k_base estimator, but MemoryProxy does
 * not depend on js-tiktoken today. Pulling that dependency into the proxy only
 * for branch-out would enlarge the production surface. This estimator follows
 * MemoryCore's CJK-aware fallback and is intentionally isolated behind two
 * functions so it can be replaced by the exact tokenizer later.
 */
export function estimateTextTokens(text: string): number {
  if (!text) return 0;
  let cjk = 0;
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if (
      (cp >= 0x4e00 && cp <= 0x9fff) ||
      (cp >= 0x3400 && cp <= 0x4dbf) ||
      (cp >= 0xf900 && cp <= 0xfaff)
    ) {
      cjk += 1;
    }
  }
  const rest = Math.max(0, text.length - cjk);
  return Math.max(1, Math.ceil(cjk / 1.7 + rest / 4));
}

function visibleContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content == null ? "" : JSON.stringify(content);
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (!part || typeof part !== "object") return "";
      const p = part as Record<string, unknown>;
      if (p.type === "text" && typeof p.text === "string") return p.text;
      if (p.type === "tool_use" || p.type === "toolCall") {
        return `${String(p.name ?? p.toolName ?? "")} ${JSON.stringify(p.arguments ?? p.input ?? "")}`;
      }
      if (p.type === "tool_result") return JSON.stringify(p.content ?? "");
      return JSON.stringify(p);
    })
    .join("\n");
}

export function estimateContextTokens(messages: Array<Record<string, unknown>>): number {
  let total = 0;
  for (const message of messages) {
    const role = String(message.role ?? "");
    const content = visibleContent(message.content);
    total += estimateTextTokens(`${role}\n${content}`) + 4;
  }
  return total;
}
