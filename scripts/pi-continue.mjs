import { join } from "node:path";
import { pathToFileURL } from "node:url";

const packageRoot = process.env.PI_CODING_AGENT_ROOT;
const sessionPath = process.env.PI_BRANCH_OUT_SESSION;
const modelRef = process.env.PI_BRANCH_OUT_MODEL;
const leafId = process.env.PI_BRANCH_OUT_LEAF_ID;
const thinkingLevel = process.env.PI_BRANCH_OUT_THINKING ?? "off";
const extensionPaths = JSON.parse(process.env.PI_BRANCH_OUT_EXTENSIONS ?? "[]");

if (!packageRoot || !sessionPath || !modelRef || !leafId) {
  throw new Error("PI_CODING_AGENT_ROOT, PI_BRANCH_OUT_SESSION, PI_BRANCH_OUT_MODEL, and PI_BRANCH_OUT_LEAF_ID are required");
}

const pi = await import(pathToFileURL(join(packageRoot, "dist/index.js")).href);
const cwd = process.cwd();
const services = await pi.createAgentSessionServices({
  cwd,
  resourceLoaderOptions: { additionalExtensionPaths: extensionPaths },
});
const errors = services.diagnostics.filter((item) => item.type === "error");
if (errors.length > 0) {
  throw new Error(errors.map((item) => item.message).join("; "));
}

const slash = modelRef.indexOf("/");
if (slash <= 0 || slash === modelRef.length - 1) throw new Error(`invalid model reference: ${modelRef}`);
const provider = modelRef.slice(0, slash);
const modelId = modelRef.slice(slash + 1);
const model = services.modelRegistry.find(provider, modelId);
if (!model) throw new Error(`model not registered: ${modelRef}`);

const sessionManager = pi.SessionManager.open(sessionPath);
if (!sessionManager.getEntry(leafId)) throw new Error(`checkpoint leaf is missing from session: ${leafId}`);
sessionManager.branch(leafId);
const restored = sessionManager.buildSessionContext();
const last = restored.messages.at(-1);
if (!last || (last.role !== "user" && last.role !== "toolResult")) {
  throw new Error(`checkpoint cannot continue from message role: ${last?.role ?? "none"}`);
}

const { session, modelFallbackMessage } = await pi.createAgentSessionFromServices({
  services,
  sessionManager,
  model,
  thinkingLevel,
});
if (modelFallbackMessage) process.stderr.write(`${modelFallbackMessage}\n`);
session.subscribe((event) => process.stdout.write(`${JSON.stringify(event)}\n`));

try {
  await session.agent.continue();
} finally {
  session.dispose();
}
