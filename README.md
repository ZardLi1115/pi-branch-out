# pi-branch-out

用于 **Pi + TDAI Memory + Harbor / EvoCodeBench** 的结构化 Branch-out 数据采集。

这套代码有一个硬边界：**不修改 TencentDB-Agent-Memory、MemoryCore 或 MemoryProxy 源码。** TDAI 只作为现成服务使用。我们通过官方 Pi 插件走 TDAI Proxy，并通过现有只读 `memory-bridge` 冻结 L1/L0 候选；Budget Controller、上下文分配和 Branch-out 全部发生在本仓库与 Pi extension 侧。

## 现在能做什么

Natural 运行保持当前 Pi + TDAI 的原始行为，不额外自动注入动态 L1/L0。Harbor 从第 2 个 step 开始，在 Pi 处理当前 instruction 之前保存：

```text
branch-checkpoints/step-002/
├── checkpoint.json
├── workspace.tar.gz
├── checkpoint-session.jsonl
├── pi-session-full/
└── recall-snapshot.json
```

`recall-snapshot.json` 是当时通过 TDAI 现有只读 `atomic/search` + `conversation/search` 得到的候选快照。之后 Branch 不再实时查询 TDAI，而是始终使用这份冻结候选，因此不会看到自然轨迹后续产生的“未来记忆”。

Branch 的唯一动作是：

```text
budget_ratio ∈ [0, 1]
```

推荐第一批使用：

```text
0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0
```

实际预算不是固定 Token，而是：

```text
feasible_budget = min(
    当前上下文剩余空间,
    冻结候选池总 Token,
    可选 hard cap
)

budget = budget_ratio × feasible_budget
```

具体 Memory 如何装入是确定性的：先按 TDAI 召回顺序放完整 L1；剩余预算再对已选 L1 逐层、轮流补 L0，直到预算或 L0 用完。没有单独的“展开粒度”动作。

## 安装

```bash
git clone https://github.com/ZardLi1115/pi-branch-out.git
cd pi-branch-out
python -m pip install -e .
```

需要本机已经能运行：

- Harbor **0.22+**（本仓库 agent 使用 `SUPPORTS_RESUME = True`。旧 Harbor 的 `harbor.agents.capabilities.AgentCapabilities` 已不存在。）
- Pi coding agent **0.73.1**。批量任务建议先运行
  `scripts/build-pi-runtime.sh` 生成 Linux 离线 runtime，再用
  `--pi-runtime-archive` 注入，避免每个容器通过 NVM/npm 重装。
- TDAI Proxy / MemoryCore
- 一个 Harbor 多 step 任务。仓库里有一条可跑的 EvoCodeBench 示例：
  `examples/harbor_tasks/evocodebench-run-cmd`（`chat.utils.run_cmd`）。原始 `data.jsonl` 不是 Harbor task。

并准备好官方 TDAI Pi 插件，例如：

```text
/path/to/TencentDB-Agent-Memory/MemoryCore/pi-plugin/index.ts
```

本仓库不会改这个文件。`--pi-extension` 如果是本机文件，运行时会自动上传到 Harbor agent 环境。`extensions/tdai-conversation-id.ts` 会由 agent 自动上传，不必再手动加一次。

## TDAI 环境变量

至少准备：

```bash
export TDAI_PROXY_URL="http://<Harbor容器可访问的proxy地址>:8096"
export TDAI_SPACE_ID="default"
export TDAI_TEAM_ID="<team>"
export TDAI_AGENT_ID="<agent>"
export TDAI_USER_KEY="<user-key>"
export TDAI_MODEL="<model>"
```

`TDAI_TASK_ID` 对 memory 来说可选；但若 Proxy 开了 `sessionInit`，没有 task 时第一次 LLM 调用会被劫持成 `ask_followup_question` 表单（`content=None`）。采集时建议创建一个真实 task 并设置 `TDAI_TASK_ID`。

注意：如果 Harbor task 跑在 Docker 容器里，`127.0.0.1:8096` 通常指容器自己，不是宿主机。请使用容器实际能访问的 Proxy 地址，例如同一 Docker network 的服务名，或你的环境已经配置好的 `host.docker.internal`。Agent 会把 `TDAI_PROXY_URL` / `OPENAI_BASE_URL` 里的 loopback 改写成 `host.docker.internal`。

## 先跑一条 Natural

```bash
pi-branch-out natural \
  --task ./examples/harbor_tasks/evocodebench-run-cmd \
  --jobs-dir ./natural_runs/evocodebench-run-cmd \
  --model tdai/<model> \
  --pi-thinking medium \
  --pi-extension /path/to/TencentDB-Agent-Memory/MemoryCore/pi-plugin/index.ts
```

第 1 个 Harbor step 用来建立真实 Pi/TDAI session，因此只作为 baseline，不做严格 Branch。第 2 个 step 起，checkpoint 中应出现：

```text
recall_snapshot_status = "ready"
```

如果是 `bridge-error`，先检查 `TDAI_PROXY_URL` 是否能从 Harbor agent 环境访问，以及 Natural 第 1 step 是否已经成功通过 TDAI Proxy。

## 从一个 checkpoint 跑单个 Branch

```bash
pi-branch-out branch \
  --task /data/EvoCodeBench/<task> \
  --checkpoint ./natural_runs/.../branch-checkpoints/step-003 \
  --budget-ratio 0.4 \
  --output-root ./branch_runs \
  --model tdai/<model> \
  --pi-extension /path/to/TencentDB-Agent-Memory/MemoryCore/pi-plugin/index.ts
```

## RoadmapBench 单步任务

RoadmapBench 的一整份 roadmap 是一个 Harbor step。Natural 采集必须改用 Pi
内部 model-call 边界：

```bash
pi-branch-out natural \
  --task /path/to/RoadmapBench \
  --jobs-dir ./natural_runs/roadmapbench \
  --model tdai/gpt-5.6-luna \
  --pi-thinking medium \
  --pi-runtime-archive ./runtime/pi-runtime-linux-amd64.tar.gz \
  --checkpoint-boundary model-call \
  --pi-extension /path/to/TencentDB-Agent-Memory/MemoryCore/pi-plugin/index.ts
```

该入口默认向 Harbor 传
`--environment-build-timeout-multiplier 2.0`；RoadmapBench 的默认 600 秒构建
上限因此提升为 1200 秒。正式批采前可先用 Harbor 的 `--install-only --agent
oracle` 跑一遍数据集，将 115 个官方任务镜像拉取/构建进本机缓存。

Harbor 会把数据集根目录展开为全部任务。每个 trial 的
`agent/model-call-checkpoints/model-call-states.jsonl` 记录所有模型调用；从
`call-002/` 起保存可恢复的 Pi leaf id、冻结 recall snapshot，以及相对任务初始
commit 的 workspace binary diff / 未跟踪文件包。所有点共享 Pi 最终的 append-only
session 文件，避免为每个调用复制一遍不断增长的 transcript。

从这些 checkpoint 启动 `branch` 或 `branch-grid` 时，采集器通过 Pi 官方 SDK 的
`Agent.continue()` 恢复 tool loop，不会再次添加 roadmap 用户消息。没有
`[[steps]]` 的原始 RoadmapBench task 会完整 clone，并继续使用它自己的 verifier。

`default_actual_memory_tokens`、`default_mapped_action` 和
`actual_injected_content_sha256` 特指方向 F 的动态 L1/L0 注入。当前腾讯默认链路不
自动注入 L1/L0，所以 Natural 分别为 0、0 和空内容 SHA-256；L2/L3 与工具提示仍由
TDAI 正常工作，但客户端不会把不可观测的服务端静态注入冒充为精确数据。

Branch 会：

```text
恢复 workspace
→ fork 当时的 Pi 原生 session
→ 读取冻结 recall-snapshot.json
→ 只在当前 step 强制 budget_ratio=0.4
→ Pi context hook 注入选中的 L1/L0
→ 写 budget-observation.json
→ 当前 step 结束后解除干预
→ 后续 Harbor step 恢复原始 Pi+TDAI 行为
→ 一直跑到任务结束
```

## 一次跑多个 Budget

```bash
pi-branch-out branch-grid \
  --task /data/EvoCodeBench/<task> \
  --checkpoint ./natural_runs/.../branch-checkpoints/step-003 \
  --ratios 0,0.2,0.4,0.6,0.8,1.0 \
  --output-root ./branch_runs \
  --model tdai/<model> \
  --pi-extension /path/to/TencentDB-Agent-Memory/MemoryCore/pi-plugin/index.ts
```

每个 ratio 使用独立 Harbor run 目录。首轮建议串行执行，确认 namespace / session 不互相污染后再考虑并发。

## 采集结果在哪里

Natural 数据主要在：

```text
natural_runs/<task>/...
├── pi-step-*.stdout.jsonl
├── pi-step-*.stderr.txt
└── branch-checkpoints/step-*/
```

每个 Branch 目录主要包含：

```text
branch_request.json
branch_result.json
jobs/
```

Harbor Agent 日志还会保存：

```text
budget-observation-step-*.json
```

其中记录 requested/applied ratio、冻结候选池大小、feasible budget、实际 budget、注入 Token、L1/L0 id。正式数据默认 fail-closed：如果外部 Pi adapter 没真正执行指定 Budget，就不会把 branch 当作有效结果。

## 冷启动怎么做

当前 Pi + TDAI 本身没有我们新增的动态 L1/L0 Budget Policy，所以 Natural 的真实 F 动作记为 `0`。第一批数据建议先跑一批 untouched Natural，再从 step 2+ 的 checkpoint 对邻近几个 Budget 做 Branch-out。这样第一版 CQL 训练前就已经有“同一个 State、不同 Budget”的真实长期结果，不需要先训练一个 MLP，也不需要随机破坏 Natural 质量。

等第一版 MLP + CQL 出来以后，只需要替换 Budget ratio 的来源：

```text
Branch override > MLP policy > baseline
```

确定性的 Recall Snapshot、Budget Controller 和 L1/L0 allocator 不需要改。

## 测试

```bash
pytest -q
npm install --no-save tsx@4
npx tsx --test tdai/tests/adaptive-memory.test.ts
```

GitHub CI 会同时跑 Python 和 TypeScript 核心测试。

## 本机踩坑（不要提交密钥）

下面这些是 Harbor + Pi 0.73.1 + 当前 TDAI 镜像上踩过的坑。密钥、本机 Codex URL、`.tdai.env`、`natural_runs/`、`branch_runs/` **都不要进 git**。

### 两条 provider 路径不要混

- `--model cpa/<id>`：Pi 直连 Codex / 兼容 OpenAI Responses 的上游。需要 `CUSTOM_API_KEY` + `OPENAI_BASE_URL`（以及可选 `CPA_MODEL`）。**没有 memory。** 只适合接线冒烟。
- `--model tdai/<id>`：走官方 TDAI Pi 插件，才有注入 / L0 写入 / frozen recall。这是采集路径。

Pi 的 `models.json` 里 `apiKey` 必须是**裸环境变量名** `CUSTOM_API_KEY`。写成 `"$CUSTOM_API_KEY"` 时，`resolveConfigValue()` 会把它当字面量送出去，上游返回 401 Invalid API key。

### Pi 0.73.1 没有 `before_provider_headers`

官方插件用这个事件写 `x-conversation-id: pi-${sessionId}`。0.73.1 实际事件是 `session_start` / `before_agent_start` / `before_provider_request`（后者只能改 body）。结果是：

- Proxy 按 API key hash 当 `sessionKey`，`conversationId=null`，`injectedSkipped=true`
- `memory-bridge` 用 `pi-${sid}` 去查，401 `session not initialized`
- Natural step 2+ 的 `recall_snapshot_status` 变成 `bridge-error`，Branch 无法启动

本仓库不改 TDAI。`extensions/tdai-conversation-id.ts` 会在 session id 已知后**完整重注册** `tdai` provider（只改 headers 会把已存的 `apiKey` 清掉），带上静态 `x-conversation-id`。Harbor agent 会自动上传这个文件。

修好后 step 2 应看到 `recall_snapshot_status = "ready"`，Proxy 日志应有 `injectedSkipped=false` 和 `write-l0`。

### Frozen recall 的其它坑

- `_current_pi_session_id` 必须调 `python3`。Ubuntu 24.04 没有无版本的 `python`；heredoc 会把 `command not found` 吞成 rc=0，快照变成 `pi-session-missing`。
- Step 1 的 `session-not-initialized` 是设计如此：当时还没有 Pi session，不能做严格 Branch。
- `memory-bridge` 不要用 `curl -f`：4xx 时 body 被丢掉，错误只剩空白。
- 容器里查 session 文件用 in-container glob；host 侧 `download_dir` 只能用来做 checkpoint，不能用来代替 session id。

### TDAI Proxy 镜像 vs 源码

当前 `memory-proxy:latest` 的 `extractSpaceIdFromPath` allowlist **不含** `pi`（本地源码有）。`/pi/<space>/...` 会 401 `missing service_id`。采集时设 `TDAI_AGENT_SOURCE=codebuddy`（`claude-code` / `opencode` / `dsh` / `cursor` 也可以）。Proxy 要用 `PROXY_FULL_STACK=1`（auth + sessionInit + tdai）。Auth 要求 `MEMORY_CORE_GATEWAY_API_KEY` 为空。

### Docker / Harbor

- 容器访问宿主机服务一律 `host.docker.internal`，不要 `127.0.0.1`。
- Docker VM 内存小时，task `memory_mb` 建议 2048。
- 示例 Dockerfile 用了 DaoCloud / Aliyun 镜像；网络能直连 Docker Hub 时改回官方源即可。
- `--print` 的 Pi 若 stdin 开着会一直等输入。agent 里用 `</dev/null`。
- 非交互环境请加载 nvm 后再调 `pi`。

### 不要提交的东西

- 任何 API key / user key / `.env` / `.tdai.env`
- 本机 Codex / cliproxy 地址和端口
- `natural_runs/`、`branch_runs/`、Harbor job 产物
- 对 `TencentDB-Agent-Memory` 的源码修改（本仓库约定不碰）
