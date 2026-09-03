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
git checkout feat/pi-tdai-harbor-branchout
python -m pip install -e .
```

需要本机已经能运行：

- Harbor
- Pi coding agent
- TDAI Proxy / MemoryCore
- EvoCodeBench 的 Harbor task

并准备好官方 TDAI Pi 插件，例如：

```text
/path/to/TencentDB-Agent-Memory/MemoryCore/pi-plugin/index.ts
```

本仓库不会改这个文件。`--pi-extension` 如果是本机文件，运行时会自动上传到 Harbor agent 环境。

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

`TDAI_TASK_ID` 可选。

注意：如果 Harbor task 跑在 Docker 容器里，`127.0.0.1:8096` 通常指容器自己，不是宿主机。请使用容器实际能访问的 Proxy 地址，例如同一 Docker network 的服务名，或你的环境已经配置好的 `host.docker.internal`。

## 先跑一条 Natural

```bash
pi-branch-out natural \
  --task /data/EvoCodeBench/<task> \
  --jobs-dir ./natural_runs/<task> \
  --model tdai/<model> \
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
