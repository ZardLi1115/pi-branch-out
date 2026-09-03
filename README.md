# pi-branch-out

用于 **Pi + TDAI Memory + Harbor 多轮编码任务** 的结构化 Branch-out 采集。

核心目标不是重新拼接一段历史 Prompt，而是从真实运行轨迹中恢复同一个决策点：

```text
自然轨迹
  ↓
Harbor 某轮开始前保存 checkpoint
  ├─ 代码工作区
  ├─ Pi 原生 JSONL session
  └─ TDAI 状态（本地模式可直接快照）
  ↓
从 checkpoint 建立反事实分支
  ↓
只强制当前这一轮 Memory Budget
  ↓
下一轮起恢复正常策略
  ↓
继续运行到任务结束
```

这套结构面向 EvoCodeBench 这类 Harbor multi-step 任务。每个 benchmark step 就对应一次新的用户需求，也是当前方案中一次 Memory Budget 决策的自然边界。

## 设计原则

1. **保留 Pi 原生结构化历史**：checkpoint 沿 Pi session 的 `parentId` 链裁剪，再用 `--fork` 继续，不把历史扁平化成文本。
2. **Coding 环境必须一起恢复**：代码、依赖和生成文件都会影响后续行为，因此 checkpoint 同时保存工作区。
3. **TDAI 状态必须一致**：本地状态目录可直接打包恢复；如果使用外部 Gateway，需要额外提供隔离 namespace 或服务端 snapshot/restore。
4. **只干预一次**：分支的第一轮使用指定 `budget_ratio`，后续 Harbor step 通过 Pi `--continue` 正常继续。
5. **Budget 是比例，不是固定 Token**：`budget_ratio` 作用于 TDAI Runtime 计算出的本轮可行 Memory 预算，因此长任务可以自然超过固定 16K 上限。

建议首版动作集合：

```text
0.0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0
```

`0.0` 表示当前轮关闭动态 L1/L0 自动注入，`1.0` 表示允许使用全部本轮可行动态 Memory 预算。

## 安装

```bash
pip install -e .
```

运行环境还需要：

- Harbor
- Pi coding agent
- 能在 Pi 内工作的 TDAI Memory 适配
- EvoCodeBench task 目录

Harbor 必须启用 `--resume-trajectory`，这样同一任务后续 step 会调用 Agent 的 `resume()`，Pi session 才能连续。

## 1. 采自然轨迹

```bash
pi-branch-out natural \
  --task /data/EvoCodeBench/<task> \
  --jobs-dir ./natural_runs/<task> \
  --model <provider/model> \
  --pi-extension /path/in/container/to/tdai-pi-adapter.ts
```

如果 TDAI 状态保存在 Harbor agent 容器内，还可以指定：

```bash
--tdai-state-dir /path/to/tdai/state
```

Agent 会在每个 Harbor step **处理当前用户需求之前** 保存：

```text
branch-checkpoints/step-002/
├── checkpoint.json
├── workspace.tar.gz
├── checkpoint-session.jsonl
├── pi-session-full/
└── tdai-state.tar.gz       # 仅本地 TDAI 状态模式
```

第一轮开始前还没有 Pi 历史，因此 `checkpoint-session.jsonl` 可以为空；该 checkpoint 仍可作为“新会话 + 不同预算”的分支起点。

## 2. 从某个 checkpoint Branch-out

例如强制当前轮使用 80% 可行 Memory 预算：

```bash
pi-branch-out branch \
  --task /data/EvoCodeBench/<task> \
  --checkpoint ./natural_runs/.../branch-checkpoints/step-005 \
  --budget-ratio 0.8 \
  --output-root ./branch_runs \
  --model <provider/model> \
  --pi-extension /path/in/container/to/tdai-pi-adapter.ts
```

执行过程：

```text
读取 checkpoint
  ↓
生成只包含当前轮及后续轮次的 Harbor task
  ↓
恢复代码工作区
  ↓
恢复 TDAI 状态
  ↓
Pi --fork checkpoint-session.jsonl
  ↓
当前轮注入 budget_ratio=0.8
  ↓
当前轮结束
  ↓
后续轮 Pi --continue，不再强制预算
  ↓
Harbor verifier 正常运行直到 task 结束
```

分支目录会保存 `branch_request.json`、Harbor jobs 和 `branch_result.json`。

## TDAI Budget 接口

本仓库提供 `extensions/tdai-budget-override.ts`，分支第一轮会设置：

```text
TDAI_MEMORY_BUDGET_RATIO_OVERRIDE=<0..1>
TDAI_MEMORY_BUDGET_OVERRIDE_ONE_SHOT=1
```

TDAI 的自适应 Recall 实现需要消费这两个值：

```text
用户输入
  ↓
TDAI 宽召回候选池
  ↓
如果存在 Branch override：使用指定 budget_ratio
否则：使用当前 Policy 输出
  ↓
计算 feasible_memory_budget
  ↓
actual_budget = budget_ratio × feasible_memory_budget
  ↓
确定性分配：完整 L1 优先，剩余预算渐进展开 L0
```

**当前仓库只实现 Branch-out 与 override 传递，不修改 TencentDB-Agent-Memory 本体。** 在 TDAI Runtime 尚未读取 `TDAI_MEMORY_BUDGET_RATIO_OVERRIDE` 前，分支虽然可以正确恢复 Pi/Harbor 状态，但不同预算不会真正改变 Memory 注入内容。

正式采集时还应让 TDAI 每轮记录：请求的 `budget_ratio`、本轮 `feasible_budget`、实际 Memory Token、最终注入的 L1/L0 ID，便于验证分支 Action 是否真正执行。

## 外部 TDAI Gateway

如果 TDAI 数据不在 Harbor 容器，而是共享远端 Gateway，不能仅复制本地目录来恢复 checkpoint。建议二选一：

- 每条 natural / branch rollout 使用独立的 Memory namespace，并从 checkpoint 克隆 namespace；
- 给 Gateway 增加显式 snapshot/restore API。

在完成其中一种方案前，不应并行运行共享同一 TDAI namespace 的同任务分支，否则分支会互相污染。

## 与训练数据的关系

Branch-out 本身不训练模型。它负责得到可信的反事实轨迹：

```text
同一个 State
  ├─ 20% Budget → 后续轨迹 → Reward / Cost
  ├─ 60% Budget → 后续轨迹 → Reward / Cost
  └─ 100% Budget → 后续轨迹 → Reward / Cost
```

之后再把自然轨迹和分支轨迹整理为：

```text
(s_t, a_t, r_t, s_t+1, done)
```

供 CQL + 小型 MLP 训练使用。
