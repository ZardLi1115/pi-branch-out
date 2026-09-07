# LongMemEval Budget Policy 采集合同

## 实验语义

本路径固定 TencentDB-Agent-Memory `v2.0.0-beta.1`。每个 LongMemEval 问题
使用独立 agent namespace；每个 `haystack_session_id` 保持为独立 MemoryCore
session，并只写该 session 的 user/assistant 消息。写入完成并等待 pipeline 后，
采集器只查询一次：

oracle 文件不保证 session 顺序；采集器按 `haystack_dates` 升序稳定排序后写入，
同时保留原 `haystack_session_id` 和源数组下标，避免 temporal / knowledge-update
任务被文件排列顺序扰动。

- L1：`POST /v3/atomic/search`，原问题作为 query，`limit=5`；
- L3：`POST /v3/core/read`；
- L2：`POST /v3/scenario/ls`。

这三份结果组成冻结 state。Persona（L3）和 Scene Navigation（L2）对所有动作
保持相同；离散预算只控制 L1。动作 1.0 是 beta.1
OpenClaw 的完整 top-5 默认召回，动作 0 不注入 L1。采集器复用仓库的完整片段
allocator，并用 beta.1 OpenClaw 原生 `<relevant-memories>` 包装重新计数。

不同 nominal action 渲染出相同内容时只产生一次模型答案和一次 judge 结果，其他
动作写入 `action_aliases`。因此同一题的训练记录具有相同 `state_id`，但不会把
等价动作伪装成多次独立 API 观测。

## 数据准备与实例启动

所有数据统一放在 `.local-tdai/longmemeval-collection/`，该目录被 Git 忽略：

```powershell
pwsh -File scripts/prepare-longmemeval.ps1

pwsh -File scripts/start-local-tdai.ps1 `
  -InstanceName longmemeval-chat -CorePort 8423 -ProxyPort 8099 `
  -PromptMode chat `
  -L1IdleTimeoutSeconds 2 -L2DelayAfterL1Seconds 2 `
  -L2MinIntervalSeconds 5 -L2MaxIntervalSeconds 30
```

缩短的是离线灌库后的 pipeline 调度等待，不改变 L1 query、top-k、allocator 或
回答动作。正式启动和冒烟必须由 `wake-run` 包装。

`PromptMode=chat` 是 LongMemEval 的必要语义。RoadmapBench 实例默认使用
`code`；该模式的 L1 提取规则会主动排除个人生活和个人偏好，不能用于本评测。

先使用 oracle 做接线验证；它的 10,960 条消息全部满足 beta.1 的单条 8192 字符
限制：

```powershell
pwsh -File scripts/run-longmemeval.ps1 `
  -Data .local-tdai/longmemeval-collection/source/longmemeval_oracle.json `
  -OutputRoot .local-tdai/longmemeval-collection/runs/oracle-v1 `
  -Runtime .local-tdai/longmemeval-chat/runtime.json `
  -Limit 1
```

回答模型和 judge 默认都使用本机 Codex `custom` provider 的 `gpt-5.6-luna`。
回答模板采用 LongMemEval 官方 retrieved-facts 语义，不额外要求保守拒答；版本会写入
manifest，避免不同 prompt 的数据混用。
judge prompt 与 LongMemEval 官方 `evaluate_qa.py` 一致，但这不是官方 GPT-4o judge，
manifest 会标记为 `official-prompt-custom-judge`。若以后提供官方 judge endpoint，
需单独运行标准脚本或将 judge 模型改成 `gpt-4o-2024-08-06`。

## 产物

```text
runs/<batch>/
├── dataset-manifest.json
├── status.jsonl
├── hypotheses/action-*.jsonl
└── items/<question-id>/
    ├── source-reference.json
    ├── ingest.jsonl
    ├── pipeline-status.json
    ├── candidate-snapshot.json
    ├── state.json
    ├── action-plan.json
    ├── samples.jsonl
    ├── actions/<effective-action-id>/{request,response,judge}.json
    └── complete.json
```

源历史不在每个 item 下重复复制；`source-reference.json` 保存 source/history hash、
evidence session IDs 和任何截断记录。候选原文、score、Persona、场景路径及原始 API
响应保存在 `candidate-snapshot.json`。

在线 state 同时保存 top-5 `l1_contents`、`persona_text` 和 `scene_paths`；这些都是
决策时可见信息，后续可由固定编码器构建 MLP 特征，标准答案不会进入 state。

可续跑批量入口会逐题记录失败并继续，已完成题不会重复写入：

```powershell
pwsh -File scripts/run-longmemeval-batch.ps1 -BatchName oracle-v1 -Limit 20
```

批量入口使用固定 `oracle-stratified-v1.json`：六种 question type 与 abstention
先在各层内按哈希稳定排序，再轮转组成 500 题顺序。因此把同一 batch 从 `Limit 20`
扩大到 `Limit 50` 不会改变前 20 题，也不会误采文件开头单一题型。

训练导出：

```powershell
python scripts/export-longmemeval-training.py `
  --collection-root .local-tdai/longmemeval-collection/runs/oracle-v1 `
  --output-dir .local-tdai/longmemeval-collection/training/oracle-v1
```

导出器按 question ID 稳定切分 80/10/10，生成现有 CQL trainer 可读的
`state-prefixes.jsonl`、`default-labels.jsonl`、`transitions.jsonl` 和
`equivalent-action-aliases.jsonl`。默认 `reward` 是 LongMemEval 二元质量分；可显式
传 `--cost-coefficient` 加入成本惩罚。`--cost-measure answer-billable-token-proxy`
使用回答调用的输入（扣缓存读）加输出 Token；`--cost-measure injected-l1-tokens`
只惩罚预算动作实际新增的 L1 Token。judge 用量始终只算研究开销。

要按固定 selection 前缀重建 n200 并扫描 L1 Token 成本系数，可运行：

```powershell
pwsh -File scripts/run-longmemeval-cost-sweep.ps1 `
  -Limit 200 `
  -CostCoefficients 0,0.1,0.3,1.0 `
  -CostNormalizerTokens 100 `
  -Seeds 7,17,29
```

该脚本只读取已完成的采集结果，不重新调用 answer/judge API。各 cost coefficient 使用
独立 training/policy 目录，不覆盖完整 n500 数据。selection 文件与 limit 会写入 manifest，
保证子集成员可以复核。

`default-labels.jsonl` 中的 1.0 只表示 beta.1 OpenClaw 默认会注入完整 top-5，
是行为模仿标签，不表示该动作是该题的最优预算。实际 Q 值只从已执行并评分的
`transitions.jsonl` 学习。

等价 action 会在训练导出中展开为全部 nominal ratio，但共享同一真实 observation，
并设置 `sample_weight=1/alias_count`。这样 CQL 不会把已证实等价的 ratio 当成未观测
动作，同时同一 API 结果在 loss 中的总权重仍为 1，不会虚增证据量。特征版本
`visible-state-hash-v4-memory-text` 会编码 query、top-5 L1 原文、Persona 和场景路径。

小模型训练后必须用同一批次的配对 dev/test action 做离线冻结评估：

```powershell
pi-branch-out train-policy `
  --dataset-dir .local-tdai/longmemeval-collection/training/oracle-v1 `
  --output-dir .local-tdai/longmemeval-collection/policies/oracle-v1-seed7

python scripts/evaluate-longmemeval-policy.py `
  --dataset-dir .local-tdai/longmemeval-collection/training/oracle-v1 `
  --policy-dir .local-tdai/longmemeval-collection/policies/oracle-v1-seed7 `
  --output .local-tdai/longmemeval-collection/policies/oracle-v1-seed7/evaluation.json
```

50 题阶段的 dev/test 仍太小，结果只用于训练接线验收，不用于替换默认策略。
评估同时报告 `all` 与 `informative`：后者只包含至少两个 action 获得不同质量分的
state，防止全档同分题让任意策略看起来同样优秀。固定预算除质量外还报告平均
L1 注入 Token，便于按预先确定的质量容忍范围比较成本，而不是只按准确率选策略。

beta.1 的 `conversation/add` 没有幂等键，因此写入请求绝不自动重试。若响应结果
不确定，item 会写 `ingest-uncertain.json` 并隔离失败；必须换新 batch/namespace，
不能在原 namespace 上猜测重跑。

每个未完成 item 还会保存 `namespace.json`。基础设施中断后，部分写入的旧 item
必须整体移入 outage audit，再由新 namespace 重新灌入；不能在旧 namespace 上重复
追加。批量入口会恢复已有但停止的容器并等待健康，运行中快速连接失败会触发熔断，
避免把后续题目全部标成独立数据失败。

若 Docker Desktop 前端仍在但 daemon pipe 无响应，应通过 wake-run 执行：

```powershell
pwsh -File scripts/restart-docker-desktop.ps1
```

批量脚本对每次 `docker info/update/start` 都有独立超时，不会再永久卡在 Docker CLI。

## 超长 L0 边界

cleaned-S 有 152/500 个实例含至少一条超过 8192 字符的干扰消息（没有 evidence
session 被命中），beta.1 无法原样写入。默认 `--overlong-policy error`，绝不静默
改变 benchmark。可选 `skip-instance` 或显式 `truncate`；后者会在
`source-reference.json` 记录原长度与存储长度。分块会改变 user-round 计数和
pipeline 触发语义，因此本实现不提供隐式分块模式。

cleaned-M 约 500 sessions / 题，成本远高于 oracle 和 cleaned-S。在 oracle 冒烟、
候选非空率、评分稳定性及动作差异率验收前，不进入 cleaned-M 批采。
