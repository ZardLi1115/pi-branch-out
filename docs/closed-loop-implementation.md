# TDAI 记忆预算闭环：实现与运行合同

本文件对应 RoadmapBench 采集路径。TDAI 后端使用
`TencentCloud/TencentDB-Agent-Memory`，本仓库不修改其源码。

## 已实现边界

- 动作表固定为 `0/0.2/0.4/0.6/0.8/1.0`，版本为
  `budget-ratios-v1`。
- allocator 版本为 `complete-render-v2`：完整渲染重计数、完整片段删除、
  L1 优先、L0 原检索顺序、无显式关系的 L0 独立展示。
- model-call checkpoint 保存 workspace git binary delta、未跟踪文件包、Pi leaf、
  append-only session 引用、冻结候选、候选原文/来源、动作预演表和 SHA-256。
- 所有 model call 追加轻量账本；未抽中节点的候选字段为 `null/unobserved`，不会
  冒充 0。默认限额为每任务 2 个断点、间隔 10 次调用、10% 抽样、最多 8 次探测。
  随机数由 task、sampling batch、call index 的 SHA-256 确定，可复查复现。
- `collector-summary.json` 记录探测次数、保存次数、累计探测/保存耗时及保存字节数。
- branch 当前调用优先于 policy；当前调用结束后，同一冻结版本 policy 继续逐调用决策。
- 默认标签与 CQL transition 物理分离；state prefix 按内容指纹只存一次。
- input、output、cache read、cache write 按 Pi 的互斥 usage schema 分开计费，
  Memory Token 已属于 input，不再单独收费。
- `done` 与 `truncated` 分离；缺终局分、缺初始/分叉分、缺账单的轨迹 fail-closed。

## 后端隔离硬约束

腾讯当前 `memory-bridge` 只开放 search/read；Pi 插件没有 memory namespace、禁写、
L0/L1 snapshot/restore 参数。L0/L1 查询可按 task/session 收敛，但 L2/L3 仍共享
team+agent。因此冻结候选只构成局部注入实验，不能证明完整长程反事实。

正式 Branch 必须使用 `--tdai-isolation-mode isolated-instance`，并用
`--backend-proxy-url` 实际覆盖 Harbor 子进程看到的 `TDAI_PROXY_URL`。多动作 grid
还要求 URL 与 instance id 同时含 `{run_id}`，保证每个有效内容动作使用不同实例。
共享模式只能配合 `--allow-shared-backend-long-branch` 做非训练接线检查。

## Responses / Codex 兼容要求

当上游使用 Codex custom provider 的 Responses API 时，设置
`TDAI_WIRE_API=responses`。外围兼容 extension 会执行三项协议对齐，而不修改
TencentDB-Agent-Memory：

- 模型请求同时携带相同值的 `x-conversation-id` 与 `session-id`，前者供
  memory-bridge 使用，后者触发 Codex handler 的 sessionInit。
- bridge 在热进程中先尝试 `codex:<conversation-id>`；只有收到明确的
  `session not initialized` 才回退裸 ID，供重启后的持久化 BindingRepo 查询。
- Pi 省略 Responses message item 的可选 `type` 时，在发送前为已有 role 的 item
  补 `type: message`，否则当前 TDAI Codex recorder 无法识别 user input，L0 会
  被静默跳过。

本地 Proxy 必须在启动阶段启用持久化 ProxyStorage，例如 SQLite：

```yaml
storage:
  enabled: true
  backend: sqlite
  sqlite:
    dbPath: /data/tdai-memory-proxy/storage.db
```

仅设置 fs fallback 目录不够：当前 Proxy 要到首次 injection pipeline 构建时才安装
fallback BindingRepo，可能错过第一轮 sessionInit 的 binding 写入。

2026-09-05 的 RoadmapBench `glz-3.0.0-roadmap` 验收结果：78 次模型调用，
77 个调用级 checkpoint 全部 recall ready；最大候选 86 条/107080 Token；官方
3/3 phases、reward 1.0；缓存率按 `cache_read/(input+cache_read)` 计算为
95.16%。运行产物保存在 git 忽略的 `.local-tdai/`，不作为仓库测试 fixture。

## 推荐顺序

1. 用 `split-roadmapbench` 从官方 `data/tasks_overview.jsonl` 生成稳定任务级 split。
2. `scripts/collect-roadmapbench.py` 先以无模型 Agent 测初始分，再运行 untouched
   Natural，收集 model-call checkpoint、usage 和最终官方分数。
3. `select-checkpoints` 每任务选 1～2 个存在有效内容差异且覆盖不足的节点。
4. 对选中节点先运行 `score-checkpoint`；再从相同快照运行预算 0 恢复对照，使用
   `verify-recovery` 检查官方分数和零注入。
5. 在独立 TDAI 实例上运行替代动作。`branch-grid` 根据 checkpoint 的动作预演
   SHA-256，在 API 调用前跳过等价档位。
6. `export-training` 生成：
   - `state-prefixes.jsonl`
   - `default-labels.jsonl`
   - `transitions.jsonl`
   - `equivalent-action-aliases.jsonl`
   - `dataset-manifest.json`
7. `train-policy` 训练一层 ReLU 小 Q 网络。默认标签不全为 0 时才分类预热；全为
   0 时从动作 0 初始化，随后 CQL 只读取真实动作 transition。
8. 用 `--policy-file` 与 `--policy-version` 采新轨迹。在线 extension 使用与训练端
   相同的 `visible-state-hash-v3-history` 特征，只读取决策时可见字段。
9. `summarize-evaluation` 按完整任务配对汇总平均完成分、满分成功率、API 成本和
   policy p95 时延。首个 variant 是 baseline，质量容忍范围必须在运行前传入。

## 奖励与数据资格

阶段分只接受同时带 `official_verifier=true` 与 `isolated_copy=true` 的
`checkpoint-score.json`。未评分节点质量增量为 0，下一次真实评分吸收累计差值；
最后一次评分使用终局分，因此质量项望远镜求和为：

```text
终局完成分 - 初始完成分
```

单步奖励再减去该次调用的归一化实际账单成本。人为停止的轨迹标为
`truncated=true`，不会伪装成环境终止。

## 仍由外围基础设施提供

- 每条 Branch 的独立 TDAI/MemoryCore/存储实例及其启动、销毁和健康检查。
- API 密钥、模型价格、总调用/费用/时间额度。
- 正式批量任务的 `wake-run` 调度。

这些信息未就绪时，只能运行单元测试、离线导出测试和共享后端的非训练局部检查，
不能宣称完成长程反事实验收。
