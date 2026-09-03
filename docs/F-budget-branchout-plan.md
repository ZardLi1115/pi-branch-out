# 方向 F：基于 Budget 的自适应记忆注入与 Branch-out

## 这个方向到底在做什么

TDAI 已经负责记忆的存储、检索和长期维护，我们不改它。方向 F 只解决一个很具体的问题：**这一轮到底应该给动态记忆多少上下文空间。**

当前 Pi + TDAI 的主链路里，Pi 通过官方插件把请求发给 TDAI Proxy，TDAI 继续负责原来的 L2/L3、工具能力和 L0 写入。我们额外在 Pi 一侧增加一个很薄的适配层：先通过 TDAI 已有的只读 `atomic/search` 和 `conversation/search` 拿到候选，再决定这一轮最多允许多少 L1/L0 被自动塞进 Pi 的上下文。

因此我们不是训练模型去判断“哪条记忆最有用”，也不是重新做一套检索。TDAI 的召回顺序继续保留，我们只学习资源分配。动作可以写成一个比例：

```text
0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0
```

实际 Token 不是固定值。每轮先算：

```text
feasible budget = min(
  当前上下文还能放多少，
  当前冻结候选池一共多少 Token，
  可选的工程安全上限
)
```

然后：

```text
actual budget = action ratio × feasible budget
```

这样长任务的 100% 可以是 30K、50K，短任务也可能只有几 K，不会被固定的 1K / 4K / 16K 卡死。

给定 Budget 后，具体怎么装内容完全确定。先按 TDAI 原召回顺序放完整 L1，L1 不截断。如果还有空间，就给已经选中的 L1 轮流补第一层 L0，再补第二层、第三层，一直补到预算或候选耗尽。因此“展开粒度”不再是动作，预算越大自然展开越深。

---

## 冷启动怎么做，为什么不直接随机 Budget

第一版没有 MLP，也不应该为了采数据让 Natural 轨迹随机乱跳 Budget。这样很容易把任务质量打坏，而且采出来的数据和真实 TDAI 使用方式离得太远。

所以第一阶段先跑 **原始 Pi + TDAI baseline**。我们的适配层不改 Prompt，只在每个 Harbor step 开始前做旁路记录。当前 Pi + TDAI 本身没有我们新增的动态 L1/L0 自动注入，因此从方向 F 的口径看，真实 baseline action 是 0。这个 0 只代表“不开启 F 新增的自动 L1/L0 注入”，并不代表 TDAI 整体没有记忆，原来的 L2/L3、工具检索和写入仍然照常工作。

为了后续有一个更接近旧式固定 Top-K Recall 的参考，也可以额外计算一个 shadow anchor，例如假设固定放 Top-5 L1，看它占当前候选池的多少比例，再映射到最近的 Budget 档。这个 anchor 只是辅助选择 Branch，不冒充当前 Pi + TDAI 的真实执行动作。

Natural 轨迹最重要的产物不是标签，而是 checkpoint 和冻结的 Recall Snapshot。以 EvoCodeBench + Harbor 为例，一个多轮任务会连续修改同一个代码仓库。我们在第 2 个 Harbor step 开始，每轮真正交给 Pi 之前保存：

```text
Pi 原生 session
代码 workspace
当前 Harbor step
当前 Query
TDAI L1/L0 Recall Snapshot
```

第 1 step 不做严格 Branch，因为新 Pi session 在第一次请求前 TDAI Proxy 还没有完成 session 初始化。第 1 step 正常跑完后，从第 2 step 开始 Memory Bridge 已经可以在真正执行当前请求前读取候选。

Recall Snapshot 必须冻结。原因很简单：如果自然轨迹已经跑到了第 8 轮，再回头分叉第 3 轮，这时候重新查询 TDAI，很可能会看到第 4～8 轮产生的“未来记忆”。所以 Branch 绝不重新 Recall，而是直接使用第 3 轮当时保存的候选快照。

---

## 第一次 MLP 训练前要先做少量 Branch-out

只用 baseline 轨迹训练 MLP 没什么意义，它最多学会模仿 baseline，不知道多给或少给 Memory 会发生什么。因此第一次 CQL 训练之前就要做少量反事实分支。

例如某个 checkpoint 的 baseline 是 0，我们可以从完全相同的 Pi session、代码状态和 Recall Snapshot 出发，分别试：

```text
0.2 → 后续继续跑到任务结束
0.4 → 后续继续跑到任务结束
0.8 → 后续继续跑到任务结束
```

Branch 只强制当前这个 Harbor step 的 Budget。当前 step 结束后，后面的 step 回到正常 baseline 行为。这样测到的是“在这个状态下只改变一次 Budget，会对后面整个任务造成什么长期影响”，而不是把整个任务都改成固定 0.8。

正式采集可以用 `branch-grid` 一次跑多个比例。每个 Branch 都保存实际预算、注入 Token、L1/L0 id，以及 Harbor 的后续轨迹和 verifier 结果。后面把这些数据整理成：

```text
(s, a, r, s_next, done)
```

就可以训练第一版 CQL + 小型 MLP。这里不需要 SFT，也不训练 LLM。MLP 输入状态，输出几个 Budget Action 的 Q 值，CQL 用离线轨迹训练它，并对数据覆盖很少的动作保持保守。

第一版训练完成后，Budget 来源从 baseline 切到 MLP：

```text
Branch override > MLP policy > baseline fallback
```

如果 MLP 在某个状态给出的动作附近数据覆盖太差，可以回退到 baseline 或限制它只在已覆盖的相邻档位里选，先不要让第一版策略直接跳到极端动作。

---

## 最后的自优化闭环

第一版 MLP 上线到实验环境以后，再用它跑新的 EvoCodeBench Natural 轨迹。新的策略会遇到新的 State，也会选择和 baseline 不一样的 Budget。我们继续从这些新轨迹里挑 checkpoint 做 Branch-out，优先测试当前动作的相邻档位，偶尔测试 0 或 1.0 补覆盖。

整个循环就是：

```text
原始 Pi + TDAI baseline
        ↓
Natural 轨迹 + 冻结 Recall Snapshot
        ↓
少量 Branch-out
        ↓
第一批 Offline RL 数据
        ↓
CQL 训练 MLP v1
        ↓
MLP v1 跑新轨迹
        ↓
继续 Branch-out 补反事实数据
        ↓
重新训练 MLP v2
        ↓
持续迭代
```

评估不要只看一个混合 Reward。至少同时看任务质量和成本：Harbor verifier 的用例通过率 / 完整任务成功率，以及每个任务的真实 API 成本、输入 Token。只有在质量不明显下降的前提下成本更低，或者相同成本下长任务成功率更高，才算方向 F 真正有效。

这套设计里 TDAI 始终是现成的 Memory Backend，我们不修改它；可学习部分也始终只有 Budget Policy。检索、候选冻结和实际上下文拼装都是可复现的确定性逻辑，所以后面即使 CQL 最终没有带来明显收益，也可以很容易退回固定 Budget 或原始 Pi + TDAI baseline，不会把整个记忆系统绑死在一个模型上。
