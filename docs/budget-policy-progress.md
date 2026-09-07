# TDAI 记忆预算策略：过程与阶段进展

本项目研究在 TDAI 检索候选已经确定后，如何为每次回答动态选择 L1 注入预算。第一阶段
固定 TencentDB-Agent-Memory `v2.0.0-beta.1` 的 OpenClaw 行为：按问题召回最多五条
L1，同时保持 Persona、Scene Navigation 和 L2/L3 不变，再对同一状态实际执行
`0/0.2/0.4/0.6/0.8/1.0` 六档预算。LongMemEval oracle 500 题已全部完成，得到
3,000 条 nominal transitions、2,418 条可训练 transitions 和 1,528 次真实 API
observation；回答和 judge 均使用 `gpt-5.6-luna`，judge 使用官方 prompt，但不是官方
GPT-4o，因此标记为 `official-prompt-custom-judge`。

## 当前效果

目前离线 test 表现最好的单个模型是 v5 positional、`cql_alpha=1`、seed 17，策略版本
`cql-10093bdfddce`。这里的默认 top-5 即固定 Action 1.0：注入当前最多五条 L1 的
完整渲染内容，不表示占满模型上下文窗口。

| Test（41 题） | 默认 top-5 | 当前最佳单模型 | 变化 |
|---|---:|---:|---:|
| 准确率 | 56.10% | 56.10% | 0.00 pp |
| 平均 L1 Token | 107.27 | 84.83 | -22.44 |
| L1 Token 节省率 | 0% | 20.9% | +20.9 pp |
| informative test（11 题）准确率 | 100% | 100% | 0.00 pp |

该结果说明单个候选模型已经在这份 test 上做到同质量、少注入约五分之一 L1。但它是从
多个 seed 的 test 表现中选出的，存在 test selection bias，不能视为无偏的上线结果。
三 seed 平均仍低于默认；当前模型应被看作研究候选，而不是已验收的生产策略。

成本系数实验使用 `reward = quality - λ × injected_l1_tokens / 100`。下表为 n500、旧 v4
特征、三个 seed 的 test/all 平均结果；它显示继续增大 λ 会稳定减少 Token，但会直接
牺牲质量。

| λ | 准确率 | 相对默认 top-5 | 平均 L1 Token | Token 节省率 |
|---:|---:|---:|---:|---:|
| 默认 top-5 | 56.10% | — | 107.27 | — |
| 0 | 50.41% | -5.69 pp | 74.18 | 30.85% |
| 0.1 | 48.78% | -7.32 pp | 59.63 | 44.41% |
| 0.3 | 40.65% | -15.45 pp | 32.46 | 69.74% |
| 1.0 | 36.59% | -19.51 pp | 6.88 | 93.59% |

随后固定 `λ=0`，比较 CQL 系数和位置特征。v4 将 Query、全部 L1、Persona 和 Scene
混入同一 128 维文本哈希；v5 将五个 L1 位置分槽编码，并加入每档预算的实际注入量和
选中条数。下表同样是三个 seed 的 test/all 平均值，所有模型都使用 dev 最佳 epoch。

| 特征 | `cql_alpha` | 准确率 | 平均 L1 Token | 相对默认准确率 | Token 节省率 |
|---|---:|---:|---:|---:|---:|
| v4 mixed hash | 0 | 44.72% | 51.31 | -11.38 pp | 52.17% |
| v4 mixed hash | 0.1 | 43.90% | 56.37 | -12.20 pp | 47.45% |
| v4 mixed hash | 1.0 | 45.53% | 65.59 | -10.57 pp | 38.85% |
| v5 positional | 0 | 47.97% | 73.88 | -8.13 pp | 31.13% |
| v5 positional | 0.1 | 50.41% | 81.47 | -5.69 pp | 24.05% |
| v5 positional | 1.0 | 52.85% | 85.72 | -3.25 pp | 20.09% |

v5 在三个 alpha 上都提高了 test 质量，并减少了“默认答对、策略答错”的问题，证明保留
候选位置和动作结构是有效改进。`cql_alpha=0` 没有消除泛化损失，说明问题并不只是
CQL 保守项。基于三个 seed 的 Q margin 做过一次默认回退门控：dev 准确率由 44.23%
升至 46.15%，节省 26.73% Token，且没有破坏默认正确题；冻结到 test 后准确率却由
56.10% 降至 51.22%，仅节省 11.82%，仍破坏了两道默认正确题，因此该门控未通过验收。

## 训练与评估方法

每个 LongMemEval 问题先按时间顺序把历史 session 写入独立 MemoryCore namespace，等待
L1/L2/L3 pipeline 稳定，然后只召回一次候选并冻结。六档预算复用同一候选、排序、
分配器和完整渲染 Token 估算；产生相同内容的动作只调用一次回答 API，导出时展开 nominal
alias，并设置 `sample_weight=1/alias_count`，避免重复证据被放大。任务按 question ID
稳定切分 train/dev/test，同一问题的全部动作始终位于同一 split。

策略模型是单隐藏层 ReLU MLP，一次输出六个动作的 Q 值。v5 输入仅包含决策时可见信息：
Query、Persona、Scene、五个有顺序的 L1 文本槽位及长度/score/mask，以及六档动作的
预算 Token、实际注入 Token 和选中 L1 数量；不使用标准答案、judge 或未来结果。文本
使用固定哈希表示，数值做确定性变换。模型先用默认行为标签做可选分类预热，再用真实
action reward 训练；LongMemEval 是单步 terminal decision，因此没有下一状态回报。
CQL alpha 控制保守项，成本 λ 在导出数据时进入 reward，两者互相独立。训练可在每个
epoch 用 dev 平均 reward 选 checkpoint，同 reward 时选择平均注入 Token 更低的版本。

主评估始终报告固定六档、默认 top-5、策略准确率和实际 L1 Token，并按问题输出配对损益、
bootstrap CI、`fixed1_only_correct` 等四象限。`informative` 仅用于诊断不同动作确实产生
不同质量的 state，主结论仍以完整 test 为准。L1 Token 是当前 TDAI 完整渲染估算，不
等同于整次 API 请求费用；输入、输出、缓存和 judge 研究成本在原始采集中分开记录。

## 当前卡点与方向

现有动作空间并不缺少无损节省机会：在 test 上事后选择“不低于默认质量的最小动作”，
可以保持 56.10% 准确率并把平均 L1 Token 从 107.27 降到 13.17，乐观上界为节省
87.72%；只看默认已经答对的 23 题，仍可节省 73.79%。真正瓶颈是模型无法在回答前
可靠识别哪些题可以安全缩减；dev 的 informative state 只有 14 个，Q margin 也不是经
校准的风险概率。

下一步应从混合 reward Q-learning 转向 paired risk classifier，直接学习“默认 top-5
正确但低预算会错”的风险，并把 top-5 作为强制 fallback；使用 class-balanced loss、
交叉验证和更大的独立校准集，只在低风险的小部分问题上缩减。若目标是同时提高准确率，
还需新增去重、候选重排、靠近 top-5 的细粒度动作，并对新注入内容重新生成和评分。
并行进行的 TDAI v2.0.1 实验已经部署为无默认注入、运行时禁写，只允许 Agent 主动调用
L1/L0/L2 只读工具；该批次尚在采集中，结果不属于本模型当前指标。

离线复核环境为 Python 3.11+ 与 NumPy：

```powershell
python -m pip install -e ".[training]"
python scripts/predict-longmemeval-policy.py `
  --policy model/policy.json `
  --state <state.json>
```
