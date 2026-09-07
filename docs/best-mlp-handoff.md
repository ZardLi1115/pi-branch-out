# TDAI Budget Policy 初版 MLP 交接

本包交付当前离线评测表现最好的单个 MLP：`visible-state-hash-v5-positional-l1-actions`、
`cql_alpha=1`、seed 17，策略版本 `cql-10093bdfddce`。模型在 LongMemEval oracle 的固定
test 集 41 题上准确率为 56.10%，与 TDAI v2.0.0-beta.1 默认完整 top-5 L1 相同；平均
L1 注入量由 107.27 降为 84.83 Token，减少 22.44 Token（20.9%）。在 dev 52 题上，
模型准确率 46.15%，默认 top-5 为 44.23%。这些结果使用 `gpt-5.6-luna` 和 LongMemEval
官方 judge prompt，但 judge 模型不是官方 GPT-4o，协议标记为
`official-prompt-custom-judge`。

实现机制是单隐藏层 ReLU MLP，一次输出六档预算 `0/0.2/0.4/0.6/0.8/1.0` 的 Q 值。
输入为决策时可见状态：Query、Persona、Scene、五个有位置的 L1 槽位，以及六档动作的
预算、实际注入 Token 和选中 L1 数量。缺失槽位带 mask，文本使用固定哈希表示；模型不
读取标准答案、judge 分数或未来信息。训练数据来自同一 state 下六档真实回答与评分，
内容等价动作共享 observation 并做权重校正。训练器支持 CQL、成本 reward 和 dev 最佳
epoch；本模型使用质量 reward、CQL alpha 1，并由 dev reward 选择 checkpoint。

离线推理使用 Python 3.11+ 和 NumPy：

```powershell
python -m pip install -e ".[training]"
python scripts/predict-longmemeval-policy.py `
  --policy model/policy.json `
  --state <state.json>
```

当前最大卡点是泛化和默认保护。该模型是按 test 表现从多个 seed 中选出的研究候选，
存在 test selection bias；三个 seed 的平均 test 准确率仍低于默认 top-5。基于 dev 校准
的 ensemble fallback 在 test 上仍将 2 个默认正确样本变错，只节省 11.82%，因此没有
通过上线验收。现有 TypeScript 在线插件只兼容 v4 特征，本包提供的是可复核的 Python
离线推理，不代表已经接入生产。L1 Token 还是 TDAI 字符估算值，不等同于总 API 账单。

现有配对数据仍显示很大的可挖空间：test 上事后选择“不低于默认质量的最小动作”可以
保持 56.10% 准确率，同时理论上节省 87.72% L1 Token；只看默认已经答对的题，仍可
节省 73.79%。下一步应训练专门的 paired risk classifier，预测“默认正确但低预算会错”
的风险，以 top-5 为强制回退；扩大独立校准集和 informative state 后再冻结评测。若要
进一步提高而非只保质量，应增加去重、重排和靠近 top-5 的细粒度动作，并对新内容重新
运行回答和评分。
