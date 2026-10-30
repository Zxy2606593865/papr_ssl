# P4-06 — 每种 Backbone 的最佳层与最终 Backbone 比较

输入：

```text
P4-05:
63 Backbone/Layer configurations
3 seeds/configuration
189 raw layer-seed results
generic_dev_score = prototype_macro_f1
generic_test sealed
```

P4-06 是首个允许选择获胜项的 P4 阶段。

选择规则：

```text
对每种 Backbone：
    最大化 3-seed mean generic_dev_score
    均值完全相同时 -> 选择较小的 hidden_state_index

随后：
    对比每种 Backbone 的获胜项
    最大化同一 3-seed mean generic_dev_score
```

P4-06 不允许用资源成本取代 embedding 质量作为选择依据。资源成本由 P4-07 随后记录。

运行：

```powershell
python scripts/select_p4_06_backbone_layer.py
python scripts/audit_p4_06_selection.py
```

输出：

```text
artifacts/p4_runs/p4_06_selection/
├── p4_06_selection.json
└── p4_06_winners.csv
```

P4-06 通过后：
- P4-07 记录参数量、latency、memory 和 cache 成本。
- P4-08 冻结选定的 Backbone 与精确 `hidden_state_index`，作为 P5 唯一主线。
