# P4-08 — 冻结唯一 Teacher 主线

P4-08 不进行新的模型训练或排序。

It consumes:
- P4-06 selected Teacher feature source
- P4-07 resource-cost record

并将且仅将一条 P5 主线冻结为：

```text
Backbone:
microsoft/wavlm-large

revision:
c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c

outputs.hidden_states index:
15

hidden dimension:
1024
```

本次冻结采用的 P4 结果：

```text
3-seed mean generic_dev_score = 0.886472...
sample std                    = 0.006107...
95% Student-t CI              = [0.871301..., 0.901644...]
```

另外两种 Backbone 的获胜项仅作为消融基线保留。

## 运行

```powershell
python scripts/freeze_p4_08_mainline.py
python scripts/audit_p4_08_mainline.py
```

输出：

```text
artifacts/p4_runs/p4_08_mainline/
├── p4_08_mainline_lock.json
└── p5_input_contract.json
```

## P5 交接

P5 固定使用 P4-08 的 Backbone 和 hidden-state source，只比较：

```text
Mean DR
vs
Attention DR
```

with seeds:

```text
17, 29, 43
```

注意：当前 P4 caches 是 masked-mean `[D]` caches。Attention DR 需要帧级 `[T,D]` features，因此 P5 只需为冻结的 WavLM-large `hidden_states[15]` 入围项物化 frame-level cache。

P5 主线不会重新启用其他 Backbone。
`generic_test` 继续保持封存。
