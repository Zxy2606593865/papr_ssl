# H7-00 — Final Configuration Freeze

## 当前在干什么？

现在不再改模型，也暂时不做 API。

H6-04 已经把：

```text
Raw WAV Enrollment
→ Raw WAV Query
→ Global + DTW
→ C/W/U
→ ACCEPT / CONFIRM / REJECT
```

完整闭环验证完毕。

下一步在打开 `generic_test` 前，先做一次 **最终冻结（freeze）**。

## 为什么必须 freeze？

因为 `generic_test` 是最后一块真正未见测试数据。

一旦看过它的结果，如果我们再根据结果去：

```text
改阈值
换 embedding
换 layer
调 DTW λ
重新训练 Head
```

那它就不再是严格意义上的最终测试集。

所以 H7-00 要把当前所有关键配置和文件 SHA256 固定下来。

## 本次冻结内容

```text
Backbone:
  microsoft/wavlm-large
  revision = c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
  hidden_states[15]
  raw float32 waveform
  frozen

Representation:
  Attention DR
  256D
  L2 norm

Temporal:
  same 1024→256 projection
  downsample = 3
  DTW local cost = 1-cos
  path normalized
  band ratio = 0.25
  top-k = 3
  templates/class = 2
  fixed lambda = 0.50

Ranking:
  fixed Global + DTW
  H4 learned ranker NOT PROMOTED

Decision:
  shared C/W/U linear head
  H6 three-state thresholds
  ACCEPT / CONFIRM / REJECT
  scores are NOT calibrated probabilities

Enrollment:
  frozen 1-shot / 2-shot policy
  LOO/shrinkage deferred
  medoid deferred
```

## 运行

```powershell
python scripts/run_h7_00_final_freeze.py
python scripts/audit_h7_00_final_freeze.py
```

目标：

```text
H7-00 STATUS: PASS
AUDIT STATUS: PASS
generic_test accessed: NO
```

输出：

```text
artifacts/h7_00_final_freeze/frozen_config.json
artifacts/h7_00_final_freeze/freeze_seal.json
```

`freeze_seal.json` 会记录最终配置文件 SHA256。

## H7-00 PASS 后

才进入：

```text
H7-01 FINAL GENERIC TEST
```

H7-01 不能再做超参数搜索，只能按被冻结配置一次性评估并报告结果。
