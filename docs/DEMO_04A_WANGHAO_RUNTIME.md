# Demo-04A — Wanghao Real-WAV H6 Runtime

## 当前在干什么？

`papr_audio_toolkit` 已经导出：

```text
5 registered intents
2-shot enrollment / intent
5 query / intent
3 unknown intents
2 unknown query / intent
```

总共：

```text
10 enrollment WAV
25 known query WAV
6 unknown query WAV
= 41 manifest rows
```

现在切回 `papr_ssl`，不再训练模型。

流程：

```text
Enrollment WAV
→ H6 RawWavFeatureAdapter
→ WavLM-large[15]
→ Attention DR 256D global
+ temporal 256D
→ UserMemory

Query WAV
→ 同一 Adapter
→ Global Prototype + Top3 DTW
→ λ=0.50
→ C/W/U Decision Head
→ ACCEPT / CONFIRM / REJECT
```

## 这一步验证什么？

这是当前最重要的一次真实外部域工程验证：

```text
MDSC 开发出的 representation/runtime
↓
王灏真实录音
↓
2-shot 个性化
↓
跨 source query
```

它不是 pristine research test，因为：
- 王灏数据已经用于切分、聚类、人工挑选 Demo intent；
- 但 Query WAV 没有用于 H6 训练或 threshold fitting；
- `generic_test` 仍然保持 sealed。

## 一个重要限制

当前 H6 C/W/U Head 和三态 policy 是在：

```text
20 registered intents
```

的 MDSC episode 协议下开发的。

王灏 Demo 现在只有：

```text
5 registered intents
```

所以如果：
- embedding 能把正确 intent 排第一；
- 但大量出现 CONFIRM / REJECT；

这可能是 5-intent vs 20-intent 的 decision-policy distribution shift，
不能直接解释为 representation 失败。

第一轮禁止修改 threshold。

先测原样迁移能力。

## 使用

把本 bundle 覆盖到 `papr_ssl` 根目录。

然后从 `papr_ssl` 根目录运行：

```powershell
python scripts/run_demo04a_wanghao_runtime.py `
  --dataset-root "..\papr_audio_toolkit\data\exports\wanghao_demo" `
  --output-dir artifacts\demo_04a_wanghao_runtime
```

完成后：

```powershell
python scripts/audit_demo04a_wanghao_runtime.py
```

## 输出

```text
artifacts/demo_04a_wanghao_runtime/
├── wanghao_user_memory.pt
├── predictions.jsonl
└── result.json
```

重点指标：

```text
KNOWN
CA       correct accept
WA       wrong accept
CONFIRM
REJECT

UNKNOWN
ACCEPT
CONFIRM
REJECT
```

以及：

```text
PER INTENT
```

## 解释顺序

第一优先看：

```text
Known WA
```

因为这是最危险错误。

第二看：

```text
Known CA
```

第三看：

```text
Unknown ACCEPT
```

不要只看总体 accuracy。

## 下一步

根据 Demo-04A 结果分三种情况：

### A. Known CA 高、WA 低、Unknown Reject/Confirm 合理
直接进入 Demo API / UI。

### B. Top-1 intent 基本对，但大量 CONFIRM/REJECT
说明表示可能可用，但 H6 三态 policy 在 5-intent 真实域上发生分布偏移。
下一步做独立的 Demo policy calibration，不能改 representation。

### C. 大量 Wrong Intent
说明真实域 representation / clustering / enrollment 本身有问题。
先做 embedding-level error analysis，而不是调 threshold。
