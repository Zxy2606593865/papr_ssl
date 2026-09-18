# Demo-04C — Decision Feature Diagnostics

## 为什么现在做这个？

Demo-04B 已经把 3 个失败分成了三种不同机制：

```text
Known Reject #1
true = today_feel_happy
top1 = today_feel_happy
```

说明 representation/ranking 仍然正确，但 C/W/U Head 判成 U。

```text
Known Reject #2
true = today_feel_happy
top1 = Introduction_the_school
true intent = rank2
```

这里确实发生了 ranking error，但 H6 policy 最终 REJECT，
所以安全层阻止了 Wrong Accept。

```text
Unknown False Accept
true unknown = greet / 大家好
top1 = Self-introduction / 大家好 我叫王灏
```

这是典型前缀 hard negative。

## 现在要回答的具体问题

H6 decision head 实际输入不是只有 fused score。

它还包含：
- top1 fused score
- top1-top2 margin
- global score / margin
- DTW score / margin
- global/DTW winner agreement
- duration_log_ratio
- shot
- support cosine mean/min

所以不能只看最终 C/W/U。

Demo-04C 会把这 11 个 feature 全部导出，并计算：

```text
z = (x - training_mean) / training_std
```

以及每个 feature 对 C/W/U logit 的贡献：

```text
contribution = weight * z
```

这样就能知道：
- Known Reject #1 为什么 U 高达 0.8446；
- Known Reject #2 为什么 U 高达 0.9207；
- “大家好”为什么 duration evidence 没能阻止 ACCEPT。

## 运行

从 `papr_ssl` 根目录：

```powershell
python scripts/analyze_demo04c_decision_features.py `
  --dataset-root "..\papr_audio_toolkit\data\exports\wanghao_demo" `
  --output-dir artifacts\demo_04c_wanghao_feature_diagnostics
```

会重新执行 31 个 Query 的 frozen feature extraction，但：
- 不训练；
- 不改 threshold；
- 不改 representation；
- 不访问 generic_test。

## 输出

```text
artifacts/demo_04c_wanghao_feature_diagnostics/
├── query_feature_diagnostics.csv
├── failure_logit_contributions.csv
└── summary.json
```

## 同时要做人工听检

Demo-04B 已经复制了 3 条失败 WAV：

```text
artifacts/demo_04b_wanghao_failure_analysis/failure_wavs/
```

请分别确认：

1. 两条 `today_feel_happy` 是否完整、清晰地说了：
   `谢谢 祝你生活愉快`

2. `greet` 是否确实只说：
   `大家好`

如果 Known Reject #2 本身是截断、噪声或人工标签有问题，
就不能把它直接当成 representation error。
