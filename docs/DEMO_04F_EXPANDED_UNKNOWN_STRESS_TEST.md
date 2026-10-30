# Demo-04F — Wanghao 扩展未知样本压力测试

## 为什么接下来做这项测试

Demo-04E did **not** produce a robust prefix feature:

```text
Unknown false accept:
prefix-suffix DTW = -0.0019
start-end cosine  = -0.1033
```

这两个数值都与更广泛的 Known / Unknown 分布重叠。

一个值得关注的现象是，在 Demo-04E 打印的五个同等或更短的 Known 对照样本中，hard negative 的 `start-end` 值更负。但目前只有：

```text
1 hard-negative sample
```

这不足以制定生产规则。

修改 H6 前，先利用现有数据。

## 已确认但未注册的 Wanghao 意图

toolkit 中包含人工确认、但不属于五个已注册 Demo 意图的类别，包括：

```text
greet
see_you_again
coffee_ready
hold_on
make_coffee
michael_jordan
teacher
```

Demo-04F 使用完全冻结的 Wanghao-v2 memory，评估每种此类未注册意图的**所有可用已确认样本**。

核心问题是：

```text
Does "大家好" -> Self-introduction false accept repeat across all greet samples?
```

如果该现象系统性重复，就能说明存在真实的语义前缀失效模式。

如果只有一个样本失败，而其他 greet 样本均被正确拒识，就不要围绕单个现象设计特殊规则。

## 运行

在 `papr_ssl` 根目录运行：

```powershell
python scripts/run_demo04f_expanded_unknown.py `
  --toolkit-root "..\papr_audio_toolkit" `
  --dataset-root "..\papr_audio_toolkit\data\exports\wanghao_demo_v2" `
  --memory artifacts\demo_04a_wanghao_runtime_v2\wanghao_user_memory.pt `
  --output-dir artifacts\demo_04f_unknown_expansion
```

## 输出

```text
artifacts/demo_04f_unknown_expansion/
├── predictions.jsonl
├── per_intent_summary.csv
└── result.json
```

## 结果解读

这仍是工程压力测试，而非独立基准：
- 只有一位真实用户；
- 意图由人工确认；
- 若干未注册意图只有一条源录音；
- 未采用按 session 分离的采集协议。

本步骤不会更改任何 threshold 或策略。
