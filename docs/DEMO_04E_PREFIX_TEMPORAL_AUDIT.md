# Demo-04E — 前缀 / 端点时序审计

## Demo-04D 的结论

The false accept:

```text
Unknown: 大家好
Top1:    大家好 我叫王灏
Q/S = 0.720
```

它比 enrollment 短语更短。

But three correct Known queries are equally or even more short:

```text
known correct equally/more short = 3/24
```

因此：

```text
signed duration alone
```

不是安全的决策规则。

## 下一项假设

真正的前缀 hard negative，其**时序形状**应不同于完整短语的合理快速/缩短发音。

对于 top-1 enrollment 意图，Demo-04E 测量：

```text
full_dtw_sim
prefix_dtw_sim
suffix_dtw_sim
prefix_minus_suffix

start_cosine
end_cosine
start_minus_end
```

Interpretation:

```text
prefix_minus_suffix >> 0
```

表示 query 与 enrollment 短语开头的相似度高于其结尾。

```text
start_minus_end >> 0
```

表示起始端点的匹配度明显高于结束端点。

关键比较并非只看所有 query 的整体情况。报告会专门列出与 hard negative 同样短或更短的有效 Known query。

## 为什么比再加一个手写时长阈值更合适

The current H6 DTW produces a high score for:

```text
大家好
vs
大家好 我叫王灏
```

因此，仅凭时长无法判断 query 是：
- a faster full utterance, or
- only the prefix of the enrolled utterance.

端点/前缀诊断用于检验这一缺失的区分能力。

## 运行

在 `papr_ssl` 根目录运行：

```powershell
python scripts/audit_demo04e_prefix_temporal.py `
  --dataset-root "..\papr_audio_toolkit\data\exports\wanghao_demo_v2" `
  --predictions artifacts\demo_04a_wanghao_runtime_v2\predictions.jsonl `
  --memory artifacts\demo_04a_wanghao_runtime_v2\wanghao_user_memory.pt `
  --output-dir artifacts\demo_04e_prefix_temporal_audit
```

不进行训练、threshold 拟合或 runtime 修改，也不访问 `generic_test`。

## 本次审计后的决策

如果 hard negative 呈现明确的前缀/端点特征，而同样短的 Known 对照没有该特征，下一步就在决策层试做**前缀风险特征**。

如果分布仍然高度重叠，就不要再添加启发式规则。更合适的下一步是收集更多真实 hard negative，并在按 session 分离的真实用户数据上拟合和评估一组小型 open-set 决策特征。
