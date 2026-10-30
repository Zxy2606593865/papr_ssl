# P5 — Mean DR 与 Attention DR 对比

P4 已完成，唯一特征来源已冻结：

```text
microsoft/wavlm-large
revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
outputs.hidden_states[15]
frame dim = 1024
```

帧级 cache 已通过检查：

```text
train 3756
dev 442
total 4198
total frames 509646
float32
generic_test sealed
```

P5 只改变一个因素：

```text
Mean DR
vs
Attention DR
```

两者都会生成 64D L2-normalized embedding，并采用以下设置训练：

```text
SCAF K=3
margin=0.2 rad
scale=30
20 epochs
AdamW lr=1e-3
weight_decay=1e-4
grad_clip=5
8-way x 4-shot
seeds 17,29,43
dev selection = prototype Macro-F1
```

Attention DR 使用最简的加性时序 attention：

```text
score_t = v^T tanh(W h_t)
alpha = masked softmax(score)
pooled = sum_t alpha_t h_t
Linear(1024 -> 64)
L2 Norm
```

## 重要的 P5 复现 Gate

Mean 分支作为 control，其 3-seed mean 应接近 P4 WavLM[15] 的参考值：

```text
0.886472
```

若绝对差异超过 0.02，则拒绝本次比较，并先调查实现或数据 pipeline。

## 运行

```powershell
python -m unittest discover -s tests -v

python scripts/run_p5_mean_vs_attention.py --device cuda

python scripts/aggregate_p5_mean_vs_attention.py
```

最后一个脚本会报告：
- Mean 的 3-seed mean/std/95% CI
- Attention 的 3-seed mean/std/95% CI
- Mean 相对 P4 的复现差异
- Attention - Mean delta
- 选定的 DR method
- `generic_test` 仍保持封存
