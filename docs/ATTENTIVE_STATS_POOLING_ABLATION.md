# Attentive Statistics Pooling 消融实验

## 目标

当前 Project-1 Teacher 结构如下：

```text
WavLM-large
-> hidden_states[15]
-> Attention Pooling
-> 256D
-> SCAF
```

多层融合实验未能提升模型效果，因此该分支已关闭。

本实验探讨一个新问题：

> Does retaining temporal dispersion information improve the 256D Teacher?

## Control: matched attentive mean

```text
WavLM[15] [T,1024]
        |
        v
Additive Attention
        |
        v
weighted mean mu [1024]
        |
        v
Linear 1024 -> 256
        |
        v
L2 Norm
        |
        v
SCAF
```

## Candidate: Attentive Statistics Pooling

```text
WavLM[15] [T,1024]
        |
        v
same Additive Attention
       / \
      /   \
     v     v
    mu    sigma
      \   /
       \ /
     concat [2048]
        |
        v
Linear 2048 -> 256
        |
        v
L2 Norm
        |
        v
SCAF
```

Equations:

```math
alpha_t = softmax(v^T tanh(W h_t))

mu = sum_t alpha_t h_t

sigma = sqrt(sum_t alpha_t h_t^2 - mu^2)

z = L2Norm(Linear([mu, sigma]))
```

两者在概念上的唯一区别是是否保留时间标准差。

## Attention architecture control

脚本会检查当前 `build_dr("attention")` 实现，并推断其 additive attention scorer 的 hidden size。匹配对照组和 ASP 候选组都使用相同的 hidden size。

这样可以避免在不知情的情况下改变 attention 容量。

## 已冻结的设置

```text
Backbone: WavLM-large
Layer: hidden_states[15]
Feature cache: existing P5 layer15 cache
Embedding: 256D
SCAF: K=3, margin=0.2 rad, scale=30
Epochs: 20
Optimizer: AdamW
LR: 1e-3
Weight decay: 1e-4
Grad clip: 5
Sampler: 8-way x 4-shot
Seeds: 17 / 29 / 43
Selection: dev prototype Macro-F1
generic_test: sealed
```

## 运行顺序

### 1. 运行前审计

```powershell
python scripts/audit_attentive_stats_setup.py
```

### 2. Smoke test

```powershell
python scripts/run_attentive_stats_ablation.py `
  --smoke `
  --output-dir artifacts/p6_attentive_stats_ablation_smoke
```

### 3. 完整实验

```powershell
python scripts/run_attentive_stats_ablation.py
```

### 4. 汇总

```powershell
python scripts/summarize_attentive_stats_ablation.py
```

### 5. 审计

```powershell
python scripts/audit_attentive_stats_ablation.py
```

## 决策规则

不得仅凭一个 seed 将 ASP 纳入主线。

首先要求：
- 匹配的 attentive-mean 对照组在预设的 0.02 绝对容差内复现此前的 256D 水平；
- ASP 的三 seed 均值更高；
- ASP 未出现异常不稳定。

满足这些条件后，使用当前 256D Teacher 的相同评估流程，运行 15-shot 个性化 / open-set 评估。

如果 ASP 没有带来收益，则关闭 pooling 变体分支，转向下一项实质性优化：temporal latent sequence + DTW。

面向硬件项目的 canonical 64D Teacher 保持不变。
