# WavLM 多层融合实验

## 科学问题

The current Project-1 Teacher uses:

```text
WavLM-large hidden_states[15]
-> Attention DR
-> 256D
-> SCAF
```

维度 sweep 将 closed-set 3-seed mean 提高到约：

```text
0.912667 Macro-F1
```

本实验单独检验一个新问题：

> Can nearby WavLM layers provide complementary information beyond layer 15?

## Arms

### Control 对照组

```text
WavLM layer 15
-> same Attention DR
-> 256D
-> same SCAF
```

### 可学习的多层融合

```text
WavLM layers 13,14,15,16,17
        |
        v
global trainable softmax scalar weights
        |
        v
weighted sum [T,1024]
        |
        v
same Attention DR
        |
        v
256D
        |
        v
same SCAF
```

可训练融合仅包含五个 scalar logits：

```math
alpha_l = exp(w_l) / sum_j exp(w_j)

H_fused = sum_l alpha_l H^(l)
```

WavLM 始终完全冻结。

## Cache

实验开始前，必须先为 layers 13..17 建立仅含 train/dev 的 hidden-state cache。

默认以 float16 存储以减少磁盘占用。Control 和 fusion 两组使用同一份 cache，以控制精度影响。

预计大小约为单层的五倍，再因 float16 减半。按此前的帧数计算，feature tensors 约占 4.9 GiB。

不加载 test rows。

## 运行顺序

### 1. 构建 Cache

```powershell
python scripts/build_wavlm_layers13_17_cache.py --device cuda
```

### 2. Smoke Test

```powershell
python scripts/run_wavlm_multilayer_fusion.py `
  --smoke `
  --output-dir artifacts/p6_wavlm_multilayer_fusion_smoke
```

### 3. 完整实验

```powershell
python scripts/run_wavlm_multilayer_fusion.py
```

### 4. 汇总

```powershell
python scripts/summarize_wavlm_multilayer_fusion.py
```

### 5. 审计

```powershell
python scripts/audit_wavlm_multilayer_fusion.py
```

## 冻结的设置

```text
MDSC Core30 train/dev
Attention DR
256D embedding
SCAF K=3
margin=0.2 rad
scale=30
20 epochs
AdamW lr=1e-3
weight decay=1e-4
grad clip=5
8-way x 4-shot
seeds 17/29/43
selection = dev prototype Macro-F1
generic_test = sealed
```

## 晋级规则

不得仅因某一个 seed 分数更高就晋级多层融合方案。

首先必须满足：
- a reasonable 3-seed mean improvement over the same-cache single15 control;
- no pathological variance;
- sensible learned layer weights.

若通过，下一步实验将沿用 256D Teacher 已执行过的 15-shot 个性化/open-set 比较。

Project 2 的规范 64D Teacher 保持不变。
