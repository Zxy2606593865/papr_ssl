# P3 科研 Gate — 真实实验 01

这**不是 P3-09**。P3-01 至 P3-08 已构成完整工程计划。本文件使用真实数据执行这些契约。

## 首个真实实验条件

```text
Task:      MDSC-Core30
Backbone:  facebook/wav2vec2-base
Revision:  0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
Layer:     outputs.hidden_states[12]
Pooling:   masked mean, offline cache
Head:      Mean DR -> 64D L2
Loss:      SCAF K=3, m=0.2, s=30
Seed:      17 first
Test:      sealed
```

冻结的 Core30 policy 在 30 个 classes 中包含 3756 条 train utterances 和 442 条 dev utterances。首份真实 cache 只包含 train+dev；P3 模型开发不会物化 Core30 test audio。

## 为什么先使用 Wav2Vec2？

目标是在扩大实验矩阵前，先验证一套端到端 P3 真实条件。Wav2Vec2-base 是三个 P1 backbones 中最小的一个，其最终 hidden-state index 已冻结并验证。

## Step 1 — Preflight

在项目根目录运行：

```powershell
python scripts/preflight_p3_real_run01.py --check-all-audio
```

预期结果：

```text
train rows: 3756
dev rows:   442
classes:    30
REAL RUN 01 PREFLIGHT: PASS
```

默认 MDSC 根目录为：

```text
datasets/public/mdsc
```

仅当本地 P2 audit 使用其他位置时，才覆盖 `--mdsc-root`。

## Step 2 — 物化真实 SSL Cache

```powershell
python scripts/materialize_p3_core30_wav2vec2_cache.py `
  --layers 12 `
  --device cuda `
  --batch-size 4
```

注意：Wav2Vec2 仅对语义 waveform 长度完全相同的 groups 执行 forward。SSL forward 不添加右侧 padding，与安全的 P2-06 policy 一致。

The cache is written under:

```text
artifacts/ssl_feature_cache/
facebook__wav2vec2-base__rev-0b5b8e868dd8__layer-12__masked-mean-f32/
```

不要为 seed 29/43 重新生成 cache。SSL backbone 已冻结，因此三个 seeds 完全复用此 cache。

如果 CUDA 显存不足，请调低：

```text
--batch-size 4
```

to `2` or `1`. This changes throughput, not the semantic feature definition,
because every chunk contains only equal-length waveforms and no padding.

## Step 3 — 首个真实 Teacher Run

```powershell
python scripts/run_p3_core30_wav2vec2_seed17.py --device cuda
```

该命令依次执行：

```text
untrained random 64D projection baseline
        ↓
20-epoch Mean DR + SCAF training
        ↓
dev metrics every epoch
        ↓
P3-07 dev-only checkpoint selection
        ↓
checkpoint SHA-256 validation
        ↓
P3-08 run_manifest.json
```

Baseline 与训练后的模型使用相同的缓存 SSL features，并在训练前采用相同的 seed-17 随机初始化。因此，该比较只衡量 Mean DR + SCAF 学习带来的影响。

输出：

```text
artifacts/p3_runs/
└── mdsc_core30__wav2vec2_base__layer12/
    └── seed_0017/
        ├── untrained_projection_baseline.json
        ├── checkpoints/
        ├── metrics.jsonl
        ├── checkpoint_hashes.jsonl
        ├── selected_checkpoint.json
        ├── baseline_vs_trained.json
        └── run_manifest.json
```

## 结果解读

完成 seed 17 后，不要宣布 P3 科研 Gate 整体通过。

Seed 17 只能回答：

> Can the real Core30 + Wav2Vec2 baseline train successfully and improve over
> its own untrained projection?

若实验成功，应使用 seeds 29 和 43 运行相同的冻结实验。P3 科研结论必须依据三个 seeds 的结果，而不是最佳的单个 seed。

## 需要收集的科研 Gate 证据

对真实 run 验证：

```text
finite train loss
no NaN / Inf
64D unit-norm error within contract
collapsed = false
selected checkpoint based only on generic_dev_score
trained dev score vs untrained projection
checkpoint SHA-256
run manifest
generic_test sealed
```

三个 seeds 全部完成后，才能决定该 Backbone/Layer 是否满足 P3 科研 Gate。
