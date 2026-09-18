# P5 — Mean DR vs Attention DR

P4 is closed and the sole feature source is frozen:

```text
microsoft/wavlm-large
revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
outputs.hidden_states[15]
frame dim = 1024
```

The frame cache has already passed:

```text
train 3756
dev 442
total 4198
total frames 509646
float32
generic_test sealed
```

P5 changes one factor only:

```text
Mean DR
vs
Attention DR
```

Both produce a 64D L2-normalized embedding and are trained with:

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

Attention DR is a minimal additive temporal attention:

```text
score_t = v^T tanh(W h_t)
alpha = masked softmax(score)
pooled = sum_t alpha_t h_t
Linear(1024 -> 64)
L2 Norm
```

## Important P5 reproduction gate

The Mean branch is the control. Its 3-seed mean should remain close to the P4
WavLM[15] reference:

```text
0.886472
```

P5 allows at most an absolute 0.02 difference before the comparison is rejected
for implementation/data-pipeline investigation.

## Run

```powershell
python -m unittest discover -s tests -v

python scripts/run_p5_mean_vs_attention.py --device cuda

python scripts/aggregate_p5_mean_vs_attention.py
```

The final script reports:
- Mean 3-seed mean/std/95% CI
- Attention 3-seed mean/std/95% CI
- Mean reproduction difference vs P4
- Attention - Mean delta
- selected DR method
- generic_test remains sealed
