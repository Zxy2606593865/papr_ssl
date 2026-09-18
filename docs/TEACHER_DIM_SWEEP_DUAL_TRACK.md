# Teacher Embedding Dimension Sweep — Dual-Track Policy

Two project goals are now separated.

## Project 1: Agent speech front-end
Performance first. The server-side Teacher may use a larger embedding if it improves recognition.

## Project 2: Edge hardware
Efficiency first. The existing canonical 64D Teacher is preserved for later Student distillation.

So this experiment does **not** replace 64D.

### Canonical 64D stays frozen

```text
WavLM-large
hidden_states[15]
-> Attention DR
-> 64D L2 embedding
-> SCAF
```

### New sweep

Only projection output dimension changes:

```text
WavLM[15] cached [T,1024]
        |
        v
same Attention Pooling
        |
        v
64 / 128 / 256 / 512
        |
        v
L2 Norm
        |
        v
SCAF (K=3, margin=0.2 rad, scale=30)
```

Frozen training setup:

```text
MDSC Core30 train/dev
20 epochs
AdamW lr=1e-3
weight_decay=1e-4
grad clip=5
8-way x 4-shot
seeds 17 / 29 / 43
dev prototype Macro-F1 selection
generic_test sealed
```

## Run order

```powershell
python scripts/audit_teacher_dim_sweep.py
```

Then smoke test:

```powershell
python scripts/run_teacher_dim_sweep.py `
  --smoke `
  --output-dir artifacts/p6_teacher_dim_sweep_smoke
```

If PASS, run full sweep:

```powershell
python scripts/run_teacher_dim_sweep.py
```

Finally:

```powershell
python scripts/summarize_teacher_dim_sweep.py
```

The 64D sweep result is only a reproduction control. It never overwrites the canonical 64D Teacher.

After this sweep, compare `canonical 64D` vs `best high-dimensional candidate` under the same 15-shot personalized/open-set evaluation before promoting a new Project-1 Teacher.
