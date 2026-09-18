# P3 Real Multi-seed Completion

Seed 17 has already completed on the frozen real condition:

```text
MDSC-Core30
Wav2Vec2-base
revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
layer 12
Mean DR -> 64D L2
SCAF K=3, m=0.2, s=30
20 epochs
8-way x 4-shot
```

Do not change the budget after seeing seed 17.

The selected epoch for seed 17 is epoch 20. This is useful evidence that the
current budget may not have reached a plateau, but changing the epoch budget
for seeds 29/43 would invalidate the frozen multi-seed comparison.

Run:

```powershell
python scripts/run_p3_core30_wav2vec2_seed.py --seed 29 --device cuda
python scripts/run_p3_core30_wav2vec2_seed.py --seed 43 --device cuda
```

Each seed gets:

```text
seed_0029/
seed_0043/
```

with its own:

```text
untrained_projection_baseline.json
checkpoints/
metrics.jsonl
checkpoint_hashes.jsonl
selected_checkpoint.json
baseline_vs_trained.json
run_manifest.json
```

The same previously materialized Wav2Vec2 cache is reused.

After both complete:

```powershell
python scripts/summarize_p3_core30_wav2vec2_multiseed.py
```

This creates:

```text
scientific_gate_summary.json
experiment_manifest.json
```

and reports baseline, selected trained score, and absolute gain as mean ± sample
standard deviation over seeds 17/29/43.

If all three selected checkpoints land at or near epoch 20 and the dev curves
are still rising, finish this frozen 20-epoch experiment first. Any longer
training budget should then be declared as a separate experiment, not silently
substituted into the existing three-seed result.
