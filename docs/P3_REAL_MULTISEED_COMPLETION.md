# P3 真实数据 Multi-seed 实验完成情况

Seed 17 已在冻结的真实实验条件下完成：

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

查看 seed 17 的结果后，不得改变训练预算。

Seed 17 选择的 epoch 是 20。这提示当前预算可能尚未达到平台期，但若改变 seeds 29/43 的 epoch 预算，就会使冻结的 multi-seed 比较失效。

Run:

```powershell
python scripts/run_p3_core30_wav2vec2_seed.py --seed 29 --device cuda
python scripts/run_p3_core30_wav2vec2_seed.py --seed 43 --device cuda
```

每个 seed 对应一个目录：

```text
seed_0029/
seed_0043/
```

并分别包含：

```text
untrained_projection_baseline.json
checkpoints/
metrics.jsonl
checkpoint_hashes.jsonl
selected_checkpoint.json
baseline_vs_trained.json
run_manifest.json
```

复用此前物化的同一份 Wav2Vec2 cache。

两者均完成后运行：

```powershell
python scripts/summarize_p3_core30_wav2vec2_multiseed.py
```

该命令会生成：

```text
scientific_gate_summary.json
experiment_manifest.json
```

并报告 seeds 17/29/43 上 baseline、选中训练分数和绝对增益的均值 ± 样本标准差。

如果三个被选 checkpoint 都落在 epoch 20 或附近，且 dev 曲线仍在上升，应先完成本次冻结的 20-epoch 实验。之后若要延长训练预算，必须作为独立实验声明，不得静默替换现有三 seed 结果。
