# P4-07 — 模型规模与资源成本记录

P4-07 不负责选择 Teacher。

Teacher 已在 P4-06 按以下指标选定：

```text
3-seed mean generic_dev_score
= prototype_macro_f1
```

P4-07 只记录每种 Backbone 获胜配置的资源成本。

## 指标

For each winner:

```text
parameter_count_total
float32 parameter storage bytes
selected hidden dimension
model-forward latency: median / mean / p95
waveform-to-hidden latency: median / mean / p95
CUDA resident allocated memory
CUDA peak allocated memory
forward peak extra memory
selected masked-mean cache size
```

Latency benchmark 使用以下设置：

```text
1.0 s waveform
16 kHz
batch size = 1
float32
10 warmups
30 timed repeats
```

W2v-BERT 2.0 reports two latency values:

```text
model_forward_latency_ms
    input_features already prepared
    ↓
    W2v-BERT model

waveform_to_hidden_latency_ms
    waveform
    ↓
    CPU AutoFeatureExtractor
    ↓
    H2D transfer
    ↓
    W2v-BERT model
```

For Wav2Vec2 / WavLM, waveform-to-hidden includes waveform H2D transfer +
model forward, but not audio file decoding.

## 重要解读

所选 hidden state 由标准 Hugging Face 完整 forward 读取：

```text
output_hidden_states=True
```

不假设有 early exit 或 encoder 截断。因此，选择中间层 `hidden_state_index` 并不自动降低 backbone 计算量。未来如进行 early-exit/truncation 优化，应单独开展实验。

## 运行

P4-06 必须已生成：

```text
artifacts/p4_runs/p4_06_selection/p4_06_selection.json
```

随后运行：

```powershell
python scripts/profile_p4_07_resource_cost.py --device cuda
python scripts/audit_p4_07_resource_cost.py
```

输出：

```text
artifacts/p4_runs/p4_07_resource_cost/
├── p4_07_resource_cost.json
└── p4_07_resource_cost.csv
```

通过时必须满足：
- all 3 Backbone winners profiled
- positive parameter/latency measurements
- standard full HF forward
- no early exit/truncation
- resource cost did not alter P4-06 Teacher selection
- generic_test not accessed

P4-07 通过后，P4-08 会冻结 P4-06 获胜项，作为 P5 唯一主线。
