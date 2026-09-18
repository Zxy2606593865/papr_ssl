# P4-07 — Model Scale / Resource Cost Recording

P4-07 does not choose the Teacher.

The Teacher was chosen in P4-06 using:

```text
3-seed mean generic_dev_score
= prototype_macro_f1
```

P4-07 only records the cost of the three per-Backbone winner configurations.

## Metrics

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

The latency benchmark uses:

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

## Important interpretation

The selected hidden state is read from the standard Hugging Face full forward:

```text
output_hidden_states=True
```

No early exit or encoder truncation is assumed. Therefore selecting an
intermediate hidden_state_index does not automatically imply reduced backbone
compute. Any future early-exit/truncation optimization would be a separate
experiment.

## Run

P4-06 must already have produced:

```text
artifacts/p4_runs/p4_06_selection/p4_06_selection.json
```

Then:

```powershell
python scripts/profile_p4_07_resource_cost.py --device cuda
python scripts/audit_p4_07_resource_cost.py
```

Outputs:

```text
artifacts/p4_runs/p4_07_resource_cost/
├── p4_07_resource_cost.json
└── p4_07_resource_cost.csv
```

PASS requires:
- all 3 Backbone winners profiled
- positive parameter/latency measurements
- standard full HF forward
- no early exit/truncation
- resource cost did not alter P4-06 Teacher selection
- generic_test not accessed

After P4-07 PASS, P4-08 freezes the P4-06 winner as the sole P5 mainline.
