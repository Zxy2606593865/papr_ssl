# PAPR-SSL Teacher Integration Report

**Document:** `docs/T_INTEGRATION_REPORT.md`

**Project:** PAPR-SSL v1.0

**Phase:** P1 — Teacher Integration

**Status:** **PASS**
**Date:** 2026-09-10

---

## 1. Purpose

This document records the completion and acceptance evidence for **P1 — Teacher Integration** in PAPR-SSL.

The objective of P1 is to establish a reproducible, engineering-runnable teacher-side SSL backbone integration layer before any large-scale public-data experiments, teacher selection, representation-head training, SCAF training, knowledge distillation, or Student development.

P1 validates three pretrained SSL backbones:

- `facebook/wav2vec2-base`
- `microsoft/wavlm-large`
- `facebook/w2v-bert-2.0`

The scope of this report is restricted to **real-checkpoint integration and engineering validation**. It does **not** include P2+ dataset engineering, Teacher training, Mean DR/SCAF optimization, Attention DR comparison, KD, or Student evaluation.

---

## 2. P1 Acceptance Criteria

P1 is considered complete only when all of the following are satisfied:

1. A dedicated PAPR-SSL runtime environment exists and is reproducible.
2. All three real pretrained SSL checkpoints can be loaded.
3. All three models can complete real read-only forward passes.
4. All backbones return a common `SSLBackboneOutput` contract.
5. Hidden-state indexing is explicit and validated.
6. Batch and mask behavior is understood and does not silently misalign.
7. Formal candidate layers are checked as VALID/INVALID on real checkpoints.
8. CPU/GPU engineering resource usage is measured on the actual development machine.
9. Hugging Face revisions are pinned from mutable `main` to immutable 40-character commit hashes.
10. Real-checkpoint integration tests exist but are not mixed into the default offline unit-test suite.
11. No silent fallback from one backbone implementation to another is permitted.

All criteria above are satisfied.

---

## 3. Development Environment

### 3.1 Local development machine

The P1 resource measurements were collected on the actual Windows development machine.

- Operating system: Windows 10/11 kernel family (`Windows-10-10.0.26200-SP0` as reported by Python)
- CPU: Intel64 Family 6 Model 183, GenuineIntel
- Logical CPU count: 24
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- CUDA runtime reported by PyTorch: 12.8

### 3.2 Python environment

Dedicated Conda environment:

```text
papr_ssl
```

Key versions recorded during P1:

```text
Python              3.11.16
torch               2.11.0+cu128
torchaudio          2.11.0+cu128
transformers        5.13.0
numpy               1.26.4
PyYAML              6.0.3
pytest              9.1.1
soundfile           0.14.0
safetensors         0.8.0
huggingface_hub     1.30.0
CUDA runtime        12.8
```

P1-10 additionally records:

```text
artifacts/p1_10/pip_freeze.txt
artifacts/p1_10/key_dependencies.json
```

These files provide the full Python package snapshot and the key dependency snapshot used for the final pinned-checkpoint validation.

---

## 4. Common SSL Backbone Contract

All three backbones are integrated through a common output contract:

```text
SSLBackboneOutput
├── features:    float32 [B, T, D]
├── frame_mask:  bool    [B, T]
└── hidden_layer: int
```

The selected `hidden_layer` corresponds to the **exact index in `outputs.hidden_states`**.

The P1 waveform-level input contract used for smoke and resource validation is:

```text
batch size      = 1
sample rate     = 16000 Hz
window length   = 1.0 s
samples/window  = 16000
waveform dtype  = float32
```

The three backbones are allowed to retain their own internal frontend implementations. P1 does not force an artificial frontend unification.

---

## 5. P1-04 — Wav2Vec2 Real Integration

### 5.1 Model

```text
kind:        wav2vec2
model_name:  facebook/wav2vec2-base
```

Pinned revision:

```text
0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
```

Configured hidden layer:

```text
12
```

### 5.2 Real forward result

Observed model structure:

```text
num_hidden_layers       = 12
hidden_size             = 768
hidden_states_count     = 13
valid hidden indices    = [0, 12]
```

Real wrapper output:

```text
features    = [1, 49, 768]
frame_mask  = [1, 49]
dtype       = float32 / bool
finite      = True
```

The selected raw hidden state and wrapper output shape match exactly.

### 5.3 Frontend

Wav2Vec2 receives raw waveform input:

```text
input_values = [1, 16000]
```

P1 frontend alignment check:

```text
PASS
```

### 5.4 Load-report note

When loading `Wav2Vec2Model` from `facebook/wav2vec2-base`, Transformers reports several pretraining-only parameters as `UNEXPECTED`, such as quantizer/projector weights.

This is accepted for P1 because the project intentionally uses the bare Wav2Vec2 representation model rather than the complete pretraining objective stack.

No missing representation-path parameter or silent fallback was observed.

---

## 6. P1-05 — WavLM Real Integration

### 6.1 Model

```text
kind:        wavlm
model_name:  microsoft/wavlm-large
```

Pinned revision:

```text
c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
```

Configured hidden layer:

```text
24
```

### 6.2 Real forward result

Observed structure:

```text
num_hidden_layers       = 24
hidden_size             = 1024
hidden_states_count     = 25
valid hidden indices    = [0, 24]
```

Real wrapper output:

```text
features    = [1, 49, 1024]
frame_mask  = [1, 49]
dtype       = float32 / bool
finite      = True
```

### 6.3 Frontend

WavLM receives raw waveform input:

```text
input_values = [1, 16000]
```

Frontend alignment check:

```text
PASS
```

---

## 7. P1-06 — W2v-BERT 2.0 Real Integration

### 7.1 Model

```text
kind:        w2v_bert2
model_name:  facebook/w2v-bert-2.0
```

Pinned revision:

```text
da985ba0987f70aaeb84a80f2851cfac8c697a7b
```

Configured hidden layer:

```text
24
```

### 7.2 Real forward result

Observed structure:

```text
num_hidden_layers       = 24
hidden_size             = 1024
hidden_states_count     = 25
valid hidden indices    = [0, 24]
```

Real wrapper output:

```text
features    = [1, 49, 1024]
frame_mask  = [1, 49]
dtype       = float32 / bool
finite      = True
```

### 7.3 Frontend

W2v-BERT 2.0 uses its own feature extractor before the Transformer stack.

Observed model inputs:

```text
input_features = [1, 49, 160]  float32
attention_mask = [1, 49]       int64
```

Wrapper output mask:

```text
frame_mask = [1, 49] bool
```

Frontend and output temporal dimensions are aligned.

P1 frontend alignment check:

```text
PASS
```

---

## 8. P1-07 — Batch / Padding / Mask Validation

The current P1 public wrapper API accepts fixed-window waveforms and does not expose explicit waveform lengths or a waveform-validity mask.

Therefore, arbitrary zero-padded tail semantics cannot be inferred honestly: a zero-valued tail may represent either padding or valid silence.

P1-07 consequently distinguishes between:

- singleton validation;
- true multi-sample full-length batch validation;
- true padded-batch semantics, marked N/A under the current fixed-window contract.

Formal status:

```text
PASS_WITH_PADDED_NA
```

### 8.1 Wav2Vec2

```text
batch waveform     = [3, 16000]
features           = [3, 49, 768]
frame_mask         = [3, 49]
valid frames       = [49, 49, 49]
singleton/batch max diff ≈ 2.62e-4
```

### 8.2 WavLM

```text
batch waveform     = [3, 16000]
features           = [3, 49, 1024]
frame_mask         = [3, 49]
valid frames       = [49, 49, 49]
singleton/batch max diff ≈ 3.21e-4
```

### 8.3 W2v-BERT 2.0

```text
batch waveform     = [3, 16000]
features           = [3, 49, 1024]
frame_mask         = [3, 49]
valid frames       = [49, 49, 49]
singleton/batch max diff ≈ 5.92e-4
```

The small singleton-vs-batch differences are within normal floating-point differences produced by GPU execution paths and do not indicate a contract violation.

---

## 9. P1-08 — Candidate Layer Validity

Candidate layers are taken from the formal Teacher layer-sweep configuration and expanded by the project's `load_layer_sweep()` implementation.

This step checks structural validity only.

It does **not** claim that a VALID layer is the best representation layer.

### 9.1 Wav2Vec2

Observed valid hidden-state range:

```text
[0, 12]
```

Formal candidates:

```text
8   VALID
12  VALID
```

Summary:

```text
VALID   = [8, 12]
INVALID = []
```

### 9.2 WavLM

Observed valid hidden-state range:

```text
[0, 24]
```

Formal candidates:

```text
8   VALID
12  VALID
16  VALID
20  VALID
22  VALID
```

Summary:

```text
VALID   = [8, 12, 16, 20, 22]
INVALID = []
```

### 9.3 W2v-BERT 2.0

Observed valid hidden-state range:

```text
[0, 24]
```

Formal candidates:

```text
8   VALID
12  VALID
16  VALID
20  VALID
22  VALID
```

Summary:

```text
VALID   = [8, 12, 16, 20, 22]
INVALID = []
```

Overall P1-08 result:

```text
PASS
```

These candidate sets may proceed to P4 representation-quality comparison.

---

## 10. P1-09 — CPU/GPU Resource Benchmark

### 10.1 Protocol

Formal P1-09 benchmark protocol:

```text
batch size     = 1
window         = 1.0 s
sample rate    = 16000 Hz
samples        = 16000
inference      = torch.inference_mode()
warmup runs    = 10
timed runs     = 50
```

CPU and CUDA measurements were executed in separate processes.

Teacher integration is not optimized for minimum latency. The requirement is engineering runnability on the actual development machine.

### 10.2 Final benchmark summary

| Backbone | CPU Mean | CPU P95 | CPU Peak RSS | GPU Mean | GPU P95 | GPU Peak Allocated | GPU Peak Reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wav2Vec2 Base | 37.07 ms | 39.45 ms | 1058.4 MB | 7.06 ms | 7.96 ms | 397.3 MB | 456 MB |
| WavLM Large | 106.45 ms | 141.77 ms | 1891.9 MB | 19.70 ms | 24.58 ms | 1261.6 MB | 1282 MB |
| W2v-BERT 2.0 | 201.25 ms | 217.57 ms | 2858.4 MB | 32.84 ms | 34.56 ms | 2231.2 MB | 2246 MB |

### 10.3 Load time

| Backbone | CPU Load | GPU Load |
|---|---:|---:|
| Wav2Vec2 Base | 2.78 s | 2.75 s |
| WavLM Large | 3.68 s | 3.95 s |
| W2v-BERT 2.0 | 3.69 s | 5.06 s |

### 10.4 CPU process-memory observations

Wav2Vec2:

```text
baseline RSS              ≈ 495.3 MB
after load RSS            ≈ 649.5 MB
observed peak RSS         ≈ 1058.4 MB
peak increment            ≈ 563.1 MB
```

WavLM:

```text
baseline RSS              ≈ 495.9 MB
after load RSS            ≈ 680.7 MB
observed peak RSS         ≈ 1891.9 MB
peak increment            ≈ 1396.0 MB
```

W2v-BERT 2.0:

```text
baseline RSS              ≈ 496.0 MB
after load RSS            ≈ 622.1 MB
observed peak RSS         ≈ 2858.4 MB
peak increment            ≈ 2362.4 MB
```

CPU host-memory measurement uses Windows `GetProcessMemoryInfo` / `PROCESS_MEMORY_COUNTERS_EX`, which captures process-level native allocations rather than Python-only allocations.

### 10.5 GPU memory observations

Wav2Vec2:

```text
resident allocated after load ≈ 360.3 MB
forward peak allocated        ≈ 397.3 MB
forward peak reserved         ≈ 456.0 MB
```

WavLM:

```text
resident allocated after load ≈ 1203.4 MB
forward peak allocated        ≈ 1261.6 MB
forward peak reserved         ≈ 1282.0 MB
```

W2v-BERT 2.0:

```text
resident allocated after load ≈ 2214.5 MB
forward peak allocated        ≈ 2231.2 MB
forward peak reserved         ≈ 2246.0 MB
```

### 10.6 Engineering conclusion

All three Teacher backbones are engineering-runnable on the current development machine.

Even W2v-BERT 2.0, the heaviest P1 candidate, completes a 1-second input forward in approximately:

```text
CPU mean ≈ 201 ms
GPU mean ≈ 32.8 ms
```

Therefore, local development is not restricted to a trivial smoke-only path. The machine is sufficient for real integration, debugging, bounded feature extraction, and Teacher-side development.

For larger public-data sweeps, repeated layer extraction, or long-running experiments, a server/GPU environment may still be more operationally efficient. That is a workflow consideration rather than a P1 engineering blocker.

P1-09 result:

```text
PASS
```

---

## 11. P1-10 — Immutable Revision Pinning

All three Hugging Face checkpoints were changed from mutable:

```text
revision: main
```

to immutable 40-character commit hashes.

### 11.1 Final pinned model table

| Kind | Model | Immutable revision |
|---|---|---|
| `wav2vec2` | `facebook/wav2vec2-base` | `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8` |
| `wavlm` | `microsoft/wavlm-large` | `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` |
| `w2v_bert2` | `facebook/w2v-bert-2.0` | `da985ba0987f70aaeb84a80f2851cfac8c697a7b` |

Pinned revisions were written into the Teacher YAML configurations.

### 11.2 Post-pin real smoke validation

All three models were reloaded from their immutable revisions and completed real CUDA forward passes.

Wav2Vec2:

```text
revision          = 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
hidden states     = 13
features          = [1, 49, 768]
shape match       = True
frontend align    = True
SMOKE RESULT      = PASS
```

WavLM:

```text
revision          = c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
hidden states     = 25
features          = [1, 49, 1024]
shape match       = True
frontend align    = True
SMOKE RESULT      = PASS
```

W2v-BERT 2.0:

```text
revision          = da985ba0987f70aaeb84a80f2851cfac8c697a7b
hidden states     = 25
features          = [1, 49, 1024]
shape match       = True
frontend align    = True
SMOKE RESULT      = PASS
```

The pinned revisions therefore preserve the same real-backbone contract established earlier in P1.

P1-10 result:

```text
PASS
```

---

## 12. P1-11 — Real-Checkpoint Integration Tests

P1-11 adds a dedicated real-model integration-test layer without contaminating the default offline unit-test suite.

Integration test files:

```text
tests/
├── conftest.py
└── integration/
    ├── __init__.py
    ├── README.md
    └── test_real_ssl_backbones.py
```

### 12.1 Default offline suite

The default unit tests do not load large Hugging Face checkpoints.

Command:

```powershell
python -m unittest discover -s tests -v
```

Final result:

```text
Ran 22 tests
OK
```

Status:

```text
22 / 22 PASS
```

### 12.2 Explicit real-model test mode

Real-checkpoint tests require explicit opt-in:

```powershell
python -m pytest tests/integration/test_real_ssl_backbones.py `
  --run-real-models `
  --hf-mode cache `
  --real-model-device cuda `
  -v
```

Final result:

```text
collected 4 items

test_pinned_real_ssl_backbone_contract[wav2vec2]   PASSED
test_pinned_real_ssl_backbone_contract[wavlm]      PASSED
test_pinned_real_ssl_backbone_contract[w2v_bert2]  PASSED
test_real_model_configs_are_immutably_pinned       PASSED

4 passed in 6.54s
```

Status:

```text
4 / 4 PASS
```

### 12.3 Network/cache isolation

`--hf-mode cache` explicitly uses local Hugging Face cache/offline mode.

Therefore, routine P1 integration verification can be run without silently downloading large checkpoints.

Network access can be explicitly enabled only when intentionally requested.

### 12.4 Integration-test assertions

The real-model tests validate:

- expected backbone kind;
- expected model identifier;
- expected immutable revision;
- revision is not `main`;
- revision length is 40 characters;
- expected configured hidden layer;
- expected model hidden-state depth;
- expected hidden dimension;
- successful real read-only forward;
- common `SSLBackboneOutput` shape/dtype contract;
- finite output features;
- valid frame mask;
- no silent model-kind fallback.

P1-11 result:

```text
PASS
```

---

## 13. P1 Task Status

| Task | Description | Result |
|---|---|---|
| P1-01 | Independent runtime environment | PASS |
| P1-02 | Dependency/version plan | PASS |
| P1-03 | Shared SSL smoke CLI | PASS |
| P1-04 | Wav2Vec2 real-checkpoint integration | PASS |
| P1-05 | WavLM real-checkpoint integration | PASS |
| P1-06 | W2v-BERT 2.0 real-checkpoint integration | PASS |
| P1-07 | Batch / padding / mask validation | PASS_WITH_PADDED_NA |
| P1-08 | Candidate-layer VALID/INVALID validation | PASS |
| P1-09 | CPU/GPU resource benchmark | PASS |
| P1-10 | Immutable revision pinning + dependency snapshot | PASS |
| P1-11 | Explicit real-model integration tests | PASS |
| P1-12 | Teacher integration report | PASS |

---

## 14. Final P1 Gate Decision

### 14.1 Functional integration

```text
Wav2Vec2      PASS
WavLM         PASS
W2v-BERT 2.0  PASS
```

All three backbones complete real forward passes and satisfy the common representation contract.

### 14.2 Hidden-state indexing

```text
Wav2Vec2      13 observed hidden states, valid [0,12]
WavLM         25 observed hidden states, valid [0,24]
W2v-BERT 2.0  25 observed hidden states, valid [0,24]
```

No hidden-layer indexing ambiguity remains for the current Transformers/model revisions.

### 14.3 Mask alignment

```text
Wav2Vec2      PASS
WavLM         PASS
W2v-BERT 2.0  PASS
```

No temporal mask/output misalignment was observed.

True variable-length padded-batch semantics remain outside the fixed-window P1 wrapper contract and are explicitly recorded as N/A rather than inferred.

### 14.4 Reproducibility

```text
model revisions pinned      PASS
dependency versions saved   PASS
post-pin real smoke         PASS
```

All production candidate checkpoints use immutable commit hashes.

### 14.5 Test isolation

```text
offline unit tests          22/22 PASS
real integration tests       4/4 PASS
real tests explicit opt-in   PASS
cache-only validation        PASS
```

Real model tests do not contaminate the ordinary offline test path.

### 14.6 Engineering runnability

All three models fit and execute successfully on the actual development machine.

Peak GPU allocation for the heaviest model, W2v-BERT 2.0, is approximately:

```text
2231.2 MB
```

and the measured CUDA mean forward latency for a 1-second input is approximately:

```text
32.84 ms
```

No P1 engineering blocker remains.

---

# P1 FINAL RESULT: PASS

**Teacher Integration is accepted and closed.**

The project may proceed to **P2 — Public Data Engineering**.

P2 should build on the frozen P1 contracts and must not silently alter:

- model identifiers;
- pinned model revisions;
- hidden-layer indexing semantics;
- waveform-level input contract;
- `SSLBackboneOutput`;
- default offline-test isolation;
- no-silent-fallback behavior.

Any later change to one of these items should be treated as a contract change and revalidated against the relevant P1 tests.
