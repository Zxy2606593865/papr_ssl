# PAPR-SSL Bootstrap Report

## 1. New project path

```text
D:\02_开发项目\02_AI视觉与机器人\02_语音识别\papr_ssl
```

The directory did not exist at the mandatory preflight check. It was created as
an independent project. The Legacy reference project under `whisper/papr` was
read only and was not imported, copied, modified, moved, or subjected to Git
operations during this bootstrap.

## 2. Directory tree

```text
papr_ssl/
├── .gitignore
├── README.md
├── pyproject.toml
├── configs/
│   └── teacher/
│       └── wav2vec2_mean.yaml
├── src/
│   └── papr_ssl/
│       ├── __init__.py
│       ├── contracts/
│       │   ├── __init__.py
│       │   └── teacher.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── windowing.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── teacher/
│       │   │   ├── __init__.py
│       │   │   ├── encoder.py
│       │   │   ├── backbones/
│       │   │   │   ├── __init__.py
│       │   │   │   └── wav2vec2.py
│       │   │   └── heads/
│       │   │       ├── __init__.py
│       │   │       └── mean_dr.py
│       │   └── student/
│       │       └── __init__.py
│       └── losses/
│           ├── __init__.py
│           └── scaf.py
├── tests/
│   ├── contracts/
│   │   └── test_teacher_contract.py
│   ├── data/
│   │   └── test_windowing.py
│   ├── losses/
│   │   └── test_scaf.py
│   └── teacher/
│       ├── test_backbone.py
│       ├── test_encoder.py
│       └── test_mean_dr.py
├── scripts/
│   └── README.md
├── datasets/
│   └── README.md
├── artifacts/
│   └── README.md
└── docs/
    └── BOOTSTRAP_REPORT.md
```

Only current Phase T0 files were created. Empty future implementation trees for
training/evaluation/utilities were not generated.

## 3. Created files

The project contains root metadata, one real Teacher YAML, a `src/` package,
four implementation modules, package exports, six offline test modules, local
data/artifact boundary documentation, and this report.

The key implementation files are:

- `src/papr_ssl/contracts/teacher.py`
- `src/papr_ssl/data/windowing.py`
- `src/papr_ssl/models/teacher/backbones/wav2vec2.py`
- `src/papr_ssl/models/teacher/heads/mean_dr.py`
- `src/papr_ssl/models/teacher/encoder.py`
- `src/papr_ssl/losses/scaf.py`

## 4. Implemented algorithms and components

### Wav2Vec2 backbone wrapper

- Uses Hugging Face `AutoModel.from_pretrained` only when a model is not
  dependency-injected.
- Returns a selected hidden state as `[B,T,D]` plus boolean `[B,T]` frame mask.
- Supports configurable `model_name`, `revision`, `hidden_layer`, and
  `freeze_backbone`.
- Validates hidden-layer range from model config when available and revalidates
  against real runtime hidden states.
- Defaults to frozen parameters and persistent eval mode.
- Does not pool, project to 64D, classify, compute SCAF, or perform KD.

### Mean DR baseline

- Computes a masked mean over time.
- Explicitly rejects an all-mask/zero-valid-frame sample.
- Projects `D → 64` with one linear layer.
- Rejects zero projected vectors and returns L2-normalized embeddings.

### Sub-center ArcFace (SCAF)

- Stores trainable centers with shape `[C,K,D]`.
- Normalizes embeddings and centers.
- Computes `[B,C,K]` cosine similarities and reduces only over the `K`
  subcenters within each class.
- Applies the angular margin only to the correct class.
- Uses cosine clamping, stable square root, and the monotonic ArcFace fallback.
- Returns cross-entropy loss and can return `[B,C]` logits for tests/metrics.
- SCAF is not referenced by Teacher inference.

### TeacherEncoder

- Composes a backbone and representation head only.
- Enforces float32 `[B,16000]`, finite, `[-1,1]` input windows.
- Returns the validated `TeacherOutput` contract.
- `encode()` is no-grad; deterministic behavior is verified in eval mode with a
  fake backbone.

### Window protocol

- Defines float32 `[B,16000]` waveform batches at 16 kHz.
- Enforces finite values in `[-1,1]` and non-empty unique `sample_id` values.
- Does not define recording, utterance, view, audio I/O, or manifest formats yet.

## 5. Explicitly NOT implemented

- WavLM, HuBERT, and Whisper Teacher
- Attention DR
- Knowledge distillation and `kd.py`
- BC-ResNet or any Student network
- Adapter/personalization algorithm
- Prototype or open-set evaluation implementation
- PTQ, QAT, INT8, TFLite, host, or ESP32 runtime
- Formal training/evaluation CLI
- Model download, feature cache, enrollment, or experiment artifacts
- Legacy PAPR package compatibility/imports

No files named `wavlm.py`, `hubert.py`, `attention_dr.py`, `kd.py`,
`bc_resnet.py`, `adapter.py`, or `quantization.py` exist.

## 6. Teacher contract

Input:

```text
waveform: torch.float32 [B,16000]
sample rate: 16000 Hz
finite and in [-1,1]
```

Output:

```text
TeacherOutput.embedding: torch.float32 [B,64]
TeacherOutput.frame_count: integer [B]
```

Every embedding row must be finite and L2 unit-normalized. Every frame count
must be positive. The output contract has no Wav2Vec2, SCAF, or DR dependency.

## 7. Test results

### Static compilation

```text
python -m compileall src tests
PASS
```

The final compile run completed without warnings.

### Offline unit tests

Executed with the existing `whisper_v0` Python environment, without installing
dependencies or using its Legacy project package:

| Area | Tests | Result |
|---|---:|---|
| Teacher contract | 1 | PASS |
| Window protocol | 2 | PASS |
| SCAF shape and correct-class margin | 2 | PASS |
| Wav2Vec2 config/injected fake model | 2 | PASS |
| Teacher composition/determinism | 1 | PASS |
| Mean DR masked mean/all-mask failure | 2 | PASS |
| **Total** | **10** | **10/10 PASS** |

`pyproject.toml` and `configs/teacher/wav2vec2_mean.yaml` were parsed by existing
local interpreters: **PASS**.

Static boundary audits:

- Import of Legacy `papr` package under `src/`, `tests/`, or configs: none.
- Absolute Windows runtime path or `os.getcwd()` under runtime/config/tests:
  none.
- Dataset, artifact, checkpoint, Hugging Face cache, log, and local-learning
  ignore rules: verified.

## 8. Dependency requirements

Proposed in `pyproject.toml`:

- Python `>=3.10`
- PyTorch `>=2.1`
- Transformers `>=4.40`
- PyYAML `>=6.0`

`unittest` is used from the standard library. These are dependency proposals,
not a lock file. No package was installed or upgraded.

## 9. Network/model-download status

- Network used only to read current official Transformers API documentation.
- Wav2Vec2 model/checkpoint downloads: **not attempted**.
- Hugging Face cache writes: **none**.
- Real Wav2Vec2 forward smoke test: **deferred integration test**.
- All unit tests use dependency-injected fake models/backbones.

## 10. Git status

An independent Git repository was initialized at the new project with branch
`main`. No file was staged, committed, pushed, or rebased.

Expected bootstrap status is untracked project content:

```text
?? .gitignore
?? README.md
?? artifacts/
?? configs/
?? datasets/
?? docs/
?? pyproject.toml
?? scripts/
?? src/
?? tests/
```

Local `.learnings/` and generated Python caches are ignored.

## 11. Known risks

1. The real configured checkpoint has not been downloaded or executed, so its
   actual output frame count/dimension and processor mask policy still need an
   integration gate.
2. Frame-mask conversion uses the Hugging Face Wav2Vec2 model helper
   `_get_feature_vector_attention_mask` for padded batches. It is a private API
   and must be rechecked when Transformers is pinned or upgraded.
3. `revision: main` is not immutable. After the first successful real smoke
   test, pin a verified model commit hash for reproducibility.
4. The dependency proposal is intentionally broad and has not been resolved
   into a lock file or isolated environment.
5. SCAF mathematical unit tests pass, but optimization behavior and E0/E0b
   qualification are untested until a real dataset/training protocol exists.
6. The in-memory Teacher contract is frozen; the cross-framework NPZ/JSON
   teacher-target serialization contract remains a later, separately tested
   deliverable.

## 12. Recommended next step

Proceed in this order:

```text
real Wav2Vec2 checkpoint smoke test
→ confirm processor/mask policy and real [B,T,D]
→ pin checkpoint revision and dependency versions
→ Teacher Mean DR training
→ E0 baseline qualification
```

Do not begin KD until the real Teacher Mean DR baseline and its 64D target
contract pass qualification.

Recommended initial commit message (not executed):

```text
feat: bootstrap PAPR-SSL Wav2Vec2 teacher baseline
```
