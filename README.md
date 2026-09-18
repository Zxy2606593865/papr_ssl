# PAPR-SSL

**PAPR-SSL is not a general ASR system.** It is the active service-side project
for personalized atypical-phrase understanding, built on a frozen SSL Teacher
research line and a personalized runtime.

The workspace has two separate development lines:

- **Project 1 — PAPR-SSL + Agent**: this repository owns the frozen Teacher,
  Enrollment Memory, Prototype/Top-3 DTW, C/W/U decision, three-state runtime,
  `intent_id → canonical_text`, Web Demo and browser TTS. Backend Orchestrator
  and Agent Adapter are the next phase.
- **Project 2 — PAPR-Edge-Lite**: no dedicated in-scope repository currently
  exists in this workspace. Teacher target export, Student/KD, PCEN/LogMel,
  BC-ResNet and edge deployment are not implemented yet. The Legacy
  `..\whisper\papr` repository is explicitly excluded from this cleanup.

Start with:

- [Current development status](docs/DEVELOPMENT_STATUS.md)
- [Dual-project map](docs/PAPR_PROJECT_MAP.md)
- [Repository audit](docs/PAPR_REPOSITORY_AUDIT.md)

## Current Teacher candidates

```text
same float32 waveform [B,16000]
                 │
       ┌─────────┼─────────────┐
       │         │             │
   Wav2Vec2    WavLM    W2v-BERT 2.0
       │         │             │
       └─────────┼─────────────┘
                 ↓
 exact outputs.hidden_states index
                 ↓
          shared Mean DR
                 ↓
      unit-normalized 64D
                 ↓
            shared SCAF
```

The fair-comparison boundary is the common waveform input and
`SSLBackboneOutput`; internal model frontends are not forced to be identical.
Wav2Vec2 and WavLM consume raw `input_values`. W2v-BERT 2.0 first converts the
waveform to model-specific acoustic `input_features` with its feature extractor.

## Hidden-layer indexing

`hidden_layer` is always the exact zero-based index into
`outputs.hidden_states`. Index `0` is normally the initial embedding output and
indices `1..N` are successive encoder outputs for the current Hugging Face
models. PAPR-SSL never applies an implicit `+1` or `-1`, never clamps an invalid
index, and never falls back to the final layer. Real model depth and indexing
must be verified in T-Integration.

## Current implementation

- Wav2Vec2, WavLM, and W2v-BERT 2.0 wrappers with offline dependency injection
- Shared `SSLBackboneOutput`: float32 `[B,T,D]`, bool `[B,T]`, exact layer index
- Shared masked Mean DR with configurable input `D` and fixed 64D output
- Shared Sub-center ArcFace configuration: `K=3`, margin `0.2`, scale `30`
- Teacher and fixed-window contracts
- Offline layer-sweep expansion
- Frozen Attention-DR runtime path with Enrollment Memory and Prototype search
- Top-3 DTW evidence, C/W/U decision, and ACCEPT / CONFIRM / REJECT output
- `intent_id → canonical_text` mapping
- Raw-WAV adapter, Wanghao real-data Demo, Web backend/frontend and browser TTS

Backend Orchestrator, Agent Adapter/Coze integration, Student/KD, PCEN/LogMel,
BC-ResNet, PTQ/QAT, TFLite, and ESP32 deployment are not implemented.

## Configuration status

The three Teacher configurations use the same head, output dimension, SCAF,
and seed. Only their backbone configuration differs. Their Hugging Face
revisions are pinned to immutable commit hashes.

## Evaluation rule

**The test set is not used for model selection.** Backbone, hidden layer,
checkpoint, DR choice, thresholds, and hyperparameters are selected only with
the development protocol in [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).
`generic_test` remains sealed until roadmap stage T9.

## Offline development

```powershell
python -m compileall src tests
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Tests use fake models and a fake W2v-BERT feature extractor. They do not install
packages, load datasets, download checkpoints, or train models.

## Repository boundaries

The local Legacy repository is:

```text
D:\02_开发项目\02_AI视觉与机器人\02_语音识别\whisper\papr
```

It is excluded from this cleanup and is not the Project-2 development entry.
The shared audio data project is `..\papr_audio_toolkit`. PAPR-SSL does not
import sibling repositories' internal Python modules; projects exchange
versioned files and manifests.
