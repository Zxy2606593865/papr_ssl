# PAPR-SSL

**PAPR-SSL is not an ASR system. PAPR-SSL is an interchangeable SSL Teacher
research framework.** It studies fixed-length embeddings for personalized
atypical speech, few-shot intent recognition, open-set rejection, and eventual
edge deployment.

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

Attention DR, Layer Fusion, KD, Student, Adapter, prototype runtime, PTQ/QAT,
TFLite, and ESP32 deployment are not implemented.

## Configuration status

The three Teacher configurations use the same head, output dimension, SCAF,
and seed. Only their backbone configuration differs. Their current
`revision: main` values are **UNPINNED DEVELOPMENT REVISION** values. T-Integration
must replace them with immutable revisions after real smoke tests succeed.

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

## Legacy reference

The local Legacy reference is:

```text
D:\02_开发项目\02_AI视觉与机器人\02_语音识别\whisper\papr
```

That path is documentation only. Runtime code and YAML do not contain it, and
PAPR-SSL never imports the Legacy `papr` package. Comparisons use independent
evaluation artifacts.
