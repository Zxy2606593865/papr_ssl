# SSL Backbone Refactor Report

## 1. Baseline before change

The existing Bootstrap implementation contained Wav2Vec2, Mean DR, SCAF,
TeacherEncoder, Teacher/Window contracts, and offline tests. Before any source
change:

```text
python -m compileall src tests: PASS
offline unittest discovery: 10/10 PASS
```

The project was already an empty-history Git repository: no files were tracked
or committed, and all project content appeared as untracked top-level paths.

## 2. New three-backbone architecture

```text
same float32 waveform [B,16000]
                 │
       ┌─────────┼─────────────┐
       │         │             │
   Wav2Vec2    WavLM    W2v-BERT 2.0
       │         │             │
       └─────────┼─────────────┘
                 ↓
       SSLBackboneOutput [B,T,D]
                 ↓
          shared Mean DR
                 ↓
             float32 64D
                 ↓
            shared SCAF
```

The comparison boundary is common; internal frontends remain model-specific.
No training loop, evaluation implementation, or numerical result was created.

## 3. Common contract

`SSLBackboneOutput` is defined in
`src/papr_ssl/models/teacher/backbones/base.py` with:

```text
features: float32 [B,T,D], finite
frame_mask: bool [B,T], at least one valid frame per sample
hidden_layer: non-negative int
```

It does not require equal `T`, `D`, or frontend across backbones. A read-only
`mask` property preserves the Bootstrap output-access pattern while the common
name is now `frame_mask`.

## 4. Wav2Vec2 adaptation

The Wav2Vec2 wrapper retains `model_name`, `revision`, `hidden_layer`,
`freeze_backbone`, dependency injection, raw `input_values`, frozen eval
behavior, and runtime range validation. Its former output dataclass name remains
an alias to the common contract. Pooling, Mean DR, SCAF, classification, and KD
remain outside the wrapper.

## 5. WavLM implementation

`WavLMBackbone` is a separate raw-waveform wrapper with the same external
constructor controls and common output. A real model loads through
`WavLMModel.from_pretrained`; injected fake models keep all unit tests offline.
No claim is made that WavLM is better than Wav2Vec2.

## 6. W2v-BERT 2.0 implementation

`W2vBert2Backbone` supports paired injection of `model` and
`feature_extractor`. When neither is injected, the real integration path uses
`AutoFeatureExtractor` and `Wav2Vec2BertModel`. Supplying only one component
raises `ValueError`, preventing an accidental network fallback during an
offline test.

The feature extractor receives each 16 kHz waveform and must return
`input_features [B,T,F]` and `attention_mask [B,T]`. Both are validated and sent
to the model. The selected hidden state and mask must have the same frame axis.

## 7. W2v-BERT 2.0 frontend difference

Wav2Vec2 and WavLM receive raw waveform through `input_values`; their internal
convolutional frontend produces model frames. W2v-BERT 2.0 instead receives
feature-extractor acoustic `input_features`. This difference is preserved rather
than hidden behind a fabricated shared frontend. Fairness begins at the common
waveform and resumes at `SSLBackboneOutput`.

## 8. Hidden-layer indexing semantics

Across all three backbones, `hidden_layer` is the **exact zero-based index into
`outputs.hidden_states`**. There is no implicit `+1`/`-1`, encoder-layer alias,
clamp, or final-layer fallback.

Range is checked from `config.num_hidden_layers` when present and checked again
against the real runtime tuple. Invalid indices raise `ValueError`. Layer-sweep
candidates remain unverified proposals until T-Integration; a real invalid
candidate must be recorded as `INVALID_LAYER`.

## 9. Factory

`create_ssl_backbone(kind, **kwargs)` supports exactly:

- `wav2vec2`
- `wavlm`
- `w2v_bert2`

The factory uses an explicit constructor mapping. Unknown kinds raise
`ValueError`; there is no dynamic evaluation, plugin loading, or fallback.

## 10. Layer sweep

`LayerSweepCandidate` contains `backbone_kind`, project-relative `base_config`,
and exact `hidden_layer`. The expander reads YAML and creates combinations only;
it does not import/load a checkpoint, train, or access data. Duplicate
`(backbone_kind, hidden_layer)` pairs raise `ValueError`.

`configs/teacher/layer_sweep.yaml` expands to 12 candidates across all three
backbones. T-Integration must validate each candidate against the real model;
the expander intentionally cannot declare a layer valid.

## 11. Evaluation protocol

`docs/EVALUATION_PROTOCOL.md` freezes the responsibilities of
`generic_train`, `generic_enrollment`, `generic_dev_cal`, and
`generic_dev_score`. It prohibits test-set model, layer, checkpoint, threshold,
and hyperparameter selection.

`generic_test` remains sealed until the backbone, layer, DR, checkpoint,
threshold policy, and evaluation code are frozen. Its support/known/unknown
subsets and 1/5/10-shot rules are defined, together with within-domain,
cross-domain, and three-level evidence requirements. The metric contract covers
Known Macro F1, correct accept, wrong intent, known reject, unknown reject, FAR,
FRR, ACC/DET at 1% and 5% FAR, AUC, embedding diagnostics, and margins. No fake
metric values were produced.

## 12. Regression tests

The original 10 Bootstrap test methods still pass. Wav2Vec2 dependency
injection, range failure, Teacher composition, contracts, Window protocol,
Mean DR, and SCAF behavior remain covered.

## 13. New tests

Twelve test methods were added, bringing the suite to **22/22 PASS**:

- common `SSLBackboneOutput` contract;
- Wav2Vec2 common fields and compatibility mask;
- WavLM raw-waveform fake-model flow and invalid layer;
- W2v-BERT fake extractor → `input_features` → fake model flow;
- W2v-BERT missing extractor and invalid layer failures;
- all three factory kinds and unknown-kind failure;
- one shared MeanDR across three different feature dimensions;
- layer-sweep expansion and duplicate failure;
- all three Teacher configs share head, 64D, SCAF, and seed.

Final validation:

```text
python -m compileall src tests: PASS, no warning
offline unittest discovery: PASS, 22/22
Legacy papr import scan: no matches
runtime/config/test absolute-path scan: no matches
forbidden implementation-file scan: no matches
checkpoint/model artifact scan: no matches
```

## 14. Network/download status

Network access was used only to read current official Hugging Face documentation
for WavLM and Wav2Vec2-BERT APIs. No model, processor, feature extractor,
dataset, or checkpoint was downloaded.

The existing `whisper_v0` environment reports Transformers 5.13.0 and exposes
`WavLMModel`, `Wav2Vec2BertModel`, and `AutoFeatureExtractor`. This is an
availability observation, not a real-forward validation. Its existing
SciPy/NumPy compatibility warning remains unrelated and was not repaired by
changing dependencies.

## 15. Explicitly NOT implemented

- Attention DR or Layer Fusion
- Knowledge Distillation
- BC-ResNet or any Student
- Adapter/personalization algorithm
- Prototype runtime or metric calculation code
- PTQ, QAT, INT8, TFLite, host, or ESP32 code
- Formal training or checkpoint selection
- Real Wav2Vec2/WavLM/W2v-BERT forward
- Dataset access or final-test access
- Model/dependency downloads or upgrades

Mean DR and SCAF remain single shared implementations.

## 16. Risks

1. All three `revision: main` values are **UNPINNED DEVELOPMENT REVISION** and
   must become immutable after successful real smoke tests.
2. Real `[T,D]`, mask behavior, memory use, throughput, and tuple indexing remain
   unverified.
3. Wav2Vec2/WavLM padded-mask conversion relies on Hugging Face's private
   `_get_feature_vector_attention_mask` helper and requires integration checks
   for the pinned Transformers version.
4. W2v-BERT requires CPU feature extraction followed by transfer to the model
   device; integration must measure this cost and confirm exact mask alignment.
5. The dependency lower bounds are proposals, not an environment lock. The
   observed host environment is not proof of reproducibility.
6. Layer candidates are not valid merely because they expand successfully.
7. No evaluation implementation exists; the protocol is a contract only.

## 17. Git status

The repository still has no initial commit, so ordinary `git diff` and
`git diff --stat` are empty and cannot represent untracked source changes.
`git status --short` reports the expected untracked project roots:

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

The audit therefore also used `git ls-files --others --exclude-standard`, static
source scans, compileall, and the complete offline suite. A pre-existing,
untracked `docs/论文/` reference corpus was observed and left untouched; it is
not attributed to this refactor. Local `.learnings/` and Python caches remain
ignored. No add, commit, push, rebase, or Legacy-project Git command ran.

## 18. Recommended next phase

Proceed to **T-Integration**, not Attention DR:

1. Smoke-test the real Wav2Vec2 checkpoint.
2. Smoke-test the real WavLM checkpoint.
3. Smoke-test W2v-BERT 2.0 with its real feature extractor.
4. Verify processor/feature-extractor policies.
5. Record real `[T,D]` for every candidate.
6. Verify real frame masks and exact tuple indexing.
7. Measure peak memory and throughput.
8. Pin immutable model revisions.
9. Pin dependency versions and create an environment lock.
10. Only then train shared Mean DR + SCAF and select on `generic_dev_score`.

Recommended commit message (not executed):

```text
feat: add interchangeable SSL teacher backbones
```
