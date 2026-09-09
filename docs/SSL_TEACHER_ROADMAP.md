# SSL Teacher Roadmap

| Stage | Scope | Gate |
|---|---|---|
| T0 | Wav2Vec2 + Mean DR | Offline contract and unit tests pass. |
| T1 | WavLM + Mean DR | Offline wrapper/factory tests pass; real integration deferred. |
| T2 | W2v-BERT 2.0 + Mean DR | Feature-extractor → `input_features` flow passes offline tests. |
| T3 | Layer sweep per backbone | Candidate expansion is unique; real indices validated without fallback. |
| T4 | Select backbone + layer on `generic_dev_score` | Same split, budget, head, SCAF, and metrics. |
| T5 | Mean DR vs Attention DR | Begin only after T4 is frozen. |
| T6 | Best single layer vs Layer Fusion | Begin only after T5 is frozen. |
| T7 | E0 Teacher qualification | All development gates and reports pass. |
| T8 | Freeze Teacher | Pin code, model revision, dependency lock, checkpoint, and protocol. |
| T9 | Open `generic_test` | First permitted access to final generic test. |
| T10 | Cross-domain evaluation | Frozen Teacher and threshold policy; no Teacher retraining. |
| T11 | Teacher target export | Versioned/hash-verified cross-framework target contract. |
| T12 | KD Student | Start only after Teacher targets and qualification are frozen. |

`generic_test` is sealed before T9. It must not influence stages T0–T8, including
backbone, hidden-layer, checkpoint, threshold, Attention DR, or Layer Fusion
selection.

The immediate next phase is **T-Integration**, not Attention DR: run real
Wav2Vec2, WavLM, and W2v-BERT 2.0 smoke tests; verify processors/frontends,
actual `[T,D]`, masks, exact hidden-state indexing, memory, and throughput; then
pin model revisions and dependency versions.
