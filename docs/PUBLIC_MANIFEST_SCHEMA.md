# PAPR-SSL Public Audio Manifest Schema v1

**Phase:** P2-03
**Schema ID:** `papr_ssl.audio_manifest.v1`

## 1. Purpose

The manifest is the stable interface between dataset-specific raw layouts and
downstream PAPR-SSL code.

Raw audio is not copied into the manifest. Each JSONL row records where one
source audio item lives and the metadata required to use it reproducibly.

The schema is intentionally independent of GSC/MDSC directory layout.

## 2. Storage format

The canonical storage format is **JSON Lines (`.jsonl`)**:

```json
{"utt_id":"...","dataset":"gsc_v2","audio_relpath":"yes/xxx.wav",...}
{"utt_id":"...","dataset":"mdsc","audio_relpath":"Uncontrol/dev/eval/wav/...",...}
```

One line = one audio record.

Paths MUST be relative to the configured dataset root. Absolute Windows paths
must not be serialized into portable manifests.

## 3. Common fields

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `schema_version` | string | yes | `papr_ssl.audio_manifest.v1` |
| `utt_id` | string | yes | Globally stable utterance/record ID |
| `dataset` | enum | yes | `gsc_v2` or `mdsc` |
| `audio_relpath` | string | yes | Dataset-root-relative audio path |
| `speaker_id` | string/null | speech: yes | Source speaker identifier |
| `label` | string/null | conditional | Source class/phrase label |
| `transcript` | string/null | conditional | Source transcript text |
| `language` | enum | yes | `en` or `zh-CN` |
| `split` | enum | yes | `train`, `dev`, `test` |
| `domain` | enum | yes | `standard`, `control`, `dysarthria` |
| `role` | enum | yes | `train`, `enrollment`, `eval`, `none` |
| `record_type` | enum | yes | `speech` or `noise` |
| `sample_rate_hz` | int | yes | Raw file sample rate |
| `num_channels` | int | yes | Raw channel count |
| `num_frames` | int | yes | Raw frame count |
| `duration_sec` | float | yes | Raw duration in seconds |
| `source_meta` | object | yes | Dataset-specific metadata not promoted to the common schema |

For a speech record, at least one of `label` and `transcript` must be present.

## 4. GSC mapping

GSC uses:

```text
dataset      = gsc_v2
language     = en
domain       = standard
split        = official train/dev/test mapping
speaker_id   = filename prefix before _nohash_
label        = class directory name
transcript   = same keyword string
role         = train
record_type  = speech
```

Why is `role=train` used even for GSC validation/test?

`split` is the dataset partition. `role` is reserved for personalized
enrollment/evaluation semantics. GSC has no enrollment/eval hierarchy, so its
ordinary speech items use the neutral training-role value in schema v1.

The six `_background_noise_` WAV files are not ordinary KWS utterances. If
serialized, they use:

```text
record_type = noise
role        = none
speaker_id  = null
label       = null
```

They must not be counted as one of the 35 speech classes.

## 5. MDSC mapping

MDSC uses:

```text
dataset       = mdsc
language      = zh-CN
domain        = control | dysarthria
split         = official train/dev/test
role          = train | enrollment | eval
speaker_id    = source speaker ID, e.g. CF0010 / DF0014 / DM0002
label         = source phrase/keyword label
transcript    = source transcription
record_type   = speech
```

Mapping of top-level source folders:

```text
Control   -> domain=control
Uncontrol -> domain=dysarthria
```

The official enrollment/eval hierarchy must be preserved rather than
flattened.

The 14 stereo files discovered in P2-02 remain represented with
`num_channels=2` in the raw manifest. Mono conversion belongs to a later
preprocessing stage and must not rewrite raw source facts.

## 6. `label` vs `transcript`

These fields are deliberately separate.

`label` is the task/source class identity used for grouping or classification.

`transcript` is the textual content supplied by the dataset.

For GSC they are normally identical:

```text
label      = "backward"
transcript = "backward"
```

For MDSC they may also initially match, but the distinction is preserved so a
later benchmark can map multiple surface transcripts to one canonical intent
without destroying the original source text.

## 7. `split` vs `role`

They answer different questions.

```text
split:
Which official dataset partition owns this item?
train / dev / test

role:
What is this item used for inside a personalized protocol?
train / enrollment / eval / none
```

Example:

```text
MDSC Uncontrol/dev/enrollment/...
split  = dev
role   = enrollment
```

This distinction is required to preserve the original MDSC benchmark layout.

## 8. Raw facts vs derived preprocessing

The v1 manifest stores RAW source facts:

```text
sample_rate_hz
num_channels
num_frames
duration_sec
```

It does not claim that the model consumes the raw file unchanged.

Later preprocessing may perform:

```text
stereo -> mono
resample (if ever needed)
window/crop/pad
normalization
```

Those operations require a separate processed-sample contract and must not
overwrite the raw manifest fields.

## 9. Stable path policy

Do not store:

```text
D:\02_开发项目\...\speech_commands_v2\yes\xxx.wav
```

Store:

```text
yes/xxx.wav
```

The runtime resolves:

```text
dataset_root + audio_relpath
```

This keeps manifests portable across Windows workstations, AutoDL, Linux
servers, and future training environments.

## 10. P2-03 freeze decision

P2-03 freezes the common schema only.

P2-04 will implement:

```text
GSCAdapter
MDSCAdapter
        ↓
AudioManifestRecord
        ↓
*.jsonl
```

No model training, competition-specific optimization, new split creation, or
audio rewriting belongs to P2-03.
