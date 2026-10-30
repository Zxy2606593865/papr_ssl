# PAPR-SSL 公共音频 Manifest Schema v1

**Phase:** P2-03
**Schema ID:** `papr_ssl.audio_manifest.v1`

## 1. 用途

manifest 是数据集专属原始目录结构与下游 PAPR-SSL 代码之间的稳定接口。

manifest 不复制原始音频。每行 JSONL 记录一条源音频的位置，以及可复现使用所需的元数据。

此 schema 刻意与 GSC/MDSC 目录结构解耦。

## 2. 存储格式

The canonical storage format is **JSON Lines (`.jsonl`)**:

```json
{"utt_id":"...","dataset":"gsc_v2","audio_relpath":"yes/xxx.wav",...}
{"utt_id":"...","dataset":"mdsc","audio_relpath":"Uncontrol/dev/eval/wav/...",...}
```

一行对应一条音频记录。

路径必须相对于配置的数据集根目录。不得将 Windows 绝对路径序列化到可移植的 manifest 中。

## 3. 通用字段

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `schema_version` | string | yes | `papr_ssl.audio_manifest.v1` |
| `utt_id` | string | yes | 全局稳定的 utterance/record ID |
| `dataset` | enum | yes | `gsc_v2` or `mdsc` |
| `audio_relpath` | string | yes | 相对于数据集根目录的音频路径 |
| `speaker_id` | string/null | speech: yes | 源说话人标识符 |
| `label` | string/null | conditional | 源类别/短语标签 |
| `transcript` | string/null | conditional | 源转写文本 |
| `language` | enum | yes | `en` or `zh-CN` |
| `split` | enum | yes | `train`, `dev`, `test` |
| `domain` | enum | yes | `standard`, `control`, `dysarthria` |
| `role` | enum | yes | `train`、`enrollment`、`eval`、`none` |
| `record_type` | enum | yes | `speech` 或 `noise` |
| `sample_rate_hz` | int | yes | 原始文件采样率 |
| `num_channels` | int | yes | 原始声道数 |
| `num_frames` | int | yes | 原始帧数 |
| `duration_sec` | float | yes | 原始时长（秒） |
| `source_meta` | object | yes | 尚未纳入通用 schema 的数据集专属元数据 |

对于 speech 记录，`label` 和 `transcript` 至少要提供一个。

## 4. GSC 映射

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

为什么 GSC validation/test 也使用 `role=train`？

`split` 表示数据集分区。`role` 专用于个性化 enrollment/evaluation 语义。GSC 没有 enrollment/eval 层级，因此在 schema v1 中，普通语音样本使用中性的训练角色值。

六个 `_background_noise_` WAV 文件不是普通 KWS utterance。若将其序列化，应使用：

```text
record_type = noise
role        = none
speaker_id  = null
label       = null
```

不得将它们计入 35 个语音类别。

## 5. MDSC 映射

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

必须保留官方 enrollment/eval 层级，不得将其扁平化。

P2-02 中发现的 14 个 stereo 文件，在 raw manifest 中仍以 `num_channels=2` 表示。Mono 转换属于后续 preprocessing 阶段，不得改写原始来源信息。

## 6. `label` 与 `transcript`

这两个字段有意分开。

`label` 是用于分组或分类的任务/源类别标识。

`transcript` 是数据集提供的文本内容。

对于 GSC，两者通常相同：

```text
label      = "backward"
transcript = "backward"
```

对于 MDSC，两者起初也可能相同；保留这一区分后，后续基准可以将多个表层转写映射到一个规范意图，同时不破坏原始文本。

## 7. `split` 与 `role`

两者回答不同的问题。

```text
split:
此条目属于哪个官方数据集分区？
train / dev / test

role:
在个性化协议中，此条目用于什么用途？
train / enrollment / eval / none
```

Example:

```text
MDSC Uncontrol/dev/enrollment/...
split  = dev
role   = enrollment
```

必须保留这一区分，才能保留 MDSC 原始基准布局。

## 8. 原始信息与派生预处理

v1 manifest 存储原始来源信息：

```text
sample_rate_hz
num_channels
num_frames
duration_sec
```

它不表示模型会未经处理直接读取原始文件。

Later preprocessing may perform:

```text
stereo -> mono
resample (if ever needed)
window/crop/pad
normalization
```

这些操作需要独立的 processed-sample contract，且不得覆盖 raw manifest 字段。

## 9. 稳定路径策略

Do not store:

```text
D:\02_开发项目\...\speech_commands_v2\yes\xxx.wav
```

Store:

```text
yes/xxx.wav
```

runtime 会解析：

```text
dataset_root + audio_relpath
```

这样可以让 manifest 在 Windows 工作站、AutoDL、Linux server 和未来训练环境之间保持可移植。

## 10. P2-03 冻结决定

P2-03 仅冻结通用 schema。

P2-04 将实现：

```text
GSCAdapter
MDSCAdapter
        ↓
AudioManifestRecord
        ↓
*.jsonl
```

P2-03 不涉及模型训练、竞赛专属优化、新建 split 或改写音频。
