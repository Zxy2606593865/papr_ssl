# H4-00B — Core30 Frozen Feature Materialization

## 当前在干什么？

H4-00 已经确认：

```text
WavLM[15] frame cache:
TRAIN 3756 / 3756
DEV    442 / 442
```

但 H4 要在 34 个 TRAIN speaker 上学习一个共享 Task Head，而我们之前保存的
256D global embedding 主要服务于 DEV / 15-shot 实验，没有完整保存 3756 条 TRAIN embedding。

所以现在不是训练新模型，而是把已经确定的 frozen Teacher 对 TRAIN/DEV
全部样本“跑一遍现成特征头”，把 H4 需要的输入材料准备齐。

可以理解为：

```text
已有 WavLM[15] 缓存 [T,1024]
        ↓
已经训练好的 Attention + 256D Projection
        ├─ Global 256D
        └─ framewise 256D → downsample×3 → Temporal sequence
```

这里所有参数都冻结。

## 为什么必须做？

H4 的正确实验关系应该是：

```text
34 TRAIN speakers
→ 产生候选证据
→ 拟合共享 evidence / decision Head

4 unseen DEV speakers
→ 完全不参与 Head fitting
→ 只做开发评价
```

没有 TRAIN 的 256D global embedding，就无法严格做这个实验。

## 一个审计修正

H4-00 的自动搜索把：

```text
within_speaker_dtw.npz
```

列成了“Best global embedding artifact”。

它只是因为包含 `utt_id` 且覆盖 442 个 DEV utterance，被通用搜索逻辑选中了；
它本身不是 global embedding 文件。

这不影响“H4 TRAIN global 缺失”这个主要结论，但 H4-00B 不再依赖模糊自动发现，
而是直接生成一个结构明确的正式 artifact。

## 输出

```text
artifacts/h4_core30_features/
├─ global/
│  └─ core30_global256.npz
├─ temporal/
│  ├─ index.jsonl
│  └─ utterances/*.pt
└─ manifest.json
```

Global NPZ 包含：

```text
TRAIN:
  utt_id
  speaker_id
  label
  label_index
  embedding [3756,256]

DEV:
  utt_id
  speaker_id
  label
  label_index
  embedding [442,256]
```

Temporal 共 4198 条。

## 为什么还要做 DEV reproduction？

我们已经有以前的 442 条 DEV 256D embedding。

新导出后要求：

```text
新 DEV embedding ≈ 历史 DEV embedding
```

如果一致，说明：

> 我们只是补齐 TRAIN 特征，没有偷偷改变 Teacher / Neck。

这是非常重要的复现保护。

## 运行

```powershell
python scripts/prepare_h4_core30_features.py
python scripts/audit_h4_core30_features.py
```

如果 Audit PASS，下一步才正式进入：

```text
H4-01 Shared Evidence Head
```

届时第一次真正使用：

```text
34 TRAIN speakers → fit
4 unseen DEV speakers → evaluate
```
