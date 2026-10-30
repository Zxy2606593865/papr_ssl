# P5 前置条件 — Mean 与 Attention DR 的帧级 Cache

P4 已完成。

Frozen source:

```text
microsoft/wavlm-large
revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
outputs.hidden_states[15]
D = 1024
```

P4 使用 masked-mean `[D]` caches。P5 Attention DR 需要时序轴，因此 P5 首先只为获胜项物化帧级 `[T,D]` features。

不重新启用其他 Backbone。

## Cache 格式

Variable-length utterances are packed into shards:

```text
shard_00000.pt
  features: [sum_T, 1024] float32
  offsets:  [N+1]
  utt_ids:  N strings
```

使用 `index.jsonl` 将每条 utterance 映射到对应帧范围。

## 运行

仅使用本地 HF cache：

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"

python scripts/materialize_p5_wavlm15_frame_cache.py `
  --device cuda `
  --batch-size 2
```

如果 CUDA 显存紧张，可以使用 `--batch-size 1`，因为这不会改变特征定义。

随后运行：

```powershell
python scripts/audit_p5_wavlm15_frame_cache.py
```

通过时必须满足：

```text
train = 3756
dev = 442
total = 4198
hidden dim = 1024
feature dtype = float32
generic_test accessed = NO
```

帧级 cache 通过后，P5 可以比较：

```text
A. Mean DR
   masked mean over [T,1024]
   -> Linear(1024,64)
   -> L2

B. Attention DR
   masked temporal attention over [T,1024]
   -> 64D projection
   -> L2
```

两个分支必须使用相同的冻结帧级 cache、相同的 train/dev split、相同的 SCAF 设置和 optimizer budget，以及 seeds 17/29/43。
