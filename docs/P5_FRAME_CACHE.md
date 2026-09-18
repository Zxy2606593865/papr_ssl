# P5 prerequisite — frame-level cache for Mean vs Attention DR

P4 is closed.

Frozen source:

```text
microsoft/wavlm-large
revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
outputs.hidden_states[15]
D = 1024
```

P4 used masked-mean `[D]` caches. P5 Attention DR needs the temporal axis,
therefore P5 first materializes only the winner's frame-level `[T,D]` features.

No other Backbone is reopened.

## Cache format

Variable-length utterances are packed into shards:

```text
shard_00000.pt
  features: [sum_T, 1024] float32
  offsets:  [N+1]
  utt_ids:  N strings
```

with `index.jsonl` mapping every utterance to its frame range.

## Run

Use local HF cache only:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"

python scripts/materialize_p5_wavlm15_frame_cache.py `
  --device cuda `
  --batch-size 2
```

If CUDA memory is tight, `--batch-size 1` is allowed because it does not
change the feature definition.

Then:

```powershell
python scripts/audit_p5_wavlm15_frame_cache.py
```

PASS requires:

```text
train = 3756
dev = 442
total = 4198
hidden dim = 1024
feature dtype = float32
generic_test accessed = NO
```

After the frame cache passes, P5 can compare:

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

Both branches must use the same frozen frame cache, the same train/dev split,
the same SCAF settings, the same optimizer budget, and seeds 17/29/43.
