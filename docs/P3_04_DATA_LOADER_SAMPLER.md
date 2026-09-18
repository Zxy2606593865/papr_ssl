# P3-04 — Training Data Loader / Sampler

## Goal

Make the statistical composition of P3 training batches explicit and
reproducible.

The train loader must not silently inherit the long-tail frequency distribution
of the source dataset.

## Data path

P3-04 trains on the P3-02 offline feature cache:

```text
task-view index
      +
SSL cache identity
      ↓
OfflineFeatureTaskDataset
      ↓
cached masked-mean vector [D]
      +
train-defined integer class id
      ↓
ClassBalancedBatchSampler
```

No SSL backbone forward is executed here.

## Train class map

Class ids are defined **only from train**:

```text
sorted unique train task labels
    ↓
0 .. C-1
```

Dev must reuse the exact train class map.

A dev label absent from train is a protocol error.

## Train sampler

P3-04 uses explicit:

```text
C-way × K-shot
```

batch composition.

Default sampler contract:

```text
classes_per_batch = 8
samples_per_class = 4
batch_size = 32
seed = Teacher config seed
batches_per_epoch = ceil(train_size / 32)
```

The values are configurable, but every run must record them later in the P3-08
run manifest.

### Why class-balanced?

Suppose the raw train distribution is:

```text
A: 1000
B: 300
C: 80
```

Plain `shuffle=True` will expose class A to SCAF much more frequently than C.

The P3 sampler instead chooses classes independently of raw class frequency.
Class exposure over one epoch differs by at most one batch appearance.

Inside each selected class:

- if at least K samples exist, K distinct examples are drawn for that batch;
- if fewer than K exist, replacement is allowed explicitly.

This behavior is deterministic from:

```text
seed + epoch
```

Call:

```python
sampler.set_epoch(epoch)
```

before each training epoch.

## Dev loader

Dev is fundamentally different from train:

```text
shuffle = false
rebalancing = false
drop_last = false
full deterministic pass
```

Dev metrics must reflect the actual frozen dev set, not a balanced resample.

## Test protection

P3-04 deliberately refuses to construct `split="test"` datasets.

P3-07 still keeps generic test sealed. Test loading belongs only to the final
evaluation stage after model/checkpoint selection is frozen.

## Cache join integrity

The dataset joins task rows and cached features by:

```text
(dataset, utt_id)
```

and verifies:

- row exists in cache;
- split agrees;
- when the task index carries `sample_hash`, it must match the cache;
- cache identity is already validated by P3-02 reader.

The task label itself remains outside the audio-centric cache.

## Output batch contract

Train/dev batches contain:

```text
features      float32 [B,D]
labels        int64   [B]
task_labels   list[str]
datasets      list[str]
utt_ids       list[str]
splits        list[str]
sample_hashes list[str]
frame_counts  int64   [B]
```

P3-05 will consume only the cached `[B,D]` vectors plus labels for Mean DR +
SCAF training, while metadata is retained for auditability.

## P3-04 Gate

PASS requires:

1. exact C-way × K-shot train batches;
2. balanced class exposure despite imbalanced source counts;
3. deterministic sampling for the same seed/epoch;
4. `set_epoch()` changes the next epoch's draw;
5. train defines the class map and dev reuses it;
6. dev is a deterministic, complete, non-rebalanced pass;
7. cache/task split and optional sample-hash mismatches fail fast;
8. test construction is refused.
