#!/usr/bin/env python
"""Verify the real Core30 task-view -> P3 feature-cache join before training."""

from __future__ import annotations

import argparse
from pathlib import Path

from papr_ssl.cache.ssl_feature_cache import SSLCacheIdentity
from papr_ssl.training.feature_data import OfflineFeatureTaskDataset


MODEL_ID = "facebook/wav2vec2-base"
REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
LAYER = 12


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"
        ),
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/ssl_feature_cache"),
    )
    args = p.parse_args()

    identity = SSLCacheIdentity(
        model_id=MODEL_ID,
        model_revision=REVISION,
        layer=LAYER,
    )
    cache_dir = args.cache_root / identity.slug

    train = OfflineFeatureTaskDataset(
        task_index=args.task_index,
        cache_dir=cache_dir,
        cache_identity=identity,
        split="train",
    )
    dev = OfflineFeatureTaskDataset(
        task_index=args.task_index,
        cache_dir=cache_dir,
        cache_identity=identity,
        split="dev",
        class_to_index=train.class_to_index,
    )

    # Force one real shard read from each split.
    train0 = train[0]
    dev0 = dev[0]

    assert len(train) == 3756, len(train)
    assert len(dev) == 442, len(dev)
    assert len(train.class_to_index) == 30, len(train.class_to_index)
    assert train.reader.hidden_dim == 768, train.reader.hidden_dim
    assert tuple(train0["features"].shape) == (768,)
    assert tuple(dev0["features"].shape) == (768,)
    assert train0["dataset"] == "mdsc"
    assert dev0["dataset"] == "mdsc"

    print("=" * 100)
    print("PAPR-SSL REAL CORE30 TASK-VIEW / FEATURE-CACHE JOIN")
    print("=" * 100)
    print(f"task index:                {args.task_index}")
    print(f"cache dir:                 {cache_dir}")
    print(f"train rows joined:         {len(train)}")
    print(f"dev rows joined:           {len(dev)}")
    print(f"classes:                   {len(train.class_to_index)}")
    print(f"cached feature dim:        {train.reader.hidden_dim}")
    print(f"train sample dataset:      {train0['dataset']}")
    print(f"train sample utt_id:       {train0['utt_id']}")
    print(f"train feature shape:       {tuple(train0['features'].shape)}")
    print(f"dev feature shape:         {tuple(dev0['features'].shape)}")
    print("test dataset constructed:  NO")
    print("-" * 100)
    print("REAL CORE30 CACHE JOIN: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
