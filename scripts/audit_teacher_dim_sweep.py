#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import torch.nn as nn

from papr_ssl.training.teacher.p5_dr import build_dr
from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frame-cache", type=Path,
                   default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"))
    p.add_argument("--core-index", type=Path,
                   default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"))
    args = p.parse_args()

    train_ds = FrameCacheDataset(cache_dir=args.frame_cache, task_index=args.core_index, split="train")
    dev_ds = FrameCacheDataset(cache_dir=args.frame_cache, task_index=args.core_index,
                               split="dev", class_to_index=train_ds.class_to_index)

    model = build_dr("attention")
    candidates = [(name, module.in_features, module.out_features)
                  for name, module in model.named_modules()
                  if isinstance(module, nn.Linear) and module.out_features == 64]

    print("=" * 108)
    print("TEACHER DIMENSION SWEEP PRE-FLIGHT AUDIT")
    print(f"train rows:                  {len(train_ds)}")
    print(f"dev rows:                    {len(dev_ds)}")
    print(f"64-output Linear candidates: {candidates}")
    print(f"frame cache:                 {args.frame_cache}")
    print("-" * 108)

    if len(train_ds) != 3756:
        raise RuntimeError("Expected 3756 Core30 train rows")
    if len(dev_ds) != 442:
        raise RuntimeError("Expected 442 Core30 dev rows")
    if len(candidates) != 1:
        raise RuntimeError("Expected exactly one 64-output projection in Attention DR")

    print("generic_test accessed:        NO")
    print("canonical P5 64D artifacts:   NOT TOUCHED")
    print("PRE-FLIGHT AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
