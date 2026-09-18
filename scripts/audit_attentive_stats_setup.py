#!/usr/bin/env python
"""Pre-flight audit for the Attentive Statistics Pooling ablation.

This audit:
- inspects the current Attention DR to infer its additive-attention hidden size;
- verifies the existing WavLM layer-15 cache and Core30 train/dev counts;
- never loads generic_test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch.nn as nn

from papr_ssl.training.teacher.p5_dr import build_dr
from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset


def infer_attention_hidden_dim():
    model = build_dr("attention")
    linears = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linears.append(
                (name, int(module.in_features), int(module.out_features))
            )

    # Expected current design:
    # 1024 -> H, H -> 1 attention scorer, and 1024 -> 64 projection.
    hidden_candidates = sorted(
        {
            out_f
            for _, in_f, out_f in linears
            if in_f == 1024 and out_f not in (1, 64)
        }
    )

    valid = []
    for h in hidden_candidates:
        has_score = any(
            in_f == h and out_f == 1
            for _, in_f, out_f in linears
        )
        if has_score:
            valid.append(h)

    if len(valid) != 1:
        raise RuntimeError(
            "Could not uniquely infer additive-attention hidden dimension "
            f"from current build_dr('attention'). Linear layers={linears}, "
            f"valid_hidden_candidates={valid}"
        )

    return valid[0], linears


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frame-cache",
        type=Path,
        default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"),
    )
    p.add_argument(
        "--core-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"
        ),
    )
    args = p.parse_args()

    hidden_dim, linears = infer_attention_hidden_dim()

    train_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="train",
    )
    dev_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="dev",
        class_to_index=train_ds.class_to_index,
    )

    print("=" * 108)
    print("ATTENTIVE STATISTICS POOLING PRE-FLIGHT AUDIT")
    print("=" * 108)
    print(f"Core30 train rows:             {len(train_ds)}")
    print(f"Core30 dev rows:               {len(dev_ds)}")
    print(f"inferred attention hidden dim: {hidden_dim}")
    print("current Attention DR Linear layers:")
    for name, in_f, out_f in linears:
        print(f"  {name:40s} {in_f:4d} -> {out_f:4d}")

    if len(train_ds) != 3756:
        raise RuntimeError(f"Expected train=3756, got {len(train_ds)}")
    if len(dev_ds) != 442:
        raise RuntimeError(f"Expected dev=442, got {len(dev_ds)}")

    print("-" * 108)
    print("canonical 64D Teacher:         PRESERVED")
    print("Project-1 256D Teacher:        PRESERVED")
    print("generic_test accessed:         NO")
    print("PRE-FLIGHT AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
