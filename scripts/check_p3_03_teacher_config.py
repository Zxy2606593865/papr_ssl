#!/usr/bin/env python
"""Validate the frozen P3-03 Teacher training-config contract."""

from __future__ import annotations

from pathlib import Path

from papr_ssl.training.teacher_training_config import (
    config_signature,
    load_teacher_training_config,
)


CONFIGS = (
    Path("configs/p3/teacher/wav2vec2_base.yaml"),
    Path("configs/p3/teacher/wavlm_large.yaml"),
    Path("configs/p3/teacher/w2v_bert2.yaml"),
)


def main() -> int:
    configs = [load_teacher_training_config(path) for path in CONFIGS]

    if len({config_signature(cfg) for cfg in configs}) != 1:
        raise RuntimeError(
            "P3-03 violation: backbone configs do not share one "
            "Mean-DR/SCAF contract."
        )

    print("=" * 96)
    print("PAPR-SSL P3-03 TEACHER TRAINING CONFIG")
    print("=" * 96)
    for cfg in configs:
        print(
            f"{cfg.backbone.name:16s} "
            f"layer={cfg.backbone.layer:2d}  "
            f"seed={cfg.seed:2d}  "
            f"frozen={str(cfg.backbone.frozen):5s}  "
            f"cache={str(cfg.backbone.use_offline_cache):5s}"
        )
        print(f"  model_id:       {cfg.backbone.model_id}")
        print(f"  revision:       {cfg.backbone.model_revision}")

    first = configs[0]
    print("-" * 96)
    print(f"head:             {first.head.kind}")
    print(f"embedding_dim:    {first.head.embedding_dim}")
    print(f"L2 normalize:     {first.head.l2_normalize}")
    print(f"SCAF K:           {first.scaf.k}")
    print(f"SCAF margin:      {first.scaf.margin}")
    print(f"SCAF scale:       {first.scaf.scale}")
    print(f"base seed:        {first.seed}")
    print("-" * 96)
    print("P3-03 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
