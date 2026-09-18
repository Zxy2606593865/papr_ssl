#!/usr/bin/env python
"""Audit P5 frame-level cache integrity for frozen WavLM-large layer 15."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


EXPECTED_TOTAL = 4198
EXPECTED_TRAIN = 3756
EXPECTED_DEV = 442
EXPECTED_DIM = 1024
EXPECTED_LAYER = 15
EXPECTED_MODEL_ID = "microsoft/wavlm-large"
EXPECTED_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
SCHEMA = "papr_ssl.p5_frame_cache.v1"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "artifacts/p5_frame_cache/wavlm_large_layer15"
        ),
    )
    args = p.parse_args()

    manifest = json.loads(
        (args.cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["schema"] != SCHEMA:
        raise RuntimeError("frame cache schema mismatch")
    if manifest["model_id"] != EXPECTED_MODEL_ID:
        raise RuntimeError("unexpected model_id")
    if manifest["model_revision"] != EXPECTED_REVISION:
        raise RuntimeError("unexpected revision")
    if int(manifest["hidden_state_index"]) != EXPECTED_LAYER:
        raise RuntimeError("unexpected hidden_state_index")
    if int(manifest["hidden_dim"]) != EXPECTED_DIM:
        raise RuntimeError("unexpected hidden_dim")
    if manifest["feature_dtype"] != "float32":
        raise RuntimeError("P5 frame cache must be float32")
    if manifest["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    rows = [
        json.loads(line)
        for line in (args.cache_dir / "index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL} index rows, got {len(rows)}"
        )
    if len({row["utt_id"] for row in rows}) != EXPECTED_TOTAL:
        raise RuntimeError("duplicate utt_id in frame cache")

    split_counts = {
        "train": sum(row["split"] == "train" for row in rows),
        "dev": sum(row["split"] == "dev" for row in rows),
    }
    if split_counts != {
        "train": EXPECTED_TRAIN,
        "dev": EXPECTED_DEV,
    }:
        raise RuntimeError(f"split count mismatch: {split_counts}")
    if any(row["split"] not in ("train", "dev") for row in rows):
        raise RuntimeError("unexpected split found in P5 frame cache")

    by_shard: dict[str, list[dict]] = {}
    for row in rows:
        by_shard.setdefault(row["shard"], []).append(row)

    total_frames = 0
    for shard_name, shard_rows in sorted(by_shard.items()):
        shard = torch.load(
            args.cache_dir / shard_name,
            map_location="cpu",
            weights_only=False,
        )
        if shard["schema"] != SCHEMA:
            raise RuntimeError(f"{shard_name}: schema mismatch")
        feat = shard["features"]
        offsets = shard["offsets"]
        utt_ids = shard["utt_ids"]

        if feat.dtype != torch.float32:
            raise RuntimeError(f"{shard_name}: dtype is not float32")
        if feat.ndim != 2 or int(feat.shape[1]) != EXPECTED_DIM:
            raise RuntimeError(f"{shard_name}: invalid feature shape")
        if offsets.ndim != 1:
            raise RuntimeError(f"{shard_name}: invalid offsets")
        if len(utt_ids) + 1 != int(offsets.numel()):
            raise RuntimeError(f"{shard_name}: offsets/utt_ids mismatch")
        if int(offsets[0]) != 0:
            raise RuntimeError(f"{shard_name}: first offset != 0")
        if int(offsets[-1]) != int(feat.shape[0]):
            raise RuntimeError(f"{shard_name}: final offset mismatch")
        if not torch.isfinite(feat).all():
            raise RuntimeError(f"{shard_name}: non-finite feature")

        lookup = {row["utt_id"]: row for row in shard_rows}
        for i, utt_id in enumerate(utt_ids):
            row = lookup[utt_id]
            start = int(offsets[i])
            end = int(offsets[i + 1])
            if end <= start:
                raise RuntimeError(f"{shard_name}: zero-length sequence")
            if row["start_frame"] != start or row["end_frame"] != end:
                raise RuntimeError(f"{shard_name}: index bounds mismatch")
            if row["frame_count"] != end - start:
                raise RuntimeError(f"{shard_name}: frame_count mismatch")

        total_frames += int(feat.shape[0])

    if total_frames != int(manifest["total_frame_count"]):
        raise RuntimeError("total frame count mismatch")

    cache_bytes = sum(
        p.stat().st_size
        for p in args.cache_dir.rglob("*")
        if p.is_file()
    )

    print("=" * 108)
    print("P5 WAVLM-LARGE hidden_states[15] FRAME CACHE AUDIT")
    print("=" * 108)
    print(f"train utterances:      {split_counts['train']}")
    print(f"dev utterances:        {split_counts['dev']}")
    print(f"total utterances:      {len(rows)}")
    print(f"total frames:          {total_frames}")
    print(f"hidden dim:            {EXPECTED_DIM}")
    print("feature dtype:         float32")
    print(f"cache size:            {cache_bytes / (1024**3):.3f} GiB")
    print("generic_test accessed: NO")
    print("-" * 108)
    print("P5 FRAME CACHE STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
