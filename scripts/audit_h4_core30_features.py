#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("artifacts/h4_core30_features"),
    )
    args = p.parse_args()

    manifest = json.loads(
        (args.feature_dir / "manifest.json").read_text(encoding="utf-8")
    )
    d = np.load(
        args.feature_dir / "global" / "core30_global256.npz",
        allow_pickle=False,
    )

    assert d["train_embedding"].shape == (3756, 256)
    assert d["dev_embedding"].shape == (442, 256)
    assert len(d["train_utt_id"]) == 3756
    assert len(d["dev_utt_id"]) == 442
    assert manifest["temporal"]["train_rows"] == 3756
    assert manifest["temporal"]["dev_rows"] == 442
    assert manifest["temporal"]["rows"] == 4198
    assert manifest["trainable_parameters_updated"] is False
    assert manifest["checkpoint_modified"] is False
    assert manifest["generic_test"] == "sealed_not_accessed"

    index_path = args.feature_dir / "temporal" / "index.jsonl"
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 4198

    missing = [
        r["utt_id"]
        for r in rows
        if not (args.feature_dir / "temporal" / r["cache_relpath"]).exists()
    ]
    if missing:
        raise RuntimeError(
            f"Missing temporal files: {len(missing)}; first={missing[:5]}"
        )

    repro = manifest["dev_reproduction_vs_existing_256d"]
    if repro["reference_present"]:
        assert repro["status"] == "PASS"

    print("=" * 108)
    print("H4-00B CORE30 FEATURE MATERIALIZATION AUDIT")
    print("=" * 108)
    print(f"global TRAIN:            {d['train_embedding'].shape}")
    print(f"global DEV:              {d['dev_embedding'].shape}")
    print(f"temporal TRAIN/DEV:      3756 / 442")
    if repro["reference_present"]:
        print(
            f"DEV reproduction:        PASS "
            f"(max_abs={repro['max_abs_diff']:.3e})"
        )
    else:
        print("DEV reproduction:        SKIPPED")
    print("trainable params updated:NO")
    print("generic_test accessed:   NO")
    print("AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
