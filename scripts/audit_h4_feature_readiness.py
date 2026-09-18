#!/usr/bin/env python
"""H4-00 — Shared Evidence Head feature readiness audit.

Why this exists
---------------
H2-00 showed that the official full Core30 unseen-speaker benchmark is truly
1/2-shot. User×phrase variance is not scientifically estimable at n=1/2, and
temporal medoid selection is also not meaningful with only 1-2 enrollment
examples.

Therefore the public-data mainline moves to H4:
    shared low-capacity evidence fusion / C-W-U decision.

Before fitting H4 on TRAIN speakers, we must verify that frozen 256D global
embeddings and temporal features exist for ALL Core30 TRAIN/DEV utterances.

This script only audits artifacts. It does not train, does not load TEST audio,
and does not touch generic_test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


def normalize_phrase(text):
    if text is None:
        return ""
    x = str(text).replace("<p>", "")
    x = re.sub(r"\s+", "", x)
    return x.casefold()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get(row, keys, default=""):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return default


def row_split(row):
    return str(get(row, ("split",))).strip().lower()


def row_utt(row):
    v = get(row, ("utt_id", "id", "audio_id"))
    if v == "":
        raise KeyError(f"No utt id in keys={sorted(row)}")
    return str(v)


def row_phrase(row):
    return normalize_phrase(
        get(row, ("task_label", "label_text", "label", "transcript", "standard_text", "text"))
    )


def collect_core30_ids(index_path: Path):
    ids = {"train": set(), "dev": set()}
    phrases = set()
    for row in read_jsonl(index_path):
        sp = row_split(row)
        if sp == "test":
            continue
        if sp not in ids:
            continue
        uid = row_utt(row)
        ids[sp].add(uid)
        ph = row_phrase(row)
        if ph:
            phrases.add(ph)
    if len(phrases) != 30:
        raise RuntimeError(f"Expected 30 Core30 phrases, got {len(phrases)}")
    return ids


def discover_npz(root: Path):
    return sorted(root.rglob("*.npz"))


def inspect_npz(path: Path):
    info = {
        "path": str(path),
        "keys": [],
        "utt_id_keys": [],
        "row_counts": {},
        "matched_train": 0,
        "matched_dev": 0,
        "error": None,
    }
    try:
        d = np.load(path, allow_pickle=False)
        info["keys"] = list(d.files)
        for key in d.files:
            arr = d[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                info["row_counts"][key] = int(arr.shape[0])

        # Common id-array names used by this project.
        for key in d.files:
            lk = key.lower()
            if "utt" in lk and "id" in lk:
                arr = d[key]
                if arr.ndim == 1:
                    try:
                        ids = set(arr.astype(str).tolist())
                    except Exception:
                        continue
                    info["utt_id_keys"].append(key)
                    yield key, ids, info
    except Exception as e:
        info["error"] = repr(e)
    yield None, set(), info


def inspect_temporal_indices(artifacts_root: Path):
    results = []
    for path in sorted(artifacts_root.rglob("index.jsonl")):
        ids = set()
        count = 0
        has_cache_path = False
        try:
            for row in read_jsonl(path):
                count += 1
                try:
                    ids.add(row_utt(row))
                except Exception:
                    pass
                if any(k in row for k in ("cache_relpath", "path", "feature_path")):
                    has_cache_path = True
            results.append({
                "path": str(path),
                "rows": count,
                "unique_utt_ids": len(ids),
                "has_cache_path_field": has_cache_path,
                "ids": ids,
            })
        except Exception as e:
            results.append({
                "path": str(path),
                "rows": 0,
                "unique_utt_ids": 0,
                "has_cache_path_field": False,
                "ids": set(),
                "error": repr(e),
            })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--core30-index",
        type=Path,
        default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"),
    )
    p.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h4_00_feature_readiness"),
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    core_ids = collect_core30_ids(args.core30_index)
    train_ids = core_ids["train"]
    dev_ids = core_ids["dev"]

    npz_rows = []
    best_global = None

    for path in discover_npz(args.artifacts_root):
        last_info = None
        emitted = False
        for id_key, ids, info in inspect_npz(path):
            last_info = info
            if id_key is None:
                continue
            emitted = True
            train_match = len(ids & train_ids)
            dev_match = len(ids & dev_ids)
            row = {
                "path": str(path),
                "utt_id_key": id_key,
                "unique_ids": len(ids),
                "train_match": train_match,
                "train_total": len(train_ids),
                "train_coverage": train_match / len(train_ids) if train_ids else 0.0,
                "dev_match": dev_match,
                "dev_total": len(dev_ids),
                "dev_coverage": dev_match / len(dev_ids) if dev_ids else 0.0,
                "npz_keys": info["keys"],
            }
            npz_rows.append(row)
            score = (train_match + dev_match, train_match, dev_match)
            if best_global is None or score > best_global[0]:
                best_global = (score, row)

        if not emitted and last_info is not None and last_info["error"]:
            npz_rows.append({
                "path": str(path),
                "utt_id_key": None,
                "unique_ids": 0,
                "train_match": 0,
                "train_total": len(train_ids),
                "train_coverage": 0.0,
                "dev_match": 0,
                "dev_total": len(dev_ids),
                "dev_coverage": 0.0,
                "npz_keys": last_info["keys"],
                "error": last_info["error"],
            })

    temporal_rows = []
    best_temporal = None
    for info in inspect_temporal_indices(args.artifacts_root):
        ids = info.pop("ids")
        train_match = len(ids & train_ids)
        dev_match = len(ids & dev_ids)
        row = {
            **info,
            "train_match": train_match,
            "train_total": len(train_ids),
            "train_coverage": train_match / len(train_ids) if train_ids else 0.0,
            "dev_match": dev_match,
            "dev_total": len(dev_ids),
            "dev_coverage": dev_match / len(dev_ids) if dev_ids else 0.0,
        }
        temporal_rows.append(row)
        score = (train_match + dev_match, train_match, dev_match)
        if best_temporal is None or score > best_temporal[0]:
            best_temporal = (score, row)

    best_global_row = best_global[1] if best_global else None
    best_temporal_row = best_temporal[1] if best_temporal else None

    global_train_ready = bool(
        best_global_row and best_global_row["train_coverage"] >= 0.999999
    )
    global_dev_ready = bool(
        best_global_row and best_global_row["dev_coverage"] >= 0.999999
    )
    temporal_train_ready = bool(
        best_temporal_row and best_temporal_row["train_coverage"] >= 0.999999
    )
    temporal_dev_ready = bool(
        best_temporal_row and best_temporal_row["dev_coverage"] >= 0.999999
    )

    verdict = {
        "global_train_ready": global_train_ready,
        "global_dev_ready": global_dev_ready,
        "temporal_train_ready": temporal_train_ready,
        "temporal_dev_ready": temporal_dev_ready,
        "ready_for_h4_full_global_temporal_fit": (
            global_train_ready
            and global_dev_ready
            and temporal_train_ready
            and temporal_dev_ready
        ),
    }

    out = {
        "schema": "papr_ssl.h4_00_feature_readiness.v1",
        "core30": {
            "train_utt_ids": len(train_ids),
            "dev_utt_ids": len(dev_ids),
        },
        "best_global_artifact": best_global_row,
        "best_temporal_index": best_temporal_row,
        "verdict": verdict,
        "all_npz_candidates": npz_rows,
        "all_temporal_index_candidates": temporal_rows,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 104)
    print("H4-00 SHARED EVIDENCE HEAD FEATURE READINESS AUDIT")
    print("=" * 104)
    print(f"Core30 TRAIN utts:          {len(train_ids)}")
    print(f"Core30 DEV utts:            {len(dev_ids)}")
    print("-" * 104)

    if best_global_row:
        print("Best global embedding artifact:")
        print(f"  {best_global_row['path']}")
        print(
            f"  TRAIN coverage: {best_global_row['train_match']}/{len(train_ids)} "
            f"({100*best_global_row['train_coverage']:.2f}%)"
        )
        print(
            f"  DEV coverage:   {best_global_row['dev_match']}/{len(dev_ids)} "
            f"({100*best_global_row['dev_coverage']:.2f}%)"
        )
    else:
        print("Best global embedding artifact: NONE FOUND")

    print("-" * 104)
    if best_temporal_row:
        print("Best temporal feature index:")
        print(f"  {best_temporal_row['path']}")
        print(
            f"  TRAIN coverage: {best_temporal_row['train_match']}/{len(train_ids)} "
            f"({100*best_temporal_row['train_coverage']:.2f}%)"
        )
        print(
            f"  DEV coverage:   {best_temporal_row['dev_match']}/{len(dev_ids)} "
            f"({100*best_temporal_row['dev_coverage']:.2f}%)"
        )
    else:
        print("Best temporal feature index: NONE FOUND")

    print("-" * 104)
    print(f"global TRAIN ready:         {'YES' if global_train_ready else 'NO'}")
    print(f"global DEV ready:           {'YES' if global_dev_ready else 'NO'}")
    print(f"temporal TRAIN ready:       {'YES' if temporal_train_ready else 'NO'}")
    print(f"temporal DEV ready:         {'YES' if temporal_dev_ready else 'NO'}")
    print(
        "H4 full feature fit ready:  "
        + ("YES" if verdict["ready_for_h4_full_global_temporal_fit"] else "NO")
    )
    print("generic_test accessed:      NO")
    print("H4-00 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
