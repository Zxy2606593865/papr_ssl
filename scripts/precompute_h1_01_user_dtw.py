#!/usr/bin/env python
"""H1-01 prerequisite: precompute within-speaker DTW distances for Core30 DEV.

Why:
- H1-01 repeatedly changes which utterances are support/query.
- The temporal sequences themselves never change.
- Precomputing pairwise DTW once makes the 20 repeated episodes cheap.

Scope:
- Core30 DEV only: 442 utterances, 4 unseen speakers.
- Only within-speaker pairs are computed.
- No TEST data.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


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


def row_utt(row):
    v = get(row, ("utt_id", "id", "audio_id"))
    if v == "":
        raise KeyError(f"No utt id field in {sorted(row)}")
    return str(v)


def row_speaker(row):
    v = get(row, ("speaker_id", "speaker", "user_id", "subject_id"))
    if v == "":
        raise KeyError(f"No speaker field in {sorted(row)}")
    return str(v)


def load_temporal_index(cache_dir: Path):
    out = {}
    with (cache_dir / "index.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["utt_id"]] = cache_dir / row["cache_relpath"]
    return out


def l2norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_seq(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    x = obj["features"].to(torch.float32).numpy()
    return l2norm(x)


def dtw_cosine_distance(a, b, band_ratio=0.25):
    a = l2norm(a)
    b = l2norm(b)
    n, m = len(a), len(b)
    band = max(abs(n - m), int(math.ceil(band_ratio * max(n, m))))

    inf = np.inf
    prev_cost = np.full(m + 1, inf, dtype=np.float64)
    prev_len = np.zeros(m + 1, dtype=np.int32)
    prev_cost[0] = 0.0

    for i in range(1, n + 1):
        cur_cost = np.full(m + 1, inf, dtype=np.float64)
        cur_len = np.zeros(m + 1, dtype=np.int32)
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        ai = a[i - 1]

        for j in range(j0, j1 + 1):
            local = 1.0 - float(np.clip(np.dot(ai, b[j - 1]), -1.0, 1.0))
            candidates = (
                (prev_cost[j], prev_len[j]),
                (cur_cost[j - 1], cur_len[j - 1]),
                (prev_cost[j - 1], prev_len[j - 1]),
            )
            best_cost, best_len = min(candidates, key=lambda x: x[0])
            if not np.isfinite(best_cost):
                continue
            cur_cost[j] = best_cost + local
            cur_len[j] = best_len + 1

        prev_cost, prev_len = cur_cost, cur_len

    if not np.isfinite(prev_cost[m]) or prev_len[m] <= 0:
        raise RuntimeError(f"No DTW path for lengths n={n}, m={m}")

    return float(prev_cost[m] / prev_len[m])


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embedding-npz",
        type=Path,
        default=Path(
            "artifacts/p6_teacher_256_15shot/embeddings/"
            "teacher_256d_15shot_embeddings.npz"
        ),
    )
    p.add_argument(
        "--mdsc-manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--temporal-cache",
        type=Path,
        default=Path("artifacts/p6_temporal_dtw/features"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h1_01_unseen_speaker_closedset/dtw"),
    )
    p.add_argument("--band-ratio", type=float, default=0.25)
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.embedding_npz, allow_pickle=False)
    utt_ids = d["known_utt_id"].astype(str)
    if len(utt_ids) != 442:
        raise RuntimeError(f"Expected 442 Core30 DEV utterances, got {len(utt_ids)}")

    mdsc = {}
    for row in read_jsonl(args.mdsc_manifest):
        if str(row.get("split", "")).lower() != "dev":
            continue
        uid = row_utt(row)
        if uid in set(utt_ids):
            mdsc[uid] = row_speaker(row)

    missing = [u for u in utt_ids if u not in mdsc]
    if missing:
        raise RuntimeError(f"Missing speaker metadata for {len(missing)} utts; first={missing[:5]}")

    temporal = load_temporal_index(args.temporal_cache)
    missing_t = [u for u in utt_ids if u not in temporal]
    if missing_t:
        raise RuntimeError(f"Missing temporal feature for {len(missing_t)} utts; first={missing_t[:5]}")

    speakers = np.asarray([mdsc[u] for u in utt_ids], dtype=str)
    groups = defaultdict(list)
    for i, s in enumerate(speakers):
        groups[s].append(i)

    if len(groups) != 4:
        raise RuntimeError(f"Expected 4 DEV speakers, got {len(groups)}")

    dist = np.full((len(utt_ids), len(utt_ids)), np.nan, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)

    seq_cache = {}

    def seq(i):
        if i not in seq_cache:
            seq_cache[i] = load_seq(temporal[utt_ids[i]])
        return seq_cache[i]

    total_pairs = sum(len(idx) * (len(idx) - 1) // 2 for idx in groups.values())
    done = 0

    for speaker, idxs in sorted(groups.items()):
        print(f"[speaker {speaker}] utterances={len(idxs)}")
        for ai in range(len(idxs)):
            i = idxs[ai]
            for bj in range(ai + 1, len(idxs)):
                j = idxs[bj]
                value = dtw_cosine_distance(
                    seq(i),
                    seq(j),
                    band_ratio=args.band_ratio,
                )
                dist[i, j] = value
                dist[j, i] = value
                done += 1

                if done % 500 == 0 or done == total_pairs:
                    print(f"[DTW] {done}/{total_pairs} within-speaker pairs")

    np.savez_compressed(
        args.output_dir / "within_speaker_dtw.npz",
        utt_id=utt_ids,
        speaker_id=speakers,
        dtw_distance=dist,
    )

    manifest = {
        "schema": "papr_ssl.h1_01_within_speaker_dtw.v1",
        "rows": len(utt_ids),
        "speakers": len(groups),
        "within_speaker_pairs": total_pairs,
        "band_ratio": args.band_ratio,
        "local_cost": "1 - cosine",
        "path_normalization": "cumulative_cost / path_length",
        "cross_speaker_entries": "NaN by design",
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("H1-01 WITHIN-SPEAKER DTW PRECOMPUTE")
    print(f"utterances:            {len(utt_ids)}")
    print(f"DEV speakers:          {len(groups)}")
    print(f"DTW pairs:             {total_pairs}")
    print("generic_test accessed: NO")
    print("PRECOMPUTE STATUS: PASS")


if __name__ == "__main__":
    main()
