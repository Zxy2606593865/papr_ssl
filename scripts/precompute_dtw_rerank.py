#!/usr/bin/env python
"""Precompute low-cost DTW reranking evidence.

Redundancy-control policy:
- global 256D Teacher first generates top-3 candidate classes;
- DTW is computed ONLY for those top-3 classes;
- each class keeps only 2 representative enrollment templates;
- representatives are the 2 enrollment embeddings closest to the class
  global prototype;
- no new neural encoder is introduced.

DTW local distance:
    d(a,b) = 1 - cosine(a,b)

DTW uses a fixed Sakoe-Chiba band.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def l2norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_temporal_index(cache_dir: Path):
    out = {}
    with (cache_dir / "index.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["utt_id"]] = cache_dir / row["cache_relpath"]
    return out


def load_seq(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    x = obj["features"].to(torch.float32).numpy()
    x = l2norm(x)
    return x


def dtw_cosine_distance(a, b, band_ratio=0.25):
    """Path-length normalized cosine DTW distance. Pure NumPy, banded."""
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
                (prev_cost[j], prev_len[j]),      # insertion
                (cur_cost[j - 1], cur_len[j - 1]),# deletion
                (prev_cost[j - 1], prev_len[j - 1]), # match
            )
            best_cost, best_len = min(candidates, key=lambda x: x[0])
            if not np.isfinite(best_cost):
                continue
            cur_cost[j] = best_cost + local
            cur_len[j] = best_len + 1

        prev_cost, prev_len = cur_cost, cur_len

    if not np.isfinite(prev_cost[m]) or prev_len[m] <= 0:
        raise RuntimeError(f"No DTW path for lengths {n}, {m}, band={band}")

    return float(prev_cost[m] / prev_len[m])


def choose_representatives(enrollment):
    """2 reps/class nearest to normalized mean prototype in global 256D."""
    e = l2norm(enrollment)
    proto = l2norm(e.mean(axis=1))
    reps = np.zeros((30, 2), dtype=np.int64)
    for c in range(30):
        sim = e[c] @ proto[c]
        order = np.argsort(-sim)
        reps[c] = order[:2]
    return proto, reps


def precompute_for_queries(
    query_global,
    query_ids,
    enrollment,
    enrollment_ids,
    proto,
    reps,
    temporal_index,
    top_k,
    band_ratio,
    role,
):
    q = l2norm(query_global)
    global_scores = q @ proto.T
    topk_cls = np.argsort(-global_scores, axis=1)[:, :top_k]
    topk_global = np.take_along_axis(global_scores, topk_cls, axis=1)
    topk_dtw = np.zeros_like(topk_global, dtype=np.float64)

    template_cache = {}
    for c in range(30):
        template_cache[c] = []
        for j in reps[c]:
            utt_id = enrollment_ids[c, j]
            template_cache[c].append(load_seq(temporal_index[utt_id]))

    for i, utt_id in enumerate(query_ids):
        qseq = load_seq(temporal_index[utt_id])

        for k, c in enumerate(topk_cls[i]):
            distances = [
                dtw_cosine_distance(qseq, tseq, band_ratio=band_ratio)
                for tseq in template_cache[int(c)]
            ]
            best_dist = min(distances)
            # local cosine distance in [0,2], convert to similarity roughly [0,1].
            topk_dtw[i, k] = 1.0 - 0.5 * best_dist

        if (i + 1) % 50 == 0 or i + 1 == len(query_ids):
            print(f"[DTW {role}] {i+1}/{len(query_ids)}")

    return {
        "global_scores": global_scores.astype(np.float32),
        "topk_class": topk_cls.astype(np.int64),
        "topk_global": topk_global.astype(np.float32),
        "topk_dtw": topk_dtw.astype(np.float32),
    }


def main() -> int:
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
        "--temporal-cache",
        type=Path,
        default=Path("artifacts/p6_temporal_dtw/features"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_temporal_dtw/precomputed"),
    )
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--templates-per-class", type=int, default=2)
    p.add_argument("--band-ratio", type=float, default=0.25)
    args = p.parse_args()

    if args.templates_per_class != 2:
        raise ValueError("This controlled experiment is frozen to 2 templates/class")
    if args.top_k != 3:
        raise ValueError("This controlled experiment is frozen to top_k=3")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tm = json.loads(
        (args.temporal_cache / "manifest.json").read_text(encoding="utf-8")
    )
    if tm["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("Temporal cache test seal violated")

    d = np.load(args.embedding_npz, allow_pickle=False)
    enrollment = d["enrollment"].astype(np.float64)
    enrollment_ids = d["enrollment_utt_id"].astype(str)
    known_global = d["known_embedding"].astype(np.float64)
    known_ids = d["known_utt_id"].astype(str)
    unknown_global = d["unknown_embedding"].astype(np.float64)
    unknown_ids = d["unknown_utt_id"].astype(str)

    proto, reps = choose_representatives(enrollment)
    temporal_index = load_temporal_index(args.temporal_cache)

    print("[DTW] representative enrollment indices per class:")
    print(reps)

    known = precompute_for_queries(
        known_global,
        known_ids,
        enrollment,
        enrollment_ids,
        proto,
        reps,
        temporal_index,
        args.top_k,
        args.band_ratio,
        "known",
    )
    unknown = precompute_for_queries(
        unknown_global,
        unknown_ids,
        enrollment,
        enrollment_ids,
        proto,
        reps,
        temporal_index,
        args.top_k,
        args.band_ratio,
        "unknown",
    )

    np.savez_compressed(
        args.output_dir / "dtw_rerank_evidence.npz",
        representative_indices=reps,
        known_global_scores=known["global_scores"],
        known_topk_class=known["topk_class"],
        known_topk_global=known["topk_global"],
        known_topk_dtw=known["topk_dtw"],
        unknown_global_scores=unknown["global_scores"],
        unknown_topk_class=unknown["topk_class"],
        unknown_topk_global=unknown["topk_global"],
        unknown_topk_dtw=unknown["topk_dtw"],
    )

    manifest = {
        "schema": "papr_ssl.temporal_dtw_evidence.v1",
        "top_k_classes": args.top_k,
        "templates_per_class": args.templates_per_class,
        "representative_policy": "2 enrollment samples closest to class 256D prototype",
        "dtw": {
            "local_cost": "1 - cosine_similarity",
            "normalization": "divide cumulative cost by DTW path length",
            "band": "Sakoe-Chiba",
            "band_ratio": args.band_ratio,
            "similarity": "1 - 0.5 * normalized_DTW_distance",
        },
        "redundancy_control": {
            "second_encoder": False,
            "global_teacher_reused": True,
            "rerank_only_top3": True,
            "temporal_templates_per_class": 2,
        },
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("DTW RERANK EVIDENCE PRECOMPUTE COMPLETE")
    print("new neural encoder:      NO")
    print("DTW candidate classes:   top-3 only")
    print("templates per class:     2")
    print("generic_test accessed:   NO")
    print("PRECOMPUTE STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
