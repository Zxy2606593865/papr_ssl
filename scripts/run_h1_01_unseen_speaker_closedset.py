#!/usr/bin/env python
"""H1-01 — Unseen-speaker personalized closed-set baseline.

Scientific question:
    A completely unseen DEV speaker supplies only 1 or 2 labeled enrollment
    recordings per phrase. Can the already-frozen 256D Teacher recognize the
    remaining recordings from that same speaker?

This script compares:
1) Global 256D prototype only
2) Global 256D prototype + Top-3 DTW reranking

No new Head is trained.
No rejector is used.
No unknown data is used in this step.

Protocol:
- 4 DEV speakers were never used to fit the Teacher/Head.
- For every speaker x phrase:
    N recordings -> support/enrollment
    remaining recordings -> query
- N in {1,2}
- 20 deterministic support selections
- fixed DTW fusion lambda=0.50, inherited from the prior development baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SHOT_LEVELS = (1, 2)
DEFAULT_REPEATS = 20
FUSION_LAMBDA = 0.50
TOP_K = 3


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


def row_utt(row):
    return str(get(row, ("utt_id", "id", "audio_id")))


def row_speaker(row):
    return str(get(row, ("speaker_id", "speaker", "user_id", "subject_id")))


def row_phrase(row):
    return normalize_phrase(
        get(row, ("transcript", "standard_text", "text", "label"))
    )


def stable_rank(seed, speaker, phrase, utt):
    h = hashlib.sha256()
    for value in (str(seed), speaker, phrase, utt):
        h.update(value.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def l2norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def macro_f1(true, pred, n_classes=30):
    vals = []
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        den = 2 * tp + fp + fn
        vals.append((2 * tp / den) if den else 0.0)
    return float(np.mean(vals))


def accuracy(true, pred):
    return float(np.mean(true == pred))


def recall_at_k(topk, true):
    return float(np.mean([t in row for row, t in zip(topk, true)]))


def summarize(vals):
    x = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "sample_std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


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
        "--dtw-npz",
        type=Path,
        default=Path(
            "artifacts/h1_01_unseen_speaker_closedset/dtw/"
            "within_speaker_dtw.npz"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h1_01_unseen_speaker_closedset/eval"),
    )
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--base-seed", type=int, default=20260915)
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.embedding_npz, allow_pickle=False)
    utt_ids = emb["known_utt_id"].astype(str)
    z = l2norm(emb["known_embedding"].astype(np.float64))

    if z.shape != (442, 256):
        raise RuntimeError(f"Expected known embeddings [442,256], got {z.shape}")

    # Metadata map for exactly the 442 Core30 DEV utterances.
    wanted = set(utt_ids)
    meta = {}
    for row in read_jsonl(args.mdsc_manifest):
        if str(row.get("split", "")).lower() != "dev":
            continue
        uid = row_utt(row)
        if uid not in wanted:
            continue
        meta[uid] = {
            "speaker": row_speaker(row),
            "phrase": row_phrase(row),
        }

    missing = [u for u in utt_ids if u not in meta]
    if missing:
        raise RuntimeError(f"Missing metadata for {len(missing)} utts")

    speakers = np.asarray([meta[u]["speaker"] for u in utt_ids], dtype=str)
    phrases = np.asarray([meta[u]["phrase"] for u in utt_ids], dtype=str)
    phrase_vocab = sorted(set(phrases))
    speaker_vocab = sorted(set(speakers))

    if len(phrase_vocab) != 30:
        raise RuntimeError(f"Expected 30 phrases, got {len(phrase_vocab)}")
    if len(speaker_vocab) != 4:
        raise RuntimeError(f"Expected 4 DEV speakers, got {len(speaker_vocab)}")

    class_of = {p: i for i, p in enumerate(phrase_vocab)}

    # Validate DTW matrix alignment.
    dd = np.load(args.dtw_npz, allow_pickle=False)
    dtw_ids = dd["utt_id"].astype(str)
    dtw_dist = dd["dtw_distance"].astype(np.float64)
    if not np.array_equal(dtw_ids, utt_ids):
        raise RuntimeError("DTW NPZ utt_id order does not match embedding NPZ")
    if dtw_dist.shape != (442, 442):
        raise RuntimeError(f"Unexpected DTW matrix shape {dtw_dist.shape}")

    groups = defaultdict(list)
    for i, (s, ph) in enumerate(zip(speakers, phrases)):
        groups[(s, ph)].append(i)

    # H0-02 promised that every DEV speaker x Core30 phrase supports 2-shot + query.
    for s in speaker_vocab:
        for ph in phrase_vocab:
            if len(groups[(s, ph)]) < 3:
                raise RuntimeError(
                    f"{s}/{ph} has only {len(groups[(s,ph)])} rows; "
                    "cannot support 2-shot + query"
                )

    runs = []

    for shot in SHOT_LEVELS:
        for repeat in range(args.repeats):
            seed = args.base_seed + repeat

            speaker_results = []
            pooled_true_g = []
            pooled_pred_g = []
            pooled_pred_d = []
            pooled_top3 = []

            for speaker in speaker_vocab:
                support = {}
                queries = []

                for ph in phrase_vocab:
                    idx = sorted(
                        groups[(speaker, ph)],
                        key=lambda i: stable_rank(
                            seed, speaker, ph, utt_ids[i]
                        ),
                    )
                    support[ph] = idx[:shot]
                    queries.extend(idx[shot:])

                # Global prototype per class for this speaker only.
                prototypes = []
                for ph in phrase_vocab:
                    pz = l2norm(z[support[ph]].mean(axis=0, keepdims=True))[0]
                    prototypes.append(pz)
                prototypes = np.stack(prototypes, axis=0)

                qz = z[queries]
                true = np.asarray(
                    [class_of[phrases[i]] for i in queries],
                    dtype=np.int64,
                )

                global_scores = qz @ prototypes.T
                order = np.argsort(-global_scores, axis=1)
                pred_global = order[:, 0].astype(np.int64)
                top3 = order[:, :TOP_K]

                # DTW reranks only the global Top-3.
                pred_dtw = np.empty(len(queries), dtype=np.int64)

                for qi, query_idx in enumerate(queries):
                    candidates = top3[qi]
                    fused = []

                    for c in candidates:
                        ph = phrase_vocab[int(c)]
                        best_dist = min(
                            float(dtw_dist[query_idx, si])
                            for si in support[ph]
                        )
                        if not np.isfinite(best_dist):
                            raise RuntimeError(
                                f"Non-finite DTW distance query={utt_ids[query_idx]} "
                                f"support_class={ph}"
                            )

                        dtw_similarity = 1.0 - 0.5 * best_dist
                        score = (
                            FUSION_LAMBDA * global_scores[qi, c]
                            + (1.0 - FUSION_LAMBDA) * dtw_similarity
                        )
                        fused.append(score)

                    pred_dtw[qi] = int(
                        candidates[int(np.argmax(fused))]
                    )

                sm = {
                    "speaker_id": speaker,
                    "query_rows": len(queries),
                    "global_macro_f1": macro_f1(true, pred_global),
                    "global_accuracy": accuracy(true, pred_global),
                    "global_recall_at_3": recall_at_k(top3, true),
                    "dtw_macro_f1": macro_f1(true, pred_dtw),
                    "dtw_accuracy": accuracy(true, pred_dtw),
                }
                speaker_results.append(sm)

                pooled_true_g.append(true)
                pooled_pred_g.append(pred_global)
                pooled_pred_d.append(pred_dtw)
                pooled_top3.append(top3)

            true_all = np.concatenate(pooled_true_g)
            pg_all = np.concatenate(pooled_pred_g)
            pd_all = np.concatenate(pooled_pred_d)
            top3_all = np.concatenate(pooled_top3)

            pooled = {
                "global_macro_f1": macro_f1(true_all, pg_all),
                "global_accuracy": accuracy(true_all, pg_all),
                "global_recall_at_3": recall_at_k(top3_all, true_all),
                "dtw_macro_f1": macro_f1(true_all, pd_all),
                "dtw_accuracy": accuracy(true_all, pd_all),
            }
            pooled["delta_macro_f1"] = (
                pooled["dtw_macro_f1"] - pooled["global_macro_f1"]
            )
            pooled["delta_accuracy"] = (
                pooled["dtw_accuracy"] - pooled["global_accuracy"]
            )

            runs.append(
                {
                    "shot": shot,
                    "repeat": repeat,
                    "seed": seed,
                    "fusion_lambda": FUSION_LAMBDA,
                    "speaker_results": speaker_results,
                    "pooled": pooled,
                }
            )

            print(
                f"[{shot}-shot repeat {repeat+1:02d}/{args.repeats}] "
                f"Global F1={pooled['global_macro_f1']:.4f} "
                f"ACC={pooled['global_accuracy']:.4f} "
                f"R@3={pooled['global_recall_at_3']:.4f} | "
                f"DTW F1={pooled['dtw_macro_f1']:.4f} "
                f"ACC={pooled['dtw_accuracy']:.4f} | "
                f"dF1={pooled['delta_macro_f1']:+.4f}"
            )

    summary = {}
    for shot in SHOT_LEVELS:
        rr = [r for r in runs if r["shot"] == shot]
        summary[str(shot)] = {
            "global_macro_f1": summarize(
                [r["pooled"]["global_macro_f1"] for r in rr]
            ),
            "global_accuracy": summarize(
                [r["pooled"]["global_accuracy"] for r in rr]
            ),
            "global_recall_at_3": summarize(
                [r["pooled"]["global_recall_at_3"] for r in rr]
            ),
            "dtw_macro_f1": summarize(
                [r["pooled"]["dtw_macro_f1"] for r in rr]
            ),
            "dtw_accuracy": summarize(
                [r["pooled"]["dtw_accuracy"] for r in rr]
            ),
            "delta_macro_f1": summarize(
                [r["pooled"]["delta_macro_f1"] for r in rr]
            ),
            "delta_accuracy": summarize(
                [r["pooled"]["delta_accuracy"] for r in rr]
            ),
            "dtw_better_f1_repeats": int(
                sum(r["pooled"]["delta_macro_f1"] > 0 for r in rr)
            ),
            "dtw_equal_f1_repeats": int(
                sum(abs(r["pooled"]["delta_macro_f1"]) < 1e-12 for r in rr)
            ),
            "dtw_worse_f1_repeats": int(
                sum(r["pooled"]["delta_macro_f1"] < 0 for r in rr)
            ),
        }

    out = {
        "schema": "papr_ssl.h1_01_unseen_speaker_closedset.v1",
        "task": "MDSC Core30 unseen-speaker personalized closed-set",
        "dev_speakers": speaker_vocab,
        "speaker_count": 4,
        "class_count": 30,
        "shot_levels": list(SHOT_LEVELS),
        "repeats": args.repeats,
        "support_selection": "deterministic seeded per speaker x phrase",
        "query_policy": "all remaining distinct recordings after support reservation",
        "systems": {
            "global": "same-speaker support mean prototype in frozen 256D Teacher space",
            "dtw": (
                "global Top-3 + within-speaker support DTW rerank; "
                f"fixed lambda={FUSION_LAMBDA}"
            ),
        },
        "new_trainable_head_added": False,
        "unknown_data_used": False,
        "rejector_used": False,
        "cross_session_claim": False,
        "summary": summary,
        "runs": runs,
        "generic_test": "sealed_not_accessed",
    }

    (args.output_dir / "h1_01_unseen_speaker_closedset.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("H1-01 UNSEEN-SPEAKER PERSONALIZED CLOSED-SET SUMMARY")
    print("=" * 108)
    for shot in SHOT_LEVELS:
        s = summary[str(shot)]
        print(
            f"{shot}-shot | "
            f"GLOBAL F1={s['global_macro_f1']['mean']:.6f} "
            f"ACC={s['global_accuracy']['mean']:.6f} "
            f"R@3={s['global_recall_at_3']['mean']:.6f} | "
            f"DTW F1={s['dtw_macro_f1']['mean']:.6f} "
            f"ACC={s['dtw_accuracy']['mean']:.6f} | "
            f"dF1={s['delta_macro_f1']['mean']:+.6f} | "
            f"DTW better/equal/worse="
            f"{s['dtw_better_f1_repeats']}/"
            f"{s['dtw_equal_f1_repeats']}/"
            f"{s['dtw_worse_f1_repeats']}"
        )

    print("-" * 108)
    print("new trainable Head:     NO")
    print("unknown data used:      NO")
    print("rejector used:          NO")
    print("cross-session claim:    NO")
    print("generic_test accessed:  NO")
    print("H1-01 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
