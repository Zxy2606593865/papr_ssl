#!/usr/bin/env python
"""H1-02 — Unseen-speaker personalized open-set baseline.

Goal
----
Test whether the frozen personalized recognizer can:
1) correctly accept a registered phrase;
2) reject a phrase that this user did NOT register.

This is still a baseline:
- no new neural Head;
- no Unknown Exposure training;
- no LOO/Shrinkage;
- no C/W/U decision model.

Protocol
--------
- MDSC Core30 DEV only: 4 speakers unseen during training.
- For each speaker and repeat:
    * 20 phrases are registered (known).
    * remaining 10 Core30 phrases are unregistered (unknown relative to this user).
    * registered phrase: N support recordings + remaining query recordings.
    * unregistered phrase: all recordings are unknown queries.
- N in {1,2}.
- Global and Global+DTW are evaluated separately.
- Rejection uses ONE scalar threshold on top-1 score.
- For each held-out DEV speaker, threshold is calibrated only on the OTHER 3
  DEV speakers (leave-one-speaker-out calibration).
- Calibration rule: maximize known Correct Accept subject to unknown FAR <= 10%.
- generic_test stays sealed.

Important
---------
Unknown here means "not in this user's current enrollment set", not a permanent
unknown class.
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
REPEATS = 20
REGISTERED_CLASSES = 20
TOP_K = 3
FUSION_LAMBDA = 0.50
TARGET_UNKNOWN_FAR = 0.10


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


def stable_hash(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def l2norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def choose_registered(phrase_vocab, repeat):
    ranked = sorted(
        phrase_vocab,
        key=lambda ph: stable_hash("registered_vocab", repeat, ph),
    )
    return set(ranked[:REGISTERED_CLASSES])


def choose_support(indices, shot, repeat, speaker, phrase, utt_ids):
    return sorted(
        indices,
        key=lambda i: stable_hash(
            "support", repeat, speaker, phrase, utt_ids[i]
        ),
    )[:shot]


def macro_f1(true, pred, class_ids):
    vals = []
    for c in class_ids:
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        den = 2 * tp + fp + fn
        vals.append((2 * tp / den) if den else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def build_speaker_episode(
    speaker,
    shot,
    repeat,
    phrase_vocab,
    class_of,
    groups,
    z,
    dtw_dist,
    utt_ids,
):
    registered = choose_registered(phrase_vocab, repeat)
    registered_sorted = sorted(registered)
    registered_class_ids = [class_of[p] for p in registered_sorted]

    support = {}
    known_queries = []
    unknown_queries = []

    for ph in phrase_vocab:
        idx = list(groups[(speaker, ph)])
        if ph in registered:
            sup = choose_support(
                idx, shot, repeat, speaker, ph, utt_ids
            )
            support[ph] = sup
            sup_set = set(sup)
            known_queries.extend(i for i in idx if i not in sup_set)
        else:
            unknown_queries.extend(idx)

    # Same-speaker personalized prototypes only.
    prototypes = []
    for ph in registered_sorted:
        p = l2norm(z[support[ph]].mean(axis=0, keepdims=True))[0]
        prototypes.append(p)
    prototypes = np.stack(prototypes, axis=0)

    def score_queries(query_indices):
        qz = z[query_indices]
        global_scores = qz @ prototypes.T
        order = np.argsort(-global_scores, axis=1)
        topk = order[:, : min(TOP_K, len(registered_sorted))]
        pred_global_local = order[:, 0]
        score_global = global_scores[np.arange(len(query_indices)), pred_global_local]

        pred_dtw_local = np.empty(len(query_indices), dtype=np.int64)
        score_dtw = np.empty(len(query_indices), dtype=np.float64)

        for qi, query_idx in enumerate(query_indices):
            candidates = topk[qi]
            fused_scores = []

            for local_c in candidates:
                ph = registered_sorted[int(local_c)]
                best_dist = min(
                    float(dtw_dist[query_idx, si])
                    for si in support[ph]
                )
                if not np.isfinite(best_dist):
                    raise RuntimeError(
                        f"Non-finite DTW distance speaker={speaker} "
                        f"query={utt_ids[query_idx]} phrase={ph}"
                    )
                dtw_similarity = 1.0 - 0.5 * best_dist
                fused = (
                    FUSION_LAMBDA * global_scores[qi, local_c]
                    + (1.0 - FUSION_LAMBDA) * dtw_similarity
                )
                fused_scores.append(fused)

            best_pos = int(np.argmax(fused_scores))
            pred_dtw_local[qi] = int(candidates[best_pos])
            score_dtw[qi] = float(fused_scores[best_pos])

        pred_global = np.asarray(
            [registered_class_ids[int(i)] for i in pred_global_local],
            dtype=np.int64,
        )
        pred_dtw = np.asarray(
            [registered_class_ids[int(i)] for i in pred_dtw_local],
            dtype=np.int64,
        )

        return {
            "global_pred": pred_global,
            "global_score": score_global.astype(np.float64),
            "dtw_pred": pred_dtw,
            "dtw_score": score_dtw,
        }

    known = score_queries(known_queries)
    unknown = score_queries(unknown_queries)

    known_true = np.asarray(
        [class_of[phrase_vocab[class_of[phrase_vocab[0]]]] for _ in []],
        dtype=np.int64,
    )
    # Simpler and explicit true labels.
    phrase_by_index = {}
    for ph in phrase_vocab:
        for i in groups[(speaker, ph)]:
            phrase_by_index[i] = ph

    known_true = np.asarray(
        [class_of[phrase_by_index[i]] for i in known_queries],
        dtype=np.int64,
    )

    return {
        "speaker": speaker,
        "shot": shot,
        "repeat": repeat,
        "registered_phrases": registered_sorted,
        "registered_class_ids": registered_class_ids,
        "known_true": known_true,
        "known_global_pred": known["global_pred"],
        "known_global_score": known["global_score"],
        "known_dtw_pred": known["dtw_pred"],
        "known_dtw_score": known["dtw_score"],
        "unknown_global_pred": unknown["global_pred"],
        "unknown_global_score": unknown["global_score"],
        "unknown_dtw_pred": unknown["dtw_pred"],
        "unknown_dtw_score": unknown["dtw_score"],
    }


def select_threshold(cal_episodes, system, target_far):
    known_scores = []
    known_correct = []
    unknown_scores = []

    for e in cal_episodes:
        ks = e[f"known_{system}_score"]
        kp = e[f"known_{system}_pred"]
        yt = e["known_true"]
        us = e[f"unknown_{system}_score"]

        known_scores.append(ks)
        known_correct.append(kp == yt)
        unknown_scores.append(us)

    known_scores = np.concatenate(known_scores)
    known_correct = np.concatenate(known_correct)
    unknown_scores = np.concatenate(unknown_scores)

    # Accept when score >= threshold.
    candidates = np.unique(
        np.concatenate([
            known_scores,
            unknown_scores,
            [np.inf, -np.inf],
        ])
    )

    best = None
    for t in candidates:
        unknown_far = float(np.mean(unknown_scores >= t))
        if unknown_far > target_far + 1e-12:
            continue

        correct_accept = float(
            np.mean((known_scores >= t) & known_correct)
        )
        known_accept = float(np.mean(known_scores >= t))

        key = (correct_accept, -unknown_far, known_accept, t)
        if best is None or key > best["key"]:
            best = {
                "threshold": float(t),
                "cal_correct_accept": correct_accept,
                "cal_unknown_far": unknown_far,
                "cal_known_accept": known_accept,
                "key": key,
            }

    if best is None:
        raise RuntimeError("No feasible threshold found")

    best.pop("key")
    return best


def evaluate_episode(e, system, threshold):
    yt = e["known_true"]
    pred = e[f"known_{system}_pred"]
    score = e[f"known_{system}_score"]
    uscore = e[f"unknown_{system}_score"]

    accept_known = score >= threshold
    correct = pred == yt

    ca = float(np.mean(accept_known & correct))
    wi = float(np.mean(accept_known & ~correct))
    kr = float(np.mean(~accept_known))
    ua = float(np.mean(uscore >= threshold))
    ur = 1.0 - ua

    closed_f1 = macro_f1(
        yt,
        pred,
        class_ids=e["registered_class_ids"],
    )
    closed_acc = float(np.mean(correct))

    return {
        "known_count": int(len(yt)),
        "unknown_count": int(len(uscore)),
        "closed_macro_f1": closed_f1,
        "closed_accuracy": closed_acc,
        "correct_accept": ca,
        "wrong_intent": wi,
        "known_reject": kr,
        "unknown_accept": ua,
        "unknown_reject": ur,
        "known_identity_sum": ca + wi + kr,
    }


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
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
        default=Path("artifacts/h1_02_unseen_speaker_openset"),
    )
    p.add_argument("--repeats", type=int, default=REPEATS)
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.embedding_npz, allow_pickle=False)
    utt_ids = emb["known_utt_id"].astype(str)
    z = l2norm(emb["known_embedding"].astype(np.float64))
    if z.shape != (442, 256):
        raise RuntimeError(f"Expected [442,256], got {z.shape}")

    wanted = set(utt_ids)
    meta = {}
    for row in read_jsonl(args.mdsc_manifest):
        if str(row.get("split", "")).lower() != "dev":
            continue
        uid = row_utt(row)
        if uid in wanted:
            meta[uid] = {
                "speaker": row_speaker(row),
                "phrase": row_phrase(row),
            }

    if len(meta) != len(utt_ids):
        raise RuntimeError(
            f"Metadata mismatch: {len(meta)} for {len(utt_ids)} utts"
        )

    speakers = np.asarray([meta[u]["speaker"] for u in utt_ids], dtype=str)
    phrases = np.asarray([meta[u]["phrase"] for u in utt_ids], dtype=str)
    speaker_vocab = sorted(set(speakers))
    phrase_vocab = sorted(set(phrases))
    if len(speaker_vocab) != 4 or len(phrase_vocab) != 30:
        raise RuntimeError(
            f"Expected 4 speakers/30 phrases, got "
            f"{len(speaker_vocab)}/{len(phrase_vocab)}"
        )

    class_of = {p: i for i, p in enumerate(phrase_vocab)}

    dd = np.load(args.dtw_npz, allow_pickle=False)
    if not np.array_equal(dd["utt_id"].astype(str), utt_ids):
        raise RuntimeError("DTW utt_id order mismatch")
    dtw_dist = dd["dtw_distance"].astype(np.float64)

    groups = defaultdict(list)
    for i, (s, ph) in enumerate(zip(speakers, phrases)):
        groups[(s, ph)].append(i)

    for s in speaker_vocab:
        for ph in phrase_vocab:
            if len(groups[(s, ph)]) < 3:
                raise RuntimeError(
                    f"{s}/{ph} has <3 rows; 2-shot protocol impossible"
                )

    all_results = []

    for shot in SHOT_LEVELS:
        for repeat in range(args.repeats):
            episodes = {
                s: build_speaker_episode(
                    speaker=s,
                    shot=shot,
                    repeat=repeat,
                    phrase_vocab=phrase_vocab,
                    class_of=class_of,
                    groups=groups,
                    z=z,
                    dtw_dist=dtw_dist,
                    utt_ids=utt_ids,
                )
                for s in speaker_vocab
            }

            for heldout in speaker_vocab:
                cal = [
                    episodes[s]
                    for s in speaker_vocab
                    if s != heldout
                ]
                score_ep = episodes[heldout]

                fold_result = {
                    "shot": shot,
                    "repeat": repeat,
                    "heldout_speaker": heldout,
                    "registered_class_count": REGISTERED_CLASSES,
                    "unknown_class_count": 30 - REGISTERED_CLASSES,
                }

                for system in ("global", "dtw"):
                    cal_info = select_threshold(
                        cal,
                        system=system,
                        target_far=TARGET_UNKNOWN_FAR,
                    )
                    metrics = evaluate_episode(
                        score_ep,
                        system=system,
                        threshold=cal_info["threshold"],
                    )
                    fold_result[system] = {
                        "calibration": cal_info,
                        "metrics": metrics,
                    }

                all_results.append(fold_result)

            # Print pooled summary over the 4 held-out speaker folds.
            rr = [
                r for r in all_results
                if r["shot"] == shot and r["repeat"] == repeat
            ]
            for system in ("global", "dtw"):
                ca = np.mean(
                    [r[system]["metrics"]["correct_accept"] for r in rr]
                )
                wi = np.mean(
                    [r[system]["metrics"]["wrong_intent"] for r in rr]
                )
                kr = np.mean(
                    [r[system]["metrics"]["known_reject"] for r in rr]
                )
                ur = np.mean(
                    [r[system]["metrics"]["unknown_reject"] for r in rr]
                )
                if system == "global":
                    prefix = (
                        f"[{shot}-shot repeat {repeat+1:02d}/{args.repeats}] "
                    )
                    print(
                        prefix
                        + f"GLOBAL CA={ca:.4f} WI={wi:.4f} "
                        f"KR={kr:.4f} UR={ur:.4f}"
                    )
                else:
                    print(
                        " " * 29
                        + f"DTW    CA={ca:.4f} WI={wi:.4f} "
                        f"KR={kr:.4f} UR={ur:.4f}"
                    )

    summary = {}
    for shot in SHOT_LEVELS:
        rr = [r for r in all_results if r["shot"] == shot]
        summary[str(shot)] = {}

        for system in ("global", "dtw"):
            summary[str(shot)][system] = {}
            for metric in (
                "closed_macro_f1",
                "closed_accuracy",
                "correct_accept",
                "wrong_intent",
                "known_reject",
                "unknown_accept",
                "unknown_reject",
            ):
                summary[str(shot)][system][metric] = summarize(
                    [r[system]["metrics"][metric] for r in rr]
                )

        for metric in (
            "correct_accept",
            "wrong_intent",
            "known_reject",
            "unknown_reject",
        ):
            summary[str(shot)][f"delta_dtw_minus_global_{metric}"] = summarize(
                [
                    r["dtw"]["metrics"][metric]
                    - r["global"]["metrics"][metric]
                    for r in rr
                ]
            )

    out = {
        "schema": "papr_ssl.h1_02_unseen_speaker_openset.v1",
        "task": "MDSC Core30 unseen-speaker personalized open-set baseline",
        "speaker_count": 4,
        "class_count": 30,
        "registered_classes_per_episode": REGISTERED_CLASSES,
        "unregistered_classes_per_episode": 30 - REGISTERED_CLASSES,
        "shot_levels": list(SHOT_LEVELS),
        "repeats": args.repeats,
        "systems": {
            "global": "256D personalized prototype score",
            "dtw": (
                "Global Top-3 + same-speaker DTW; fixed lambda="
                f"{FUSION_LAMBDA}"
            ),
        },
        "rejection": {
            "type": "single top1-score threshold baseline",
            "target_calibration_unknown_far": TARGET_UNKNOWN_FAR,
            "calibration": (
                "leave-one-DEV-speaker-out: 3 speakers calibrate threshold, "
                "held-out speaker is scored"
            ),
            "selection_rule": (
                "maximize known Correct Accept subject to unknown FAR <= 10%"
            ),
        },
        "unknown_definition": (
            "10 Core30 phrases not registered in the current user's episode"
        ),
        "new_neural_head_added": False,
        "unknown_exposure_training_used": False,
        "loo_shrinkage_used": False,
        "cross_session_claim": False,
        "generic_test": "sealed_not_accessed",
        "summary": summary,
        "fold_results": all_results,
    }

    (args.output_dir / "h1_02_unseen_speaker_openset.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("H1-02 UNSEEN-SPEAKER PERSONALIZED OPEN-SET SUMMARY")
    print("=" * 108)
    for shot in SHOT_LEVELS:
        for system in ("global", "dtw"):
            s = summary[str(shot)][system]
            print(
                f"{shot}-shot {system.upper():6s} | "
                f"CA={s['correct_accept']['mean']:.6f} "
                f"WI={s['wrong_intent']['mean']:.6f} "
                f"KR={s['known_reject']['mean']:.6f} "
                f"UR={s['unknown_reject']['mean']:.6f} "
                f"ClosedF1={s['closed_macro_f1']['mean']:.6f}"
            )
        d = summary[str(shot)]
        print(
            f"{shot}-shot DTW-GLOBAL | "
            f"dCA={d['delta_dtw_minus_global_correct_accept']['mean']:+.6f} "
            f"dWI={d['delta_dtw_minus_global_wrong_intent']['mean']:+.6f} "
            f"dKR={d['delta_dtw_minus_global_known_reject']['mean']:+.6f} "
            f"dUR={d['delta_dtw_minus_global_unknown_reject']['mean']:+.6f}"
        )

    print("-" * 108)
    print("new neural Head:         NO")
    print("Unknown Exposure train:  NO")
    print("LOO/Shrinkage:           NO")
    print("cross-session claim:     NO")
    print("generic_test accessed:   NO")
    print("H1-02 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
