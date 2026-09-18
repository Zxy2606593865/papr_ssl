#!/usr/bin/env python
"""H4-01 — First trainable Task Head: shared evidence ranker.

What this is
------------
This is the FIRST step that actually FITS a Task Head.

The Backbone/Neck stay frozen. For each candidate phrase we build a small
evidence vector from already-computed features:

    global cosine score
    global candidate margin
    DTW similarity
    DTW candidate margin
    support count (1/2-shot)
    query/support duration mismatch
    second-template availability

A single shared linear logistic ranker learns:
    "how likely is this candidate to be the correct registered phrase?"

It is shared across all phrases/users; no 30-way fixed output layer is learned.

Fit:
    MDSC TRAIN speakers only (34 speakers)
Evaluate:
    MDSC DEV speakers only (4 unseen speakers)

This H4-01 is CLOSED-SET ranking only.
No rejector, no Unknown Exposure, no ACCEPT/REJECT decision yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SHOT_LEVELS = (1, 2)
FIT_REPEATS = 3
DEV_REPEATS = 20
TOP_K = 3
FIXED_LAMBDA = 0.50
FEATURE_NAMES = (
    "global_score",
    "global_margin",
    "dtw_similarity",
    "dtw_margin",
    "log_support_n",
    "duration_log_ratio",
    "second_template_available",
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


def stable_support(indices, shot, repeat, speaker, label, utt_ids):
    return sorted(
        indices,
        key=lambda i: stable_hash(
            "support", repeat, speaker, label, utt_ids[i]
        ),
    )[:shot]


def load_temporal_index(feature_dir: Path):
    index = {}
    path = feature_dir / "temporal" / "index.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index[row["utt_id"]] = {
            **row,
            "path": feature_dir / "temporal" / row["cache_relpath"],
        }
    return index


class TemporalStore:
    def __init__(self, index):
        self.index = index
        self.cache = {}
        self.dtw_cache = {}

    def seq(self, utt_id):
        if utt_id not in self.cache:
            obj = torch.load(
                self.index[utt_id]["path"],
                map_location="cpu",
                weights_only=False,
            )
            x = obj["features"].to(torch.float32).numpy()
            self.cache[utt_id] = l2norm(x)
        return self.cache[utt_id]

    def frames(self, utt_id):
        return int(self.index[utt_id]["frames"])

    def dtw(self, a_id, b_id, band_ratio=0.25):
        key = tuple(sorted((a_id, b_id)))
        if key in self.dtw_cache:
            return self.dtw_cache[key]

        a = self.seq(a_id)
        b = self.seq(b_id)
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
                local = 1.0 - float(
                    np.clip(np.dot(ai, b[j - 1]), -1.0, 1.0)
                )
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
            raise RuntimeError(f"No DTW path for {a_id}, {b_id}")

        out = float(prev_cost[m] / prev_len[m])
        self.dtw_cache[key] = out
        return out


def load_split(npz, split):
    return {
        "utt_id": npz[f"{split}_utt_id"].astype(str),
        "speaker_id": npz[f"{split}_speaker_id"].astype(str),
        "label": npz[f"{split}_label"].astype(str),
        "label_index": npz[f"{split}_label_index"].astype(np.int64),
        "embedding": l2norm(npz[f"{split}_embedding"].astype(np.float64)),
    }


def groups_for_split(data):
    groups = defaultdict(list)
    for i, (s, label) in enumerate(
        zip(data["speaker_id"], data["label"])
    ):
        groups[(s, label)].append(i)
    return groups


def macro_f1(true, pred, class_ids):
    vals = []
    for c in class_ids:
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        den = 2 * tp + fp + fn
        vals.append((2 * tp / den) if den else 0.0)
    return float(np.mean(vals))


def make_episode_rows(
    *,
    data,
    groups,
    temporal,
    shot,
    repeat,
    top_k=TOP_K,
):
    speakers = sorted(set(data["speaker_id"]))
    labels = sorted(set(data["label"]))
    class_of = {
        label: int(
            data["label_index"][np.where(data["label"] == label)[0][0]]
        )
        for label in labels
    }

    candidate_features = []
    candidate_targets = []
    query_records = []

    for speaker in speakers:
        # Keep only labels that truly support N-shot + at least one query.
        eligible = [
            label
            for label in labels
            if len(groups[(speaker, label)]) >= shot + 1
        ]
        if len(eligible) < 2:
            continue

        support = {}
        queries = []
        for label in eligible:
            idx = groups[(speaker, label)]
            sup = stable_support(
                idx,
                shot,
                repeat,
                speaker,
                label,
                data["utt_id"],
            )
            support[label] = sup
            sup_set = set(sup)
            queries.extend(i for i in idx if i not in sup_set)

        prototypes = []
        for label in eligible:
            p = l2norm(
                data["embedding"][support[label]].mean(
                    axis=0, keepdims=True
                )
            )[0]
            prototypes.append(p)
        prototypes = np.stack(prototypes, axis=0)

        local_class_ids = np.asarray(
            [class_of[label] for label in eligible],
            dtype=np.int64,
        )

        for qidx in queries:
            qz = data["embedding"][qidx]
            true_global_class = int(data["label_index"][qidx])
            global_scores = qz @ prototypes.T
            order = np.argsort(-global_scores)
            candidates = order[: min(top_k, len(order))]

            # DTW similarities for all current candidates first, so we can
            # compute candidate-relative DTW margin.
            dtw_sims = []
            duration_ratios = []
            second_flags = []

            q_utt = data["utt_id"][qidx]
            q_frames = temporal.frames(q_utt)

            for local_c in candidates:
                label = eligible[int(local_c)]
                sims = []
                support_frames = []

                for sidx in support[label]:
                    s_utt = data["utt_id"][sidx]
                    d = temporal.dtw(q_utt, s_utt)
                    sims.append(1.0 - 0.5 * d)
                    support_frames.append(temporal.frames(s_utt))

                dtw_sims.append(float(max(sims)))
                med_frames = float(np.median(support_frames))
                duration_ratios.append(
                    abs(math.log(max(q_frames, 1) / max(med_frames, 1.0)))
                )
                second_flags.append(1.0 if len(sims) >= 2 else 0.0)

            dtw_sims = np.asarray(dtw_sims, dtype=np.float64)

            start = len(candidate_features)

            for pos, local_c in enumerate(candidates):
                g = float(global_scores[local_c])
                others_g = [
                    float(global_scores[o])
                    for o in candidates
                    if int(o) != int(local_c)
                ]
                g_margin = g - max(others_g) if others_g else 0.0

                dtw = float(dtw_sims[pos])
                other_dtw = [
                    float(v)
                    for j, v in enumerate(dtw_sims)
                    if j != pos
                ]
                dtw_margin = dtw - max(other_dtw) if other_dtw else 0.0

                feat = [
                    g,
                    g_margin,
                    dtw,
                    dtw_margin,
                    math.log(float(shot)),
                    float(duration_ratios[pos]),
                    float(second_flags[pos]),
                ]

                candidate_features.append(feat)
                pred_class = int(local_class_ids[int(local_c)])
                candidate_targets.append(
                    1.0 if pred_class == true_global_class else 0.0
                )

            stop = len(candidate_features)
            query_records.append(
                {
                    "speaker_id": speaker,
                    "query_utt": q_utt,
                    "true_class": true_global_class,
                    "candidate_local_positions": candidates.astype(int).tolist(),
                    "candidate_global_class_ids": [
                        int(local_class_ids[int(c)]) for c in candidates
                    ],
                    "feature_slice": [start, stop],
                    "global_scores": [
                        float(global_scores[c]) for c in candidates
                    ],
                    "dtw_scores": dtw_sims.astype(float).tolist(),
                }
            )

    return (
        np.asarray(candidate_features, dtype=np.float32),
        np.asarray(candidate_targets, dtype=np.float32),
        query_records,
    )


def fit_logistic(X, y, epochs=400, lr=0.03, weight_decay=1e-3):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    Xs = (X - mean) / std

    xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    model = torch.nn.Linear(X.shape[1], 1)
    torch.manual_seed(20260915)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)

    pos = float(y.sum())
    neg = float(len(y) - pos)
    pos_weight = torch.tensor(
        [neg / max(pos, 1.0)],
        dtype=torch.float32,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best = None
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(xt).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            yt,
            pos_weight=pos_weight,
        )
        loss.backward()
        opt.step()

        current = float(loss.detach())
        if best is None or current < best["loss"]:
            best = {
                "loss": current,
                "epoch": epoch,
                "state": {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                },
            }

    model.load_state_dict(best["state"])
    return model, mean, std, best


def rank_queries(
    X,
    query_records,
    model,
    mean,
    std,
):
    xs = torch.tensor((X - mean) / std, dtype=torch.float32)
    with torch.inference_mode():
        learned = torch.sigmoid(
            model(xs).squeeze(1)
        ).numpy()

    true = []
    pred_global = []
    pred_fixed = []
    pred_learned = []

    for q in query_records:
        a, b = q["feature_slice"]
        global_scores = np.asarray(q["global_scores"])
        dtw_scores = np.asarray(q["dtw_scores"])
        candidate_ids = np.asarray(q["candidate_global_class_ids"])

        fixed = (
            FIXED_LAMBDA * global_scores
            + (1.0 - FIXED_LAMBDA) * dtw_scores
        )
        learned_scores = learned[a:b]

        true.append(int(q["true_class"]))
        pred_global.append(int(candidate_ids[np.argmax(global_scores)]))
        pred_fixed.append(int(candidate_ids[np.argmax(fixed)]))
        pred_learned.append(int(candidate_ids[np.argmax(learned_scores)]))

    return (
        np.asarray(true, dtype=np.int64),
        np.asarray(pred_global, dtype=np.int64),
        np.asarray(pred_fixed, dtype=np.int64),
        np.asarray(pred_learned, dtype=np.int64),
    )


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
        "--feature-dir",
        type=Path,
        default=Path("artifacts/h4_core30_features"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h4_01_shared_evidence_ranker"),
    )
    p.add_argument("--fit-repeats", type=int, default=FIT_REPEATS)
    p.add_argument("--dev-repeats", type=int, default=DEV_REPEATS)
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(
        args.feature_dir / "global" / "core30_global256.npz",
        allow_pickle=False,
    )
    train = load_split(d, "train")
    dev = load_split(d, "dev")

    if len(set(train["speaker_id"])) != 34:
        raise RuntimeError("Expected 34 TRAIN speakers")
    if len(set(dev["speaker_id"])) != 4:
        raise RuntimeError("Expected 4 DEV speakers")

    temporal_index = load_temporal_index(args.feature_dir)
    needed = set(train["utt_id"]) | set(dev["utt_id"])
    missing = sorted(needed - set(temporal_index))
    if missing:
        raise RuntimeError(
            f"Missing temporal features: {len(missing)} first={missing[:5]}"
        )
    temporal = TemporalStore(temporal_index)

    train_groups = groups_for_split(train)
    dev_groups = groups_for_split(dev)

    fit_X = []
    fit_y = []

    print("=" * 108)
    print("H4-01 BUILD TRAIN EVIDENCE")
    print("=" * 108)

    for shot in SHOT_LEVELS:
        for repeat in range(args.fit_repeats):
            X, y, _ = make_episode_rows(
                data=train,
                groups=train_groups,
                temporal=temporal,
                shot=shot,
                repeat=repeat,
            )
            fit_X.append(X)
            fit_y.append(y)
            print(
                f"[TRAIN {shot}-shot {repeat+1}/{args.fit_repeats}] "
                f"candidate rows={len(X)} positives={int(y.sum())}"
            )

    fit_X = np.concatenate(fit_X, axis=0)
    fit_y = np.concatenate(fit_y, axis=0)

    model, mean, std, fit_info = fit_logistic(fit_X, fit_y)

    weights = model.weight.detach().cpu().numpy()[0]
    bias = float(model.bias.detach().cpu().numpy()[0])

    ranker = {
        "schema": "papr_ssl.h4_01_shared_evidence_ranker.v1",
        "feature_names": list(FEATURE_NAMES),
        "standardize_mean": mean.tolist(),
        "standardize_std": std.tolist(),
        "weight": weights.tolist(),
        "bias": bias,
        "fit_loss": fit_info["loss"],
        "fit_epoch": fit_info["epoch"],
        "fit_candidate_rows": int(len(fit_X)),
        "fit_positive_rows": int(fit_y.sum()),
        "fit_speakers": 34,
        "frozen_backbone_neck": True,
    }
    (args.output_dir / "ranker.json").write_text(
        json.dumps(ranker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    results = []

    print("=" * 108)
    print("H4-01 EVALUATE UNSEEN DEV SPEAKERS")
    print("=" * 108)

    class_ids = sorted(set(dev["label_index"].tolist()))

    for shot in SHOT_LEVELS:
        for repeat in range(args.dev_repeats):
            X, _, queries = make_episode_rows(
                data=dev,
                groups=dev_groups,
                temporal=temporal,
                shot=shot,
                repeat=repeat,
            )
            true, pg, pf, pl = rank_queries(
                X, queries, model, mean, std
            )

            row = {
                "shot": shot,
                "repeat": repeat,
                "query_rows": int(len(true)),
                "global_macro_f1": macro_f1(true, pg, class_ids),
                "fixed_dtw_macro_f1": macro_f1(true, pf, class_ids),
                "learned_macro_f1": macro_f1(true, pl, class_ids),
                "global_accuracy": float(np.mean(true == pg)),
                "fixed_dtw_accuracy": float(np.mean(true == pf)),
                "learned_accuracy": float(np.mean(true == pl)),
            }
            row["learned_minus_fixed_f1"] = (
                row["learned_macro_f1"]
                - row["fixed_dtw_macro_f1"]
            )
            row["learned_minus_global_f1"] = (
                row["learned_macro_f1"]
                - row["global_macro_f1"]
            )
            results.append(row)

            print(
                f"[DEV {shot}-shot {repeat+1:02d}/{args.dev_repeats}] "
                f"Global={row['global_macro_f1']:.4f} "
                f"FixedDTW={row['fixed_dtw_macro_f1']:.4f} "
                f"Learned={row['learned_macro_f1']:.4f} "
                f"dLearned-Fixed={row['learned_minus_fixed_f1']:+.4f}"
            )

    summary = {}
    for shot in SHOT_LEVELS:
        rr = [r for r in results if r["shot"] == shot]
        summary[str(shot)] = {
            "global_macro_f1": summarize(
                [r["global_macro_f1"] for r in rr]
            ),
            "fixed_dtw_macro_f1": summarize(
                [r["fixed_dtw_macro_f1"] for r in rr]
            ),
            "learned_macro_f1": summarize(
                [r["learned_macro_f1"] for r in rr]
            ),
            "global_accuracy": summarize(
                [r["global_accuracy"] for r in rr]
            ),
            "fixed_dtw_accuracy": summarize(
                [r["fixed_dtw_accuracy"] for r in rr]
            ),
            "learned_accuracy": summarize(
                [r["learned_accuracy"] for r in rr]
            ),
            "learned_minus_fixed_f1": summarize(
                [r["learned_minus_fixed_f1"] for r in rr]
            ),
            "learned_minus_global_f1": summarize(
                [r["learned_minus_global_f1"] for r in rr]
            ),
            "learned_better_than_fixed_repeats": int(
                sum(r["learned_minus_fixed_f1"] > 0 for r in rr)
            ),
            "learned_equal_fixed_repeats": int(
                sum(abs(r["learned_minus_fixed_f1"]) < 1e-12 for r in rr)
            ),
            "learned_worse_than_fixed_repeats": int(
                sum(r["learned_minus_fixed_f1"] < 0 for r in rr)
            ),
        }

    out = {
        "schema": "papr_ssl.h4_01_shared_evidence_ranker_eval.v1",
        "task": "unseen-speaker personalized closed-set candidate ranking",
        "fit": {
            "speakers": 34,
            "split": "train",
            "shots": [1, 2],
            "repeats": args.fit_repeats,
        },
        "eval": {
            "speakers": 4,
            "split": "dev",
            "shots": [1, 2],
            "repeats": args.dev_repeats,
        },
        "systems": {
            "global": "256D cosine prototype",
            "fixed_dtw": "Global Top3 + DTW fixed lambda=0.50",
            "learned": "shared linear logistic evidence ranker",
        },
        "feature_names": list(FEATURE_NAMES),
        "new_trainable_head_added": True,
        "head_type": "shared linear logistic candidate ranker",
        "backbone_neck_updated": False,
        "unknown_data_used": False,
        "rejector_used": False,
        "generic_test": "sealed_not_accessed",
        "summary": summary,
        "runs": results,
    }
    (args.output_dir / "h4_01_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("H4-01 SHARED EVIDENCE RANKER SUMMARY")
    print("=" * 108)
    for shot in SHOT_LEVELS:
        s = summary[str(shot)]
        print(
            f"{shot}-shot | "
            f"GLOBAL F1={s['global_macro_f1']['mean']:.6f} | "
            f"FIXED-DTW F1={s['fixed_dtw_macro_f1']['mean']:.6f} | "
            f"LEARNED F1={s['learned_macro_f1']['mean']:.6f} | "
            f"dLearned-Fixed={s['learned_minus_fixed_f1']['mean']:+.6f} | "
            f"better/equal/worse="
            f"{s['learned_better_than_fixed_repeats']}/"
            f"{s['learned_equal_fixed_repeats']}/"
            f"{s['learned_worse_than_fixed_repeats']}"
        )
    print("-" * 108)
    print("Task Head fitted:        YES")
    print("Head type:               shared linear logistic ranker")
    print("Backbone/Neck updated:   NO")
    print("Unknown/Rejector used:   NO")
    print("generic_test accessed:   NO")
    print("H4-01 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
