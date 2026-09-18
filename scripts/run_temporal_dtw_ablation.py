#!/usr/bin/env python
"""20-split ablation: global 256D consistency vs global+DTW reranker.

This is an application-level DEV experiment. generic_test remains sealed.

Baseline:
    existing global 256D prototype + enrollment-consistency rejector

Candidate:
    same global 256D system
    + top-3 DTW reranking
    + 3 DTW-derived rejector features

Lambda is selected only on the known calibration half from a tiny frozen grid.
Tie-break prefers larger lambda, i.e. less DTW influence.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

LAMBDA_GRID = (0.50, 0.75, 1.00)
GATES = {
    "macro_f1_min": 0.80,
    "correct_accept_min": 0.80,
    "wrong_intent_max": 0.10,
    "known_reject_max": 0.20,
    "unknown_reject_min": 0.85,
}


def l2norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def consistency_features(query, enrollment, pred, primary_score, primary_margin):
    q = l2norm(query)
    e = l2norm(enrollment)
    pe = e[pred]
    sims = np.einsum("nd,nkd->nk", q, pe)
    ss = np.sort(sims, axis=1)

    mean15 = sims.mean(axis=1)
    std15 = sims.std(axis=1)
    min15 = ss[:, 0]
    q25 = np.quantile(sims, .25, axis=1)
    med = np.median(sims, axis=1)
    q75 = np.quantile(sims, .75, axis=1)
    max15 = ss[:, -1]
    top3 = ss[:, -3:].mean(axis=1)
    top5 = ss[:, -5:].mean(axis=1)
    bottom3 = ss[:, :3].mean(axis=1)

    return np.stack(
        [
            primary_score,
            primary_margin,
            mean15,
            std15,
            min15,
            q25,
            med,
            q75,
            max15,
            top3,
            top5,
            bottom3,
            primary_score - mean15,
            max15 - min15,
        ],
        axis=1,
    ).astype(np.float64)


def global_predictions(global_scores):
    order = np.argsort(global_scores, axis=1)
    pred = order[:, -1]
    second = order[:, -2]
    row = np.arange(len(global_scores))
    s1 = global_scores[row, pred]
    s2 = global_scores[row, second]
    return pred.astype(np.int64), s1, (s1 - s2)


def fused_predictions(topk_class, topk_global, topk_dtw, lam):
    fused = lam * topk_global + (1.0 - lam) * topk_dtw
    order = np.argsort(fused, axis=1)
    best_local = order[:, -1]
    second_local = order[:, -2]
    row = np.arange(len(fused))
    pred = topk_class[row, best_local]
    s1 = fused[row, best_local]
    s2 = fused[row, second_local]

    dtw_best = topk_dtw[row, best_local]
    dtw_second = topk_dtw[row, second_local]
    dtw_margin = dtw_best - dtw_second

    return (
        pred.astype(np.int64),
        s1.astype(np.float64),
        (s1 - s2).astype(np.float64),
        dtw_best.astype(np.float64),
        dtw_margin.astype(np.float64),
    )


def closed_macro_f1(true, pred):
    f = []
    for c in range(30):
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        den = 2 * tp + fp + fn
        f.append((2 * tp / den) if den else 0.0)
    return float(np.mean(f))


def choose_lambda(true_cal, topk_class, topk_global, topk_dtw):
    candidates = []
    for lam in LAMBDA_GRID:
        pred, *_ = fused_predictions(topk_class, topk_global, topk_dtw, lam)
        f1 = closed_macro_f1(true_cal, pred)
        candidates.append((f1, lam))
    # tie -> larger lambda -> more conservative / closer to current global Teacher
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return float(candidates[0][1]), candidates


def split_known(y, seed):
    rng = random.Random(seed)
    cal, score = [], []
    for c in range(30):
        idx = np.flatnonzero(y == c).tolist()
        rng.shuffle(idx)
        n = len(idx) // 2
        cal.extend(idx[:n])
        score.extend(idx[n:])
    return np.asarray(sorted(cal)), np.asarray(sorted(score))


def split_unknown(n, seed):
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    k = n // 2
    return np.asarray(sorted(idx[:k])), np.asarray(sorted(idx[k:]))


def standardizer(x):
    m = x.mean(0)
    s = x.std(0)
    return m, np.where(s < 1e-12, 1.0, s)


def zscore(x, m, s):
    return (x - m) / s


def fit_logistic(x, y, l2=1e-2):
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)

    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    n = float(len(y))
    wp = n / (2 * n_pos)
    wn = n / (2 * n_neg)
    sw = torch.where(
        yt > .5,
        torch.full_like(yt, wp),
        torch.full_like(yt, wn),
    )

    w = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)

    opt = torch.optim.LBFGS(
        [w, b],
        lr=1.0,
        max_iter=300,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = xt @ w + b
        loss = (
            sw
            * torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yt, reduction="none"
            )
        ).mean() + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def sigmoid(x, w, b):
    a = x @ w + b
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    e = np.exp(a[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def macro_f1_open(accepted, true, pred):
    f = []
    for c in range(30):
        pc = accepted & (pred == c)
        tc = true == c
        tp = int(np.sum(pc & tc))
        fp = int(np.sum(pc & ~tc))
        fn = int(np.sum(~pc & tc))
        den = 2 * tp + fp + fn
        f.append((2 * tp / den) if den else 0.0)
    return float(np.mean(f))


def metrics(true, pred, kp, up, tau):
    ka = kp >= tau
    ua = up >= tau
    correct = ka & (pred == true)
    wrong = ka & (pred != true)
    return {
        "macro_f1": macro_f1_open(ka, true, pred),
        "correct_accept": float(correct.mean()),
        "wrong_intent": float(wrong.mean()),
        "known_reject": float((~ka).mean()),
        "unknown_reject": float((~ua).mean()),
        "unknown_accept_far": float(ua.mean()),
    }


def gate(m):
    return (
        m["macro_f1"] >= GATES["macro_f1_min"]
        and m["correct_accept"] >= GATES["correct_accept_min"]
        and m["wrong_intent"] <= GATES["wrong_intent_max"]
        and m["known_reject"] <= GATES["known_reject_max"]
        and m["unknown_reject"] >= GATES["unknown_reject_min"]
    )


def viol(m):
    return (
        max(0, GATES["macro_f1_min"] - m["macro_f1"]) / GATES["macro_f1_min"]
        + max(0, GATES["correct_accept_min"] - m["correct_accept"]) / GATES["correct_accept_min"]
        + max(0, m["wrong_intent"] - GATES["wrong_intent_max"]) / GATES["wrong_intent_max"]
        + max(0, m["known_reject"] - GATES["known_reject_max"]) / GATES["known_reject_max"]
        + max(0, GATES["unknown_reject_min"] - m["unknown_reject"]) / GATES["unknown_reject_min"]
    )


def calibrate(true, pred, kp, up):
    u = np.unique(np.concatenate([kp, up]))
    cand = np.concatenate(
        [[np.nextafter(u[0], -np.inf)], u, [np.nextafter(u[-1], np.inf)]]
    )
    bestf = None
    bestfk = None
    bestd = None
    bestdk = None

    for tau in cand:
        m = metrics(true, pred, kp, up, float(tau))
        if gate(m):
            key = (
                m["macro_f1"],
                m["correct_accept"],
                m["unknown_reject"],
                -m["wrong_intent"],
                -m["known_reject"],
                tau,
            )
            if bestfk is None or key > bestfk:
                bestfk = key
                bestf = (True, float(tau))
        else:
            v = viol(m)
            key = (
                -v,
                m["macro_f1"],
                m["correct_accept"],
                m["unknown_reject"],
            )
            if bestdk is None or key > bestdk:
                bestdk = key
                bestd = (False, float(tau))

    return bestf if bestf is not None else bestd


def prepare_arm_features(
    enrollment,
    known_emb,
    unknown_emb,
    global_scores_k,
    global_scores_u,
    evidence,
    lam,
    arm,
):
    if arm == "global":
        kp, ks1, kmargin = global_predictions(global_scores_k)
        up, us1, umargin = global_predictions(global_scores_u)

        kx = consistency_features(known_emb, enrollment, kp, ks1, kmargin)
        ux = consistency_features(unknown_emb, enrollment, up, us1, umargin)
        return kp, up, kx, ux

    if arm == "dtw":
        kp, ks1, kmargin, kdtw, kdtwm = fused_predictions(
            evidence["known_topk_class"],
            evidence["known_topk_global"],
            evidence["known_topk_dtw"],
            lam,
        )
        up, us1, umargin, udtw, udtwm = fused_predictions(
            evidence["unknown_topk_class"],
            evidence["unknown_topk_global"],
            evidence["unknown_topk_dtw"],
            lam,
        )

        kx = consistency_features(known_emb, enrollment, kp, ks1, kmargin)
        ux = consistency_features(unknown_emb, enrollment, up, us1, umargin)

        # only three extra scalar features; no new neural network.
        kextra = np.stack(
            [kdtw, kdtwm, ks1 - kdtw],
            axis=1,
        )
        uextra = np.stack(
            [udtw, udtwm, us1 - udtw],
            axis=1,
        )
        return kp, up, np.concatenate([kx, kextra], axis=1), np.concatenate([ux, uextra], axis=1)

    raise ValueError(arm)


def summarize(vals):
    x = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "sample_std": float(x.std(ddof=1)),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
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
        "--dtw-evidence",
        type=Path,
        default=Path(
            "artifacts/p6_temporal_dtw/precomputed/"
            "dtw_rerank_evidence.npz"
        ),
    )
    p.add_argument(
        "--split-policy",
        type=Path,
        default=Path(
            "artifacts/p6_15shot_stability/repeated_splits/"
            "p6_15shot_stability.json"
        ),
    )
    p.add_argument(
        "--reference256",
        type=Path,
        default=Path(
            "artifacts/p6_teacher_256_15shot/repeated_splits/"
            "p6_teacher_256_vs_64_15shot.json"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_temporal_dtw/evaluation"),
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splitp = json.loads(args.split_policy.read_text(encoding="utf-8"))
    ref256 = json.loads(args.reference256.read_text(encoding="utf-8"))
    if splitp["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("split policy test seal violated")
    if ref256["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("256D reference test seal violated")

    d = np.load(args.embedding_npz, allow_pickle=False)
    enrollment = d["enrollment"].astype(np.float64)
    known_emb = d["known_embedding"].astype(np.float64)
    known_true = d["known_true"].astype(np.int64)
    unknown_emb = d["unknown_embedding"].astype(np.float64)

    e = np.load(args.dtw_evidence, allow_pickle=False)
    evidence = {k: e[k] for k in e.files}
    global_scores_k = e["known_global_scores"].astype(np.float64)
    global_scores_u = e["unknown_global_scores"].astype(np.float64)

    n = int(splitp["num_splits"])
    base = int(splitp["base_seed"])
    if n != 20:
        raise RuntimeError(f"Expected 20 splits, got {n}")

    global_runs = []
    dtw_runs = []

    for i in range(n):
        seed = base + i
        kc, ks = split_known(known_true, seed)
        uc, us = split_unknown(len(unknown_emb), seed + 1_000_003)

        # conservative lambda selection using ONLY known calibration labels.
        lambda_candidates = []
        for lam in LAMBDA_GRID:
            pred_all, *_ = fused_predictions(
                evidence["known_topk_class"][kc],
                evidence["known_topk_global"][kc],
                evidence["known_topk_dtw"][kc],
                lam,
            )
            lambda_candidates.append(
                (closed_macro_f1(known_true[kc], pred_all), lam)
            )
        lambda_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        lam = float(lambda_candidates[0][1])

        for arm, runs in (("global", global_runs), ("dtw", dtw_runs)):
            use_lam = 1.0 if arm == "global" else lam
            kp, up, kx, ux = prepare_arm_features(
                enrollment,
                known_emb,
                unknown_emb,
                global_scores_k,
                global_scores_u,
                evidence,
                use_lam,
                arm,
            )

            xcal = np.concatenate([kx[kc], ux[uc]], axis=0)
            ycal = np.concatenate(
                [
                    np.ones(len(kc), dtype=np.int64),
                    np.zeros(len(uc), dtype=np.int64),
                ]
            )

            m, s = standardizer(xcal)
            w, b = fit_logistic(zscore(xcal, m, s), ycal)

            pkc = sigmoid(zscore(kx[kc], m, s), w, b)
            puc = sigmoid(zscore(ux[uc], m, s), w, b)
            feasible, tau = calibrate(
                known_true[kc], kp[kc], pkc, puc
            )

            pks = sigmoid(zscore(kx[ks], m, s), w, b)
            pus = sigmoid(zscore(ux[us], m, s), w, b)

            met = metrics(
                known_true[ks],
                kp[ks],
                pks,
                pus,
                tau,
            )

            runs.append(
                {
                    "split_index": i,
                    "seed": seed,
                    "arm": arm,
                    "lambda": use_lam,
                    "lambda_calibration_candidates": lambda_candidates if arm == "dtw" else None,
                    "calibration_feasible": bool(feasible),
                    "probability_threshold": float(tau),
                    "score_metrics": met,
                    "absolute_gate_pass": bool(feasible and gate(met)),
                }
            )

        gm = global_runs[-1]["score_metrics"]
        dm = dtw_runs[-1]["score_metrics"]
        print(
            f"[split {i+1:02d}/{n}] lambda={lam:.2f} | "
            f"GLOBAL F1={gm['macro_f1']:.4f} CA={gm['correct_accept']:.4f} UR={gm['unknown_reject']:.4f} | "
            f"DTW F1={dm['macro_f1']:.4f} CA={dm['correct_accept']:.4f} UR={dm['unknown_reject']:.4f} | "
            f"dF1={dm['macro_f1']-gm['macro_f1']:+.4f}"
        )

    names = (
        "macro_f1",
        "correct_accept",
        "wrong_intent",
        "known_reject",
        "unknown_reject",
        "unknown_accept_far",
    )

    global_summary = {
        k: summarize([r["score_metrics"][k] for r in global_runs])
        for k in names
    }
    dtw_summary = {
        k: summarize([r["score_metrics"][k] for r in dtw_runs])
        for k in names
    }
    delta_summary = {
        k: summarize(
            [
                dr["score_metrics"][k] - gr["score_metrics"][k]
                for gr, dr in zip(global_runs, dtw_runs)
            ]
        )
        for k in names
    }

    lambda_hist = {
        str(lam): int(sum(r["lambda"] == lam for r in dtw_runs))
        for lam in LAMBDA_GRID
    }

    out = {
        "schema": "papr_ssl.temporal_dtw_ablation.v1",
        "lambda_grid": list(LAMBDA_GRID),
        "lambda_selection": "known calibration Macro-F1 only; ties prefer larger lambda",
        "global_summary": global_summary,
        "dtw_summary": dtw_summary,
        "dtw_minus_global_summary": delta_summary,
        "global_gate_pass_rate": float(np.mean([r["absolute_gate_pass"] for r in global_runs])),
        "dtw_gate_pass_rate": float(np.mean([r["absolute_gate_pass"] for r in dtw_runs])),
        "global_calibration_feasible_rate": float(np.mean([r["calibration_feasible"] for r in global_runs])),
        "dtw_calibration_feasible_rate": float(np.mean([r["calibration_feasible"] for r in dtw_runs])),
        "lambda_histogram": lambda_hist,
        "reference_current_256d_summary": ref256["score_metric_summary"],
        "global_runs": global_runs,
        "dtw_runs": dtw_runs,
        "redundancy_control": {
            "second_encoder": False,
            "dtw_top_k": 3,
            "templates_per_class": 2,
            "extra_rejector_features": 3,
        },
        "canonical_64d_teacher": "preserved_not_modified",
        "project1_256d_teacher": "preserved_not_modified",
        "generic_test": "sealed_not_accessed",
    }

    (args.output_dir / "temporal_dtw_ablation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("TEMPORAL DTW RERANKER ABLATION SUMMARY")
    print("=" * 108)
    for k in (
        "macro_f1",
        "correct_accept",
        "wrong_intent",
        "known_reject",
        "unknown_reject",
    ):
        print(
            f"{k:20s} "
            f"global={global_summary[k]['mean']:.6f} "
            f"dtw={dtw_summary[k]['mean']:.6f} "
            f"delta={delta_summary[k]['mean']:+.6f}"
        )
    print("-" * 108)
    print(f"global Gate pass rate: {out['global_gate_pass_rate']:.3f}")
    print(f"DTW Gate pass rate:    {out['dtw_gate_pass_rate']:.3f}")
    print(f"lambda histogram:      {lambda_hist}")
    print("second encoder added:  NO")
    print("generic_test accessed: NO")
    print("EVALUATION STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
