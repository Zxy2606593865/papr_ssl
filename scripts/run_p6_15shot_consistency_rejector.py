#!/usr/bin/env python
"""15-shot enrollment-consistency rejector repeated-split ablation.

This experiment keeps the Teacher and intent prediction frozen, but uses the
FULL 15-shot enrollment distribution rather than only the mean prototype.

For each query:
1) predict the intent from the 15-shot mean prototype, unchanged;
2) compute similarity to all 15 enrollment embeddings of that predicted class;
3) build a low-dimensional consistency feature vector;
4) train a tiny balanced logistic rejector on the calibration half only;
5) choose one exact empirical KNOWN probability threshold on calibration only;
6) evaluate on the held-out score half.

generic_test remains sealed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


GATES = {
    "macro_f1_min": 0.80,
    "correct_accept_min": 0.80,
    "wrong_intent_max": 0.10,
    "known_reject_max": 0.20,
    "unknown_reject_min": 0.85,
}


def l2norm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def compute_query_features(query, enrollment):
    """Return predicted class plus enrollment-consistency features.

    query: [N,64]
    enrollment: [30,15,64]
    """
    q = l2norm(query)
    e = l2norm(enrollment)
    centroids = l2norm(np.mean(e, axis=1))  # [30,64]

    class_scores = q @ centroids.T
    top2_idx = np.argsort(class_scores, axis=1)[:, -2:]
    pred = top2_idx[:, 1]
    second = top2_idx[:, 0]

    row = np.arange(len(q))
    s1 = class_scores[row, pred]
    s2 = class_scores[row, second]
    margin = s1 - s2

    # Similarity to all 15 enrollment utterances of the predicted class.
    pred_enrollment = e[pred]  # [N,15,64]
    sims = np.einsum("nd,nkd->nk", q, pred_enrollment)  # [N,15]
    sims_sorted = np.sort(sims, axis=1)

    mean15 = np.mean(sims, axis=1)
    std15 = np.std(sims, axis=1)
    min15 = sims_sorted[:, 0]
    q25 = np.quantile(sims, 0.25, axis=1)
    median15 = np.median(sims, axis=1)
    q75 = np.quantile(sims, 0.75, axis=1)
    max15 = sims_sorted[:, -1]
    top3mean = np.mean(sims_sorted[:, -3:], axis=1)
    top5mean = np.mean(sims_sorted[:, -5:], axis=1)
    bottom3mean = np.mean(sims_sorted[:, :3], axis=1)

    # Low-capacity fixed feature vector; no class identity is supplied.
    x = np.stack(
        [
            s1,
            margin,
            mean15,
            std15,
            min15,
            q25,
            median15,
            q75,
            max15,
            top3mean,
            top5mean,
            bottom3mean,
            s1 - mean15,
            max15 - min15,
        ],
        axis=1,
    )

    return pred.astype(np.int64), x.astype(np.float64)


def stratified_known_split(true_labels, seed):
    rng = random.Random(seed)
    cal, score = [], []
    for c in range(30):
        idx = np.flatnonzero(true_labels == c).tolist()
        rng.shuffle(idx)
        n_cal = len(idx) // 2
        cal.extend(idx[:n_cal])
        score.extend(idx[n_cal:])
    return np.asarray(sorted(cal)), np.asarray(sorted(score))


def random_half_split(n, seed):
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_cal = n // 2
    return (
        np.asarray(sorted(idx[:n_cal])),
        np.asarray(sorted(idx[n_cal:])),
    )


def fit_standardizer(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def standardize(x, mean, std):
    return (x - mean[None, :]) / std[None, :]


def fit_logistic(x, y, l2=1e-2, max_iter=300):
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)

    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    n = float(len(y))
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    sw = torch.where(
        yt > 0.5,
        torch.full_like(yt, w_pos),
        torch.full_like(yt, w_neg),
    )

    w = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [w, b],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = xt @ w + b
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, yt, reduction="none"
        )
        loss = torch.mean(sw * bce) + l2 * torch.sum(w * w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def sigmoid_logits(x, w, b):
    logits = x @ w + b
    out = np.empty_like(logits, dtype=np.float64)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    e = np.exp(logits[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def macro_f1(accepted, true, pred):
    f1s = []
    for c in range(30):
        pc = accepted & (pred == c)
        tc = true == c
        tp = int(np.sum(pc & tc))
        fp = int(np.sum(pc & ~tc))
        fn = int(np.sum(~pc & tc))
        den = 2 * tp + fp + fn
        f1s.append((2 * tp / den) if den else 0.0)
    return float(np.mean(f1s))


def metrics(known_true, known_pred, kp, up, tau):
    ka = kp >= tau
    ua = up >= tau
    correct = ka & (known_pred == known_true)
    wrong = ka & (known_pred != known_true)
    return {
        "macro_f1": macro_f1(ka, known_true, known_pred),
        "correct_accept": float(np.mean(correct)),
        "wrong_intent": float(np.mean(wrong)),
        "known_reject": float(np.mean(~ka)),
        "unknown_reject": float(np.mean(~ua)),
        "unknown_accept_far": float(np.mean(ua)),
    }


def gate_pass(m):
    return (
        m["macro_f1"] >= GATES["macro_f1_min"]
        and m["correct_accept"] >= GATES["correct_accept_min"]
        and m["wrong_intent"] <= GATES["wrong_intent_max"]
        and m["known_reject"] <= GATES["known_reject_max"]
        and m["unknown_reject"] >= GATES["unknown_reject_min"]
    )


def violation(m):
    return (
        max(0.0, GATES["macro_f1_min"] - m["macro_f1"])
        / GATES["macro_f1_min"]
        + max(0.0, GATES["correct_accept_min"] - m["correct_accept"])
        / GATES["correct_accept_min"]
        + max(0.0, m["wrong_intent"] - GATES["wrong_intent_max"])
        / GATES["wrong_intent_max"]
        + max(0.0, m["known_reject"] - GATES["known_reject_max"])
        / GATES["known_reject_max"]
        + max(0.0, GATES["unknown_reject_min"] - m["unknown_reject"])
        / GATES["unknown_reject_min"]
    )


def threshold_candidates(values):
    u = np.unique(np.asarray(values, dtype=np.float64))
    return np.concatenate(
        [
            np.asarray([np.nextafter(u[0], -np.inf)]),
            u,
            np.asarray([np.nextafter(u[-1], np.inf)]),
        ]
    )


def calibrate(known_true, known_pred, kp, up):
    candidates = threshold_candidates(np.concatenate([kp, up]))
    best_f = None
    best_f_key = None
    best_d = None
    best_d_key = None
    feasible = 0

    for tau in candidates:
        m = metrics(known_true, known_pred, kp, up, float(tau))
        if gate_pass(m):
            feasible += 1
            key = (
                m["macro_f1"],
                m["correct_accept"],
                m["unknown_reject"],
                -m["wrong_intent"],
                -m["known_reject"],
                float(tau),
            )
            if best_f_key is None or key > best_f_key:
                best_f_key = key
                best_f = (float(tau), m)
        else:
            v = violation(m)
            key = (
                -v,
                m["macro_f1"],
                m["correct_accept"],
                m["unknown_reject"],
            )
            if best_d_key is None or key > best_d_key:
                best_d_key = key
                best_d = (float(tau), m, v)

    if best_f is not None:
        tau, m = best_f
        return {
            "calibration_feasible": True,
            "probability_threshold": tau,
            "calibration_metrics": m,
            "feasible_threshold_count": feasible,
            "candidate_threshold_count": len(candidates),
        }

    tau, m, v = best_d
    return {
        "calibration_feasible": False,
        "probability_threshold": tau,
        "calibration_metrics": m,
        "normalized_gate_violation": float(v),
        "feasible_threshold_count": 0,
        "candidate_threshold_count": len(candidates),
    }


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "sample_std": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embeddings",
        type=Path,
        default=Path(
            "artifacts/p6_15shot_consistency/embeddings/"
            "attention_15shot_consistency_embeddings.npz"
        ),
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "artifacts/p6_15shot_stability/repeated_splits/"
            "p6_15shot_stability.json"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/p6_15shot_consistency/repeated_splits"
        ),
    )
    p.add_argument("--l2", type=float, default=1e-2)
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("baseline generic_test seal violated")

    d = np.load(args.embeddings, allow_pickle=False)
    enrollment = d["enrollment"].astype(np.float64)
    known_embedding = d["known_embedding"].astype(np.float64)
    known_true = d["known_true"].astype(np.int64)
    unknown_embedding = d["unknown_embedding"].astype(np.float64)

    known_pred, known_x = compute_query_features(
        known_embedding, enrollment
    )
    unknown_pred, unknown_x = compute_query_features(
        unknown_embedding, enrollment
    )

    num_splits = int(baseline["num_splits"])
    base_seed = int(baseline["base_seed"])
    runs = []

    for i in range(num_splits):
        seed = base_seed + i
        k_cal, k_score = stratified_known_split(known_true, seed)
        u_cal, u_score = random_half_split(
            len(unknown_embedding), seed + 1_000_003
        )

        x_cal = np.concatenate([known_x[k_cal], unknown_x[u_cal]], axis=0)
        y_cal = np.concatenate(
            [
                np.ones(len(k_cal), dtype=np.int64),
                np.zeros(len(u_cal), dtype=np.int64),
            ],
            axis=0,
        )
        mean, std = fit_standardizer(x_cal)
        x_cal_n = standardize(x_cal, mean, std)

        w, b = fit_logistic(x_cal_n, y_cal, l2=args.l2)
        kp_cal = sigmoid_logits(standardize(known_x[k_cal], mean, std), w, b)
        up_cal = sigmoid_logits(standardize(unknown_x[u_cal], mean, std), w, b)

        cal = calibrate(
            known_true[k_cal],
            known_pred[k_cal],
            kp_cal,
            up_cal,
        )

        kp_score = sigmoid_logits(
            standardize(known_x[k_score], mean, std), w, b
        )
        up_score = sigmoid_logits(
            standardize(unknown_x[u_score], mean, std), w, b
        )
        m = metrics(
            known_true[k_score],
            known_pred[k_score],
            kp_score,
            up_score,
            cal["probability_threshold"],
        )
        passed = bool(cal["calibration_feasible"] and gate_pass(m))

        base_run = baseline["runs"][i]
        if int(base_run["seed"]) != seed:
            raise RuntimeError("baseline split mismatch")
        base_m = base_run["score_metrics"]

        delta = {
            name: float(m[name] - base_m[name])
            for name in (
                "macro_f1",
                "correct_accept",
                "wrong_intent",
                "known_reject",
                "unknown_reject",
                "unknown_accept_far",
            )
        }

        runs.append(
            {
                "split_index": i,
                "seed": seed,
                "calibration_feasible": bool(
                    cal["calibration_feasible"]
                ),
                "feasible_threshold_count": int(
                    cal["feasible_threshold_count"]
                ),
                "probability_threshold": float(
                    cal["probability_threshold"]
                ),
                "score_metrics": m,
                "absolute_gate_pass": passed,
                "delta_vs_two_threshold_baseline": delta,
            }
        )

        print(
            f"[split {i+1:02d}/{num_splits}] "
            f"cal_feasible={cal['calibration_feasible']} "
            f"gate={'PASS' if passed else 'FAIL'} "
            f"F1={m['macro_f1']:.4f} "
            f"CA={m['correct_accept']:.4f} "
            f"KR={m['known_reject']:.4f} "
            f"UR={m['unknown_reject']:.4f} "
            f"| dCA={delta['correct_accept']:+.4f} "
            f"dUR={delta['unknown_reject']:+.4f}"
        )

    names = (
        "macro_f1",
        "correct_accept",
        "wrong_intent",
        "known_reject",
        "unknown_reject",
        "unknown_accept_far",
    )

    summary = {
        name: summarize([r["score_metrics"][name] for r in runs])
        for name in names
    }
    delta_summary = {
        name: summarize(
            [r["delta_vs_two_threshold_baseline"][name] for r in runs]
        )
        for name in names
    }

    component_rates = {
        "macro_f1": float(np.mean([
            r["score_metrics"]["macro_f1"] >= GATES["macro_f1_min"]
            for r in runs
        ])),
        "correct_accept": float(np.mean([
            r["score_metrics"]["correct_accept"] >= GATES["correct_accept_min"]
            for r in runs
        ])),
        "wrong_intent": float(np.mean([
            r["score_metrics"]["wrong_intent"] <= GATES["wrong_intent_max"]
            for r in runs
        ])),
        "known_reject": float(np.mean([
            r["score_metrics"]["known_reject"] <= GATES["known_reject_max"]
            for r in runs
        ])),
        "unknown_reject": float(np.mean([
            r["score_metrics"]["unknown_reject"] >= GATES["unknown_reject_min"]
            for r in runs
        ])),
    }

    output = {
        "schema": "papr_ssl.p6_15shot_consistency_rejector.v1",
        "phase": "P6",
        "frozen": {
            "teacher": "WavLM-large hidden_states[15]",
            "head": "AttentionDR 64D",
            "enrollment": "15-shot",
            "intent_prediction": "mean-prototype top1 cosine unchanged",
        },
        "rejector": {
            "type": "balanced_l2_logistic_regression",
            "feature_source": (
                "full 15-shot enrollment-consistency statistics"
            ),
            "feature_count": int(known_x.shape[1]),
            "l2": float(args.l2),
        },
        "num_splits": num_splits,
        "base_seed": base_seed,
        "absolute_gates": GATES,
        "calibration_feasible_rate": float(np.mean([
            r["calibration_feasible"] for r in runs
        ])),
        "absolute_gate_pass_count": int(sum(
            r["absolute_gate_pass"] for r in runs
        )),
        "absolute_gate_pass_rate": float(np.mean([
            r["absolute_gate_pass"] for r in runs
        ])),
        "component_gate_pass_rates": component_rates,
        "score_metric_summary": summary,
        "delta_vs_two_threshold_baseline_summary": delta_summary,
        "runs": runs,
        "generic_test": "sealed_not_accessed",
        "interpretation_note": (
            "DEV has already been observed; development ablation only."
        ),
    }

    out_path = args.output_dir / "p6_15shot_consistency_results.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 15-SHOT ENROLLMENT-CONSISTENCY REJECTOR SUMMARY")
    print("=" * 108)
    print(
        f"calibration feasible: "
        f"{sum(r['calibration_feasible'] for r in runs)}/{num_splits} "
        f"({output['calibration_feasible_rate']:.3f})"
    )
    print(
        f"absolute Gate PASS:   "
        f"{output['absolute_gate_pass_count']}/{num_splits} "
        f"({output['absolute_gate_pass_rate']:.3f})"
    )
    print("-" * 108)

    for name in (
        "macro_f1",
        "correct_accept",
        "wrong_intent",
        "known_reject",
        "unknown_reject",
    ):
        s = summary[name]
        ds = delta_summary[name]
        print(
            f"{name:20s} "
            f"mean={s['mean']:.6f} "
            f"std={s['sample_std']:.6f} "
            f"delta_vs_baseline={ds['mean']:+.6f}"
        )

    print("-" * 108)
    print("component Gate pass rates:")
    for name, rate in component_rates.items():
        print(f"{name:20s} {rate:.3f}")

    print("-" * 108)
    print("generic_test accessed: NO")
    print("P6 CONSISTENCY REJECTOR STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
