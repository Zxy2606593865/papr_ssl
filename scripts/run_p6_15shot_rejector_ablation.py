#!/usr/bin/env python
"""P6 15-shot lightweight rejector ablation.

Purpose
-------
Keep the Teacher, 64D embedding, 15-shot mean prototype, and intent prediction
completely frozen. Replace the hand-written 2D decision boundary
(score_threshold + margin_threshold) with a tiny learned KNOWN/UNKNOWN rejector.

This is a DEVELOPMENT ablation only. generic_test remains sealed.

Rejector input features (fixed):
    s1       = top-1 cosine similarity
    m        = top1 - top2 margin
    s1^2
    m^2
    s1*m

Model:
    balanced L2-regularized logistic regression

Training:
    calibration half only

Threshold:
    exact empirical threshold selection on calibration half only

Evaluation:
    held-out score half, using the same 20 repeated splits as the frozen
    15-shot stability diagnostic.

Important:
- Intent prediction is NOT changed.
- Teacher is NOT retrained.
- Prototype is NOT changed.
- This experiment isolates the rejection policy.
"""

from __future__ import annotations

import argparse
import json
import math
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


def stratified_known_split(true_labels: np.ndarray, seed: int):
    rng = random.Random(seed)
    cal, score = [], []
    for c in range(30):
        idx = np.flatnonzero(true_labels == c).tolist()
        rng.shuffle(idx)
        n_cal = len(idx) // 2
        if n_cal == 0 or n_cal == len(idx):
            raise RuntimeError(f"class {c} cannot be split")
        cal.extend(idx[:n_cal])
        score.extend(idx[n_cal:])
    return np.asarray(sorted(cal)), np.asarray(sorted(score))


def random_half_split(n: int, seed: int):
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_cal = n // 2
    return (
        np.asarray(sorted(idx[:n_cal])),
        np.asarray(sorted(idx[n_cal:])),
    )


def subset_known(all_data, idx):
    return {
        "score": all_data["score"][idx],
        "margin": all_data["margin"][idx],
        "pred": all_data["pred"][idx],
        "true": all_data["true"][idx],
    }


def subset_unknown(all_data, idx):
    return {
        "score": all_data["score"][idx],
        "margin": all_data["margin"][idx],
        "pred": all_data["pred"][idx],
    }


def feature_map(score: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """Fixed low-capacity feature map for KNOWN/UNKNOWN calibration."""
    s = np.asarray(score, dtype=np.float64)
    m = np.asarray(margin, dtype=np.float64)
    return np.stack(
        [s, m, s * s, m * m, s * m],
        axis=1,
    )


def fit_standardizer(x: np.ndarray):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def apply_standardizer(x: np.ndarray, mean, std):
    return (x - mean[None, :]) / std[None, :]


def fit_logistic_rejector(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1e-2,
    max_iter: int = 300,
):
    """Deterministic balanced logistic regression in torch.float64."""
    torch.manual_seed(0)

    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)

    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    n = float(len(y))
    if n_pos <= 0 or n_neg <= 0:
        raise RuntimeError("rejector needs both KNOWN and UNKNOWN samples")

    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    sample_w = torch.where(
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
            logits,
            yt,
            reduction="none",
        )
        loss = torch.mean(sample_w * bce) + l2 * torch.sum(w * w)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        logits = xt @ w + b
        probs = torch.sigmoid(logits)
        final_loss = float(closure().detach()) if False else float(
            torch.mean(
                sample_w
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yt, reduction="none"
                )
            )
            + l2 * torch.sum(w * w)
        )

    return {
        "w": w.detach().cpu().numpy(),
        "b": float(b.detach().cpu()),
        "train_loss": final_loss,
        "class_weight_known": w_pos,
        "class_weight_unknown": w_neg,
    }


def predict_prob(x: np.ndarray, model) -> np.ndarray:
    logits = x @ model["w"] + model["b"]
    # Stable sigmoid.
    out = np.empty_like(logits, dtype=np.float64)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    expx = np.exp(logits[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def macro_f1(accepted: np.ndarray, true: np.ndarray, pred: np.ndarray) -> float:
    f1s = []
    for c in range(30):
        pred_c = accepted & (pred == c)
        true_c = true == c
        tp = int(np.sum(pred_c & true_c))
        fp = int(np.sum(pred_c & ~true_c))
        fn = int(np.sum(~pred_c & true_c))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1s))


def metrics_from_accept(
    known,
    unknown,
    known_accept: np.ndarray,
    unknown_accept: np.ndarray,
):
    correct = known_accept & (known["pred"] == known["true"])
    wrong = known_accept & (known["pred"] != known["true"])
    return {
        "macro_f1": macro_f1(
            known_accept,
            known["true"],
            known["pred"],
        ),
        "correct_accept": float(np.mean(correct)),
        "wrong_intent": float(np.mean(wrong)),
        "known_reject": float(np.mean(~known_accept)),
        "unknown_reject": float(np.mean(~unknown_accept)),
        "unknown_accept_far": float(np.mean(unknown_accept)),
    }


def gate_pass(m: dict) -> bool:
    return (
        m["macro_f1"] >= GATES["macro_f1_min"]
        and m["correct_accept"] >= GATES["correct_accept_min"]
        and m["wrong_intent"] <= GATES["wrong_intent_max"]
        and m["known_reject"] <= GATES["known_reject_max"]
        and m["unknown_reject"] >= GATES["unknown_reject_min"]
    )


def gate_violation(m: dict) -> float:
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


def empirical_threshold_candidates(values: np.ndarray) -> np.ndarray:
    u = np.unique(np.asarray(values, dtype=np.float64))
    return np.concatenate(
        [
            np.asarray([np.nextafter(u[0], -np.inf)]),
            u,
            np.asarray([np.nextafter(u[-1], np.inf)]),
        ]
    )


def exact_probability_calibration(
    known,
    unknown,
    known_prob: np.ndarray,
    unknown_prob: np.ndarray,
):
    candidates = empirical_threshold_candidates(
        np.concatenate([known_prob, unknown_prob])
    )

    best_feasible = None
    best_feasible_key = None
    best_diag = None
    best_diag_key = None
    feasible_count = 0

    for tau in candidates:
        ka = known_prob >= tau
        ua = unknown_prob >= tau
        m = metrics_from_accept(known, unknown, ka, ua)
        if gate_pass(m):
            feasible_count += 1
            key = (
                m["macro_f1"],
                m["correct_accept"],
                m["unknown_reject"],
                -m["wrong_intent"],
                -m["known_reject"],
                float(tau),
            )
            if best_feasible_key is None or key > best_feasible_key:
                best_feasible_key = key
                best_feasible = {
                    "probability_threshold": float(tau),
                    "metrics": m,
                }
        else:
            v = gate_violation(m)
            key = (
                -v,
                m["macro_f1"],
                m["correct_accept"],
                m["unknown_reject"],
                -m["known_reject"],
                -m["wrong_intent"],
            )
            if best_diag_key is None or key > best_diag_key:
                best_diag_key = key
                best_diag = {
                    "probability_threshold": float(tau),
                    "metrics": m,
                    "normalized_gate_violation": float(v),
                }

    selected = best_feasible if best_feasible is not None else best_diag
    selected["calibration_feasible"] = best_feasible is not None
    selected["candidate_threshold_count"] = int(len(candidates))
    selected["feasible_threshold_count"] = int(feasible_count)
    return selected


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "sample_std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "artifacts/p6_15shot_stability/scores/"
            "attention_15shot_all_dev_scores.npz"
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
        default=Path("artifacts/p6_15shot_rejector"),
    )
    p.add_argument("--l2", type=float, default=1e-2)
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("baseline generic_test seal violated")

    num_splits = int(baseline["num_splits"])
    base_seed = int(baseline["base_seed"])

    d = np.load(args.scores, allow_pickle=False)
    known_all = {
        "score": d["known_score"].astype(np.float64),
        "margin": d["known_margin"].astype(np.float64),
        "pred": d["known_pred"].astype(np.int64),
        "true": d["known_true"].astype(np.int64),
    }
    unknown_all = {
        "score": d["unknown_score"].astype(np.float64),
        "margin": d["unknown_margin"].astype(np.float64),
        "pred": d["unknown_pred"].astype(np.int64),
    }

    if len(known_all["score"]) != 442:
        raise RuntimeError("expected 442 known dev")
    if len(unknown_all["score"]) != 1178:
        raise RuntimeError("expected 1178 unknown dev")

    runs = []

    for i in range(num_splits):
        seed = base_seed + i

        k_cal_idx, k_score_idx = stratified_known_split(
            known_all["true"], seed
        )
        u_cal_idx, u_score_idx = random_half_split(
            len(unknown_all["score"]),
            seed + 1_000_003,
        )

        k_cal = subset_known(known_all, k_cal_idx)
        k_score = subset_known(known_all, k_score_idx)
        u_cal = subset_unknown(unknown_all, u_cal_idx)
        u_score = subset_unknown(unknown_all, u_score_idx)

        xk_cal = feature_map(k_cal["score"], k_cal["margin"])
        xu_cal = feature_map(u_cal["score"], u_cal["margin"])
        x_cal = np.concatenate([xk_cal, xu_cal], axis=0)
        y_cal = np.concatenate(
            [
                np.ones(len(xk_cal), dtype=np.int64),
                np.zeros(len(xu_cal), dtype=np.int64),
            ],
            axis=0,
        )

        mean, std = fit_standardizer(x_cal)
        x_cal_n = apply_standardizer(x_cal, mean, std)

        model = fit_logistic_rejector(
            x_cal_n,
            y_cal,
            l2=args.l2,
        )

        pk_cal = predict_prob(
            apply_standardizer(xk_cal, mean, std),
            model,
        )
        pu_cal = predict_prob(
            apply_standardizer(xu_cal, mean, std),
            model,
        )

        calibrated = exact_probability_calibration(
            k_cal,
            u_cal,
            pk_cal,
            pu_cal,
        )

        xk_score = apply_standardizer(
            feature_map(k_score["score"], k_score["margin"]),
            mean,
            std,
        )
        xu_score = apply_standardizer(
            feature_map(u_score["score"], u_score["margin"]),
            mean,
            std,
        )
        pk_score = predict_prob(xk_score, model)
        pu_score = predict_prob(xu_score, model)

        tau = calibrated["probability_threshold"]
        score_metrics = metrics_from_accept(
            k_score,
            u_score,
            pk_score >= tau,
            pu_score >= tau,
        )
        passed = bool(
            calibrated["calibration_feasible"]
            and gate_pass(score_metrics)
        )

        baseline_run = baseline["runs"][i]
        if int(baseline_run["seed"]) != seed:
            raise RuntimeError("baseline split seed mismatch")

        base_m = baseline_run["score_metrics"]
        delta = {
            name: float(score_metrics[name] - base_m[name])
            for name in (
                "macro_f1",
                "correct_accept",
                "wrong_intent",
                "known_reject",
                "unknown_reject",
                "unknown_accept_far",
            )
        }

        run = {
            "split_index": i,
            "seed": seed,
            "calibration_feasible": bool(
                calibrated["calibration_feasible"]
            ),
            "feasible_threshold_count": int(
                calibrated["feasible_threshold_count"]
            ),
            "probability_threshold": float(tau),
            "rejector_train_loss": float(model["train_loss"]),
            "score_metrics": score_metrics,
            "absolute_gate_pass": passed,
            "delta_vs_two_threshold_baseline": delta,
        }
        runs.append(run)

        print(
            f"[split {i+1:02d}/{num_splits}] "
            f"cal_feasible={run['calibration_feasible']} "
            f"gate={'PASS' if passed else 'FAIL'} "
            f"F1={score_metrics['macro_f1']:.4f} "
            f"CA={score_metrics['correct_accept']:.4f} "
            f"KR={score_metrics['known_reject']:.4f} "
            f"UR={score_metrics['unknown_reject']:.4f} "
            f"| dCA={delta['correct_accept']:+.4f} "
            f"dUR={delta['unknown_reject']:+.4f}"
        )

    metrics = (
        "macro_f1",
        "correct_accept",
        "wrong_intent",
        "known_reject",
        "unknown_reject",
        "unknown_accept_far",
    )

    summary = {
        name: summarize([r["score_metrics"][name] for r in runs])
        for name in metrics
    }
    delta_summary = {
        name: summarize(
            [r["delta_vs_two_threshold_baseline"][name] for r in runs]
        )
        for name in metrics
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
        "schema": "papr_ssl.p6_15shot_rejector_ablation.v1",
        "phase": "P6",
        "purpose": (
            "development ablation of rejection policy with Teacher and "
            "prototype frozen"
        ),
        "frozen": {
            "teacher": "WavLM-large hidden_states[15]",
            "head": "AttentionDR 64D",
            "enrollment": "15-shot",
            "prototype": "mean_then_l2",
            "intent_prediction": "top1 cosine class unchanged",
        },
        "rejector": {
            "type": "balanced_l2_logistic_regression",
            "input_features": [
                "top1_cosine",
                "top1_top2_margin",
                "top1_cosine_squared",
                "margin_squared",
                "top1_cosine_x_margin",
            ],
            "standardization": "fit_on_calibration_only",
            "l2": float(args.l2),
            "probability_threshold": (
                "exact empirical calibration on calibration half only"
            ),
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
            "DEV has already been observed; this is a development ablation, "
            "not an independent final evaluation."
        ),
    }

    out_path = args.output_dir / "p6_15shot_rejector_results.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 15-SHOT LIGHTWEIGHT REJECTOR SUMMARY")
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
    print("P6 REJECTOR ABLATION STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
