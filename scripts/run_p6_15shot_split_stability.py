#!/usr/bin/env python
"""Repeated split stability diagnostic for frozen 15-shot mean prototype.

Default:
- 20 deterministic repeated 50/50 splits of the already-observed DEV pool
- known DEV split is class-stratified
- unknown DEV split is deterministic random 50/50
- exact empirical threshold calibration on the calibration half
- evaluation on the held-out score half

This is NOT a new untouched Gate because DEV has already been observed in prior
P6 development iterations. It is a stability diagnostic only.

generic_test remains sealed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

GATES = {
    "macro_f1_min": 0.80,
    "correct_accept_min": 0.80,
    "wrong_intent_max": 0.10,
    "known_reject_max": 0.20,
    "unknown_reject_min": 0.85,
}


def empirical_candidates(values: np.ndarray) -> np.ndarray:
    u = np.unique(np.asarray(values, dtype=np.float64))
    return np.concatenate(
        [
            np.asarray([np.nextafter(u[0], -np.inf)]),
            u,
            np.asarray([np.nextafter(u[-1], np.inf)]),
        ]
    )


def macro_f1_vectorized(accepted, true, pred):
    total = np.zeros(accepted.shape[0], dtype=np.float64)
    for c in range(30):
        tc = true == c
        pc = pred == c
        tp = np.sum(accepted & tc[None, :] & pc[None, :], axis=1)
        fp = np.sum(accepted & (~tc)[None, :] & pc[None, :], axis=1)
        fn = int(np.sum(tc)) - tp
        denom = 2.0 * tp + fp + fn
        total += np.divide(
            2.0 * tp,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0,
        )
    return total / 30.0


def metric_vectors(known, unknown, ts, margin_candidates):
    k_score = known["score"] >= ts
    u_score = unknown["score"] >= ts
    ka = (
        known["margin"][None, :] >= margin_candidates[:, None]
    ) & k_score[None, :]
    ua = (
        unknown["margin"][None, :] >= margin_candidates[:, None]
    ) & u_score[None, :]

    correct = known["pred"] == known["true"]
    wrong = ~correct
    return {
        "macro_f1": macro_f1_vectorized(ka, known["true"], known["pred"]),
        "correct_accept": np.mean(ka & correct[None, :], axis=1),
        "wrong_intent": np.mean(ka & wrong[None, :], axis=1),
        "known_reject": 1.0 - np.mean(ka, axis=1),
        "unknown_reject": 1.0 - np.mean(ua, axis=1),
        "unknown_accept_far": np.mean(ua, axis=1),
    }


def feasible_mask(v):
    return (
        (v["macro_f1"] >= GATES["macro_f1_min"])
        & (v["correct_accept"] >= GATES["correct_accept_min"])
        & (v["wrong_intent"] <= GATES["wrong_intent_max"])
        & (v["known_reject"] <= GATES["known_reject_max"])
        & (v["unknown_reject"] >= GATES["unknown_reject_min"])
    )


def violation_vector(v):
    return (
        np.maximum(0.0, GATES["macro_f1_min"] - v["macro_f1"])
        / GATES["macro_f1_min"]
        + np.maximum(
            0.0,
            GATES["correct_accept_min"] - v["correct_accept"],
        )
        / GATES["correct_accept_min"]
        + np.maximum(
            0.0,
            v["wrong_intent"] - GATES["wrong_intent_max"],
        )
        / GATES["wrong_intent_max"]
        + np.maximum(
            0.0,
            v["known_reject"] - GATES["known_reject_max"],
        )
        / GATES["known_reject_max"]
        + np.maximum(
            0.0,
            GATES["unknown_reject_min"] - v["unknown_reject"],
        )
        / GATES["unknown_reject_min"]
    )


def scalar_metrics(known, unknown, ts, tm):
    ka = (known["score"] >= ts) & (known["margin"] >= tm)
    ua = (unknown["score"] >= ts) & (unknown["margin"] >= tm)
    correct = ka & (known["pred"] == known["true"])
    wrong = ka & (known["pred"] != known["true"])

    f1s = []
    for c in range(30):
        pred_c = ka & (known["pred"] == c)
        true_c = known["true"] == c
        tp = int(np.sum(pred_c & true_c))
        fp = int(np.sum(pred_c & ~true_c))
        fn = int(np.sum(~pred_c & true_c))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)

    return {
        "macro_f1": float(np.mean(f1s)),
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


def exact_calibrate(known, unknown):
    score_candidates = empirical_candidates(
        np.concatenate([known["score"], unknown["score"]])
    )
    margin_candidates = empirical_candidates(
        np.concatenate([known["margin"], unknown["margin"]])
    )

    best_feasible = None
    best_feasible_key = None
    best_diag = None
    best_diag_key = None
    feasible_count = 0

    for ts in score_candidates:
        v = metric_vectors(known, unknown, float(ts), margin_candidates)
        ok = feasible_mask(v)
        idxs = np.flatnonzero(ok)
        feasible_count += int(idxs.size)

        for j in idxs:
            key = (
                float(v["macro_f1"][j]),
                float(v["correct_accept"][j]),
                float(v["unknown_reject"][j]),
                -float(v["wrong_intent"][j]),
                -float(v["known_reject"][j]),
                float(ts),
                float(margin_candidates[j]),
            )
            if best_feasible_key is None or key > best_feasible_key:
                best_feasible_key = key
                best_feasible = {
                    "score_threshold": float(ts),
                    "margin_threshold": float(margin_candidates[j]),
                }

        if best_feasible is None:
            vio = violation_vector(v)
            j = int(np.argmin(vio))
            key = (
                -float(vio[j]),
                float(v["macro_f1"][j]),
                float(v["correct_accept"][j]),
                float(v["unknown_reject"][j]),
            )
            if best_diag_key is None or key > best_diag_key:
                best_diag_key = key
                best_diag = {
                    "score_threshold": float(ts),
                    "margin_threshold": float(margin_candidates[j]),
                    "normalized_gate_violation": float(vio[j]),
                }

    selected = best_feasible if best_feasible is not None else best_diag
    selected["calibration_feasible"] = best_feasible is not None
    selected["feasible_pair_count"] = int(feasible_count)
    return selected


def stratified_known_split(true_labels, seed):
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


def random_half_split(n, seed):
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
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/p6_15shot_stability/scores/manifest.json"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_15shot_stability/repeated_splits"),
    )
    p.add_argument("--num-splits", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=20260911)
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("test seal violated")
    if int(manifest["shots_per_intent"]) != 15:
        raise RuntimeError("expected 15-shot score export")

    d = np.load(args.scores, allow_pickle=False)
    known_all = {
        "score": d["known_score"],
        "margin": d["known_margin"],
        "pred": d["known_pred"],
        "true": d["known_true"],
    }
    unknown_all = {
        "score": d["unknown_score"],
        "margin": d["unknown_margin"],
        "pred": d["unknown_pred"],
    }

    if len(known_all["score"]) != 442:
        raise RuntimeError("expected 442 known dev")
    if len(unknown_all["score"]) != 1178:
        raise RuntimeError("expected 1178 unknown dev")

    runs = []
    for i in range(args.num_splits):
        seed = args.base_seed + i
        k_cal_idx, k_score_idx = stratified_known_split(
            known_all["true"],
            seed,
        )
        u_cal_idx, u_score_idx = random_half_split(
            len(unknown_all["score"]),
            seed + 1_000_003,
        )

        k_cal = subset_known(known_all, k_cal_idx)
        k_score = subset_known(known_all, k_score_idx)
        u_cal = subset_unknown(unknown_all, u_cal_idx)
        u_score = subset_unknown(unknown_all, u_score_idx)

        calibrated = exact_calibrate(k_cal, u_cal)
        score_metrics = scalar_metrics(
            k_score,
            u_score,
            calibrated["score_threshold"],
            calibrated["margin_threshold"],
        )
        passed = bool(
            calibrated["calibration_feasible"]
            and gate_pass(score_metrics)
        )

        row = {
            "split_index": i,
            "seed": seed,
            "known_cal_count": int(len(k_cal_idx)),
            "known_score_count": int(len(k_score_idx)),
            "unknown_cal_count": int(len(u_cal_idx)),
            "unknown_score_count": int(len(u_score_idx)),
            "calibration_feasible": bool(
                calibrated["calibration_feasible"]
            ),
            "feasible_pair_count": int(
                calibrated["feasible_pair_count"]
            ),
            "score_threshold": float(
                calibrated["score_threshold"]
            ),
            "margin_threshold": float(
                calibrated["margin_threshold"]
            ),
            "score_metrics": score_metrics,
            "absolute_gate_pass": passed,
        }
        runs.append(row)

        print(
            f"[split {i+1:02d}/{args.num_splits}] "
            f"cal_feasible={row['calibration_feasible']} "
            f"gate={'PASS' if passed else 'FAIL'} "
            f"F1={score_metrics['macro_f1']:.4f} "
            f"CA={score_metrics['correct_accept']:.4f} "
            f"KR={score_metrics['known_reject']:.4f} "
            f"UR={score_metrics['unknown_reject']:.4f}"
        )

    metrics = (
        "macro_f1",
        "correct_accept",
        "wrong_intent",
        "known_reject",
        "unknown_reject",
        "unknown_accept_far",
    )
    metric_summary = {
        name: summarize(
            [r["score_metrics"][name] for r in runs]
        )
        for name in metrics
    }

    gate_component_pass_rates = {
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
        "schema": "papr_ssl.p6_15shot_repeated_split_stability.v1",
        "phase": "P6",
        "purpose": (
            "development stability diagnostic; not untouched final Gate"
        ),
        "num_splits": args.num_splits,
        "base_seed": args.base_seed,
        "known_split": "class-stratified approximately 50/50",
        "unknown_split": "deterministic random 50/50",
        "calibration": (
            "exact empirical score+margin search on each calibration half"
        ),
        "absolute_gates": GATES,
        "calibration_feasible_rate": float(np.mean([
            r["calibration_feasible"] for r in runs
        ])),
        "absolute_gate_pass_rate": float(np.mean([
            r["absolute_gate_pass"] for r in runs
        ])),
        "absolute_gate_pass_count": int(sum(
            r["absolute_gate_pass"] for r in runs
        )),
        "component_gate_pass_rates": gate_component_pass_rates,
        "score_metric_summary": metric_summary,
        "score_threshold_summary": summarize(
            [r["score_threshold"] for r in runs]
        ),
        "margin_threshold_summary": summarize(
            [r["margin_threshold"] for r in runs]
        ),
        "runs": runs,
        "generic_test": "sealed_not_accessed",
        "interpretation_note": (
            "DEV was already observed in prior P6 iterations; repeated splits "
            "estimate development stability only and must not be described as "
            "independent test repetitions."
        ),
    }

    out_path = args.output_dir / "p6_15shot_stability.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 15-SHOT REPEATED-SPLIT STABILITY SUMMARY")
    print("=" * 108)
    print(
        f"calibration feasible: "
        f"{sum(r['calibration_feasible'] for r in runs)}/"
        f"{args.num_splits} "
        f"({output['calibration_feasible_rate']:.3f})"
    )
    print(
        f"absolute Gate PASS:   "
        f"{output['absolute_gate_pass_count']}/"
        f"{args.num_splits} "
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
        s = metric_summary[name]
        print(
            f"{name:20s} "
            f"mean={s['mean']:.6f} "
            f"std={s['sample_std']:.6f} "
            f"p10={s['p10']:.6f} "
            f"median={s['median']:.6f} "
            f"p90={s['p90']:.6f}"
        )

    print("-" * 108)
    print("component Gate pass rates:")
    for name, rate in gate_component_pass_rates.items():
        print(f"{name:20s} {rate:.3f}")

    print("-" * 108)
    print("generic_test accessed: NO")
    print("P6 15-SHOT STABILITY STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
