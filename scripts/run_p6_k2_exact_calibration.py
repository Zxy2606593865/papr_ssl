#!/usr/bin/env python
"""Exact empirical score+margin threshold search for P6 K=2.

Why this is "exact" on the calibration set
-------------------------------------------
For accept rule:
    score >= score_threshold AND margin >= margin_threshold

the classification decisions can only change when a threshold crosses an
observed calibration score or margin. Therefore it is sufficient to evaluate:
- every unique observed score value
- every unique observed margin value
- one value below minimum
- one value above maximum

This exhausts every distinct empirical accept/reject partition expressible by
the two-threshold rule on generic_dev_cal.

Threshold selection uses generic_dev_cal ONLY.
generic_dev_score is evaluated once after the thresholds are fixed.
generic_test is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GATES = {
    "macro_f1_min": 0.80,
    "correct_accept_min": 0.80,
    "wrong_intent_max": 0.10,
    "known_reject_max": 0.20,
    "unknown_reject_min": 0.85,
}
E0B_MAX_DROP = 0.02


def empirical_candidates(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    unique = np.unique(values)
    if unique.size == 0:
        raise RuntimeError("empty empirical candidate set")
    below = np.nextafter(unique[0], -np.inf)
    above = np.nextafter(unique[-1], np.inf)
    return np.concatenate(
        [
            np.asarray([below], dtype=np.float64),
            unique,
            np.asarray([above], dtype=np.float64),
        ]
    )


def macro_f1_vectorized(
    accepted: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
) -> np.ndarray:
    """accepted: [M,N], returns Macro-F1 [M]."""
    m = accepted.shape[0]
    total = np.zeros(m, dtype=np.float64)
    for c in range(30):
        true_c = true == c
        pred_c = pred == c
        tp = np.sum(accepted & true_c[None, :] & pred_c[None, :], axis=1)
        fp = np.sum(accepted & (~true_c)[None, :] & pred_c[None, :], axis=1)
        true_count = int(np.sum(true_c))
        fn = true_count - tp
        denom = 2.0 * tp + fp + fn
        f1 = np.divide(
            2.0 * tp,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0,
        )
        total += f1
    return total / 30.0


def metric_vectors_for_score_threshold(
    *,
    known: dict[str, np.ndarray],
    unknown: dict[str, np.ndarray],
    score_threshold: float,
    margin_candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    known_score_ok = known["score"] >= score_threshold
    unknown_score_ok = unknown["score"] >= score_threshold

    known_margin_ok = (
        known["margin"][None, :] >= margin_candidates[:, None]
    )
    unknown_margin_ok = (
        unknown["margin"][None, :] >= margin_candidates[:, None]
    )

    known_accept = known_margin_ok & known_score_ok[None, :]
    unknown_accept = unknown_margin_ok & unknown_score_ok[None, :]

    correct_mask = known["pred"] == known["true"]
    wrong_mask = ~correct_mask

    correct_accept = np.mean(
        known_accept & correct_mask[None, :],
        axis=1,
    )
    wrong_intent = np.mean(
        known_accept & wrong_mask[None, :],
        axis=1,
    )
    known_reject = 1.0 - np.mean(known_accept, axis=1)
    unknown_reject = 1.0 - np.mean(unknown_accept, axis=1)
    macro_f1 = macro_f1_vectorized(
        known_accept,
        known["true"],
        known["pred"],
    )

    return {
        "macro_f1": macro_f1,
        "correct_accept": correct_accept,
        "wrong_intent": wrong_intent,
        "known_reject": known_reject,
        "unknown_reject": unknown_reject,
        "unknown_accept_far": 1.0 - unknown_reject,
    }


def feasible_mask(v: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (v["macro_f1"] >= GATES["macro_f1_min"])
        & (v["correct_accept"] >= GATES["correct_accept_min"])
        & (v["wrong_intent"] <= GATES["wrong_intent_max"])
        & (v["known_reject"] <= GATES["known_reject_max"])
        & (v["unknown_reject"] >= GATES["unknown_reject_min"])
    )


def violation_vector(v: dict[str, np.ndarray]) -> np.ndarray:
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


def scalar_metrics(
    known: dict[str, np.ndarray],
    unknown: dict[str, np.ndarray],
    score_threshold: float,
    margin_threshold: float,
) -> dict:
    ka = (
        (known["score"] >= score_threshold)
        & (known["margin"] >= margin_threshold)
    )
    ua = (
        (unknown["score"] >= score_threshold)
        & (unknown["margin"] >= margin_threshold)
    )
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
        "known_count": int(len(ka)),
        "unknown_count": int(len(ua)),
    }


def metric_gate_pass(m: dict) -> bool:
    return (
        m["macro_f1"] >= GATES["macro_f1_min"]
        and m["correct_accept"] >= GATES["correct_accept_min"]
        and m["wrong_intent"] <= GATES["wrong_intent_max"]
        and m["known_reject"] <= GATES["known_reject_max"]
        and m["unknown_reject"] >= GATES["unknown_reject_min"]
    )


def load_split_arrays(npz_path: Path) -> tuple[dict, dict, dict, dict]:
    d = np.load(npz_path, allow_pickle=False)
    known_cal = {
        "score": d["known_cal_score"],
        "margin": d["known_cal_margin"],
        "pred": d["known_cal_pred"],
        "true": d["known_cal_true"],
    }
    known_score = {
        "score": d["known_score_score"],
        "margin": d["known_score_margin"],
        "pred": d["known_score_pred"],
        "true": d["known_score_true"],
    }
    unknown_cal = {
        "score": d["unknown_cal_score"],
        "margin": d["unknown_cal_margin"],
        "pred": d["unknown_cal_pred"],
    }
    unknown_score = {
        "score": d["unknown_score_score"],
        "margin": d["unknown_score_margin"],
        "pred": d["unknown_score_pred"],
    }
    return known_cal, known_score, unknown_cal, unknown_score


def exact_calibrate(
    known: dict[str, np.ndarray],
    unknown: dict[str, np.ndarray],
) -> dict:
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
    pair_count = int(score_candidates.size * margin_candidates.size)
    feasible_pair_count = 0

    for i, score_threshold in enumerate(score_candidates, start=1):
        vectors = metric_vectors_for_score_threshold(
            known=known,
            unknown=unknown,
            score_threshold=float(score_threshold),
            margin_candidates=margin_candidates,
        )

        ok = feasible_mask(vectors)
        feasible_idx = np.flatnonzero(ok)
        feasible_pair_count += int(feasible_idx.size)

        for j in feasible_idx:
            # Frozen tie-break: maximize Macro-F1, Correct Accept,
            # Unknown Reject; minimize Wrong Intent, Known Reject;
            # then prefer larger score/margin thresholds deterministically.
            key = (
                float(vectors["macro_f1"][j]),
                float(vectors["correct_accept"][j]),
                float(vectors["unknown_reject"][j]),
                -float(vectors["wrong_intent"][j]),
                -float(vectors["known_reject"][j]),
                float(score_threshold),
                float(margin_candidates[j]),
            )
            if best_feasible_key is None or key > best_feasible_key:
                best_feasible_key = key
                best_feasible = {
                    "score_threshold": float(score_threshold),
                    "margin_threshold": float(margin_candidates[j]),
                    "metrics": {
                        name: float(values[j])
                        for name, values in vectors.items()
                    },
                }

        if best_feasible is None:
            vio = violation_vector(vectors)
            for j in np.flatnonzero(np.isfinite(vio)):
                key = (
                    -float(vio[j]),
                    float(vectors["macro_f1"][j]),
                    float(vectors["correct_accept"][j]),
                    float(vectors["unknown_reject"][j]),
                    -float(vectors["wrong_intent"][j]),
                    -float(vectors["known_reject"][j]),
                    float(score_threshold),
                    float(margin_candidates[j]),
                )
                if best_diag_key is None or key > best_diag_key:
                    best_diag_key = key
                    best_diag = {
                        "score_threshold": float(score_threshold),
                        "margin_threshold": float(margin_candidates[j]),
                        "metrics": {
                            name: float(values[j])
                            for name, values in vectors.items()
                        },
                        "violation": float(vio[j]),
                    }

        if i % 100 == 0 or i == score_candidates.size:
            print(
                f"[exact search] score threshold {i}/"
                f"{score_candidates.size}; feasible_pairs={feasible_pair_count}"
            )

    selected = best_feasible if best_feasible is not None else best_diag
    if selected is None:
        raise RuntimeError("exact threshold search produced no candidate")

    selected["calibration_feasible"] = best_feasible is not None
    selected["search"] = {
        "method": "exact_empirical_boundary_enumeration",
        "score_candidate_count": int(score_candidates.size),
        "margin_candidate_count": int(margin_candidates.size),
        "threshold_pair_count": pair_count,
        "feasible_pair_count": feasible_pair_count,
    }
    return selected


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--score-dir",
        type=Path,
        default=Path("artifacts/p6_dev_k2_exact/scores"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_dev_k2_exact/qualification"),
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"exact qualification exists; refusing overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (args.score_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("score export test seal violated")
    if int(manifest["subprototype_k"]) != 2:
        raise RuntimeError("expected K=2 score export")

    results = {}
    for method in ("mean", "attention"):
        known_cal, known_score, unknown_cal, unknown_score = load_split_arrays(
            args.score_dir / f"{method}_dev_scores.npz"
        )

        print("=" * 108)
        print(f"P6 K=2 EXACT CALIBRATION: {method.upper()}")
        print("=" * 108)

        calibrated = exact_calibrate(known_cal, unknown_cal)
        score_metrics = scalar_metrics(
            known_score,
            unknown_score,
            calibrated["score_threshold"],
            calibrated["margin_threshold"],
        )
        score_gate_pass = bool(
            calibrated["calibration_feasible"]
            and metric_gate_pass(score_metrics)
        )

        results[method] = {
            "calibration": calibrated,
            "generic_dev_score_metrics": score_metrics,
            "generic_dev_score_gate_pass": score_gate_pass,
        }

        print(
            "calibration_feasible: "
            f"{calibrated['calibration_feasible']}"
        )
        print(
            "score_threshold:      "
            f"{calibrated['score_threshold']:.12f}"
        )
        print(
            "margin_threshold:     "
            f"{calibrated['margin_threshold']:.12f}"
        )
        print(
            "searched pairs:       "
            f"{calibrated['search']['threshold_pair_count']:,}"
        )
        print(
            "feasible pairs:       "
            f"{calibrated['search']['feasible_pair_count']:,}"
        )
        print("DEV SCORE METRICS")
        for name, value in score_metrics.items():
            if isinstance(value, float):
                print(f"{name:20s} {value:.6f}")
        print(f"dev score Gate:       {score_gate_pass}")

    mean_f1 = results["mean"]["generic_dev_score_metrics"]["macro_f1"]
    attention_f1 = results["attention"]["generic_dev_score_metrics"]["macro_f1"]
    e0b_drop = mean_f1 - attention_f1
    e0b_pass = bool(e0b_drop <= E0B_MAX_DROP)
    eligible = bool(
        results["attention"]["generic_dev_score_gate_pass"]
        and e0b_pass
    )

    output = {
        "schema": "papr_ssl.p6_k2_exact_qualification.v1",
        "phase": "P6",
        "protocol_amendment": {
            "trigger": (
                "K=2 coarse quantile calibration reported "
                "calibration_feasible=False on generic_dev_cal"
            ),
            "change": (
                "replace 101-quantile threshold grid with exact empirical "
                "boundary enumeration"
            ),
            "unchanged": [
                "backbone",
                "hidden_state_index",
                "head",
                "checkpoint",
                "K=2 subprototype definition",
                "generic_dev_cal partition",
                "generic_dev_score partition",
                "absolute Gates",
            ],
            "generic_test_used_to_motivate_change": False,
        },
        "absolute_gates": GATES,
        "results": results,
        "p6_04_mainline_absolute_gate_pass": results["attention"][
            "generic_dev_score_gate_pass"
        ],
        "p6_05_e0b": {
            "baseline_method": "mean",
            "mainline_method": "attention",
            "baseline_minus_mainline_drop": e0b_drop,
            "max_allowed_drop": E0B_MAX_DROP,
            "pass": e0b_pass,
        },
        "eligible_for_p6_06_freeze": eligible,
        "generic_test": "sealed_not_accessed",
    }

    out_path = args.output_dir / "p6_k2_exact_dev_gate.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 K=2 EXACT DEVELOPMENT GATE SUMMARY")
    print("=" * 108)
    print(
        "P6-04 absolute Gate: "
        f"{'PASS' if output['p6_04_mainline_absolute_gate_pass'] else 'FAIL'}"
    )
    print(
        "P6-05 E0b Gate:      "
        f"{'PASS' if e0b_pass else 'FAIL'} "
        f"(baseline-mainline={e0b_drop:+.6f})"
    )
    print(f"eligible for P6-06 freeze: {eligible}")
    print("generic_test accessed: NO")
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
