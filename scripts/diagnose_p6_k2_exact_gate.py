#!/usr/bin/env python
"""P6 Gate-failure diagnostic using frozen K=2 empirical DEV calibration scores.

This script does NOT select a new threshold policy.
It does NOT touch generic_test.
It summarizes the Pareto limits of the already-frozen two-threshold rule
on generic_dev_cal so we can distinguish:
- threshold-search resolution failure
from
- intrinsic overlap under the current embedding/prototype policy.

Primary diagnostic questions:
1) If Correct Accept >= 0.80, what is the maximum Unknown Reject achievable?
2) If Unknown Reject >= 0.85, what is the maximum Correct Accept achievable?
3) Under Wrong Intent <= 0.10 and Known Reject <= 0.20, what is the maximum
   Unknown Reject?
4) What is the nearest empirical operating point to the absolute Gate?
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


def candidates(values: np.ndarray) -> np.ndarray:
    u = np.unique(np.asarray(values, dtype=np.float64))
    return np.concatenate(
        [
            np.asarray([np.nextafter(u[0], -np.inf)]),
            u,
            np.asarray([np.nextafter(u[-1], np.inf)]),
        ]
    )


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


def metrics(k: dict, u: dict, ts: float, tm: float) -> dict:
    ka = (k["score"] >= ts) & (k["margin"] >= tm)
    ua = (u["score"] >= ts) & (u["margin"] >= tm)
    correct = ka & (k["pred"] == k["true"])
    wrong = ka & (k["pred"] != k["true"])
    return {
        "score_threshold": float(ts),
        "margin_threshold": float(tm),
        "macro_f1": macro_f1(ka, k["true"], k["pred"]),
        "correct_accept": float(np.mean(correct)),
        "wrong_intent": float(np.mean(wrong)),
        "known_reject": float(np.mean(~ka)),
        "unknown_reject": float(np.mean(~ua)),
        "unknown_accept_far": float(np.mean(ua)),
    }


def violation(m: dict) -> float:
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


def load_cal(npz_path: Path):
    d = np.load(npz_path, allow_pickle=False)
    known = {
        "score": d["known_cal_score"],
        "margin": d["known_cal_margin"],
        "pred": d["known_cal_pred"],
        "true": d["known_cal_true"],
    }
    unknown = {
        "score": d["unknown_cal_score"],
        "margin": d["unknown_cal_margin"],
        "pred": d["unknown_cal_pred"],
    }
    return known, unknown


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "artifacts/p6_dev_k2_exact/scores/attention_dev_scores.npz"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/p6_dev_k2_exact/"
            "p6_gate_failure_diagnostic.json"
        ),
    )
    args = p.parse_args()

    k, u = load_cal(args.scores)
    score_candidates = candidates(np.concatenate([k["score"], u["score"]]))
    margin_candidates = candidates(np.concatenate([k["margin"], u["margin"]]))

    best_ca_given_ur = None
    best_ur_given_ca = None
    best_ur_given_known_constraints = None
    nearest = None

    # Enumerate all empirical operating points exactly.
    total = len(score_candidates) * len(margin_candidates)
    visited = 0

    for ts in score_candidates:
        for tm in margin_candidates:
            m = metrics(k, u, float(ts), float(tm))
            visited += 1

            if m["unknown_reject"] >= GATES["unknown_reject_min"]:
                key = (
                    m["correct_accept"],
                    m["macro_f1"],
                    -m["known_reject"],
                    -m["wrong_intent"],
                )
                if (
                    best_ca_given_ur is None
                    or key > best_ca_given_ur[0]
                ):
                    best_ca_given_ur = (key, m)

            if m["correct_accept"] >= GATES["correct_accept_min"]:
                key = (
                    m["unknown_reject"],
                    m["macro_f1"],
                    -m["known_reject"],
                    -m["wrong_intent"],
                )
                if (
                    best_ur_given_ca is None
                    or key > best_ur_given_ca[0]
                ):
                    best_ur_given_ca = (key, m)

            if (
                m["wrong_intent"] <= GATES["wrong_intent_max"]
                and m["known_reject"] <= GATES["known_reject_max"]
            ):
                key = (
                    m["unknown_reject"],
                    m["correct_accept"],
                    m["macro_f1"],
                )
                if (
                    best_ur_given_known_constraints is None
                    or key > best_ur_given_known_constraints[0]
                ):
                    best_ur_given_known_constraints = (key, m)

            v = violation(m)
            key = (
                -v,
                m["macro_f1"],
                m["correct_accept"],
                m["unknown_reject"],
                -m["known_reject"],
                -m["wrong_intent"],
            )
            if nearest is None or key > nearest[0]:
                nearest = (key, m, v)

        if visited % 100000 < len(margin_candidates):
            print(f"[diagnostic] {visited:,}/{total:,}")

    # Distribution summaries help explain overlap.
    correct_known = k["pred"] == k["true"]
    wrong_known = ~correct_known

    def q(x):
        x = np.asarray(x, dtype=np.float64)
        return {
            "n": int(x.size),
            "p05": float(np.quantile(x, 0.05)),
            "p25": float(np.quantile(x, 0.25)),
            "p50": float(np.quantile(x, 0.50)),
            "p75": float(np.quantile(x, 0.75)),
            "p95": float(np.quantile(x, 0.95)),
        }

    out = {
        "schema": "papr_ssl.p6_gate_failure_diagnostic.v1",
        "source": args.scores.as_posix(),
        "search": {
            "score_candidate_count": int(len(score_candidates)),
            "margin_candidate_count": int(len(margin_candidates)),
            "threshold_pair_count": int(total),
            "exact_empirical_enumeration": True,
        },
        "absolute_gates": GATES,
        "pareto_limits": {
            "max_correct_accept_given_unknown_reject_ge_085": (
                None if best_ca_given_ur is None else best_ca_given_ur[1]
            ),
            "max_unknown_reject_given_correct_accept_ge_080": (
                None if best_ur_given_ca is None else best_ur_given_ca[1]
            ),
            "max_unknown_reject_given_wrong_intent_le_010_and_known_reject_le_020": (
                None
                if best_ur_given_known_constraints is None
                else best_ur_given_known_constraints[1]
            ),
            "nearest_gate_operating_point": {
                **nearest[1],
                "normalized_gate_violation": float(nearest[2]),
            },
        },
        "calibration_distributions": {
            "known_correct_score": q(k["score"][correct_known]),
            "known_wrong_score": q(k["score"][wrong_known]),
            "unknown_score": q(u["score"]),
            "known_correct_margin": q(k["margin"][correct_known]),
            "known_wrong_margin": q(k["margin"][wrong_known]),
            "unknown_margin": q(u["margin"]),
        },
        "interpretation_contract": (
            "diagnostic only; no threshold/model selection; "
            "generic_test remains sealed"
        ),
        "generic_test": "sealed_not_accessed",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 EXACT GATE FAILURE DIAGNOSTIC")
    print("=" * 108)

    a = out["pareto_limits"][
        "max_correct_accept_given_unknown_reject_ge_085"
    ]
    b = out["pareto_limits"][
        "max_unknown_reject_given_correct_accept_ge_080"
    ]
    c = out["pareto_limits"][
        "max_unknown_reject_given_wrong_intent_le_010_and_known_reject_le_020"
    ]
    n = out["pareto_limits"]["nearest_gate_operating_point"]

    if a:
        print(
            "max Correct Accept with Unknown Reject >= 0.85: "
            f"{a['correct_accept']:.6f}"
        )
        print(
            "  MacroF1 / KR / WI / UR: "
            f"{a['macro_f1']:.6f} / {a['known_reject']:.6f} / "
            f"{a['wrong_intent']:.6f} / {a['unknown_reject']:.6f}"
        )
    else:
        print("no point with Unknown Reject >= 0.85")

    if b:
        print(
            "max Unknown Reject with Correct Accept >= 0.80: "
            f"{b['unknown_reject']:.6f}"
        )
        print(
            "  MacroF1 / CA / KR / WI: "
            f"{b['macro_f1']:.6f} / {b['correct_accept']:.6f} / "
            f"{b['known_reject']:.6f} / {b['wrong_intent']:.6f}"
        )
    else:
        print("no point with Correct Accept >= 0.80")

    if c:
        print(
            "max Unknown Reject with WI <= 0.10 and KR <= 0.20: "
            f"{c['unknown_reject']:.6f}"
        )
        print(
            "  MacroF1 / CA / KR / WI: "
            f"{c['macro_f1']:.6f} / {c['correct_accept']:.6f} / "
            f"{c['known_reject']:.6f} / {c['wrong_intent']:.6f}"
        )

    print("-" * 108)
    print("nearest empirical Gate point:")
    print(
        f"MacroF1={n['macro_f1']:.6f} "
        f"CA={n['correct_accept']:.6f} "
        f"WI={n['wrong_intent']:.6f} "
        f"KR={n['known_reject']:.6f} "
        f"UR={n['unknown_reject']:.6f}"
    )
    print(
        "normalized Gate violation: "
        f"{n['normalized_gate_violation']:.6f}"
    )
    print("generic_test accessed: NO")
    print("P6 DIAGNOSTIC STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
