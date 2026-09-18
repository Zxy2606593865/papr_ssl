#!/usr/bin/env python
"""H6-01 — Three-state ACCEPT / CONFIRM / REJECT policy.

Uses the already-trained H5-01 C/W/U Head.

Goal
----
H5 showed a useful safety trade-off:
- fewer wrong accepts
- much higher unknown rejection
- but more known queries were simply counted as "reject"

H6 introduces the missing middle state:

    ACCEPT  -> evidence strongly supports C
    REJECT  -> evidence strongly supports U
    CONFIRM -> everything uncertain in between

This is the first business-facing decision policy.

Calibration
-----------
Uses only the 6 TRAIN calibration speakers already frozen by H5-01.

ACCEPT threshold:
    maximize Correct Accept subject to Unknown Accept <= 10%

REJECT threshold:
    maximize Unknown Reject subject to Known Reject <= 5%

DEV:
    4 unseen speakers, evaluation only.

Important:
- C/W/U softmax values are decision scores, not calibrated probabilities.
- CONFIRM is not counted as automatic recognition success.
- generic_test remains sealed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

TARGET_UNKNOWN_ACCEPT = 0.10
MAX_KNOWN_REJECT = 0.05


def load_h5_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("h5_impl", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import H5 helper script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_head(head_path: Path):
    h = json.loads(head_path.read_text(encoding="utf-8"))
    mean = np.asarray(h["standardize_mean"], dtype=np.float64)
    std = np.asarray(h["standardize_std"], dtype=np.float64)
    weight = np.asarray(h["weight"], dtype=np.float32)
    bias = np.asarray(h["bias"], dtype=np.float32)

    model = torch.nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        model.weight.copy_(torch.tensor(weight))
        model.bias.copy_(torch.tensor(bias))
    model.eval()
    return h, model, mean, std


def select_accept_threshold(rows, scores, h5):
    # Reuse the exact H5 accept-threshold rule.
    return h5.select_head_threshold(
        rows,
        scores,
        target_far=TARGET_UNKNOWN_ACCEPT,
    )


def select_reject_threshold(rows, scores):
    y_event = np.asarray([r["event"] for r in rows], dtype=str)
    known = y_event != "U"
    unknown = y_event == "U"

    u_score = scores[:, 2]
    argmax = scores.argmax(axis=1)
    reject_candidate = argmax == 2

    candidates = np.unique(
        np.concatenate([u_score, [np.inf, -np.inf]])
    )

    best = None
    for t in candidates:
        reject = reject_candidate & (u_score >= t)

        known_reject = float(np.mean(reject[known])) if np.any(known) else 0.0
        if known_reject > MAX_KNOWN_REJECT + 1e-12:
            continue

        unknown_reject = (
            float(np.mean(reject[unknown])) if np.any(unknown) else 0.0
        )

        # Primary: reject as many unknowns as possible.
        # Tie-break: reject fewer knowns, then prefer stricter threshold.
        key = (unknown_reject, -known_reject, t)
        if best is None or key > best["key"]:
            best = {
                "threshold": float(t),
                "cal_known_reject": known_reject,
                "cal_unknown_reject": unknown_reject,
                "key": key,
            }

    if best is None:
        raise RuntimeError("No feasible REJECT threshold")
    best.pop("key")
    return best


def apply_policy(rows, scores, tau_accept, tau_reject):
    event = np.asarray([r["event"] for r in rows], dtype=str)
    argmax = scores.argmax(axis=1)
    c_score = scores[:, 0]
    u_score = scores[:, 2]

    status = np.full(len(rows), "CONFIRM", dtype=object)

    accept = (argmax == 0) & (c_score >= tau_accept)
    reject = (~accept) & (argmax == 2) & (u_score >= tau_reject)

    status[accept] = "ACCEPT"
    status[reject] = "REJECT"

    known = event != "U"
    unknown = event == "U"
    correct = event == "C"
    wrong = event == "W"

    known_count = int(np.sum(known))
    unknown_count = int(np.sum(unknown))

    metrics = {
        "known_count": known_count,
        "unknown_count": unknown_count,

        "correct_accept": float(np.mean((status[known] == "ACCEPT") & correct[known])),
        "wrong_accept": float(np.mean((status[known] == "ACCEPT") & wrong[known])),
        "known_confirm": float(np.mean(status[known] == "CONFIRM")),
        "known_reject": float(np.mean(status[known] == "REJECT")),

        "unknown_accept": float(np.mean(status[unknown] == "ACCEPT")),
        "unknown_confirm": float(np.mean(status[unknown] == "CONFIRM")),
        "unknown_reject": float(np.mean(status[unknown] == "REJECT")),
    }

    metrics["known_identity_sum"] = (
        metrics["correct_accept"]
        + metrics["wrong_accept"]
        + metrics["known_confirm"]
        + metrics["known_reject"]
    )
    metrics["unknown_identity_sum"] = (
        metrics["unknown_accept"]
        + metrics["unknown_confirm"]
        + metrics["unknown_reject"]
    )

    # Selective risk among all automatically ACCEPTed queries in this evaluated
    # mixture. This is mixture-dependent and must not be treated as universal.
    accepted = status == "ACCEPT"
    auto_error = ((event == "W") | (event == "U")) & accepted
    metrics["accept_coverage_all_queries"] = float(np.mean(accepted))
    metrics["selective_risk_on_accept"] = (
        float(np.sum(auto_error) / np.sum(accepted))
        if np.sum(accepted) > 0
        else 0.0
    )

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("artifacts/h4_core30_features"),
    )
    p.add_argument(
        "--h5-result",
        type=Path,
        default=Path("artifacts/h5_01_cwu_decision_head/h5_01_eval.json"),
    )
    p.add_argument(
        "--h5-head",
        type=Path,
        default=Path("artifacts/h5_01_cwu_decision_head/cwu_head.json"),
    )
    p.add_argument(
        "--h5-script",
        type=Path,
        default=Path("scripts/run_h5_01_cwu_decision_head.py"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h6_01_three_state_policy"),
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    h5_result = json.loads(args.h5_result.read_text(encoding="utf-8"))
    if h5_result["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("H5 generic_test seal not intact")

    h5 = load_h5_module(args.h5_script)
    head_json, model, mean, std = load_head(args.h5_head)

    d = np.load(
        args.feature_dir / "global" / "core30_global256.npz",
        allow_pickle=False,
    )
    train = h5.load_split(d, "train")
    dev = h5.load_split(d, "dev")
    train_groups = h5.groups_for_split(train)
    dev_groups = h5.groups_for_split(dev)

    train_speakers = sorted(set(train["speaker_id"]))
    dev_speakers = sorted(set(dev["speaker_id"]))
    fit_speakers, cal_speakers = h5.split_train_speakers(
        train_speakers, cal_count=6
    )

    expected_cal = set(head_json["calibration_speakers"])
    if set(cal_speakers) != expected_cal:
        raise RuntimeError(
            "Calibration speaker split differs from H5-01 frozen split"
        )

    temporal_index = h5.load_temporal_index(args.feature_dir)
    temporal = h5.TemporalStore(temporal_index)

    print("=" * 108)
    print("H6-01 REBUILD CALIBRATION EVIDENCE")
    print("=" * 108)
    cal_rows = h5.collect_rows(
        data=train,
        groups=train_groups,
        temporal=temporal,
        speakers=cal_speakers,
        shots=h5.SHOT_LEVELS,
        repeats=h5.CAL_REPEATS,
    )
    X_cal, _, _, cal_shot = h5.rows_to_arrays(cal_rows)
    cal_scores = h5.predict_scores(model, mean, std, X_cal)

    thresholds = {}
    for shot in h5.SHOT_LEVELS:
        mask = cal_shot == shot
        rows_s = [r for r, m in zip(cal_rows, mask) if m]
        scores_s = cal_scores[mask]

        accept_info = select_accept_threshold(rows_s, scores_s, h5)
        reject_info = select_reject_threshold(rows_s, scores_s)

        thresholds[str(shot)] = {
            "accept": accept_info,
            "reject": reject_info,
        }

    print("=" * 108)
    print("H6-01 REBUILD UNSEEN DEV EVIDENCE")
    print("=" * 108)
    dev_rows = h5.collect_rows(
        data=dev,
        groups=dev_groups,
        temporal=temporal,
        speakers=set(dev_speakers),
        shots=h5.SHOT_LEVELS,
        repeats=h5.DEV_REPEATS,
    )
    X_dev, _, _, dev_shot = h5.rows_to_arrays(dev_rows)
    dev_scores = h5.predict_scores(model, mean, std, X_dev)

    summary = {}
    for shot in h5.SHOT_LEVELS:
        mask = dev_shot == shot
        rows_s = [r for r, m in zip(dev_rows, mask) if m]
        scores_s = dev_scores[mask]

        t = thresholds[str(shot)]
        metrics = apply_policy(
            rows_s,
            scores_s,
            tau_accept=t["accept"]["threshold"],
            tau_reject=t["reject"]["threshold"],
        )

        summary[str(shot)] = {
            "metrics": metrics,
            "thresholds": t,
        }

    out = {
        "schema": "papr_ssl.h6_01_three_state_policy.v1",
        "policy": {
            "ACCEPT": "argmax C and C-score >= tau_accept",
            "REJECT": "not ACCEPT and argmax U and U-score >= tau_reject",
            "CONFIRM": "all remaining queries",
        },
        "calibration_constraints": {
            "max_unknown_accept": TARGET_UNKNOWN_ACCEPT,
            "max_known_reject": MAX_KNOWN_REJECT,
        },
        "data": {
            "head_fit_speakers": 28,
            "calibration_speakers": 6,
            "unseen_dev_speakers": 4,
            "registered_classes_per_episode": 20,
            "unregistered_classes_per_episode": 10,
        },
        "summary": summary,
        "scores_are_calibrated_probabilities": False,
        "confirm_counts_as_auto_success": False,
        "backbone_neck_updated": False,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "h6_01_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("H6-01 ACCEPT / CONFIRM / REJECT SUMMARY")
    print("=" * 108)
    for shot in h5.SHOT_LEVELS:
        m = summary[str(shot)]["metrics"]
        print(
            f"{shot}-shot KNOWN   | "
            f"CorrectAccept={m['correct_accept']:.6f} "
            f"WrongAccept={m['wrong_accept']:.6f} "
            f"Confirm={m['known_confirm']:.6f} "
            f"Reject={m['known_reject']:.6f}"
        )
        print(
            f"{shot}-shot UNKNOWN | "
            f"Accept={m['unknown_accept']:.6f} "
            f"Confirm={m['unknown_confirm']:.6f} "
            f"Reject={m['unknown_reject']:.6f}"
        )
        print(
            f"{shot}-shot SYSTEM  | "
            f"AcceptCoverage={m['accept_coverage_all_queries']:.6f} "
            f"SelectiveRisk={m['selective_risk_on_accept']:.6f}"
        )
    print("-" * 108)
    print("CONFIRM auto-success:     NO")
    print("C/W/U probabilities:     NO (decision scores)")
    print("Backbone/Neck updated:   NO")
    print("generic_test accessed:   NO")
    print("H6-01 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
