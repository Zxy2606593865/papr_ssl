#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--result",
        type=Path,
        default=Path("artifacts/h5_01_cwu_decision_head/h5_01_eval.json"),
    )
    p.add_argument(
        "--head",
        type=Path,
        default=Path("artifacts/h5_01_cwu_decision_head/cwu_head.json"),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))
    h = json.loads(args.head.read_text(encoding="utf-8"))

    assert d["generic_test"] == "sealed_not_accessed"
    assert d["backbone_neck_updated"] is False
    assert d["learned_h4_ranker_used"] is False
    assert d["unknown_exposure_training_used"] is False
    assert d["loo_shrinkage_used"] is False
    assert d["protocol"]["train_head_fit_speakers"] == 28
    assert d["protocol"]["train_calibration_speakers"] == 6
    assert d["protocol"]["dev_unseen_speakers"] == 4
    assert h["scores_are_calibrated_probabilities"] is False

    print("=" * 108)
    print("H5-01 C/W/U DECISION HEAD AUDIT")
    print("=" * 108)

    for shot in ("1", "2"):
        s = d["summary"][shot]
        b = s["baseline"]
        c = s["cwu_head"]
        delta = s["delta_head_minus_baseline"]

        print(
            f"{shot}-shot BASE  | "
            f"CA={b['correct_accept']:.6f} "
            f"WI={b['wrong_intent']:.6f} "
            f"KR={b['known_reject']:.6f} "
            f"UR={b['unknown_reject']:.6f}"
        )
        print(
            f"{shot}-shot C/W/U | "
            f"CA={c['correct_accept']:.6f} "
            f"WI={c['wrong_intent']:.6f} "
            f"KR={c['known_reject']:.6f} "
            f"UR={c['unknown_reject']:.6f} "
            f"EventF1={c['event_macro_f1']:.6f}"
        )
        print(
            f"{shot}-shot DELTA | "
            f"dCA={delta['correct_accept']:+.6f} "
            f"dWI={delta['wrong_intent']:+.6f} "
            f"dKR={delta['known_reject']:+.6f} "
            f"dUR={delta['unknown_reject']:+.6f}"
        )

    print("-" * 108)
    print("Head-fit:                28 TRAIN speakers")
    print("Threshold calibration:   6 TRAIN speakers")
    print("Evaluation:              4 unseen DEV speakers")
    print("C/W/U scores calibrated: NO (decision scores only)")
    print("Backbone/Neck updated:   NO")
    print("generic_test accessed:   NO")
    print("AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
