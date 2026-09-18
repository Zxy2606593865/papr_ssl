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
        default=Path(
            "artifacts/h1_02_unseen_speaker_openset/"
            "h1_02_unseen_speaker_openset.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))

    assert d["generic_test"] == "sealed_not_accessed"
    assert d["new_neural_head_added"] is False
    assert d["unknown_exposure_training_used"] is False
    assert d["loo_shrinkage_used"] is False
    assert d["cross_session_claim"] is False
    assert d["registered_classes_per_episode"] == 20
    assert d["unregistered_classes_per_episode"] == 10

    print("=" * 108)
    print("H1-02 UNSEEN-SPEAKER PERSONALIZED OPEN-SET AUDIT")
    print("=" * 108)

    for shot in ("1", "2"):
        for system in ("global", "dtw"):
            s = d["summary"][shot][system]
            print(
                f"{shot}-shot {system.upper():6s} | "
                f"CA={s['correct_accept']['mean']:.6f} "
                f"WI={s['wrong_intent']['mean']:.6f} "
                f"KR={s['known_reject']['mean']:.6f} "
                f"UR={s['unknown_reject']['mean']:.6f} "
                f"ClosedF1={s['closed_macro_f1']['mean']:.6f}"
            )

        ds = d["summary"][shot]
        print(
            f"{shot}-shot DTW-GLOBAL | "
            f"dCA={ds['delta_dtw_minus_global_correct_accept']['mean']:+.6f} "
            f"dWI={ds['delta_dtw_minus_global_wrong_intent']['mean']:+.6f} "
            f"dKR={ds['delta_dtw_minus_global_known_reject']['mean']:+.6f} "
            f"dUR={ds['delta_dtw_minus_global_unknown_reject']['mean']:+.6f}"
        )

    print("-" * 108)
    print("Protocol:               20 registered / 10 unregistered phrases")
    print("Calibration:            3 DEV speakers -> 1 held-out DEV speaker")
    print("Target calibration FAR: 10%")
    print("new neural Head:        NO")
    print("generic_test accessed:  NO")
    print("AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
