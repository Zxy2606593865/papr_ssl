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
            "artifacts/h1_01_unseen_speaker_closedset/eval/"
            "h1_01_unseen_speaker_closedset.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))

    if d["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if d["new_trainable_head_added"]:
        raise RuntimeError("H1-01 must not add a new trainable Head")
    if d["unknown_data_used"]:
        raise RuntimeError("H1-01 must be closed-set only")
    if d["rejector_used"]:
        raise RuntimeError("H1-01 must not use rejector")
    if d["cross_session_claim"]:
        raise RuntimeError("MDSC lacks explicit session metadata")

    print("=" * 108)
    print("H1-01 UNSEEN-SPEAKER PERSONALIZED CLOSED-SET AUDIT")
    print("=" * 108)

    for shot in ("1", "2"):
        s = d["summary"][shot]
        print(
            f"{shot}-shot | "
            f"GLOBAL F1={s['global_macro_f1']['mean']:.6f} "
            f"ACC={s['global_accuracy']['mean']:.6f} "
            f"R@3={s['global_recall_at_3']['mean']:.6f} | "
            f"DTW F1={s['dtw_macro_f1']['mean']:.6f} "
            f"ACC={s['dtw_accuracy']['mean']:.6f} | "
            f"dF1={s['delta_macro_f1']['mean']:+.6f} | "
            f"better/equal/worse="
            f"{s['dtw_better_f1_repeats']}/"
            f"{s['dtw_equal_f1_repeats']}/"
            f"{s['dtw_worse_f1_repeats']}"
        )

    print("-" * 108)
    print("DEV speakers:           4 unseen speakers")
    print("new trainable Head:     NO")
    print("unknown data used:      NO")
    print("rejector used:          NO")
    print("cross-session claim:    NO")
    print("generic_test accessed:  NO")
    print("AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
