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
            "artifacts/h4_01_shared_evidence_ranker/h4_01_eval.json"
        ),
    )
    p.add_argument(
        "--ranker",
        type=Path,
        default=Path(
            "artifacts/h4_01_shared_evidence_ranker/ranker.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))
    r = json.loads(args.ranker.read_text(encoding="utf-8"))

    assert d["generic_test"] == "sealed_not_accessed"
    assert d["new_trainable_head_added"] is True
    assert d["backbone_neck_updated"] is False
    assert d["unknown_data_used"] is False
    assert d["rejector_used"] is False
    assert d["fit"]["speakers"] == 34
    assert d["eval"]["speakers"] == 4
    assert r["frozen_backbone_neck"] is True

    print("=" * 108)
    print("H4-01 SHARED EVIDENCE RANKER AUDIT")
    print("=" * 108)

    for shot in ("1", "2"):
        s = d["summary"][shot]
        print(
            f"{shot}-shot | "
            f"GLOBAL F1={s['global_macro_f1']['mean']:.6f} | "
            f"FIXED-DTW F1={s['fixed_dtw_macro_f1']['mean']:.6f} | "
            f"LEARNED F1={s['learned_macro_f1']['mean']:.6f} | "
            f"dLearned-Fixed={s['learned_minus_fixed_f1']['mean']:+.6f} | "
            f"better/equal/worse="
            f"{s['learned_better_than_fixed_repeats']}/"
            f"{s['learned_equal_fixed_repeats']}/"
            f"{s['learned_worse_than_fixed_repeats']}"
        )

    print("-" * 108)
    print("Head fitted on:          34 TRAIN speakers")
    print("Evaluated on:            4 unseen DEV speakers")
    print("Head type:               shared linear logistic candidate ranker")
    print("Backbone/Neck updated:   NO")
    print("generic_test accessed:   NO")
    print("AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
