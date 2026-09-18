#!/usr/bin/env python
"""Audit P6 10-shot vs 15-shot development ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--result",
        type=Path,
        default=Path(
            "artifacts/p6_shot_ablation_10_15/"
            "p6_10_15_shot_results.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))
    if d["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if d["shots_evaluated"] != [10, 15]:
        raise RuntimeError("expected exactly 10-shot and 15-shot")

    print("=" * 108)
    print("P6 10-SHOT / 15-SHOT AUDIT")
    print("=" * 108)

    for shot in (10, 15):
        r = d["results"][str(shot)]
        c = r["calibration"]
        m = r["generic_dev_score_metrics"]
        print(
            f"{shot}-shot mean prototype | "
            f"cal_feasible={c['calibration_feasible']} | "
            f"feasible_pairs={c['search']['feasible_pair_count']:,}"
        )
        print(
            f"  Macro F1       {m['macro_f1']:.6f}   >= 0.80"
        )
        print(
            f"  Correct Accept {m['correct_accept']:.6f}   >= 0.80"
        )
        print(
            f"  Wrong Intent   {m['wrong_intent']:.6f}   <= 0.10"
        )
        print(
            f"  Known Reject   {m['known_reject']:.6f}   <= 0.20"
        )
        print(
            f"  Unknown Reject {m['unknown_reject']:.6f}   >= 0.85"
        )
        print(
            f"  absolute Gate: "
            f"{'PASS' if r['absolute_gate_pass'] else 'FAIL'}"
        )

    print("-" * 108)
    delta = d["delta_15_minus_10"]
    print(
        "15-shot - 10-shot: "
        f"F1={delta['macro_f1']:+.6f}, "
        f"CA={delta['correct_accept']:+.6f}, "
        f"WI={delta['wrong_intent']:+.6f}, "
        f"KR={delta['known_reject']:+.6f}, "
        f"UR={delta['unknown_reject']:+.6f}"
    )
    print("generic_test accessed: NO")
    print("P6 SHOT ABLATION AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
