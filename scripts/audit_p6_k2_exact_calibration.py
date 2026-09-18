#!/usr/bin/env python
"""Audit P6 K=2 exact empirical threshold result."""

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
            "artifacts/p6_dev_k2_exact/qualification/"
            "p6_k2_exact_dev_gate.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))
    if d["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    a = d["results"]["attention"]
    cal = a["calibration"]
    m = a["generic_dev_score_metrics"]

    print("=" * 108)
    print("P6 K=2 EXACT EMPIRICAL THRESHOLD AUDIT")
    print("=" * 108)
    print(f"calibration feasible: {cal['calibration_feasible']}")
    print(
        f"search pairs:         "
        f"{cal['search']['threshold_pair_count']:,}"
    )
    print(
        f"feasible pairs:       "
        f"{cal['search']['feasible_pair_count']:,}"
    )
    print(
        f"score threshold:      {cal['score_threshold']:.12f}"
    )
    print(
        f"margin threshold:     {cal['margin_threshold']:.12f}"
    )
    print("-" * 108)
    print(f"Macro F1:       {m['macro_f1']:.6f} >= 0.80")
    print(f"Correct Accept: {m['correct_accept']:.6f} >= 0.80")
    print(f"Wrong Intent:   {m['wrong_intent']:.6f} <= 0.10")
    print(f"Known Reject:   {m['known_reject']:.6f} <= 0.20")
    print(f"Unknown Reject: {m['unknown_reject']:.6f} >= 0.85")
    print("-" * 108)
    print(
        f"P6-04 absolute Gate: "
        f"{'PASS' if d['p6_04_mainline_absolute_gate_pass'] else 'FAIL'}"
    )
    print(
        f"P6-05 E0b Gate: "
        f"{'PASS' if d['p6_05_e0b']['pass'] else 'FAIL'}"
    )
    print(
        f"eligible_for_P6_06: "
        f"{d['eligible_for_p6_06_freeze']}"
    )
    print("generic_test accessed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
