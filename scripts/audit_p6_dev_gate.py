#!/usr/bin/env python
"""Audit P6 development Gate and optional P6-06 Teacher lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gate-json",
        type=Path,
        default=Path(
            "artifacts/p6_dev/qualification/p6_dev_gate.json"
        ),
    )
    p.add_argument(
        "--teacher-lock",
        type=Path,
        default=Path("artifacts/p6_teacher/teacher_lock.json"),
    )
    args = p.parse_args()

    gate = json.loads(args.gate_json.read_text(encoding="utf-8"))
    if gate["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if gate["p6_04_mainline_absolute_gate_pass"] is not True:
        raise RuntimeError("P6-04 absolute Gate FAIL")
    if gate["p6_05_e0b"]["pass"] is not True:
        raise RuntimeError("P6-05 E0b Gate FAIL")

    m = gate["results"]["attention"]["generic_dev_score_metrics"]
    print("=" * 108)
    print("P6 DEVELOPMENT GATE AUDIT")
    print("=" * 108)
    print(f"Macro F1:       {m['macro_f1']:.6f}   >= 0.80")
    print(f"Correct Accept: {m['correct_accept']:.6f}   >= 0.80")
    print(f"Wrong Intent:   {m['wrong_intent']:.6f}   <= 0.10")
    print(f"Known Reject:   {m['known_reject']:.6f}   <= 0.20")
    print(f"Unknown Reject: {m['unknown_reject']:.6f}   >= 0.85")
    print(
        "E0b drop:       "
        f"{gate['p6_05_e0b']['baseline_minus_mainline_drop']:+.6f} "
        "<= 0.02"
    )
    print("generic_test accessed: NO")

    if args.teacher_lock.is_file():
        lock = json.loads(args.teacher_lock.read_text(encoding="utf-8"))
        if lock["generic_test"] != "sealed_not_accessed":
            raise RuntimeError("Teacher lock test seal violated")
        if lock["test_may_tune_model_or_thresholds"] is not False:
            raise RuntimeError("test tuning must be prohibited")
        print("-" * 108)
        print("P6-06 Teacher lock: PRESENT")
        print(
            f"score/margin threshold: "
            f"{lock['threshold_policy']['score_threshold']:.6f} / "
            f"{lock['threshold_policy']['margin_threshold']:.6f}"
        )
        print("P6-06 STATUS: FROZEN / READY FOR P6-07")
    else:
        print("-" * 108)
        print("P6-06 Teacher lock: NOT YET CREATED")

    print("-" * 108)
    print("P6 DEV GATE STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
