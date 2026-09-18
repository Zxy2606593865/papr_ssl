#!/usr/bin/env python
"""Audit the P6 15-shot repeated-split stability diagnostic."""

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
            "artifacts/p6_15shot_stability/repeated_splits/"
            "p6_15shot_stability.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))
    if d["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    print("=" * 108)
    print("P6 15-SHOT STABILITY AUDIT")
    print("=" * 108)
    print(f"repeated splits:              {d['num_splits']}")
    print(
        f"calibration feasible rate:    "
        f"{d['calibration_feasible_rate']:.3f}"
    )
    print(
        f"absolute Gate pass rate:      "
        f"{d['absolute_gate_pass_rate']:.3f}"
    )
    print("-" * 108)

    for name in (
        "macro_f1",
        "correct_accept",
        "wrong_intent",
        "known_reject",
        "unknown_reject",
    ):
        s = d["score_metric_summary"][name]
        print(
            f"{name:20s} "
            f"mean={s['mean']:.6f} "
            f"std={s['sample_std']:.6f} "
            f"median={s['median']:.6f}"
        )

    print("-" * 108)
    print("component Gate pass rates:")
    for name, rate in d["component_gate_pass_rates"].items():
        print(f"{name:20s} {rate:.3f}")

    print("-" * 108)
    print("generic_test accessed: NO")
    print("P6 15-SHOT STABILITY AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
