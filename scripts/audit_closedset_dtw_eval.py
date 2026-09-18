#!/usr/bin/env python
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
            "artifacts/p6_temporal_dtw/closedset_eval/"
            "closedset_dtw_eval.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))

    if d["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if d["unknown_data_used"]:
        raise RuntimeError("Closed-set evaluation unexpectedly used unknown data")
    if d["rejector_used"]:
        raise RuntimeError("Closed-set evaluation unexpectedly used rejector")
    if d["threshold_used"]:
        raise RuntimeError("Closed-set evaluation unexpectedly used threshold")

    print("=" * 108)
    print("CLOSED-SET GLOBAL vs DTW AUDIT")
    print("=" * 108)

    for metric in ("macro_f1", "accuracy"):
        g = d["global_summary"][metric]["mean"]
        t = d["dtw_summary"][metric]["mean"]
        delta = d["dtw_minus_global_summary"][metric]["mean"]
        print(
            f"{metric:12s} global={g:.6f} dtw={t:.6f} delta={delta:+.6f}"
        )

    print(f"lambda histogram: {d['lambda_histogram']}")
    print("-" * 108)
    print("full DEV descriptive only:")
    for name, metrics in d["full_dev_descriptive_only"].items():
        print(
            f"  {name:20s} "
            f"F1={metrics['macro_f1']:.6f} "
            f"ACC={metrics['accuracy']:.6f}"
        )

    print("-" * 108)
    print("unknown data used:     NO")
    print("rejector used:         NO")
    print("threshold used:        NO")
    print("generic_test accessed: NO")
    print("AUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
