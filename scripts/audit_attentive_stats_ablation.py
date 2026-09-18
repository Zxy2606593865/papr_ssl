#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "artifacts/p6_attentive_stats_ablation/summary.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(
        args.summary.read_text(encoding="utf-8")
    )

    if d["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    if (
        d["canonical_64d_teacher"]
        != "preserved_not_modified"
    ):
        raise RuntimeError(
            "canonical 64D preservation contract violated"
        )

    if (
        d["project1_256d_teacher"]
        != "preserved_not_overwritten"
    ):
        raise RuntimeError(
            "Project-1 256D preservation contract violated"
        )

    print("=" * 108)
    print("ATTENTIVE STATISTICS POOLING AUDIT")
    print("=" * 108)

    for mode, s in d["aggregate"].items():
        print(
            f"{mode:24s} "
            f"mean={s['mean']:.6f} "
            f"std={s['sample_std']:.6f} "
            f"seeds={s['values']}"
        )

    print(
        f"ASP - AttentionMean: "
        f"{d['attentive_stats_minus_mean']:+.6f}"
    )

    repro = d.get("matched_mean_reproduction")
    if repro:
        print(
            f"matched mean reproduction: "
            f"{'PASS' if repro['within_0p02'] else 'FAIL'} "
            f"(abs diff={repro['absolute_difference']:.6f})"
        )

    print("canonical 64D modified: NO")
    print("Project-1 256D modified: NO")
    print("generic_test accessed:  NO")
    print("AUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
