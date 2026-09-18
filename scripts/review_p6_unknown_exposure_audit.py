#!/usr/bin/env python
"""Audit the P6 train-side unknown-exposure data audit artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--report",
        type=Path,
        default=Path(
            "artifacts/p6_unknown_exposure_audit/"
            "p6_unknown_exposure_audit.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(args.report.read_text(encoding="utf-8"))
    if d["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    leak = d["development_leakage_checks"]
    if leak["candidate_vs_core30_phrase_overlap"] != 0:
        raise RuntimeError("Core30 phrase leakage")
    if leak["candidate_vs_open_dev_phrase_overlap"] != 0:
        raise RuntimeError("open-set DEV phrase leakage")

    c = d["counts"]
    ss = d["candidate_phrase_support"]

    print("=" * 108)
    print("P6 UNKNOWN EXPOSURE DATA AUDIT REVIEW")
    print("=" * 108)
    print(f"train rows:             {c['train_rows']:,}")
    print(f"candidate rows:         {c['candidate_rows']:,}")
    print(f"candidate phrases:      {c['candidate_phrase_count']:,}")
    print(f"candidate speakers:     {c['candidate_speaker_count']:,}")
    print(f"domains:                {c['candidate_domain_counts']}")
    print("-" * 108)
    print(
        f"support median/p90/max: "
        f"{ss['median']:.1f} / {ss['p90']:.1f} / {ss['max']}"
    )
    print(
        f"phrases >=5/10/15:      "
        f"{ss['ge_5']} / {ss['ge_10']} / {ss['ge_15']}"
    )
    print("-" * 108)
    print("Core30 overlap:         0")
    print("open-set DEV overlap:   0")
    print("generic_test accessed:  NO")
    print("P6 UNKNOWN EXPOSURE DATA AUDIT REVIEW: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
