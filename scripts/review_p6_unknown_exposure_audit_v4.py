#!/usr/bin/env python
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
            "artifacts/p6_unknown_exposure_audit_v4/"
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
    s = d["candidate_phrase_support"]

    print("=" * 108)
    print("P6 UNKNOWN EXPOSURE DATA AUDIT v4 REVIEW")
    print("=" * 108)
    print(f"train rows:             {c['train_rows']:,}")
    print(f"candidate rows:         {c['candidate_rows']:,}")
    print(f"candidate phrases:      {c['candidate_phrase_count']:,}")
    print(f"candidate speakers:     {c['candidate_speaker_count']:,}")
    print(f"domains:                {c['candidate_domain_counts']}")
    print("-" * 108)
    print(
        f"support median/p90/max: "
        f"{s['median']:.1f} / {s['p90']:.1f} / {s['max']}"
    )
    print(
        f"phrases >=5/10/15:      "
        f"{s['ge_5']} / {s['ge_10']} / {s['ge_15']}"
    )
    print("-" * 108)
    print("Core30 overlap:         0")
    print("open-set DEV overlap:   0")
    print("generic_test accessed:  NO")
    print("P6 UNKNOWN EXPOSURE DATA AUDIT v4 REVIEW: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
