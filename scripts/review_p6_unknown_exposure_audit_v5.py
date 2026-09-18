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
            "artifacts/p6_unknown_exposure_audit_v5/"
            "p6_unknown_exposure_audit.json"
        ),
    )
    args = p.parse_args()

    d = json.loads(args.report.read_text(encoding="utf-8"))
    if d["generic_test"] != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    c = d["counts"]
    leak = d["development_leakage_checks"]
    if c["core30_phrase_count"] != 30:
        raise RuntimeError("bad Core30 count")
    if c["open_dev_utt_count"] != 1178:
        raise RuntimeError("bad open DEV utterance count")
    if c["open_dev_real_phrase_count"] < 10:
        raise RuntimeError("open DEV real phrase recovery looks wrong")
    if leak["candidate_vs_core30_phrase_overlap"] != 0:
        raise RuntimeError("Core30 leakage")
    if leak["candidate_vs_open_dev_phrase_overlap"] != 0:
        raise RuntimeError("open DEV phrase leakage")

    s = d["candidate_phrase_support"]

    print("=" * 108)
    print("P6 UNKNOWN EXPOSURE DATA AUDIT v5 REVIEW")
    print("=" * 108)
    print(f"MDSC rows:              {c['mdsc_rows']:,}")
    print(f"MDSC train rows:        {c['mdsc_train_rows']:,}")
    print(f"Core30 phrases:         {c['core30_phrase_count']}")
    print(f"open DEV utterances:    {c['open_dev_utt_count']:,}")
    print(f"open DEV real phrases:  {c['open_dev_real_phrase_count']:,}")
    print("-" * 108)
    print(f"candidate rows:         {c['candidate_rows']:,}")
    print(f"candidate phrases:      {c['candidate_phrase_count']:,}")
    print(f"candidate speakers:     {c['candidate_speaker_count']:,}")
    print(f"domains:                {c['candidate_domain_counts']}")
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
    print("P6 UNKNOWN EXPOSURE DATA AUDIT v5 REVIEW: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
