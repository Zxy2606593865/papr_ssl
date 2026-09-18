#!/usr/bin/env python
"""Run P2-07C personalized MDSC protocol integrity validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from papr_ssl.data.manifest_io import read_manifest_jsonl
from papr_ssl.data.protocol_integrity import (
    analyze_mdsc_personalized_protocol,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mdsc-manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/p2_07/task_views/"
            "mdsc_personalized_protocol_integrity.json"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    records = read_manifest_jsonl(args.mdsc_manifest)
    report = analyze_mdsc_personalized_protocol(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("PAPR-SSL P2-07C MDSC PERSONALIZED PROTOCOL INTEGRITY")
    print("=" * 100)
    print(f"Scoped records: {report['scope_record_count']}")
    print(f"Speakers:       {report['speaker_count']}")
    print(
        "Split speakers: "
        f"dev={report['split_speaker_counts']['dev']} "
        f"test={report['split_speaker_counts']['test']}"
    )
    print(
        "Enroll/wake observed counts: "
        f"{report['target_repetition_distribution']['enrollment_per_wake_observed_values']}"
    )
    print(
        "Eval/wake observed counts:   "
        f"{report['target_repetition_distribution']['eval_per_wake_observed_values']}"
    )
    print("-" * 100)

    for row in report["speakers"]:
        print(
            f"{row['speaker_id']:10s} split={row['split']:4s} | "
            f"target enroll={row['target_enrollment_count']:3d} "
            f"eval={row['target_eval_count']:3d} | "
            f"wake coverage enroll={row['enrollment_wake_coverage']:2d}/10 "
            f"eval={row['eval_wake_coverage']:2d}/10 | "
            f"nonwake enroll={row['nonwake_enrollment_count']:3d} "
            f"eval={row['nonwake_eval_count']:3d}"
        )

    print("-" * 100)
    for key, value in report["gate"].items():
        print(f"{key:48s} {value}")
    print(f"Report: {args.out}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
