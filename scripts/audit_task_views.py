#!/usr/bin/env python
"""P2-07A audit GSC/MDSC label inventories and candidate task views."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from papr_ssl.data.manifest_io import read_manifest_jsonl
from papr_ssl.data.task_view_audit import (
    build_label_stats,
    build_mdsc_candidate_views,
    summarize_views,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gsc-manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/gsc_v2.jsonl"),
    )
    p.add_argument(
        "--mdsc-manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/p2_07"),
    )
    p.add_argument("--top-n", type=int, default=50)
    return p.parse_args()


def write_stats_csv(path, stats):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "count",
        "speaker_count",
        "train_count",
        "enrollment_count",
        "eval_count",
        "control_count",
        "dysarthria_count",
        "train_speaker_count",
        "enrollment_speaker_count",
        "eval_speaker_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in stats:
            w.writerow(row.to_dict())


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gsc = read_manifest_jsonl(args.gsc_manifest)
    mdsc = read_manifest_jsonl(args.mdsc_manifest)

    gsc_stats = build_label_stats(gsc)
    mdsc_stats = build_label_stats(mdsc)

    gsc_speakers = len({r.speaker_id for r in gsc if r.speaker_id})
    mdsc_speakers = len({r.speaker_id for r in mdsc if r.speaker_id})

    mdsc_views = build_mdsc_candidate_views(
        mdsc_stats,
        total_speakers=mdsc_speakers,
    )

    gsc_labels = sorted(s.label for s in gsc_stats)

    report = {
        "schema": "papr_ssl.task_view_audit.v1",
        "phase": "P2-07A",
        "gsc_v2": {
            "record_count": len(gsc),
            "speaker_count": gsc_speakers,
            "unique_labels": len(gsc_stats),
            "all_labels": gsc_labels,
            "recommended_development_view": {
                "name": "gsc_all_35_closed_set",
                "labels": gsc_labels,
                "note": (
                    "Descriptive project view for backbone/embedding screening. "
                    "This does not redefine the official dataset."
                ),
            },
        },
        "mdsc": {
            "record_count": len(mdsc),
            "speaker_count": mdsc_speakers,
            "unique_labels": len(mdsc_stats),
            "candidate_view_counts": summarize_views(mdsc_views),
            "candidate_views": mdsc_views,
            "top_by_count": [
                s.to_dict()
                for s in sorted(
                    mdsc_stats,
                    key=lambda x: (-x.count, x.label),
                )[: args.top_n]
            ],
            "top_by_speaker_coverage": [
                s.to_dict()
                for s in sorted(
                    mdsc_stats,
                    key=lambda x: (-x.speaker_count, -x.count, x.label),
                )[: args.top_n]
            ],
        },
        "gate": {
            "gsc_unique_labels_is_35": len(gsc_stats) == 35,
            "mdsc_unique_labels_is_3858": len(mdsc_stats) == 3858,
            "mdsc_has_enrollment_eval_overlap": (
                len(mdsc_views["enrollment_and_eval_overlap"]) > 0
            ),
        },
        "interpretation_boundary": (
            "Candidate views are set-theoretic inventories only. "
            "P2-07A does not automatically declare any MDSC phrase a target, "
            "non-target, wake word, command, or unknown class."
        ),
    }

    report["gate"]["overall"] = (
        "PASS"
        if all(
            v is True
            for k, v in report["gate"].items()
            if k != "overall"
        )
        else "FAIL"
    )

    json_path = args.out_dir / "task_view_audit.json"
    csv_path = args.out_dir / "mdsc_label_stats.csv"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_stats_csv(csv_path, mdsc_stats)

    print("=" * 92)
    print("PAPR-SSL P2-07A TASK VIEW / LABEL INVENTORY AUDIT")
    print("=" * 92)
    print(f"GSC labels:       {len(gsc_stats)}")
    print(f"MDSC labels:      {len(mdsc_stats)}")
    print(f"MDSC speakers:    {mdsc_speakers}")
    print("-" * 92)
    for name, count in report["mdsc"]["candidate_view_counts"].items():
        print(f"{name:38s} {count}")
    print("-" * 92)
    print("Top MDSC labels by frequency:")
    for s in report["mdsc"]["top_by_count"][:20]:
        print(
            f"  {s['label']!r}: count={s['count']}, "
            f"speakers={s['speaker_count']}, "
            f"train={s['train_count']}, "
            f"enroll={s['enrollment_count']}, "
            f"eval={s['eval_count']}, "
            f"control={s['control_count']}, "
            f"dys={s['dysarthria_count']}"
        )
    print("-" * 92)
    print(f"Gate:             {report['gate']['overall']}")
    print(f"JSON:             {json_path}")
    print(f"CSV:              {csv_path}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
