#!/usr/bin/env python
"""PAPR-SSL P2-05A manifest QA and label-semantics audit.

Checks:
- JSONL readability and schema validation
- duplicate utt_id / duplicate audio_relpath
- file existence
- split/domain/role cross-tabs
- duration and channel distributions
- MDSC label/transcript frequency structure
- explicit listing of stereo MDSC items

This script is read-only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from papr_ssl.data.manifest_io import read_manifest_jsonl
from papr_ssl.data.manifest_schema import AudioManifestRecord


def parse_args() -> argparse.Namespace:
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
        "--gsc-root",
        type=Path,
        default=Path("datasets/public/speech_commands_v2"),
    )
    p.add_argument(
        "--mdsc-root",
        type=Path,
        default=Path("datasets/public/mdsc"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/p2_05/manifest_qa.json"),
    )
    p.add_argument("--top-labels", type=int, default=100)
    return p.parse_args()


def cross_tab(records: list[AudioManifestRecord], *fields: str) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in records:
        key = "|".join(str(getattr(r, f)) for f in fields)
        c[key] += 1
    return dict(sorted(c.items()))


def duration_bins(records: list[AudioManifestRecord]) -> dict[str, int]:
    c = Counter()
    for r in records:
        d = r.duration_sec
        if d < 1.0:
            c["<1s"] += 1
        elif d <= 1.0:
            c["=1s"] += 1
        elif d <= 2.0:
            c["1-2s"] += 1
        elif d <= 4.0:
            c["2-4s"] += 1
        elif d <= 8.0:
            c["4-8s"] += 1
        else:
            c[">8s"] += 1
    order = ["<1s", "=1s", "1-2s", "2-4s", "4-8s", ">8s"]
    return {k: c[k] for k in order}


def duplicates(values: list[str]) -> list[str]:
    c = Counter(values)
    return sorted(k for k, v in c.items() if v > 1)


def file_existence(
    records: list[AudioManifestRecord],
    root: Path,
) -> dict[str, Any]:
    missing = []
    for r in records:
        p = root / r.audio_relpath
        if not p.is_file():
            missing.append(r.audio_relpath)
    return {
        "missing_count": len(missing),
        "missing_examples": missing[:50],
    }


def general_qa(
    records: list[AudioManifestRecord],
    root: Path,
) -> dict[str, Any]:
    utt_dups = duplicates([r.utt_id for r in records])
    path_dups = duplicates([r.audio_relpath for r in records])

    return {
        "count": len(records),
        "duplicate_utt_id_count": len(utt_dups),
        "duplicate_utt_id_examples": utt_dups[:50],
        "duplicate_audio_relpath_count": len(path_dups),
        "duplicate_audio_relpath_examples": path_dups[:50],
        "file_existence": file_existence(records, root),
        "split": dict(Counter(r.split for r in records)),
        "domain": dict(Counter(r.domain for r in records)),
        "role": dict(Counter(r.role for r in records)),
        "channels": dict(Counter(str(r.num_channels) for r in records)),
        "sample_rates": dict(Counter(str(r.sample_rate_hz) for r in records)),
        "duration_bins": duration_bins(records),
        "cross_split_domain": cross_tab(records, "split", "domain"),
        "cross_split_role": cross_tab(records, "split", "role"),
        "cross_domain_role": cross_tab(records, "domain", "role"),
        "unique_speakers": len({r.speaker_id for r in records if r.speaker_id}),
        "unique_labels": len({r.label for r in records if r.label is not None}),
    }


def mdsc_label_qa(
    records: list[AudioManifestRecord],
    top_n: int,
) -> dict[str, Any]:
    label_counts = Counter(r.label for r in records if r.label is not None)

    labels_by_split: dict[str, set[str]] = defaultdict(set)
    labels_by_domain: dict[str, set[str]] = defaultdict(set)
    labels_by_role: dict[str, set[str]] = defaultdict(set)

    for r in records:
        if r.label is None:
            continue
        labels_by_split[r.split].add(r.label)
        labels_by_domain[r.domain].add(r.label)
        labels_by_role[r.role].add(r.label)

    speakers_by_label: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.label is not None and r.speaker_id is not None:
            speakers_by_label[r.label].add(r.speaker_id)

    freq_hist = Counter(label_counts.values())

    top = [
        {
            "label": label,
            "count": count,
            "speaker_count": len(speakers_by_label[label]),
        }
        for label, count in label_counts.most_common(top_n)
    ]

    singletons = sorted(label for label, n in label_counts.items() if n == 1)

    # Labels shared across important partitions.
    train = labels_by_split["train"]
    dev = labels_by_split["dev"]
    test = labels_by_split["test"]
    control = labels_by_domain["control"]
    dys = labels_by_domain["dysarthria"]
    enrollment = labels_by_role["enrollment"]
    eval_labels = labels_by_role["eval"]

    return {
        "unique_label_count": len(label_counts),
        "frequency_histogram_count_to_num_labels": {
            str(k): v for k, v in sorted(freq_hist.items())
        },
        "top_labels": top,
        "singleton_label_count": len(singletons),
        "singleton_examples": singletons[:100],
        "unique_by_split": {k: len(v) for k, v in labels_by_split.items()},
        "unique_by_domain": {k: len(v) for k, v in labels_by_domain.items()},
        "unique_by_role": {k: len(v) for k, v in labels_by_role.items()},
        "overlap": {
            "train_dev": len(train & dev),
            "train_test": len(train & test),
            "dev_test": len(dev & test),
            "control_dysarthria": len(control & dys),
            "enrollment_eval": len(enrollment & eval_labels),
        },
        "semantic_note": (
            "A high unique-label count may indicate that raw transcript text "
            "contains many sentence/phrase types and should not automatically "
            "be treated as the final KWS class inventory."
        ),
    }


def stereo_items(records: list[AudioManifestRecord]) -> list[dict[str, Any]]:
    return [
        {
            "utt_id": r.utt_id,
            "audio_relpath": r.audio_relpath,
            "speaker_id": r.speaker_id,
            "label": r.label,
            "split": r.split,
            "domain": r.domain,
            "role": r.role,
            "num_channels": r.num_channels,
            "duration_sec": r.duration_sec,
        }
        for r in records
        if r.num_channels != 1
    ]


def main() -> int:
    args = parse_args()

    gsc_records = read_manifest_jsonl(args.gsc_manifest)
    mdsc_records = read_manifest_jsonl(args.mdsc_manifest)

    report = {
        "schema": "papr_ssl.manifest_qa.v1",
        "phase": "P2-05A",
        "read_only": True,
        "gsc_v2": general_qa(gsc_records, args.gsc_root),
        "mdsc": general_qa(mdsc_records, args.mdsc_root),
        "mdsc_label_semantics": mdsc_label_qa(mdsc_records, args.top_labels),
        "mdsc_stereo_items": stereo_items(mdsc_records),
    }

    # Gate criteria intentionally cover structural/data-integrity errors only.
    gsc_ok = (
        report["gsc_v2"]["duplicate_utt_id_count"] == 0
        and report["gsc_v2"]["duplicate_audio_relpath_count"] == 0
        and report["gsc_v2"]["file_existence"]["missing_count"] == 0
        and report["gsc_v2"]["count"] == 105829
    )
    mdsc_ok = (
        report["mdsc"]["duplicate_utt_id_count"] == 0
        and report["mdsc"]["duplicate_audio_relpath_count"] == 0
        and report["mdsc"]["file_existence"]["missing_count"] == 0
        and report["mdsc"]["count"] == 18630
    )

    report["gate"] = {
        "gsc_v2": "PASS" if gsc_ok else "FAIL",
        "mdsc": "PASS" if mdsc_ok else "FAIL",
        "overall": "PASS" if gsc_ok and mdsc_ok else "FAIL",
        "note": (
            "Label-semantic findings are diagnostic and do not fail the "
            "structural manifest gate by themselves."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("PAPR-SSL P2-05A MANIFEST QA")
    print("=" * 88)
    print(f"GSC Gate:      {report['gate']['gsc_v2']}")
    print(f"MDSC Gate:     {report['gate']['mdsc']}")
    print(f"Overall:       {report['gate']['overall']}")
    print("-" * 88)
    print(f"GSC rows:      {report['gsc_v2']['count']}")
    print(f"GSC labels:    {report['gsc_v2']['unique_labels']}")
    print(f"GSC duration:  {report['gsc_v2']['duration_bins']}")
    print("-" * 88)
    print(f"MDSC rows:     {report['mdsc']['count']}")
    print(f"MDSC labels:   {report['mdsc']['unique_labels']}")
    print(f"MDSC duration: {report['mdsc']['duration_bins']}")
    print(f"MDSC stereo:   {len(report['mdsc_stereo_items'])}")
    print(
        "MDSC label overlap enrollment/eval: "
        f"{report['mdsc_label_semantics']['overlap']['enrollment_eval']}"
    )
    print("Top MDSC labels:")
    for row in report["mdsc_label_semantics"]["top_labels"][:20]:
        print(
            f"  {row['label']!r}: "
            f"count={row['count']}, speakers={row['speaker_count']}"
        )
    print("-" * 88)
    print(f"Report:        {args.out}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
