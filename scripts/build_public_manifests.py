#!/usr/bin/env python
"""Build PAPR-SSL unified JSONL manifests for GSC v2 and MDSC."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from papr_ssl.data.gsc_adapter import GSCAdapter
from papr_ssl.data.manifest_io import write_manifest_jsonl
from papr_ssl.data.manifest_schema import AudioManifestRecord
from papr_ssl.data.mdsc_adapter import MDSCAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gsc-root",
        type=Path,
        default=Path("datasets/public/speech_commands_v2"),
    )
    parser.add_argument(
        "--mdsc-root",
        type=Path,
        default=Path("datasets/public/mdsc"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/p2_04/manifests"),
    )
    parser.add_argument(
        "--include-gsc-background-noise",
        action="store_true",
        help="Serialize the six GSC background-noise assets as record_type=noise.",
    )
    return parser.parse_args()


def summarize(records: list[AudioManifestRecord]) -> dict:
    return {
        "count": len(records),
        "datasets": dict(Counter(r.dataset for r in records)),
        "splits": dict(Counter(r.split for r in records)),
        "domains": dict(Counter(r.domain for r in records)),
        "roles": dict(Counter(r.role for r in records)),
        "record_types": dict(Counter(r.record_type for r in records)),
        "languages": dict(Counter(r.language for r in records)),
        "channels": dict(Counter(str(r.num_channels) for r in records)),
        "sample_rates": dict(Counter(str(r.sample_rate_hz) for r in records)),
        "unique_speakers": len(
            {r.speaker_id for r in records if r.speaker_id is not None}
        ),
        "unique_labels": len(
            {r.label for r in records if r.label is not None}
        ),
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Building GSC v2 manifest...")
    gsc_records = list(
        GSCAdapter(
            args.gsc_root,
            include_background_noise=args.include_gsc_background_noise,
        ).iter_records()
    )

    print("Building MDSC manifest...")
    mdsc_records = list(MDSCAdapter(args.mdsc_root).iter_records())

    gsc_out = args.out_dir / "gsc_v2.jsonl"
    mdsc_out = args.out_dir / "mdsc.jsonl"
    summary_out = args.out_dir / "manifest_summary.json"

    gsc_count = write_manifest_jsonl(gsc_records, gsc_out)
    mdsc_count = write_manifest_jsonl(mdsc_records, mdsc_out)

    summary = {
        "phase": "P2-04B",
        "gsc_v2": summarize(gsc_records),
        "mdsc": summarize(mdsc_records),
        "files": {
            "gsc_v2": str(gsc_out),
            "mdsc": str(mdsc_out),
        },
    }
    summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("PAPR-SSL P2-04B MANIFEST BUILD")
    print("=" * 88)
    print(f"GSC rows:   {gsc_count}")
    print(f"MDSC rows:  {mdsc_count}")
    print(f"GSC:        {gsc_out}")
    print(f"MDSC:       {mdsc_out}")
    print(f"Summary:    {summary_out}")
    print("-" * 88)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
