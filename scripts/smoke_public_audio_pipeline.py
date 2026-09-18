#!/usr/bin/env python
"""Real-data smoke test for the frozen P2-05B audio pipeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from papr_ssl.data.audio_pipeline import (
    collate_audio_examples,
    load_audio_example,
)
from papr_ssl.data.manifest_io import read_manifest_jsonl


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
        default=Path("artifacts/p2_05/audio_pipeline_smoke.json"),
    )
    return p.parse_args()


def pick_gsc(records):
    shortest = min(records, key=lambda r: r.num_frames)
    exact = next(r for r in records if r.num_frames == 16000)
    return [shortest, exact]


def pick_mdsc(records):
    shortest = min(records, key=lambda r: r.num_frames)
    longest = max(records, key=lambda r: r.num_frames)
    stereo = next((r for r in records if r.num_channels > 1), None)
    picks = [shortest, longest]
    if stereo is not None and stereo.utt_id not in {x.utt_id for x in picks}:
        picks.append(stereo)
    return picks


def example_summary(ex):
    return {
        "utt_id": ex.record.utt_id,
        "dataset": ex.record.dataset,
        "stored_num_samples": int(ex.waveform.numel()),
        "valid_num_samples": int(ex.valid_num_samples),
        "source_channels": int(ex.record.num_channels),
        "duration_sec": float(ex.record.duration_sec),
        "split": ex.record.split,
        "domain": ex.record.domain,
        "role": ex.record.role,
    }


def main() -> int:
    args = parse_args()

    gsc = read_manifest_jsonl(args.gsc_manifest)
    mdsc = read_manifest_jsonl(args.mdsc_manifest)

    gsc_examples = [
        load_audio_example(r, {"gsc_v2": args.gsc_root})
        for r in pick_gsc(gsc)
    ]
    mdsc_examples = [
        load_audio_example(r, {"mdsc": args.mdsc_root})
        for r in pick_mdsc(mdsc)
    ]

    gsc_batch = collate_audio_examples(gsc_examples)
    mdsc_batch = collate_audio_examples(mdsc_examples)

    report = {
        "schema": "papr_ssl.audio_pipeline_smoke.v1",
        "phase": "P2-05B",
        "gsc_examples": [example_summary(x) for x in gsc_examples],
        "mdsc_examples": [example_summary(x) for x in mdsc_examples],
        "gsc_batch": {
            "shape": list(gsc_batch.waveforms.shape),
            "lengths": gsc_batch.lengths.tolist(),
            "valid_samples_per_row": [
                int(x) for x in gsc_batch.waveform_mask.sum(dim=1).tolist()
            ],
        },
        "mdsc_batch": {
            "shape": list(mdsc_batch.waveforms.shape),
            "lengths": mdsc_batch.lengths.tolist(),
            "valid_samples_per_row": [
                int(x) for x in mdsc_batch.waveform_mask.sum(dim=1).tolist()
            ],
        },
        "gate": "PASS",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("PAPR-SSL P2-05B REAL AUDIO PIPELINE SMOKE")
    print("=" * 88)
    print("GSC examples:")
    for row in report["gsc_examples"]:
        print(
            f"  {row['utt_id']} | stored={row['stored_num_samples']} "
            f"valid={row['valid_num_samples']} "
            f"channels={row['source_channels']}"
        )
    print(f"GSC batch:  {report['gsc_batch']}")
    print("-" * 88)
    print("MDSC examples:")
    for row in report["mdsc_examples"]:
        print(
            f"  {row['utt_id']} | stored={row['stored_num_samples']} "
            f"valid={row['valid_num_samples']} "
            f"channels={row['source_channels']}"
        )
    print(f"MDSC batch: {report['mdsc_batch']}")
    print("-" * 88)
    print(f"Gate:       {report['gate']}")
    print(f"Report:     {args.out}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
