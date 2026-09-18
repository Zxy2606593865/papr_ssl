#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

from papr_ssl.data.manifest_io import read_manifest_jsonl
from papr_ssl.data.phrase_inventory_audit import (
    build_mdsc_phrase_inventory,
    summarize_phrase_inventory,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/p2_07/phrase_inventory"),
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_mdsc_phrase_inventory(read_manifest_jsonl(args.manifest))
    summary = summarize_phrase_inventory(inventory)

    summary_path = args.out_dir / "mdsc_phrase_inventory_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = args.out_dir / "mdsc_phrase_inventory.csv"
    fields = [
        "phrase", "utterance_count", "speaker_count",
        "control_utterance_count", "dysarthria_utterance_count",
        "control_speaker_count", "dysarthria_speaker_count",
        "train_utterance_count", "train_speaker_count",
        "control_train_speaker_count", "dysarthria_train_speaker_count",
        "dev_utterance_count", "test_utterance_count",
        "enrollment_count", "eval_count", "raw_variant_count",
        "is_canonical_wake_word",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in inventory:
            d = row.to_dict()
            w.writerow({k: d[k] for k in fields})

    print("=" * 88)
    print("P2-07D MDSC PHRASE INVENTORY")
    print("=" * 88)
    print("phrases:", summary["inventory"]["phrase_count"])
    print("cross-domain:", summary["inventory"]["cross_domain_phrase_count"])
    print("wake phrases:", summary["inventory"]["wake_phrase_count"])
    print("-" * 88)
    print("TRAIN-ONLY SUPPORT SWEEP")
    for r in summary["threshold_sweep"]:
        print(
            f"C>={r['min_control_train_speakers']:2d}, "
            f"D>={r['min_dysarthria_train_speakers']:2d} -> "
            f"{r['phrase_count']:4d} phrases, "
            f"{r['non_wake_phrase_count']:4d} non-wake, "
            f"{r['train_utterance_count']:5d} train utts"
        )
    print("-" * 88)
    print("DESCRIPTIVE ONLY: no phrase set frozen, no semantic merging.")
    print("summary:", summary_path)
    print("csv:    ", csv_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
