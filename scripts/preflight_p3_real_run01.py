#!/usr/bin/env python
"""Preflight the first real P3 run without loading the SSL model."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import soundfile as sf

from papr_ssl.training.teacher.real_experiment import (
    build_mdsc_audio_catalog,
    load_real_task_rows,
    qualified_utt_relpath,
    resolve_task_audio_path,
    summarize_real_rows,
)


EXPECTED_TRAIN = 3756
EXPECTED_DEV = 442
EXPECTED_CLASSES = 30
EXPECTED_MDSC_WAVS = 18630


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"
        ),
    )
    p.add_argument(
        "--mdsc-root",
        type=Path,
        default=Path("datasets/public/mdsc"),
    )
    p.add_argument("--check-all-audio", action="store_true")
    args = p.parse_args()

    rows = load_real_task_rows(
        args.task_index,
        splits=("train", "dev"),
    )
    summary = summarize_real_rows(rows)

    if summary["split_counts"].get("train") != EXPECTED_TRAIN:
        raise RuntimeError(
            "Core30 train count mismatch: "
            f"{summary['split_counts'].get('train')} != {EXPECTED_TRAIN}"
        )
    if summary["split_counts"].get("dev") != EXPECTED_DEV:
        raise RuntimeError(
            "Core30 dev count mismatch: "
            f"{summary['split_counts'].get('dev')} != {EXPECTED_DEV}"
        )
    if summary["class_count"] != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Core30 class count mismatch: "
            f"{summary['class_count']} != {EXPECTED_CLASSES}"
        )

    # Current canonical rows resolve directly from dataset-qualified utt_id.
    # Build the 18,630-WAV stem catalog only for any legacy/plain rows.
    legacy_rows = [
        row for row in rows
        if row.audio_relpath is None and qualified_utt_relpath(row) is None
    ]
    audio_catalog = None
    if legacy_rows:
        print("[1/2] Building legacy MDSC stem catalog...")
        audio_catalog = build_mdsc_audio_catalog(args.mdsc_root)
        if len(audio_catalog) != EXPECTED_MDSC_WAVS:
            raise RuntimeError(
                f"MDSC WAV catalog size mismatch: "
                f"{len(audio_catalog)} != {EXPECTED_MDSC_WAVS}"
            )
    else:
        print("[1/2] Canonical qualified utt_id detected; catalog fallback not needed.")

    print("[2/2] Resolving Core30 task rows to physical WAV files...")
    header_counts = Counter()
    for row in rows:
        path = resolve_task_audio_path(
            row=row,
            mdsc_root=args.mdsc_root,
            audio_catalog=audio_catalog,
        )
        if args.check_all_audio:
            info = sf.info(path)
            header_counts[(int(info.samplerate), int(info.channels))] += 1
            if int(info.samplerate) != 16000:
                raise RuntimeError(
                    f"non-16k audio found: {path} ({info.samplerate} Hz)"
                )

    print("=" * 96)
    print("PAPR-SSL P3 REAL RUN 01 PREFLIGHT")
    print("=" * 96)
    print(f"task index:                    {args.task_index}")
    print(f"MDSC root:                     {args.mdsc_root}")
    print(f"train rows:                    {summary['split_counts']['train']}")
    print(f"dev rows:                      {summary['split_counts']['dev']}")
    print(f"classes:                       {summary['class_count']}")
    print(f"rows with explicit path:       {summary['rows_with_explicit_path']}")
    print(f"rows with qualified utt_id:    {summary['rows_with_qualified_utt_id']}")
    print(f"legacy rows needing catalog:   {len(legacy_rows)}")
    print(f"audio files resolved:          {len(rows)} / {len(rows)}")
    print("test audio materialized:       NO")
    if args.check_all_audio:
        print(f"audio header groups:           {dict(header_counts)}")
    print("-" * 96)
    print("REAL RUN 01 PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
