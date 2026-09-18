#!/usr/bin/env python
"""P2-07B materialize frozen task-semantic indexes from raw manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from papr_ssl.data.manifest_io import read_manifest_jsonl
from papr_ssl.data.task_views import (
    GSC_ALL_35,
    MDSC_WAKE_WORDS,
    canonical_mdsc_wake_word,
    gsc_closed_set_index,
    mdsc_common30_index,
    mdsc_personalized_wws_index,
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
        default=Path("artifacts/p2_07/task_views"),
    )
    return p.parse_args()


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row.to_dict(), ensure_ascii=False)
                + "\n"
            )


def summarize(rows):
    return {
        "count": len(rows),
        "splits": dict(Counter(r.split for r in rows)),
        "roles": dict(Counter(r.role for r in rows)),
        "domains": dict(Counter(r.domain for r in rows)),
        "task_labels": dict(Counter(r.task_label for r in rows)),
        "target_count": sum(int(r.is_target) for r in rows),
        "non_target_count": sum(int(not r.is_target) for r in rows),
        "speaker_count": len(
            {r.speaker_id for r in rows if r.speaker_id is not None}
        ),
    }


def canonical_wake_inventory(mdsc):
    counts = Counter()
    raw_variants = defaultdict(Counter)
    role_counts = defaultdict(Counter)
    domain_counts = defaultdict(Counter)

    for r in mdsc:
        if r.record_type != "speech":
            continue
        canonical = canonical_mdsc_wake_word(r.label)
        if canonical is None:
            continue

        counts[canonical] += 1
        raw_variants[canonical][r.label] += 1
        role_counts[canonical][r.role] += 1
        domain_counts[canonical][r.domain] += 1

    return {
        label: {
            "count": counts[label],
            "raw_variants": dict(raw_variants[label]),
            "roles": dict(role_counts[label]),
            "domains": dict(domain_counts[label]),
        }
        for label in MDSC_WAKE_WORDS
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gsc = read_manifest_jsonl(args.gsc_manifest)
    mdsc = read_manifest_jsonl(args.mdsc_manifest)

    gsc_view = gsc_closed_set_index(gsc)
    personalized = mdsc_personalized_wws_index(mdsc)
    common30 = mdsc_common30_index(mdsc)

    wake_inventory = canonical_wake_inventory(mdsc)

    wake_counts = {
        label: info["count"]
        for label, info in wake_inventory.items()
    }

    wake_count_gate = (
        len(wake_counts) == 10
        and all(v == 230 for v in wake_counts.values())
        and sum(wake_counts.values()) == 2300
    )

    # Every canonical wake word should expose the same dataset-level role counts:
    # 170 train + 8 enrollment + 52 eval = 230.
    wake_role_gate = all(
        info["roles"].get("train", 0) == 170
        and info["roles"].get("enrollment", 0) == 8
        and info["roles"].get("eval", 0) == 52
        for info in wake_inventory.values()
    )

    personalized_by_split = defaultdict(list)
    for row in personalized:
        personalized_by_split[row.split].append(row)

    # Personalized view must contain both positive and negative eval records.
    eval_binary_gate = True
    for split in ("dev", "test"):
        eval_rows = [
            r for r in personalized_by_split[split]
            if r.role == "eval"
        ]
        if not eval_rows:
            eval_binary_gate = False
            continue
        if not any(r.is_target for r in eval_rows):
            eval_binary_gate = False
        if not any(not r.is_target for r in eval_rows):
            eval_binary_gate = False

    report = {
        "schema": "papr_ssl.task_policy.v1",
        "phase": "P2-07B",
        "frozen_policy": {
            "gsc": {
                "view": "gsc_all_35_closed_set",
                "class_count": 35,
                "labels": list(GSC_ALL_35),
                "purpose": "generic SSL / embedding development screening",
            },
            "mdsc": {
                "primary_view": "mdsc_personalized_wws_10",
                "task": "speaker-dependent dysarthria wake-up word spotting",
                "wake_word_count": 10,
                "wake_words": list(MDSC_WAKE_WORDS),
                "positive_rule": (
                    "task-normalized transcript matches one of the 10 wake words"
                ),
                "negative_rule": (
                    "all other transcripts are non-wake for binary WWS evaluation"
                ),
                "personalized_scope": (
                    "domain=dysarthria, split in {dev,test}, "
                    "role in {enrollment,eval}"
                ),
                "enrollment_policy": (
                    "target enrollment utterances construct speaker-specific "
                    "wake-word prototypes; non-wake enrollment remains negative/"
                    "calibration context and is never merged into a wake prototype"
                ),
                "evaluation_policy": (
                    "eval wake samples are positives; eval non-wake samples are "
                    "negatives; threshold selection belongs to dev, not test"
                ),
                "optional_diagnostic_view": "mdsc_common_30_closed_set",
            },
            "normalization": {
                "raw_manifest_rewrite": False,
                "remove_pause_token": "<p>",
                "remove_whitespace": True,
                "casefold": True,
                "semantic_synonym_merging": False,
            },
        },
        "observed_wake_inventory": wake_inventory,
        "views": {
            "gsc_all_35_closed_set": summarize(gsc_view),
            "mdsc_personalized_wws_10": summarize(personalized),
            "mdsc_common_30_closed_set": summarize(common30),
        },
        "gate": {
            "gsc_has_35_classes": (
                len({r.task_label for r in gsc_view}) == 35
            ),
            "mdsc_has_10_canonical_wake_words": wake_count_gate,
            "mdsc_wake_role_counts_match": wake_role_gate,
            "personalized_eval_has_positive_and_negative": eval_binary_gate,
        },
    }

    report["gate"]["overall"] = (
        "PASS"
        if all(
            value is True
            for key, value in report["gate"].items()
            if key != "overall"
        )
        else "FAIL"
    )

    write_jsonl(
        args.out_dir / "gsc_all35.index.jsonl",
        gsc_view,
    )
    write_jsonl(
        args.out_dir / "mdsc_personalized_wws10.index.jsonl",
        personalized,
    )
    write_jsonl(
        args.out_dir / "mdsc_common30.index.jsonl",
        common30,
    )

    summary_path = args.out_dir / "task_policy_summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 96)
    print("PAPR-SSL P2-07B FROZEN TASK POLICY")
    print("=" * 96)
    print("Canonical MDSC wake words:")
    for label, info in wake_inventory.items():
        print(
            f"  {label!r}: count={info['count']}, "
            f"roles={info['roles']}, "
            f"raw_variants={info['raw_variants']}"
        )
    print("-" * 96)
    print(
        "GSC all-35:          "
        f"{report['views']['gsc_all_35_closed_set']}"
    )
    print(
        "MDSC personalized:   "
        f"{report['views']['mdsc_personalized_wws_10']}"
    )
    print(
        "MDSC common-30:      "
        f"{report['views']['mdsc_common_30_closed_set']}"
    )
    print("-" * 96)
    print(f"Gate:                 {report['gate']['overall']}")
    print(f"Summary:              {summary_path}")
    print("=" * 96)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
