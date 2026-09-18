#!/usr/bin/env python
"""H2-00 — Enrollment-statistics readiness audit.

Purpose
-------
H2 intends to use per-user/per-phrase LOO statistics and shrinkage.
Those statistics need more than 1-2 enrollment samples:

- n=1: no leave-one-out statistic exists.
- n=2: the two LOO cosine values are symmetric/equal, so class-specific
       variance is not informative.
- n>=3: user/phrase-specific dispersion starts to become estimable.

This metadata-only audit asks whether MDSC TRAIN/DEV contain enough repeated
recordings per speaker x Core30 phrase for n=3/4/5 support + >=1 query.

TEST/generic_test remains sealed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SUPPORT_LEVELS = (1, 2, 3, 4, 5)
MIN_INTENTS = 2


def normalize_phrase(text):
    if text is None:
        return ""
    x = str(text).replace("<p>", "")
    x = re.sub(r"\s+", "", x)
    return x.casefold()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get(row, keys, default=""):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return default


def row_split(row):
    return str(get(row, ("split",))).strip().lower()


def row_speaker(row):
    v = get(row, ("speaker_id", "speaker", "user_id", "subject_id"))
    if v == "":
        raise KeyError(f"No speaker id in keys={sorted(row)}")
    return str(v)


def row_phrase(row):
    return normalize_phrase(get(row, ("transcript", "standard_text", "text", "label")))


def task_phrase(row):
    return normalize_phrase(get(row, ("task_label", "label_text", "label", "transcript")))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def audit(rows, core30, split_name):
    counts = Counter()
    speakers = sorted({r["speaker"] for r in rows})
    for r in rows:
        if r["phrase"] in core30:
            counts[(r["speaker"], r["phrase"])] += 1

    pair_rows = []
    for s in speakers:
        for ph in sorted(core30):
            n = counts[(s, ph)]
            pair_rows.append({
                "split": split_name,
                "speaker_id": s,
                "phrase": ph,
                "utterance_count": n,
                **{
                    f"supports_{k}shot_plus_query": int(n >= k + 1)
                    for k in SUPPORT_LEVELS
                }
            })

    summary = []
    for k in SUPPORT_LEVELS:
        eligible = defaultdict(list)
        for (s, ph), n in counts.items():
            if n >= k + 1:
                eligible[s].append(ph)

        episode_speakers = [s for s in speakers if len(eligible[s]) >= MIN_INTENTS]
        eligible_pairs = sum(len(eligible[s]) for s in episode_speakers)
        total_pairs = len(speakers) * len(core30)
        covered_pair_fraction = eligible_pairs / total_pairs if total_pairs else 0.0

        # "Full 30-intent speaker" means every Core30 class supports k-shot + query.
        full_speakers = [s for s in speakers if len(eligible[s]) == len(core30)]

        summary.append({
            "split": split_name,
            "support_n": k,
            "episode_speakers_min2_intents": len(episode_speakers),
            "full_30intent_speakers": len(full_speakers),
            "eligible_speaker_phrase_pairs": eligible_pairs,
            "all_speaker_phrase_pairs": total_pairs,
            "eligible_pair_fraction": covered_pair_fraction,
            "loo_mean_available": int(k >= 2),
            "class_specific_variance_informative": int(k >= 3),
            "status_for_full_h2_stats": (
                "FEASIBLE" if k >= 3 and eligible_pairs > 0 else "NOT_FEASIBLE"
            ),
        })

    return summary, pair_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mdsc-manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--core30-index",
        type=Path,
        default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h2_00_enrollment_stats_readiness"),
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    core30 = set()
    for row in read_jsonl(args.core30_index):
        if row_split(row) in ("train", "dev"):
            ph = task_phrase(row)
            if ph:
                core30.add(ph)
    if len(core30) != 30:
        raise RuntimeError(f"Expected 30 Core30 phrases, got {len(core30)}")

    train, dev = [], []
    test_ignored = 0
    for row in read_jsonl(args.mdsc_manifest):
        sp = row_split(row)
        if sp == "test":
            test_ignored += 1
            continue
        if sp not in ("train", "dev"):
            continue
        rec = {
            "speaker": row_speaker(row),
            "phrase": row_phrase(row),
        }
        (train if sp == "train" else dev).append(rec)

    train_summary, train_pairs = audit(train, core30, "train")
    dev_summary, dev_pairs = audit(dev, core30, "dev")

    write_csv(args.output_dir/"readiness_summary.csv", train_summary + dev_summary)
    write_csv(args.output_dir/"speaker_phrase_counts.csv", train_pairs + dev_pairs)

    out = {
        "schema": "papr_ssl.h2_00_enrollment_stats_readiness.v1",
        "scientific_rule": {
            "n1": "No LOO statistic exists.",
            "n2": "LOO similarities are symmetric/equal; class-specific variance is uninformative.",
            "n3plus": "Class-specific dispersion becomes estimable, still with high small-sample uncertainty.",
        },
        "train": train_summary,
        "dev": dev_summary,
        "test_rows_ignored": test_ignored,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir/"summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    def show(title, rows):
        print(title)
        print(
            f"{'n':>3s} {'ep_spk':>7s} {'full30':>7s} {'pairs':>7s} "
            f"{'fraction':>9s} {'variance?':>10s} {'H2 stats':>12s}"
        )
        for r in rows:
            print(
                f"{r['support_n']:3d} "
                f"{r['episode_speakers_min2_intents']:7d} "
                f"{r['full_30intent_speakers']:7d} "
                f"{r['eligible_speaker_phrase_pairs']:7d} "
                f"{100*r['eligible_pair_fraction']:8.2f}% "
                f"{'YES' if r['class_specific_variance_informative'] else 'NO':>10s} "
                f"{r['status_for_full_h2_stats']:>12s}"
            )

    print("="*100)
    print("H2-00 ENROLLMENT STATISTICS READINESS AUDIT")
    print("="*100)
    show("TRAIN", train_summary)
    print("-"*100)
    show("DEV", dev_summary)
    print("-"*100)
    print("Interpretation:")
    print("  n=1: no LOO")
    print("  n=2: LOO mean exists, but user/phrase variance is not informative")
    print("  n>=3: full LOO + shrinkage statistics become scientifically meaningful")
    print(f"test rows ignored:     {test_ignored}")
    print("generic_test accessed: NO")
    print("H2-00 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
