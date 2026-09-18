#!/usr/bin/env python
"""H0-02 — Unseen-speaker personalized episode feasibility audit.

Corrected protocol
------------------
The official MDSC train/dev split is speaker-disjoint. For few-shot
personalization this is useful rather than fatal:

HEAD-FIT:
    use TRAIN speakers; within each speaker/phrase split distinct recordings
    into support and query.

UNSEEN-SPEAKER DEV:
    use DEV speakers only; for each unseen speaker, reserve N recordings of a
    phrase as support and the remaining recordings as query.

TEST:
    stays sealed until the entire protocol/head/calibration is frozen.

This script is metadata-only. It never loads audio, embeddings, or TEST rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

SHOT_LEVELS = (1, 2, 5, 10, 15)
MIN_INTENTS = 2


def normalize_phrase(text):
    if text is None:
        return ""
    x = str(text).replace("<p>", "")
    x = re.sub(r"\s+", "", x)
    return x.casefold()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"{path}:{n}: {e}") from e


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
        raise KeyError(f"No speaker field in keys={sorted(row)}")
    return str(v)


def row_utt(row):
    v = get(row, ("utt_id", "id", "audio_id"))
    if v == "":
        raise KeyError(f"No utt id field in keys={sorted(row)}")
    return str(v)


def row_phrase(row):
    return normalize_phrase(
        get(row, ("transcript", "standard_text", "text", "label"))
    )


def task_phrase(row):
    return normalize_phrase(
        get(row, ("task_label", "label_text", "label", "transcript"))
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def pct(x):
    return f"{100*x:.2f}%"


def audit_split(rows, core30, split_name):
    by_spk_phrase = Counter()
    by_spk_all_phrase = defaultdict(set)
    by_spk_rows = Counter()
    by_spk_outside_rows = Counter()
    by_spk_outside_phrases = defaultdict(set)

    for r in rows:
        s, ph = r["speaker_id"], r["phrase"]
        by_spk_rows[s] += 1
        if not ph:
            continue
        by_spk_all_phrase[s].add(ph)
        if ph in core30:
            by_spk_phrase[(s, ph)] += 1
        else:
            by_spk_outside_rows[s] += 1
            by_spk_outside_phrases[s].add(ph)

    speakers = sorted({r["speaker_id"] for r in rows})
    shot_summary = []
    speaker_shot_rows = []

    total_core_rows = sum(
        count for (s, ph), count in by_spk_phrase.items()
    )

    for shot in SHOT_LEVELS:
        eligible = defaultdict(dict)
        for (s, ph), count in by_spk_phrase.items():
            # N support + at least one independent query recording.
            if count >= shot + 1:
                eligible[s][ph] = count

        episode_speakers = sorted(
            s for s, pmap in eligible.items()
            if len(pmap) >= MIN_INTENTS
        )

        eligible_pairs = sum(len(eligible[s]) for s in episode_speakers)
        support_rows = shot * eligible_pairs
        remaining_known_queries = sum(
            count - shot
            for s in episode_speakers
            for count in eligible[s].values()
        )

        possible_known_rows = sum(
            count
            for s in episode_speakers
            for count in eligible[s].values()
        )

        clean_unknown_speakers = sum(
            by_spk_outside_rows[s] > 0 for s in episode_speakers
        )

        # Relative unknowns from unregistered Core30 phrases of same speaker.
        relative_unknown_rows = 0
        relative_unknown_phrases = 0
        for s in episode_speakers:
            registered = set(eligible[s])
            for ph in core30:
                if ph not in registered:
                    c = by_spk_phrase[(s, ph)]
                    if c > 0:
                        relative_unknown_rows += c
                        relative_unknown_phrases += 1

        intent_counts = [len(eligible[s]) for s in episode_speakers]
        query_counts = [
            sum(c-shot for c in eligible[s].values())
            for s in episode_speakers
        ]

        shot_summary.append({
            "split": split_name,
            "shot": shot,
            "episode_speakers_min2_intents": len(episode_speakers),
            "eligible_speaker_phrase_pairs": eligible_pairs,
            "support_rows_required": support_rows,
            "remaining_known_query_rows": remaining_known_queries,
            "eligible_known_rows_before_support_reservation": possible_known_rows,
            "coverage_of_all_core30_rows": (
                possible_known_rows / total_core_rows if total_core_rows else 0.0
            ),
            "speakers_with_clean_outside_core_unknown": clean_unknown_speakers,
            "relative_unknown_rows_from_nonregistered_core30": relative_unknown_rows,
            "relative_unknown_speaker_phrase_pairs": relative_unknown_phrases,
            "median_registered_intents": (
                float(median(intent_counts)) if intent_counts else 0.0
            ),
            "max_registered_intents": max(intent_counts) if intent_counts else 0,
            "median_remaining_known_queries": (
                float(median(query_counts)) if query_counts else 0.0
            ),
            "status": "FEASIBLE" if episode_speakers else "NOT_FEASIBLE",
        })

        for s in speakers:
            pmap = eligible.get(s, {})
            speaker_shot_rows.append({
                "split": split_name,
                "speaker_id": s,
                "shot": shot,
                "eligible_intents": len(pmap),
                "support_rows": shot * len(pmap),
                "remaining_known_query_rows": sum(c-shot for c in pmap.values()),
                "outside_core_unknown_rows": by_spk_outside_rows[s],
                "outside_core_unknown_phrases": len(by_spk_outside_phrases[s]),
                "episode_eligible": int(len(pmap) >= MIN_INTENTS),
            })

    phrase_count_rows = []
    for s in speakers:
        for ph in sorted(core30):
            c = by_spk_phrase[(s, ph)]
            phrase_count_rows.append({
                "split": split_name,
                "speaker_id": s,
                "phrase": ph,
                "utterance_count": c,
                **{
                    f"supports_{shot}shot_plus_query": int(c >= shot + 1)
                    for shot in SHOT_LEVELS
                },
            })

    return shot_summary, speaker_shot_rows, phrase_count_rows


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
        default=Path("artifacts/h0_02_unseen_speaker_episode_audit"),
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Core30 inventory, train/dev only.
    core30 = set()
    for r in read_jsonl(args.core30_index):
        if row_split(r) in ("train", "dev"):
            ph = task_phrase(r)
            if ph:
                core30.add(ph)
    if len(core30) != 30:
        raise RuntimeError(f"Expected Core30=30, got {len(core30)}")

    train, dev = [], []
    test_ignored = 0
    for r in read_jsonl(args.mdsc_manifest):
        sp = row_split(r)
        if sp == "test":
            test_ignored += 1
            continue
        if sp not in ("train", "dev"):
            continue
        row = {
            "utt_id": row_utt(r),
            "speaker_id": row_speaker(r),
            "phrase": row_phrase(r),
            "split": sp,
        }
        (train if sp == "train" else dev).append(row)

    train_spk = {r["speaker_id"] for r in train}
    dev_spk = {r["speaker_id"] for r in dev}
    overlap = train_spk & dev_spk

    train_sum, train_spk_rows, train_pair_rows = audit_split(
        train, core30, "train"
    )
    dev_sum, dev_spk_rows, dev_pair_rows = audit_split(
        dev, core30, "dev"
    )

    write_csv(args.output_dir/"shot_feasibility_train.csv", train_sum)
    write_csv(args.output_dir/"shot_feasibility_dev_unseen_speakers.csv", dev_sum)
    write_csv(args.output_dir/"speaker_shot_summary.csv", train_spk_rows + dev_spk_rows)
    write_csv(args.output_dir/"speaker_phrase_counts.csv", train_pair_rows + dev_pair_rows)

    summary = {
        "schema": "papr_ssl.h0_02_unseen_speaker_episode_audit.v1",
        "protocol": {
            "head_fit": (
                "TRAIN speakers only; within-speaker distinct recordings are "
                "partitioned into support and query."
            ),
            "unseen_speaker_dev": (
                "DEV speakers only; N support recordings per speaker/phrase "
                "are reserved from DEV, and remaining recordings are query."
            ),
            "final_test": (
                "TEST stays sealed; after model/head/calibration freeze, use "
                "the same within-speaker support/query construction on unseen TEST speakers."
            ),
            "cross_session_claim": False,
            "reason": "No explicit session/day metadata in H0-01.",
        },
        "rows": {"train": len(train), "dev": len(dev)},
        "speakers": {
            "train": len(train_spk),
            "dev": len(dev_spk),
            "train_dev_overlap": len(overlap),
            "speaker_disjoint": len(overlap) == 0,
        },
        "core30_phrase_count": 30,
        "train_shot_feasibility": train_sum,
        "dev_unseen_speaker_shot_feasibility": dev_sum,
        "test_rows_ignored": test_ignored,
        "generic_test_accessed": False,
        "scientific_interpretation": (
            "Speaker-disjoint train/dev is desirable for unseen-speaker few-shot "
            "personalization, provided each evaluation speaker has enough distinct "
            "recordings per phrase to split into support and query."
        ),
    }
    (args.output_dir/"summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    def print_block(title, rows):
        print(title)
        print(
            f"{'shot':>5s} {'ep_spk':>7s} {'pairs':>7s} "
            f"{'support':>8s} {'query':>8s} {'coverage':>10s} "
            f"{'cleanU_spk':>10s} {'status':>13s}"
        )
        for r in rows:
            print(
                f"{r['shot']:5d} "
                f"{r['episode_speakers_min2_intents']:7d} "
                f"{r['eligible_speaker_phrase_pairs']:7d} "
                f"{r['support_rows_required']:8d} "
                f"{r['remaining_known_query_rows']:8d} "
                f"{pct(r['coverage_of_all_core30_rows']):>10s} "
                f"{r['speakers_with_clean_outside_core_unknown']:10d} "
                f"{r['status']:>13s}"
            )

    print("="*108)
    print("H0-02 UNSEEN-SPEAKER PERSONALIZED EPISODE AUDIT")
    print("="*108)
    print(f"train speakers:       {len(train_spk)}")
    print(f"dev speakers:         {len(dev_spk)}")
    print(f"train/dev overlap:    {len(overlap)}")
    print(f"speaker-disjoint:     {'YES' if len(overlap)==0 else 'NO'}")
    print("-"*108)
    print_block("HEAD-FIT EPISODES FROM TRAIN SPEAKERS", train_sum)
    print("-"*108)
    print_block("UNSEEN-SPEAKER DEV EPISODES", dev_sum)
    print("-"*108)
    print("cross-session claim:   NO")
    print(f"test rows ignored:     {test_ignored}")
    print("generic_test accessed: NO")
    print("H0-02 STATUS: COMPLETE")


if __name__ == "__main__":
    main()
