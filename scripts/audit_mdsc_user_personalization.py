#!/usr/bin/env python
"""H0-01 — MDSC user-level personalized episode feasibility audit.

This script does NOT train a model and does NOT load audio, embeddings, or
generic_test examples. It audits metadata only and restricts all statistics to
train/dev.

Questions answered
------------------
1. How many speakers overlap between train and dev?
2. For each speaker x Core30 phrase, how many train support and dev query rows exist?
3. Which speakers can support 1/2/5/10/15-shot personalized episodes?
4. How many known DEV queries are coverable at each shot level?
5. For the same speakers, how many DEV utterances/phrases could serve as
   user-relative unknown queries?
6. Does MDSC expose explicit session/day metadata that can support a genuine
   cross-session protocol?

Important
---------
- Core30 support comes from TRAIN only.
- Known query comes from DEV only.
- TEST is ignored.
- "Unknown" is always defined relative to the current user's enrolled set.
- The report separately counts clean outside-Core30 DEV unknowns.
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
MIN_INTENTS_FOR_EPISODE = 2

SESSION_KEY_RE = re.compile(
    r"(session|session_id|recording_session|visit|day|date|recording_date|"
    r"record_date|take|round|batch)",
    re.IGNORECASE,
)


def normalize_phrase(text: str) -> str:
    """Mirror the frozen task-level phrase normalization conservatively."""
    if text is None:
        return ""
    x = str(text).replace("<p>", "")
    x = re.sub(r"\s+", "", x)
    return x.casefold()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Bad JSONL at {path}:{line_no}: {e}") from e


def row_split(row: dict) -> str:
    return str(row.get("split", "")).strip().lower()


def row_utt_id(row: dict) -> str:
    for key in ("utt_id", "id", "audio_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    raise KeyError(f"Could not find utt_id-like field in row keys={sorted(row)}")


def row_speaker(row: dict) -> str:
    for key in ("speaker_id", "speaker", "user_id", "subject_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    raise KeyError(
        f"Could not find speaker_id-like field in MDSC manifest row keys={sorted(row)}"
    )


def row_domain(row: dict) -> str:
    for key in ("domain", "group", "condition"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "UNKNOWN"


def row_phrase(row: dict) -> str:
    for key in ("transcript", "standard_text", "text", "label"):
        value = row.get(key)
        if value not in (None, ""):
            return normalize_phrase(value)
    return ""


def task_phrase(row: dict) -> str:
    for key in ("task_label", "label_text", "label", "transcript"):
        value = row.get(key)
        if value not in (None, ""):
            return normalize_phrase(value)
    return ""


def nested_items(d: dict, prefix=""):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from nested_items(v, key)
        else:
            yield key, v


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fmt_pct(x):
    return f"{100.0 * x:.2f}%"


def md_table(headers, rows):
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mdsc-manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--core30-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h0_01_mdsc_user_audit"),
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output dir: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load MDSC metadata, TRAIN/DEV only. TEST rows are ignored.
    # ------------------------------------------------------------------
    mdsc_by_utt = {}
    mdsc_train = []
    mdsc_dev = []
    ignored_test_rows = 0

    session_key_counts = Counter()
    session_value_examples = defaultdict(list)

    for row in read_jsonl(args.mdsc_manifest):
        split = row_split(row)
        if split == "test":
            ignored_test_rows += 1
            continue
        if split not in ("train", "dev"):
            continue

        utt_id = row_utt_id(row)
        speaker = row_speaker(row)
        phrase = row_phrase(row)

        clean = {
            "utt_id": utt_id,
            "speaker_id": speaker,
            "split": split,
            "domain": row_domain(row),
            "phrase": phrase,
        }
        mdsc_by_utt[utt_id] = clean

        if split == "train":
            mdsc_train.append(clean)
        else:
            mdsc_dev.append(clean)

        # Explicit metadata only. We do not infer sessions from paths.
        for key, value in nested_items(row):
            if SESSION_KEY_RE.search(key) and value not in (None, "", [], {}):
                session_key_counts[key] += 1
                if len(session_value_examples[key]) < 5:
                    session_value_examples[key].append(str(value))

    # ------------------------------------------------------------------
    # 2) Recover the frozen Core30 phrase inventory from TRAIN/DEV task index.
    # ------------------------------------------------------------------
    core30_phrases = set()
    core30_utt_ids = set()

    for row in read_jsonl(args.core30_index):
        split = row_split(row)
        if split == "test":
            continue
        if split not in ("train", "dev"):
            continue

        phrase = task_phrase(row)
        if phrase:
            core30_phrases.add(phrase)
        try:
            core30_utt_ids.add(row_utt_id(row))
        except KeyError:
            pass

    if len(core30_phrases) != 30:
        raise RuntimeError(
            f"Expected exactly 30 normalized Core30 phrases, got "
            f"{len(core30_phrases)}: {sorted(core30_phrases)}"
        )

    # ------------------------------------------------------------------
    # 3) Speaker-level and speaker x phrase statistics.
    # ------------------------------------------------------------------
    train_speakers = {r["speaker_id"] for r in mdsc_train}
    dev_speakers = {r["speaker_id"] for r in mdsc_dev}
    overlap_speakers = sorted(train_speakers & dev_speakers)

    pair = defaultdict(lambda: {"train": 0, "dev": 0, "domain": "UNKNOWN"})
    speaker_domain = {}
    speaker_train_rows = Counter()
    speaker_dev_rows = Counter()
    speaker_train_phrases = defaultdict(set)
    speaker_dev_phrases = defaultdict(set)
    speaker_dev_outside_core_rows = Counter()
    speaker_dev_outside_core_phrases = defaultdict(set)

    for r in mdsc_train:
        s, ph = r["speaker_id"], r["phrase"]
        speaker_domain.setdefault(s, r["domain"])
        speaker_train_rows[s] += 1
        if ph:
            speaker_train_phrases[s].add(ph)
            if ph in core30_phrases:
                pair[(s, ph)]["train"] += 1
                pair[(s, ph)]["domain"] = r["domain"]

    for r in mdsc_dev:
        s, ph = r["speaker_id"], r["phrase"]
        speaker_domain.setdefault(s, r["domain"])
        speaker_dev_rows[s] += 1
        if ph:
            speaker_dev_phrases[s].add(ph)
            if ph in core30_phrases:
                pair[(s, ph)]["dev"] += 1
                pair[(s, ph)]["domain"] = r["domain"]
            else:
                speaker_dev_outside_core_rows[s] += 1
                speaker_dev_outside_core_phrases[s].add(ph)

    # Detailed speaker x Core30 phrase CSV
    pair_rows = []
    for speaker in sorted(train_speakers | dev_speakers):
        for phrase in sorted(core30_phrases):
            stats = pair[(speaker, phrase)]
            pair_rows.append(
                {
                    "speaker_id": speaker,
                    "domain": speaker_domain.get(speaker, "UNKNOWN"),
                    "phrase": phrase,
                    "train_support_count": stats["train"],
                    "dev_query_count": stats["dev"],
                    **{
                        f"eligible_{shot}shot": int(
                            stats["train"] >= shot and stats["dev"] >= 1
                        )
                        for shot in SHOT_LEVELS
                    },
                }
            )

    write_csv(
        args.output_dir / "speaker_phrase_core30.csv",
        pair_rows,
        [
            "speaker_id",
            "domain",
            "phrase",
            "train_support_count",
            "dev_query_count",
            *[f"eligible_{s}shot" for s in SHOT_LEVELS],
        ],
    )

    # Per-shot feasibility
    shot_rows = []
    per_shot_speaker = {}
    per_shot_registered = {}

    total_core30_dev_rows_overlap = sum(
        1
        for r in mdsc_dev
        if r["speaker_id"] in overlap_speakers and r["phrase"] in core30_phrases
    )

    for shot in SHOT_LEVELS:
        speaker_registered = defaultdict(set)
        speaker_known_query_rows = Counter()

        for (speaker, phrase), stats in pair.items():
            if (
                speaker in overlap_speakers
                and stats["train"] >= shot
                and stats["dev"] >= 1
            ):
                speaker_registered[speaker].add(phrase)
                speaker_known_query_rows[speaker] += stats["dev"]

        episode_speakers = sorted(
            s
            for s, phrases in speaker_registered.items()
            if len(phrases) >= MIN_INTENTS_FOR_EPISODE
        )

        # Unknown relative to this user's registered set.
        relative_unknown_rows = Counter()
        relative_unknown_phrases = defaultdict(set)
        outside_core_unknown_rows = Counter()
        outside_core_unknown_phrases = defaultdict(set)

        for r in mdsc_dev:
            s, ph = r["speaker_id"], r["phrase"]
            if s not in episode_speakers or not ph:
                continue

            if ph not in speaker_registered[s]:
                relative_unknown_rows[s] += 1
                relative_unknown_phrases[s].add(ph)

            if ph not in core30_phrases:
                outside_core_unknown_rows[s] += 1
                outside_core_unknown_phrases[s].add(ph)

        speakers_with_clean_unknown = [
            s for s in episode_speakers
            if outside_core_unknown_rows[s] > 0
        ]
        speakers_with_relative_unknown = [
            s for s in episode_speakers
            if relative_unknown_rows[s] > 0
        ]

        eligible_pairs = sum(len(v) for v in speaker_registered.values())
        covered_known_queries = sum(
            speaker_known_query_rows[s] for s in episode_speakers
        )
        coverage = (
            covered_known_queries / total_core30_dev_rows_overlap
            if total_core30_dev_rows_overlap
            else 0.0
        )

        intent_counts = [len(speaker_registered[s]) for s in episode_speakers]
        known_q_counts = [speaker_known_query_rows[s] for s in episode_speakers]

        row = {
            "shot": shot,
            "eligible_speaker_phrase_pairs": eligible_pairs,
            "episode_speakers_min2_intents": len(episode_speakers),
            "speakers_with_clean_outside_core_unknown": len(speakers_with_clean_unknown),
            "speakers_with_any_relative_unknown": len(speakers_with_relative_unknown),
            "known_dev_queries_covered": covered_known_queries,
            "core30_dev_query_coverage": coverage,
            "median_registered_intents_per_episode_speaker": (
                float(median(intent_counts)) if intent_counts else 0.0
            ),
            "max_registered_intents_per_episode_speaker": (
                max(intent_counts) if intent_counts else 0
            ),
            "median_known_dev_queries_per_episode_speaker": (
                float(median(known_q_counts)) if known_q_counts else 0.0
            ),
            "status": "FEASIBLE" if episode_speakers else "NOT_FEASIBLE",
        }
        shot_rows.append(row)
        per_shot_speaker[shot] = episode_speakers
        per_shot_registered[shot] = {
            s: sorted(speaker_registered[s]) for s in episode_speakers
        }

    write_csv(
        args.output_dir / "shot_feasibility.csv",
        shot_rows,
        [
            "shot",
            "eligible_speaker_phrase_pairs",
            "episode_speakers_min2_intents",
            "speakers_with_clean_outside_core_unknown",
            "speakers_with_any_relative_unknown",
            "known_dev_queries_covered",
            "core30_dev_query_coverage",
            "median_registered_intents_per_episode_speaker",
            "max_registered_intents_per_episode_speaker",
            "median_known_dev_queries_per_episode_speaker",
            "status",
        ],
    )

    # Speaker summary
    speaker_rows = []
    for s in sorted(train_speakers | dev_speakers):
        row = {
            "speaker_id": s,
            "domain": speaker_domain.get(s, "UNKNOWN"),
            "in_train": int(s in train_speakers),
            "in_dev": int(s in dev_speakers),
            "train_rows": speaker_train_rows[s],
            "dev_rows": speaker_dev_rows[s],
            "train_distinct_phrases": len(speaker_train_phrases[s]),
            "dev_distinct_phrases": len(speaker_dev_phrases[s]),
            "dev_outside_core30_rows": speaker_dev_outside_core_rows[s],
            "dev_outside_core30_distinct_phrases": len(
                speaker_dev_outside_core_phrases[s]
            ),
        }
        for shot in SHOT_LEVELS:
            registered = per_shot_registered[shot].get(s, [])
            row[f"registered_intents_{shot}shot"] = len(registered)
            row[f"episode_eligible_{shot}shot"] = int(
                s in per_shot_speaker[shot]
            )
        speaker_rows.append(row)

    write_csv(
        args.output_dir / "speaker_summary.csv",
        speaker_rows,
        list(speaker_rows[0].keys()) if speaker_rows else ["speaker_id"],
    )

    # Session metadata audit
    session_audit = {
        "explicit_session_metadata_available": bool(session_key_counts),
        "candidate_keys": [
            {
                "key": key,
                "nonempty_train_dev_rows": count,
                "examples": session_value_examples[key],
            }
            for key, count in session_key_counts.most_common()
        ],
        "interpretation": (
            "Explicit candidate session/day metadata exists; inspect semantics "
            "before using it for cross-session splitting."
            if session_key_counts
            else
            "No explicit session/day field was found in train/dev metadata. "
            "Do NOT infer true sessions from file paths without dataset documentation."
        ),
    }
    (args.output_dir / "session_metadata_audit.json").write_text(
        json.dumps(session_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 4) Summary and scientific verdict.
    # ------------------------------------------------------------------
    domain_overlap = Counter(
        speaker_domain.get(s, "UNKNOWN") for s in overlap_speakers
    )

    summary = {
        "schema": "papr_ssl.h0_01_mdsc_user_personalization_audit.v1",
        "scope": "metadata-only, train/dev only",
        "mdsc_manifest": str(args.mdsc_manifest),
        "core30_index": str(args.core30_index),
        "test_rows_ignored": ignored_test_rows,
        "generic_test_audio_accessed": False,
        "generic_test_embeddings_accessed": False,
        "core30_phrase_count": len(core30_phrases),
        "rows": {
            "train": len(mdsc_train),
            "dev": len(mdsc_dev),
        },
        "speakers": {
            "train": len(train_speakers),
            "dev": len(dev_speakers),
            "train_dev_overlap": len(overlap_speakers),
            "overlap_ids": overlap_speakers,
            "overlap_domain_counts": dict(domain_overlap),
        },
        "total_core30_dev_rows_on_overlap_speakers": total_core30_dev_rows_overlap,
        "shot_feasibility": shot_rows,
        "session_metadata": session_audit,
        "protocol_definition": {
            "support": "same speaker + same Core30 phrase, TRAIN only",
            "known_query": "same speaker + registered Core30 phrase, DEV only",
            "clean_unknown_query": (
                "same speaker DEV phrase outside Core30 inventory"
            ),
            "relative_unknown_query": (
                "same speaker DEV phrase not present in that user's current registered set"
            ),
            "min_registered_intents_per_episode": MIN_INTENTS_FOR_EPISODE,
        },
        "scientific_limits": [
            "This audit does not prove child-domain generalization.",
            "This audit does not prove cross-session personalization unless explicit session metadata exists.",
            "FEASIBLE means that metadata counts permit constructing at least one user-level episode under the stated rule; it is not a performance result.",
            "Existing 91.45% DEV result must not be re-labelled as user-level personalized performance.",
        ],
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Markdown report
    shot_table = []
    for r in shot_rows:
        shot_table.append(
            [
                r["shot"],
                r["episode_speakers_min2_intents"],
                r["eligible_speaker_phrase_pairs"],
                r["known_dev_queries_covered"],
                fmt_pct(r["core30_dev_query_coverage"]),
                r["speakers_with_clean_outside_core_unknown"],
                r["median_registered_intents_per_episode_speaker"],
                r["status"],
            ]
        )

    report = f"""# H0-01 MDSC 用户级 Personalized Episode 可行性审计

## 1. 审计边界

本审计仅读取 MDSC **train/dev 元数据**，不加载音频、不加载 embedding、不训练模型。
test 行被忽略；`generic_test` 音频与 embedding 均未访问。

用户级 episode 的冻结定义：

- Support：同一 `speaker_id`、同一 Core30 短语、**TRAIN**。
- Known Query：同一 `speaker_id`、已注册 Core30 短语、**DEV**。
- Clean Unknown Query：同一 `speaker_id` 的 DEV 短语，但该短语位于 Core30 词表之外。
- Relative Unknown Query：同一用户 DEV 中不属于该用户当前注册集合的短语。
- 一个 episode speaker 至少要求 {MIN_INTENTS_FOR_EPISODE} 个可注册 intent。

## 2. 基础统计

- Train rows：**{len(mdsc_train)}**
- Dev rows：**{len(mdsc_dev)}**
- Train speakers：**{len(train_speakers)}**
- Dev speakers：**{len(dev_speakers)}**
- Train/Dev overlap speakers：**{len(overlap_speakers)}**
- Core30 phrase count：**{len(core30_phrases)}**
- Test rows ignored：**{ignored_test_rows}**

Train/Dev overlap domain：

```json
{json.dumps(dict(domain_overlap), ensure_ascii=False, indent=2)}
```

## 3. Shot 可行性

{md_table(
    [
        "shot",
        "episode speakers",
        "eligible speaker×phrase",
        "known DEV queries",
        "query coverage",
        "speakers with clean unknown",
        "median registered intents",
        "status",
    ],
    shot_table,
)}

注意：`FEASIBLE` 只代表按当前元数据可以构造至少一个 user-level episode，
**不是模型性能通过，也不是儿童域有效性证明**。

## 4. Session / Day 元数据

- Explicit session/day metadata available：**{session_audit["explicit_session_metadata_available"]}**
- 结论：{session_audit["interpretation"]}

候选字段：

```json
{json.dumps(session_audit["candidate_keys"], ensure_ascii=False, indent=2)}
```

如果没有明确 session/day 字段，后续只能先做“同 speaker 的 train-support / dev-query”
personalized baseline，不能把它表述成 cross-session baseline。

## 5. H0-01 结论

下一步是否进入 H1，按下面规则判断：

1. 如果 `train_dev_overlap = 0`：MDSC 无法按现有 split 构造同用户 train→dev baseline，H1 在 MDSC 上 BLOCKED。
2. 如果某个 shot 的 `episode speakers > 0`：该 shot 可以进行 user-level personalized baseline。
3. 如果 `speakers with clean unknown > 0`：该 shot 可以进一步构造同用户的 open-set episode。
4. 如果无明确 session/day 字段：不得声称 cross-session personalization。
5. 真实儿童数据仍然是最终 personalized / cross-session 结论的必要证据。

## 6. 输出文件

- `summary.json`
- `speaker_summary.csv`
- `speaker_phrase_core30.csv`
- `shot_feasibility.csv`
- `session_metadata_audit.json`

这些文件用于 H1 构建真实 user-level support/query episode，不修改任何现有 Teacher、Neck、DTW 或正式 artifact。
"""

    (args.output_dir / "H0_01_MDSC_USER_EPISODE_AUDIT.md").write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 108)
    print("H0-01 MDSC USER-LEVEL PERSONALIZATION FEASIBILITY AUDIT")
    print("=" * 108)
    print(f"train rows:             {len(mdsc_train)}")
    print(f"dev rows:               {len(mdsc_dev)}")
    print(f"train speakers:         {len(train_speakers)}")
    print(f"dev speakers:           {len(dev_speakers)}")
    print(f"train/dev overlap:      {len(overlap_speakers)}")
    print(f"Core30 phrases:         {len(core30_phrases)}")
    print("-" * 108)
    print(
        f"{'shot':>6s} {'episode_spk':>12s} {'pairs':>8s} "
        f"{'known_q':>10s} {'coverage':>10s} {'clean_unknown_spk':>18s} {'status':>14s}"
    )
    for r in shot_rows:
        print(
            f"{r['shot']:6d} "
            f"{r['episode_speakers_min2_intents']:12d} "
            f"{r['eligible_speaker_phrase_pairs']:8d} "
            f"{r['known_dev_queries_covered']:10d} "
            f"{fmt_pct(r['core30_dev_query_coverage']):>10s} "
            f"{r['speakers_with_clean_outside_core_unknown']:18d} "
            f"{r['status']:>14s}"
        )
    print("-" * 108)
    print(
        "explicit session/day metadata: "
        f"{'YES' if session_audit['explicit_session_metadata_available'] else 'NO'}"
    )
    print(f"test rows ignored:       {ignored_test_rows}")
    print("generic_test audio:      NOT ACCESSED")
    print("generic_test embeddings: NOT ACCESSED")
    print("H0-01 STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
