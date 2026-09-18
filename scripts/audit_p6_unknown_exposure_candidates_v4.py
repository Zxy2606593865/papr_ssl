#!/usr/bin/env python
"""Robust P6 Unknown Exposure train-side data audit.

Fixes the earlier failure:
    RuntimeError: expected 30 normalized Core30 phrases, got 0

Root cause:
The Core30/Open-set task-index schema should be parsed with the project's
existing `load_real_task_rows(...)` helper instead of assuming raw manifest
field names such as `label` / `label_text`.

This script:
- uses `load_real_task_rows` for Core30/Open-set task indices;
- uses tolerant raw-manifest parsing only for the full MDSC source manifest;
- never uses generic_test audio/features;
- excludes current open-set DEV phrase identities from train-side exposure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from papr_ssl.training.teacher.real_experiment import load_real_task_rows

SCHEMA = "papr_ssl.p6_unknown_exposure_audit.v4"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def first_value(row: dict, keys: Iterable[str]) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def normalize_phrase(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text))
    s = re.sub(r"<\s*p\s*>", "", s, flags=re.IGNORECASE)
    s = "".join(s.split())
    return s.casefold()


def extract_attr(obj: Any, names: Iterable[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value not in (None, ""):
                return value
    if hasattr(obj, "__dict__"):
        d = vars(obj)
        for name in names:
            if name in d and d[name] not in (None, ""):
                return d[name]
    return None


def phrase_from_task_row(row: Any) -> str:
    """Extract normalized phrase from a project RealTaskRow-like object."""
    value = extract_attr(
        row,
        (
            "task_label",
            "label_text",
            "label",
            "normalized_phrase",
            "transcript",
            "standard_text",
            "text",
            "phrase",
            "intent",
            "target_text",
            "canonical_text",
        ),
    )
    if value is None:
        # Some task-row classes preserve source metadata.
        meta = extract_attr(row, ("source_meta", "meta", "metadata"))
        if isinstance(meta, dict):
            value = first_value(
                meta,
                (
                    "task_label",
                    "label_text",
                    "label",
                    "normalized_phrase",
                    "transcript",
                    "standard_text",
                    "text",
                    "phrase",
                    "intent",
                    "target_text",
                    "canonical_text",
                ),
            )
    return "" if value is None else normalize_phrase(value)


def task_phrase_set(path: Path, splits: tuple[str, ...]) -> tuple[set[str], list[Any]]:
    rows = load_real_task_rows(path, splits=splits)
    phrases = {phrase_from_task_row(r) for r in rows}
    phrases.discard("")
    return phrases, rows


def row_split(row: dict) -> str:
    value = first_value(row, ("split", "subset", "partition"))
    return "" if value is None else str(value).strip().lower()


def row_phrase_raw(row: dict) -> str:
    value = first_value(
        row,
        (
            "task_label",
            "label",
            "label_text",
            "normalized_phrase",
            "transcript",
            "standard_text",
            "text",
            "phrase",
            "intent",
            "target_text",
            "canonical_text",
            "sentence",
        ),
    )
    if value is None:
        meta = row.get("source_meta")
        if isinstance(meta, dict):
            value = first_value(
                meta,
                (
                    "task_label",
                    "label",
                    "label_text",
                    "normalized_phrase",
                    "transcript",
                    "standard_text",
                    "text",
                    "phrase",
                    "intent",
                    "target_text",
                    "canonical_text",
                    "sentence",
                ),
            )
    return "" if value is None else str(value)


def phrase_of_manifest_row(row: dict) -> str:
    return normalize_phrase(row_phrase_raw(row))


def row_speaker(row: dict) -> str:
    value = first_value(
        row,
        ("speaker_id", "speaker", "spk_id", "subject_id", "participant_id"),
    )
    if value is None and isinstance(row.get("source_meta"), dict):
        value = first_value(
            row["source_meta"],
            ("speaker_id", "speaker", "spk_id", "subject_id", "participant_id"),
        )
    return "" if value is None else str(value)


def row_domain(row: dict) -> str:
    value = first_value(row, ("domain", "group", "speaker_group", "condition"))
    if value is None and isinstance(row.get("source_meta"), dict):
        value = first_value(
            row["source_meta"],
            ("domain", "group", "speaker_group", "condition"),
        )
    return "" if value is None else str(value)


def manifest_stats(path: Path) -> dict:
    try:
        rows = read_jsonl(path)
    except Exception:
        return {"path": str(path), "valid": False}

    if not rows:
        return {"path": str(path), "valid": False}

    train = sum(row_split(r) == "train" for r in rows)
    phrase_nonempty = sum(bool(phrase_of_manifest_row(r)) for r in rows)
    return {
        "path": str(path),
        "valid": True,
        "total_rows": len(rows),
        "train_rows": int(train),
        "phrase_nonempty_rows": int(phrase_nonempty),
    }


def discover_manifest(project_root: Path, excluded: set[Path]) -> tuple[Path, list[dict]]:
    candidates = []
    for base_name in ("artifacts", "data", "datasets"):
        base = project_root / base_name
        if not base.exists():
            continue
        for path in base.rglob("*.jsonl"):
            try:
                rp = path.resolve()
                if rp in excluded or path.stat().st_size > 200 * 1024 * 1024:
                    continue
            except OSError:
                continue
            st = manifest_stats(path)
            if not st.get("valid"):
                continue
            if st["train_rows"] < 1000 or st["phrase_nonempty_rows"] < 1000:
                continue
            candidates.append(st)

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover a full MDSC JSONL manifest. "
            "Pass --mdsc-manifest explicitly."
        )

    candidates.sort(
        key=lambda x: (
            x["train_rows"],
            x["total_rows"],
            x["phrase_nonempty_rows"],
        ),
        reverse=True,
    )
    return Path(candidates[0]["path"]), candidates[:10]


def support_summary(counter: Counter) -> dict:
    import numpy as np

    vals = sorted(counter.values())
    if not vals:
        return {
            "phrase_count": 0,
            "utterance_count": 0,
            "min": 0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0,
            "ge_2": 0,
            "ge_5": 0,
            "ge_10": 0,
            "ge_15": 0,
            "ge_20": 0,
        }
    a = np.asarray(vals, dtype=np.float64)
    return {
        "phrase_count": int(len(vals)),
        "utterance_count": int(sum(vals)),
        "min": int(min(vals)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "max": int(max(vals)),
        "ge_2": int(sum(v >= 2 for v in vals)),
        "ge_5": int(sum(v >= 5 for v in vals)),
        "ge_10": int(sum(v >= 10 for v in vals)),
        "ge_15": int(sum(v >= 15 for v in vals)),
        "ge_20": int(sum(v >= 20 for v in vals)),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--mdsc-manifest", type=Path, default=None)
    p.add_argument(
        "--core-index",
        type=Path,
        default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"),
    )
    p.add_argument(
        "--open-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/"
            "mdsc_core30_open_set_eval.index.jsonl"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_unknown_exposure_audit_v4"),
    )
    args = p.parse_args()

    root = args.project_root.resolve()
    core_index = (root / args.core_index).resolve()
    open_index = (root / args.open_index).resolve()
    output_dir = (root / args.output_dir).resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse TASK INDICES through the project's own schema-aware loader.
    # IMPORTANT: load_real_task_rows intentionally permits train/dev only.
    # Core30 vocabulary is fully recoverable from train+dev, so do NOT request test.
    core30, core_rows = task_phrase_set(
        core_index, ("train", "dev")
    )
    print(f"[audit] Core30 normalized phrases recovered: {len(core30)}")
    if len(core30) != 30:
        sample = []
        for r in core_rows[:3]:
            sample.append(vars(r) if hasattr(r, "__dict__") else repr(r))
        raise RuntimeError(
            "Core30 task-index parsing still did not yield 30 phrases. "
            f"got={len(core30)}. First task rows={sample}"
        )

    open_dev_phrases, open_dev_rows = task_phrase_set(open_index, ("dev",))

    print(f"[audit] Core30 normalized phrases: {len(core30)}")
    print(f"[audit] open-set DEV normalized phrases: {len(open_dev_phrases)}")

    if args.mdsc_manifest is None:
        manifest, discovery = discover_manifest(
            root, excluded={core_index, open_index}
        )
        print(f"[audit] auto-discovered full MDSC manifest: {manifest}")
    else:
        manifest = args.mdsc_manifest
        if not manifest.is_absolute():
            manifest = (root / manifest).resolve()
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        discovery = [manifest_stats(manifest)]

    full_rows = read_jsonl(manifest)
    train_rows = [r for r in full_rows if row_split(r) == "train"]
    if len(train_rows) < 1000:
        raise RuntimeError(
            f"selected manifest has only {len(train_rows)} train rows; "
            "probably not the full MDSC manifest"
        )

    candidates = []
    exclusion = Counter()

    for r in train_rows:
        phr = phrase_of_manifest_row(r)
        if not phr:
            exclusion["empty_phrase"] += 1
            continue
        if phr in core30:
            exclusion["core30_phrase"] += 1
            continue
        if phr in open_dev_phrases:
            exclusion["open_dev_phrase"] += 1
            continue
        candidates.append(r)

    phrase_counter = Counter(phrase_of_manifest_row(r) for r in candidates)
    phrase_counter.pop("", None)
    candidate_phrases = set(phrase_counter)

    overlap_core = candidate_phrases & core30
    overlap_dev = candidate_phrases & open_dev_phrases
    if overlap_core or overlap_dev:
        raise RuntimeError(
            f"leakage after filtering: core={len(overlap_core)}, "
            f"open_dev={len(overlap_dev)}"
        )

    speaker_set = {row_speaker(r) for r in candidates if row_speaker(r)}
    domains = Counter(row_domain(r) or "<unknown>" for r in candidates)

    phrase_speakers = defaultdict(set)
    phrase_domains = defaultdict(Counter)
    for r in candidates:
        phr = phrase_of_manifest_row(r)
        spk = row_speaker(r)
        if spk:
            phrase_speakers[phr].add(spk)
        phrase_domains[phr][row_domain(r) or "<unknown>"] += 1

    candidate_path = output_dir / "unknown_exposure_train_candidates.jsonl"
    with candidate_path.open("w", encoding="utf-8") as f:
        for r in candidates:
            out = dict(r)
            out["p6_unknown_exposure_normalized_phrase"] = (
                phrase_of_manifest_row(r)
            )
            out["p6_unknown_exposure_role"] = "train_candidate"
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary_path = output_dir / "unknown_exposure_phrase_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        for phr, count in sorted(
            phrase_counter.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            obj = {
                "normalized_phrase": phr,
                "utterance_count": int(count),
                "speaker_count": int(len(phrase_speakers[phr])),
                "domain_counts": dict(phrase_domains[phr]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    support = support_summary(phrase_counter)
    report = {
        "schema": SCHEMA,
        "source_manifest": {
            "path": str(manifest),
            "sha256": sha256(manifest),
            "auto_discovery_candidates": discovery,
        },
        "task_index_parser": "papr_ssl.training.teacher.real_experiment.load_real_task_rows",
        "counts": {
            "full_manifest_rows": len(full_rows),
            "train_rows": len(train_rows),
            "core30_phrase_count": len(core30),
            "open_dev_phrase_count": len(open_dev_phrases),
            "candidate_rows": len(candidates),
            "candidate_phrase_count": len(phrase_counter),
            "candidate_speaker_count": len(speaker_set),
            "candidate_domain_counts": dict(domains),
            "exclusion_counts": dict(exclusion),
        },
        "candidate_phrase_support": support,
        "development_leakage_checks": {
            "candidate_vs_core30_phrase_overlap": len(overlap_core),
            "candidate_vs_open_dev_phrase_overlap": len(overlap_dev),
        },
        "generic_test_used_for_selection": False,
        "generic_test_audio_or_embedding_accessed": False,
        "generic_test": "sealed_not_accessed",
    }

    report_path = output_dir / "p6_unknown_exposure_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 UNKNOWN EXPOSURE TRAIN-SIDE DATA AUDIT v4")
    print("=" * 108)
    print(f"manifest:                     {manifest}")
    print(f"full manifest rows:           {len(full_rows):,}")
    print(f"train rows:                   {len(train_rows):,}")
    print(f"Core30 phrases:               {len(core30):,}")
    print(f"open-set DEV phrases:         {len(open_dev_phrases):,}")
    print("-" * 108)
    print(f"candidate rows:               {len(candidates):,}")
    print(f"candidate phrases:            {len(phrase_counter):,}")
    print(f"candidate speakers:           {len(speaker_set):,}")
    print(f"candidate domains:            {dict(domains)}")
    print("-" * 108)
    print(
        "phrase support: "
        f"median={support['median']:.1f}, "
        f"p90={support['p90']:.1f}, max={support['max']}"
    )
    print(
        "phrases with support >= "
        f"2:{support['ge_2']}, 5:{support['ge_5']}, "
        f"10:{support['ge_10']}, 15:{support['ge_15']}, "
        f"20:{support['ge_20']}"
    )
    print("-" * 108)
    print(f"candidate/Core30 overlap:     {len(overlap_core)}")
    print(f"candidate/open-DEV overlap:   {len(overlap_dev)}")
    print("generic_test used for selection: NO")
    print("generic_test audio/features:     NO")
    print("P6 UNKNOWN EXPOSURE DATA AUDIT v4 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
