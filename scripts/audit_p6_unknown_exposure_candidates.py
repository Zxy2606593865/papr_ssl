#!/usr/bin/env python
"""Audit train-side MDSC candidates for open-set exposure training.

This is a DATA AUDIT ONLY.
It does NOT train a model and does NOT access generic_test audio/features.

Frozen intent:
- Core30 remains the KNOWN vocabulary.
- candidate UNKNOWN exposure comes only from MDSC train rows outside Core30.
- open-set DEV phrases are excluded from exposure candidates, so the current
  development unknown benchmark is not directly trained on.
- generic_test is not used for selection.

The script can auto-discover the full MDSC manifest when --mdsc-manifest is
omitted. It prefers a JSONL file containing the largest number of MDSC rows
and a substantial train split.

Outputs:
- p6_unknown_exposure_audit.json
- unknown_exposure_train_candidates.jsonl
- unknown_exposure_phrase_summary.jsonl
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

SCHEMA = "papr_ssl.p6_unknown_exposure_audit.v1"


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


def row_split(row: dict) -> str:
    value = first_value(row, ("split", "subset", "partition"))
    return "" if value is None else str(value).strip().lower()


def row_dataset(row: dict) -> str:
    value = first_value(row, ("dataset", "dataset_name", "source_dataset"))
    return "" if value is None else str(value).strip().lower()


def row_phrase_raw(row: dict) -> str:
    value = first_value(
        row,
        (
            "label",
            "label_text",
            "transcript",
            "standard_text",
            "text",
            "phrase",
            "intent",
        ),
    )
    return "" if value is None else str(value)


def row_speaker(row: dict) -> str:
    value = first_value(
        row,
        ("speaker_id", "speaker", "spk_id", "subject_id", "participant_id"),
    )
    return "" if value is None else str(value)


def row_domain(row: dict) -> str:
    value = first_value(row, ("domain", "group", "speaker_group", "condition"))
    return "" if value is None else str(value)


def row_utt_id(row: dict) -> str:
    value = first_value(row, ("utt_id", "utterance_id", "id"))
    if value is not None:
        return str(value)
    rel = first_value(row, ("audio_relpath", "audio_path", "path", "wav_path"))
    return "" if rel is None else str(rel)


def normalize_phrase(text: str) -> str:
    """Frozen task-style phrase normalization.

    - Unicode NFKC
    - remove literal <p> markers case-insensitively
    - remove all whitespace
    - casefold Latin / other cased scripts
    """
    s = unicodedata.normalize("NFKC", str(text))
    s = re.sub(r"<\s*p\s*>", "", s, flags=re.IGNORECASE)
    s = "".join(s.split())
    return s.casefold()


def phrase_of(row: dict) -> str:
    return normalize_phrase(row_phrase_raw(row))


def is_mdsc_row(row: dict) -> bool:
    ds = row_dataset(row)
    if ds:
        return "mdsc" in ds or "aishell" in ds or "dysarth" in ds
    # Full MDSC manifests from earlier P2 work may omit a dataset field.
    # In that case use conservative structural hints.
    rel = str(
        first_value(row, ("audio_relpath", "audio_path", "path", "wav_path"))
        or ""
    ).lower()
    dom = row_domain(row).lower()
    return (
        "mdsc" in rel
        or "dysarthria" in rel
        or "control" in rel
        or "dysarth" in dom
        or dom == "control"
    )


def manifest_stats(path: Path, max_rows: int | None = None) -> dict:
    total = 0
    mdsc = 0
    train = 0
    phrase_nonempty = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    return {"path": str(path), "valid": False}
                if not isinstance(row, dict):
                    continue
                total += 1
                if is_mdsc_row(row):
                    mdsc += 1
                if row_split(row) == "train":
                    train += 1
                if phrase_of(row):
                    phrase_nonempty += 1
                if max_rows and total >= max_rows:
                    break
    except Exception:
        return {"path": str(path), "valid": False}

    return {
        "path": str(path),
        "valid": True,
        "total_rows": total,
        "mdsc_rows": mdsc,
        "train_rows": train,
        "phrase_nonempty_rows": phrase_nonempty,
    }


def discover_manifest(project_root: Path, excluded: set[Path]) -> tuple[Path, list[dict]]:
    roots = [
        project_root / "artifacts",
        project_root / "data",
        project_root / "datasets",
    ]
    candidates: list[dict] = []

    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*.jsonl"):
            rp = path.resolve()
            if rp in excluded:
                continue
            try:
                if path.stat().st_size > 200 * 1024 * 1024:
                    continue
            except OSError:
                continue
            st = manifest_stats(path)
            if not st.get("valid"):
                continue
            # We need a genuine train-side source with phrases.
            if st["train_rows"] < 1000 or st["phrase_nonempty_rows"] < 1000:
                continue
            candidates.append(st)

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover a full MDSC JSONL manifest. "
            "Pass it explicitly with --mdsc-manifest."
        )

    # Prefer the broadest manifest, with train coverage dominating the rank.
    candidates.sort(
        key=lambda x: (
            x["mdsc_rows"],
            x["train_rows"],
            x["total_rows"],
            x["phrase_nonempty_rows"],
        ),
        reverse=True,
    )

    best = Path(candidates[0]["path"])
    return best, candidates[:10]


def class_set_from_index(rows: list[dict]) -> set[str]:
    out = {phrase_of(r) for r in rows if phrase_of(r)}
    return out


def support_summary(counter: Counter) -> dict:
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

    import numpy as np

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


def sha256_text(path: Path) -> str:
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
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"
        ),
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
        default=Path("artifacts/p6_unknown_exposure_audit"),
    )
    args = p.parse_args()

    project_root = args.project_root.resolve()
    core_index = (project_root / args.core_index).resolve()
    open_index = (project_root / args.open_index).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not core_index.exists():
        raise FileNotFoundError(core_index)
    if not open_index.exists():
        raise FileNotFoundError(open_index)

    core_rows = read_jsonl(core_index)
    open_rows = read_jsonl(open_index)

    core30 = class_set_from_index(core_rows)
    if len(core30) != 30:
        raise RuntimeError(
            f"expected 30 normalized Core30 phrases, got {len(core30)}"
        )

    # IMPORTANT: only DEV labels are used from the open-set index.
    # Test rows are deliberately ignored for selection.
    open_dev_rows = [r for r in open_rows if row_split(r) == "dev"]
    open_dev_phrases = class_set_from_index(open_dev_rows)

    if args.mdsc_manifest is None:
        manifest, discovery = discover_manifest(
            project_root,
            excluded={core_index, open_index},
        )
        print(f"[audit] auto-discovered MDSC manifest: {manifest}")
    else:
        manifest = args.mdsc_manifest
        if not manifest.is_absolute():
            manifest = (project_root / manifest).resolve()
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        discovery = [manifest_stats(manifest)]

    full_rows = read_jsonl(manifest)

    train_rows = [r for r in full_rows if row_split(r) == "train"]
    if len(train_rows) < 1000:
        raise RuntimeError(
            f"selected manifest has only {len(train_rows)} train rows; "
            "this is probably not the full MDSC manifest"
        )

    # Candidate exposure = train-side phrase outside known Core30 and outside
    # current open-set DEV phrase identities.
    candidates = []
    exclusion_counts = Counter()
    raw_train_phrase_counter = Counter()

    for r in train_rows:
        phr = phrase_of(r)
        if not phr:
            exclusion_counts["empty_phrase"] += 1
            continue
        raw_train_phrase_counter[phr] += 1
        if phr in core30:
            exclusion_counts["core30_phrase"] += 1
            continue
        if phr in open_dev_phrases:
            exclusion_counts["open_dev_phrase"] += 1
            continue
        candidates.append(r)

    candidate_phrase_counter = Counter(phrase_of(r) for r in candidates)
    candidate_speakers = {row_speaker(r) for r in candidates if row_speaker(r)}
    candidate_domains = Counter(row_domain(r) or "<unknown>" for r in candidates)

    # Leakage checks we are allowed to make during development.
    candidate_phrases = set(candidate_phrase_counter)
    overlap_core = sorted(candidate_phrases & core30)
    overlap_open_dev = sorted(candidate_phrases & open_dev_phrases)
    if overlap_core or overlap_open_dev:
        raise RuntimeError(
            "candidate pool leakage detected despite filtering: "
            f"core={len(overlap_core)}, open_dev={len(overlap_open_dev)}"
        )

    # Preserve rows with an explicit normalized phrase field added.
    candidate_path = output_dir / "unknown_exposure_train_candidates.jsonl"
    with candidate_path.open("w", encoding="utf-8") as f:
        for r in candidates:
            out = dict(r)
            out["p6_unknown_exposure_normalized_phrase"] = phrase_of(r)
            out["p6_unknown_exposure_role"] = "train_candidate"
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    phrase_meta: dict[str, dict] = {}
    phrase_speakers: dict[str, set[str]] = defaultdict(set)
    phrase_domains: dict[str, Counter] = defaultdict(Counter)

    for r in candidates:
        phr = phrase_of(r)
        spk = row_speaker(r)
        if spk:
            phrase_speakers[phr].add(spk)
        phrase_domains[phr][row_domain(r) or "<unknown>"] += 1

    phrase_summary_path = output_dir / "unknown_exposure_phrase_summary.jsonl"
    with phrase_summary_path.open("w", encoding="utf-8") as f:
        for phr, count in sorted(
            candidate_phrase_counter.items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            row = {
                "normalized_phrase": phr,
                "utterance_count": int(count),
                "speaker_count": int(len(phrase_speakers[phr])),
                "domain_counts": dict(phrase_domains[phr]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema": SCHEMA,
        "purpose": (
            "audit train-side non-Core30 speech available for future "
            "open-set exposure training"
        ),
        "source_manifest": {
            "path": str(manifest),
            "sha256": sha256_text(manifest),
            "auto_discovery_candidates": discovery,
        },
        "frozen_policy": {
            "known_vocabulary": "Core30 exact normalized phrase identities",
            "candidate_source_split": "train only",
            "exclude_core30_phrase_identity": True,
            "exclude_open_set_dev_phrase_identity": True,
            "generic_test_used_for_selection": False,
            "generic_test_audio_or_embedding_accessed": False,
            "test_phrase_overlap_check_deferred_until_final_frozen_audit": True,
        },
        "counts": {
            "full_manifest_rows": int(len(full_rows)),
            "train_rows": int(len(train_rows)),
            "core30_phrase_count": int(len(core30)),
            "open_dev_phrase_count": int(len(open_dev_phrases)),
            "candidate_rows": int(len(candidates)),
            "candidate_phrase_count": int(len(candidate_phrase_counter)),
            "candidate_speaker_count": int(len(candidate_speakers)),
            "candidate_domain_counts": dict(candidate_domains),
            "exclusion_counts": dict(exclusion_counts),
        },
        "candidate_phrase_support": support_summary(candidate_phrase_counter),
        "development_leakage_checks": {
            "candidate_vs_core30_phrase_overlap": int(len(overlap_core)),
            "candidate_vs_open_dev_phrase_overlap": int(len(overlap_open_dev)),
        },
        "generic_test": "sealed_not_accessed",
        "next_decision_contract": (
            "No model training should begin until this audit is reviewed. "
            "If candidate supply is adequate, freeze a deterministic "
            "train-side exposure selection policy before training Teacher-O1."
        ),
    }

    report_path = output_dir / "p6_unknown_exposure_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 UNKNOWN EXPOSURE TRAIN-SIDE DATA AUDIT")
    print("=" * 108)
    print(f"manifest:                     {manifest}")
    print(f"full manifest rows:           {len(full_rows):,}")
    print(f"train rows:                   {len(train_rows):,}")
    print(f"Core30 phrases:               {len(core30):,}")
    print(f"open-set DEV phrases:         {len(open_dev_phrases):,}")
    print("-" * 108)
    print(f"candidate rows:               {len(candidates):,}")
    print(f"candidate phrases:            {len(candidate_phrase_counter):,}")
    print(f"candidate speakers:           {len(candidate_speakers):,}")
    print(f"candidate domains:            {dict(candidate_domains)}")
    print("-" * 108)
    ss = report["candidate_phrase_support"]
    print(
        "phrase support: "
        f"median={ss['median']:.1f}, p90={ss['p90']:.1f}, "
        f"max={ss['max']}"
    )
    print(
        "phrases with support >= "
        f"2:{ss['ge_2']}, 5:{ss['ge_5']}, 10:{ss['ge_10']}, "
        f"15:{ss['ge_15']}, 20:{ss['ge_20']}"
    )
    print("-" * 108)
    print(
        "candidate/Core30 phrase overlap:  "
        f"{len(overlap_core)}"
    )
    print(
        "candidate/open-DEV overlap:       "
        f"{len(overlap_open_dev)}"
    )
    print("generic_test used for selection: NO")
    print("generic_test audio/features:         NO")
    print("P6 UNKNOWN EXPOSURE DATA AUDIT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
