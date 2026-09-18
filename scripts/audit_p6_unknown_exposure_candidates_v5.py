#!/usr/bin/env python
"""P6 Unknown Exposure audit v5.

Fixes two issues revealed by v4:
1) v4 auto-selected GSC v2 instead of MDSC.
2) the open-set task index has a generic task_label, so its unique task_label
   count is not the number of real unknown phrases. v5 maps open-set DEV utt_id
   values back to the full MDSC manifest and recovers the original phrase text.

This is a DATA AUDIT ONLY. No training. No generic_test audio/features.
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

SCHEMA = "papr_ssl.p6_unknown_exposure_audit.v5"


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


def row_split(row: dict) -> str:
    v = first_value(row, ("split", "subset", "partition"))
    return "" if v is None else str(v).strip().lower()


def row_dataset(row: dict) -> str:
    v = first_value(row, ("dataset", "dataset_name", "source_dataset"))
    if v is None and isinstance(row.get("source_meta"), dict):
        v = first_value(
            row["source_meta"],
            ("dataset", "dataset_name", "source_dataset"),
        )
    return "" if v is None else str(v).strip().lower()


def row_utt_id(row: dict) -> str:
    v = first_value(row, ("utt_id", "utterance_id", "id"))
    if v is None and isinstance(row.get("source_meta"), dict):
        v = first_value(row["source_meta"], ("utt_id", "utterance_id", "id"))
    return "" if v is None else str(v)


def raw_phrase(row: dict) -> str:
    # Prefer actual transcript/text-like fields over generic task labels.
    v = first_value(
        row,
        (
            "transcript",
            "standard_text",
            "text",
            "phrase",
            "label",
            "label_text",
            "canonical_text",
            "sentence",
            "task_label",
        ),
    )
    if v is None and isinstance(row.get("source_meta"), dict):
        v = first_value(
            row["source_meta"],
            (
                "transcript",
                "standard_text",
                "text",
                "phrase",
                "label",
                "label_text",
                "canonical_text",
                "sentence",
                "task_label",
            ),
        )
    return "" if v is None else str(v)


def phrase_of(row: dict) -> str:
    return normalize_phrase(raw_phrase(row))


def row_speaker(row: dict) -> str:
    v = first_value(
        row,
        ("speaker_id", "speaker", "spk_id", "subject_id", "participant_id"),
    )
    if v is None and isinstance(row.get("source_meta"), dict):
        v = first_value(
            row["source_meta"],
            ("speaker_id", "speaker", "spk_id", "subject_id", "participant_id"),
        )
    return "" if v is None else str(v)


def row_domain(row: dict) -> str:
    v = first_value(row, ("domain", "group", "speaker_group", "condition"))
    if v is None and isinstance(row.get("source_meta"), dict):
        v = first_value(
            row["source_meta"],
            ("domain", "group", "speaker_group", "condition"),
        )
    return "" if v is None else str(v)


def task_attr(row, name: str):
    if hasattr(row, name):
        return getattr(row, name)
    if hasattr(row, "__dict__"):
        return vars(row).get(name)
    return None


def core30_phrase_set(core_index: Path) -> set[str]:
    rows = load_real_task_rows(core_index, splits=("train", "dev"))
    phrases = set()
    for r in rows:
        v = task_attr(r, "task_label")
        if v not in (None, ""):
            phrases.add(normalize_phrase(v))
    phrases.discard("")
    return phrases


def open_dev_utt_ids(open_index: Path) -> set[str]:
    rows = load_real_task_rows(open_index, splits=("dev",))
    ids = {str(task_attr(r, "utt_id")) for r in rows if task_attr(r, "utt_id")}
    return ids


def manifest_stats(path: Path) -> dict:
    try:
        rows = read_jsonl(path)
    except Exception:
        return {"path": str(path), "valid": False}

    if not rows:
        return {"path": str(path), "valid": False}

    dataset_counts = Counter(row_dataset(r) or "<missing>" for r in rows)
    mdsc_rows = sum(
        n for ds, n in dataset_counts.items()
        if ("mdsc" in ds or "aishell" in ds or "dysarth" in ds)
    )
    train_rows = sum(row_split(r) == "train" for r in rows)
    phrase_rows = sum(bool(phrase_of(r)) for r in rows)

    return {
        "path": str(path),
        "valid": True,
        "total_rows": len(rows),
        "train_rows": int(train_rows),
        "phrase_rows": int(phrase_rows),
        "mdsc_rows": int(mdsc_rows),
        "dataset_counts": dict(dataset_counts),
    }


def discover_mdsc_manifest(
    project_root: Path,
    excluded: set[Path],
) -> tuple[Path, list[dict]]:
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

            # Strictly require a substantial MDSC population.
            if st["mdsc_rows"] < 5000:
                continue
            if st["train_rows"] < 3000:
                continue
            if st["phrase_rows"] < 3000:
                continue
            candidates.append(st)

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover an MDSC manifest with >=5000 MDSC rows. "
            "Run PowerShell `Get-ChildItem artifacts -Recurse -Filter *.jsonl | "
            "Where-Object {$_.Name -match 'mdsc'} | Select FullName` and pass "
            "the correct file with --mdsc-manifest."
        )

    candidates.sort(
        key=lambda x: (x["mdsc_rows"], x["train_rows"], x["total_rows"]),
        reverse=True,
    )

    # Refuse an ambiguous tie rather than silently selecting the wrong dataset.
    if len(candidates) > 1:
        a, b = candidates[0], candidates[1]
        if (
            a["mdsc_rows"] == b["mdsc_rows"]
            and a["train_rows"] == b["train_rows"]
            and a["total_rows"] == b["total_rows"]
        ):
            raise RuntimeError(
                "Multiple equally plausible MDSC manifests were found. "
                "Pass --mdsc-manifest explicitly.\n"
                + "\n".join(x["path"] for x in candidates[:5])
            )

    return Path(candidates[0]["path"]), candidates[:10]


def support_summary(counter: Counter) -> dict:
    import numpy as np
    vals = list(counter.values())
    if not vals:
        return {
            "phrase_count": 0,
            "utterance_count": 0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0,
            "ge_2": 0,
            "ge_5": 0,
            "ge_10": 0,
            "ge_15": 0,
        }
    a = np.asarray(vals, dtype=np.float64)
    return {
        "phrase_count": len(vals),
        "utterance_count": int(sum(vals)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "max": int(np.max(a)),
        "ge_2": int(sum(v >= 2 for v in vals)),
        "ge_5": int(sum(v >= 5 for v in vals)),
        "ge_10": int(sum(v >= 10 for v in vals)),
        "ge_15": int(sum(v >= 15 for v in vals)),
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
        default=Path("artifacts/p6_unknown_exposure_audit_v5"),
    )
    args = p.parse_args()

    root = args.project_root.resolve()
    core_index = (root / args.core_index).resolve()
    open_index = (root / args.open_index).resolve()
    outdir = (root / args.output_dir).resolve()

    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"refusing overwrite: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    core30 = core30_phrase_set(core_index)
    if len(core30) != 30:
        raise RuntimeError(f"expected Core30 phrase count 30, got {len(core30)}")

    open_ids = open_dev_utt_ids(open_index)
    if len(open_ids) != 1178:
        raise RuntimeError(
            f"expected 1178 open-set DEV utterances, got {len(open_ids)}"
        )

    if args.mdsc_manifest is None:
        manifest, discovery = discover_mdsc_manifest(
            root, excluded={core_index, open_index}
        )
        print(f"[audit] auto-discovered MDSC manifest: {manifest}")
    else:
        manifest = args.mdsc_manifest
        if not manifest.is_absolute():
            manifest = (root / manifest).resolve()
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        discovery = [manifest_stats(manifest)]

    full_rows = read_jsonl(manifest)

    # If a broad/mixed manifest is supplied, keep MDSC rows only.
    mdsc_rows = [
        r for r in full_rows
        if (
            "mdsc" in row_dataset(r)
            or "aishell" in row_dataset(r)
            or "dysarth" in row_dataset(r)
        )
    ]
    if len(mdsc_rows) < 5000:
        raise RuntimeError(
            f"selected manifest is not MDSC-like: only {len(mdsc_rows)} "
            "rows identified as MDSC"
        )

    train_rows = [r for r in mdsc_rows if row_split(r) == "train"]

    by_utt = {row_utt_id(r): r for r in mdsc_rows if row_utt_id(r)}
    missing_open_ids = sorted(open_ids - set(by_utt))
    if missing_open_ids:
        sample = missing_open_ids[:5]
        raise RuntimeError(
            f"{len(missing_open_ids)}/1178 open DEV utt_id values were not "
            f"found in the MDSC manifest. Sample={sample}"
        )

    open_dev_phrases = {phrase_of(by_utt[u]) for u in open_ids}
    open_dev_phrases.discard("")
    if len(open_dev_phrases) < 10:
        raise RuntimeError(
            "Recovered fewer than 10 real open-set DEV phrases. "
            "The raw MDSC phrase field is probably being parsed incorrectly. "
            f"Recovered={len(open_dev_phrases)}"
        )

    candidates = []
    exclusions = Counter()
    for r in train_rows:
        phr = phrase_of(r)
        if not phr:
            exclusions["empty_phrase"] += 1
            continue
        if phr in core30:
            exclusions["core30_phrase"] += 1
            continue
        if phr in open_dev_phrases:
            exclusions["open_dev_phrase"] += 1
            continue
        candidates.append(r)

    phrase_counter = Counter(phrase_of(r) for r in candidates)
    phrase_counter.pop("", None)
    candidate_phrases = set(phrase_counter)

    overlap_core = candidate_phrases & core30
    overlap_dev = candidate_phrases & open_dev_phrases
    if overlap_core or overlap_dev:
        raise RuntimeError("phrase leakage after filtering")

    speakers = {row_speaker(r) for r in candidates if row_speaker(r)}
    domains = Counter(row_domain(r) or "<unknown>" for r in candidates)

    phrase_speakers = defaultdict(set)
    phrase_domains = defaultdict(Counter)
    for r in candidates:
        phr = phrase_of(r)
        spk = row_speaker(r)
        if spk:
            phrase_speakers[phr].add(spk)
        phrase_domains[phr][row_domain(r) or "<unknown>"] += 1

    cand_path = outdir / "unknown_exposure_train_candidates.jsonl"
    with cand_path.open("w", encoding="utf-8") as f:
        for r in candidates:
            obj = dict(r)
            obj["p6_unknown_exposure_normalized_phrase"] = phrase_of(r)
            obj["p6_unknown_exposure_role"] = "train_candidate"
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    phrase_path = outdir / "unknown_exposure_phrase_summary.jsonl"
    with phrase_path.open("w", encoding="utf-8") as f:
        for phr, count in sorted(
            phrase_counter.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            f.write(
                json.dumps(
                    {
                        "normalized_phrase": phr,
                        "utterance_count": int(count),
                        "speaker_count": len(phrase_speakers[phr]),
                        "domain_counts": dict(phrase_domains[phr]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    support = support_summary(phrase_counter)
    report = {
        "schema": SCHEMA,
        "source_manifest": {
            "path": str(manifest),
            "sha256": sha256(manifest),
            "discovery_candidates": discovery,
        },
        "counts": {
            "full_manifest_rows": len(full_rows),
            "mdsc_rows": len(mdsc_rows),
            "mdsc_train_rows": len(train_rows),
            "core30_phrase_count": len(core30),
            "open_dev_utt_count": len(open_ids),
            "open_dev_real_phrase_count": len(open_dev_phrases),
            "candidate_rows": len(candidates),
            "candidate_phrase_count": len(phrase_counter),
            "candidate_speaker_count": len(speakers),
            "candidate_domain_counts": dict(domains),
            "exclusion_counts": dict(exclusions),
        },
        "candidate_phrase_support": support,
        "development_leakage_checks": {
            "candidate_vs_core30_phrase_overlap": len(overlap_core),
            "candidate_vs_open_dev_phrase_overlap": len(overlap_dev),
        },
        "methodological_note": (
            "Open-set DEV phrase identities are excluded from exposure training. "
            "generic_test remains sealed. Whether final test unknown phrases "
            "overlap train-side exposure phrases must be reported only after "
            "the model is frozen; the final evaluation should distinguish "
            "seen-unknown vs unseen-unknown phrases if overlap exists."
        ),
        "generic_test": "sealed_not_accessed",
    }
    (outdir / "p6_unknown_exposure_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 UNKNOWN EXPOSURE TRAIN-SIDE DATA AUDIT v5")
    print("=" * 108)
    print(f"manifest:                     {manifest}")
    print(f"full rows:                    {len(full_rows):,}")
    print(f"MDSC rows:                    {len(mdsc_rows):,}")
    print(f"MDSC train rows:              {len(train_rows):,}")
    print(f"Core30 phrases:               {len(core30)}")
    print(f"open DEV utterances:          {len(open_ids):,}")
    print(f"open DEV real phrases:        {len(open_dev_phrases):,}")
    print("-" * 108)
    print(f"candidate rows:               {len(candidates):,}")
    print(f"candidate phrases:            {len(phrase_counter):,}")
    print(f"candidate speakers:           {len(speakers):,}")
    print(f"candidate domains:            {dict(domains)}")
    print("-" * 108)
    print(
        "phrase support: "
        f"median={support['median']:.1f}, "
        f"p90={support['p90']:.1f}, max={support['max']}"
    )
    print(
        "phrases >=2/5/10/15:      "
        f"{support['ge_2']} / {support['ge_5']} / "
        f"{support['ge_10']} / {support['ge_15']}"
    )
    print("-" * 108)
    print(f"candidate/Core30 overlap:     {len(overlap_core)}")
    print(f"candidate/open-DEV overlap:   {len(overlap_dev)}")
    print("generic_test accessed:        NO")
    print("P6 UNKNOWN EXPOSURE DATA AUDIT v5 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
