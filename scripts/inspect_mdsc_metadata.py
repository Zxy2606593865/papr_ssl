#!/usr/bin/env python
"""PAPR-SSL P2-04A read-only MDSC metadata inspection.

Purpose:
- inspect README.txt and transcript label.txt formats;
- verify wav <-> transcript key matching;
- enumerate speaker/split/role structure;
- collect representative raw examples before implementing MDSCAdapter.

This script NEVER edits raw data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect MDSC/AISHELL-6B metadata before writing the adapter."
    )
    parser.add_argument(
        "--mdsc-root",
        type=Path,
        default=Path("datasets/public/mdsc"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/p2_04/mdsc_metadata_inspection.json"),
    )
    parser.add_argument(
        "--examples-per-label-file",
        type=int,
        default=5,
    )
    return parser.parse_args()


def read_text_fallback(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def infer_split_role_domain(path: Path, root: Path) -> dict[str, str | None]:
    parts = path.resolve().relative_to(root.resolve()).parts
    low = [x.lower() for x in parts]

    domain = None
    if low and low[0] == "control":
        domain = "control"
    elif low and low[0] == "uncontrol":
        domain = "dysarthria"

    split = next((x for x in ("train", "dev", "test") if x in low), None)
    role = next((x for x in ("enrollment", "eval") if x in low), None)
    if role is None and split == "train":
        role = "train"

    return {"domain": domain, "split": split, "role": role}


def candidate_delimiter(line: str) -> str:
    if "\t" in line:
        return "TAB"
    if "|" in line:
        return "PIPE"
    if "," in line:
        return "COMMA"
    if " " in line:
        return "SPACE"
    return "NONE"


def first_field(line: str) -> str:
    for sep in ("\t", "|", ","):
        if sep in line:
            return line.split(sep, 1)[0].strip()
    return line.split(maxsplit=1)[0].strip() if line.strip() else ""


def normalize_key(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = Path(value).name
    return Path(value).stem


def find_wav_files_for_label(label_path: Path, root: Path) -> list[Path]:
    """
    Find the most likely WAV subtree associated with a transcript/label.txt.
    This does not depend on one exact MDSC directory layout.
    """
    rel_parts = label_path.resolve().relative_to(root.resolve()).parts
    low = [p.lower() for p in rel_parts]

    # Replace the transcript component by wav and keep path prefix.
    if "transcript" in low:
        idx = low.index("transcript")
        prefix = root.joinpath(*rel_parts[:idx])
        suffix_parts = list(rel_parts[idx + 1 : -1])
        candidate = prefix / "wav"
        if suffix_parts:
            candidate = candidate.joinpath(*suffix_parts)
        if candidate.exists():
            return sorted(candidate.rglob("*.wav"))

    # Fallback: look within the nearest split/role directory.
    parent = label_path.parent
    for ancestor in [parent, *parent.parents]:
        if ancestor == root.parent:
            break
        wav_dir = ancestor / "wav"
        if wav_dir.exists():
            return sorted(wav_dir.rglob("*.wav"))
        if ancestor == root:
            break

    return []


def inspect_label_file(
    path: Path,
    *,
    root: Path,
    examples_per_file: int,
) -> dict[str, Any]:
    text, encoding = read_text_fallback(path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    delim_counts = Counter(candidate_delimiter(line) for line in lines)
    token_counts = Counter()
    keys: list[str] = []

    for line in lines:
        delim = candidate_delimiter(line)
        if delim == "TAB":
            n = len(line.split("\t"))
        elif delim == "PIPE":
            n = len(line.split("|"))
        elif delim == "COMMA":
            n = len(line.split(","))
        else:
            n = len(line.split())
        token_counts[n] += 1
        keys.append(normalize_key(first_field(line)))

    wav_files = find_wav_files_for_label(path, root)
    wav_keys = {p.stem for p in wav_files}
    label_keys = {k for k in keys if k}

    matched = wav_keys & label_keys
    missing_label_for_wav = sorted(wav_keys - label_keys)
    label_without_wav = sorted(label_keys - wav_keys)

    meta = infer_split_role_domain(path, root)

    return {
        "path": rel(path, root),
        "encoding": encoding,
        "line_count": len(lines),
        "delimiter_counts": dict(delim_counts),
        "field_count_distribution": dict(sorted(token_counts.items())),
        "examples": lines[:examples_per_file],
        "inferred": meta,
        "associated_wav_count": len(wav_files),
        "unique_label_key_count": len(label_keys),
        "matched_wav_key_count": len(matched),
        "missing_label_for_wav_count": len(missing_label_for_wav),
        "label_without_wav_count": len(label_without_wav),
        "missing_label_for_wav_examples": missing_label_for_wav[:20],
        "label_without_wav_examples": label_without_wav[:20],
    }


def speaker_inventory(root: Path) -> dict[str, Any]:
    rows = []
    speaker_counts = Counter()
    speaker_by_partition: dict[str, set[str]] = defaultdict(set)

    for wav in sorted(root.rglob("*.wav")):
        parts = wav.resolve().relative_to(root.resolve()).parts
        low = [p.lower() for p in parts]
        meta = infer_split_role_domain(wav, root)

        speaker = wav.stem.split("_", 1)[0]
        # Prefer directory-based speaker when available.
        if "wav" in low:
            idx = low.index("wav")
            if idx + 1 < len(parts) - 1:
                maybe = parts[idx + 1]
                if maybe.upper().startswith(("CF", "CM", "DF", "DM")):
                    speaker = maybe

        speaker_counts[speaker] += 1
        key = f"{meta['domain']}|{meta['split']}|{meta['role']}"
        speaker_by_partition[key].add(speaker)

    for key, speakers in sorted(speaker_by_partition.items()):
        domain, split, role = key.split("|")
        rows.append(
            {
                "domain": None if domain == "None" else domain,
                "split": None if split == "None" else split,
                "role": None if role == "None" else role,
                "speaker_count": len(speakers),
                "speakers": sorted(speakers),
            }
        )

    return {
        "total_unique_speakers": len(speaker_counts),
        "speaker_audio_counts": dict(sorted(speaker_counts.items())),
        "partitions": rows,
    }


def main() -> int:
    args = parse_args()
    root = args.mdsc_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MDSC root not found: {root}")

    readme_path = root / "README.txt"
    readme = None
    if readme_path.is_file():
        text, enc = read_text_fallback(readme_path)
        readme = {
            "path": rel(readme_path, root),
            "encoding": enc,
            "line_count": len(text.splitlines()),
            # Raw README is included because this is an internal audit artifact.
            "text": text,
        }

    label_files = sorted(root.rglob("label.txt"))
    labels = [
        inspect_label_file(
            p,
            root=root,
            examples_per_file=args.examples_per_label_file,
        )
        for p in label_files
    ]

    intelligibility = root / "Intelligibility.xlsx"

    report = {
        "schema": "papr_ssl.mdsc_metadata_inspection.v1",
        "phase": "P2-04A",
        "read_only": True,
        "root": str(root),
        "readme": readme,
        "intelligibility_xlsx": {
            "exists": intelligibility.is_file(),
            "path": rel(intelligibility, root) if intelligibility.is_file() else None,
            "size_bytes": intelligibility.stat().st_size if intelligibility.is_file() else None,
        },
        "label_file_count": len(label_files),
        "label_files": labels,
        "speaker_inventory": speaker_inventory(root),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("PAPR-SSL P2-04A MDSC METADATA INSPECTION")
    print("=" * 88)
    print(f"Root:                 {root}")
    print(f"README:               {'FOUND' if readme else 'MISSING'}")
    print(f"Intelligibility.xlsx: {'FOUND' if intelligibility.is_file() else 'MISSING'}")
    print(f"label.txt files:      {len(label_files)}")
    print(f"Unique speakers:      {report['speaker_inventory']['total_unique_speakers']}")
    print("-" * 88)

    for item in labels:
        print(item["path"])
        print(
            f"  lines={item['line_count']} "
            f"encoding={item['encoding']} "
            f"delimiter={item['delimiter_counts']} "
            f"wav={item['associated_wav_count']} "
            f"matched={item['matched_wav_key_count']} "
            f"missing_wav_labels={item['label_without_wav_count']} "
            f"wav_without_labels={item['missing_label_for_wav_count']}"
        )
        for example in item["examples"][:2]:
            print(f"    example: {example}")
    print("-" * 88)
    print(f"Report written to: {args.out}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
