#!/usr/bin/env python
"""PAPR-SSL P2-02 raw public-dataset audit.

This script performs read-only inspection of:
  1) Google Speech Commands v2 (GSC v0.02)
  2) MDSC / AISHELL-6B

It does NOT:
  - modify audio;
  - resample audio;
  - rename files;
  - generate training manifests;
  - create train/dev/test splits for MDSC;
  - train or load SSL models.

Typical PowerShell usage:

    python scripts/audit_public_datasets.py `
      --gsc-root datasets/public/speech_commands_v2 `
      --mdsc-root datasets/public/mdsc `
      --out-dir artifacts/p2_02

The audit uses audio headers only (soundfile.info), so it does not load full
waveforms into memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf


AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PAPR-SSL P2-02 raw audit for GSC v2 and MDSC/AISHELL-6B."
    )
    parser.add_argument(
        "--gsc-root",
        type=Path,
        default=Path("datasets/public/speech_commands_v2"),
        help="Root directory of extracted Google Speech Commands v2.",
    )
    parser.add_argument(
        "--mdsc-root",
        type=Path,
        default=Path("datasets/public/mdsc"),
        help="Root directory of extracted MDSC/AISHELL-6B.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/p2_02"),
        help="Output directory for JSON audit reports.",
    )
    parser.add_argument(
        "--max-audio",
        type=int,
        default=0,
        help=(
            "Optional header-inspection limit per dataset for debugging only. "
            "0 means inspect every extracted audio file."
        ),
    )
    return parser.parse_args()


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTS


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(s) for s in ARCHIVE_SUFFIXES)


def _safe_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    xs = sorted(values)
    if len(xs) == 1:
        p95 = xs[0]
    else:
        rank = (len(xs) - 1) * 0.95
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            p95 = xs[lo]
        else:
            frac = rank - lo
            p95 = xs[lo] * (1.0 - frac) + xs[hi] * frac
    return {
        "count": len(xs),
        "min": min(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p95": p95,
        "max": max(xs),
    }


def inspect_audio_headers(
    files: list[Path],
    *,
    root: Path,
    max_audio: int,
) -> dict[str, Any]:
    selected = files if max_audio <= 0 else files[:max_audio]

    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    formats: Counter[str] = Counter()
    subtypes: Counter[str] = Counter()
    durations: list[float] = []
    frames: list[int] = []
    failures: list[dict[str, str]] = []

    for path in selected:
        try:
            info = sf.info(str(path))
            sample_rates[int(info.samplerate)] += 1
            channels[int(info.channels)] += 1
            formats[str(info.format)] += 1
            subtypes[str(info.subtype)] += 1
            frames.append(int(info.frames))
            if info.samplerate:
                durations.append(float(info.frames) / float(info.samplerate))
        except Exception as exc:
            failures.append(
                {
                    "path": _rel(path, root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "discovered_audio_files": len(files),
        "inspected_audio_files": len(selected),
        "inspection_limited": max_audio > 0 and len(selected) < len(files),
        "sample_rate_hz": dict(sorted(sample_rates.items())),
        "channels": dict(sorted(channels.items())),
        "formats": dict(sorted(formats.items())),
        "subtypes": dict(sorted(subtypes.items())),
        "duration_seconds": _safe_stats(durations),
        "frame_count": _safe_stats([float(x) for x in frames]),
        "read_failures_count": len(failures),
        "read_failures": failures[:100],
    }


def archive_inventory(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not root.exists():
        return results

    for path in sorted(p for p in root.rglob("*") if _is_archive(p)):
        item: dict[str, Any] = {
            "path": _rel(path, root),
            "size_mb": path.stat().st_size / (1024.0 ** 2),
            "members": None,
            "audio_members": None,
            "top_prefixes": [],
            "error": None,
        }
        try:
            names: list[str] = []
            if path.name.lower().endswith(".zip"):
                with zipfile.ZipFile(path, "r") as zf:
                    names = [x.filename for x in zf.infolist() if not x.is_dir()]
            elif (
                path.name.lower().endswith(".tar")
                or path.name.lower().endswith(".tar.gz")
                or path.name.lower().endswith(".tgz")
            ):
                # Only used when the dataset remains archived.
                with tarfile.open(path, "r:*") as tf:
                    names = [x.name for x in tf.getmembers() if x.isfile()]

            item["members"] = len(names)
            item["audio_members"] = sum(
                Path(name).suffix.lower() in AUDIO_EXTS for name in names
            )
            prefixes = Counter(
                name.replace("\\", "/").split("/")[0]
                for name in names
                if name.strip("/")
            )
            item["top_prefixes"] = [
                {"name": name, "count": count}
                for name, count in prefixes.most_common(20)
            ]
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)

    return results


def path_structure(files: Iterable[Path], root: Path) -> dict[str, Any]:
    top = Counter()
    depth = Counter()
    examples: list[str] = []

    for path in files:
        rel = path.resolve().relative_to(root.resolve())
        parts = rel.parts
        if parts:
            top[parts[0]] += 1
        depth[len(parts)] += 1
        if len(examples) < 30:
            examples.append(rel.as_posix())

    return {
        "top_level_file_distribution": [
            {"name": name, "count": count}
            for name, count in top.most_common()
        ],
        "path_depth_distribution": dict(sorted(depth.items())),
        "examples": examples,
    }


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]


def gsc_speaker_id(filename: str) -> str:
    # Official Speech Commands files normally use:
    # speakerhash_nohash_index.wav
    stem = Path(filename).stem
    return stem.split("_nohash_")[0]


def audit_gsc(root: Path, max_audio: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": "Google Speech Commands v2",
        "expected_release": "speech_commands_v0.02",
        "root": str(root.resolve()),
        "exists": root.exists(),
    }
    if not root.exists():
        result.update({"status": "MISSING_ROOT"})
        return result

    audio_files = sorted(p for p in root.rglob("*") if _is_audio(p))
    archives = archive_inventory(root)

    if not audio_files:
        result.update(
            {
                "status": "NEEDS_EXTRACTION" if archives else "NO_AUDIO_FOUND",
                "archives": archives,
                "entries": sorted(p.name for p in root.iterdir())[:100],
            }
        )
        return result

    wav_files = [p for p in audio_files if p.suffix.lower() == ".wav"]
    rel_to_path = {_rel(p, root): p for p in wav_files}

    validation_file = root / "validation_list.txt"
    testing_file = root / "testing_list.txt"
    validation = set(read_lines(validation_file))
    testing = set(read_lines(testing_file))

    overlap = sorted(validation & testing)
    missing_validation = sorted(x for x in validation if x not in rel_to_path)
    missing_testing = sorted(x for x in testing if x not in rel_to_path)

    background = {
        rel
        for rel in rel_to_path
        if rel.startswith("_background_noise_/")
    }
    speech = set(rel_to_path) - background
    train = speech - validation - testing

    class_counts: Counter[str] = Counter()
    split_class_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    split_paths = {
        "train": train,
        "validation": validation & speech,
        "test": testing & speech,
    }

    for rel in speech:
        label = rel.split("/", 1)[0]
        class_counts[label] += 1
    for split_name, paths in split_paths.items():
        for rel in paths:
            split_class_counts[split_name][rel.split("/", 1)[0]] += 1

    speaker_sets: dict[str, set[str]] = {}
    for split_name, paths in split_paths.items():
        speaker_sets[split_name] = {gsc_speaker_id(Path(rel).name) for rel in paths}

    speaker_overlap = {
        "train_validation": sorted(
            speaker_sets["train"] & speaker_sets["validation"]
        ),
        "train_test": sorted(speaker_sets["train"] & speaker_sets["test"]),
        "validation_test": sorted(
            speaker_sets["validation"] & speaker_sets["test"]
        ),
    }

    result.update(
        {
            "status": "PASS" if not overlap and not missing_validation and not missing_testing else "WARN",
            "official_split_files": {
                "validation_list_txt": validation_file.is_file(),
                "testing_list_txt": testing_file.is_file(),
                "validation_entries": len(validation),
                "testing_entries": len(testing),
                "validation_test_overlap": overlap[:100],
                "missing_validation_paths": missing_validation[:100],
                "missing_testing_paths": missing_testing[:100],
            },
            "counts": {
                "all_audio": len(audio_files),
                "wav": len(wav_files),
                "speech_wav": len(speech),
                "background_noise_wav": len(background),
                "classes": len(class_counts),
            },
            "class_counts": dict(sorted(class_counts.items())),
            "split_counts": {
                name: len(paths) for name, paths in split_paths.items()
            },
            "split_class_counts": {
                name: dict(sorted(counts.items()))
                for name, counts in split_class_counts.items()
            },
            "speaker_counts": {
                name: len(values) for name, values in speaker_sets.items()
            },
            "speaker_overlap_counts": {
                name: len(values) for name, values in speaker_overlap.items()
            },
            "speaker_overlap_examples": {
                name: values[:30] for name, values in speaker_overlap.items()
            },
            "audio_headers": inspect_audio_headers(
                audio_files, root=root, max_audio=max_audio
            ),
            "structure": path_structure(audio_files, root),
            "archives": archives,
        }
    )
    return result


def guess_mdsc_group(rel: str) -> str:
    parts = [part.lower() for part in Path(rel).parts]
    for part in parts:
        if part == "uncontrol" or "uncontrol" in part:
            return "Uncontrol"
    for part in parts:
        if part == "control" or "control" in part:
            return "Control"
    return "UNKNOWN"


def audit_mdsc(root: Path, max_audio: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": "MDSC / AISHELL-6B",
        "root": str(root.resolve()),
        "exists": root.exists(),
    }
    if not root.exists():
        result.update({"status": "MISSING_ROOT"})
        return result

    all_files = sorted(p for p in root.rglob("*") if p.is_file())
    audio_files = [p for p in all_files if _is_audio(p)]
    archives = archive_inventory(root)

    metadata_candidates = [
        _rel(p, root)
        for p in all_files
        if p.suffix.lower() in {".xlsx", ".xls", ".csv", ".txt", ".json"}
    ]

    if not audio_files:
        result.update(
            {
                "status": "NEEDS_EXTRACTION" if archives else "NO_AUDIO_FOUND",
                "archives": archives,
                "metadata_candidates": metadata_candidates,
                "entries": sorted(p.name for p in root.iterdir())[:100],
            }
        )
        return result

    groups: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    for path in audio_files:
        rel = _rel(path, root)
        groups[guess_mdsc_group(rel)] += 1
        extension_counts[path.suffix.lower()] += 1

    # Structural summaries intentionally avoid guessing speaker IDs before the
    # real AISHELL-6B path/metadata semantics are reviewed.
    top_dirs = Counter()
    second_dirs = Counter()
    for path in audio_files:
        rel_parts = path.resolve().relative_to(root.resolve()).parts
        if len(rel_parts) >= 1:
            top_dirs[rel_parts[0]] += 1
        if len(rel_parts) >= 2:
            second_dirs["/".join(rel_parts[:2])] += 1

    result.update(
        {
            "status": "PASS",
            "counts": {
                "all_files": len(all_files),
                "all_audio": len(audio_files),
                "by_extension": dict(sorted(extension_counts.items())),
            },
            "group_from_path_name_only": dict(sorted(groups.items())),
            "top_level_audio_distribution": [
                {"path": name, "count": count}
                for name, count in top_dirs.most_common()
            ],
            "second_level_audio_distribution": [
                {"path": name, "count": count}
                for name, count in second_dirs.most_common(100)
            ],
            "metadata_candidates": metadata_candidates,
            "audio_headers": inspect_audio_headers(
                audio_files, root=root, max_audio=max_audio
            ),
            "structure": path_structure(audio_files, root),
            "archives": archives,
            "important_note": (
                "P2-02 does not infer MDSC speaker_id, label, severity, or "
                "official split semantics yet. Those must be derived from the "
                "actual README/metadata/path convention after this audit."
            ),
        }
    )
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_dataset_summary(report: dict[str, Any]) -> None:
    print("=" * 88)
    print(report["dataset"])
    print("=" * 88)
    print(f"Root:          {report['root']}")
    print(f"Status:        {report.get('status')}")
    if "counts" in report:
        print(f"Counts:        {json.dumps(report['counts'], ensure_ascii=False)}")
    if "split_counts" in report:
        print(f"Split counts:  {json.dumps(report['split_counts'], ensure_ascii=False)}")
    if "speaker_counts" in report:
        print(f"Speakers:      {json.dumps(report['speaker_counts'], ensure_ascii=False)}")
    if "group_from_path_name_only" in report:
        print(
            "Path groups:   "
            f"{json.dumps(report['group_from_path_name_only'], ensure_ascii=False)}"
        )
    audio = report.get("audio_headers")
    if audio:
        print(f"Sample rates:  {audio['sample_rate_hz']}")
        print(f"Channels:      {audio['channels']}")
        print(f"Duration:      {audio['duration_seconds']}")
        print(f"Read failures: {audio['read_failures_count']}")
    if report.get("status") == "NEEDS_EXTRACTION":
        print("Action:        Extract the raw archive(s) without renaming/restructuring.")
    print()


def main() -> int:
    args = parse_args()
    if args.max_audio < 0:
        raise ValueError("--max-audio must be >= 0")

    gsc = audit_gsc(args.gsc_root, args.max_audio)
    mdsc = audit_mdsc(args.mdsc_root, args.max_audio)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gsc_out = args.out_dir / "gsc_v2_raw_audit.json"
    mdsc_out = args.out_dir / "mdsc_raw_audit.json"
    combined_out = args.out_dir / "public_datasets_raw_audit.json"

    write_json(gsc_out, gsc)
    write_json(mdsc_out, mdsc)
    write_json(
        combined_out,
        {
            "schema": "papr_ssl.public_dataset_raw_audit.v1",
            "phase": "P2-02",
            "read_only": True,
            "gsc": gsc,
            "mdsc": mdsc,
        },
    )

    print_dataset_summary(gsc)
    print_dataset_summary(mdsc)
    print("=" * 88)
    print("P2-02 RAW DATA AUDIT")
    print("=" * 88)
    print(f"GSC:           {gsc.get('status')}")
    print(f"MDSC:          {mdsc.get('status')}")
    print(f"GSC report:    {gsc_out}")
    print(f"MDSC report:   {mdsc_out}")
    print(f"Combined:      {combined_out}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
