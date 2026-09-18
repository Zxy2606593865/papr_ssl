#!/usr/bin/env python
"""H6-03 raw-WAV parity check.

Before letting real WAV enter the H6 runtime, verify that raw extraction
reproduces the frozen H4 feature artifacts on previously seen DEV WAV files.

This is NOT a benchmark. It checks implementation parity only.

Comparison:
    raw WAV -> WavLM[15] -> frozen 256D features
vs
    previously materialized H4 cached features

generic_test is not accessed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from papr_ssl.inference.h6_raw_wav_adapter import (
    RawWavFeatureAdapter,
    resolve_selected_checkpoint,
)


PATH_KEYS = (
    "audio_relpath",
    "audio_path",
    "wav_path",
    "source_path",
    "filepath",
    "file_path",
    "path",
    "file",
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get(row, keys, default=""):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def row_utt(row):
    value = get(row, ("utt_id", "id", "audio_id"))
    if value == "":
        raise KeyError(f"No utt_id-like field in keys={sorted(row)}")
    return str(value)


def row_split(row):
    return str(get(row, ("split",))).strip().lower()


def extract_path_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in PATH_KEYS:
            if key in value and isinstance(value[key], str):
                return value[key]
    return None


def find_audio_string(row):
    for key in PATH_KEYS:
        if key in row:
            value = extract_path_value(row[key])
            if value:
                return value

    # Common nested `audio: {path: ...}` form.
    if "audio" in row:
        value = extract_path_value(row["audio"])
        if value:
            return value

    return None


def resolve_audio_path(raw, *, project_root: Path, manifest_path: Path, audio_root: Path | None):
    p = Path(raw)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if audio_root is not None:
            candidates.append(audio_root / p)
        candidates.extend(
            [
                project_root / p,
                manifest_path.parent / p,
                Path.cwd() / p,
                p,
            ]
        )

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    return None


def build_audio_map(manifest_path: Path, project_root: Path, audio_root: Path | None):
    out = {}
    missing_path_field = []
    unresolved = []

    for row in read_jsonl(manifest_path):
        split = row_split(row)
        if split == "test":
            continue
        if split not in ("train", "dev"):
            continue

        utt = row_utt(row)
        raw = find_audio_string(row)
        if raw is None:
            missing_path_field.append((utt, sorted(row.keys())))
            continue

        resolved = resolve_audio_path(
            raw,
            project_root=project_root,
            manifest_path=manifest_path,
            audio_root=audio_root,
        )
        if resolved is None:
            unresolved.append((utt, raw))
            continue

        out[utt] = resolved

    return out, missing_path_field, unresolved


def load_temporal_reference(feature_dir: Path):
    index = {}
    index_path = feature_dir / "temporal" / "index.jsonl"
    for row in read_jsonl(index_path):
        index[row["utt_id"]] = feature_dir / "temporal" / row["cache_relpath"]
    return index


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    return float(np.dot(a, b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("artifacts/h4_core30_features"),
    )
    p.add_argument(
        "--mdsc-manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--embedding-manifest",
        type=Path,
        default=Path(
            "artifacts/p6_teacher_256_15shot/embeddings/manifest.json"
        ),
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Optional dataset root if manifest stores relative WAV paths.",
    )
    p.add_argument("--samples", type=int, default=8)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h6_03_raw_wav_parity"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path.cwd().resolve()

    checkpoint = resolve_selected_checkpoint(args.embedding_manifest)
    print(f"[H6-03] checkpoint: {checkpoint}")

    d = np.load(
        args.feature_dir / "global" / "core30_global256.npz",
        allow_pickle=False,
    )
    dev_ids = d["dev_utt_id"].astype(str)
    dev_global = d["dev_embedding"].astype(np.float32)
    global_ref = {u: dev_global[i] for i, u in enumerate(dev_ids)}

    temporal_ref = load_temporal_reference(args.feature_dir)

    audio_map, missing_path_field, unresolved = build_audio_map(
        args.mdsc_manifest,
        project_root=project_root,
        audio_root=args.audio_root,
    )

    candidates = [
        u for u in sorted(dev_ids)
        if u in audio_map and u in temporal_ref
    ]

    if len(candidates) < args.samples:
        diagnostic = {
            "requested_samples": args.samples,
            "resolved_dev_audio": len(candidates),
            "manifest_rows_missing_path_field": len(missing_path_field),
            "unresolved_paths": len(unresolved),
            "first_missing_path_field": missing_path_field[:3],
            "first_unresolved": unresolved[:5],
        }
        (args.output_dir / "path_diagnostic.json").write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Only {len(candidates)} DEV WAVs could be resolved; "
            f"need {args.samples}. See {args.output_dir/'path_diagnostic.json'}. "
            f"The manifest uses relative paths such as "
            f"'Control/dev/wav/CF0010/CF0010_0001.wav'. "
            f"Rerun with --audio-root pointing to the directory that directly contains "
            f"the Control/ and Dysarthria/ (or equivalent) dataset folders."
        )

    selected = candidates[: args.samples]

    adapter = RawWavFeatureAdapter(
        checkpoint=checkpoint,
        device=args.device,
    )

    rows = []
    all_global_cos = []
    all_temporal_cos = []
    global_max_abs = []
    temporal_max_abs = []

    print("=" * 108)
    print("H6-03 RAW WAV -> FROZEN FEATURE PARITY")
    print("=" * 108)

    for n, utt in enumerate(selected, start=1):
        raw = adapter.extract_wav(audio_map[utt])

        ref_g = global_ref[utt].astype(np.float32)

        ref_obj = torch.load(
            temporal_ref[utt],
            map_location="cpu",
            weights_only=False,
        )
        ref_t = ref_obj["features"].to(torch.float32).numpy()
        raw_t = np.asarray(raw["temporal_sequence"], dtype=np.float32)

        shape_match = tuple(raw_t.shape) == tuple(ref_t.shape)

        g_cos = cosine(raw["global_embedding"], ref_g)
        g_abs = float(
            np.max(
                np.abs(
                    np.asarray(raw["global_embedding"], dtype=np.float32)
                    - ref_g
                )
            )
        )

        if shape_match:
            # Per-frame cosine: robust to tiny float16 cache rounding.
            rt = raw_t / np.maximum(
                np.linalg.norm(raw_t, axis=1, keepdims=True), 1e-12
            )
            rf = ref_t / np.maximum(
                np.linalg.norm(ref_t, axis=1, keepdims=True), 1e-12
            )
            frame_cos = np.sum(rt * rf, axis=1)
            t_cos_mean = float(np.mean(frame_cos))
            t_cos_min = float(np.min(frame_cos))
            t_abs = float(np.max(np.abs(raw_t - ref_t)))
        else:
            t_cos_mean = float("nan")
            t_cos_min = float("nan")
            t_abs = float("inf")

        row = {
            "utt_id": utt,
            "wav_path": str(audio_map[utt]),
            "raw_hidden_frames": int(raw["hidden_frames"]),
            "raw_temporal_shape": list(raw_t.shape),
            "ref_temporal_shape": list(ref_t.shape),
            "temporal_shape_match": shape_match,
            "global_cosine": g_cos,
            "global_max_abs_diff": g_abs,
            "temporal_frame_cosine_mean": t_cos_mean,
            "temporal_frame_cosine_min": t_cos_min,
            "temporal_max_abs_diff": t_abs,
        }
        rows.append(row)

        all_global_cos.append(g_cos)
        global_max_abs.append(g_abs)

        if shape_match:
            all_temporal_cos.append(t_cos_min)
            temporal_max_abs.append(t_abs)

        print(
            f"[{n}/{len(selected)}] {utt} | "
            f"Gcos={g_cos:.8f} Gabs={g_abs:.3e} | "
            f"Tshape={'OK' if shape_match else 'FAIL'} "
            f"TcosMin={t_cos_min:.8f} Tabs={t_abs:.3e}"
        )

    all_shape_match = all(r["temporal_shape_match"] for r in rows)
    min_global_cos = float(min(all_global_cos))
    max_global_abs = float(max(global_max_abs))
    min_temporal_cos = (
        float(min(all_temporal_cos)) if all_temporal_cos else float("-inf")
    )
    max_temporal_abs = (
        float(max(temporal_max_abs)) if temporal_max_abs else float("inf")
    )

    # Cosine thresholds are primary because the reference temporal cache is float16.
    pass_global = min_global_cos >= 0.99999
    pass_temporal = all_shape_match and min_temporal_cos >= 0.9999
    passed = pass_global and pass_temporal

    summary = {
        "schema": "papr_ssl.h6_03_raw_wav_parity.v1",
        "samples": len(rows),
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "hidden_state_index": adapter.hidden_state_index,
        "checkpoint": str(checkpoint),
        "global": {
            "min_cosine": min_global_cos,
            "max_abs_diff": max_global_abs,
            "threshold_min_cosine": 0.99999,
            "pass": pass_global,
        },
        "temporal": {
            "all_shape_match": all_shape_match,
            "min_frame_cosine": min_temporal_cos,
            "max_abs_diff": max_temporal_abs,
            "threshold_min_frame_cosine": 0.9999,
            "pass": pass_temporal,
        },
        "status": "PASS" if passed else "FAIL",
        "raw_wav_adapter_promoted": bool(passed),
        "generic_test": "sealed_not_accessed",
        "rows": rows,
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("-" * 108)
    print(f"global min cosine:       {min_global_cos:.8f}")
    print(f"temporal min frame cos:  {min_temporal_cos:.8f}")
    print(f"temporal shape match:    {'YES' if all_shape_match else 'NO'}")
    print(f"raw WAV adapter promote: {'YES' if passed else 'NO'}")
    print("generic_test accessed:   NO")
    print(f"H6-03 STATUS:            {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
