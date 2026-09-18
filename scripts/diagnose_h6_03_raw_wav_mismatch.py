#!/usr/bin/env python
"""H6-03A — Localize Raw-WAV parity mismatch against the frozen P5 frame cache.

Why
---
H6-03 showed:
    global cosine ~0.95
    temporal frame cosine ~0.88
while temporal shapes matched.

That means we are using the same utterances / roughly the same timeline, but
the numerical representation path is not identical.

This diagnostic compares the NEW raw-WAV WavLM hidden states directly against
the ORIGINAL P5 [T,1024] frame cache, before Attention/256D projection.

It tests:
- waveform loader parity (soundfile vs torchaudio)
- waveform preprocessing:
    raw waveform
    manual zero-mean/unit-variance normalization
    HF feature extractor default
    HF feature extractor + explicit attention mask
- hidden-state indices 12..18

No model parameter is trained or changed.
TEST/generic_test is not accessed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel

from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset


MODEL_ID = "microsoft/wavlm-large"
REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
LAYER_INDICES = tuple(range(12, 19))
TARGET_SR = 16000


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


def row_utt(row):
    v = get(row, ("utt_id", "id", "audio_id"))
    if v == "":
        raise KeyError(f"No utt id in keys={sorted(row)}")
    return str(v)


def row_split(row):
    return str(get(row, ("split",))).strip().lower()


def build_audio_map(manifest: Path, audio_root: Path):
    out = {}
    for row in read_jsonl(manifest):
        sp = row_split(row)
        if sp == "test":
            continue
        if sp not in ("train", "dev"):
            continue
        rel = get(
            row,
            (
                "audio_relpath",
                "audio_path",
                "wav_path",
                "source_path",
                "filepath",
                "file_path",
                "path",
                "file",
            ),
        )
        if not rel:
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = audio_root / p
        if p.exists():
            out[row_utt(row)] = p.resolve()
    return out


def load_sf(path: Path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if int(sr) != TARGET_SR:
        x_t = torch.from_numpy(x)
        x = torchaudio.functional.resample(
            x_t, int(sr), TARGET_SR
        ).numpy().astype(np.float32)
        sr = TARGET_SR
    return x, int(sr)


def load_torchaudio(path: Path):
    """Optional loader parity check.

    torchaudio 2.11 may route torchaudio.load through TorchCodec. TorchCodec is
    not required by the validated PAPR pipeline, so absence of that optional
    dependency must not block H6-03A. Return (None, None, reason) when unavailable.
    """
    try:
        x, sr = torchaudio.load(str(path))
    except (ImportError, ModuleNotFoundError) as e:
        return None, None, f"SKIPPED: {type(e).__name__}: {e}"

    x = x.to(torch.float32)
    if x.ndim == 2:
        x = x.mean(dim=0)
    if int(sr) != TARGET_SR:
        x = torchaudio.functional.resample(x, int(sr), TARGET_SR)
        sr = TARGET_SR
    return x.cpu().numpy().astype(np.float32), int(sr), None


def z_norm(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    mean = float(x.mean())
    var = float(x.var())
    return ((x - mean) / math.sqrt(var + 1e-7)).astype(np.float32)


def extract_cached_feature(item):
    def walk(obj, prefix=""):
        hits = []
        if isinstance(obj, dict):
            preferred = (
                "features",
                "feature",
                "hidden",
                "hidden_state",
                "frames",
                "x",
            )
            for key in preferred:
                if key in obj:
                    hits.extend(walk(obj[key], f"{prefix}.{key}" if prefix else key))
            for key, value in obj.items():
                if key not in preferred:
                    hits.extend(walk(value, f"{prefix}.{key}" if prefix else key))
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                hits.extend(walk(value, f"{prefix}[{i}]"))
        elif torch.is_tensor(obj):
            a = obj.detach().cpu().numpy()
            if a.ndim == 2 and a.shape[-1] == 1024:
                hits.append((prefix, a.astype(np.float32)))
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 2 and obj.shape[-1] == 1024:
                hits.append((prefix, obj.astype(np.float32)))
        return hits

    hits = walk(item)
    if not hits:
        raise RuntimeError(
            f"Could not find a [T,1024] feature tensor in FrameCacheDataset item type={type(item)}"
        )
    # Prefer exact/common `features` name if multiple.
    hits.sort(key=lambda kv: (0 if kv[0].endswith("features") else 1, kv[0]))
    return hits[0]


def frame_cosine_stats(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        return {
            "shape_match": False,
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
            "mean": float("nan"),
            "min": float("nan"),
            "p05": float("nan"),
            "max_abs": float("inf"),
        }

    aa = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bb = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    c = np.sum(aa * bb, axis=1)
    return {
        "shape_match": True,
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "mean": float(c.mean()),
        "min": float(c.min()),
        "p05": float(np.quantile(c, 0.05)),
        "max_abs": float(np.max(np.abs(a - b))),
    }


def build_inputs(feature_extractor, waveform, variant, device):
    if variant == "raw":
        return {
            "input_values": torch.from_numpy(waveform)[None].to(device)
        }

    if variant == "manual_znorm":
        return {
            "input_values": torch.from_numpy(z_norm(waveform))[None].to(device)
        }

    if variant == "hf_default":
        inp = feature_extractor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        out = {"input_values": inp["input_values"].to(device)}
        if "attention_mask" in inp:
            out["attention_mask"] = inp["attention_mask"].to(device)
        return out

    if variant == "hf_explicit_mask":
        inp = feature_extractor(
            waveform,
            sampling_rate=TARGET_SR,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_values": inp["input_values"].to(device),
            "attention_mask": inp["attention_mask"].to(device),
        }

    raise ValueError(variant)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--audio-root",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--frame-cache",
        type=Path,
        default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"),
    )
    p.add_argument(
        "--core-index",
        type=Path,
        default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"),
    )
    p.add_argument("--samples", type=int, default=3)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h6_03a_mismatch_localization"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    dev_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="dev",
    )

    ds_pos = {}
    for i, row in enumerate(dev_ds.rows):
        ds_pos[str(row["utt_id"])] = i

    audio_map = build_audio_map(args.manifest, args.audio_root)

    candidates = sorted(set(ds_pos) & set(audio_map))
    if len(candidates) < args.samples:
        raise RuntimeError(
            f"Only {len(candidates)} DEV items have both audio and P5 cache; need {args.samples}"
        )
    selected = candidates[: args.samples]

    print(f"[H6-03A] loading {MODEL_ID}@{REVISION}")
    extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID,
        revision=REVISION,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=REVISION,
    ).eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    variants = (
        "raw",
        "manual_znorm",
        "hf_default",
        "hf_explicit_mask",
    )

    per_sample = []
    aggregate = defaultdict(list)

    print("=" * 112)
    print("H6-03A RAW-WAV -> P5 FRAME-CACHE MISMATCH LOCALIZATION")
    print("=" * 112)

    for n, utt in enumerate(selected, start=1):
        wav_path = audio_map[utt]

        x_sf, sr_sf = load_sf(wav_path)
        x_ta, sr_ta, torchaudio_note = load_torchaudio(wav_path)

        if x_ta is None:
            loader_max_abs = float("nan")
            loader_cos = float("nan")
        elif len(x_sf) == len(x_ta):
            loader_max_abs = float(np.max(np.abs(x_sf - x_ta)))
            loader_cos = float(
                np.dot(x_sf, x_ta)
                / max(np.linalg.norm(x_sf) * np.linalg.norm(x_ta), 1e-12)
            )
        else:
            loader_max_abs = float("inf")
            loader_cos = float("nan")

        item = dev_ds[ds_pos[utt]]
        cache_key, cached = extract_cached_feature(item)

        sample_result = {
            "utt_id": utt,
            "wav_path": str(wav_path),
            "samples_sf": len(x_sf),
            "samples_torchaudio": (len(x_ta) if x_ta is not None else None),
            "sr_sf": sr_sf,
            "sr_torchaudio": sr_ta,
            "torchaudio_loader_note": torchaudio_note,
            "loader_max_abs": loader_max_abs,
            "loader_cosine": loader_cos,
            "cached_feature_key": cache_key,
            "cached_shape": list(cached.shape),
            "comparisons": [],
        }

        if x_ta is None:
            loader_text = "torchaudio.load=SKIPPED (TorchCodec unavailable)"
        else:
            loader_text = (
                f"loader_abs={loader_max_abs:.3e} "
                f"loader_cos={loader_cos:.8f}"
            )

        print(
            f"[{n}/{len(selected)}] {utt} | "
            f"{loader_text} | cache={cached.shape} ({cache_key})"
        )

        for variant in variants:
            model_inputs = build_inputs(
                extractor,
                x_sf,
                variant,
                device,
            )

            with torch.inference_mode():
                out = model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            for layer in LAYER_INDICES:
                hidden = out.hidden_states[layer][0].float().cpu().numpy()
                stats = frame_cosine_stats(hidden, cached)
                row = {
                    "variant": variant,
                    "hidden_state_index": layer,
                    **stats,
                }
                sample_result["comparisons"].append(row)

                if stats["shape_match"]:
                    aggregate[(variant, layer)].append(stats["mean"])

        per_sample.append(sample_result)

        # Print top 5 matches for this utterance.
        valid = [
            r for r in sample_result["comparisons"]
            if r["shape_match"] and np.isfinite(r["mean"])
        ]
        valid.sort(key=lambda r: r["mean"], reverse=True)
        for r in valid[:5]:
            print(
                f"    {r['variant']:16s} hs[{r['hidden_state_index']:2d}] "
                f"mean={r['mean']:.8f} min={r['min']:.8f} "
                f"max_abs={r['max_abs']:.3e}"
            )

    agg_rows = []
    for (variant, layer), vals in aggregate.items():
        arr = np.asarray(vals, dtype=np.float64)
        agg_rows.append(
            {
                "variant": variant,
                "hidden_state_index": layer,
                "samples": len(vals),
                "mean_of_frame_cosine_means": float(arr.mean()),
                "min_of_frame_cosine_means": float(arr.min()),
            }
        )

    agg_rows.sort(
        key=lambda r: r["mean_of_frame_cosine_means"],
        reverse=True,
    )

    best = agg_rows[0] if agg_rows else None

    # Interpretation categories.
    diagnosis = "UNRESOLVED"
    if best is not None:
        if best["mean_of_frame_cosine_means"] >= 0.9999:
            if best["hidden_state_index"] != 15:
                diagnosis = "LIKELY_HIDDEN_STATE_INDEX_MISMATCH"
            elif best["variant"] in ("raw", "manual_znorm"):
                diagnosis = "LIKELY_WAVEFORM_NORMALIZATION_MISMATCH"
            else:
                diagnosis = "EXACT_PIPELINE_VARIANT_IDENTIFIED"
        elif best["mean_of_frame_cosine_means"] >= 0.99:
            diagnosis = "NEAR_MATCH_NEEDS_SMALL_PIPELINE_AUDIT"
        else:
            diagnosis = "DEEPER_PIPELINE_OR_MODEL_IMPLEMENTATION_MISMATCH"

    result = {
        "schema": "papr_ssl.h6_03a_mismatch_localization.v1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "layer_indices_tested": list(LAYER_INDICES),
        "variants_tested": list(variants),
        "torchaudio_load_is_optional_diagnostic_only": True,
        "best_aggregate_match": best,
        "diagnosis": diagnosis,
        "aggregate_ranking": agg_rows,
        "samples": per_sample,
        "generic_test": "sealed_not_accessed",
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 112)
    print("H6-03A AGGREGATE TOP MATCHES")
    print("=" * 112)
    for row in agg_rows[:10]:
        print(
            f"{row['variant']:16s} hs[{row['hidden_state_index']:2d}] | "
            f"mean={row['mean_of_frame_cosine_means']:.8f} "
            f"worst_sample_mean={row['min_of_frame_cosine_means']:.8f}"
        )

    print("-" * 112)
    print("diagnosis:", diagnosis)
    print("generic_test accessed: NO")
    print("H6-03A STATUS: COMPLETE")


if __name__ == "__main__":
    main()
