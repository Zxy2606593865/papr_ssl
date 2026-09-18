#!/usr/bin/env python
"""Build frozen WavLM hidden-state cache for layers 13..17 (train/dev only).

Scientific purpose
------------------
This cache supports a clean comparison between:
1) single layer 15 -> AttentionDR-256D -> SCAF
2) learnable weighted fusion of layers 13..17 -> AttentionDR-256D -> SCAF

Important:
- WavLM remains completely frozen.
- generic_test is never loaded.
- Cache defaults to float16 to reduce disk use. Both experimental arms use
  this exact same cache, so cache precision is controlled.
- Each utterance is cropped to its semantic frame length before saving, so
  padding frames never become training features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WavLMModel

from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset
from papr_ssl.training.teacher.real_experiment import (
    build_mdsc_audio_catalog,
    load_real_task_rows,
    qualified_utt_relpath,
    resolve_task_audio_path,
)

MODEL_ID = "microsoft/wavlm-large"
MODEL_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
LAYERS = (13, 14, 15, 16, 17)
SAMPLE_RATE = 16000


def safe_name(utt_id: str) -> str:
    return hashlib.sha1(utt_id.encode("utf-8")).hexdigest() + ".pt"


def load_wave(path: Path) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if int(sr) != SAMPLE_RATE:
        raise RuntimeError(f"Expected 16 kHz, got {sr}: {path}")
    wav = torch.from_numpy(data).to(torch.float32).mean(dim=1)
    if wav.numel() == 0 or not torch.isfinite(wav).all():
        raise RuntimeError(f"Invalid waveform: {path}")
    return wav.contiguous()


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frame-cache15",
        type=Path,
        default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"),
        help="Existing P5 layer-15 cache; used only to recover the frozen Core30 train/dev rows/classes.",
    )
    p.add_argument(
        "--core-index",
        type=Path,
        default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"),
    )
    p.add_argument(
        "--mdsc-root",
        type=Path,
        default=Path("datasets/public/mdsc"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_wavlm_layers13_17_cache"),
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty cache: {args.output_dir}")

    data_dir = args.output_dir / "utterances"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FrameCacheDataset(
        cache_dir=args.frame_cache15,
        task_index=args.core_index,
        split="train",
    )
    dev_ds = FrameCacheDataset(
        cache_dir=args.frame_cache15,
        task_index=args.core_index,
        split="dev",
        class_to_index=train_ds.class_to_index,
    )
    if len(train_ds) != 3756 or len(dev_ds) != 442:
        raise RuntimeError(
            f"Expected Core30 train/dev 3756/442, got {len(train_ds)}/{len(dev_ds)}"
        )

    task_rows = load_real_task_rows(args.core_index, splits=("train", "dev"))
    by_utt = {r.utt_id: r for r in task_rows}
    expected_ids = [r["utt_id"] for r in train_ds.rows] + [r["utt_id"] for r in dev_ds.rows]

    missing = [u for u in expected_ids if u not in by_utt]
    if missing:
        raise RuntimeError(f"{len(missing)} Core30 train/dev utt_ids missing from task rows; first={missing[:5]}")

    legacy = [
        by_utt[u] for u in expected_ids
        if by_utt[u].audio_relpath is None and qualified_utt_relpath(by_utt[u]) is None
    ]
    catalog = build_mdsc_audio_catalog(args.mdsc_root) if legacy else None

    device = torch.device(args.device)
    model = WavLMModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        output_hidden_states=True,
        local_files_only=True,
    )
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    storage_dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
    index_rows = []
    total_frames = 0

    ordered = []
    for split, ds in (("train", train_ds), ("dev", dev_ds)):
        for row in ds.rows:
            ordered.append((split, row["utt_id"]))

    for batch_i, items in enumerate(batched(ordered, args.batch_size), start=1):
        waves = []
        paths = []
        for split, utt_id in items:
            trow = by_utt[utt_id]
            path = resolve_task_audio_path(
                row=trow,
                mdsc_root=args.mdsc_root,
                audio_catalog=catalog,
            )
            wav = load_wave(path)
            waves.append(wav)
            paths.append(path)

        max_len = max(w.numel() for w in waves)
        x = torch.zeros((len(waves), max_len), dtype=torch.float32)
        am = torch.zeros((len(waves), max_len), dtype=torch.long)
        for i, wav in enumerate(waves):
            n = wav.numel()
            x[i, :n] = wav
            am[i, :n] = 1

        x = x.to(device, non_blocking=True)
        am = am.to(device, non_blocking=True)

        with torch.inference_mode():
            out = model(
                input_values=x,
                attention_mask=am,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden = [out.hidden_states[layer].detach() for layer in LAYERS]
        t_out = hidden[0].shape[1]
        frame_mask = model._get_feature_vector_attention_mask(t_out, am).to(torch.bool)

        for i, (split, utt_id) in enumerate(items):
            valid_t = int(frame_mask[i].sum().item())
            if valid_t <= 0:
                raise RuntimeError(f"No valid WavLM frames for {utt_id}")

            stacked = torch.stack(
                [h[i, :valid_t].to(torch.float32).cpu() for h in hidden],
                dim=0,
            ).to(storage_dtype).contiguous()

            rel = Path("utterances") / safe_name(utt_id)
            torch.save(
                {
                    "schema": "papr_ssl.wavlm_multilayer_cache_item.v1",
                    "utt_id": utt_id,
                    "layers": list(LAYERS),
                    "features": stacked,  # [5,T,1024]
                    "valid_frames": valid_t,
                },
                args.output_dir / rel,
            )

            index_rows.append(
                {
                    "utt_id": utt_id,
                    "split": split,
                    "cache_relpath": str(rel).replace("\\", "/"),
                    "valid_frames": valid_t,
                    "num_layers": len(LAYERS),
                    "feature_dim": int(stacked.shape[-1]),
                    "storage_dtype": args.storage_dtype,
                }
            )
            total_frames += valid_t

        done = min(batch_i * args.batch_size, len(ordered))
        print(f"[cache] {done}/{len(ordered)} utterances")

        del out, hidden, x, am
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (args.output_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    bytes_per = 2 if args.storage_dtype == "float16" else 4
    estimated_bytes = total_frames * len(LAYERS) * 1024 * bytes_per
    manifest = {
        "schema": "papr_ssl.wavlm_multilayer_cache.v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layers": list(LAYERS),
        "feature_dim": 1024,
        "storage_dtype": args.storage_dtype,
        "train_rows": len(train_ds),
        "dev_rows": len(dev_ds),
        "total_rows": len(index_rows),
        "total_semantic_frames": total_frames,
        "estimated_feature_bytes": estimated_bytes,
        "estimated_feature_gib": estimated_bytes / (1024**3),
        "padding_policy": "save semantic frames only; padding discarded before cache",
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("WAVLM LAYERS 13..17 CACHE COMPLETE")
    print(f"rows:                 {len(index_rows)}")
    print(f"semantic frames:      {total_frames}")
    print(f"storage dtype:        {args.storage_dtype}")
    print(f"estimated features:   {manifest['estimated_feature_gib']:.3f} GiB")
    print("generic_test accessed: NO")
    print("CACHE STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
