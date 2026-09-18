#!/usr/bin/env python
"""Materialize the first real P3 offline feature cache.

MDSC-Core30 train+dev only
facebook/wav2vec2-base
exact pinned revision
selected outputs.hidden_states layer(s)
safe policy: exact semantic waveform length groups, no padding
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch
from transformers import Wav2Vec2Model

from papr_ssl.cache.ssl_feature_cache import (
    CacheSample,
    MeanFeatureCacheWriter,
    cache_identities,
    fanout_multilayer_masked_mean,
    semantic_waveform_hash,
)
from papr_ssl.training.frozen_backbone import freeze_ssl_backbone
from papr_ssl.training.teacher.real_experiment import (
    RealTaskRow,
    build_mdsc_audio_catalog,
    load_real_task_rows,
    qualified_utt_relpath,
    resolve_task_audio_path,
)


MODEL_ID = "facebook/wav2vec2-base"
REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
SAMPLE_RATE = 16000


def parse_layers(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("--layers must contain at least one integer")
    if len(set(values)) != len(values):
        raise ValueError("--layers contains duplicates")
    if any(x < 0 or x > 12 for x in values):
        raise ValueError("wav2vec2-base hidden_states index must be in 0..12")
    return values


def load_waveform(path: Path) -> torch.Tensor:
    data, sr = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    if int(sr) != SAMPLE_RATE:
        raise RuntimeError(f"expected 16 kHz, got {sr}: {path}")
    waveform = torch.from_numpy(data).to(torch.float32)
    # [T,C] -> mono [T]
    waveform = waveform.mean(dim=1)
    if waveform.numel() <= 0:
        raise RuntimeError(f"empty waveform: {path}")
    if not torch.isfinite(waveform).all():
        raise RuntimeError(f"non-finite waveform: {path}")
    return waveform.contiguous()


def group_rows_by_num_frames(
    rows: list[RealTaskRow],
    mdsc_root: Path,
    audio_catalog: dict[str, Path] | None,
) -> dict[int, list[RealTaskRow]]:
    groups: dict[int, list[RealTaskRow]] = defaultdict(list)
    for i, row in enumerate(rows, start=1):
        path = resolve_task_audio_path(
            row=row,
            mdsc_root=mdsc_root,
            audio_catalog=audio_catalog,
        )
        info = sf.info(path)
        if int(info.samplerate) != SAMPLE_RATE:
            raise RuntimeError(f"expected 16 kHz: {path}")
        groups[int(info.frames)].append(row)
        if i % 500 == 0:
            print(f"[metadata] {i}/{len(rows)}")
    return groups


@torch.inference_mode()
def forward_exact_length_chunk(
    *,
    model: Wav2Vec2Model,
    waveforms: list[torch.Tensor],
    layers: tuple[int, ...],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    lengths = {int(x.numel()) for x in waveforms}
    if len(lengths) != 1:
        raise RuntimeError(
            "safe Wav2Vec2 cache forward requires exact equal waveform lengths"
        )

    batch = torch.stack(waveforms, dim=0).to(
        device=device,
        dtype=torch.float32,
    )
    outputs = model(
        input_values=batch,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Wav2Vec2 did not return hidden_states")

    selected: dict[int, torch.Tensor] = {}
    for layer in layers:
        x = hidden_states[layer]
        selected[layer] = x.to(torch.float32)

    t = selected[layers[0]].shape[1]
    if any(x.shape[1] != t for x in selected.values()):
        raise RuntimeError("selected hidden-state layers have different T")
    frame_mask = torch.ones(
        (batch.shape[0], t),
        dtype=torch.bool,
        device=device,
    )
    return selected, frame_mask


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"
        ),
    )
    p.add_argument(
        "--mdsc-root",
        type=Path,
        default=Path("datasets/public/mdsc"),
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/ssl_feature_cache"),
    )
    p.add_argument("--layers", default="12")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    layers = parse_layers(args.layers)
    rows = load_real_task_rows(
        args.task_index,
        splits=("train", "dev"),
    )

    print("=" * 100)
    print("PAPR-SSL REAL CACHE: MDSC-Core30 + Wav2Vec2-base")
    print("=" * 100)
    print(f"rows (train+dev): {len(rows)}")
    print(f"layers:           {layers}")
    print(f"revision:         {REVISION}")
    print(f"device:           {args.device}")
    print("safe batching:    exact semantic waveform length; no padding")
    print("test audio:       NOT MATERIALIZED")

    identities = cache_identities(
        model_id=MODEL_ID,
        model_revision=REVISION,
        layers=layers,
    )

    # Refuse overwrite.
    for identity in identities:
        target = args.cache_root / identity.slug
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(
                f"cache directory is non-empty: {target}"
            )

    writers = {
        identity.layer: MeanFeatureCacheWriter(
            args.cache_root,
            identity,
            shard_size=args.shard_size,
        )
        for identity in identities
    }

    legacy_rows = [
        row for row in rows
        if row.audio_relpath is None and qualified_utt_relpath(row) is None
    ]
    if legacy_rows:
        print("[1/4] Building legacy MDSC stem catalog...")
        audio_catalog = build_mdsc_audio_catalog(args.mdsc_root)
        print(f"legacy catalog entries: {len(audio_catalog)}")
    else:
        print("[1/4] Canonical qualified utt_id detected; catalog fallback not needed.")
        audio_catalog = None

    print("[2/4] Reading WAV headers and building exact-length groups...")
    groups = group_rows_by_num_frames(rows, args.mdsc_root, audio_catalog)
    print(f"unique waveform lengths: {len(groups)}")

    print("[3/4] Loading pinned Wav2Vec2 backbone...")
    device = torch.device(args.device)
    model = Wav2Vec2Model.from_pretrained(
        MODEL_ID,
        revision=REVISION,
    )
    freeze_ssl_backbone(model)
    model.to(device)

    print("[4/4] Materializing masked-mean cache...")
    done = 0
    total = len(rows)

    for num_frames in sorted(groups):
        group = groups[num_frames]
        for start in range(0, len(group), args.batch_size):
            chunk = group[start : start + args.batch_size]

            waveforms: list[torch.Tensor] = []
            samples: list[CacheSample] = []
            for row in chunk:
                path = resolve_task_audio_path(
                    row=row,
                    mdsc_root=args.mdsc_root,
                    audio_catalog=audio_catalog,
                )
                waveform = load_waveform(path)
                if int(waveform.numel()) != num_frames:
                    raise RuntimeError(
                        f"WAV frame-count changed between header/read: {path}"
                    )
                sample_hash = semantic_waveform_hash(
                    waveform,
                    sample_rate_hz=SAMPLE_RATE,
                    valid_num_samples=int(waveform.numel()),
                )
                waveforms.append(waveform)
                samples.append(
                    CacheSample(
                        utt_id=row.utt_id,
                        dataset=row.dataset,
                        split=row.split,
                        sample_hash=sample_hash,
                        frame_count=1,  # replaced by fanout
                    )
                )

            hidden, frame_mask = forward_exact_length_chunk(
                model=model,
                waveforms=waveforms,
                layers=layers,
                device=device,
            )
            fanout_multilayer_masked_mean(
                hidden_states_by_layer=hidden,
                frame_mask=frame_mask,
                samples=samples,
                writers=writers,
            )

            done += len(chunk)
            if done % 100 < len(chunk) or done == total:
                print(f"[cache] {done}/{total}")

    manifests = {
        layer: writer.finalize()
        for layer, writer in writers.items()
    }

    print("-" * 100)
    for layer, manifest in manifests.items():
        print(f"layer {layer:2d}: {manifest}")
    print("REAL CACHE MATERIALIZATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
