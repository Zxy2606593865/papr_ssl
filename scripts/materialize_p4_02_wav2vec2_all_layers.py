#!/usr/bin/env python
"""P4-02: materialize all Wav2Vec2 hidden-state caches in one SSL pass.

Current P3 already materialized layer 12.  This script validates and reuses a
complete existing cache, and materializes only missing indices.  It never
silently overwrites a non-empty incomplete cache directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import Wav2Vec2Model

from papr_ssl.cache.ssl_feature_cache import (
    CACHE_SCHEMA,
    CacheSample,
    MeanFeatureCacheWriter,
    SSLCacheIdentity,
    fanout_multilayer_masked_mean,
    masked_mean_pool,
    semantic_waveform_hash,
)
from papr_ssl.training.frozen_backbone import freeze_ssl_backbone
from papr_ssl.training.teacher.p4_candidates import get_p4_backbone
from papr_ssl.training.teacher.real_experiment import (
    RealTaskRow,
    build_mdsc_audio_catalog,
    load_real_task_rows,
    qualified_utt_relpath,
    resolve_task_audio_path,
)


SAMPLE_RATE = 16000
EXPECTED_SAMPLE_COUNT = 4198


def load_waveform(path: Path) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if int(sr) != SAMPLE_RATE:
        raise RuntimeError(f"expected 16 kHz, got {sr}: {path}")
    waveform = torch.from_numpy(data).to(torch.float32).mean(dim=1)
    if waveform.numel() <= 0:
        raise RuntimeError(f"empty waveform: {path}")
    if not torch.isfinite(waveform).all():
        raise RuntimeError(f"non-finite waveform: {path}")
    return waveform.contiguous()


def cache_status(
    *,
    cache_root: Path,
    identity: SSLCacheIdentity,
) -> str:
    """Return missing / complete, or fail on a partial/stale directory."""
    directory = cache_root / identity.slug
    if not directory.exists():
        return "missing"
    entries = list(directory.iterdir())
    if not entries:
        return "missing"

    manifest_path = directory / "manifest.json"
    index_path = directory / "index.jsonl"
    if not manifest_path.is_file() or not index_path.is_file():
        raise RuntimeError(
            f"non-empty incomplete cache directory; refusing overwrite: {directory}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_identity = {
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "layer": identity.layer,
        "pooling": identity.pooling,
        "feature_dtype": identity.feature_dtype,
    }
    if manifest.get("schema") != CACHE_SCHEMA:
        raise RuntimeError(f"cache schema mismatch: {directory}")
    if manifest.get("identity") != expected_identity:
        raise RuntimeError(f"cache identity mismatch: {directory}")
    if int(manifest.get("hidden_dim", -1)) != 768:
        raise RuntimeError(f"cache hidden_dim mismatch: {directory}")
    if int(manifest.get("sample_count", -1)) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"cache sample_count mismatch: {directory}: "
            f"{manifest.get('sample_count')} != {EXPECTED_SAMPLE_COUNT}"
        )
    index_lines = sum(
        1
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if index_lines != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"cache index row count mismatch: {directory}: "
            f"{index_lines} != {EXPECTED_SAMPLE_COUNT}"
        )
    return "complete"


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
def forward_missing_layers(
    *,
    model: Wav2Vec2Model,
    waveforms: list[torch.Tensor],
    layers: tuple[int, ...],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    if len({int(x.numel()) for x in waveforms}) != 1:
        raise RuntimeError("exact-length Wav2Vec2 batch contract violated")

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
    if hidden_states is None or len(hidden_states) != 13:
        raise RuntimeError(
            f"expected 13 hidden states, got "
            f"{None if hidden_states is None else len(hidden_states)}"
        )

    selected = {
        layer: hidden_states[layer].to(torch.float32)
        for layer in layers
    }
    t = selected[layers[0]].shape[1]
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
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    spec = get_p4_backbone("wav2vec2_base")
    all_layers = spec.valid_hidden_state_indices
    identities = {
        layer: SSLCacheIdentity(
            model_id=spec.model_id,
            model_revision=spec.model_revision,
            layer=layer,
        )
        for layer in all_layers
    }

    status = {
        layer: cache_status(
            cache_root=args.cache_root,
            identity=identity,
        )
        for layer, identity in identities.items()
    }
    missing_layers = tuple(
        layer for layer in all_layers if status[layer] == "missing"
    )
    complete_layers = tuple(
        layer for layer in all_layers if status[layer] == "complete"
    )

    print("=" * 104)
    print("PAPR-SSL P4-02 WAV2VEC2 FULL HIDDEN-STATE CACHE FANOUT")
    print("=" * 104)
    print(f"validated indices:  {all_layers[0]}..{all_layers[-1]}")
    print(f"existing complete:  {complete_layers}")
    print(f"to materialize:     {missing_layers}")
    print("test audio:          NOT MATERIALIZED")

    if not missing_layers:
        print("-" * 104)
        print("ALL WAV2VEC2 P4-02 CACHES ALREADY COMPLETE")
        print("P4-02 CACHE STATUS: PASS")
        return 0

    rows = load_real_task_rows(
        args.task_index,
        splits=("train", "dev"),
    )
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_SAMPLE_COUNT} train+dev rows, got {len(rows)}"
        )

    writers = {
        layer: MeanFeatureCacheWriter(
            args.cache_root,
            identities[layer],
            shard_size=args.shard_size,
        )
        for layer in missing_layers
    }

    legacy_rows = [
        row for row in rows
        if row.audio_relpath is None and qualified_utt_relpath(row) is None
    ]
    audio_catalog = (
        build_mdsc_audio_catalog(args.mdsc_root)
        if legacy_rows
        else None
    )

    print("[1/3] Reading WAV headers and building exact-length groups...")
    groups = group_rows_by_num_frames(
        rows,
        args.mdsc_root,
        audio_catalog,
    )
    print(f"unique waveform lengths: {len(groups)}")

    print("[2/3] Loading pinned Wav2Vec2 backbone once...")
    device = torch.device(args.device)
    model = Wav2Vec2Model.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
    )
    freeze_ssl_backbone(model)
    model.to(device)

    print("[3/3] One-forward multi-layer masked-mean fanout...")
    done = 0
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
                        f"WAV header/read length mismatch: {path}"
                    )
                waveforms.append(waveform)
                samples.append(
                    CacheSample(
                        utt_id=row.utt_id,
                        dataset=row.dataset,
                        split=row.split,
                        sample_hash=semantic_waveform_hash(
                            waveform,
                            sample_rate_hz=SAMPLE_RATE,
                            valid_num_samples=int(waveform.numel()),
                        ),
                        frame_count=1,
                    )
                )

            hidden, frame_mask = forward_missing_layers(
                model=model,
                waveforms=waveforms,
                layers=missing_layers,
                device=device,
            )
            fanout_multilayer_masked_mean(
                hidden_states_by_layer=hidden,
                frame_mask=frame_mask,
                samples=samples,
                writers=writers,
            )
            done += len(chunk)
            if done % 100 < len(chunk) or done == len(rows):
                print(f"[cache] {done}/{len(rows)}")

    manifests = {
        layer: writer.finalize()
        for layer, writer in writers.items()
    }

    # Final full-coverage audit including reused layer12.
    for layer, identity in identities.items():
        final_status = cache_status(
            cache_root=args.cache_root,
            identity=identity,
        )
        if final_status != "complete":
            raise RuntimeError(f"layer {layer} cache did not complete")

    print("-" * 104)
    for layer in all_layers:
        tag = "reused" if layer in complete_layers else "created"
        print(f"layer {layer:2d}: COMPLETE ({tag})")
    print("-" * 104)
    print("P4-02 WAV2VEC2 CACHE FANOUT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
