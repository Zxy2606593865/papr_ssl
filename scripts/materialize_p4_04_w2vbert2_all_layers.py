#!/usr/bin/env python
"""P4-04: materialize all W2v-BERT 2.0 hidden-state caches in one SSL pass.

Frozen P2 safe policy:
    semantic waveform trim -> W2v-BERT feature extractor native batch
    -> input_features + attention_mask -> model

The task rows are resolved to their exact source waveforms first. No waveform
right-padding is fed into the acoustic feature extractor. The extractor itself
forms the native acoustic-feature batch and returns the authoritative frame mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

from papr_ssl.cache.ssl_feature_cache import (
    CACHE_SCHEMA,
    CacheSample,
    MeanFeatureCacheWriter,
    SSLCacheIdentity,
    fanout_multilayer_masked_mean,
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
EXPECTED_HIDDEN_STATE_COUNT = 25
EXPECTED_HIDDEN_DIM = 1024
EXPECTED_FRONTEND_DIM = 160


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
    if int(manifest.get("hidden_dim", -1)) != EXPECTED_HIDDEN_DIM:
        raise RuntimeError(f"cache hidden_dim mismatch: {directory}")
    if int(manifest.get("sample_count", -1)) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"cache sample_count mismatch: {directory}: "
            f"{manifest.get('sample_count')} != {EXPECTED_SAMPLE_COUNT}"
        )
    index_rows = sum(
        1
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if index_rows != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"cache index row count mismatch: {directory}: "
            f"{index_rows} != {EXPECTED_SAMPLE_COUNT}"
        )
    return "complete"


def resolve_rows_and_lengths(
    *,
    rows: list[RealTaskRow],
    mdsc_root: Path,
    audio_catalog: dict[str, Path] | None,
) -> list[tuple[int, RealTaskRow, Path]]:
    resolved: list[tuple[int, RealTaskRow, Path]] = []
    for i, row in enumerate(rows, start=1):
        path = resolve_task_audio_path(
            row=row,
            mdsc_root=mdsc_root,
            audio_catalog=audio_catalog,
        )
        info = sf.info(path)
        if int(info.samplerate) != SAMPLE_RATE:
            raise RuntimeError(f"expected 16 kHz: {path}")
        resolved.append((int(info.frames), row, path))
        if i % 500 == 0:
            print(f"[metadata] {i}/{len(rows)}")
    # Throughput/memory optimization only. Exact semantic waveform boundaries
    # are preserved before the feature extractor sees each sample.
    resolved.sort(key=lambda x: (x[0], x[1].utt_id))
    return resolved


@torch.inference_mode()
def forward_native_frontend_batch(
    *,
    model: Wav2Vec2BertModel,
    feature_extractor,
    waveforms: list[torch.Tensor],
    layers: tuple[int, ...],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, tuple[int, int, int]]:
    raw_speech = [
        waveform.detach().cpu().numpy()
        for waveform in waveforms
    ]
    batch = feature_extractor(
        raw_speech,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    input_features = batch.get("input_features")
    attention_mask = batch.get("attention_mask")

    if not isinstance(input_features, torch.Tensor):
        raise RuntimeError("feature extractor did not return input_features")
    if input_features.ndim != 3:
        raise RuntimeError(
            f"expected input_features [B,T,F], got {tuple(input_features.shape)}"
        )
    if int(input_features.shape[-1]) != EXPECTED_FRONTEND_DIM:
        raise RuntimeError(
            f"expected frontend dim={EXPECTED_FRONTEND_DIM}, "
            f"got {input_features.shape[-1]}"
        )
    if not isinstance(attention_mask, torch.Tensor):
        raise RuntimeError("feature extractor did not return attention_mask")
    if attention_mask.shape != input_features.shape[:2]:
        raise RuntimeError(
            "feature extractor attention_mask is not aligned to input_features"
        )
    if not bool(attention_mask.to(torch.bool).any(dim=1).all()):
        raise RuntimeError("feature extractor produced a zero-valid-frame sample")

    input_features = input_features.to(
        device=device,
        dtype=torch.float32,
    )
    frame_mask = attention_mask.to(
        device=device,
        dtype=torch.bool,
    )

    output = model(
        input_features=input_features,
        attention_mask=frame_mask.to(torch.long),
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("W2v-BERT 2.0 did not return hidden_states")
    if len(hidden_states) != EXPECTED_HIDDEN_STATE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_HIDDEN_STATE_COUNT} hidden states, "
            f"got {len(hidden_states)}"
        )

    selected = {
        layer: hidden_states[layer].to(torch.float32)
        for layer in layers
    }
    model_frames = selected[layers[0]].shape[1]
    if model_frames != frame_mask.shape[1]:
        raise RuntimeError(
            "W2v-BERT model hidden-state time axis does not match "
            "feature-extractor attention_mask"
        )
    if any(x.shape[1] != model_frames for x in selected.values()):
        raise RuntimeError(
            "selected W2v-BERT hidden states disagree on time dimension"
        )

    frontend_shape = (
        int(input_features.shape[0]),
        int(input_features.shape[1]),
        int(input_features.shape[2]),
    )
    return selected, frame_mask, frontend_shape


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
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Correctness-first default for W2v-BERT 2.0 on laptop GPUs. "
            "Use 2 only after confirming enough CUDA memory."
        ),
    )
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    spec = get_p4_backbone("w2v_bert2")
    layers = spec.valid_hidden_state_indices
    identities = {
        layer: SSLCacheIdentity(
            model_id=spec.model_id,
            model_revision=spec.model_revision,
            layer=layer,
        )
        for layer in layers
    }

    status = {
        layer: cache_status(
            cache_root=args.cache_root,
            identity=identity,
        )
        for layer, identity in identities.items()
    }
    missing_layers = tuple(
        x for x in layers if status[x] == "missing"
    )
    complete_layers = tuple(
        x for x in layers if status[x] == "complete"
    )

    print("=" * 108)
    print("PAPR-SSL P4-04 W2V-BERT 2.0 FULL HIDDEN-STATE CACHE FANOUT")
    print("=" * 108)
    print(f"validated indices:   {layers[0]}..{layers[-1]}")
    print(f"existing complete:   {complete_layers}")
    print(f"to materialize:      {missing_layers}")
    print("safe batching:       semantic trim -> native acoustic frontend batch")
    print("model inputs:        input_features + attention_mask")
    print("batch order:         length-sorted to reduce frontend padding")
    print("test audio:          NOT MATERIALIZED")

    if not missing_layers:
        print("-" * 108)
        print("ALL W2V-BERT 2.0 P4-04 CACHES ALREADY COMPLETE")
        print("P4-04 CACHE STATUS: PASS")
        return 0

    rows = load_real_task_rows(
        args.task_index,
        splits=("train", "dev"),
    )
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_SAMPLE_COUNT} train+dev rows, got {len(rows)}"
        )

    legacy_rows = [
        row
        for row in rows
        if row.audio_relpath is None
        and qualified_utt_relpath(row) is None
    ]
    audio_catalog = (
        build_mdsc_audio_catalog(args.mdsc_root)
        if legacy_rows
        else None
    )

    print("[1/3] Resolving exact semantic WAVs and sorting by length...")
    resolved = resolve_rows_and_lengths(
        rows=rows,
        mdsc_root=args.mdsc_root,
        audio_catalog=audio_catalog,
    )
    print(
        "waveform length range: "
        f"{resolved[0][0]}..{resolved[-1][0]} samples"
    )

    print("[2/3] Loading pinned W2v-BERT 2.0 frontend + backbone once...")
    device = torch.device(args.device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
    )
    model = Wav2Vec2BertModel.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
        output_hidden_states=True,
    )
    freeze_ssl_backbone(model)
    model.to(device)

    writers = {
        layer: MeanFeatureCacheWriter(
            args.cache_root,
            identities[layer],
            shard_size=args.shard_size,
        )
        for layer in missing_layers
    }

    print("[3/3] Native frontend + one-forward multi-layer fanout...")
    done = 0
    first_frontend_shape = None

    for start in range(0, len(resolved), args.batch_size):
        chunk = resolved[start : start + args.batch_size]

        waveforms: list[torch.Tensor] = []
        samples: list[CacheSample] = []

        for expected_frames, row, path in chunk:
            waveform = load_waveform(path)
            if int(waveform.numel()) != expected_frames:
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

        hidden, frame_mask, frontend_shape = forward_native_frontend_batch(
            model=model,
            feature_extractor=feature_extractor,
            waveforms=waveforms,
            layers=missing_layers,
            device=device,
        )
        if first_frontend_shape is None:
            first_frontend_shape = frontend_shape
            print(
                "first frontend batch: "
                f"{first_frontend_shape}  [B,T,160]"
            )

        fanout_multilayer_masked_mean(
            hidden_states_by_layer=hidden,
            frame_mask=frame_mask,
            samples=samples,
            writers=writers,
        )

        done += len(chunk)
        if done % 100 < len(chunk) or done == len(resolved):
            print(f"[cache] {done}/{len(resolved)}")

    for writer in writers.values():
        writer.finalize()

    for layer, identity in identities.items():
        if cache_status(
            cache_root=args.cache_root,
            identity=identity,
        ) != "complete":
            raise RuntimeError(
                f"W2v-BERT hidden_states[{layer}] cache did not complete"
            )

    print("-" * 108)
    for layer in layers:
        tag = "reused" if layer in complete_layers else "created"
        print(f"layer {layer:2d}: COMPLETE ({tag})")
    print("-" * 108)
    print("P4-04 W2V-BERT 2.0 CACHE FANOUT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
