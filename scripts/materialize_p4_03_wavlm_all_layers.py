#!/usr/bin/env python
"""P4-03: materialize all WavLM-large hidden-state caches in one SSL pass.

Frozen P2 safe policy:
    WavLM -> native right-padded batch + attention_mask

The script length-sorts train+dev rows before forming native padded batches to
reduce wasted padding while preserving the already validated WavLM batching
semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import WavLMModel

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
    resolved.sort(key=lambda x: (x[0], x[1].utt_id))
    return resolved


@torch.inference_mode()
def forward_native_padded(
    *,
    model: WavLMModel,
    waveforms: list[torch.Tensor],
    layers: tuple[int, ...],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    lengths = torch.tensor(
        [int(x.numel()) for x in waveforms],
        dtype=torch.int64,
    )
    padded = pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0,
    )
    sample_mask = (
        torch.arange(
            padded.shape[1],
            dtype=torch.int64,
        ).unsqueeze(0)
        < lengths.unsqueeze(1)
    )

    padded = padded.to(device=device, dtype=torch.float32)
    sample_mask = sample_mask.to(device=device, dtype=torch.bool)

    output = model(
        input_values=padded,
        attention_mask=sample_mask.to(torch.long),
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("WavLM did not return hidden_states")
    if len(hidden_states) != EXPECTED_HIDDEN_STATE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_HIDDEN_STATE_COUNT} hidden states, "
            f"got {len(hidden_states)}"
        )

    selected = {
        layer: hidden_states[layer].to(torch.float32)
        for layer in layers
    }
    frame_count = selected[layers[0]].shape[1]
    if any(x.shape[1] != frame_count for x in selected.values()):
        raise RuntimeError("WavLM selected hidden states disagree on frame count")

    converter = getattr(
        model,
        "_get_feature_vector_attention_mask",
        None,
    )
    if converter is None:
        raise RuntimeError(
            "WavLM model cannot convert raw attention mask to frame mask"
        )
    try:
        frame_mask = converter(
            frame_count,
            sample_mask,
            add_adapter=False,
        )
    except TypeError:
        frame_mask = converter(frame_count, sample_mask)
    frame_mask = frame_mask.to(
        device=device,
        dtype=torch.bool,
    )
    if not bool(frame_mask.any(dim=1).all()):
        raise RuntimeError("WavLM produced a zero-valid-frame sample")

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
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Reduce to 1 if WavLM-large runs out of GPU memory.",
    )
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    spec = get_p4_backbone("wavlm_large")
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

    print("=" * 104)
    print("PAPR-SSL P4-03 WAVLM-LARGE FULL HIDDEN-STATE CACHE FANOUT")
    print("=" * 104)
    print(f"validated indices:   {layers[0]}..{layers[-1]}")
    print(f"existing complete:   {complete_layers}")
    print(f"to materialize:      {missing_layers}")
    print("safe batching:       native padded batch + attention_mask")
    print("batch order:         length-sorted to reduce padding")
    print("test audio:          NOT MATERIALIZED")

    if not missing_layers:
        print("-" * 104)
        print("ALL WAVLM P4-03 CACHES ALREADY COMPLETE")
        print("P4-03 CACHE STATUS: PASS")
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

    print("[1/3] Resolving WAV headers and sorting by length...")
    resolved = resolve_rows_and_lengths(
        rows=rows,
        mdsc_root=args.mdsc_root,
        audio_catalog=audio_catalog,
    )
    print(
        "length range:        "
        f"{resolved[0][0]}..{resolved[-1][0]} samples"
    )

    print("[2/3] Loading pinned WavLM-large backbone once...")
    device = torch.device(args.device)
    model = WavLMModel.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
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

    print("[3/3] Native padded one-forward multi-layer fanout...")
    done = 0
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

        hidden, frame_mask = forward_native_padded(
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
        if done % 100 < len(chunk) or done == len(resolved):
            print(f"[cache] {done}/{len(resolved)}")

    for writer in writers.values():
        writer.finalize()

    # Full 25-layer post-audit.
    for layer, identity in identities.items():
        if cache_status(
            cache_root=args.cache_root,
            identity=identity,
        ) != "complete":
            raise RuntimeError(
                f"WavLM hidden_states[{layer}] cache did not complete"
            )

    print("-" * 104)
    for layer in layers:
        tag = "reused" if layer in complete_layers else "created"
        print(f"layer {layer:2d}: COMPLETE ({tag})")
    print("-" * 104)
    print("P4-03 WAVLM CACHE FANOUT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
