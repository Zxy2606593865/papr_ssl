#!/usr/bin/env python
"""P5 prerequisite: materialize frame-level WavLM-large hidden_states[15].

Frozen by P4-08:
- model_id: microsoft/wavlm-large
- revision: c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
- hidden_state_index: 15
- hidden_dim: 1024
- train/dev only; generic_test remains sealed

Storage format:
- float32 frame features
- variable-length sequences packed per shard as one [sum_T, 1024] tensor
- offsets identify each utterance inside the shard
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import WavLMModel

from papr_ssl.training.teacher.real_experiment import (
    RealTaskRow,
    build_mdsc_audio_catalog,
    load_real_task_rows,
    qualified_utt_relpath,
    resolve_task_audio_path,
)

SAMPLE_RATE = 16000
EXPECTED_TOTAL = 4198
EXPECTED_TRAIN = 3756
EXPECTED_DEV = 442
MODEL_ID = "microsoft/wavlm-large"
MODEL_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
HIDDEN_STATE_INDEX = 15
HIDDEN_DIM = 1024
SCHEMA = "papr_ssl.p5_frame_cache.v1"


def load_waveform(path: Path) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if int(sr) != SAMPLE_RATE:
        raise RuntimeError(f"expected 16 kHz, got {sr}: {path}")
    x = torch.from_numpy(data).to(torch.float32).mean(dim=1)
    if x.numel() <= 0:
        raise RuntimeError(f"empty waveform: {path}")
    if not torch.isfinite(x).all():
        raise RuntimeError(f"non-finite waveform: {path}")
    return x.contiguous()


def waveform_hash(x: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(f"sr={SAMPLE_RATE};n={x.numel()};".encode("utf-8"))
    h.update(x.detach().cpu().numpy().tobytes(order="C"))
    return h.hexdigest()


def resolve_rows(
    rows: list[RealTaskRow],
    mdsc_root: Path,
) -> list[tuple[int, RealTaskRow, Path]]:
    legacy = [
        row for row in rows
        if row.audio_relpath is None and qualified_utt_relpath(row) is None
    ]
    catalog = build_mdsc_audio_catalog(mdsc_root) if legacy else None

    out = []
    for i, row in enumerate(rows, start=1):
        path = resolve_task_audio_path(
            row=row,
            mdsc_root=mdsc_root,
            audio_catalog=catalog,
        )
        info = sf.info(path)
        if int(info.samplerate) != SAMPLE_RATE:
            raise RuntimeError(f"expected 16 kHz: {path}")
        out.append((int(info.frames), row, path))
        if i % 500 == 0:
            print(f"[metadata] {i}/{len(rows)}")
    out.sort(key=lambda item: (item[0], item[1].utt_id))
    return out


@torch.inference_mode()
def forward_batch(
    *,
    model: WavLMModel,
    waveforms: list[torch.Tensor],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[int]]:
    lengths = torch.tensor(
        [int(x.numel()) for x in waveforms],
        dtype=torch.long,
    )
    max_len = int(lengths.max().item())
    batch = torch.zeros(
        (len(waveforms), max_len),
        dtype=torch.float32,
    )
    attention_mask = torch.zeros(
        (len(waveforms), max_len),
        dtype=torch.long,
    )
    for i, x in enumerate(waveforms):
        n = int(x.numel())
        batch[i, :n] = x
        attention_mask[i, :n] = 1

    batch = batch.to(device)
    attention_mask = attention_mask.to(device)

    out = model(
        input_values=batch,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    if hs is None or len(hs) != 25:
        raise RuntimeError("unexpected WavLM hidden_states contract")

    features = hs[HIDDEN_STATE_INDEX].to(torch.float32)
    if int(features.shape[-1]) != HIDDEN_DIM:
        raise RuntimeError(
            f"hidden dim mismatch: {features.shape[-1]} != {HIDDEN_DIM}"
        )

    feature_mask = model._get_feature_vector_attention_mask(
        features.shape[1],
        attention_mask,
    ).to(torch.bool)

    result = []
    frame_counts = []
    for i in range(features.shape[0]):
        valid = int(feature_mask[i].sum().item())
        if valid <= 0:
            raise RuntimeError("zero valid WavLM frames")
        # Valid frames are packed to the left by the model-native mask.
        seq = features[i, :valid].detach().cpu().contiguous()
        if not torch.isfinite(seq).all():
            raise RuntimeError("non-finite cached frame feature")
        result.append(seq)
        frame_counts.append(valid)

    return result, frame_counts


class ShardWriter:
    def __init__(
        self,
        root: Path,
        *,
        shard_utterances: int,
    ):
        self.root = root
        self.shard_utterances = shard_utterances
        self.shard_id = 0
        self.pending_features: list[torch.Tensor] = []
        self.pending_meta: list[dict] = []
        self.index_rows: list[dict] = []

    def add(self, feature: torch.Tensor, meta: dict) -> None:
        self.pending_features.append(feature)
        self.pending_meta.append(meta)
        if len(self.pending_features) >= self.shard_utterances:
            self.flush()

    def flush(self) -> None:
        if not self.pending_features:
            return

        shard_name = f"shard_{self.shard_id:05d}.pt"
        shard_path = self.root / shard_name

        offsets = [0]
        for feat in self.pending_features:
            offsets.append(offsets[-1] + int(feat.shape[0]))

        packed = torch.cat(self.pending_features, dim=0).contiguous()
        if packed.ndim != 2 or int(packed.shape[1]) != HIDDEN_DIM:
            raise RuntimeError("invalid packed frame cache tensor")

        torch.save(
            {
                "schema": SCHEMA,
                "features": packed,
                "offsets": torch.tensor(offsets, dtype=torch.long),
                "utt_ids": [m["utt_id"] for m in self.pending_meta],
            },
            shard_path,
        )

        for i, meta in enumerate(self.pending_meta):
            row = dict(meta)
            row.update(
                {
                    "shard": shard_name,
                    "row_in_shard": i,
                    "start_frame": offsets[i],
                    "end_frame": offsets[i + 1],
                    "frame_count": offsets[i + 1] - offsets[i],
                    "hidden_dim": HIDDEN_DIM,
                    "feature_dtype": "float32",
                }
            )
            self.index_rows.append(row)

        self.shard_id += 1
        self.pending_features.clear()
        self.pending_meta.clear()

    def finalize(self) -> None:
        self.flush()
        with (self.root / "index.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for row in self.index_rows:
                f.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )


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
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/p5_frame_cache/wavlm_large_layer15"
        ),
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--shard-utterances", type=int, default=128)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"frame cache already exists; refusing overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_real_task_rows(
        args.task_index,
        splits=("train", "dev"),
    )
    split_counts = {
        "train": sum(row.split == "train" for row in rows),
        "dev": sum(row.split == "dev" for row in rows),
    }
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL} rows, got {len(rows)}"
        )
    if split_counts != {
        "train": EXPECTED_TRAIN,
        "dev": EXPECTED_DEV,
    }:
        raise RuntimeError(f"split count mismatch: {split_counts}")

    print("=" * 108)
    print("P5 FRAME-LEVEL CACHE: WAVLM-LARGE hidden_states[15]")
    print("=" * 108)
    print(f"model:              {MODEL_ID}")
    print(f"revision:           {MODEL_REVISION}")
    print(f"hidden_state_index: {HIDDEN_STATE_INDEX}")
    print(f"hidden_dim:         {HIDDEN_DIM}")
    print(f"train/dev:          {EXPECTED_TRAIN} / {EXPECTED_DEV}")
    print("generic_test:       NOT MATERIALIZED")
    print("storage dtype:      float32")

    resolved = resolve_rows(rows, args.mdsc_root)
    print(
        "waveform length range: "
        f"{resolved[0][0]}..{resolved[-1][0]} samples"
    )

    device = torch.device(args.device)
    model = WavLMModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        output_hidden_states=True,
        local_files_only=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)

    writer = ShardWriter(
        args.output_dir,
        shard_utterances=args.shard_utterances,
    )

    total_frames = 0
    done = 0
    for start in range(0, len(resolved), args.batch_size):
        chunk = resolved[start:start + args.batch_size]
        waveforms = []
        metas = []

        for expected_n, row, path in chunk:
            wav = load_waveform(path)
            if int(wav.numel()) != expected_n:
                raise RuntimeError(f"WAV length mismatch: {path}")
            waveforms.append(wav)
            metas.append(
                {
                    "utt_id": row.utt_id,
                    "dataset": row.dataset,
                    "split": row.split,
                    "sample_hash": waveform_hash(wav),
                }
            )

        frame_features, frame_counts = forward_batch(
            model=model,
            waveforms=waveforms,
            device=device,
        )

        for feat, frames, meta in zip(
            frame_features,
            frame_counts,
            metas,
        ):
            if int(feat.shape[0]) != frames:
                raise RuntimeError("frame count mismatch")
            writer.add(feat, meta)
            total_frames += frames

        done += len(chunk)
        if done % 100 < len(chunk) or done == len(resolved):
            print(
                f"[cache] {done}/{len(resolved)} "
                f"utterances, {total_frames} frames"
            )

    writer.finalize()

    manifest = {
        "schema": SCHEMA,
        "phase": "P5",
        "purpose": "frame_level_cache_for_mean_vs_attention_dr",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "hidden_state_index": HIDDEN_STATE_INDEX,
        "hidden_dim": HIDDEN_DIM,
        "feature_dtype": "float32",
        "sample_rate_hz": SAMPLE_RATE,
        "train_count": EXPECTED_TRAIN,
        "dev_count": EXPECTED_DEV,
        "sample_count": EXPECTED_TOTAL,
        "total_frame_count": total_frames,
        "batch_size": args.batch_size,
        "native_right_padding_with_attention_mask": True,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("-" * 108)
    print(f"sample_count:       {EXPECTED_TOTAL}")
    print(f"total_frame_count:  {total_frames}")
    print("generic_test:       SEALED / NOT ACCESSED")
    print("P5 FRAME CACHE MATERIALIZATION: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
