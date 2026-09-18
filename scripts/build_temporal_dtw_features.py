#!/usr/bin/env python
"""Build minimal temporal features for DTW using the SAME 256D Teacher projection.

This is intentionally NOT a second encoder.

Pipeline:
    WavLM hidden_states[15] [T,1024]
      -> reuse trained 256D Teacher projection frame-by-frame
      -> temporal mean-pool every N frames
      -> framewise L2 normalization
      -> compact temporal sequence [T',256]

Only these utterances are built:
- 30 x 15 enrollment utterances already used by the 256D personalized experiment
- 442 Core30 known DEV utterances
- 1178 open-set DEV utterances

generic_test is never loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel

from papr_ssl.training.teacher.p5_dr import build_dr
from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset, frame_collate
from papr_ssl.training.teacher.real_experiment import (
    build_mdsc_audio_catalog,
    load_real_task_rows,
    qualified_utt_relpath,
    resolve_task_audio_path,
)

MODEL_ID = "microsoft/wavlm-large"
MODEL_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
HIDDEN_STATE_INDEX = 15


def safe_name(utt_id: str) -> str:
    return hashlib.sha1(utt_id.encode("utf-8")).hexdigest() + ".pt"


def replace_projection(model: nn.Module, dim: int = 256) -> nn.Module:
    model = copy.deepcopy(model)
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and m.out_features == 64]
    if len(cands) != 1:
        raise RuntimeError(
            f"Expected exactly one 64-output Linear, got "
            f"{[(n,m.in_features,m.out_features) for n,m in cands]}"
        )
    name, old = cands[0]
    model.set_submodule(
        name,
        nn.Linear(old.in_features, dim, bias=(old.bias is not None))
    )
    return model


def load_projection(checkpoint: Path, device: torch.device):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(obj.get("embedding_dim", -1)) != 256:
        raise RuntimeError(f"Expected 256D Teacher checkpoint, got {obj.get('embedding_dim')}")

    dr = replace_projection(build_dr("attention"), 256)
    dr.load_state_dict(obj["dr_state_dict"], strict=True)
    dr.eval().to(device)

    cands = [
        (name, module)
        for name, module in dr.named_modules()
        if isinstance(module, nn.Linear)
        and module.in_features == 1024
        and module.out_features == 256
    ]
    if len(cands) != 1:
        raise RuntimeError(
            f"Expected one 1024->256 Teacher projection, got "
            f"{[(n,m.in_features,m.out_features) for n,m in cands]}"
        )
    name, proj = cands[0]
    proj.eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj, name


def temporal_project(h: torch.Tensor, projection: nn.Linear, downsample: int):
    """h [T,1024] -> [T',256]."""
    with torch.inference_mode():
        z = projection(h.to(torch.float32))
        z = F.normalize(z, dim=-1)

        chunks = []
        for start in range(0, z.shape[0], downsample):
            chunk = z[start:start + downsample]
            pooled = chunk.mean(dim=0, keepdim=True)
            chunks.append(F.normalize(pooled, dim=-1))
        return torch.cat(chunks, dim=0).to(torch.float16).cpu().contiguous()


def cached_hidden_for_utt(dataset, row_index, device):
    batch = frame_collate([dataset[row_index]])
    x = batch["features"].to(device, non_blocking=True)
    mask = batch["frame_mask"].to(device, non_blocking=True)
    valid = mask[0]
    h = x[0, valid].to(torch.float32)
    return h


def load_wave(path: Path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if int(sr) != 16000:
        raise RuntimeError(f"Expected 16kHz, got {sr}: {path}")
    wav = torch.from_numpy(data).float().mean(dim=1)
    if wav.numel() == 0 or not torch.isfinite(wav).all():
        raise RuntimeError(f"Invalid audio: {path}")
    return wav.contiguous()


def resolve_selected_checkpoint(embedding_manifest: Path) -> Path:
    m = json.loads(embedding_manifest.read_text(encoding="utf-8"))
    if m.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("256D embedding manifest test seal violated")
    selected = m["selected_checkpoint"]
    return Path(selected["checkpoint"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embedding-npz",
        type=Path,
        default=Path(
            "artifacts/p6_teacher_256_15shot/embeddings/"
            "teacher_256d_15shot_embeddings.npz"
        ),
    )
    p.add_argument(
        "--embedding-manifest",
        type=Path,
        default=Path(
            "artifacts/p6_teacher_256_15shot/embeddings/manifest.json"
        ),
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
    p.add_argument(
        "--open-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/"
            "mdsc_core30_open_set_eval.index.jsonl"
        ),
    )
    p.add_argument("--mdsc-root", type=Path, default=Path("datasets/public/mdsc"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_temporal_dtw/features"),
    )
    p.add_argument("--downsample", type=int, default=3)
    p.add_argument("--open-batch-size", type=int, default=8)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "utterances"
    data_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.embedding_npz, allow_pickle=False)
    enrollment_ids = d["enrollment_utt_id"].astype(str).reshape(-1)
    known_ids = d["known_utt_id"].astype(str)
    unknown_ids = d["unknown_utt_id"].astype(str)

    if len(enrollment_ids) != 450 or len(known_ids) != 442 or len(unknown_ids) != 1178:
        raise RuntimeError(
            f"Unexpected target counts enrollment={len(enrollment_ids)} "
            f"known={len(known_ids)} unknown={len(unknown_ids)}"
        )

    device = torch.device(args.device)
    checkpoint = resolve_selected_checkpoint(args.embedding_manifest)
    projection, projection_name = load_projection(checkpoint, device)

    train_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="train",
    )
    dev_ds = FrameCacheDataset(
        cache_dir=args.frame_cache,
        task_index=args.core_index,
        split="dev",
        class_to_index=train_ds.class_to_index,
    )

    train_lookup = {row["utt_id"]: i for i, row in enumerate(train_ds.rows)}
    dev_lookup = {row["utt_id"]: i for i, row in enumerate(dev_ds.rows)}

    index_rows = []

    def save_item(utt_id: str, role: str, seq: torch.Tensor):
        rel = Path("utterances") / safe_name(utt_id)
        torch.save(
            {
                "schema": "papr_ssl.temporal_teacher_sequence.v1",
                "utt_id": utt_id,
                "role": role,
                "features": seq,
                "shape": list(seq.shape),
            },
            args.output_dir / rel,
        )
        index_rows.append(
            {
                "utt_id": utt_id,
                "role": role,
                "cache_relpath": str(rel).replace("\\", "/"),
                "frames": int(seq.shape[0]),
                "dim": int(seq.shape[1]),
            }
        )

    print("[temporal] building 450 enrollment sequences from existing WavLM[15] cache...")
    for j, utt_id in enumerate(enrollment_ids, 1):
        i = train_lookup.get(utt_id)
        if i is None:
            raise RuntimeError(f"Enrollment utt missing from train frame cache: {utt_id}")
        h = cached_hidden_for_utt(train_ds, i, device)
        seq = temporal_project(h, projection, args.downsample)
        save_item(utt_id, "enrollment", seq)
        if j % 50 == 0 or j == len(enrollment_ids):
            print(f"[enrollment] {j}/{len(enrollment_ids)}")

    print("[temporal] building 442 known DEV sequences from existing WavLM[15] cache...")
    for j, utt_id in enumerate(known_ids, 1):
        i = dev_lookup.get(utt_id)
        if i is None:
            raise RuntimeError(f"Known DEV utt missing from frame cache: {utt_id}")
        h = cached_hidden_for_utt(dev_ds, i, device)
        seq = temporal_project(h, projection, args.downsample)
        save_item(utt_id, "known_dev", seq)
        if j % 50 == 0 or j == len(known_ids):
            print(f"[known DEV] {j}/{len(known_ids)}")

    print("[temporal] building 1178 open DEV sequences with frozen WavLM[15]...")
    open_rows = load_real_task_rows(args.open_index, splits=("dev",))
    if len(open_rows) != 1178:
        raise RuntimeError(f"Expected 1178 open DEV rows, got {len(open_rows)}")
    open_by_utt = {r.utt_id: r for r in open_rows}
    if set(unknown_ids) != set(open_by_utt):
        raise RuntimeError("Open DEV utt_id set mismatch vs existing 256D embedding NPZ")

    legacy = [
        r for r in open_rows
        if r.audio_relpath is None and qualified_utt_relpath(r) is None
    ]
    catalog = build_mdsc_audio_catalog(args.mdsc_root) if legacy else None

    backbone = WavLMModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        output_hidden_states=True,
        local_files_only=True,
    )
    backbone.eval().to(device)
    for param in backbone.parameters():
        param.requires_grad_(False)

    ids_ordered = list(unknown_ids)
    for start in range(0, len(ids_ordered), args.open_batch_size):
        batch_ids = ids_ordered[start:start + args.open_batch_size]
        waves = []
        for utt_id in batch_ids:
            row = open_by_utt[utt_id]
            path = resolve_task_audio_path(
                row=row,
                mdsc_root=args.mdsc_root,
                audio_catalog=catalog,
            )
            waves.append(load_wave(path))

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
            out = backbone(
                input_values=x,
                attention_mask=am,
                output_hidden_states=True,
                return_dict=True,
            )
            h = out.hidden_states[HIDDEN_STATE_INDEX].to(torch.float32)
            fm = backbone._get_feature_vector_attention_mask(
                h.shape[1], am
            ).to(torch.bool)

        for i, utt_id in enumerate(batch_ids):
            valid_h = h[i, fm[i]]
            seq = temporal_project(valid_h, projection, args.downsample)
            save_item(utt_id, "unknown_dev", seq)

        done = min(start + len(batch_ids), len(ids_ordered))
        print(f"[open DEV] {done}/{len(ids_ordered)}")

    with (args.output_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = np.asarray([r["frames"] for r in index_rows], dtype=np.int64)
    manifest = {
        "schema": "papr_ssl.temporal_dtw_features.v1",
        "source_teacher": "WavLM-large[15] + current trained 256D projection",
        "projection_checkpoint": str(checkpoint),
        "projection_module": projection_name,
        "temporal_branch": "reuse Teacher frame projection; no second encoder",
        "downsample_factor": args.downsample,
        "frame_dim": 256,
        "storage_dtype": "float16",
        "counts": {
            "enrollment": 450,
            "known_dev": 442,
            "unknown_dev": 1178,
        },
        "sequence_length": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "max": int(lengths.max()),
        },
        "canonical_64d_teacher": "preserved_not_modified",
        "project1_256d_teacher": "preserved_not_modified",
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("TEMPORAL DTW FEATURE BUILD COMPLETE")
    print(f"projection:            {projection_name}")
    print(f"downsample factor:     {args.downsample}")
    print(f"sequence median/p95:   {manifest['sequence_length']['median']:.1f} / "
          f"{manifest['sequence_length']['p95']:.1f}")
    print("second encoder added:  NO")
    print("generic_test accessed: NO")
    print("BUILD STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
