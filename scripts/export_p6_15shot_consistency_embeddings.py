#!/usr/bin/env python
"""Export 64D embeddings for 15-shot enrollment-consistency diagnostics.

Frozen:
- WavLM-large outputs.hidden_states[15]
- AttentionDR 64D representative checkpoint
- deterministic 15-shot enrollment used by previous P6 ablations
- generic_test remains sealed

Exports:
- enrollment embeddings [30, 15, 64]
- all known DEV embeddings [442, 64]
- all unknown DEV embeddings [1178, 64]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
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
SAMPLE_RATE = 16000
SHOT = 15
SALT = "papr_ssl_p6_protocol_v1"


def stable_key(*parts: str) -> str:
    h = hashlib.sha256()
    h.update(SALT.encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return h.hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_attention_dr(checkpoint: Path, device: torch.device):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if obj.get("method") != "attention":
        raise RuntimeError("expected AttentionDR checkpoint")
    dr = build_dr("attention")
    dr.load_state_dict(obj["dr_state_dict"], strict=True)
    dr.eval().to(device)
    for p in dr.parameters():
        p.requires_grad_(False)
    return dr


@torch.inference_mode()
def embed_dataset(dataset, dr, device):
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=frame_collate,
    )
    out = {}
    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        mask = batch["frame_mask"].to(device, non_blocking=True)
        z = dr(x, mask).cpu().to(torch.float32)
        for utt_id, emb in zip(batch["utt_ids"], z):
            out[utt_id] = emb.contiguous()
    return out


def load_wave(path: Path) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if int(sr) != SAMPLE_RATE:
        raise RuntimeError(f"expected 16 kHz: {path}")
    x = torch.from_numpy(data).to(torch.float32).mean(dim=1)
    if x.numel() <= 0 or not torch.isfinite(x).all():
        raise RuntimeError(f"invalid waveform: {path}")
    return x.contiguous()


@torch.inference_mode()
def embed_unknown_dev(
    *,
    open_index: Path,
    mdsc_root: Path,
    backbone: WavLMModel,
    dr,
    device: torch.device,
):
    rows = load_real_task_rows(open_index, splits=("dev",))
    if len(rows) != 1178:
        raise RuntimeError(f"expected 1178 open-set dev rows, got {len(rows)}")

    legacy = [
        r for r in rows
        if r.audio_relpath is None and qualified_utt_relpath(r) is None
    ]
    catalog = build_mdsc_audio_catalog(mdsc_root) if legacy else None

    out = {}
    for i, row in enumerate(rows, start=1):
        path = resolve_task_audio_path(
            row=row,
            mdsc_root=mdsc_root,
            audio_catalog=catalog,
        )
        wav = load_wave(path).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(wav, dtype=torch.long)
        model_out = backbone(
            input_values=wav,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = model_out.hidden_states[HIDDEN_STATE_INDEX].to(torch.float32)
        frame_mask = backbone._get_feature_vector_attention_mask(
            hs.shape[1], attention_mask
        ).to(torch.bool)
        z = dr(hs, frame_mask).squeeze(0).cpu().to(torch.float32)
        out[row.utt_id] = z.contiguous()
        if i % 100 == 0 or i == len(rows):
            print(f"[unknown dev] {i}/{len(rows)}")

    return rows, out


def build_15shot_rows(train_ds):
    groups = {}
    for row in train_ds.rows:
        groups.setdefault(row["label_text"], []).append(row["utt_id"])

    rows = []
    for label in sorted(groups):
        ordered = sorted(
            groups[label],
            key=lambda utt: stable_key("enrollment", label, utt),
        )
        if len(ordered) < SHOT:
            raise RuntimeError(f"class {label!r} has only {len(ordered)} samples")
        c = train_ds.class_to_index[label]
        for utt_id in ordered[:SHOT]:
            rows.append((c, label, utt_id))

    if len(rows) != 30 * SHOT:
        raise RuntimeError("expected exactly 450 enrollment utterances")
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("artifacts/p6_dev_k2/protocol"),
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
    p.add_argument(
        "--mdsc-root",
        type=Path,
        default=Path("datasets/public/mdsc"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p6_15shot_consistency/embeddings"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(
        (args.protocol_dir / "p6_k2_protocol.json").read_text(encoding="utf-8")
    )
    if protocol.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")

    meta = protocol["representative_checkpoints"]["attention"]
    checkpoint = Path(meta["checkpoint"])
    if sha256(checkpoint) != meta["checkpoint_sha256"]:
        raise RuntimeError("Attention checkpoint hash mismatch")

    device = torch.device(args.device)
    dr = load_attention_dr(checkpoint, device)

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
    if len(dev_ds) != 442:
        raise RuntimeError(f"expected 442 known DEV, got {len(dev_ds)}")

    print("[consistency] embedding known train/dev...")
    train_emb = embed_dataset(train_ds, dr, device)
    dev_emb = embed_dataset(dev_ds, dr, device)

    enrollment_rows = build_15shot_rows(train_ds)
    enrollment = np.zeros((30, SHOT, 64), dtype=np.float32)
    enrollment_utt_ids = np.empty((30, SHOT), dtype="<U256")
    class_names = np.empty((30,), dtype="<U256")
    cursor = [0] * 30

    for c, label, utt_id in enrollment_rows:
        j = cursor[c]
        enrollment[c, j] = train_emb[utt_id].numpy()
        enrollment_utt_ids[c, j] = utt_id
        class_names[c] = label
        cursor[c] += 1

    if cursor != [SHOT] * 30:
        raise RuntimeError(f"bad enrollment population: {cursor}")

    known_utt_id = np.asarray([r["utt_id"] for r in dev_ds.rows], dtype=str)
    known_true = np.asarray(
        [int(r["label_index"]) for r in dev_ds.rows], dtype=np.int64
    )
    known_embedding = np.stack(
        [dev_emb[u].numpy() for u in known_utt_id], axis=0
    ).astype(np.float32)

    print("[consistency] embedding all open-set DEV once...")
    backbone = WavLMModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        output_hidden_states=True,
        local_files_only=True,
    )
    backbone.eval().to(device)
    for param in backbone.parameters():
        param.requires_grad_(False)

    unknown_rows, unknown_emb = embed_unknown_dev(
        open_index=args.open_index,
        mdsc_root=args.mdsc_root,
        backbone=backbone,
        dr=dr,
        device=device,
    )
    unknown_utt_id = np.asarray([r.utt_id for r in unknown_rows], dtype=str)
    unknown_embedding = np.stack(
        [unknown_emb[u].numpy() for u in unknown_utt_id], axis=0
    ).astype(np.float32)

    np.savez_compressed(
        args.output_dir / "attention_15shot_consistency_embeddings.npz",
        enrollment=enrollment,
        enrollment_utt_id=enrollment_utt_ids,
        class_name=class_names,
        known_utt_id=known_utt_id,
        known_true=known_true,
        known_embedding=known_embedding,
        unknown_utt_id=unknown_utt_id,
        unknown_embedding=unknown_embedding,
    )

    manifest = {
        "schema": "papr_ssl.p6_15shot_consistency_embeddings.v1",
        "teacher": "WavLM-large hidden_states[15]",
        "head": "AttentionDR 64D",
        "shots_per_intent": SHOT,
        "enrollment_shape": [30, SHOT, 64],
        "known_dev_shape": list(known_embedding.shape),
        "unknown_dev_shape": list(unknown_embedding.shape),
        "checkpoint": meta,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 15-SHOT CONSISTENCY EMBEDDING EXPORT")
    print("=" * 108)
    print(f"enrollment:          {enrollment.shape}")
    print(f"known DEV:           {known_embedding.shape}")
    print(f"unknown DEV:         {unknown_embedding.shape}")
    print("generic_test accessed: NO")
    print("P6 CONSISTENCY EXPORT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
