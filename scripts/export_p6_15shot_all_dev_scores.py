#!/usr/bin/env python
"""Export per-sample DEV scores for the frozen 15-shot mean-prototype condition.

This is a development diagnostic only.

Frozen:
- WavLM-large outputs.hidden_states[15]
- AttentionDR 64D representative checkpoint
- 15 REAL enrollment utterances per intent
- one mean prototype per intent
- generic_test remains sealed

The export contains ALL Core30 dev and ALL open-set dev samples so that repeated
calibration/score splits can be generated without rerunning WavLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
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
        z = dr(x, mask).cpu()
        for utt_id, emb in zip(batch["utt_ids"], z):
            out[utt_id] = emb.contiguous()
    return out


def load_wave(path: Path) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if int(sr) != SAMPLE_RATE:
        raise RuntimeError(f"expected 16 kHz: {path}")
    x = torch.from_numpy(data).float().mean(dim=1)
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
        hs = model_out.hidden_states[HIDDEN_STATE_INDEX].float()
        frame_mask = backbone._get_feature_vector_attention_mask(
            hs.shape[1],
            attention_mask,
        ).bool()
        out[row.utt_id] = dr(
            hs,
            frame_mask,
        ).squeeze(0).cpu().contiguous()

        if i % 100 == 0 or i == len(rows):
            print(f"[unknown dev] {i}/{len(rows)}")
    return rows, out


def build_15shot_enrollment(train_ds):
    groups = {}
    for row in train_ds.rows:
        groups.setdefault(row["label_text"], []).append(row["utt_id"])

    enrollment = []
    for label in sorted(groups):
        ordered = sorted(
            groups[label],
            key=lambda u: stable_key("enrollment", label, u),
        )
        if len(ordered) < SHOT:
            raise RuntimeError(f"class {label!r} has fewer than {SHOT} samples")
        for utt_id in ordered[:SHOT]:
            enrollment.append(
                {
                    "utt_id": utt_id,
                    "label_text": label,
                    "label_index": train_ds.class_to_index[label],
                }
            )
    if len(enrollment) != 30 * SHOT:
        raise RuntimeError("unexpected enrollment count")
    return enrollment


def build_mean_prototypes(enrollment, train_embeddings):
    by_class = {}
    for row in enrollment:
        by_class.setdefault(int(row["label_index"]), []).append(
            train_embeddings[row["utt_id"]]
        )

    protos = []
    for c in range(30):
        x = torch.stack(by_class[c], dim=0).mean(dim=0)
        protos.append(F.normalize(x, p=2, dim=0))
    return torch.stack(protos, dim=0)


def score_embeddings(
    utt_ids: list[str],
    embeddings: dict[str, torch.Tensor],
    prototypes: torch.Tensor,
):
    z = F.normalize(
        torch.stack([embeddings[u] for u in utt_ids], dim=0),
        p=2,
        dim=-1,
    )
    sim = z @ prototypes.T
    top2 = torch.topk(sim, k=2, dim=1)
    return (
        top2.values[:, 0].numpy().astype(np.float64),
        (top2.values[:, 0] - top2.values[:, 1]).numpy().astype(np.float64),
        top2.indices[:, 0].numpy().astype(np.int64),
    )


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
        default=Path("artifacts/p6_15shot_stability/scores"),
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
        raise RuntimeError("protocol test seal violated")

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
        raise RuntimeError(f"expected 442 known dev rows, got {len(dev_ds)}")

    print("[15-shot stability] embedding known train/dev...")
    train_emb = embed_dataset(train_ds, dr, device)
    dev_emb = embed_dataset(dev_ds, dr, device)

    enrollment = build_15shot_enrollment(train_ds)
    prototypes = build_mean_prototypes(enrollment, train_emb)

    known_utt_ids = [row["utt_id"] for row in dev_ds.rows]
    known_true = np.asarray(
        [int(row["label_index"]) for row in dev_ds.rows],
        dtype=np.int64,
    )
    known_label_text = np.asarray(
        [row["label_text"] for row in dev_ds.rows],
        dtype=str,
    )
    known_score, known_margin, known_pred = score_embeddings(
        known_utt_ids,
        dev_emb,
        prototypes,
    )

    print("[15-shot stability] embedding all open-set DEV once...")
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
    unknown_utt_ids = [row.utt_id for row in unknown_rows]
    unknown_score, unknown_margin, unknown_pred = score_embeddings(
        unknown_utt_ids,
        unknown_emb,
        prototypes,
    )

    np.savez_compressed(
        args.output_dir / "attention_15shot_all_dev_scores.npz",
        known_utt_id=np.asarray(known_utt_ids, dtype=str),
        known_label_text=known_label_text,
        known_true=known_true,
        known_pred=known_pred,
        known_score=known_score,
        known_margin=known_margin,
        unknown_utt_id=np.asarray(unknown_utt_ids, dtype=str),
        unknown_pred=unknown_pred,
        unknown_score=unknown_score,
        unknown_margin=unknown_margin,
    )

    manifest = {
        "schema": "papr_ssl.p6_15shot_all_dev_scores.v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "hidden_state_index": HIDDEN_STATE_INDEX,
        "head": "AttentionDR",
        "embedding_dim": 64,
        "shots_per_intent": SHOT,
        "prototype_policy": "mean_then_l2",
        "known_dev_count": len(known_utt_ids),
        "unknown_dev_count": len(unknown_utt_ids),
        "checkpoint": meta,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 15-SHOT ALL-DEV SCORE EXPORT")
    print("=" * 108)
    print(f"known dev:           {len(known_utt_ids)}")
    print(f"unknown dev:         {len(unknown_utt_ids)}")
    print(f"enrollment:          {30 * SHOT} (30 x {SHOT})")
    print("generic_test accessed: NO")
    print("P6 15-SHOT SCORE EXPORT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
