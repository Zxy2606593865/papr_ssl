#!/usr/bin/env python
"""Export frozen P6 K=2 empirical dev score/margin arrays.

Purpose
-------
This is a protocol-preserving diagnostic/export step after the coarse quantile
calibrator reported calibration_feasible=False on generic_dev_cal.

No threshold is selected here.
No generic_test row is loaded.

Frozen model/protocol
---------------------
- microsoft/wavlm-large
- revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c
- outputs.hidden_states[15]
- P5 representative checkpoints from p6_k2_protocol.json
- K=2 independent subprototypes per class
- class score = max cosine over the 2 subprototypes
- same generic_dev_cal / generic_dev_score partitions as prior K=2 run
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


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dr(method: str, checkpoint: Path, device: torch.device):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if obj.get("method") != method:
        raise RuntimeError(
            f"checkpoint method mismatch: expected={method}, got={obj.get('method')}"
        )
    model = build_dr(method)
    model.load_state_dict(obj["dr_state_dict"], strict=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.inference_mode()
def embed_dataset(
    dataset: FrameCacheDataset,
    drs: dict[str, torch.nn.Module],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=frame_collate,
    )
    out = {method: {} for method in drs}
    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        mask = batch["frame_mask"].to(device, non_blocking=True)
        for method, dr in drs.items():
            z = dr(x, mask).cpu()
            for utt_id, emb in zip(batch["utt_ids"], z):
                out[method][utt_id] = emb.contiguous()
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
    wanted_utt_ids: set[str],
    backbone: WavLMModel,
    drs: dict[str, torch.nn.Module],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    rows = load_real_task_rows(open_index, splits=("dev",))
    rows = [row for row in rows if row.utt_id in wanted_utt_ids]
    if len(rows) != len(wanted_utt_ids):
        raise RuntimeError(
            f"wanted {len(wanted_utt_ids)} unknown dev rows, got {len(rows)}"
        )

    legacy = [
        row
        for row in rows
        if row.audio_relpath is None and qualified_utt_relpath(row) is None
    ]
    catalog = build_mdsc_audio_catalog(mdsc_root) if legacy else None

    out = {method: {} for method in drs}
    for i, row in enumerate(rows, start=1):
        path = resolve_task_audio_path(
            row=row,
            mdsc_root=mdsc_root,
            audio_catalog=catalog,
        )
        x = load_wave(path).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(x, dtype=torch.long, device=device)
        model_out = backbone(
            input_values=x,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = model_out.hidden_states[HIDDEN_STATE_INDEX].to(torch.float32)
        frame_mask = backbone._get_feature_vector_attention_mask(
            hs.shape[1],
            attention_mask,
        ).to(torch.bool)
        for method, dr in drs.items():
            out[method][row.utt_id] = (
                dr(hs, frame_mask).squeeze(0).cpu().contiguous()
            )
        if i % 100 == 0 or i == len(rows):
            print(f"[unknown dev] {i}/{len(rows)}")
    return out


def build_subprototypes(
    enrollment_rows: list[dict],
    embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    subp = torch.empty((30, 2, 64), dtype=torch.float32)
    seen = set()
    for row in enrollment_rows:
        c = int(row["label_index"])
        k = int(row["subprototype_index"])
        subp[c, k] = F.normalize(
            embeddings[row["utt_id"]],
            p=2,
            dim=0,
        )
        seen.add((c, k))
    if len(seen) != 60:
        raise RuntimeError(f"expected 60 class/subprototype entries, got {len(seen)}")
    return subp


def score_rows(
    rows: list[dict],
    embeddings: dict[str, torch.Tensor],
    subprototypes: torch.Tensor,
    *,
    known: bool,
) -> dict[str, np.ndarray]:
    z = torch.stack([embeddings[row["utt_id"]] for row in rows], dim=0)
    z = F.normalize(z, p=2, dim=-1)
    sim_all = torch.einsum("bd,ckd->bck", z, subprototypes)
    class_scores = sim_all.max(dim=-1).values
    top2 = torch.topk(class_scores, k=2, dim=1)

    result = {
        "utt_id": np.asarray([row["utt_id"] for row in rows], dtype=str),
        "score": top2.values[:, 0].numpy().astype(np.float64),
        "margin": (
            top2.values[:, 0] - top2.values[:, 1]
        ).numpy().astype(np.float64),
        "pred": top2.indices[:, 0].numpy().astype(np.int64),
    }
    if known:
        result["true"] = np.asarray(
            [int(row["label_index"]) for row in rows],
            dtype=np.int64,
        )
    return result


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
        default=Path("artifacts/p6_dev_k2_exact/scores"),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"score export already exists; refusing overwrite: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(
        (args.protocol_dir / "p6_k2_protocol.json").read_text(encoding="utf-8")
    )
    if protocol.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("protocol test seal violated")
    if int(protocol.get("subprototype_k")) != 2:
        raise RuntimeError("this script requires K=2 subprototype protocol")

    enrollment = read_jsonl(
        args.protocol_dir / "generic_enrollment_k2_subproto.jsonl"
    )
    known_cal = read_jsonl(args.protocol_dir / "generic_dev_cal_known.jsonl")
    known_score = read_jsonl(args.protocol_dir / "generic_dev_score_known.jsonl")
    unknown_cal = read_jsonl(args.protocol_dir / "generic_dev_cal_unknown.jsonl")
    unknown_score = read_jsonl(args.protocol_dir / "generic_dev_score_unknown.jsonl")

    device = torch.device(args.device)
    drs = {}
    checkpoint_meta = {}
    for method in ("mean", "attention"):
        meta = protocol["representative_checkpoints"][method]
        checkpoint = Path(meta["checkpoint"])
        if sha256(checkpoint) != meta["checkpoint_sha256"]:
            raise RuntimeError(f"{method}: checkpoint hash mismatch")
        drs[method] = load_dr(method, checkpoint, device)
        checkpoint_meta[method] = meta

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

    print("[exact] embedding known train/dev from frozen frame cache...")
    train_emb = embed_dataset(train_ds, drs, device)
    dev_emb = embed_dataset(dev_ds, drs, device)

    wanted_unknown = {
        row["utt_id"] for row in unknown_cal + unknown_score
    }
    print("[exact] embedding all open-set DEV rows only...")
    backbone = WavLMModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        output_hidden_states=True,
        local_files_only=True,
    )
    backbone.eval().to(device)
    for param in backbone.parameters():
        param.requires_grad_(False)

    unknown_emb = embed_unknown_dev(
        open_index=args.open_index,
        mdsc_root=args.mdsc_root,
        wanted_utt_ids=wanted_unknown,
        backbone=backbone,
        drs=drs,
        device=device,
    )

    manifest = {
        "schema": "papr_ssl.p6_k2_empirical_scores.v1",
        "purpose": "exact_empirical_threshold_calibration",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "hidden_state_index": HIDDEN_STATE_INDEX,
        "subprototype_k": 2,
        "checkpoint_meta": checkpoint_meta,
        "known_cal_count": len(known_cal),
        "known_score_count": len(known_score),
        "unknown_cal_count": len(unknown_cal),
        "unknown_score_count": len(unknown_score),
        "generic_test": "sealed_not_accessed",
    }

    for method in ("mean", "attention"):
        subp = build_subprototypes(enrollment, train_emb[method])

        kc = score_rows(known_cal, dev_emb[method], subp, known=True)
        ks = score_rows(known_score, dev_emb[method], subp, known=True)
        uc = score_rows(unknown_cal, unknown_emb[method], subp, known=False)
        us = score_rows(unknown_score, unknown_emb[method], subp, known=False)

        np.savez_compressed(
            args.output_dir / f"{method}_dev_scores.npz",
            known_cal_utt_id=kc["utt_id"],
            known_cal_score=kc["score"],
            known_cal_margin=kc["margin"],
            known_cal_pred=kc["pred"],
            known_cal_true=kc["true"],
            known_score_utt_id=ks["utt_id"],
            known_score_score=ks["score"],
            known_score_margin=ks["margin"],
            known_score_pred=ks["pred"],
            known_score_true=ks["true"],
            unknown_cal_utt_id=uc["utt_id"],
            unknown_cal_score=uc["score"],
            unknown_cal_margin=uc["margin"],
            unknown_cal_pred=uc["pred"],
            unknown_score_utt_id=us["utt_id"],
            unknown_score_score=us["score"],
            unknown_score_margin=us["margin"],
            unknown_score_pred=us["pred"],
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("P6 K=2 EMPIRICAL SCORE EXPORT")
    print("=" * 108)
    print(f"known cal/score:   {len(known_cal)} / {len(known_score)}")
    print(f"unknown cal/score: {len(unknown_cal)} / {len(unknown_score)}")
    print("generic_test accessed: NO")
    print("P6 K2 SCORE EXPORT STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
