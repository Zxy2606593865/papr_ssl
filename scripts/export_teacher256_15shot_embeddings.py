#!/usr/bin/env python
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
SHOT = 15
EMBED_DIM = 256
SALT = "papr_ssl_p6_protocol_v1"


def stable_key(*parts: str) -> str:
    h = hashlib.sha256()
    h.update(SALT.encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return h.hexdigest()


def replace_projection(model: nn.Module, dim: int) -> nn.Module:
    model = copy.deepcopy(model)
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and m.out_features == 64]
    if len(cands) != 1:
        raise RuntimeError(
            f"expected exactly one 64-output Linear, got "
            f"{[(n,m.in_features,m.out_features) for n,m in cands]}"
        )
    name, old = cands[0]
    new = nn.Linear(old.in_features, dim, bias=(old.bias is not None))
    model.set_submodule(name, new)
    return model


def choose_median_checkpoint(sweep_dir: Path):
    rows = []
    for seed in (17, 29, 43):
        rpath = sweep_dir / "dim_256" / f"seed_{seed}" / "result.json"
        cpath = sweep_dir / "dim_256" / f"seed_{seed}" / "best.pt"
        if not rpath.exists() or not cpath.exists():
            raise FileNotFoundError(f"missing 256D artifacts for seed {seed}")
        r = json.loads(rpath.read_text(encoding="utf-8"))
        rows.append({
            "seed": seed,
            "score": float(r["selected_generic_dev_score"]),
            "epoch": int(r["selected_epoch"]),
            "checkpoint": str(cpath),
        })
    ordered = sorted(rows, key=lambda x: (x["score"], x["seed"]))
    return ordered[1], rows


def load_dr(checkpoint: Path, device: torch.device):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(obj.get("embedding_dim", -1)) != 256:
        raise RuntimeError(f"not a 256D checkpoint: {obj.get('embedding_dim')}")
    dr = replace_projection(build_dr("attention"), 256)
    dr.load_state_dict(obj["dr_state_dict"], strict=True)
    dr.eval().to(device)
    for p in dr.parameters():
        p.requires_grad_(False)
    return dr


@torch.inference_mode()
def embed_cached(dataset, dr, device):
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(), collate_fn=frame_collate
    )
    out = {}
    for batch in loader:
        x = batch["features"].to(device)
        mask = batch["frame_mask"].to(device)
        z = torch.nn.functional.normalize(dr(x, mask).float(), dim=-1).cpu()
        for utt, emb in zip(batch["utt_ids"], z):
            out[utt] = emb.contiguous()
    return out


def load_wave(path: Path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if int(sr) != 16000:
        raise RuntimeError(f"expected 16kHz: {path}")
    return torch.from_numpy(data).float().mean(dim=1).contiguous()


@torch.inference_mode()
def embed_open_dev(open_index, mdsc_root, backbone, dr, device):
    rows = load_real_task_rows(open_index, splits=("dev",))
    if len(rows) != 1178:
        raise RuntimeError(f"expected 1178 open DEV rows, got {len(rows)}")

    legacy = [r for r in rows
              if r.audio_relpath is None and qualified_utt_relpath(r) is None]
    catalog = build_mdsc_audio_catalog(mdsc_root) if legacy else None

    out = {}
    for i, row in enumerate(rows, 1):
        path = resolve_task_audio_path(
            row=row, mdsc_root=mdsc_root, audio_catalog=catalog
        )
        wav = load_wave(path).unsqueeze(0).to(device)
        am = torch.ones_like(wav, dtype=torch.long)
        mo = backbone(
            input_values=wav,
            attention_mask=am,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = mo.hidden_states[15].float()
        fm = backbone._get_feature_vector_attention_mask(
            hs.shape[1], am
        ).bool()
        z = torch.nn.functional.normalize(
            dr(hs, fm).squeeze(0).float(), dim=-1
        ).cpu()
        out[row.utt_id] = z
        if i % 100 == 0 or i == len(rows):
            print(f"[open DEV] {i}/{len(rows)}")
    return rows, out


def build_15shot(train_ds):
    groups = {}
    for row in train_ds.rows:
        groups.setdefault(row["label_text"], []).append(row["utt_id"])

    rows = []
    for label in sorted(groups):
        ordered = sorted(
            groups[label],
            key=lambda utt: stable_key("enrollment", label, utt)
        )
        if len(ordered) < 15:
            raise RuntimeError(f"class {label} has fewer than 15 samples")
        c = train_ds.class_to_index[label]
        for utt in ordered[:15]:
            rows.append((c, label, utt))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path,
                    default=Path("artifacts/p6_teacher_dim_sweep"))
    ap.add_argument("--frame-cache", type=Path,
                    default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"))
    ap.add_argument("--core-index", type=Path,
                    default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30.index.jsonl"))
    ap.add_argument("--open-index", type=Path,
                    default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_core30_open_set_eval.index.jsonl"))
    ap.add_argument("--mdsc-root", type=Path, default=Path("datasets/public/mdsc"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("artifacts/p6_teacher_256_15shot/embeddings"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected, all_rows = choose_median_checkpoint(args.sweep_dir)
    print("[256D] checkpoint candidates:")
    for r in sorted(all_rows, key=lambda x: x["seed"]):
        print(f"  seed={r['seed']} score={r['score']:.6f} epoch={r['epoch']}")
    print(f"[256D] median representative seed={selected['seed']} score={selected['score']:.6f}")

    device = torch.device(args.device)
    dr = load_dr(Path(selected["checkpoint"]), device)

    train_ds = FrameCacheDataset(
        cache_dir=args.frame_cache, task_index=args.core_index, split="train"
    )
    dev_ds = FrameCacheDataset(
        cache_dir=args.frame_cache, task_index=args.core_index, split="dev",
        class_to_index=train_ds.class_to_index
    )
    if len(train_ds) != 3756 or len(dev_ds) != 442:
        raise RuntimeError("unexpected Core30 counts")

    train_emb = embed_cached(train_ds, dr, device)
    dev_emb = embed_cached(dev_ds, dr, device)

    enrollment = np.zeros((30, 15, 256), dtype=np.float32)
    enrollment_ids = np.empty((30, 15), dtype="<U256")
    class_names = np.empty((30,), dtype="<U256")
    cursor = [0]*30
    for c, label, utt in build_15shot(train_ds):
        j = cursor[c]
        enrollment[c, j] = train_emb[utt].numpy()
        enrollment_ids[c, j] = utt
        class_names[c] = label
        cursor[c] += 1

    known_ids = np.asarray([r["utt_id"] for r in dev_ds.rows], dtype=str)
    known_true = np.asarray([int(r["label_index"]) for r in dev_ds.rows], dtype=np.int64)
    known_embedding = np.stack([dev_emb[u].numpy() for u in known_ids]).astype(np.float32)

    backbone = WavLMModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION,
        output_hidden_states=True, local_files_only=True
    )
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    open_rows, open_emb = embed_open_dev(
        args.open_index, args.mdsc_root, backbone, dr, device
    )
    unknown_ids = np.asarray([r.utt_id for r in open_rows], dtype=str)
    unknown_embedding = np.stack([open_emb[u].numpy() for u in unknown_ids]).astype(np.float32)

    np.savez_compressed(
        args.output_dir / "teacher_256d_15shot_embeddings.npz",
        enrollment=enrollment,
        enrollment_utt_id=enrollment_ids,
        class_name=class_names,
        known_utt_id=known_ids,
        known_true=known_true,
        known_embedding=known_embedding,
        unknown_utt_id=unknown_ids,
        unknown_embedding=unknown_embedding,
    )

    (args.output_dir / "manifest.json").write_text(
        json.dumps({
            "schema": "papr_ssl.p6_teacher_256_15shot_embeddings.v1",
            "selected_checkpoint_policy": "median selected-dev-score among seeds 17/29/43",
            "selected_checkpoint": selected,
            "all_seed_candidates": all_rows,
            "embedding_dim": 256,
            "project_2_canonical_64d": "preserved_not_modified",
            "generic_test": "sealed_not_accessed",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("="*108)
    print("P6 256D TEACHER 15-SHOT EMBEDDING EXPORT")
    print(f"representative seed:    {selected['seed']}")
    print(f"known DEV shape:        {known_embedding.shape}")
    print(f"unknown DEV shape:      {unknown_embedding.shape}")
    print("canonical 64D modified: NO")
    print("generic_test accessed:  NO")
    print("EXPORT STATUS: PASS")


if __name__ == "__main__":
    main()
