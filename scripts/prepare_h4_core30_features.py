#!/usr/bin/env python
"""H4-00B — Materialize frozen Core30 global+temporal features for H4.

Context
-------
H4-00 found:
- Core30 TRAIN 3756 / DEV 442
- WavLM[15] frame cache covers all TRAIN+DEV
- 256D global embeddings exist for DEV, but not for all TRAIN
- H4 cannot fit a shared Head until TRAIN evidence is available.

This script DOES NOT train WavLM or the 256D Teacher.
It reuses:
- frozen P5 WavLM[15] frame cache
- already-selected 256D Attention-DR checkpoint

Outputs
-------
1) global/core30_global256.npz
   - all 3756 TRAIN + 442 DEV 256D embeddings
   - utt_id / speaker_id / label / label_index
2) temporal/index.jsonl + utterances/*.pt
   - framewise 1024->256 projection
   - L2 norm -> downsample x3 -> L2 norm
   - same temporal recipe as the promoted DTW branch
3) manifest.json

Safety
------
- TRAIN/DEV only
- TEST/generic_test not accessed
- no checkpoint modification
- no retraining
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from papr_ssl.training.teacher.p5_dr import build_dr
from papr_ssl.training.teacher.p5_frame_data import FrameCacheDataset, frame_collate


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get(row, keys, default=""):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return default


def row_split(row):
    return str(get(row, ("split",))).strip().lower()


def row_utt(row):
    v = get(row, ("utt_id", "id", "audio_id"))
    if v == "":
        raise KeyError(f"No utt_id-like field in keys={sorted(row)}")
    return str(v)


def row_speaker(row):
    v = get(row, ("speaker_id", "speaker", "user_id", "subject_id"))
    if v == "":
        raise KeyError(f"No speaker-like field in keys={sorted(row)}")
    return str(v)


def safe_name(utt_id: str) -> str:
    return hashlib.sha1(utt_id.encode("utf-8")).hexdigest() + ".pt"


def replace_projection(model: nn.Module, dim: int = 256) -> nn.Module:
    model = copy.deepcopy(model)
    cands = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and m.out_features == 64
    ]
    if len(cands) != 1:
        raise RuntimeError(
            f"Expected exactly one 64-output Linear, got "
            f"{[(n, m.in_features, m.out_features) for n, m in cands]}"
        )
    name, old = cands[0]
    model.set_submodule(
        name,
        nn.Linear(old.in_features, dim, bias=(old.bias is not None)),
    )
    return model


def resolve_selected_checkpoint(manifest_path: Path) -> Path:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("Embedding manifest generic_test seal is not intact")
    selected = m.get("selected_checkpoint")
    if not isinstance(selected, dict) or "checkpoint" not in selected:
        raise RuntimeError("selected_checkpoint missing from embedding manifest")
    path = Path(selected["checkpoint"])
    if not path.exists():
        raise FileNotFoundError(
            f"Selected 256D checkpoint does not exist: {path}"
        )
    return path


def load_teacher_dr(checkpoint: Path, device: torch.device):
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(obj.get("embedding_dim", -1)) != 256:
        raise RuntimeError(
            f"Expected 256D Teacher checkpoint, got "
            f"{obj.get('embedding_dim')}"
        )
    dr = replace_projection(build_dr("attention"), 256)
    dr.load_state_dict(obj["dr_state_dict"], strict=True)
    dr.eval().to(device)
    for p in dr.parameters():
        p.requires_grad_(False)
    return dr


def get_frame_projection(dr: nn.Module) -> tuple[str, nn.Linear]:
    cands = [
        (name, module)
        for name, module in dr.named_modules()
        if isinstance(module, nn.Linear)
        and module.in_features == 1024
        and module.out_features == 256
    ]
    if len(cands) != 1:
        raise RuntimeError(
            f"Expected one 1024->256 projection, got "
            f"{[(n, m.in_features, m.out_features) for n, m in cands]}"
        )
    return cands[0]


def temporal_project(
    h: torch.Tensor,
    projection: nn.Linear,
    downsample: int,
) -> torch.Tensor:
    """[T,1024] -> [T',256], same recipe as promoted DTW branch."""
    z = projection(h.to(torch.float32))
    z = F.normalize(z, dim=-1)

    chunks = []
    for start in range(0, z.shape[0], downsample):
        chunk = z[start:start + downsample]
        pooled = chunk.mean(dim=0, keepdim=True)
        chunks.append(F.normalize(pooled, dim=-1))
    return (
        torch.cat(chunks, dim=0)
        .to(torch.float16)
        .cpu()
        .contiguous()
    )


def build_speaker_map(mdsc_manifest: Path):
    out = {}
    for row in read_jsonl(mdsc_manifest):
        sp = row_split(row)
        if sp == "test":
            continue
        if sp not in ("train", "dev"):
            continue
        out[row_utt(row)] = row_speaker(row)
    return out


@torch.inference_mode()
def export_split(
    *,
    dataset: FrameCacheDataset,
    split: str,
    dr: nn.Module,
    projection: nn.Linear,
    speaker_map: dict[str, str],
    device: torch.device,
    temporal_dir: Path,
    temporal_index_rows: list[dict],
    downsample: int,
    batch_size: int,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=frame_collate,
    )

    global_by_utt = {}
    label_by_utt = {}
    label_index_by_utt = {}

    done = 0
    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        mask = batch["frame_mask"].to(device, non_blocking=True)
        labels = batch["labels"].cpu().numpy().astype(np.int64)

        global_z = F.normalize(dr(x, mask).float(), dim=-1).cpu()

        for bi, utt_id in enumerate(batch["utt_ids"]):
            if utt_id not in speaker_map:
                raise RuntimeError(f"Missing speaker_id for {utt_id}")

            global_by_utt[utt_id] = global_z[bi].numpy().astype(np.float32)
            label_index = int(labels[bi])
            label_text = dataset.rows[done + bi]["label_text"]
            label_by_utt[utt_id] = str(label_text)
            label_index_by_utt[utt_id] = label_index

            valid_h = x[bi, mask[bi]]
            seq = temporal_project(valid_h, projection, downsample)

            rel = Path("utterances") / safe_name(utt_id)
            out_path = temporal_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": "papr_ssl.h4_core30_temporal256.v1",
                    "utt_id": utt_id,
                    "split": split,
                    "speaker_id": speaker_map[utt_id],
                    "label_text": str(label_text),
                    "label_index": label_index,
                    "features": seq,
                    "shape": list(seq.shape),
                },
                out_path,
            )
            temporal_index_rows.append(
                {
                    "utt_id": utt_id,
                    "split": split,
                    "speaker_id": speaker_map[utt_id],
                    "label_text": str(label_text),
                    "label_index": label_index,
                    "cache_relpath": str(rel).replace("\\", "/"),
                    "frames": int(seq.shape[0]),
                    "dim": int(seq.shape[1]),
                }
            )

        done += len(batch["utt_ids"])
        if done % 500 == 0 or done == len(dataset):
            print(f"[{split}] {done}/{len(dataset)}")

    ordered_ids = [r["utt_id"] for r in dataset.rows]
    return {
        "utt_id": np.asarray(ordered_ids, dtype=str),
        "speaker_id": np.asarray(
            [speaker_map[u] for u in ordered_ids],
            dtype=str,
        ),
        "label": np.asarray(
            [label_by_utt[u] for u in ordered_ids],
            dtype=str,
        ),
        "label_index": np.asarray(
            [label_index_by_utt[u] for u in ordered_ids],
            dtype=np.int64,
        ),
        "embedding": np.stack(
            [global_by_utt[u] for u in ordered_ids],
            axis=0,
        ).astype(np.float32),
    }


def verify_dev_reference(
    exported_dev: dict,
    reference_npz: Path,
    atol: float,
):
    if not reference_npz.exists():
        return {
            "reference_present": False,
            "status": "SKIPPED",
            "reason": f"reference not found: {reference_npz}",
        }

    d = np.load(reference_npz, allow_pickle=False)
    ref_ids = d["known_utt_id"].astype(str)
    ref_z = d["known_embedding"].astype(np.float32)

    pos = {u: i for i, u in enumerate(exported_dev["utt_id"])}
    missing = [u for u in ref_ids if u not in pos]
    if missing:
        raise RuntimeError(
            f"Reference DEV contains IDs absent from new export: {missing[:5]}"
        )

    new_z = np.stack(
        [exported_dev["embedding"][pos[u]] for u in ref_ids],
        axis=0,
    )
    max_abs = float(np.max(np.abs(new_z - ref_z)))
    mean_abs = float(np.mean(np.abs(new_z - ref_z)))

    new_n = new_z / np.maximum(
        np.linalg.norm(new_z, axis=1, keepdims=True), 1e-12
    )
    ref_n = ref_z / np.maximum(
        np.linalg.norm(ref_z, axis=1, keepdims=True), 1e-12
    )
    cosine = np.sum(new_n * ref_n, axis=1)
    min_cos = float(cosine.min())
    mean_cos = float(cosine.mean())

    status = "PASS" if max_abs <= atol else "FAIL"
    if status != "PASS":
        raise RuntimeError(
            f"DEV reproduction failed: max_abs={max_abs:.3e} > atol={atol:.3e}"
        )

    return {
        "reference_present": True,
        "status": status,
        "rows": int(len(ref_ids)),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "min_cosine": min_cos,
        "mean_cosine": mean_cos,
        "atol": atol,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--frame-cache",
        type=Path,
        default=Path("artifacts/p5_frame_cache/wavlm_large_layer15"),
    )
    p.add_argument(
        "--core-index",
        type=Path,
        default=Path(
            "artifacts/p2_07/mdsc_policy_v2/"
            "mdsc_core30.index.jsonl"
        ),
    )
    p.add_argument(
        "--mdsc-manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--embedding-manifest",
        type=Path,
        default=Path(
            "artifacts/p6_teacher_256_15shot/embeddings/manifest.json"
        ),
    )
    p.add_argument(
        "--reference-dev-npz",
        type=Path,
        default=Path(
            "artifacts/p6_teacher_256_15shot/embeddings/"
            "teacher_256d_15shot_embeddings.npz"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/h4_core30_features"),
    )
    p.add_argument("--downsample", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--repro-atol", type=float, default=1e-5)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing overwrite non-empty output dir: {args.output_dir}"
        )

    global_dir = args.output_dir / "global"
    temporal_dir = args.output_dir / "temporal"
    global_dir.mkdir(parents=True, exist_ok=True)
    temporal_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = resolve_selected_checkpoint(args.embedding_manifest)
    print(f"[H4 features] checkpoint: {checkpoint}")

    dr = load_teacher_dr(checkpoint, device)
    projection_name, projection = get_frame_projection(dr)
    speaker_map = build_speaker_map(args.mdsc_manifest)

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

    if len(train_ds) != 3756 or len(dev_ds) != 442:
        raise RuntimeError(
            f"Unexpected Core30 sizes train={len(train_ds)} dev={len(dev_ds)}"
        )

    temporal_index_rows = []

    train = export_split(
        dataset=train_ds,
        split="train",
        dr=dr,
        projection=projection,
        speaker_map=speaker_map,
        device=device,
        temporal_dir=temporal_dir,
        temporal_index_rows=temporal_index_rows,
        downsample=args.downsample,
        batch_size=args.batch_size,
    )
    dev = export_split(
        dataset=dev_ds,
        split="dev",
        dr=dr,
        projection=projection,
        speaker_map=speaker_map,
        device=device,
        temporal_dir=temporal_dir,
        temporal_index_rows=temporal_index_rows,
        downsample=args.downsample,
        batch_size=args.batch_size,
    )

    np.savez_compressed(
        global_dir / "core30_global256.npz",
        train_utt_id=train["utt_id"],
        train_speaker_id=train["speaker_id"],
        train_label=train["label"],
        train_label_index=train["label_index"],
        train_embedding=train["embedding"],
        dev_utt_id=dev["utt_id"],
        dev_speaker_id=dev["speaker_id"],
        dev_label=dev["label"],
        dev_label_index=dev["label_index"],
        dev_embedding=dev["embedding"],
    )

    with (temporal_dir / "index.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in temporal_index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    repro = verify_dev_reference(
        dev,
        args.reference_dev_npz,
        args.repro_atol,
    )

    train_norm = np.linalg.norm(train["embedding"], axis=1)
    dev_norm = np.linalg.norm(dev["embedding"], axis=1)
    temporal_lengths = np.asarray(
        [r["frames"] for r in temporal_index_rows],
        dtype=np.int64,
    )

    manifest = {
        "schema": "papr_ssl.h4_core30_features.v1",
        "source": {
            "frame_cache": str(args.frame_cache),
            "checkpoint": str(checkpoint),
            "projection_module": projection_name,
            "teacher": "WavLM-large[15] + Attention DR 256D",
        },
        "global": {
            "train_shape": list(train["embedding"].shape),
            "dev_shape": list(dev["embedding"].shape),
            "dtype": str(train["embedding"].dtype),
            "train_norm_mean": float(train_norm.mean()),
            "train_norm_max_abs_error_from_1": float(
                np.max(np.abs(train_norm - 1.0))
            ),
            "dev_norm_mean": float(dev_norm.mean()),
            "dev_norm_max_abs_error_from_1": float(
                np.max(np.abs(dev_norm - 1.0))
            ),
        },
        "temporal": {
            "rows": len(temporal_index_rows),
            "train_rows": int(
                sum(r["split"] == "train" for r in temporal_index_rows)
            ),
            "dev_rows": int(
                sum(r["split"] == "dev" for r in temporal_index_rows)
            ),
            "dim": 256,
            "dtype": "float16",
            "downsample_factor": args.downsample,
            "frames_min": int(temporal_lengths.min()),
            "frames_median": float(np.median(temporal_lengths)),
            "frames_p95": float(np.quantile(temporal_lengths, 0.95)),
            "frames_max": int(temporal_lengths.max()),
        },
        "dev_reproduction_vs_existing_256d": repro,
        "trainable_parameters_updated": False,
        "checkpoint_modified": False,
        "test_rows_accessed": 0,
        "generic_test": "sealed_not_accessed",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 108)
    print("H4-00B CORE30 FROZEN FEATURE MATERIALIZATION")
    print("=" * 108)
    print(f"global TRAIN shape:       {train['embedding'].shape}")
    print(f"global DEV shape:         {dev['embedding'].shape}")
    print(f"temporal rows:            {len(temporal_index_rows)}")
    print(f"projection:               {projection_name}")
    print(f"downsample:               {args.downsample}")
    if repro["reference_present"]:
        print(
            f"DEV reproduction:         {repro['status']} "
            f"(max_abs={repro['max_abs_diff']:.3e}, "
            f"min_cos={repro['min_cosine']:.8f})"
        )
    else:
        print("DEV reproduction:         SKIPPED (reference absent)")
    print("trainable params updated: NO")
    print("generic_test accessed:    NO")
    print("H4-00B STATUS: PASS")


if __name__ == "__main__":
    main()
