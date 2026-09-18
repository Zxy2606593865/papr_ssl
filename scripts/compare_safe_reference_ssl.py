#!/usr/bin/env python
"""P2-06C real safe-batch policy vs exact singleton reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import yaml

from papr_ssl.data.audio_pipeline import collate_audio_examples, load_audio_example
from papr_ssl.data.manifest_io import read_manifest_jsonl
from papr_ssl.models.teacher.backbones.length_aware import LengthAwareSSLAdapter
from papr_ssl.models.teacher.backbones.safe_batch import (
    batching_policy_name,
    safe_forward_audio_batch,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument(
        "--gsc-manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/gsc_v2.jsonl"),
    )
    p.add_argument(
        "--mdsc-manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--gsc-root",
        type=Path,
        default=Path("datasets/public/speech_commands_v2"),
    )
    p.add_argument(
        "--mdsc-root",
        type=Path,
        default=Path("datasets/public/mdsc"),
    )
    p.add_argument("--json-out", type=Path, required=True)
    return p.parse_args()


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise TypeError("config root must be a mapping")
    return obj


def extract_backbone_cfg(config: Mapping[str, Any]):
    if isinstance(config.get("backbone"), Mapping):
        return dict(config["backbone"])
    teacher = config.get("teacher")
    if isinstance(teacher, Mapping) and isinstance(teacher.get("backbone"), Mapping):
        return dict(teacher["backbone"])
    raise KeyError("Could not locate backbone config")


def create_backbone(cfg):
    from papr_ssl.models.teacher.backbones.factory import create_ssl_backbone
    kind = cfg["kind"]
    return create_ssl_backbone(
        kind,
        **{k: v for k, v in cfg.items() if k != "kind"},
    )


def device_from_arg(value):
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def picks_gsc(records):
    shortest = min(records, key=lambda r: r.num_frames)
    full = next(r for r in records if r.num_frames == 16000)
    return [shortest, full]


def picks_mdsc(records):
    candidates = sorted(
        [r for r in records if 14000 <= r.num_frames <= 40000],
        key=lambda r: r.num_frames,
    )
    first = candidates[0]
    second = next(r for r in reversed(candidates) if r.num_frames != first.num_frames)
    return [first, second]


def make_batch(records, dataset_name, root, device):
    examples = [
        load_audio_example(r, {dataset_name: root})
        for r in records
    ]
    return collate_audio_examples(examples).to(device)


def compare_outputs(reference, safe):
    ref_lengths = reference.frame_mask.sum(dim=1)
    safe_lengths = safe.frame_mask.sum(dim=1)
    if not torch.equal(ref_lengths, safe_lengths):
        return {
            "gate": "FAIL",
            "reason": "valid frame counts differ",
            "reference_lengths": ref_lengths.cpu().tolist(),
            "safe_lengths": safe_lengths.cpu().tolist(),
        }

    rows = []
    gate = "PASS"
    for i, length in enumerate(ref_lengths.cpu().tolist()):
        length = int(length)
        a = reference.features[i, :length].float()
        b = safe.features[i, :length].float()
        diff = (a - b).abs()
        frame_cos = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
        pooled_cos = F.cosine_similarity(
            a.mean(dim=0, keepdim=True),
            b.mean(dim=0, keepdim=True),
            dim=-1,
            eps=1e-8,
        )[0]
        row = {
            "index": i,
            "valid_frames": length,
            "mean_abs_error": float(diff.mean().item()),
            "max_abs_error": float(diff.max().item()),
            "mean_frame_cosine": float(frame_cos.mean().item()),
            "min_frame_cosine": float(frame_cos.min().item()),
            "pooled_cosine": float(pooled_cos.item()),
        }
        if row["mean_frame_cosine"] < 0.999 or row["pooled_cosine"] < 0.999:
            gate = "FAIL"
        rows.append(row)

    return {"gate": gate, "samples": rows}


def run_one(backbone, kind, batch):
    with torch.inference_mode():
        reference = LengthAwareSSLAdapter(backbone).forward_audio_batch(batch)
        safe = safe_forward_audio_batch(
            backbone_kind=kind,
            backbone=backbone,
            batch=batch,
        )
    return compare_outputs(reference, safe)


def main():
    args = parse_args()
    device = device_from_arg(args.device)
    cfg = extract_backbone_cfg(load_yaml(args.config))
    kind = cfg["kind"]
    backbone = create_backbone(cfg).to(device).eval()

    gsc_records = read_manifest_jsonl(args.gsc_manifest)
    mdsc_records = read_manifest_jsonl(args.mdsc_manifest)

    gsc_batch = make_batch(
        picks_gsc(gsc_records), "gsc_v2", args.gsc_root, device
    )
    mdsc_batch = make_batch(
        picks_mdsc(mdsc_records), "mdsc", args.mdsc_root, device
    )

    gsc = run_one(backbone, kind, gsc_batch)
    mdsc = run_one(backbone, kind, mdsc_batch)
    overall = "PASS" if gsc["gate"] == "PASS" and mdsc["gate"] == "PASS" else "FAIL"

    result = {
        "phase": "P2-06C",
        "kind": kind,
        "device": str(device),
        "policy": batching_policy_name(kind),
        "gsc_v2": gsc,
        "mdsc": mdsc,
        "gate": overall,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("PAPR-SSL P2-06C SAFE BATCH POLICY")
    print("=" * 88)
    print(f"Backbone: {kind}")
    print(f"Policy:   {result['policy']}")
    print(f"GSC:      {gsc['gate']}")
    print(f"MDSC:     {mdsc['gate']}")
    print(f"Gate:     {overall}")
    print(f"JSON:     {args.json_out}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
