#!/usr/bin/env python
"""P2-06B real native-batch vs singleton-reference comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from papr_ssl.data.audio_pipeline import (
    collate_audio_examples,
    load_audio_example,
)
from papr_ssl.data.manifest_io import read_manifest_jsonl
from papr_ssl.experiments.native_batch_compare import compare_native_to_reference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
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
    p.add_argument("--min-frame-cosine", type=float, default=0.999)
    p.add_argument("--min-pooled-cosine", type=float, default=0.999)
    p.add_argument("--json-out", type=Path, required=True)
    return p.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise TypeError("config root must be a mapping")
    return obj


def extract_backbone_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("backbone"), Mapping):
        return dict(config["backbone"])
    teacher = config.get("teacher")
    if isinstance(teacher, Mapping) and isinstance(
        teacher.get("backbone"), Mapping
    ):
        return dict(teacher["backbone"])
    raise KeyError("Could not locate backbone config")


def create_backbone(cfg):
    from papr_ssl.models.teacher.backbones.factory import create_ssl_backbone
    kind = cfg["kind"]
    kwargs = {k: v for k, v in cfg.items() if k != "kind"}
    return create_ssl_backbone(kind, **kwargs)


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_gsc(records):
    shortest = min(records, key=lambda r: r.num_frames)
    full = next(r for r in records if r.num_frames == 16000)
    return [shortest, full]


def pick_mdsc(records):
    candidates = sorted(
        [r for r in records if 14000 <= r.num_frames <= 40000],
        key=lambda r: r.num_frames,
    )
    if len(candidates) < 2:
        raise RuntimeError("Need at least two bounded MDSC examples")
    shortest = candidates[0]
    longest = next(
        r for r in reversed(candidates)
        if r.num_frames != shortest.num_frames
    )
    return [shortest, longest]


def make_batch(records, dataset_name, dataset_root, device):
    examples = [
        load_audio_example(r, {dataset_name: dataset_root})
        for r in records
    ]
    return collate_audio_examples(examples).to(device)


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    cfg = extract_backbone_cfg(load_yaml(args.config))
    backbone = create_backbone(cfg).to(device).eval()

    gsc_records = read_manifest_jsonl(args.gsc_manifest)
    mdsc_records = read_manifest_jsonl(args.mdsc_manifest)

    gsc_batch = make_batch(
        pick_gsc(gsc_records),
        "gsc_v2",
        args.gsc_root,
        device,
    )
    mdsc_batch = make_batch(
        pick_mdsc(mdsc_records),
        "mdsc",
        args.mdsc_root,
        device,
    )

    gsc_cmp = compare_native_to_reference(
        backbone,
        gsc_batch,
        min_mean_frame_cosine=args.min_frame_cosine,
        min_pooled_cosine=args.min_pooled_cosine,
    )
    mdsc_cmp = compare_native_to_reference(
        backbone,
        mdsc_batch,
        min_mean_frame_cosine=args.min_frame_cosine,
        min_pooled_cosine=args.min_pooled_cosine,
    )

    overall = "PASS" if (
        gsc_cmp.gate == "PASS" and mdsc_cmp.gate == "PASS"
    ) else "FAIL"

    report = {
        "phase": "P2-06B",
        "config": str(args.config),
        "kind": cfg["kind"],
        "device": str(device),
        "thresholds": {
            "min_mean_frame_cosine": args.min_frame_cosine,
            "min_pooled_cosine": args.min_pooled_cosine,
        },
        "gsc_v2": {
            "waveform_lengths": gsc_batch.lengths.cpu().tolist(),
            "comparison": gsc_cmp.to_dict(),
        },
        "mdsc": {
            "waveform_lengths": mdsc_batch.lengths.cpu().tolist(),
            "comparison": mdsc_cmp.to_dict(),
        },
        "gate": overall,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("PAPR-SSL P2-06B NATIVE BATCH VS REFERENCE")
    print("=" * 88)
    print(f"Backbone: {cfg['kind']}")
    print(f"Device:   {device}")
    print(f"GSC:      {gsc_cmp.gate}")
    for row in gsc_cmp.samples:
        print(
            f"  sample={row.index} frames={row.valid_frames_native}/"
            f"{row.valid_frames_reference} "
            f"frame_cos={row.mean_frame_cosine} "
            f"pooled_cos={row.pooled_cosine} "
            f"mae={row.mean_abs_error} max={row.max_abs_error}"
        )
    print(f"MDSC:     {mdsc_cmp.gate}")
    for row in mdsc_cmp.samples:
        print(
            f"  sample={row.index} frames={row.valid_frames_native}/"
            f"{row.valid_frames_reference} "
            f"frame_cos={row.mean_frame_cosine} "
            f"pooled_cos={row.pooled_cosine} "
            f"mae={row.mean_abs_error} max={row.max_abs_error}"
        )
    print(f"Gate:     {overall}")
    print(f"JSON:     {args.json_out}")
    print("=" * 88)

    # Deliberately return zero even for a numeric FAIL so all three backbones
    # can be benchmarked in sequence; the JSON/printed Gate is authoritative.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
