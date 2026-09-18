#!/usr/bin/env python
"""P2-06A real length-aware SSL smoke for one configured backbone.

This test uses real GSC/MDSC examples, but intentionally uses the
correctness-first singleton-trimmed adapter rather than native padded batching.
"""

from __future__ import annotations

import argparse
import inspect
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
from papr_ssl.models.teacher.backbones.length_aware import LengthAwareSSLAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
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
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError("Config root must be a mapping")
    return data


def extract_backbone_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("backbone"), Mapping):
        return dict(config["backbone"])
    teacher = config.get("teacher")
    if isinstance(teacher, Mapping) and isinstance(teacher.get("backbone"), Mapping):
        return dict(teacher["backbone"])
    raise KeyError("Could not locate backbone config")


def create_backbone(cfg: dict[str, Any]):
    from papr_ssl.models.teacher.backbones.factory import create_ssl_backbone
    kind = cfg["kind"]
    kwargs = {k: v for k, v in cfg.items() if k != "kind"}
    return create_ssl_backbone(kind, **kwargs)


def pick_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_gsc(records):
    short = min(records, key=lambda r: r.num_frames)
    full = next(r for r in records if r.num_frames == 16000)
    return [short, full]


def pick_mdsc(records):
    # Keep the real-model smoke bounded on a laptop: use two different,
    # relatively short utterances rather than the 14.7-second maximum.
    candidates = sorted(
        [r for r in records if 14000 <= r.num_frames <= 40000],
        key=lambda r: r.num_frames,
    )
    if len(candidates) < 2:
        raise RuntimeError("Could not find two bounded MDSC smoke examples")
    first = candidates[0]
    second = next(
        r for r in reversed(candidates)
        if r.num_frames != first.num_frames
    )
    return [first, second]


def output_summary(out) -> dict[str, Any]:
    return {
        "features_shape": list(out.features.shape),
        "frame_mask_shape": list(out.frame_mask.shape),
        "valid_frames_per_sample": [
            int(v) for v in out.frame_mask.sum(dim=1).cpu().tolist()
        ],
        "hidden_layer": int(out.hidden_layer),
        "finite": bool(torch.isfinite(out.features).all().item()),
    }


def run_dataset(
    adapter,
    *,
    records,
    dataset_name: str,
    root: Path,
    device: torch.device,
):
    examples = [
        load_audio_example(r, {dataset_name: root})
        for r in records
    ]
    batch = collate_audio_examples(examples).to(device)

    with torch.inference_mode():
        out = adapter.forward_audio_batch(batch)

    summary = {
        "waveform_shape": list(batch.waveforms.shape),
        "waveform_lengths": batch.lengths.cpu().tolist(),
        "output": output_summary(out),
    }

    valid = summary["output"]["valid_frames_per_sample"]
    if len(set(batch.lengths.cpu().tolist())) > 1 and len(set(valid)) <= 1:
        raise ValueError(
            "Different waveform lengths produced identical valid frame counts; "
            "length-aware propagation did not take effect"
        )
    if not summary["output"]["finite"]:
        raise ValueError("Non-finite SSL features")

    return summary


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)

    cfg = load_yaml(args.config)
    backbone_cfg = extract_backbone_cfg(cfg)
    backbone = create_backbone(backbone_cfg).to(device).eval()
    adapter = LengthAwareSSLAdapter(backbone).to(device).eval()

    gsc_records = read_manifest_jsonl(args.gsc_manifest)
    mdsc_records = read_manifest_jsonl(args.mdsc_manifest)

    gsc = run_dataset(
        adapter,
        records=pick_gsc(gsc_records),
        dataset_name="gsc_v2",
        root=args.gsc_root,
        device=device,
    )
    mdsc = run_dataset(
        adapter,
        records=pick_mdsc(mdsc_records),
        dataset_name="mdsc",
        root=args.mdsc_root,
        device=device,
    )

    result = {
        "phase": "P2-06A",
        "mode": "correctness_first_exact_length_singleton_forward",
        "config": str(args.config),
        "kind": backbone_cfg["kind"],
        "device": str(device),
        "gsc_v2": gsc,
        "mdsc": mdsc,
        "gate": "PASS",
    }

    print("=" * 88)
    print("PAPR-SSL P2-06A LENGTH-AWARE SSL REAL SMOKE")
    print("=" * 88)
    print(f"Backbone: {backbone_cfg['kind']}")
    print(f"Device:   {device}")
    print(f"GSC:      {gsc}")
    print(f"MDSC:     {mdsc}")
    print("Gate:     PASS")
    print("=" * 88)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON:     {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
