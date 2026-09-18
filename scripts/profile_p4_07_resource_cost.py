#!/usr/bin/env python
"""P4-07: profile model scale and inference resource cost.

Purpose:
- record resource cost for the three P4-06 per-Backbone winners
- DO NOT re-rank or replace the P4-06 Teacher winner
- generic_test remains sealed

Benchmark protocol:
- deterministic 1.0 s / 16 kHz waveform
- batch size = 1
- torch.float32
- CUDA synchronization around timed model forward
- model-forward latency excludes disk I/O
- W2v-BERT model-forward latency excludes its CPU feature extractor;
  an additional waveform-to-hidden end-to-end latency is also reported
- selected hidden state is read from the full standard HF forward;
  no early-exit/truncated-encoder optimization is assumed
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2Model,
    WavLMModel,
    Wav2Vec2BertModel,
)

from papr_ssl.cache.ssl_feature_cache import SSLCacheIdentity
from papr_ssl.training.teacher.p4_candidates import get_p4_backbone


SAMPLE_RATE = 16000
DTYPE = torch.float32

MODEL_CLASS = {
    "wav2vec2_base": Wav2Vec2Model,
    "wavlm_large": WavLMModel,
    "w2v_bert2": Wav2Vec2BertModel,
}


def load_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def tensor_bytes(parameters) -> int:
    return sum(
        int(p.numel()) * int(p.element_size())
        for p in parameters
    )


def directory_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file()
    )


def make_waveform(duration_sec: float) -> torch.Tensor:
    n = int(round(SAMPLE_RATE * duration_sec))
    if n <= 0:
        raise ValueError("duration must produce at least one sample")
    # Deterministic non-silent waveform. Compute shape/cost is the point;
    # benchmark does not depend on semantic content.
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    waveform = (
        0.12 * torch.sin(2.0 * math.pi * 220.0 * t)
        + 0.04 * torch.sin(2.0 * math.pi * 440.0 * t)
    )
    return waveform.unsqueeze(0).contiguous()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_raw_waveform_model(
    *,
    model,
    layer: int,
    waveform_cpu: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict, tuple[int, ...]]:
    waveform_device = waveform_cpu.to(device=device, dtype=DTYPE)
    attention_mask = torch.ones_like(
        waveform_device,
        dtype=torch.long,
        device=device,
    )

    def forward_preloaded():
        out = model(
            input_values=waveform_device,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("model did not return hidden_states")
        return hs[layer]

    for _ in range(warmup):
        _ = forward_preloaded()
    synchronize(device)

    if device.type == "cuda":
        resident = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        resident = 0

    lat_ms = []
    selected = None
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        selected = forward_preloaded()
        synchronize(device)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    if selected is None:
        raise RuntimeError("no timed forward executed")

    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )

    # End-to-end here means CPU waveform -> device transfer -> model -> selected
    # hidden state. No audio decode / disk I/O.
    e2e_ms = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        x = waveform_cpu.to(device=device, dtype=DTYPE)
        m = torch.ones_like(x, dtype=torch.long, device=device)
        out = model(
            input_values=x,
            attention_mask=m,
            output_hidden_states=True,
            return_dict=True,
        )
        _ = out.hidden_states[layer]
        synchronize(device)
        e2e_ms.append((time.perf_counter() - t0) * 1000.0)

    stats = {
        "model_forward_latency_ms": {
            "median": float(statistics.median(lat_ms)),
            "mean": float(statistics.mean(lat_ms)),
            "p95": float(quantile(lat_ms, 0.95)),
            "min": float(min(lat_ms)),
            "max": float(max(lat_ms)),
            "repeats": repeats,
        },
        "waveform_to_hidden_latency_ms": {
            "median": float(statistics.median(e2e_ms)),
            "mean": float(statistics.mean(e2e_ms)),
            "p95": float(quantile(e2e_ms, 0.95)),
            "min": float(min(e2e_ms)),
            "max": float(max(e2e_ms)),
            "repeats": repeats,
            "includes_cpu_feature_extractor": False,
        },
        "cuda_memory": {
            "resident_allocated_bytes_before_timed_forward": resident,
            "peak_allocated_bytes_during_timed_forward": peak,
            "forward_peak_extra_bytes": max(0, peak - resident),
        },
    }
    return stats, tuple(int(x) for x in selected.shape)


@torch.inference_mode()
def benchmark_w2vbert(
    *,
    model,
    feature_extractor,
    layer: int,
    waveform_cpu: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict, tuple[int, ...], tuple[int, ...]]:
    raw_np = waveform_cpu.squeeze(0).numpy()

    frontend = feature_extractor(
        [raw_np],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_features_cpu = frontend["input_features"].to(torch.float32)
    attention_mask_cpu = frontend["attention_mask"].to(torch.long)

    input_features = input_features_cpu.to(device)
    attention_mask = attention_mask_cpu.to(device)

    def forward_preloaded():
        out = model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("W2v-BERT did not return hidden_states")
        return hs[layer]

    for _ in range(warmup):
        _ = forward_preloaded()
    synchronize(device)

    if device.type == "cuda":
        resident = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        resident = 0

    lat_ms = []
    selected = None
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        selected = forward_preloaded()
        synchronize(device)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    if selected is None:
        raise RuntimeError("no timed W2v-BERT forward executed")

    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )

    # End-to-end: waveform ndarray -> CPU acoustic frontend -> H2D -> model.
    e2e_ms = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        front = feature_extractor(
            [raw_np],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        feats = front["input_features"].to(
            device=device,
            dtype=torch.float32,
        )
        mask = front["attention_mask"].to(
            device=device,
            dtype=torch.long,
        )
        out = model(
            input_features=feats,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        _ = out.hidden_states[layer]
        synchronize(device)
        e2e_ms.append((time.perf_counter() - t0) * 1000.0)

    stats = {
        "model_forward_latency_ms": {
            "median": float(statistics.median(lat_ms)),
            "mean": float(statistics.mean(lat_ms)),
            "p95": float(quantile(lat_ms, 0.95)),
            "min": float(min(lat_ms)),
            "max": float(max(lat_ms)),
            "repeats": repeats,
        },
        "waveform_to_hidden_latency_ms": {
            "median": float(statistics.median(e2e_ms)),
            "mean": float(statistics.mean(e2e_ms)),
            "p95": float(quantile(e2e_ms, 0.95)),
            "min": float(min(e2e_ms)),
            "max": float(max(e2e_ms)),
            "repeats": repeats,
            "includes_cpu_feature_extractor": True,
        },
        "cuda_memory": {
            "resident_allocated_bytes_before_timed_forward": resident,
            "peak_allocated_bytes_during_timed_forward": peak,
            "forward_peak_extra_bytes": max(0, peak - resident),
        },
    }
    return (
        stats,
        tuple(int(x) for x in selected.shape),
        tuple(int(x) for x in input_features_cpu.shape),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--selection-json",
        type=Path,
        default=Path(
            "artifacts/p4_runs/p4_06_selection/"
            "p4_06_selection.json"
        ),
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/ssl_feature_cache"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p4_runs/p4_07_resource_cost"),
    )
    p.add_argument("--duration-sec", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("invalid warmup/repeats")
    if args.duration_sec <= 0:
        raise ValueError("duration-sec must be > 0")

    selection = load_json(args.selection_json)
    if selection.get("phase") != "P4-06":
        raise RuntimeError("P4-06 selection JSON required")
    if selection.get("generic_test") != "sealed_not_accessed":
        raise RuntimeError("generic_test seal violated")
    if selection.get("resource_cost_used_for_selection") is not False:
        raise RuntimeError(
            "P4-06 selection was unexpectedly resource-based"
        )

    per_backbone = selection["per_backbone_winners"]
    selected_mainline = selection["selected_mainline"]

    device = torch.device(args.device)
    waveform_cpu = make_waveform(args.duration_sec)

    records = []

    for selected in per_backbone:
        backbone = selected["backbone"]
        layer = int(selected["hidden_state_index"])
        spec = get_p4_backbone(backbone)

        print("=" * 108)
        print(
            f"P4-07 PROFILING {backbone} "
            f"hidden_states[{layer}]"
        )
        print("=" * 108)

        cleanup(device)

        model_cls = MODEL_CLASS[backbone]
        model = model_cls.from_pretrained(
            spec.model_id,
            revision=spec.model_revision,
            output_hidden_states=True,
        )
        model.eval()
        model.to(device=device, dtype=DTYPE)

        total_params = sum(int(p.numel()) for p in model.parameters())
        trainable_params = sum(
            int(p.numel())
            for p in model.parameters()
            if p.requires_grad
        )
        param_bytes = tensor_bytes(model.parameters())

        cache_identity = SSLCacheIdentity(
            model_id=spec.model_id,
            model_revision=spec.model_revision,
            layer=layer,
        )
        cache_dir = args.cache_root / cache_identity.slug
        cache_bytes = directory_size_bytes(cache_dir)

        frontend_shape = None
        if backbone == "w2v_bert2":
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                spec.model_id,
                revision=spec.model_revision,
            )
            bench, hidden_shape, frontend_shape = benchmark_w2vbert(
                model=model,
                feature_extractor=feature_extractor,
                layer=layer,
                waveform_cpu=waveform_cpu,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            input_type = "input_features+attention_mask"
            frontend = "HF AutoFeatureExtractor on CPU"
        else:
            bench, hidden_shape = benchmark_raw_waveform_model(
                model=model,
                layer=layer,
                waveform_cpu=waveform_cpu,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            input_type = "raw_waveform+attention_mask"
            frontend = "model-native raw waveform"

        if hidden_shape[-1] != int(spec.embedding_dim):
            raise RuntimeError(
                f"{backbone}: hidden dim mismatch "
                f"{hidden_shape[-1]} != {spec.embedding_dim}"
            )

        record = {
            "backbone": backbone,
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
            "selected_hidden_state_index": layer,
            "p4_06_generic_dev_score_mean": float(
                selected["generic_dev_score_mean"]
            ),
            "p4_06_generic_dev_score_sample_std": float(
                selected["generic_dev_score_sample_std"]
            ),
            "parameter_count_total": total_params,
            "parameter_count_trainable_before_freeze": trainable_params,
            "parameter_storage_bytes_float32": param_bytes,
            "hidden_dim": int(spec.embedding_dim),
            "benchmark_input": {
                "sample_rate_hz": SAMPLE_RATE,
                "duration_sec": args.duration_sec,
                "batch_size": 1,
                "dtype": "float32",
                "input_type": input_type,
                "frontend": frontend,
            },
            "selected_hidden_shape": list(hidden_shape),
            "w2vbert_frontend_shape": (
                list(frontend_shape)
                if frontend_shape is not None
                else None
            ),
            "standard_hf_full_forward": True,
            "early_exit_or_encoder_truncation_used": False,
            "benchmark": bench,
            "selected_masked_mean_cache_directory": cache_dir.as_posix(),
            "selected_masked_mean_cache_bytes": cache_bytes,
        }
        records.append(record)

        print(f"parameters:       {total_params:,}")
        print(
            "float32 param MB: "
            f"{param_bytes / (1024**2):.2f}"
        )
        print(
            "model fwd median: "
            f"{bench['model_forward_latency_ms']['median']:.3f} ms"
        )
        print(
            "waveform->hidden: "
            f"{bench['waveform_to_hidden_latency_ms']['median']:.3f} ms"
        )
        if device.type == "cuda":
            mem = bench["cuda_memory"]
            print(
                "resident alloc:   "
                f"{mem['resident_allocated_bytes_before_timed_forward'] / (1024**2):.2f} MB"
            )
            print(
                "peak alloc:       "
                f"{mem['peak_allocated_bytes_during_timed_forward'] / (1024**2):.2f} MB"
            )
            print(
                "forward extra:    "
                f"{mem['forward_peak_extra_bytes'] / (1024**2):.2f} MB"
            )

        del model
        if backbone == "w2v_bert2":
            del feature_extractor
        cleanup(device)

    output = {
        "schema": "papr_ssl.p4_07_resource_cost.v1",
        "phase": "P4-07",
        "purpose": "record_resource_cost_only",
        "teacher_selection_source": args.selection_json.as_posix(),
        "teacher_selection_basis": (
            "P4-06 embedding quality: 3-seed mean generic_dev_score"
        ),
        "teacher_selection_changed_by_resource_cost": False,
        "selected_mainline_before_resource_profiling": {
            "backbone": selected_mainline["backbone"],
            "hidden_state_index": int(
                selected_mainline["hidden_state_index"]
            ),
            "generic_dev_score_mean": float(
                selected_mainline["generic_dev_score_mean"]
            ),
        },
        "benchmark_protocol": {
            "sample_rate_hz": SAMPLE_RATE,
            "duration_sec": args.duration_sec,
            "batch_size": 1,
            "dtype": "float32",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "full_standard_hf_forward": True,
            "early_exit_or_encoder_truncation": False,
            "disk_io_included": False,
        },
        "records": records,
        "ranking_by_resource_cost_performed": False,
        "generic_test": "sealed_not_accessed",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "p4_07_resource_cost.json"
    csv_path = args.output_dir / "p4_07_resource_cost.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            "P4-07 output already exists; refusing silent overwrite"
        )

    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "backbone",
            "selected_hidden_state_index",
            "generic_dev_score_mean",
            "parameter_count_total",
            "parameter_storage_mb_float32",
            "hidden_dim",
            "model_forward_median_ms",
            "waveform_to_hidden_median_ms",
            "cuda_resident_allocated_mb",
            "cuda_peak_allocated_mb",
            "cuda_forward_peak_extra_mb",
            "selected_masked_mean_cache_mb",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            mem = r["benchmark"]["cuda_memory"]
            cache_bytes = r["selected_masked_mean_cache_bytes"]
            w.writerow(
                {
                    "backbone": r["backbone"],
                    "selected_hidden_state_index": r[
                        "selected_hidden_state_index"
                    ],
                    "generic_dev_score_mean": r[
                        "p4_06_generic_dev_score_mean"
                    ],
                    "parameter_count_total": r[
                        "parameter_count_total"
                    ],
                    "parameter_storage_mb_float32": (
                        r["parameter_storage_bytes_float32"]
                        / (1024**2)
                    ),
                    "hidden_dim": r["hidden_dim"],
                    "model_forward_median_ms": r["benchmark"][
                        "model_forward_latency_ms"
                    ]["median"],
                    "waveform_to_hidden_median_ms": r["benchmark"][
                        "waveform_to_hidden_latency_ms"
                    ]["median"],
                    "cuda_resident_allocated_mb": (
                        mem[
                            "resident_allocated_bytes_before_timed_forward"
                        ]
                        / (1024**2)
                    ),
                    "cuda_peak_allocated_mb": (
                        mem[
                            "peak_allocated_bytes_during_timed_forward"
                        ]
                        / (1024**2)
                    ),
                    "cuda_forward_peak_extra_mb": (
                        mem["forward_peak_extra_bytes"]
                        / (1024**2)
                    ),
                    "selected_masked_mean_cache_mb": (
                        None
                        if cache_bytes is None
                        else cache_bytes / (1024**2)
                    ),
                }
            )

    print("=" * 108)
    print("P4-07 RESOURCE COST RECORDING COMPLETE")
    print("=" * 108)
    print(
        "Teacher selection remains: "
        f"{selected_mainline['backbone']} "
        f"hidden_states[{selected_mainline['hidden_state_index']}]"
    )
    print("teacher selection changed by resource cost: NO")
    print("ranking by resource cost performed:         NO")
    print("generic_test accessed:                      NO")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("-" * 108)
    print("P4-07 STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
