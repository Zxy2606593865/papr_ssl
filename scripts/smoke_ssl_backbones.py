#!/usr/bin/env python
"""
PAPR-SSL P1-03 through P1-09 smoke CLI.

Purpose
-------
Load one configured SSL backbone, feed a deterministic 1-second / 16 kHz
waveform through the existing PAPR-SSL wrapper, validate the common
SSLBackboneOutput contract, directly observe the underlying Hugging Face
hidden_states tuple, report repeatable timing / CUDA memory statistics, and
validate batch / padding / frame-mask behavior for P1-07, and validate
candidate hidden-state indices for P1-08, and run formal CPU/GPU resource
benchmarking for P1-09.

This script does NOT train the model, does NOT compute SCAF, and does NOT touch
Attention DR, KD, Student, or dataset code.

Supported targets: Wav2Vec2, WavLM, and W2v-BERT 2.0.
The CLI remains shared; model-specific frontend behavior stays inside each wrapper.
"""

from __future__ import annotations

import argparse
import ctypes
import inspect
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PAPR-SSL real-checkpoint smoke test for SSL backbones."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Teacher/backbone YAML config path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. Default: auto.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=440.0,
        help="Frequency of deterministic 1-second sine-wave smoke input.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.05,
        help="Amplitude of deterministic smoke waveform.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warm-up forwards before timing. Default: 3.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of timed forwards. Default: 10.",
    )
    parser.add_argument(
        "--batch-mask-test",
        action="store_true",
        help=(
            "Run P1-07 singleton, multi-sample batch, and (when the wrapper "
            "contract exposes validity metadata) true padded-batch mask checks."
        ),
    )
    parser.add_argument(
        "--layer-check",
        action="store_true",
        help=(
            "Run P1-08 candidate hidden-state index validation using the real "
            "read-only forward already captured by this smoke run."
        ),
    )
    parser.add_argument(
        "--resource-benchmark",
        action="store_true",
        help=(
            "Run P1-09 formal resource benchmarking. Use a fresh process per "
            "model/device and pass a fixed --warmup/--runs protocol."
        ),
    )
    parser.add_argument(
        "--layer-sweep-config",
        type=Path,
        default=None,
        help=(
            "Optional path to the real layer-sweep YAML consumed by "
            "papr_ssl.experiments.layer_sweep.load_layer_sweep(). If omitted, "
            "the CLI searches configs/**/*.yaml|yml for a unique sweep payload."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional explicit P1-08 candidate hidden-state indices, e.g. "
            "--layers 0 4 8 12 13. If omitted, the script first looks for "
            "candidate-layer fields in config/layer_sweep metadata and finally "
            "falls back to evenly spaced structural probes plus one invalid index."
        ),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got: {type(data).__name__}")
    return data


def _looks_like_backbone_cfg(value: Mapping[str, Any]) -> bool:
    keys = {str(k).lower() for k in value.keys()}
    return bool(
        keys
        & {
            "kind",
            "backbone",
            "model_name",
            "model_name_or_path",
            "checkpoint",
            "hidden_layer",
        }
    )


def extract_backbone_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    direct = config.get("backbone")
    if isinstance(direct, Mapping):
        return dict(direct)

    teacher = config.get("teacher")
    if isinstance(teacher, Mapping):
        nested = teacher.get("backbone")
        if isinstance(nested, Mapping):
            return dict(nested)

    if _looks_like_backbone_cfg(config):
        return dict(config)

    raise KeyError(
        "Could not locate backbone config. Expected `backbone: {...}`, "
        "`teacher.backbone: {...}`, or a root-level backbone mapping."
    )


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_smoke_waveform(
    *,
    device: torch.device,
    frequency_hz: float,
    amplitude: float,
) -> torch.Tensor:
    if not (0.0 < amplitude <= 1.0):
        raise ValueError(f"amplitude must be in (0, 1], got {amplitude}")
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be > 0, got {frequency_hz}")

    t = torch.arange(WINDOW_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    waveform = amplitude * torch.sin(2.0 * math.pi * frequency_hz * t)
    return waveform.unsqueeze(0).to(device=device, dtype=torch.float32)


def _import_factory():
    try:
        from papr_ssl.models.teacher.backbones.factory import create_ssl_backbone
    except ImportError as exc:
        raise RuntimeError(
            "Could not import "
            "`papr_ssl.models.teacher.backbones.factory.create_ssl_backbone`. "
            "Activate the `papr_ssl` Conda environment and install the project "
            "with `python -m pip install -e . --no-deps`."
        ) from exc
    return create_ssl_backbone


def create_backbone(backbone_cfg: dict[str, Any]):
    factory = _import_factory()
    sig = inspect.signature(factory)
    params = list(sig.parameters.values())

    # Current PAPR-SSL factory is expected to be create_ssl_backbone(kind, **kwargs).
    if params and params[0].name == "kind":
        if "kind" not in backbone_cfg:
            raise KeyError("Backbone config must contain `kind`.")
        kind = backbone_cfg["kind"]
        kwargs = {k: v for k, v in backbone_cfg.items() if k != "kind"}
        return factory(kind, **kwargs)

    # Fallback for a one-mapping factory.
    positional = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) == 1 and positional[0].name not in backbone_cfg:
        return factory(backbone_cfg)

    # Fallback for keyword-driven factory.
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    if accepts_kwargs:
        return factory(**backbone_cfg)

    required_names = {
        p.name
        for p in params
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    if required_names.issubset(backbone_cfg.keys()):
        kwargs = {
            p.name: backbone_cfg[p.name]
            for p in params
            if p.name in backbone_cfg
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        return factory(**kwargs)

    raise TypeError(
        "The existing create_ssl_backbone factory signature does not match "
        "the smoke CLI's expected config-driven interface.\n"
        f"Factory signature: {sig}\n"
        f"Backbone config keys: {sorted(backbone_cfg.keys())}"
    )


def move_backbone_to_device(backbone: Any, device: torch.device) -> None:
    if hasattr(backbone, "to"):
        backbone.to(device)
    if hasattr(backbone, "eval"):
        backbone.eval()


def _find_underlying_model(backbone: Any) -> tuple[str | None, Any | None]:
    for name in ("model", "_model", "ssl_model", "backbone"):
        value = getattr(backbone, name, None)
        if value is not None and value is not backbone:
            return name, value
    return None, None


def model_structure_info(backbone: Any) -> dict[str, Any]:
    attr_name, model = _find_underlying_model(backbone)
    if model is None:
        return {
            "underlying_model_attr": None,
            "num_hidden_layers": None,
            "hidden_size": None,
            "expected_hidden_states_count": None,
        }

    config = getattr(model, "config", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    hidden_size = getattr(config, "hidden_size", None)

    expected_count = None
    if isinstance(num_hidden_layers, int):
        expected_count = num_hidden_layers + 1

    return {
        "underlying_model_attr": attr_name,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "expected_hidden_states_count": expected_count,
    }


def _tensor_meta(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    meta = {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "finite": bool(torch.isfinite(value).all().item()) if value.is_floating_point() else None,
    }
    if value.ndim == 2 and not value.is_floating_point():
        meta["nonzero_per_sample"] = [
            int(v) for v in (value != 0).sum(dim=1).detach().cpu().tolist()
        ]
    return meta


def observe_model_io_and_hidden_states(
    backbone: Any,
    waveform: torch.Tensor,
    *,
    backbone_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """
    Observe the real underlying HF model without bypassing the PAPR wrapper.

    We attach:
      - a forward-pre-hook to capture what the wrapper actually sends to the
        model (`input_values`, `input_features`, `attention_mask`, etc.);
      - a forward-hook to observe the real `outputs.hidden_states`.

    This is especially important for W2v-BERT 2.0, whose wrapper must perform:
        waveform -> feature extractor -> input_features + attention_mask -> model
    """
    _, model = _find_underlying_model(backbone)
    if model is None or not hasattr(model, "register_forward_hook"):
        raise RuntimeError(
            "Could not locate an underlying torch module for model I/O observation."
        )

    captured_hidden: dict[str, Any] = {}
    captured_inputs: dict[str, Any] = {
        "positional_args": [],
        "keyword_tensors": {},
    }

    def pre_hook(_module, args, kwargs):
        captured_inputs["positional_args"] = [
            _tensor_meta(v) for v in args
        ]
        for key, value in kwargs.items():
            meta = _tensor_meta(value)
            if meta is not None:
                captured_inputs["keyword_tensors"][key] = meta

    def post_hook(_module, _inputs, output):
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            captured_hidden["hidden_states_count"] = None
            captured_hidden["hidden_states_shapes"] = None
            return

        captured_hidden["hidden_states_count"] = len(hidden_states)
        captured_hidden["hidden_states_shapes"] = [
            list(x.shape) if isinstance(x, torch.Tensor) else None
            for x in hidden_states
        ]

    pre_handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = model.register_forward_hook(post_hook)

    call_kwargs = dict(backbone_kwargs or {})
    try:
        with torch.inference_mode():
            wrapper_output = backbone(waveform, **call_kwargs)
    finally:
        pre_handle.remove()
        post_handle.remove()

    if "hidden_states_count" not in captured_hidden:
        raise RuntimeError(
            "Underlying model forward hook fired without exposing hidden-state metadata."
        )
    if captured_hidden["hidden_states_count"] is None:
        raise RuntimeError(
            "Underlying model output did not contain `hidden_states`. "
            "The wrapper/model must run with output_hidden_states=True."
        )

    return wrapper_output, captured_hidden, captured_inputs


def validate_model_forward_inputs(
    *,
    kind: str,
    model_inputs: Mapping[str, Any],
    output_info: Mapping[str, Any],
) -> dict[str, Any]:
    keyword_tensors = model_inputs.get("keyword_tensors", {})
    positional_args = model_inputs.get("positional_args", [])

    result = {
        "keyword_tensors": dict(keyword_tensors),
        "positional_args": list(positional_args),
        "frontend_contract": "generic",
        "frontend_alignment_pass": True,
    }

    if kind in {"w2vbert2", "wav2vec2bert"}:
        result["frontend_contract"] = "input_features + attention_mask"

        input_features = keyword_tensors.get("input_features")
        attention_mask = keyword_tensors.get("attention_mask")

        if input_features is None:
            raise RuntimeError(
                "W2v-BERT 2.0 model forward did not receive keyword `input_features`."
            )
        if attention_mask is None:
            raise RuntimeError(
                "W2v-BERT 2.0 model forward did not receive keyword `attention_mask`."
            )

        feat_shape = input_features["shape"]
        mask_shape = attention_mask["shape"]

        if len(feat_shape) != 3:
            raise ValueError(
                f"W2v-BERT input_features must be [B,T,F], got {feat_shape}"
            )
        if len(mask_shape) != 2:
            raise ValueError(
                f"W2v-BERT attention_mask must be [B,T], got {mask_shape}"
            )
        if feat_shape[:2] != mask_shape:
            raise ValueError(
                "W2v-BERT feature/mask frame axes are misaligned: "
                f"input_features[:2]={feat_shape[:2]}, attention_mask={mask_shape}"
            )

        wrapper_mask_shape = output_info["frame_mask_shape"]
        if mask_shape != wrapper_mask_shape:
            raise ValueError(
                "W2v-BERT model attention_mask and PAPR frame_mask differ: "
                f"model={mask_shape}, wrapper={wrapper_mask_shape}"
            )

        if feat_shape[1] != output_info["features_shape"][1]:
            raise ValueError(
                "W2v-BERT input feature frame axis and selected hidden-state "
                "frame axis differ: "
                f"input_features T={feat_shape[1]}, "
                f"hidden T={output_info['features_shape'][1]}"
            )

        model_valid = attention_mask.get("nonzero_per_sample")
        wrapper_valid = output_info.get("valid_frames_per_sample")
        if model_valid is not None and wrapper_valid is not None and model_valid != wrapper_valid:
            raise ValueError(
                "W2v-BERT attention_mask values and PAPR frame_mask values differ: "
                f"model valid frames={model_valid}, wrapper valid frames={wrapper_valid}"
            )

    return result


def validate_ssl_backbone_output(output: Any) -> dict[str, Any]:
    missing = [
        name
        for name in ("features", "frame_mask", "hidden_layer")
        if not hasattr(output, name)
    ]
    if missing:
        raise TypeError(
            "Backbone output does not satisfy SSLBackboneOutput contract; "
            f"missing attributes: {missing}"
        )

    features = output.features
    frame_mask = output.frame_mask
    hidden_layer = output.hidden_layer

    if not isinstance(features, torch.Tensor):
        raise TypeError("output.features must be a torch.Tensor")
    if not isinstance(frame_mask, torch.Tensor):
        raise TypeError("output.frame_mask must be a torch.Tensor")

    if features.ndim != 3:
        raise ValueError(
            f"features must have shape [B,T,D], got {tuple(features.shape)}"
        )
    if frame_mask.ndim != 2:
        raise ValueError(
            f"frame_mask must have shape [B,T], got {tuple(frame_mask.shape)}"
        )
    if tuple(frame_mask.shape) != tuple(features.shape[:2]):
        raise ValueError(
            "frame_mask is not aligned with feature frames: "
            f"features[:2]={tuple(features.shape[:2])}, "
            f"mask={tuple(frame_mask.shape)}"
        )
    if features.dtype != torch.float32:
        raise TypeError(
            f"features must be float32 by contract, got {features.dtype}"
        )
    if frame_mask.dtype != torch.bool:
        raise TypeError(
            f"frame_mask must be bool by contract, got {frame_mask.dtype}"
        )
    if not torch.isfinite(features).all():
        raise ValueError("features contain NaN or Inf")
    if not frame_mask.any(dim=1).all():
        raise ValueError("At least one sample has zero valid feature frames")

    return {
        "features_shape": list(features.shape),
        "frame_mask_shape": list(frame_mask.shape),
        "features_dtype": str(features.dtype).replace("torch.", ""),
        "frame_mask_dtype": str(frame_mask.dtype).replace("torch.", ""),
        "features_finite": True,
        "hidden_layer": int(hidden_layer),
        "valid_frames_per_sample": [
            int(v) for v in frame_mask.sum(dim=1).detach().cpu().tolist()
        ],
    }


def validate_hidden_state_observation(
    *,
    output_info: Mapping[str, Any],
    raw_info: Mapping[str, Any],
) -> dict[str, Any]:
    count = raw_info["hidden_states_count"]
    shapes = raw_info["hidden_states_shapes"]
    selected = int(output_info["hidden_layer"])

    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"Invalid observed hidden_states count: {count}")
    if not isinstance(shapes, list) or len(shapes) != count:
        raise ValueError("Observed hidden-state shape list is inconsistent.")
    if selected < 0 or selected >= count:
        raise IndexError(
            f"Selected hidden_layer={selected} is outside observed range [0, {count - 1}]"
        )

    selected_raw_shape = shapes[selected]
    wrapper_shape = output_info["features_shape"]

    if selected_raw_shape != wrapper_shape:
        raise ValueError(
            "Wrapper features do not match the selected raw hidden state: "
            f"raw hidden_states[{selected}]={selected_raw_shape}, "
            f"wrapper features={wrapper_shape}"
        )

    return {
        "observed_hidden_states_count": count,
        "valid_index_range": [0, count - 1],
        "selected_raw_hidden_state_shape": selected_raw_shape,
        "all_hidden_state_shapes": shapes,
        "selected_matches_wrapper_features": True,
    }



def _forward_signature(backbone: Any) -> inspect.Signature:
    target = getattr(backbone, "forward", backbone)
    return inspect.signature(target)


def detect_validity_contract(backbone: Any) -> dict[str, str] | None:
    """Return the explicit waveform-validity input exposed by the wrapper, if any.

    We intentionally do not guess `padding_mask` semantics because libraries disagree
    on whether True means valid or padded. P1-07 must not manufacture a passing mask.
    """
    params = _forward_signature(backbone).parameters

    for name in ("waveform_lengths", "input_lengths", "valid_lengths", "lengths"):
        if name in params:
            return {"mode": "lengths", "parameter": name}

    for name in ("waveform_mask", "input_mask", "attention_mask"):
        if name in params:
            return {"mode": "valid_mask", "parameter": name}

    return None


def build_full_length_batch(
    *,
    device: torch.device,
    amplitude: float,
) -> torch.Tensor:
    frequencies = (440.0, 660.0, 880.0)
    rows = [
        build_smoke_waveform(
            device=device, frequency_hz=freq, amplitude=amplitude
        ).squeeze(0)
        for freq in frequencies
    ]
    return torch.stack(rows, dim=0)


def build_true_padded_batch(
    *,
    device: torch.device,
    amplitude: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor(
        [WINDOW_SAMPLES, 12_000, 8_000], dtype=torch.long, device=device
    )
    frequencies = (440.0, 660.0, 880.0)
    batch = torch.zeros(
        (len(frequencies), WINDOW_SAMPLES), dtype=torch.float32, device=device
    )

    for row, (freq, length) in enumerate(zip(frequencies, lengths.tolist())):
        time_axis = torch.arange(length, dtype=torch.float32, device=device) / SAMPLE_RATE
        batch[row, :length] = amplitude * torch.sin(
            2.0 * math.pi * freq * time_axis
        )

    sample_index = torch.arange(WINDOW_SAMPLES, device=device).unsqueeze(0)
    valid_mask = sample_index < lengths.unsqueeze(1)
    return batch, lengths, valid_mask


def validity_call_kwargs(
    *,
    contract: Mapping[str, str],
    lengths: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    name = contract["parameter"]
    mode = contract["mode"]
    if mode == "lengths":
        return {name: lengths}
    if mode == "valid_mask":
        value = valid_mask.to(dtype=torch.long) if name == "attention_mask" else valid_mask
        return {name: value}
    raise ValueError(f"Unknown validity contract mode: {mode}")


def validate_prefix_frame_mask(frame_mask: torch.Tensor) -> None:
    """Require each mask row to be a contiguous valid prefix followed by padding."""
    if frame_mask.dtype != torch.bool or frame_mask.ndim != 2:
        raise TypeError("frame_mask must be bool [B,T] for prefix validation")
    # A False -> True transition means padding becomes valid again later in the row.
    invalid_transition = (~frame_mask[:, :-1]) & frame_mask[:, 1:]
    if invalid_transition.any():
        raise ValueError("frame_mask is not prefix-contiguous for at least one sample")


def run_p1_07_batch_mask_test(
    *,
    backbone: Any,
    kind: str,
    singleton_output: Any,
    singleton_output_info: Mapping[str, Any],
    device: torch.device,
    amplitude: float,
) -> dict[str, Any]:
    """Validate P1-07 without confusing zero-valued audio with padding."""
    result: dict[str, Any] = {
        "singleton": {
            "status": "PASS",
            "features_shape": list(singleton_output.features.shape),
            "frame_mask_shape": list(singleton_output.frame_mask.shape),
            "valid_frames_per_sample": list(
                singleton_output_info["valid_frames_per_sample"]
            ),
        }
    }

    # 1) Real B>1 test with three independent full-length windows.
    full_batch = build_full_length_batch(device=device, amplitude=amplitude)
    full_output, full_hidden, full_model_inputs = observe_model_io_and_hidden_states(
        backbone, full_batch
    )
    full_info = validate_ssl_backbone_output(full_output)
    full_hidden_info = validate_hidden_state_observation(
        output_info=full_info, raw_info=full_hidden
    )
    full_frontend = validate_model_forward_inputs(
        kind=kind, model_inputs=full_model_inputs, output_info=full_info
    )

    if full_info["features_shape"][0] != full_batch.shape[0]:
        raise ValueError(
            "Batched features lost the batch dimension: "
            f"input B={full_batch.shape[0]}, output={full_info['features_shape']}"
        )
    if any(
        count != singleton_output_info["valid_frames_per_sample"][0]
        for count in full_info["valid_frames_per_sample"]
    ):
        raise ValueError(
            "Equal-length batch produced inconsistent valid frame counts: "
            f"singleton={singleton_output_info['valid_frames_per_sample']}, "
            f"batch={full_info['valid_frames_per_sample']}"
        )

    # Same 440 Hz first sample should be numerically stable when moved into a batch.
    reference = singleton_output.features[0].detach()
    batched_reference = full_output.features[0].detach()
    max_abs_diff = float((reference - batched_reference).abs().max().item())

    result["full_length_batch"] = {
        "status": "PASS",
        "input_shape": list(full_batch.shape),
        "features_shape": full_info["features_shape"],
        "frame_mask_shape": full_info["frame_mask_shape"],
        "valid_frames_per_sample": full_info["valid_frames_per_sample"],
        "hidden_states_count": full_hidden_info["observed_hidden_states_count"],
        "frontend_alignment_pass": full_frontend["frontend_alignment_pass"],
        "singleton_vs_batch_sample0_max_abs_diff": max_abs_diff,
    }

    # 2) True padded semantics only if the wrapper exposes validity metadata.
    validity_contract = detect_validity_contract(backbone)
    if validity_contract is None:
        result["padded_batch"] = {
            "status": "N/A",
            "reason": (
                "Current wrapper forward contract exposes waveform only and the "
                "project uses a fixed 1-second window contract. Zero-valued tail "
                "samples would therefore be audio/silence, not identifiable padding. "
                "P1-07 does not fabricate padding semantics."
            ),
            "forward_signature": str(_forward_signature(backbone)),
        }
        result["result"] = "PASS_WITH_PADDED_NA"
        return result

    padded_batch, lengths, valid_mask = build_true_padded_batch(
        device=device, amplitude=amplitude
    )
    call_kwargs = validity_call_kwargs(
        contract=validity_contract, lengths=lengths, valid_mask=valid_mask
    )
    padded_output, padded_hidden, padded_model_inputs = observe_model_io_and_hidden_states(
        backbone, padded_batch, backbone_kwargs=call_kwargs
    )
    padded_info = validate_ssl_backbone_output(padded_output)
    padded_hidden_info = validate_hidden_state_observation(
        output_info=padded_info, raw_info=padded_hidden
    )
    padded_frontend = validate_model_forward_inputs(
        kind=kind, model_inputs=padded_model_inputs, output_info=padded_info
    )
    validate_prefix_frame_mask(padded_output.frame_mask)

    valid_frames = padded_info["valid_frames_per_sample"]
    if not (valid_frames[0] >= valid_frames[1] >= valid_frames[2]):
        raise ValueError(
            "Shorter padded waveforms did not produce non-increasing valid-frame "
            f"counts: lengths={lengths.tolist()}, valid_frames={valid_frames}"
        )
    if valid_frames[0] == valid_frames[-1]:
        raise ValueError(
            "Explicitly shorter waveform produced the same number of valid frames; "
            "padding metadata is not affecting frame_mask."
        )

    result["padded_batch"] = {
        "status": "PASS",
        "validity_contract": dict(validity_contract),
        "waveform_lengths": [int(v) for v in lengths.detach().cpu().tolist()],
        "input_shape": list(padded_batch.shape),
        "features_shape": padded_info["features_shape"],
        "frame_mask_shape": padded_info["frame_mask_shape"],
        "valid_frames_per_sample": valid_frames,
        "prefix_contiguous_mask": True,
        "hidden_states_count": padded_hidden_info["observed_hidden_states_count"],
        "frontend_alignment_pass": padded_frontend["frontend_alignment_pass"],
    }
    result["result"] = "PASS"
    return result



def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _find_candidate_layers_in_config(
    config: Mapping[str, Any],
) -> tuple[list[int] | None, str | None]:
    """Find obvious candidate-layer lists in the loaded experiment config."""
    preferred_keys = {
        "candidate_layers",
        "layer_candidates",
        "candidate_hidden_layers",
        "hidden_layer_candidates",
        "sweep_layers",
        "layers",
    }

    def walk(
        node: Any,
        path: tuple[str, ...],
    ) -> tuple[list[int] | None, str | None]:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_s = str(key)
                key_l = key_s.lower()
                next_path = path + (key_s,)

                if key_l in preferred_keys and isinstance(value, (list, tuple)):
                    if value and all(
                        isinstance(v, int) and not isinstance(v, bool)
                        for v in value
                    ):
                        path_text = ".".join(p.lower() for p in next_path)
                        if key_l != "layers" or any(
                            token in path_text
                            for token in ("sweep", "candidate", "experiment")
                        ):
                            return _dedupe_ints(list(value)), ".".join(next_path)

                found, source = walk(value, next_path)
                if found is not None:
                    return found, source

        return None, None

    return walk(config, ())


def _looks_like_layer_sweep_payload(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    entries = payload.get("backbones")
    if not isinstance(entries, (list, tuple)):
        return False
    if not entries:
        return False
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        if "kind" not in entry or "base_config" not in entry or "hidden_layers" not in entry:
            return False
        layers = entry.get("hidden_layers")
        if not isinstance(layers, (list, tuple)):
            return False
    return True


def discover_layer_sweep_config(project_root: Path) -> Path | None:
    """
    Discover the real layer-sweep YAML from configs/.

    Preference:
      1. filenames containing both "layer" and "sweep";
      2. otherwise a unique YAML payload matching the layer_sweep schema.

    Ambiguity is an error rather than silently selecting the wrong experiment.
    """
    configs_dir = project_root / "configs"
    if not configs_dir.is_dir():
        return None

    matches: list[Path] = []
    for pattern in ("**/*.yaml", "**/*.yml"):
        for path in configs_dir.glob(pattern):
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = yaml.safe_load(f) or {}
            except Exception:
                continue
            if _looks_like_layer_sweep_payload(payload):
                matches.append(path.resolve())

    if not matches:
        return None

    preferred = [
        p for p in matches
        if "layer" in p.name.lower() and "sweep" in p.name.lower()
    ]
    candidates = preferred or matches

    if len(candidates) > 1:
        rendered = "\n".join(f"  - {p}" for p in candidates)
        raise RuntimeError(
            "Multiple layer-sweep YAML files were discovered. "
            "Specify one explicitly with --layer-sweep-config:\n"
            f"{rendered}"
        )

    return candidates[0]


def _normalized_relpath(path: Path | str) -> str:
    return Path(path).as_posix().lower().lstrip("./")


def load_real_layer_sweep_candidates(
    *,
    sweep_path: Path,
    current_config_path: Path,
    kind: str,
) -> tuple[list[int], str]:
    """
    Use the project's real layer_sweep loader and select candidates belonging
    to this backbone/config.

    First match both backbone_kind and base_config. If the config path cannot
    be matched exactly, fall back to kind only only when that kind appears
    under exactly one base_config in the sweep file.
    """
    try:
        from papr_ssl.experiments.layer_sweep import load_layer_sweep
    except ImportError as exc:
        raise RuntimeError(
            "Could not import papr_ssl.experiments.layer_sweep.load_layer_sweep"
        ) from exc

    sweep_path = sweep_path.expanduser().resolve()
    candidates = load_layer_sweep(sweep_path)

    def normalize_kind(value: Any) -> str:
        return (
            str(value)
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
        )

    normalized_kind = normalize_kind(kind)
    current_norm = _normalized_relpath(current_config_path)

    kind_matches = [
        c for c in candidates
        if normalize_kind(c.backbone_kind) == normalized_kind
    ]
    if not kind_matches:
        raise RuntimeError(
            f"No layer-sweep candidates found for backbone kind {kind!r} "
            f"in {sweep_path}"
        )

    exact_matches = []
    for candidate in kind_matches:
        base_norm = _normalized_relpath(candidate.base_config)
        if (
            base_norm == current_norm
            or current_norm.endswith(base_norm)
            or Path(base_norm).name == current_config_path.name.lower()
        ):
            exact_matches.append(candidate)

    selected = exact_matches
    match_mode = "kind + base_config"

    if not selected:
        base_configs = {
            _normalized_relpath(c.base_config)
            for c in kind_matches
        }
        if len(base_configs) == 1:
            selected = kind_matches
            match_mode = "kind only (unique base_config in sweep)"
        else:
            rendered = ", ".join(sorted(base_configs))
            raise RuntimeError(
                "Layer-sweep file contains multiple base_config entries for "
                f"{kind!r}, but none matches current config "
                f"{current_config_path}. Candidates: {rendered}"
            )

    layers = _dedupe_ints([int(c.hidden_layer) for c in selected])
    if not layers:
        raise RuntimeError(
            f"Layer sweep produced no candidates for {kind!r}"
        )

    return (
        layers,
        f"{sweep_path} via load_layer_sweep ({match_mode})",
    )



def resolve_layer_candidates(
    *,
    explicit_layers: list[int] | None,
    layer_sweep_config: Path | None,
    project_root: Path,
    current_config_path: Path,
    config: Mapping[str, Any],
    kind: str,
    observed_count: int,
    configured_layer: int | None,
) -> tuple[list[int], str]:
    # Explicit CLI candidates are useful for diagnostics, but P1-08 should
    # normally use the real layer-sweep YAML.
    if explicit_layers:
        return _dedupe_ints(explicit_layers), "CLI --layers"

    sweep_path = layer_sweep_config
    if sweep_path is None:
        sweep_path = discover_layer_sweep_config(project_root)

    if sweep_path is not None:
        return load_real_layer_sweep_candidates(
            sweep_path=sweep_path,
            current_config_path=current_config_path,
            kind=kind,
        )

    # Backward-compatible support for candidate lists embedded directly in the
    # loaded config, if the project ever adopts that layout.
    from_config, config_source = _find_candidate_layers_in_config(config)
    if from_config:
        return from_config, f"config:{config_source}"

    # Structural fallback is diagnostic only. It must NOT be mistaken for the
    # actual P1-08 layer_sweep candidate validation.
    max_valid = observed_count - 1
    anchors = [
        0,
        max_valid // 4,
        max_valid // 2,
        (3 * max_valid) // 4,
        max_valid,
    ]
    if configured_layer is not None:
        anchors.append(int(configured_layer))
    anchors.append(observed_count)

    return (
        sorted(set(anchors)),
        "DIAGNOSTIC FALLBACK: auto structural probes + invalid boundary; "
        "real layer-sweep YAML not found",
    )



def run_p1_08_layer_check(
    *,
    candidates: list[int],
    candidate_source: str,
    hidden_state_info: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a VALID/INVALID table from the real hidden_states tuple captured by
    the read-only smoke forward.

    A candidate is VALID iff it is a legal exact hidden_states index and its
    tensor has a concrete [B,T,D] shape.
    """
    count = int(hidden_state_info["observed_hidden_states_count"])
    shapes = hidden_state_info["all_hidden_state_shapes"]

    rows: list[dict[str, Any]] = []
    valid_layers: list[int] = []
    invalid_layers: list[int] = []

    for layer in candidates:
        row: dict[str, Any] = {"layer": int(layer)}

        if layer < 0 or layer >= count:
            row.update(
                {
                    "status": "INVALID",
                    "reason": (
                        "outside observed hidden_states index range "
                        f"[0, {count - 1}]"
                    ),
                    "shape": None,
                }
            )
            invalid_layers.append(int(layer))
        else:
            shape = shapes[layer]
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or any(
                    not isinstance(dim, int) or dim <= 0
                    for dim in shape
                )
            ):
                row.update(
                    {
                        "status": "INVALID",
                        "reason": (
                            f"hidden_states[{layer}] does not expose "
                            "a valid [B,T,D] tensor shape"
                        ),
                        "shape": shape,
                    }
                )
                invalid_layers.append(int(layer))
            else:
                row.update(
                    {
                        "status": "VALID",
                        "reason": None,
                        "shape": shape,
                    }
                )
                valid_layers.append(int(layer))

        rows.append(row)

    if not valid_layers:
        raise RuntimeError("P1-08 found no VALID candidate layer.")

    formal_source = not candidate_source.startswith("DIAGNOSTIC FALLBACK:")
    return {
        "mode": "read_only_hidden_states_observation",
        "candidate_source": candidate_source,
        "formal_layer_sweep_source": formal_source,
        "observed_hidden_states_count": count,
        "valid_index_range": [0, count - 1],
        "candidates": rows,
        "valid_layers": valid_layers,
        "invalid_layers": invalid_layers,
        "result": "PASS" if formal_source else "DIAGNOSTIC_ONLY",
    }



def _bytes_to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 ** 2)


def _windows_process_memory_bytes() -> dict[str, int] | None:
    """
    Return current/peak process memory on Windows via GetProcessMemoryInfo.

    Use explicit WinAPI signatures. Relying on ctypes' default c_int return /
    argument conversion can corrupt HANDLE values on 64-bit Windows and make
    GetProcessMemoryInfo silently return FALSE.
    """
    if os.name != "nt":
        return None

    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)

        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE

        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        handle = kernel32.GetCurrentProcess()
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)

        ok = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            error_code = ctypes.get_last_error()
            raise OSError(
                error_code,
                "GetProcessMemoryInfo failed",
            )

        return {
            "rss": int(counters.WorkingSetSize),
            "peak_rss": int(counters.PeakWorkingSetSize),
            "pagefile": int(counters.PagefileUsage),
            "peak_pagefile": int(counters.PeakPagefileUsage),
            "private": int(counters.PrivateUsage),
        }
    except Exception:
        return None


def process_memory_snapshot() -> dict[str, float | str | None]:
    """
    Process-level host-memory snapshot.

    On the actual PAPR-SSL development machine (Windows), WorkingSetSize is
    used so PyTorch native tensor/model allocations are included. tracemalloc
    is intentionally not used because it misses most native PyTorch memory.
    """
    win = _windows_process_memory_bytes()
    if win is not None:
        return {
            "backend": "windows_GetProcessMemoryInfo",
            "rss_mb": _bytes_to_mb(win["rss"]),
            "process_lifetime_peak_rss_mb": _bytes_to_mb(win["peak_rss"]),
            "pagefile_mb": _bytes_to_mb(win["pagefile"]),
            "process_lifetime_peak_pagefile_mb": _bytes_to_mb(win["peak_pagefile"]),
            "private_mb": _bytes_to_mb(win.get("private")),
        }

    # Portable fallback. ru_maxrss is a peak rather than current RSS and has
    # platform-specific units; current RSS remains unavailable without psutil.
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak = float(usage.ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        if sys.platform == "darwin":
            peak_mb = peak / (1024.0 ** 2)
        else:
            peak_mb = peak / 1024.0
        return {
            "backend": "resource_getrusage",
            "rss_mb": None,
            "process_lifetime_peak_rss_mb": peak_mb,
            "pagefile_mb": None,
            "process_lifetime_peak_pagefile_mb": None,
        }
    except Exception:
        return {
            "backend": "unavailable",
            "rss_mb": None,
            "process_lifetime_peak_rss_mb": None,
            "pagefile_mb": None,
            "process_lifetime_peak_pagefile_mb": None,
        }


class ProcessRSSSampler:
    """Low-overhead RSS sampler for untimed memory probes."""

    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_mb: float | None = None

    def _sample_once(self) -> None:
        snap = process_memory_snapshot()
        rss = snap.get("rss_mb")
        if isinstance(rss, (int, float)):
            rss_f = float(rss)
            if self.peak_rss_mb is None or rss_f > self.peak_rss_mb:
                self.peak_rss_mb = rss_f

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)
        self._sample_once()

    def start(self) -> "ProcessRSSSampler":
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="papr-ssl-rss-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample_once()
        return self.peak_rss_mb


def cuda_memory_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "allocated_mb": None,
            "reserved_mb": None,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        }

    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 ** 2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 ** 2),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 ** 2),
    }


def system_resource_metadata(device: torch.device) -> dict[str, Any]:
    cpu_name = (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.uname().processor
        or None
    )
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": cpu_name,
        "logical_cpu_count": os.cpu_count(),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }


def run_p1_09_memory_probe(
    *,
    backbone: Any,
    waveform: torch.Tensor,
    device: torch.device,
    host_baseline: Mapping[str, Any],
    host_after_load: Mapping[str, Any],
    load_peak_rss_mb: float | None,
    cuda_after_load: Mapping[str, Any],
    cuda_load_peak: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Run one untimed forward dedicated to peak-memory measurement.

    Latency is measured separately so the RSS sampler cannot perturb timing.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _sync(device)

    sampler = ProcessRSSSampler().start()
    with torch.inference_mode():
        _ = backbone(waveform)
        _sync(device)
    forward_peak_rss_mb = sampler.stop()

    host_after_probe = process_memory_snapshot()
    cuda_after_probe = cuda_memory_snapshot(device)

    baseline_rss = host_baseline.get("rss_mb")
    total_peak_candidates = [
        x for x in (load_peak_rss_mb, forward_peak_rss_mb)
        if isinstance(x, (int, float))
    ]
    host_peak = max(total_peak_candidates) if total_peak_candidates else None
    host_increment = None
    if isinstance(host_peak, (int, float)) and isinstance(baseline_rss, (int, float)):
        host_increment = max(0.0, float(host_peak) - float(baseline_rss))

    gpu_peak_alloc = cuda_after_probe.get("peak_allocated_mb")
    gpu_peak_reserved = cuda_after_probe.get("peak_reserved_mb")
    gpu_load_alloc = cuda_after_load.get("allocated_mb")
    gpu_load_reserved = cuda_after_load.get("reserved_mb")

    return {
        "host_memory": {
            "backend": host_after_probe.get("backend"),
            "baseline_rss_mb": host_baseline.get("rss_mb"),
            "after_load_rss_mb": host_after_load.get("rss_mb"),
            "load_peak_rss_mb": load_peak_rss_mb,
            "forward_peak_rss_mb": forward_peak_rss_mb,
            "overall_observed_peak_rss_mb": host_peak,
            "overall_peak_increment_from_baseline_mb": host_increment,
            "after_probe_rss_mb": host_after_probe.get("rss_mb"),
            "process_lifetime_peak_rss_mb": host_after_probe.get(
                "process_lifetime_peak_rss_mb"
            ),
        },
        "cuda_memory": {
            "after_load_allocated_mb": gpu_load_alloc,
            "after_load_reserved_mb": gpu_load_reserved,
            "load_peak_allocated_mb": cuda_load_peak.get("peak_allocated_mb"),
            "load_peak_reserved_mb": cuda_load_peak.get("peak_reserved_mb"),
            "forward_peak_allocated_mb": gpu_peak_alloc,
            "forward_peak_reserved_mb": gpu_peak_reserved,
            "forward_over_resident_allocated_mb": (
                max(0.0, float(gpu_peak_alloc) - float(gpu_load_alloc))
                if isinstance(gpu_peak_alloc, (int, float))
                and isinstance(gpu_load_alloc, (int, float))
                else None
            ),
            "forward_over_resident_reserved_mb": (
                max(0.0, float(gpu_peak_reserved) - float(gpu_load_reserved))
                if isinstance(gpu_peak_reserved, (int, float))
                and isinstance(gpu_load_reserved, (int, float))
                else None
            ),
        },
    }


def build_p1_09_resource_report(
    *,
    report: Mapping[str, Any],
    system_meta: Mapping[str, Any],
    memory_probe: Mapping[str, Any],
) -> dict[str, Any]:
    bench = report["timing"]["benchmark"]
    return {
        "protocol": {
            "batch_size": int(report["input_shape"][0]),
            "samples_per_item": int(report["input_shape"][1]),
            "sample_rate_hz": int(report["sample_rate"]),
            "warmup_runs": int(bench["warmup_runs"]),
            "timed_runs": int(bench["timed_runs"]),
            "inference_mode": True,
            "dtype_contract": "float32 features",
        },
        "system": dict(system_meta),
        "load_seconds": float(report["timing"]["load_seconds"]),
        "latency_ms": {
            "mean": float(bench["mean_ms"]),
            "p50": float(bench["p50_ms"]),
            "p95": float(bench["p95_ms"]),
            "min": float(bench["min_ms"]),
            "max": float(bench["max_ms"]),
        },
        "memory": dict(memory_probe),
        "engineering_runnable": True,
        "result": "PASS",
    }



def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_forward(
    *,
    backbone: Any,
    waveform: torch.Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if runs <= 0:
        raise ValueError("--runs must be > 0")

    with torch.inference_mode():
        for _ in range(warmup):
            _ = backbone(waveform)
            _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(runs):
            _sync(device)
            start = time.perf_counter()
            _ = backbone(waveform)
            _sync(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    sorted_ms = sorted(latencies_ms)

    def percentile(p: float) -> float:
        if len(sorted_ms) == 1:
            return sorted_ms[0]
        rank = (len(sorted_ms) - 1) * p
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return sorted_ms[lo]
        frac = rank - lo
        return sorted_ms[lo] * (1.0 - frac) + sorted_ms[hi] * frac

    peak_allocated_mb = None
    peak_reserved_mb = None
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

    return {
        "warmup_runs": warmup,
        "timed_runs": runs,
        "latencies_ms": latencies_ms,
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "peak_cuda_allocated_mb": peak_allocated_mb,
        "peak_cuda_reserved_mb": peak_reserved_mb,
    }


def config_metadata(backbone_cfg: Mapping[str, Any]) -> dict[str, Any]:
    def first(*names: str):
        for name in names:
            if name in backbone_cfg:
                return backbone_cfg[name]
        return None

    return {
        "kind": first("kind", "type", "name"),
        "checkpoint": first(
            "checkpoint",
            "model_name",
            "model_name_or_path",
            "pretrained_model_name_or_path",
        ),
        "revision": first("revision"),
        "configured_hidden_layer": first(
            "hidden_layer", "selected_layer", "layer"
        ),
    }


def write_json_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def print_report(report: Mapping[str, Any]) -> None:
    print("=" * 76)
    print("PAPR-SSL P1-03 THROUGH P1-09 SSL BACKBONE SMOKE")
    print("=" * 76)

    cfg = report["config"]
    print(f"Config:                   {report['config_path']}")
    print(f"Backbone kind:            {cfg.get('kind')}")
    print(f"Checkpoint:               {cfg.get('checkpoint')}")
    print(f"Revision:                 {cfg.get('revision')}")
    print(f"Configured layer:         {cfg.get('configured_hidden_layer')}")
    print(f"Device:                   {report['device']}")
    print(f"Input waveform:           {report['input_shape']} @ {SAMPLE_RATE} Hz")

    structure = report["model_structure"]
    print("-" * 76)
    print(f"Model holder attr:        {structure.get('underlying_model_attr')}")
    print(f"Config hidden layers:     {structure.get('num_hidden_layers')}")
    print(f"Config hidden size:       {structure.get('hidden_size')}")
    print(f"Expected hidden states:   {structure.get('expected_hidden_states_count')}")

    raw = report["hidden_state_observation"]
    print(f"Observed hidden states:   {raw['observed_hidden_states_count']}")
    print(f"Valid hidden index range: {raw['valid_index_range']}")
    print(
        f"Raw selected state:       "
        f"{raw['selected_raw_hidden_state_shape']}"
    )
    print(
        f"Raw/wrapper shape match:  "
        f"{raw['selected_matches_wrapper_features']}"
    )

    model_inputs = report["model_forward_inputs"]
    print("-" * 76)
    print(f"Frontend contract:        {model_inputs['frontend_contract']}")
    for key, meta in model_inputs["keyword_tensors"].items():
        print(
            f"Model input {key}: "
            f"shape={meta['shape']}, dtype={meta['dtype']}, device={meta['device']}"
        )
    if model_inputs["positional_args"]:
        for idx, meta in enumerate(model_inputs["positional_args"]):
            if meta is not None:
                print(
                    f"Model positional[{idx}]: "
                    f"shape={meta['shape']}, dtype={meta['dtype']}, device={meta['device']}"
                )
    print(f"Frontend alignment PASS: {model_inputs['frontend_alignment_pass']}")

    out = report["output"]
    print("-" * 76)
    print(f"Selected layer:           {out['hidden_layer']}")
    print(f"Features shape:           {out['features_shape']}")
    print(f"Frame mask shape:         {out['frame_mask_shape']}")
    print(f"Features dtype:           {out['features_dtype']}")
    print(f"Frame mask dtype:         {out['frame_mask_dtype']}")
    print(f"Finite:                   {out['features_finite']}")
    print(f"Valid frames/sample:      {out['valid_frames_per_sample']}")

    batch_mask = report.get("batch_mask_test")
    if batch_mask is not None:
        print("-" * 76)
        print("P1-07 BATCH / PADDING / MASK")
        print(f"Singleton:                {batch_mask['singleton']['status']}")
        full = batch_mask["full_length_batch"]
        print(f"Full-length batch:        {full['status']}")
        print(f"Batch input shape:        {full['input_shape']}")
        print(f"Batch features shape:     {full['features_shape']}")
        print(f"Batch frame mask shape:   {full['frame_mask_shape']}")
        print(f"Batch valid frames:       {full['valid_frames_per_sample']}")
        print(
            "Singleton/batch max diff: "
            f"{full['singleton_vs_batch_sample0_max_abs_diff']:.6g}"
        )
        padded = batch_mask["padded_batch"]
        print(f"Padded semantics:         {padded['status']}")
        if padded["status"] == "PASS":
            print(f"Validity contract:        {padded['validity_contract']}")
            print(f"Waveform lengths:         {padded['waveform_lengths']}")
            print(f"Padded valid frames:      {padded['valid_frames_per_sample']}")
            print(f"Prefix-contiguous mask:   {padded['prefix_contiguous_mask']}")
        else:
            print(f"Padded reason:            {padded['reason']}")
        print(f"P1-07 RESULT:             {batch_mask['result']}")

    layer_check = report.get("layer_check")
    if layer_check is not None:
        print("-" * 76)
        print("P1-08 CANDIDATE LAYER VALIDITY")
        print(
            f"Candidate source:          "
            f"{layer_check['candidate_source']}"
        )
        print(
            f"Formal sweep source:       "
            f"{layer_check['formal_layer_sweep_source']}"
        )
        print(
            f"Observed hidden states:    "
            f"{layer_check['observed_hidden_states_count']}"
        )
        print(
            f"Valid index range:         "
            f"{layer_check['valid_index_range']}"
        )
        for row in layer_check["candidates"]:
            suffix = (
                f" shape={row['shape']}"
                if row["shape"] is not None
                else ""
            )
            reason = (
                f" ({row['reason']})"
                if row["reason"]
                else ""
            )
            print(
                f"Layer {row['layer']:>3}:               "
                f"{row['status']}{suffix}{reason}"
            )
        print(
            f"VALID layers:              "
            f"{layer_check['valid_layers']}"
        )
        print(
            f"INVALID layers:            "
            f"{layer_check['invalid_layers']}"
        )
        print(
            f"P1-08 RESULT:              "
            f"{layer_check['result']}"
        )

    resource = report.get("resource_benchmark")
    if resource is not None:
        print("-" * 76)
        print("P1-09 RESOURCE BENCHMARK")
        protocol = resource["protocol"]
        print(
            f"Protocol:                  batch={protocol['batch_size']}, "
            f"{protocol['samples_per_item']} samples @ "
            f"{protocol['sample_rate_hz']} Hz"
        )
        print(
            f"Warmup / timed runs:       "
            f"{protocol['warmup_runs']} / {protocol['timed_runs']}"
        )
        system = resource["system"]
        print(f"Platform:                  {system.get('platform')}")
        print(f"CPU:                       {system.get('cpu')}")
        print(f"Logical CPUs:              {system.get('logical_cpu_count')}")
        print(f"GPU:                       {system.get('gpu')}")
        print(f"Load time:                 {resource['load_seconds']:.3f} s")
        lat = resource["latency_ms"]
        print(f"Forward mean:              {lat['mean']:.3f} ms")
        print(f"Forward P50:               {lat['p50']:.3f} ms")
        print(f"Forward P95:               {lat['p95']:.3f} ms")

        host = resource["memory"]["host_memory"]
        print(f"Host memory backend:       {host.get('backend')}")
        if host.get("baseline_rss_mb") is not None:
            print(
                f"Host baseline RSS:         "
                f"{host['baseline_rss_mb']:.1f} MB"
            )
        if host.get("after_load_rss_mb") is not None:
            print(
                f"Host RSS after load:       "
                f"{host['after_load_rss_mb']:.1f} MB"
            )
        if host.get("overall_observed_peak_rss_mb") is not None:
            print(
                f"Host observed peak RSS:    "
                f"{host['overall_observed_peak_rss_mb']:.1f} MB"
            )
        if host.get("overall_peak_increment_from_baseline_mb") is not None:
            print(
                f"Host peak over baseline:   "
                f"{host['overall_peak_increment_from_baseline_mb']:.1f} MB"
            )

        cuda = resource["memory"]["cuda_memory"]
        if cuda.get("after_load_allocated_mb") is not None:
            print(
                f"CUDA resident allocated:   "
                f"{cuda['after_load_allocated_mb']:.1f} MB"
            )
            print(
                f"CUDA resident reserved:    "
                f"{cuda['after_load_reserved_mb']:.1f} MB"
            )
            print(
                f"CUDA forward peak alloc:   "
                f"{cuda['forward_peak_allocated_mb']:.1f} MB"
            )
            print(
                f"CUDA forward peak reserve: "
                f"{cuda['forward_peak_reserved_mb']:.1f} MB"
            )
            print(
                f"CUDA forward overhead:     "
                f"{cuda['forward_over_resident_allocated_mb']:.1f} MB"
            )
        else:
            print("CUDA memory:               N/A (CPU run)")

        print(f"Engineering runnable:      {resource['engineering_runnable']}")
        print(f"P1-09 RESULT:              {resource['result']}")

    timing = report["timing"]
    print("-" * 76)
    print(f"Backbone load time:       {timing['load_seconds']:.3f} s")
    print(f"Warm-up runs:             {timing['benchmark']['warmup_runs']}")
    print(f"Timed runs:               {timing['benchmark']['timed_runs']}")
    print(f"Mean latency:             {timing['benchmark']['mean_ms']:.3f} ms")
    print(f"P50 latency:              {timing['benchmark']['p50_ms']:.3f} ms")
    print(f"P95 latency:              {timing['benchmark']['p95_ms']:.3f} ms")
    print(f"Min latency:              {timing['benchmark']['min_ms']:.3f} ms")
    print(f"Max latency:              {timing['benchmark']['max_ms']:.3f} ms")

    alloc = timing["benchmark"]["peak_cuda_allocated_mb"]
    reserv = timing["benchmark"]["peak_cuda_reserved_mb"]
    if alloc is not None:
        print(f"Peak CUDA allocated:      {alloc:.1f} MB")
        print(f"Peak CUDA reserved:       {reserv:.1f} MB")
    else:
        print("Peak CUDA allocated:      N/A")
        print("Peak CUDA reserved:       N/A")

    print("=" * 76)
    print("SMOKE RESULT: PASS")
    print("=" * 76)


def main() -> int:
    args = parse_args()
    device = select_device(args.device)

    config = load_yaml(args.config)
    backbone_cfg = extract_backbone_cfg(config)
    metadata = config_metadata(backbone_cfg)

    # P1 reference implementation is intentionally Wav2Vec2-first.
    kind = str(metadata.get("kind") or "").lower().replace("-", "").replace("_", "")
    if kind and kind not in {"wav2vec2", "wav2vec", "wavlm", "w2vbert2", "wav2vec2bert"}:
        raise RuntimeError(
            "Unsupported backbone kind for the shared PAPR-SSL smoke CLI. "
            "Expected wav2vec2, wavlm, or w2v_bert2."
        )

    waveform = build_smoke_waveform(
        device=device,
        frequency_hz=args.frequency_hz,
        amplitude=args.amplitude,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        _sync(device)
        torch.cuda.reset_peak_memory_stats(device)

    host_baseline = process_memory_snapshot()
    load_sampler = ProcessRSSSampler().start() if args.resource_benchmark else None

    load_start = time.perf_counter()
    backbone = create_backbone(backbone_cfg)
    move_backbone_to_device(backbone, device)
    _sync(device)
    load_seconds = time.perf_counter() - load_start

    load_peak_rss_mb = load_sampler.stop() if load_sampler is not None else None
    host_after_load = process_memory_snapshot()
    cuda_load_peak = cuda_memory_snapshot(device)
    cuda_after_load = {
        "allocated_mb": cuda_load_peak.get("allocated_mb"),
        "reserved_mb": cuda_load_peak.get("reserved_mb"),
    }

    structure = model_structure_info(backbone)

    # One authoritative wrapper forward with a hook on the real HF model.
    observed_output, raw_hidden, model_inputs = observe_model_io_and_hidden_states(
        backbone, waveform
    )
    output_info = validate_ssl_backbone_output(observed_output)
    hidden_state_info = validate_hidden_state_observation(
        output_info=output_info,
        raw_info=raw_hidden,
    )
    forward_input_info = validate_model_forward_inputs(
        kind=kind,
        model_inputs=model_inputs,
        output_info=output_info,
    )

    batch_mask_test = None
    if args.batch_mask_test:
        batch_mask_test = run_p1_07_batch_mask_test(
            backbone=backbone,
            kind=kind,
            singleton_output=observed_output,
            singleton_output_info=output_info,
            device=device,
            amplitude=args.amplitude,
        )

    layer_check = None
    if args.layer_check:
        candidate_layers, candidate_source = resolve_layer_candidates(
            explicit_layers=args.layers,
            layer_sweep_config=args.layer_sweep_config,
            project_root=Path.cwd(),
            current_config_path=args.config,
            config=config,
            kind=kind,
            observed_count=(
                hidden_state_info["observed_hidden_states_count"]
            ),
            configured_layer=metadata.get("configured_hidden_layer"),
        )
        layer_check = run_p1_08_layer_check(
            candidates=candidate_layers,
            candidate_source=candidate_source,
            hidden_state_info=hidden_state_info,
        )

    # Repeatable performance measurement after the correctness checks.
    benchmark = benchmark_forward(
        backbone=backbone,
        waveform=waveform,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
    )

    memory_probe = None
    system_meta = None
    if args.resource_benchmark:
        system_meta = system_resource_metadata(device)
        memory_probe = run_p1_09_memory_probe(
            backbone=backbone,
            waveform=waveform,
            device=device,
            host_baseline=host_baseline,
            host_after_load=host_after_load,
            load_peak_rss_mb=load_peak_rss_mb,
            cuda_after_load=cuda_after_load,
            cuda_load_peak=cuda_load_peak,
        )

    report = {
        "schema": "papr_ssl.smoke_backbone.v8",
        "config_path": str(args.config),
        "config": metadata,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "input_shape": list(waveform.shape),
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "model_structure": structure,
        "hidden_state_observation": hidden_state_info,
        "model_forward_inputs": forward_input_info,
        "output": output_info,
        "batch_mask_test": batch_mask_test,
        "layer_check": layer_check,
        "timing": {
            "load_seconds": load_seconds,
            "benchmark": benchmark,
        },
        "resource_benchmark": None,
        "result": "PASS",
    }

    if args.resource_benchmark:
        report["resource_benchmark"] = build_p1_09_resource_report(
            report=report,
            system_meta=system_meta or {},
            memory_probe=memory_probe or {},
        )

    print_report(report)

    if args.json_out is not None:
        write_json_report(args.json_out, report)
        print(f"\nJSON report written to: {args.json_out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nSMOKE RESULT: INTERRUPTED", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print("=" * 76, file=sys.stderr)
        print("SMOKE RESULT: FAIL", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("=" * 76, file=sys.stderr)
        raise
