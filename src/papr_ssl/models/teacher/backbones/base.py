"""Minimal common contract and validation helpers for SSL backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SSLBackboneOutput:
    """Backbone-independent selected hidden-state output.

    `hidden_layer` is always the exact zero-based index used to select an item
    from `outputs.hidden_states`. No implicit encoder-layer conversion occurs.
    """

    features: Tensor
    frame_mask: Tensor
    hidden_layer: int

    def __post_init__(self) -> None:
        if not isinstance(self.features, Tensor) or self.features.ndim != 3:
            raise ValueError("features must be a torch.Tensor with shape [B,T,D]")
        if 0 in self.features.shape:
            raise ValueError("features dimensions must be non-zero")
        if self.features.dtype != torch.float32:
            raise TypeError("features must use torch.float32")
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError("features contain NaN or Inf")
        if not isinstance(self.frame_mask, Tensor):
            raise TypeError("frame_mask must be a torch.Tensor")
        if self.frame_mask.shape != self.features.shape[:2]:
            raise ValueError("frame_mask must have shape [B,T]")
        if self.frame_mask.dtype != torch.bool:
            raise TypeError("frame_mask must use torch.bool")
        if self.frame_mask.device != self.features.device:
            raise ValueError("frame_mask and features must be on the same device")
        if not bool(self.frame_mask.any(dim=1).all()):
            raise ValueError("every sample must contain at least one valid frame")
        validate_hidden_layer(self.hidden_layer)

    @property
    def mask(self) -> Tensor:
        """Compatibility alias retained for the Bootstrap API."""

        return self.frame_mask


def validate_hidden_layer(hidden_layer: int) -> None:
    if not isinstance(hidden_layer, int) or isinstance(hidden_layer, bool):
        raise TypeError("hidden_layer must be an integer")
    if hidden_layer < 0:
        raise ValueError("hidden_layer must be non-negative")


def validate_hidden_layer_against_config(model: nn.Module, hidden_layer: int) -> None:
    """Validate an exact hidden-state tuple index when config exposes depth."""

    validate_hidden_layer(hidden_layer)
    config = getattr(model, "config", None)
    layer_count = getattr(config, "num_hidden_layers", None)
    if layer_count is None:
        return
    maximum = int(layer_count)
    if hidden_layer > maximum:
        raise ValueError(
            f"hidden_layer={hidden_layer} is out of range; "
            f"model exposes outputs.hidden_states indices 0..{maximum}"
        )


def select_hidden_state(
    output: Any,
    *,
    hidden_layer: int,
    expected_batch: int,
) -> Tensor:
    """Select one exact hidden-state tuple index and normalize dtype only."""

    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("SSL model output did not include hidden_states")
    if hidden_layer >= len(hidden_states):
        raise ValueError(
            f"hidden_layer={hidden_layer} is out of range for runtime "
            f"outputs.hidden_states with valid indices 0..{len(hidden_states) - 1}"
        )
    features = hidden_states[hidden_layer]
    if not isinstance(features, Tensor) or features.ndim != 3:
        raise RuntimeError("selected hidden state is not a Tensor with shape [B,T,D]")
    if features.shape[0] != expected_batch:
        raise RuntimeError("SSL model output batch size changed unexpectedly")
    return features.to(dtype=torch.float32)


def validate_waveforms(waveforms: Tensor, sample_mask: Optional[Tensor] = None) -> None:
    if not isinstance(waveforms, Tensor):
        raise TypeError("waveforms must be a torch.Tensor")
    if waveforms.ndim != 2 or waveforms.shape[0] == 0 or waveforms.shape[1] == 0:
        raise ValueError("waveforms must have non-empty shape [B,S]")
    if waveforms.dtype != torch.float32:
        raise TypeError("waveforms must use torch.float32")
    if not bool(torch.isfinite(waveforms).all()):
        raise ValueError("waveforms contain NaN or Inf")
    if sample_mask is not None:
        if not isinstance(sample_mask, Tensor) or sample_mask.shape != waveforms.shape:
            raise ValueError("sample_mask must have shape [B,S]")
        if sample_mask.dtype != torch.bool:
            raise TypeError("sample_mask must use torch.bool")
        if not bool(sample_mask.any(dim=1).all()):
            raise ValueError("every sample must contain valid input samples")


def freeze_model(model: nn.Module) -> None:
    model.requires_grad_(False)
    model.eval()


def model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


def raw_sample_mask_to_frame_mask(
    model: nn.Module,
    features: Tensor,
    sample_mask: Optional[Tensor],
) -> Tensor:
    """Convert raw-sample masks using the HF waveform-model helper."""

    batch_size, frame_count = features.shape[:2]
    if sample_mask is None:
        return torch.ones(
            (batch_size, frame_count), dtype=torch.bool, device=features.device
        )
    converter = getattr(model, "_get_feature_vector_attention_mask", None)
    if converter is None:
        raise RuntimeError("model cannot convert a sample mask to a frame mask")
    try:
        frame_mask = converter(frame_count, sample_mask, add_adapter=False)
    except TypeError:
        frame_mask = converter(frame_count, sample_mask)
    return frame_mask.to(device=features.device, dtype=torch.bool)
