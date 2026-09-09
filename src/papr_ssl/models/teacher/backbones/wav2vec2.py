"""Raw-waveform Wav2Vec2 frame-representation wrapper."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
from torch import Tensor, nn

from .base import (
    SSLBackboneOutput,
    freeze_model,
    raw_sample_mask_to_frame_mask,
    select_hidden_state,
    validate_hidden_layer,
    validate_hidden_layer_against_config,
    validate_waveforms,
)


Wav2Vec2BackboneOutput = SSLBackboneOutput


class Wav2Vec2Backbone(nn.Module):
    """Select one exact `outputs.hidden_states` index from Wav2Vec2."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        *,
        revision: Optional[str] = "main",
        hidden_layer: int = 12,
        freeze_backbone: bool = True,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if not str(model_name).strip():
            raise ValueError("model_name must be non-empty")
        validate_hidden_layer(hidden_layer)
        self.model_name = str(model_name).strip()
        self.revision = revision
        self.hidden_layer = hidden_layer
        self.freeze_backbone = bool(freeze_backbone)
        self.model = model if model is not None else self._load_pretrained_model()
        validate_hidden_layer_against_config(self.model, self.hidden_layer)
        if self.freeze_backbone:
            freeze_model(self.model)

    def _load_pretrained_model(self) -> nn.Module:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load a real Wav2Vec2 checkpoint"
            ) from exc
        return AutoModel.from_pretrained(
            self.model_name,
            revision=self.revision,
            output_hidden_states=True,
        )

    @property
    def output_dim(self) -> Optional[int]:
        value = getattr(getattr(self.model, "config", None), "hidden_size", None)
        return None if value is None else int(value)

    def train(self, mode: bool = True) -> "Wav2Vec2Backbone":
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def forward(
        self,
        waveforms: Tensor,
        sample_mask: Optional[Tensor] = None,
    ) -> SSLBackboneOutput:
        validate_waveforms(waveforms, sample_mask)
        call_kwargs = {
            "input_values": waveforms,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if sample_mask is not None:
            call_kwargs["attention_mask"] = sample_mask.to(dtype=torch.long)
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            output = self.model(**call_kwargs)
        features = select_hidden_state(
            output,
            hidden_layer=self.hidden_layer,
            expected_batch=waveforms.shape[0],
        )
        frame_mask = raw_sample_mask_to_frame_mask(
            self.model, features, sample_mask
        )
        return SSLBackboneOutput(
            features=features,
            frame_mask=frame_mask,
            hidden_layer=self.hidden_layer,
        )
