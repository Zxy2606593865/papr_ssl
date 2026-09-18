"""W2v-BERT 2.0 wrapper with its model-specific acoustic frontend."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from papr_ssl.contracts.teacher import SAMPLE_RATE

from .base import (
    SSLBackboneOutput,
    freeze_model,
    model_device,
    sample_mask_to_lengths,
    select_hidden_state,
    validate_hidden_layer,
    validate_hidden_layer_against_config,
    validate_waveforms,
)


class W2vBert2Backbone(nn.Module):
    """Convert waveform to acoustic features, then select W2v-BERT hidden state."""

    def __init__(
        self,
        model_name: str = "facebook/w2v-bert-2.0",
        *,
        revision: Optional[str] = "main",
        hidden_layer: int = 24,
        freeze_backbone: bool = True,
        model: Optional[nn.Module] = None,
        feature_extractor: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if not str(model_name).strip():
            raise ValueError("model_name must be non-empty")
        validate_hidden_layer(hidden_layer)
        if (model is None) != (feature_extractor is None):
            raise ValueError(
                "model and feature_extractor must be injected together"
            )

        self.model_name = str(model_name).strip()
        self.revision = revision
        self.hidden_layer = hidden_layer
        self.freeze_backbone = bool(freeze_backbone)

        if model is None:
            model, feature_extractor = self._load_pretrained_components()

        self.model = model
        self.feature_extractor = feature_extractor

        validate_hidden_layer_against_config(self.model, self.hidden_layer)
        if self.freeze_backbone:
            freeze_model(self.model)

    def _load_pretrained_components(self) -> Tuple[nn.Module, Any]:
        try:
            from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load W2v-BERT 2.0 components"
            ) from exc

        extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name,
            revision=self.revision,
        )
        model = Wav2Vec2BertModel.from_pretrained(
            self.model_name,
            revision=self.revision,
            output_hidden_states=True,
        )
        return model, extractor

    @property
    def output_dim(self) -> Optional[int]:
        value = getattr(getattr(self.model, "config", None), "hidden_size", None)
        return None if value is None else int(value)

    def train(self, mode: bool = True) -> "W2vBert2Backbone":
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def forward(
        self,
        waveforms: Tensor,
        sample_mask: Optional[Tensor] = None,
    ) -> SSLBackboneOutput:
        """Native batched forward with optional raw-sample validity.

        When `sample_mask` is supplied, each row is trimmed to its semantic
        source length BEFORE entering AutoFeatureExtractor. The extractor then
        pads the variable-length raw-speech list once and emits its own
        feature-level attention mask. The W2v-BERT model itself is still called
        exactly once for the whole batch.
        """
        validate_waveforms(waveforms, sample_mask)

        input_features, frame_mask = self._extract_features(
            waveforms,
            sample_mask=sample_mask,
        )

        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            output = self.model(
                input_features=input_features,
                attention_mask=frame_mask.to(dtype=torch.long),
                output_hidden_states=True,
                return_dict=True,
            )

        features = select_hidden_state(
            output,
            hidden_layer=self.hidden_layer,
            expected_batch=waveforms.shape[0],
        )

        if frame_mask.shape != features.shape[:2]:
            raise RuntimeError(
                "feature-extractor attention_mask does not match model frame output"
            )

        return SSLBackboneOutput(
            features=features,
            frame_mask=frame_mask.to(
                device=features.device,
                dtype=torch.bool,
            ),
            hidden_layer=self.hidden_layer,
        )

    def _extract_features(
        self,
        waveforms: Tensor,
        sample_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if self.feature_extractor is None:
            raise RuntimeError("W2v-BERT 2.0 feature_extractor is missing")

        if sample_mask is None:
            raw_speech = [
                row.detach().cpu().numpy()
                for row in waveforms
            ]
        else:
            lengths = sample_mask_to_lengths(sample_mask)
            raw_speech = [
                waveforms[i, : int(length)].detach().cpu().numpy()
                for i, length in enumerate(lengths.detach().cpu().tolist())
            ]

        batch = self.feature_extractor(
            raw_speech,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        if not isinstance(batch, Mapping):
            try:
                batch = dict(batch)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "feature_extractor output must be mapping-like"
                ) from exc

        input_features = batch.get("input_features")
        attention_mask = batch.get("attention_mask")

        if not isinstance(input_features, Tensor) or input_features.ndim != 3:
            raise RuntimeError(
                "feature_extractor must return input_features [B,T,F]"
            )
        if input_features.shape[0] != waveforms.shape[0]:
            raise RuntimeError(
                "feature_extractor output batch size changed unexpectedly"
            )
        if not isinstance(attention_mask, Tensor):
            raise RuntimeError(
                "feature_extractor must return attention_mask [B,T]"
            )
        if attention_mask.shape != input_features.shape[:2]:
            raise RuntimeError(
                "feature_extractor attention_mask must have shape [B,T]"
            )

        attention_mask = attention_mask.to(dtype=torch.bool)
        if not bool(attention_mask.any(dim=1).all()):
            raise ValueError("every sample must contain valid acoustic frames")
        if attention_mask.shape[1] > 1:
            false_to_true = (~attention_mask[:, :-1]) & attention_mask[:, 1:]
            if bool(false_to_true.any()):
                raise ValueError(
                    "feature_extractor attention_mask must be prefix-contiguous"
                )

        device = model_device(self.model)
        return (
            input_features.to(
                device=device,
                dtype=torch.float32,
            ),
            attention_mask.to(
                device=device,
                dtype=torch.bool,
            ),
        )
