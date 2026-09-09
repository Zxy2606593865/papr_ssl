"""Small explicit factory for supported SSL Teacher backbones."""

from __future__ import annotations

from typing import Any

from torch import nn

from .w2v_bert2 import W2vBert2Backbone
from .wav2vec2 import Wav2Vec2Backbone
from .wavlm import WavLMBackbone


SUPPORTED_BACKBONES = ("wav2vec2", "wavlm", "w2v_bert2")


def create_ssl_backbone(kind: str, **kwargs: Any) -> nn.Module:
    """Create one supported backbone; unknown kinds never fall back."""

    normalized = str(kind).strip().lower()
    constructors = {
        "wav2vec2": Wav2Vec2Backbone,
        "wavlm": WavLMBackbone,
        "w2v_bert2": W2vBert2Backbone,
    }
    try:
        constructor = constructors[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown backbone kind {kind!r}; expected one of {SUPPORTED_BACKBONES}"
        ) from exc
    return constructor(**kwargs)
