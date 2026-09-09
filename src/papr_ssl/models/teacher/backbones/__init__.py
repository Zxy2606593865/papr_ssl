"""Interchangeable SSL acoustic backbones."""

from .base import SSLBackboneOutput
from .factory import SUPPORTED_BACKBONES, create_ssl_backbone
from .w2v_bert2 import W2vBert2Backbone
from .wav2vec2 import Wav2Vec2Backbone, Wav2Vec2BackboneOutput
from .wavlm import WavLMBackbone

__all__ = [
    "SSLBackboneOutput",
    "SUPPORTED_BACKBONES",
    "W2vBert2Backbone",
    "Wav2Vec2Backbone",
    "Wav2Vec2BackboneOutput",
    "WavLMBackbone",
    "create_ssl_backbone",
]
