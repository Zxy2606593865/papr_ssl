from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from papr_ssl.models.teacher.backbones.factory import create_ssl_backbone
from papr_ssl.models.teacher.backbones.w2v_bert2 import W2vBert2Backbone
from papr_ssl.models.teacher.backbones.wav2vec2 import Wav2Vec2Backbone
from papr_ssl.models.teacher.backbones.wavlm import WavLMBackbone


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1, hidden_size=4)
        self.anchor = nn.Parameter(torch.ones(1))


class _FakeFeatureExtractor:
    pass


class BackboneFactoryTest(unittest.TestCase):
    def test_factory_supports_all_three_kinds(self) -> None:
        wav2vec2 = create_ssl_backbone(
            "wav2vec2", hidden_layer=1, model=_FakeModel()
        )
        wavlm = create_ssl_backbone("wavlm", hidden_layer=1, model=_FakeModel())
        w2v_bert2 = create_ssl_backbone(
            "w2v_bert2",
            hidden_layer=1,
            model=_FakeModel(),
            feature_extractor=_FakeFeatureExtractor(),
        )
        self.assertIsInstance(wav2vec2, Wav2Vec2Backbone)
        self.assertIsInstance(wavlm, WavLMBackbone)
        self.assertIsInstance(w2v_bert2, W2vBert2Backbone)

    def test_unknown_kind_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown backbone kind"):
            create_ssl_backbone("automatic-best-model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
