from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from papr_ssl.models.teacher.backbones.base import SSLBackboneOutput
from papr_ssl.models.teacher.backbones.wav2vec2 import Wav2Vec2Backbone


class _FakeWav2Vec2Model(nn.Module):
    def __init__(self, num_hidden_layers: int = 2, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
        )
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, input_values: torch.Tensor, **_: object) -> SimpleNamespace:
        batch_size = input_values.shape[0]
        states = tuple(
            torch.full((batch_size, 3, self.config.hidden_size), float(index))
            for index in range(self.config.num_hidden_layers + 1)
        )
        return SimpleNamespace(hidden_states=states)


class Wav2Vec2BackboneTest(unittest.TestCase):
    def test_hidden_layer_out_of_range_fails_without_download(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            Wav2Vec2Backbone(hidden_layer=3, model=_FakeWav2Vec2Model())

    def test_injected_model_returns_selected_layer(self) -> None:
        backbone = Wav2Vec2Backbone(
            hidden_layer=2,
            freeze_backbone=True,
            model=_FakeWav2Vec2Model(),
        )
        output = backbone(torch.zeros(2, 16_000))
        self.assertIsInstance(output, SSLBackboneOutput)
        self.assertEqual(tuple(output.features.shape), (2, 3, 4))
        self.assertTrue(bool((output.features == 2.0).all()))
        self.assertTrue(bool(output.frame_mask.all()))
        self.assertTrue(bool(output.mask.all()))
        self.assertEqual(output.hidden_layer, 2)
        self.assertFalse(backbone.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
