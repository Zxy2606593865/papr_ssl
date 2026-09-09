from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from papr_ssl.models.teacher.backbones.wavlm import WavLMBackbone


class _FakeWavLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=3, hidden_size=6)
        self.anchor = nn.Parameter(torch.ones(1))
        self.received_input_values = None

    def forward(self, input_values: torch.Tensor, **_: object) -> SimpleNamespace:
        self.received_input_values = input_values
        states = tuple(
            torch.full((input_values.shape[0], 4, 6), float(index))
            for index in range(4)
        )
        return SimpleNamespace(hidden_states=states)


class WavLMBackboneTest(unittest.TestCase):
    def test_fake_model_uses_raw_waveform_and_common_contract(self) -> None:
        model = _FakeWavLM()
        backbone = WavLMBackbone(hidden_layer=2, model=model)
        waveforms = torch.zeros(2, 16_000, dtype=torch.float32)
        output = backbone(waveforms)
        self.assertIs(model.received_input_values, waveforms)
        self.assertEqual(tuple(output.features.shape), (2, 4, 6))
        self.assertEqual(output.hidden_layer, 2)
        self.assertTrue(bool(output.frame_mask.all()))
        self.assertFalse(model.training)

    def test_hidden_layer_out_of_range_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            WavLMBackbone(hidden_layer=4, model=_FakeWavLM())


if __name__ == "__main__":
    unittest.main(verbosity=2)
