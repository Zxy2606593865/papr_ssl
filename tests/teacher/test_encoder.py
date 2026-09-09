from __future__ import annotations

import unittest

import torch
from torch import nn

from papr_ssl.models.teacher.backbones.wav2vec2 import Wav2Vec2BackboneOutput
from papr_ssl.models.teacher.encoder import TeacherEncoder
from papr_ssl.models.teacher.heads.mean_dr import MeanDR


class _FakeBackbone(nn.Module):
    def forward(self, waveforms: torch.Tensor) -> Wav2Vec2BackboneOutput:
        batch_size = waveforms.shape[0]
        base = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        features = base.unsqueeze(0).expand(batch_size, -1, -1).clone()
        mask = torch.ones(batch_size, 5, dtype=torch.bool)
        return Wav2Vec2BackboneOutput(
            features=features,
            frame_mask=mask,
            hidden_layer=2,
        )


class TeacherEncoderTest(unittest.TestCase):
    def test_fake_backbone_composition_is_deterministic_in_eval(self) -> None:
        torch.manual_seed(17)
        encoder = TeacherEncoder(_FakeBackbone(), MeanDR(input_dim=4, output_dim=64))
        encoder.eval()
        waveforms = torch.zeros(2, 16_000, dtype=torch.float32)

        first = encoder.encode(waveforms)
        second = encoder.encode(waveforms)

        self.assertEqual(tuple(first.embedding.shape), (2, 64))
        self.assertEqual(first.embedding.dtype, torch.float32)
        self.assertTrue(torch.equal(first.embedding, second.embedding))
        self.assertTrue(torch.equal(first.frame_count, torch.tensor([5, 5])))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(first.embedding, dim=1),
                torch.ones(2),
                atol=1e-5,
                rtol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
