from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from papr_ssl.models.teacher.backbones.base import SSLBackboneOutput
from papr_ssl.models.teacher.backbones.length_aware import LengthAwareSSLAdapter


class FakeStrideBackbone(nn.Module):
    """One output frame per two input samples; all frames valid."""

    def __init__(self, hidden_layer: int = 3, dim: int = 4) -> None:
        super().__init__()
        self.hidden_layer = hidden_layer
        self.dim = dim
        self.seen_lengths: list[int] = []

    def forward(self, waveform: torch.Tensor) -> SSLBackboneOutput:
        self.seen_lengths.append(int(waveform.shape[1]))
        t = max(1, waveform.shape[1] // 2)
        base = waveform[:, : t * 2 : 2].unsqueeze(-1)
        features = base.repeat(1, 1, self.dim).to(torch.float32)
        mask = torch.ones(
            (waveform.shape[0], t),
            dtype=torch.bool,
            device=waveform.device,
        )
        return SSLBackboneOutput(
            features=features,
            frame_mask=mask,
            hidden_layer=self.hidden_layer,
        )


class FakeTailMaskedBackbone(nn.Module):
    """Simulates a frontend that emits internally padded invalid tail frames."""

    def __init__(self, hidden_layer: int = 7, dim: int = 3) -> None:
        super().__init__()
        self.hidden_layer = hidden_layer
        self.dim = dim

    def forward(self, waveform: torch.Tensor) -> SSLBackboneOutput:
        # Always expose two extra internal alignment frames.
        valid_t = max(1, waveform.shape[1] // 4)
        total_t = valid_t + 2

        features = torch.ones(
            (1, total_t, self.dim),
            dtype=torch.float32,
            device=waveform.device,
        )
        # Make the invalid tail obviously non-zero; adapter must still remove it.
        features[:, valid_t:, :] = 99.0

        mask = torch.zeros(
            (1, total_t),
            dtype=torch.bool,
            device=waveform.device,
        )
        mask[:, :valid_t] = True

        return SSLBackboneOutput(
            features=features,
            frame_mask=mask,
            hidden_layer=self.hidden_layer,
        )


class LengthAwareSSLAdapterTest(unittest.TestCase):
    def test_lengths_trim_each_sample_before_backbone(self) -> None:
        base = FakeStrideBackbone()
        adapter = LengthAwareSSLAdapter(base)

        waveforms = torch.zeros((2, 8), dtype=torch.float32)
        lengths = torch.tensor([8, 4], dtype=torch.int64)

        out = adapter(waveforms, waveform_lengths=lengths)

        self.assertEqual(base.seen_lengths, [8, 4])
        self.assertEqual(tuple(out.features.shape), (2, 4, 4))
        self.assertEqual(out.frame_mask.sum(dim=1).tolist(), [4, 2])
        self.assertTrue(torch.all(out.features[1, 2:] == 0))

    def test_mask_only_contract(self) -> None:
        base = FakeStrideBackbone()
        adapter = LengthAwareSSLAdapter(base)

        waveforms = torch.zeros((2, 8), dtype=torch.float32)
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )

        out = adapter(waveforms, waveform_mask=mask)
        self.assertEqual(base.seen_lengths, [8, 4])
        self.assertEqual(out.frame_mask.sum(dim=1).tolist(), [4, 2])

    def test_lengths_and_mask_must_agree(self) -> None:
        adapter = LengthAwareSSLAdapter(FakeStrideBackbone())
        waveforms = torch.zeros((1, 8), dtype=torch.float32)
        lengths = torch.tensor([4], dtype=torch.int64)
        mask = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool
        )

        with self.assertRaises(ValueError):
            adapter(
                waveforms,
                waveform_lengths=lengths,
                waveform_mask=mask,
            )

    def test_non_prefix_waveform_mask_rejected(self) -> None:
        adapter = LengthAwareSSLAdapter(FakeStrideBackbone())
        waveforms = torch.zeros((1, 6), dtype=torch.float32)
        mask = torch.tensor(
            [[1, 1, 0, 1, 0, 0]], dtype=torch.bool
        )

        with self.assertRaises(ValueError):
            adapter(waveforms, waveform_mask=mask)

    def test_full_length_behavior_remains_available(self) -> None:
        base = FakeStrideBackbone()
        adapter = LengthAwareSSLAdapter(base)
        waveforms = torch.zeros((3, 10), dtype=torch.float32)

        out = adapter(waveforms)

        self.assertEqual(base.seen_lengths, [10, 10, 10])
        self.assertEqual(out.frame_mask.sum(dim=1).tolist(), [5, 5, 5])

    def test_forward_audio_batch_bridge(self) -> None:
        adapter = LengthAwareSSLAdapter(FakeStrideBackbone())
        batch = SimpleNamespace(
            waveforms=torch.zeros((2, 8), dtype=torch.float32),
            lengths=torch.tensor([8, 6], dtype=torch.int64),
            waveform_mask=torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0, 0],
                ],
                dtype=torch.bool,
            ),
        )

        out = adapter.forward_audio_batch(batch)
        self.assertEqual(out.frame_mask.sum(dim=1).tolist(), [4, 3])

    def test_model_provided_invalid_tail_is_respected(self) -> None:
        adapter = LengthAwareSSLAdapter(FakeTailMaskedBackbone())

        waveforms = torch.zeros((2, 12), dtype=torch.float32)
        lengths = torch.tensor([12, 8], dtype=torch.int64)

        out = adapter(waveforms, waveform_lengths=lengths)

        # valid frames = floor(length/4): [3, 2]
        self.assertEqual(out.frame_mask.sum(dim=1).tolist(), [3, 2])
        self.assertEqual(tuple(out.features.shape), (2, 3, 3))

        # 99-valued invalid internal tail must never survive.
        self.assertFalse(torch.any(out.features == 99.0))

        # Cross-sample repadding remains zero.
        self.assertTrue(torch.all(out.features[1, 2:] == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
