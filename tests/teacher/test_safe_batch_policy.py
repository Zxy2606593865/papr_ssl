from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from papr_ssl.models.teacher.backbones.base import SSLBackboneOutput
from papr_ssl.models.teacher.backbones.length_aware import LengthAwareSSLAdapter
from papr_ssl.models.teacher.backbones.safe_batch import (
    batching_policy_name,
    safe_forward_audio_batch,
)


class PaddingSensitiveBackbone(nn.Module):
    """Fake model whose features depend on the visible raw row length.

    It mimics the important failure mode measured for Wav2Vec2: a shorter sample
    numerically changes when right-padded in a larger raw batch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, int]] = []

    def forward(self, waveforms, sample_mask=None):
        b, s = waveforms.shape
        self.calls.append((b, s))
        t = max(1, s // 2)

        # Every valid feature value depends on the raw tensor width `s`.
        value = torch.full(
            (b, t, 2),
            float(s),
            dtype=torch.float32,
            device=waveforms.device,
        )

        if sample_mask is None:
            frame_mask = torch.ones(
                (b, t),
                dtype=torch.bool,
                device=waveforms.device,
            )
        else:
            sample_lengths = sample_mask.sum(dim=1)
            frame_lengths = torch.clamp(sample_lengths // 2, min=1)
            idx = torch.arange(t, device=waveforms.device).unsqueeze(0)
            frame_mask = idx < frame_lengths.unsqueeze(1)

        return SSLBackboneOutput(
            features=value,
            frame_mask=frame_mask,
            hidden_layer=1,
        )


class NativeFriendlyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, waveforms, sample_mask=None):
        self.calls += 1
        b, s = waveforms.shape
        t = max(1, s // 2)
        features = torch.ones(
            (b, t, 3),
            dtype=torch.float32,
            device=waveforms.device,
        )
        if sample_mask is None:
            mask = torch.ones((b, t), dtype=torch.bool, device=waveforms.device)
        else:
            lengths = sample_mask.sum(dim=1) // 2
            lengths = torch.clamp(lengths, min=1)
            idx = torch.arange(t, device=waveforms.device).unsqueeze(0)
            mask = idx < lengths.unsqueeze(1)
        return SSLBackboneOutput(features, mask, 2)


def make_batch():
    waveforms = torch.zeros((3, 8), dtype=torch.float32)
    lengths = torch.tensor([8, 4, 8], dtype=torch.int64)
    mask = torch.tensor(
        [
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,1,1,1,1],
        ],
        dtype=torch.bool,
    )
    return SimpleNamespace(
        waveforms=waveforms,
        lengths=lengths,
        waveform_mask=mask,
    )


class SafeBatchPolicyTest(unittest.TestCase):
    def test_wav2vec2_groups_by_exact_length_and_restores_order(self):
        base = PaddingSensitiveBackbone()
        batch = make_batch()

        safe = safe_forward_audio_batch(
            backbone_kind="wav2vec2",
            backbone=base,
            batch=batch,
        )
        reference = LengthAwareSSLAdapter(base).forward_audio_batch(batch)

        self.assertEqual(
            safe.frame_mask.sum(dim=1).tolist(),
            reference.frame_mask.sum(dim=1).tolist(),
        )
        self.assertTrue(torch.equal(safe.features, reference.features))

        # Two unique raw lengths -> two grouped calls.
        self.assertIn((1, 4), base.calls)
        self.assertIn((2, 8), base.calls)

        # Original row order [8,4,8] is restored.
        self.assertEqual(safe.frame_mask.sum(dim=1).tolist(), [4, 2, 4])
        self.assertTrue(torch.all(safe.features[0, :4] == 8.0))
        self.assertTrue(torch.all(safe.features[1, :2] == 4.0))
        self.assertTrue(torch.all(safe.features[2, :4] == 8.0))

    def test_wavlm_uses_one_native_batch_call(self):
        base = NativeFriendlyBackbone()
        out = safe_forward_audio_batch(
            backbone_kind="wavlm",
            backbone=base,
            batch=make_batch(),
        )
        self.assertEqual(base.calls, 1)
        self.assertEqual(out.frame_mask.sum(dim=1).tolist(), [4, 2, 4])

    def test_w2v_bert2_uses_one_native_batch_call(self):
        base = NativeFriendlyBackbone()
        out = safe_forward_audio_batch(
            backbone_kind="w2v_bert2",
            backbone=base,
            batch=make_batch(),
        )
        self.assertEqual(base.calls, 1)
        self.assertEqual(out.frame_mask.sum(dim=1).tolist(), [4, 2, 4])

    def test_frozen_policy_names(self):
        self.assertEqual(
            batching_policy_name("wav2vec2"),
            "grouped_exact_waveform_length",
        )
        self.assertEqual(
            batching_policy_name("wavlm"),
            "native_padded_batch",
        )
        self.assertEqual(
            batching_policy_name("w2v_bert2"),
            "native_padded_batch",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
