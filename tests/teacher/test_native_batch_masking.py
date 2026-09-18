from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from papr_ssl.models.teacher.backbones.base import (
    sample_mask_to_lengths,
    validate_waveforms,
)
from papr_ssl.models.teacher.backbones.w2v_bert2 import W2vBert2Backbone


class FakeConfig:
    num_hidden_layers = 1
    hidden_size = 4


class FakeW2vBertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = FakeConfig()
        self.last_attention_mask = None
        self.calls = 0

    def forward(
        self,
        *,
        input_features,
        attention_mask,
        output_hidden_states,
        return_dict,
    ):
        self.calls += 1
        self.last_attention_mask = attention_mask.detach().clone()

        x = input_features[..., :1].repeat(1, 1, 4)
        h0 = x.to(torch.float32)
        h1 = (x + 1.0).to(torch.float32)
        return SimpleNamespace(hidden_states=(h0, h1))


class FakeFeatureExtractor:
    def __init__(self) -> None:
        self.seen_lengths = []

    def __call__(
        self,
        raw_speech,
        *,
        sampling_rate,
        return_tensors,
        padding,
        return_attention_mask,
    ):
        self.seen_lengths = [len(x) for x in raw_speech]
        frame_lengths = [max(1, len(x) // 2) for x in raw_speech]
        max_t = max(frame_lengths)

        feats = torch.zeros(
            (len(raw_speech), max_t, 2),
            dtype=torch.float32,
        )
        mask = torch.zeros(
            (len(raw_speech), max_t),
            dtype=torch.long,
        )
        for i, (raw, t) in enumerate(zip(raw_speech, frame_lengths)):
            value = float(np.asarray(raw).mean()) if len(raw) else 0.0
            feats[i, :t, 0] = value
            feats[i, :t, 1] = float(t)
            mask[i, :t] = 1

        return {
            "input_features": feats,
            "attention_mask": mask,
        }


class NativeMaskContractTest(unittest.TestCase):
    def test_sample_mask_to_lengths(self) -> None:
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        self.assertEqual(sample_mask_to_lengths(mask).tolist(), [4, 2])

    def test_non_prefix_sample_mask_rejected(self) -> None:
        waveforms = torch.zeros((1, 6), dtype=torch.float32)
        mask = torch.tensor(
            [[1, 1, 0, 1, 0, 0]],
            dtype=torch.bool,
        )
        with self.assertRaises(ValueError):
            validate_waveforms(waveforms, mask)

    def test_w2v_bert2_native_batch_trims_raw_rows_before_extractor(self) -> None:
        model = FakeW2vBertModel()
        extractor = FakeFeatureExtractor()
        backbone = W2vBert2Backbone(
            model_name="fake",
            hidden_layer=1,
            freeze_backbone=True,
            model=model,
            feature_extractor=extractor,
        )

        waveforms = torch.zeros((2, 8), dtype=torch.float32)
        waveforms[0, :8] = 1.0
        waveforms[1, :4] = 2.0
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )

        out = backbone(waveforms, sample_mask=mask)

        # Variable-length raw arrays are passed in ONE extractor call.
        self.assertEqual(extractor.seen_lengths, [8, 4])

        # The neural model itself is also called once for the whole batch.
        self.assertEqual(model.calls, 1)

        self.assertEqual(tuple(out.features.shape), (2, 4, 4))
        self.assertEqual(out.frame_mask.sum(dim=1).tolist(), [4, 2])
        self.assertEqual(
            model.last_attention_mask.sum(dim=1).tolist(),
            [4, 2],
        )

    def test_w2v_bert2_without_mask_preserves_old_full_row_behavior(self) -> None:
        model = FakeW2vBertModel()
        extractor = FakeFeatureExtractor()
        backbone = W2vBert2Backbone(
            model_name="fake",
            hidden_layer=1,
            freeze_backbone=True,
            model=model,
            feature_extractor=extractor,
        )

        waveforms = torch.zeros((2, 8), dtype=torch.float32)
        out = backbone(waveforms)

        self.assertEqual(extractor.seen_lengths, [8, 8])
        self.assertEqual(out.frame_mask.sum(dim=1).tolist(), [4, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
