from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from papr_ssl.models.teacher.backbones.w2v_bert2 import W2vBert2Backbone


class _FakeFeatureExtractor:
    def __init__(self) -> None:
        self.called = False

    def __call__(self, raw_speech: object, **kwargs: object) -> dict:
        self.called = True
        self.raw_speech = raw_speech
        self.kwargs = kwargs
        batch_size = len(raw_speech)
        input_features = torch.arange(
            batch_size * 5 * 8, dtype=torch.float32
        ).reshape(batch_size, 5, 8)
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 0]] * batch_size, dtype=torch.long
        )
        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }


class _FakeW2vBertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=3, hidden_size=6)
        self.anchor = nn.Parameter(torch.ones(1))
        self.received_input_features = None
        self.received_attention_mask = None

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        self.received_input_features = input_features
        self.received_attention_mask = attention_mask
        states = tuple(
            torch.full((input_features.shape[0], input_features.shape[1], 6), float(index))
            for index in range(4)
        )
        return SimpleNamespace(hidden_states=states)


class W2vBert2BackboneTest(unittest.TestCase):
    def test_feature_extractor_to_input_features_pipeline(self) -> None:
        extractor = _FakeFeatureExtractor()
        model = _FakeW2vBertModel()
        backbone = W2vBert2Backbone(
            hidden_layer=2,
            model=model,
            feature_extractor=extractor,
        )
        output = backbone(torch.zeros(2, 16_000, dtype=torch.float32))

        self.assertTrue(extractor.called)
        self.assertEqual(extractor.kwargs["sampling_rate"], 16_000)
        self.assertEqual(tuple(model.received_input_features.shape), (2, 5, 8))
        self.assertEqual(tuple(model.received_attention_mask.shape), (2, 5))
        self.assertEqual(tuple(output.features.shape), (2, 5, 6))
        self.assertEqual(output.hidden_layer, 2)
        self.assertTrue(torch.equal(output.frame_mask, torch.tensor(
            [[True, True, True, True, False], [True, True, True, True, False]]
        )))

    def test_missing_feature_extractor_fails_without_download(self) -> None:
        with self.assertRaisesRegex(ValueError, "injected together"):
            W2vBert2Backbone(model=_FakeW2vBertModel())

    def test_hidden_layer_out_of_range_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            W2vBert2Backbone(
                hidden_layer=4,
                model=_FakeW2vBertModel(),
                feature_extractor=_FakeFeatureExtractor(),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
