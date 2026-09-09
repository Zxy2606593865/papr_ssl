from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from papr_ssl.models.teacher.heads.mean_dr import MeanDR


class MeanDRTest(unittest.TestCase):
    def test_masked_mean_and_projection(self) -> None:
        head = MeanDR(input_dim=2, output_dim=2)
        with torch.no_grad():
            head.projection.weight.copy_(torch.eye(2))
            head.projection.bias.zero_()

        features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])

        pooled = head.masked_mean(features, mask)
        embedding = head(features, mask)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 3.0]])))
        self.assertTrue(torch.allclose(embedding, F.normalize(pooled, dim=1)))

    def test_all_mask_fails(self) -> None:
        head = MeanDR(input_dim=3)
        with self.assertRaisesRegex(ValueError, "all-mask"):
            head(torch.randn(2, 4, 3), torch.zeros(2, 4, dtype=torch.bool))

    def test_shared_head_supports_different_backbone_dimensions(self) -> None:
        for input_dim in (4, 7, 13):
            with self.subTest(input_dim=input_dim):
                head = MeanDR(input_dim=input_dim, output_dim=64)
                output = head(
                    torch.randn(2, 5, input_dim),
                    torch.ones(2, 5, dtype=torch.bool),
                )
                self.assertEqual(tuple(output.shape), (2, 64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
