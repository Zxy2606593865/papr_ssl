from __future__ import annotations

import math
import unittest

import torch

from papr_ssl.losses.scaf import SubCenterArcFaceLoss


class SubCenterArcFaceLossTest(unittest.TestCase):
    def test_logits_shape(self) -> None:
        torch.manual_seed(17)
        criterion = SubCenterArcFaceLoss(
            num_classes=2,
            embedding_dim=64,
            subcenters=3,
        )
        embeddings = torch.randn(4, 64)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        loss, logits = criterion(embeddings, labels, return_logits=True)
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_margin_changes_only_correct_class(self) -> None:
        margin = 0.2
        scale = 30.0
        criterion = SubCenterArcFaceLoss(
            num_classes=2,
            embedding_dim=64,
            subcenters=3,
            margin_rad=margin,
            scale=scale,
        )
        with torch.no_grad():
            criterion.centers.zero_()
            criterion.centers[0, :, 0] = 1.0
            criterion.centers[1, :, 1] = 1.0

        embedding = torch.zeros(1, 64)
        embedding[0, 0] = math.cos(0.4)
        embedding[0, 1] = math.sin(0.4)
        labels = torch.tensor([0], dtype=torch.long)

        base = criterion.compute_logits(embedding, apply_margin=False)
        with_margin = criterion.compute_logits(embedding, labels, apply_margin=True)

        self.assertAlmostEqual(with_margin[0, 1].item(), base[0, 1].item(), places=5)
        self.assertLess(with_margin[0, 0].item(), base[0, 0].item())
        self.assertAlmostEqual(
            with_margin[0, 0].item(), math.cos(0.4 + margin) * scale, places=4
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
