from __future__ import annotations

import unittest

import torch

from papr_ssl.models.teacher.backbones.base import SSLBackboneOutput


class SSLBackboneOutputTest(unittest.TestCase):
    def test_common_contract(self) -> None:
        output = SSLBackboneOutput(
            features=torch.randn(2, 5, 7, dtype=torch.float32),
            frame_mask=torch.tensor(
                [[True, True, True, True, False], [True, True, True, True, True]]
            ),
            hidden_layer=3,
        )
        self.assertEqual(tuple(output.features.shape), (2, 5, 7))
        self.assertEqual(output.features.dtype, torch.float32)
        self.assertEqual(output.frame_mask.dtype, torch.bool)
        self.assertEqual(output.hidden_layer, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
