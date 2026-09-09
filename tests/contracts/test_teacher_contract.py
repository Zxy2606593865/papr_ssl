from __future__ import annotations

import unittest

import torch

from papr_ssl.contracts.teacher import TeacherOutput


class TeacherContractTest(unittest.TestCase):
    def test_valid_teacher_output(self) -> None:
        embedding = torch.nn.functional.normalize(torch.randn(3, 64), dim=1)
        output = TeacherOutput(
            embedding=embedding.to(torch.float32),
            frame_count=torch.tensor([49, 50, 48], dtype=torch.int64),
        )
        self.assertEqual(tuple(output.embedding.shape), (3, 64))
        self.assertEqual(output.embedding.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(output.embedding).all()))
        self.assertTrue(
            bool(
                torch.allclose(
                    torch.linalg.vector_norm(output.embedding, dim=1),
                    torch.ones(3),
                    atol=1e-5,
                    rtol=1e-5,
                )
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
