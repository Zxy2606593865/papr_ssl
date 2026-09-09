from __future__ import annotations

import unittest

import torch

from papr_ssl.data.windowing import WindowBatch


class WindowBatchTest(unittest.TestCase):
    def test_fixed_window_contract(self) -> None:
        batch = WindowBatch(
            waveforms=torch.zeros(2, 16_000, dtype=torch.float32),
            sample_ids=("window-a", "window-b"),
        )
        self.assertEqual(tuple(batch.waveforms.shape), (2, 16_000))
        self.assertEqual(batch.sample_rate, 16_000)

    def test_out_of_range_samples_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[-1, 1\]"):
            WindowBatch(
                waveforms=torch.full((1, 16_000), 1.1, dtype=torch.float32),
                sample_ids=("bad-window",),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
