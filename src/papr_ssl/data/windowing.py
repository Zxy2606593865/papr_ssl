"""Minimal fixed-window data protocol used by Phase T0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from papr_ssl.contracts.teacher import SAMPLE_RATE, WINDOW_SAMPLES


@dataclass(frozen=True)
class WindowBatch:
    """A batch of normalized one-second speech windows."""

    waveforms: Tensor
    sample_ids: Tuple[str, ...]
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        if not isinstance(self.waveforms, Tensor):
            raise TypeError("waveforms must be a torch.Tensor")
        if self.waveforms.ndim != 2 or self.waveforms.shape[1] != WINDOW_SAMPLES:
            raise ValueError(
                f"waveforms must have shape [B,{WINDOW_SAMPLES}], "
                f"got {tuple(self.waveforms.shape)}"
            )
        if self.waveforms.shape[0] == 0:
            raise ValueError("waveforms batch must be non-empty")
        if self.waveforms.dtype != torch.float32:
            raise TypeError("waveforms must use torch.float32")
        if not bool(torch.isfinite(self.waveforms).all()):
            raise ValueError("waveforms contain NaN or Inf")
        if self.waveforms.numel() and float(self.waveforms.abs().max()) > 1.0:
            raise ValueError("waveform samples must lie in [-1, 1]")
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {SAMPLE_RATE}")
        if len(self.sample_ids) != self.waveforms.shape[0]:
            raise ValueError("sample_ids length must equal batch size")

        cleaned = tuple(str(item).strip() for item in self.sample_ids)
        if any(not item for item in cleaned):
            raise ValueError("sample_ids must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("sample_ids must be unique within a batch")
        object.__setattr__(self, "sample_ids", cleaned)
