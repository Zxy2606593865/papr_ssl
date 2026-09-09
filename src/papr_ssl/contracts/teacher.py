"""Framework-level contract for final Teacher embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 16_000
EMBEDDING_DIM = 64


@dataclass(frozen=True)
class TeacherOutput:
    """Validated output of a Teacher encoder.

    The contract is intentionally independent of backbone, representation head,
    and training loss choices.
    """

    embedding: Tensor
    frame_count: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.embedding, Tensor):
            raise TypeError("embedding must be a torch.Tensor")
        if self.embedding.ndim != 2 or self.embedding.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"embedding must have shape [B,{EMBEDDING_DIM}], "
                f"got {tuple(self.embedding.shape)}"
            )
        if self.embedding.dtype != torch.float32:
            raise TypeError("embedding must use torch.float32")
        if not bool(torch.isfinite(self.embedding).all()):
            raise ValueError("embedding contains NaN or Inf")

        norms = torch.linalg.vector_norm(self.embedding, dim=1)
        if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)):
            raise ValueError("embedding rows must have L2 unit norm")

        if not isinstance(self.frame_count, Tensor):
            raise TypeError("frame_count must be a torch.Tensor")
        if self.frame_count.ndim != 1 or self.frame_count.shape[0] != self.embedding.shape[0]:
            raise ValueError("frame_count must have shape [B]")
        if self.frame_count.dtype not in (torch.int32, torch.int64):
            raise TypeError("frame_count must use an integer dtype")
        if not bool((self.frame_count > 0).all()):
            raise ValueError("frame_count must be positive")
