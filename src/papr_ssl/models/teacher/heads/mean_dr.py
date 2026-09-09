"""Masked-mean dimensionality-reduction baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MeanDR(nn.Module):
    """Masked temporal mean followed by a linear projection and L2 norm."""

    def __init__(self, input_dim: int, output_dim: int = 64, eps: float = 1e-12) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.eps = float(eps)
        self.projection = nn.Linear(self.input_dim, self.output_dim)

    def masked_mean(self, features: Tensor, mask: Tensor) -> Tensor:
        self._validate_inputs(features, mask)
        valid_counts = mask.sum(dim=1)
        if not bool((valid_counts > 0).all()):
            invalid = torch.nonzero(valid_counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(f"all-mask samples are not allowed; indices={invalid}")
        weights = mask.to(dtype=features.dtype).unsqueeze(-1)
        return (features * weights).sum(dim=1) / valid_counts.to(features.dtype).unsqueeze(1)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        pooled = self.masked_mean(features, mask)
        projected = self.projection(pooled)
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("projected embeddings contain NaN or Inf")
        norms = torch.linalg.vector_norm(projected, dim=1)
        if not bool((norms > self.eps).all()):
            invalid = torch.nonzero(norms <= self.eps, as_tuple=False).flatten().tolist()
            raise ValueError(f"cannot normalize zero embeddings; indices={invalid}")
        return F.normalize(projected, p=2, dim=1, eps=self.eps)

    def _validate_inputs(self, features: Tensor, mask: Tensor) -> None:
        if not isinstance(features, Tensor) or features.ndim != 3:
            raise ValueError("features must have shape [B,T,D]")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"feature dimension must be {self.input_dim}, got {features.shape[-1]}"
            )
        if not torch.is_floating_point(features):
            raise TypeError("features must use a floating-point dtype")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features contain NaN or Inf")
        if not isinstance(mask, Tensor) or mask.shape != features.shape[:2]:
            raise ValueError("mask must have shape [B,T]")
        if mask.dtype != torch.bool:
            raise TypeError("mask must use torch.bool")
