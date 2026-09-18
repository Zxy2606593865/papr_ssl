"""P5 dimensionality-reduction heads: Mean DR and Attention DR."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MeanDR(nn.Module):
    def __init__(self, input_dim: int = 1024, embedding_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(input_dim, embedding_dim)

    def forward(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = frame_mask.unsqueeze(-1).to(features.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (features * mask).sum(dim=1) / denom
        z = self.proj(pooled)
        return F.normalize(z, p=2, dim=-1)


class AttentionDR(nn.Module):
    """Minimal additive temporal attention pooling followed by 64D projection.

    score_t = v^T tanh(W h_t)
    alpha = softmax(masked score)
    pooled = sum_t alpha_t h_t
    z = L2Norm(Linear(pooled))
    """

    def __init__(
        self,
        input_dim: int = 1024,
        embedding_dim: int = 64,
        attention_dim: int = 128,
    ):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )
        self.proj = nn.Linear(input_dim, embedding_dim)

    def forward(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.attn(features).squeeze(-1)
        scores = scores.masked_fill(~frame_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)
        pooled = torch.sum(features * alpha.unsqueeze(-1), dim=1)
        z = self.proj(pooled)
        return F.normalize(z, p=2, dim=-1)


def build_dr(method: str) -> nn.Module:
    if method == "mean":
        return MeanDR()
    if method == "attention":
        return AttentionDR()
    raise ValueError(f"unknown P5 DR method: {method}")
