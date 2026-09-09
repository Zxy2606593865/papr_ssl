"""Numerically stable Sub-center ArcFace (SCAF) training loss."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SubCenterArcFaceLoss(nn.Module):
    """Sub-center ArcFace classification loss.

    Centers have shape `[C,K,D]`. Similarities are reduced with a maximum over
    `K` independently for each class, yielding `[B,C]`. The angular margin is
    then applied only to each sample's correct class.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 64,
        subcenters: int = 3,
        margin_rad: float = 0.2,
        scale: float = 30.0,
        eps: float = 1e-7,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if subcenters <= 0:
            raise ValueError("subcenters must be positive")
        if not 0.0 <= margin_rad < math.pi / 2:
            raise ValueError("margin_rad must be in [0, pi/2)")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if not 0.0 < eps < 1.0:
            raise ValueError("eps must be in (0,1)")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")

        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.subcenters = int(subcenters)
        self.margin_rad = float(margin_rad)
        self.scale = float(scale)
        self.eps = float(eps)
        self.reduction = reduction

        self.centers = nn.Parameter(
            torch.empty(self.num_classes, self.subcenters, self.embedding_dim)
        )
        nn.init.normal_(self.centers, mean=0.0, std=0.01)

    def compute_logits(
        self,
        embeddings: Tensor,
        labels: Optional[Tensor] = None,
        *,
        apply_margin: bool = True,
    ) -> Tensor:
        """Return scaled `[B,C]` logits, optionally without target margin."""

        self._validate_embeddings(embeddings)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1, eps=self.eps)

        center_norms = torch.linalg.vector_norm(self.centers, dim=-1)
        if not bool((center_norms > self.eps).all()):
            raise ValueError("SCAF contains a zero-norm center")
        normalized_centers = F.normalize(self.centers, p=2, dim=-1, eps=self.eps)

        cos_all = torch.einsum("bd,ckd->bck", normalized_embeddings, normalized_centers)
        cos_class = cos_all.amax(dim=2)

        if not apply_margin:
            return cos_class * self.scale

        checked_labels = self._validate_labels(labels, embeddings.shape[0])
        stable_cosine = cos_class.clamp(min=-1.0 + self.eps, max=1.0 - self.eps)
        target_cosine = stable_cosine.gather(1, checked_labels[:, None]).squeeze(1)

        sine = torch.sqrt(torch.clamp(1.0 - target_cosine.square(), min=self.eps))
        phi = target_cosine * math.cos(self.margin_rad) - sine * math.sin(self.margin_rad)

        threshold = math.cos(math.pi - self.margin_rad)
        monotonic_correction = math.sin(math.pi - self.margin_rad) * self.margin_rad
        phi = torch.where(
            target_cosine > threshold,
            phi,
            target_cosine - monotonic_correction,
        )

        margin_logits = cos_class.clone()
        margin_logits.scatter_(1, checked_labels[:, None], phi[:, None])
        return margin_logits * self.scale

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        *,
        return_logits: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        labels = self._validate_labels(labels, embeddings.shape[0])
        logits = self.compute_logits(embeddings, labels, apply_margin=True)
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        if return_logits:
            return loss, logits
        return loss

    def _validate_embeddings(self, embeddings: Tensor) -> None:
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [B,D]")
        if embeddings.shape[0] == 0:
            raise ValueError("embeddings batch must be non-empty")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embedding dimension must be {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        if not torch.is_floating_point(embeddings):
            raise TypeError("embeddings must use a floating-point dtype")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("embeddings contain NaN or Inf")
        if not bool((torch.linalg.vector_norm(embeddings, dim=1) > self.eps).all()):
            raise ValueError("embeddings must not contain zero vectors")

    def _validate_labels(self, labels: Optional[Tensor], batch_size: int) -> Tensor:
        if not isinstance(labels, Tensor):
            raise TypeError("labels must be a torch.Tensor")
        if labels.ndim != 1 or labels.shape[0] != batch_size:
            raise ValueError("labels must have shape [B]")
        if labels.dtype != torch.long:
            raise TypeError("labels must use torch.long")
        if not bool(((labels >= 0) & (labels < self.num_classes)).all()):
            raise ValueError("labels contain an out-of-range class index")
        return labels
