"""P3-05 closed-set embedding diagnostics on development data.

Checkpoint selection in P3/P4 uses one frozen scalar:

    generic_dev_score := prototype_macro_f1

where class prototypes are computed from TRAIN embeddings only, and DEV is
classified by cosine similarity to those train prototypes.

Open-set thresholding is intentionally not part of P3-05; that belongs to P6.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


GENERIC_DEV_SCORE_NAME = "prototype_macro_f1"


@dataclass(frozen=True)
class DevEmbeddingMetrics:
    sample_count: int
    class_count: int
    prototype_accuracy: float
    prototype_macro_f1: float
    intra_class_cosine_mean: float
    wrong_class_cosine_max_mean: float
    top1_top2_margin_mean: float
    prototype_inter_cosine_mean: float
    embedding_norm_mean: float
    embedding_norm_max_abs_error: float
    embedding_std_mean: float
    collapsed: bool
    generic_dev_score: float
    generic_dev_score_name: str = GENERIC_DEV_SCORE_NAME

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_embedding(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "embedding"):
        value = output.embedding
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, Mapping):
        value = output.get("embedding")
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(
        "Mean DR output must be a Tensor or expose Tensor field `embedding`."
    )


def cached_mean_dr_forward(
    mean_dr: nn.Module,
    cached_features: torch.Tensor,
) -> torch.Tensor:
    """Run the existing Mean DR on already pooled P3-02 vectors.

    P3-02 stores masked-mean vectors [B,D]. To reuse the project's canonical
    MeanDR implementation without duplicating its projection/L2 logic, each
    cached vector is represented as a one-frame sequence [B,1,D] with a valid
    [B,1] mask. Masked mean of a one-frame sequence is exactly that vector.
    """
    if cached_features.ndim != 2:
        raise ValueError("cached_features must have shape [B,D]")
    x = cached_features.to(torch.float32).unsqueeze(1)
    mask = torch.ones(
        (x.shape[0], 1),
        dtype=torch.bool,
        device=x.device,
    )
    embedding = _extract_embedding(mean_dr(x, mask))
    if embedding.ndim != 2 or embedding.shape[1] != 64:
        raise RuntimeError(
            f"Mean DR must output [B,64], got {tuple(embedding.shape)}"
        )
    if not torch.isfinite(embedding).all():
        raise RuntimeError("Mean DR produced NaN/Inf embeddings")
    return embedding


def assert_unit_norm(
    embedding: torch.Tensor,
    *,
    atol: float = 1e-4,
) -> None:
    if embedding.ndim != 2 or embedding.shape[1] != 64:
        raise RuntimeError("embedding must have shape [B,64]")
    norms = torch.linalg.vector_norm(embedding, dim=1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=atol,
        rtol=0.0,
    ):
        max_err = torch.max(torch.abs(norms - 1.0)).item()
        raise RuntimeError(
            "64D unit-norm contract violated; "
            f"max_abs_error={max_err:.6g}"
        )


@torch.inference_mode()
def encode_loader(
    mean_dr: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = mean_dr.training
    mean_dr.eval()

    zs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for batch in loader:
        features = batch["features"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        labels = batch["labels"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        z = cached_mean_dr_forward(mean_dr, features)
        assert_unit_norm(z)
        zs.append(z.cpu())
        ys.append(labels.cpu())

    if was_training:
        mean_dr.train()

    if not zs:
        raise RuntimeError("evaluation loader produced no batches")
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def build_train_prototypes(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    if train_embeddings.ndim != 2 or train_embeddings.shape[1] != 64:
        raise ValueError("train_embeddings must be [N,64]")
    if train_labels.ndim != 1:
        raise ValueError("train_labels must be [N]")
    if train_embeddings.shape[0] != train_labels.shape[0]:
        raise ValueError("embedding/label count mismatch")

    prototypes: list[torch.Tensor] = []
    for class_id in range(num_classes):
        mask = train_labels == class_id
        if not torch.any(mask):
            raise RuntimeError(
                f"train reference set has no samples for class {class_id}"
            )
        proto = train_embeddings[mask].mean(dim=0)
        if torch.linalg.vector_norm(proto).item() <= 1e-12:
            raise RuntimeError(
                f"zero-norm train prototype for class {class_id}"
            )
        prototypes.append(F.normalize(proto, dim=0))
    return torch.stack(prototypes, dim=0)


def _macro_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    num_classes: int,
) -> float:
    f1s: list[float] = []
    for c in range(num_classes):
        true_c = y_true == c
        pred_c = y_pred == c
        tp = int(torch.sum(true_c & pred_c).item())
        fp = int(torch.sum(~true_c & pred_c).item())
        fn = int(torch.sum(true_c & ~pred_c).item())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(sum(f1s) / len(f1s))


def _prototype_inter_cosine_mean(
    prototypes: torch.Tensor,
) -> float:
    c = prototypes.shape[0]
    if c <= 1:
        return 0.0
    sim = prototypes @ prototypes.T
    mask = ~torch.eye(c, dtype=torch.bool)
    return float(sim[mask].mean().item())


def evaluate_embedding_space(
    *,
    mean_dr: nn.Module,
    train_reference_loader: Iterable[Mapping[str, Any]],
    dev_loader: Iterable[Mapping[str, Any]],
    num_classes: int,
    device: torch.device,
) -> DevEmbeddingMetrics:
    """Evaluate DEV only using prototypes computed from TRAIN only."""
    train_z, train_y = encode_loader(
        mean_dr,
        train_reference_loader,
        device=device,
    )
    dev_z, dev_y = encode_loader(
        mean_dr,
        dev_loader,
        device=device,
    )

    prototypes = build_train_prototypes(
        train_z,
        train_y,
        num_classes=num_classes,
    )

    similarity = dev_z @ prototypes.T
    pred = similarity.argmax(dim=1)
    accuracy = float((pred == dev_y).float().mean().item())
    macro_f1 = _macro_f1(
        dev_y,
        pred,
        num_classes=num_classes,
    )

    own = similarity[
        torch.arange(dev_y.shape[0]),
        dev_y,
    ]

    wrong = similarity.clone()
    wrong[
        torch.arange(dev_y.shape[0]),
        dev_y,
    ] = float("-inf")
    wrong_max = wrong.max(dim=1).values

    if num_classes >= 2:
        top2 = torch.topk(similarity, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        margin_mean = float(margin.mean().item())
    else:
        margin_mean = 0.0

    norms = torch.linalg.vector_norm(dev_z, dim=1)
    std_mean = float(dev_z.std(dim=0, unbiased=False).mean().item())
    collapsed = bool(std_mean < 1e-6)

    return DevEmbeddingMetrics(
        sample_count=int(dev_z.shape[0]),
        class_count=int(num_classes),
        prototype_accuracy=accuracy,
        prototype_macro_f1=macro_f1,
        intra_class_cosine_mean=float(own.mean().item()),
        wrong_class_cosine_max_mean=float(wrong_max.mean().item()),
        top1_top2_margin_mean=margin_mean,
        prototype_inter_cosine_mean=_prototype_inter_cosine_mean(
            prototypes
        ),
        embedding_norm_mean=float(norms.mean().item()),
        embedding_norm_max_abs_error=float(
            torch.max(torch.abs(norms - 1.0)).item()
        ),
        embedding_std_mean=std_mean,
        collapsed=collapsed,
        generic_dev_score=macro_f1,
    )
