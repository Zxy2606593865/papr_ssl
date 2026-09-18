"""P3-05 Mean DR + SCAF training on offline SSL feature caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from papr_ssl.training.teacher.evaluate import (
    cached_mean_dr_forward,
    assert_unit_norm,
    evaluate_embedding_space,
)


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0")


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be >= 0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _construct_by_signature(
    cls: type[nn.Module],
    semantic_values: Mapping[str, Any],
    aliases: Mapping[str, tuple[str, ...]],
) -> nn.Module:
    sig = inspect.signature(cls)
    kwargs: dict[str, Any] = {}

    reverse: dict[str, str] = {}
    for semantic, names in aliases.items():
        for name in names:
            reverse[name] = semantic

    for name, p in sig.parameters.items():
        if name == "self":
            continue
        semantic = reverse.get(name)
        if semantic is not None:
            kwargs[name] = semantic_values[semantic]
            continue
        if p.default is inspect.Parameter.empty and p.kind not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise RuntimeError(
                f"Cannot construct {cls.__name__}: unsupported required "
                f"constructor parameter {name!r}; signature={sig}"
            )
    return cls(**kwargs)


def build_project_mean_dr(
    *,
    input_dim: int,
    embedding_dim: int = 64,
) -> nn.Module:
    from papr_ssl.models.teacher.heads.mean_dr import MeanDR

    return _construct_by_signature(
        MeanDR,
        {
            "input_dim": input_dim,
            "embedding_dim": embedding_dim,
        },
        {
            "input_dim": (
                "input_dim",
                "in_dim",
                "in_features",
                "feature_dim",
                "hidden_dim",
            ),
            "embedding_dim": (
                "embedding_dim",
                "output_dim",
                "out_dim",
                "out_features",
            ),
        },
    )


def build_project_scaf(
    *,
    num_classes: int,
    embedding_dim: int = 64,
    k: int = 3,
    margin: float = 0.2,
    scale: float = 30.0,
) -> nn.Module:
    from papr_ssl.losses.scaf import SubCenterArcFaceLoss

    return _construct_by_signature(
        SubCenterArcFaceLoss,
        {
            "num_classes": num_classes,
            "embedding_dim": embedding_dim,
            "k": k,
            "margin": margin,
            "scale": scale,
        },
        {
            "num_classes": (
                "num_classes",
                "class_count",
                "n_classes",
            ),
            "embedding_dim": (
                "embedding_dim",
                "feature_dim",
                "dim",
                "in_features",
            ),
            "k": (
                "k",
                "num_subcenters",
                "num_sub_centers",
                "subcenters",
            ),
            "margin": ("margin", "m"),
            "scale": ("scale", "s"),
        },
    )


def scaf_loss(
    scaf: nn.Module,
    embedding: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Normalize several reasonable existing SCAF return conventions."""
    result = scaf(embedding, labels)

    if isinstance(result, torch.Tensor):
        if result.ndim == 0:
            return result
        raise RuntimeError(
            "SCAF returned a non-scalar Tensor; expected scalar loss"
        )

    if isinstance(result, Mapping):
        value = result.get("loss")
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            return value

    if isinstance(result, (tuple, list)):
        for value in result:
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                return value

    if hasattr(result, "loss"):
        value = result.loss
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            return value

    raise TypeError(
        "Could not extract scalar SCAF loss from forward result "
        f"{type(result).__name__}"
    )


def checkpoint_sha256(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def train_one_epoch(
    *,
    mean_dr: nn.Module,
    scaf: nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> dict[str, float]:
    mean_dr.train()
    scaf.train()

    total_loss = 0.0
    total_examples = 0

    for batch in train_loader:
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

        optimizer.zero_grad(set_to_none=True)
        z = cached_mean_dr_forward(mean_dr, features)
        assert_unit_norm(z)
        loss = scaf_loss(scaf, z, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("non-finite train loss")

        loss.backward()

        params = [
            p
            for module in (mean_dr, scaf)
            for p in module.parameters()
            if p.requires_grad
        ]
        if not params:
            raise RuntimeError("Mean DR + SCAF expose no trainable parameters")

        torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=grad_clip_norm,
        )
        optimizer.step()

        b = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * b
        total_examples += b

    if total_examples == 0:
        raise RuntimeError("train loader produced zero examples")

    return {
        "train_loss": total_loss / total_examples,
        "train_examples": float(total_examples),
    }


def save_epoch_checkpoint(
    *,
    path: str | Path,
    epoch: int,
    mean_dr: nn.Module,
    scaf: nn.Module,
    optimizer: torch.optim.Optimizer,
    seed: int,
    train_metrics: Mapping[str, Any],
    dev_metrics: Mapping[str, Any],
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "papr_ssl.teacher_checkpoint.v1",
        "phase": "P3-05",
        "epoch": int(epoch),
        "seed": int(seed),
        "mean_dr_state_dict": mean_dr.state_dict(),
        "scaf_state_dict": scaf.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": dict(train_metrics),
        "dev_metrics": dict(dev_metrics),
    }
    torch.save(payload, path)
    return checkpoint_sha256(path)


def run_training(
    *,
    mean_dr: nn.Module,
    scaf: nn.Module,
    train_loader: Any,
    train_reference_loader: Iterable[Mapping[str, Any]],
    dev_loader: Iterable[Mapping[str, Any]],
    num_classes: int,
    seed: int,
    run_dir: str | Path,
    device: torch.device,
    optimization: OptimizationConfig,
) -> list[dict[str, Any]]:
    """Train all epochs and save every epoch; no checkpoint selection here."""
    optimization.validate()
    seed_everything(seed)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    hashes_path = run_dir / "checkpoint_hashes.jsonl"

    # Avoid accidentally appending to an old run.
    if metrics_path.exists() or hashes_path.exists():
        raise FileExistsError(
            "run_dir already contains P3-05 metrics/hash logs; "
            "use a new immutable run directory"
        )

    mean_dr.to(device)
    scaf.to(device)

    trainable = [
        p
        for module in (mean_dr, scaf)
        for p in module.parameters()
        if p.requires_grad
    ]
    if not trainable:
        raise RuntimeError("no trainable Mean DR/SCAF parameters")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )

    history: list[dict[str, Any]] = []

    for epoch in range(1, optimization.epochs + 1):
        sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch - 1)

        train_metrics = train_one_epoch(
            mean_dr=mean_dr,
            scaf=scaf,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=optimization.grad_clip_norm,
        )

        dev = evaluate_embedding_space(
            mean_dr=mean_dr,
            train_reference_loader=train_reference_loader,
            dev_loader=dev_loader,
            num_classes=num_classes,
            device=device,
        )
        dev_metrics = dev.to_dict()

        checkpoint_path = (
            run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
        )
        sha = save_epoch_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            mean_dr=mean_dr,
            scaf=scaf,
            optimizer=optimizer,
            seed=seed,
            train_metrics=train_metrics,
            dev_metrics=dev_metrics,
        )

        row = {
            "epoch": epoch,
            **train_metrics,
            "dev": dev_metrics,
            "checkpoint": checkpoint_path.as_posix(),
            "checkpoint_sha256": sha,
        }
        _append_jsonl(metrics_path, row)
        _append_jsonl(
            hashes_path,
            {
                "epoch": epoch,
                "checkpoint": checkpoint_path.as_posix(),
                "sha256": sha,
            },
        )
        history.append(row)

    return history
