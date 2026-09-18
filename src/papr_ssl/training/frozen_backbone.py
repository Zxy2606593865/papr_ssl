"""P3-01 contract: frozen SSL backbone.

The backbone is treated as a fixed feature extractor:
- every parameter has requires_grad=False;
- actual backbone forward execution is forced into eval mode;
- optimizer construction must exclude all backbone parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class FrozenBackboneState:
    total_parameters: int
    trainable_parameters: int
    training: bool


def _force_eval_before_forward(
    module: nn.Module,
    _inputs: tuple[object, ...],
) -> None:
    """Guarantee eval semantics immediately before every backbone forward."""
    module.eval()


def freeze_ssl_backbone(
    backbone: nn.Module,
    *,
    install_eval_guard: bool = True,
) -> nn.Module:
    """Freeze an SSL backbone in-place.

    This operation is intentionally idempotent.
    """
    backbone.requires_grad_(False)
    backbone.eval()

    if install_eval_guard and not hasattr(
        backbone, "_papr_ssl_eval_guard_handle"
    ):
        handle = backbone.register_forward_pre_hook(
            _force_eval_before_forward
        )
        # A normal Python attribute is sufficient; it is not part of state_dict.
        setattr(backbone, "_papr_ssl_eval_guard_handle", handle)

    assert_ssl_backbone_frozen(backbone)
    return backbone


def assert_ssl_backbone_frozen(backbone: nn.Module) -> None:
    """Fail fast if any SSL parameter is trainable."""
    trainable = [
        name
        for name, param in backbone.named_parameters()
        if param.requires_grad
    ]
    if trainable:
        preview = ", ".join(trainable[:8])
        raise RuntimeError(
            "P3-01 violation: SSL backbone contains trainable parameters: "
            f"{preview}"
        )


def force_ssl_backbone_eval(backbone: nn.Module) -> None:
    """Re-assert eval mode after a surrounding model.train() call."""
    backbone.eval()


def frozen_backbone_state(backbone: nn.Module) -> FrozenBackboneState:
    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(
        p.numel() for p in backbone.parameters() if p.requires_grad
    )
    return FrozenBackboneState(
        total_parameters=total,
        trainable_parameters=trainable,
        training=backbone.training,
    )


def trainable_parameters(
    module: nn.Module,
) -> Iterable[nn.Parameter]:
    """Yield only parameters that are allowed to be optimized."""
    for param in module.parameters():
        if param.requires_grad:
            yield param


def assert_optimizer_excludes_backbone(
    optimizer: torch.optim.Optimizer,
    backbone: nn.Module,
) -> None:
    """Ensure no backbone Parameter object is present in the optimizer."""
    backbone_ids = {id(p) for p in backbone.parameters()}
    optimizer_ids = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
    }

    overlap = backbone_ids & optimizer_ids
    if overlap:
        raise RuntimeError(
            "P3-01 violation: optimizer contains SSL backbone parameters."
        )


def snapshot_parameters(
    module: nn.Module,
) -> dict[str, torch.Tensor]:
    """CPU clone used by the P3-01 mutation gate."""
    return {
        name: param.detach().cpu().clone()
        for name, param in module.named_parameters()
    }


def assert_parameters_unchanged(
    before: dict[str, torch.Tensor],
    module: nn.Module,
) -> None:
    """Bitwise check that frozen parameters did not change."""
    after = {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
    }

    if set(before) != set(after):
        raise RuntimeError(
            "P3-01 violation: backbone parameter names changed."
        )

    changed = [
        name
        for name in before
        if not torch.equal(before[name], after[name])
    ]
    if changed:
        preview = ", ".join(changed[:8])
        raise RuntimeError(
            "P3-01 violation: frozen SSL parameters changed: "
            f"{preview}"
        )
