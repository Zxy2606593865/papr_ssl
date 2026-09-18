#!/usr/bin/env python
"""Standalone P3-01 contract smoke check."""

from __future__ import annotations

import torch
from torch import nn

from papr_ssl.training.frozen_backbone import (
    assert_optimizer_excludes_backbone,
    assert_parameters_unchanged,
    assert_ssl_backbone_frozen,
    freeze_ssl_backbone,
    frozen_backbone_state,
    snapshot_parameters,
    trainable_parameters,
)


class SmokeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(self.proj(x))


class SmokeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = freeze_ssl_backbone(SmokeBackbone())
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        return self.head(self.backbone(x))


def main() -> int:
    torch.manual_seed(0)
    model = SmokeTeacher()

    assert_ssl_backbone_frozen(model.backbone)
    state = frozen_backbone_state(model.backbone)

    optimizer = torch.optim.AdamW(
        list(trainable_parameters(model)),
        lr=1e-2,
    )
    assert_optimizer_excludes_backbone(
        optimizer,
        model.backbone,
    )

    before = snapshot_parameters(model.backbone)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.randn(8, 16)).pow(2).mean()
    loss.backward()
    optimizer.step()

    assert model.backbone.training is False
    assert all(
        p.grad is None for p in model.backbone.parameters()
    )
    assert any(
        p.grad is not None for p in model.head.parameters()
    )
    assert_parameters_unchanged(before, model.backbone)

    print("=" * 88)
    print("PAPR-SSL P3-01 FROZEN BACKBONE CONTRACT")
    print("=" * 88)
    print(f"backbone parameters:           {state.total_parameters}")
    print(f"trainable backbone parameters: {state.trainable_parameters}")
    print(f"backbone eval after forward:   {not model.backbone.training}")
    print("backbone gradients:            NONE")
    print("optimizer contains backbone:   NO")
    print("backbone changed after step:   NO")
    print("-" * 88)
    print("P3-01 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
