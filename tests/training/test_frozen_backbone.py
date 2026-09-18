from __future__ import annotations

import unittest

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


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.dropout(self.linear(x))


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ToyBackbone()
        freeze_ssl_backbone(self.backbone)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class FrozenBackboneTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_all_backbone_parameters_are_frozen(self):
        backbone = freeze_ssl_backbone(ToyBackbone())
        assert_ssl_backbone_frozen(backbone)
        state = frozen_backbone_state(backbone)
        self.assertGreater(state.total_parameters, 0)
        self.assertEqual(state.trainable_parameters, 0)
        self.assertFalse(state.training)

    def test_parent_train_cannot_make_actual_backbone_forward_train_mode(self):
        model = ToyTeacher()
        model.train()

        # Parent train() recursively flips the child first...
        self.assertTrue(model.backbone.training)

        # ...but the P3-01 pre-forward guard restores eval immediately before
        # backbone computation.
        _ = model(torch.randn(4, 8))
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.head.training)

    def test_backbone_receives_no_gradients(self):
        model = ToyTeacher()
        model.train()

        loss = model(torch.randn(4, 8)).sum()
        loss.backward()

        self.assertTrue(
            all(p.grad is None for p in model.backbone.parameters())
        )
        self.assertTrue(
            any(p.grad is not None for p in model.head.parameters())
        )

    def test_optimizer_contains_only_trainable_parameters(self):
        model = ToyTeacher()
        optimizer = torch.optim.AdamW(
            list(trainable_parameters(model)),
            lr=1e-3,
        )
        assert_optimizer_excludes_backbone(
            optimizer,
            model.backbone,
        )

        optimizer_ids = {
            id(p)
            for group in optimizer.param_groups
            for p in group["params"]
        }
        self.assertTrue(
            all(id(p) in optimizer_ids for p in model.head.parameters())
        )

    def test_backbone_is_bitwise_unchanged_after_optimizer_step(self):
        model = ToyTeacher()
        optimizer = torch.optim.AdamW(
            list(trainable_parameters(model)),
            lr=1e-2,
        )

        before = snapshot_parameters(model.backbone)

        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(8, 8)).pow(2).mean()
        loss.backward()
        optimizer.step()

        assert_parameters_unchanged(before, model.backbone)


if __name__ == "__main__":
    unittest.main(verbosity=2)
