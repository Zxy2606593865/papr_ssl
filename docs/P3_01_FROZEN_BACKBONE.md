# P3-01 — Frozen SSL Backbone Contract

## Goal

During P3 baseline training, the pretrained SSL backbone is a **fixed feature
extractor**.

The trainable representation head must not update Wav2Vec2, WavLM, or
W2v-BERT 2.0 parameters.

## Frozen contract

For every SSL backbone:

```text
parameter.requires_grad = False
backbone forward executes in eval mode
optimizer excludes backbone parameters
backbone gradient = None
backbone parameters are unchanged after optimizer.step()
```

## Why both `requires_grad=False` and `eval()`?

They solve different problems.

`requires_grad=False` means:

> Do not compute/update parameter gradients.

`eval()` means:

> Execute inference-time behavior for modules such as Dropout and
> BatchNorm-like stateful layers.

Freezing parameters alone does not automatically call `eval()`.

## Why install an eval guard?

A surrounding teacher model will normally call:

```python
teacher.train()
```

PyTorch recursively switches child modules into train mode. Therefore a
backbone that was previously set to `eval()` can be flipped back to
`training=True`.

P3-01 installs a forward-pre-hook that re-applies `eval()` immediately before
every actual SSL forward. Thus the backbone computation always uses eval
semantics while the trainable head remains in train mode.

## Optimizer rule

Never construct the optimizer with `model.parameters()` blindly.

Use:

```python
optimizer = torch.optim.AdamW(
    list(trainable_parameters(model)),
    lr=...,
)
```

and validate:

```python
assert_optimizer_excludes_backbone(
    optimizer,
    model.backbone,
)
```

## Integration pattern

Immediately after loading a real SSL backbone:

```python
backbone = build_ssl_backbone(...)
freeze_ssl_backbone(backbone)
```

Then only the P3 representation head / SCAF parameters remain trainable.

## Gate

P3-01 passes when:

1. every backbone parameter has `requires_grad=False`;
2. backbone forward uses eval mode even after parent `train()`;
3. backbone gradients remain `None`;
4. optimizer contains no backbone parameters;
5. a real optimizer step leaves backbone parameters bitwise unchanged.

The standalone smoke script validates the contract mechanics. P3-02 will bind
this contract to the real Teacher / 64D representation head.
