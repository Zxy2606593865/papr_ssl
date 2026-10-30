# P3-01 — 冻结 SSL Backbone 契约

## 目标

在 P3 基线训练期间，预训练 SSL backbone 是**固定特征提取器**。

可训练的表征 head 不得更新 Wav2Vec2、WavLM 或 W2v-BERT 2.0 参数。

## 冻结契约

对每个 SSL backbone：

```text
parameter.requires_grad = False
backbone forward executes in eval mode
optimizer excludes backbone parameters
backbone gradient = None
backbone parameters are unchanged after optimizer.step()
```

## 为什么同时使用 `requires_grad=False` 和 `eval()`？

它们解决不同的问题。

`requires_grad=False` 表示：

> 不计算或更新参数梯度。

`eval()` 表示：

> 让 Dropout 和类似 BatchNorm 的有状态层按推理时的行为运行。

只冻结参数不会自动调用 `eval()`。

## 为什么要安装 eval guard？

外层 teacher 模型通常会调用：

```python
teacher.train()
```

PyTorch 会递归地将子模块切换到 train 模式。因此，先前设置为 `eval()` 的 backbone 可能又被切换为 `training=True`。

P3-01 安装 forward-pre-hook，在每次实际执行 SSL forward 之前重新调用 `eval()`。这样 backbone 计算始终采用 eval 语义，而可训练 head 仍处于 train 模式。

## Optimizer 规则

不要不加筛选地用 `model.parameters()` 创建 optimizer。

使用：

```python
optimizer = torch.optim.AdamW(
    list(trainable_parameters(model)),
    lr=...,
)
```

并验证：

```python
assert_optimizer_excludes_backbone(
    optimizer,
    model.backbone,
)
```

## 集成方式

加载真实 SSL backbone 后立即执行：

```python
backbone = build_ssl_backbone(...)
freeze_ssl_backbone(backbone)
```

之后仅 P3 representation head / SCAF 参数保持可训练。

## Gate

满足以下条件时，P3-01 通过：

1. 每个 backbone 参数的 `requires_grad=False`；
2. 即使父模块调用 `train()`，backbone forward 仍使用 eval 模式；
3. backbone 梯度始终为 `None`；
4. optimizer 不包含 backbone 参数；
5. 执行一次真实 optimizer step 后，backbone 参数逐位不变。

独立 smoke script 验证契约机制。P3-02 会将此契约应用到真实 Teacher / 64D 表征 head。
