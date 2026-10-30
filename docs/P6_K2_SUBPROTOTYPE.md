# P6 K=2 Subprototype 回退方案

K=1 calibration 不可行，因此启用已定义的 P6-01 回退方案。

需区分以下两种做法：

- `K=2 enrollment average` = two samples averaged into one prototype.
- `K=2 subprototype` = keep two independent prototypes per class.

本方案实现第二种定义：

```text
class_score(c) = max(cos(z,p_c1), cos(z,p_c2))
```

Top1/top2 margin 根据 30 个 class scores 计算。

运行：

```powershell
python scripts/prepare_p6_k2_subprototype.py
python scripts/run_p6_k2_subprototype.py --device cuda
python scripts/audit_p6_k2_subprototype.py
```

K=1 artifacts 保留在 `artifacts/p6_dev`；K=2 写入 `artifacts/p6_dev_k2`。`generic_test` 继续封存。
