# 轻量时序 DTW 重排序器

## 这会让模型过于冗余吗？

不会。本实验正是为避免冗余而设计的。

它**不会**增加第二个 encoder。

现有 Project-1 Teacher 保持为：

```text
Audio
-> WavLM-large hidden_states[15]
-> Attention
-> 256D global embedding
```

DTW 分支复用**同一组 WavLM layer-15 frame feature**，并逐帧复用已训练 Teacher 的 `1024 -> 256` projection。

因此，新增分支仅包含：

```text
same WavLM[15] frames
-> same trained 1024->256 projection
-> temporal mean-pool every 3 frames
-> DTW
```

没有第二个 WavLM、CNN 或 Transformer。

## 控制 DTW 的开销

全局 256D prototype 系统仍是主要分类器。

只有全局系统生成 top-3 候选类别后，才运行 DTW。

每个类别仅保留 2 个时序 enrollment template。

Therefore each query requires only:

```text
3 candidate classes x 2 templates = 6 DTW comparisons
```

而不是进行：

```text
30 classes x 15 enrollment utterances = 450 DTW comparisons
```

DTW template 比较次数因此减少 75 倍。

## 架构

```text
                         WavLM-large[15]
                         /             \
                        /               \
                       v                 v
              Attention pooling     same frame projection
                       |                 |
                       v                 v
                    256D global      [T',256] temporal
                       |                 |
                       v                 |
              30-class prototype         |
                       |                 |
                       v                 |
                  top-3 classes ---------+
                                         |
                                         v
                             DTW to 2 templates/class
                                         |
                                         v
                               conservative score fusion
                                         |
                                         v
                              consistency + rejector
```

## 已冻结的 DTW 参数

为避免无约束搜索，参数固定如下：

```text
temporal downsample factor = 3
top-k candidate classes = 3
templates/class = 2
DTW local cost = 1 - cosine
Sakoe-Chiba band ratio = 0.25
fusion lambda grid = {0.50, 0.75, 1.00}
```

Lambda 仅使用**已知类校准数据**选择。遇到平局时优先选择较大的 lambda，使系统尽量接近现有全局 Teacher，除非 DTW 能带来可测量的收益。

## 运行顺序

### 1. 构建时序特征

```powershell
python scripts/build_temporal_dtw_features.py --device cuda
```

### 2. 预计算 DTW 证据

```powershell
python scripts/precompute_dtw_rerank.py
```

这是最慢的 CPU 阶段。只需计算一次，后续重复 split 均可复用。

### 3. 运行 20-split 应用消融实验

```powershell
python scripts/run_temporal_dtw_ablation.py
```

### 4. 审计

```powershell
python scripts/audit_temporal_dtw_ablation.py
```

## 纳入主线的规则

不得仅因 DTW 有学术趣味而保留它。

只有当它能提升 20-split 应用指标或 Gate 稳定性时才予以保留，重点关注：

```text
Macro-F1
Correct Accept
Unknown Reject
Gate pass rate
```

且不能明显增加 Wrong Intent。

如果 DTW 没有帮助，就移除该分支并保留更简单的 256D Teacher。

## 项目边界

Project 1 / Agent：
- 允许使用 DTW，因为推理在服务端运行。

Project 2 / 硬件：
- canonical 64D 路径保持不变。
- edge Student 不会自动继承此 DTW 分支。

`generic_test` 继续保持封存。
