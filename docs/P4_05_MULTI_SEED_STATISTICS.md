# P4-05 — 三个 Seed 的统计汇总

P4-02、P4-03 和 P4-04 已完成所有冻结的 layer sweeps。

输入覆盖范围：

```text
Wav2Vec2-base   13 layers x 3 seeds = 39
WavLM-large     25 layers x 3 seeds = 75
W2v-BERT 2.0    25 layers x 3 seeds = 75

Total:
63 Backbone/Layer configurations
189 layer-seed results
```

P4-05 只执行统计汇总。

本阶段**不**会：
- 对 layer 排名；
- 选择最佳 layer；
- 选择最佳 Backbone；
- 访问 generic_test。

对每种 Backbone/Layer 配置：

```text
seed17
seed29
seed43
   ↓
mean
sample std (ddof=1)
standard error
two-sided 95% Student-t CI (df=2)
```

置信区间计算如下：

```text
mean ± t(0.975, df=2) * s / sqrt(3)
```

其中：

```text
t = 4.302652729911275
```

运行：

```powershell
python scripts/aggregate_p4_05_statistics.py
```

输出：

```text
artifacts/p4_runs/p4_05_statistics/
├── p4_05_multiseed_statistics.json
└── p4_05_multiseed_statistics.csv
```

随后执行审计：

```powershell
python scripts/audit_p4_05_statistics.py
```

通过时必须满足：

```text
13 Wav2Vec2 configurations
25 WavLM configurations
25 W2v-BERT configurations
63 aggregate configurations
189 raw layer-seed results
3 seeds/config
sample std ddof=1
95% CI = Student-t df=2
ranking performed = NO
best layer selected = NO
best backbone selected = NO
generic_test accessed = NO
```

P4-06 是首个允许为每种 Backbone 选择最佳层并比较三种 Backbone 获胜项的阶段。
