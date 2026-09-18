# P4-05 — Three-seed Statistical Aggregation

P4-02, P4-03 and P4-04 have completed all frozen layer sweeps.

Input coverage:

```text
Wav2Vec2-base   13 layers x 3 seeds = 39
WavLM-large     25 layers x 3 seeds = 75
W2v-BERT 2.0    25 layers x 3 seeds = 75

Total:
63 Backbone/Layer configurations
189 layer-seed results
```

P4-05 performs statistics only.

It does **not**:
- rank layers,
- select a best layer,
- select a best Backbone,
- access generic_test.

For every Backbone/Layer configuration:

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

The confidence interval is:

```text
mean ± t(0.975, df=2) * s / sqrt(3)
```

with:

```text
t = 4.302652729911275
```

Run:

```powershell
python scripts/aggregate_p4_05_statistics.py
```

Outputs:

```text
artifacts/p4_runs/p4_05_statistics/
├── p4_05_multiseed_statistics.json
└── p4_05_multiseed_statistics.csv
```

Then audit:

```powershell
python scripts/audit_p4_05_statistics.py
```

PASS requires:

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

P4-06 will be the first phase allowed to select the best layer per Backbone
and then compare the three Backbone winners.
