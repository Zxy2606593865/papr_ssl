# P4-06 — Best Layer per Backbone and Final Backbone Comparison

Input:

```text
P4-05:
63 Backbone/Layer configurations
3 seeds/configuration
189 raw layer-seed results
generic_dev_score = prototype_macro_f1
generic_test sealed
```

P4-06 is the first P4 stage allowed to select winners.

Selection rule:

```text
For each Backbone:
    maximize 3-seed mean generic_dev_score
    exact mean tie -> lower hidden_state_index

Then:
    compare the three per-Backbone winners
    maximize the same 3-seed mean generic_dev_score
```

Resource cost is not allowed to override embedding quality in P4-06.
P4-07 records cost afterward.

Run:

```powershell
python scripts/select_p4_06_backbone_layer.py
python scripts/audit_p4_06_selection.py
```

Outputs:

```text
artifacts/p4_runs/p4_06_selection/
├── p4_06_selection.json
└── p4_06_winners.csv
```

After P4-06 passes:
- P4-07 records parameter count / latency / memory / cache cost.
- P4-08 freezes the selected Backbone + exact hidden_state_index as the
  single mainline for P5.
