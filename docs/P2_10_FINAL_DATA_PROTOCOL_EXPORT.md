# P2-10 — 最终数据协议导出

P2-10 是 P2 阶段的归档与定稿环节。它不会重新扫描原始音频、重建 manifests、重新定义类别、改变数据划分或执行语义合并。

该阶段将已冻结的 P2 输出打包到：

```text
docs/DATA_PROTOCOL.md

artifacts/data_audit/
├── gsc_raw_audit.json
├── mdsc_raw_audit.json
├── mdsc_phrase_inventory_summary.json
├── mdsc_task_policy_v2_summary.json
├── p2_final_closure.json
├── data_audit_summary.json
└── manifest.json
```

完整测试套件通过后运行：

```powershell
python scripts/export_p2_final_data_protocol.py `
  --unit-tests-passed 73
```

如果本地 raw-audit 源路径不同，请显式传入 `--gsc-raw-audit` 和 `--mdsc-raw-audit`。
