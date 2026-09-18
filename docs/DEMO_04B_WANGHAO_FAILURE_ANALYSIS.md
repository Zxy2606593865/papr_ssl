# Demo-04B — Wanghao Failure Analysis

## Demo-04A baseline

当前真实王灏 2-shot、5 registered intent、source-disjoint query：

```text
KNOWN
CA      = 0.92
WA      = 0.00
REJECT  = 0.08

UNKNOWN
ACCEPT  = 0.1667
REJECT  = 0.8333
```

已知类共 25 条：
- 23 条 correct ACCEPT
- 2 条 REJECT
- 0 条 wrong ACCEPT

Unknown 共 6 条：
- 5 条 REJECT
- 1 条 false ACCEPT

## 为什么先不调 threshold？

三个失败样本必须先区分：

### Known REJECT

若候选排序 top1 仍然是真实 intent：

```text
representation/ranking 基本正确
→ decision policy 太保守或域偏移
```

若 top1 已经错：

```text
representation/enrollment/cluster quality 问题
```

### Unknown False Accept

当前已知：

```text
unknown = greet / “大家好”
predicted = Self-introduction / “大家好 我叫王灏”
```

这是语音内容高度相似的 hard negative。

需要查看候选分数、margin 和 C/W/U evidence，不能简单归因于模型差。

## 运行

从 `papr_ssl` 根目录：

```powershell
python scripts/analyze_demo04a_failures.py `
  --dataset-root "..\papr_audio_toolkit\data\exports\wanghao_demo" `
  --output-dir artifacts\demo_04b_wanghao_failure_analysis
```

输出：

```text
artifacts/demo_04b_wanghao_failure_analysis/
├── summary.json
├── failure_review.csv
└── failure_wavs/
```

只需重新听 3 条失败 WAV。

## 这一步禁止

```text
重新训练
改 threshold
打开 generic_test
```

先确定错误类型，再决定下一步。
