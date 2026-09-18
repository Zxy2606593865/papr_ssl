#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from papr_ssl.data.manifest_io import read_manifest_jsonl
from papr_ssl.data.mdsc_task_policy_v2 import (
    mdsc_command20_index,
    mdsc_core30_index,
    mdsc_core30_open_set_eval_index,
    summarize_mdsc_policy,
)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row.to_dict(), ensure_ascii=False) + "\n"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/p2_04/manifests/mdsc.jsonl"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/p2_07/mdsc_policy_v2"),
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = read_manifest_jsonl(args.manifest)

    core30 = mdsc_core30_index(records)
    command20 = mdsc_command20_index(records)
    open_set = mdsc_core30_open_set_eval_index(records)
    summary = summarize_mdsc_policy(core30, command20, open_set)

    write_jsonl(args.out_dir / "mdsc_core30.index.jsonl", core30)
    write_jsonl(args.out_dir / "mdsc_command20.index.jsonl", command20)
    write_jsonl(args.out_dir / "mdsc_core30_open_set_eval.index.jsonl", open_set)

    summary_path = args.out_dir / "mdsc_task_policy_v2_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 92)
    print("PAPR-SSL P2-07E REVISED MDSC TASK POLICY")
    print("=" * 92)
    for name, view in summary["views"].items():
        print(
            f"{name:38s} "
            f"count={view['count']:5d} "
            f"classes={view.get('class_count', '-')}"
        )
    print("-" * 92)
    for key, value in summary["gate"].items():
        print(f"{key:52s} {value}")
    print("-" * 92)
    print("Important: open-set utterances are NOT one SCAF training class.")
    print("Summary:", summary_path)
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
