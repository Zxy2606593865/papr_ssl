#!/usr/bin/env python
"""Prepare stable real-Wanghao preset WAVs for Demo-05.

Selection policy:
- use the corrected Demo-04A-v2 prediction record;
- for each registered demo phrase, choose a frozen correct-ACCEPT query;
- choose one frozen REJECT unknown sample for rejection demonstration;
- copy WAVs into the web static directory;
- do not alter the research dataset.

These are presentation presets, not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


CORE_INTENTS = [
    ("Self-introduction", "个性化语音样本 01"),
    ("Introduction_the_school", "个性化语音样本 02"),
    ("Scan-to-pay", "个性化语音样本 03"),
    ("Please_enjoy_your_meal", "个性化语音样本 04"),
    ("today_feel_happy", "个性化语音样本 05"),
]

UNKNOWN_INTENT = "see_you_again"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Corrected wanghao_demo_v2 root",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/demo_04a_wanghao_runtime_v2/predictions.jsonl"),
    )
    p.add_argument(
        "--web-root",
        type=Path,
        default=Path("demo/demo05_web"),
    )
    args = p.parse_args()

    dataset_root = args.dataset_root.resolve()
    web_root = args.web_root.resolve()
    audio_dir = web_root / "static/audio/presets"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild only generated preset WAVs/config.
    for pth in audio_dir.glob("*.wav"):
        pth.unlink()

    rows = read_jsonl(args.predictions)
    presets = []

    for idx, (intent_id, display_name) in enumerate(CORE_INTENTS, start=1):
        matches = [
            r for r in rows
            if r.get("kind") == "known"
            and r.get("true_intent_id") == intent_id
            and r.get("correct_accept") is True
            and r.get("runtime_status") == "ACCEPT"
        ]
        if not matches:
            raise RuntimeError(f"No frozen correct-ACCEPT query for {intent_id}")
        row = matches[0]
        src = dataset_root / row["audio_path"]
        if not src.exists():
            raise FileNotFoundError(src)

        filename = f"preset_{idx:02d}_{src.name}"
        dst = audio_dir / filename
        shutil.copy2(src, dst)

        presets.append({
            "id": f"known_{idx:02d}",
            "display_name": display_name,
            "description": "王灏真实录音 · 个性化识别演示",
            "demo_kind": "recognition",
            "wav_relpath": f"static/audio/presets/{filename}",
            "audio_url": f"/static/audio/presets/{filename}",
            "source_segment_id": row["segment_id"],
            "source_file": row["source_file"],
            "expected_canonical_text": row["true_canonical_text"],
            "selection_note": "Frozen correct-ACCEPT query from Demo-04A-v2",
        })

    unknown_matches = [
        r for r in rows
        if r.get("kind") == "unknown"
        and r.get("ground_truth_unregistered_intent_id") == UNKNOWN_INTENT
        and r.get("runtime_status") == "REJECT"
    ]
    if not unknown_matches:
        raise RuntimeError(f"No frozen REJECT query for {UNKNOWN_INTENT}")

    row = unknown_matches[0]
    src = dataset_root / row["audio_path"]
    if not src.exists():
        raise FileNotFoundError(src)

    filename = f"preset_06_{src.name}"
    dst = audio_dir / filename
    shutil.copy2(src, dst)

    presets.append({
        "id": "unknown_01",
        "display_name": "未注册表达样本",
        "description": "王灏真实录音 · 拒识能力演示",
        "demo_kind": "rejection",
        "wav_relpath": f"static/audio/presets/{filename}",
        "audio_url": f"/static/audio/presets/{filename}",
        "source_segment_id": row["segment_id"],
        "source_file": row["source_file"],
        "expected_canonical_text": None,
        "selection_note": "Frozen REJECT query from Demo-04A-v2",
    })

    payload = {
        "schema": "papr_ssl.demo05.presets.v1",
        "student_name": "王灏",
        "note": (
            "Presentation presets selected from the already-frozen corrected "
            "engineering evaluation. Not a benchmark result."
        ),
        "presets": presets,
    }
    (web_root / "demo_presets.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print("DEMO-05 PRESET PREPARATION")
    print("=" * 96)
    for x in presets:
        print(
            f"{x['id']:<12} {x['display_name']:<18} "
            f"segment={x['source_segment_id']}"
        )
    print("-" * 96)
    print(f"preset config:            {web_root/'demo_presets.json'}")
    print(f"audio dir:                {audio_dir}")
    print("DEMO-05 PREP STATUS:      PASS")


if __name__ == "__main__":
    main()
