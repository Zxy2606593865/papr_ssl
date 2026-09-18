#!/usr/bin/env python
"""Verify that the latest qualified-utt-id hotfix is actually imported."""

from __future__ import annotations

import inspect

from papr_ssl.training.teacher import real_experiment as rexp


def main() -> int:
    source = inspect.getsource(rexp.resolve_task_audio_path)

    assert hasattr(rexp, "qualified_utt_relpath"), (
        "LATEST HOTFIX NOT ACTIVE: qualified_utt_relpath is missing"
    )
    assert "encoded_rel = qualified_utt_relpath(row)" in source, (
        "LATEST HOTFIX NOT ACTIVE: resolve_task_audio_path still uses the "
        "old catalog-only implementation"
    )
    assert "audio_catalog[row.utt_id]" not in source, (
        "LATEST HOTFIX NOT ACTIVE: old audio_catalog[row.utt_id] lookup found"
    )

    row = rexp.RealTaskRow(
        dataset="mdsc",
        utt_id="mdsc:Control/dev/wav/CF0010/CF0010_0001.wav",
        split="dev",
        task_label="smoke",
    )
    rel = rexp.qualified_utt_relpath(row)
    assert rel == "Control/dev/wav/CF0010/CF0010_0001.wav", rel

    print("=" * 92)
    print("P3 REAL QUALIFIED-UTT-ID HOTFIX VERIFICATION")
    print("=" * 92)
    print(f"imported module:           {rexp.__file__}")
    print("qualified_utt_relpath:     PRESENT")
    print(f"decoded sample utt_id:     {rel}")
    print("old catalog-only lookup:   ABSENT")
    print("-" * 92)
    print("QUALIFIED-UTT-ID HOTFIX: ACTIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
