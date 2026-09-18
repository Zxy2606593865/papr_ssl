#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--result",
        type=Path,
        default=Path("artifacts/h6_03_raw_wav_parity/summary.json"),
    )
    args = p.parse_args()

    d = json.loads(args.result.read_text(encoding="utf-8"))

    assert d["schema"] == "papr_ssl.h6_03_raw_wav_parity.v1"
    assert d["generic_test"] == "sealed_not_accessed"
    assert d["samples"] > 0

    print("=" * 108)
    print("H6-03 RAW WAV ADAPTER PARITY AUDIT")
    print("=" * 108)
    print(f"samples:                  {d['samples']}")
    print(f"model:                    {d['model_id']}")
    print(f"revision:                 {d['revision']}")
    print(f"hidden_states index:      {d['hidden_state_index']}")
    print(f"global min cosine:        {d['global']['min_cosine']:.8f}")
    print(f"global max abs diff:      {d['global']['max_abs_diff']:.3e}")
    print(f"temporal shape match:     {d['temporal']['all_shape_match']}")
    print(f"temporal min frame cos:   {d['temporal']['min_frame_cosine']:.8f}")
    print(f"temporal max abs diff:    {d['temporal']['max_abs_diff']:.3e}")
    print("-" * 108)
    print(f"raw WAV adapter promoted: {'YES' if d['raw_wav_adapter_promoted'] else 'NO'}")
    print("generic_test accessed:    NO")

    if d["status"] != "PASS":
        raise SystemExit("AUDIT STATUS: FAIL — do not use raw WAV runtime yet.")

    print("AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
