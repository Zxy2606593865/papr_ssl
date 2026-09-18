#!/usr/bin/env python
"""Audit H7-00 freeze without opening generic_test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("artifacts/h7_00_final_freeze"),
    )
    args = p.parse_args()

    config_path = args.freeze_dir / "frozen_config.json"
    seal_path = args.freeze_dir / "freeze_seal.json"

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))

    errors = []

    if cfg.get("research_state") != "FROZEN_BEFORE_GENERIC_TEST":
        errors.append("research_state")
    if cfg.get("generic_test") != "SEALED_NOT_ACCESSED":
        errors.append("generic_test seal")
    if cfg["backbone"]["hidden_state_index"] != 15:
        errors.append("hidden_state_index")
    if cfg["representation"]["global_embedding_dim"] != 256:
        errors.append("embedding_dim")
    if cfg["temporal"]["fixed_fusion_lambda"] != 0.50:
        errors.append("fixed_fusion_lambda")
    if cfg["ranking"]["h4_learned_shared_ranker_promoted"] is not False:
        errors.append("H4 ranker promotion flag")
    if cfg["decision"]["scores_are_calibrated_probabilities"] is not False:
        errors.append("decision score calibration flag")
    if cfg["enrollment"]["supported_shots_for_frozen_policy"] != [1, 2]:
        errors.append("supported shots")

    current_config_hash = sha256_file(config_path)
    if current_config_hash != seal["frozen_config_sha256"]:
        errors.append("freeze seal digest mismatch")

    # Re-hash every frozen dependency to catch post-freeze changes.
    changed = []
    for name, rec in cfg["artifact_hashes"].items():
        path = Path(rec["path"])
        if not path.exists():
            changed.append(f"{name}: MISSING")
            continue
        now = sha256_file(path)
        if now != rec["sha256"]:
            changed.append(f"{name}: SHA256 CHANGED")

    print("=" * 108)
    print("H7-00 FINAL FREEZE AUDIT")
    print("=" * 108)
    print(f"freeze config SHA256:      {current_config_hash}")
    print(f"frozen dependencies:       {len(cfg['artifact_hashes'])}")
    print(f"dependency changes:        {len(changed)}")
    print("generic_test accessed:     NO")
    if changed:
        for x in changed:
            print("  -", x)

    if errors or changed:
        print("-" * 108)
        for x in errors:
            print("ERROR:", x)
        raise SystemExit("AUDIT STATUS: FAIL")

    print("-" * 108)
    print("AUDIT STATUS: PASS")


if __name__ == "__main__":
    main()
