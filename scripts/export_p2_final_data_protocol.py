#!/usr/bin/env python
"""P2-10 final export: package frozen P2 artifacts into canonical outputs."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_json(src: Path, dst: Path):
    data = load(src)
    write(dst, data)
    return data


def render_protocol(phrase, policy):
    core = policy["views"]["mdsc_core30_exact_phrase"]
    cmd = policy["views"]["mdsc_command20_exact_phrase"]
    opn = policy["views"]["mdsc_core30_open_set_eval"]
    return f'''# PAPR-SSL Data Protocol\n\nStatus: **FROZEN — P2 PASS / CLOSED**\n\n## 1. Principle\n\nP2 standardizes data access, manifests, audio handling, safe batching, task views, and evaluation roles. Raw audio and raw manifests are not rewritten. Task-specific labels are derived views.\n\n## 2. GSC v2\n\n- Role: generic standard-speech KWS / embedding benchmark.\n- Task: **GSC-35** closed-set classification.\n- Audio: 16 kHz; model-side length 16000 samples; short clips are right-padded; valid lengths are retained for masking.\n\n## 3. MDSC\n\nMDSC is treated as a multi-use Mandarin dysarthria speech resource, not as a 3784-class phrase dataset.\n\nAfter approved surface normalization:\n\n```text\nnormalized phrases: {phrase["inventory"]["phrase_count"]}\ncross-domain phrases: {phrase["inventory"]["cross_domain_phrase_count"]}\n```\n\nNo semantic synonym merging is performed.\n\n### 3.1 MDSC-Core30 — primary public phrase benchmark\n\n```text\nutterances: {core["count"]}\nclasses:    {core["class_count"]}\ntrain/dev/test: {core["splits"]["train"]} / {core["splits"]["dev"]} / {core["splits"]["test"]}\ncontrol/dysarthria: {core["domains"]["control"]} / {core["domains"]["dysarthria"]}\n```\n\nPurpose: dysarthria phrase representation, SSL backbone/layer/head comparison, and Control-vs-Dysarthria analysis. The class inventory is justified by stable train-side speaker coverage.\n\n### 3.2 MDSC-Command20 — command-only diagnostic\n\n```text\nutterances: {cmd["count"]}\nclasses:    {cmd["class_count"]}\ntrain/dev/test: {cmd["splits"]["train"]} / {cmd["splits"]["dev"]} / {cmd["splits"]["test"]}\n```\n\nPurpose: evaluate ordinary command representation without relying on wake-word semantics.\n\n### 3.3 MDSC-WWS10 — auxiliary personalization benchmark\n\nPurpose: 1-shot personalized wake-word spotting and prototype/few-shot personalization analysis. It does not define the final product activation mechanism.\n\n### 3.4 MDSC open-set / rejection evaluation\n\n```text\nutterances: {opn["count"]}\ndev/test: {opn["splits"]["dev"]} / {opn["splits"]["test"]}\ncontrol/dysarthria: {opn["domains"]["control"]} / {opn["domains"]["dysarthria"]}\n```\n\nOpen-set utterances are **not** collapsed into one supervised SCAF class. They are used for dev-side threshold calibration, sealed test-side rejection evaluation, hard-negative analysis, or future auxiliary objectives.\n\n## 4. Task-dependent labels\n\nExample:\n\n```text\n"打开空调"\nWWS10   -> non-wake / negative\nCore30  -> target class "打开空调"\nOpenSet -> not open-set because it belongs to Core30\n```\n\nTherefore NON_WAKE and OPEN_SET are task-view labels, not global labels written into the raw manifest.\n\n## 5. MDSC audio policy\n\n- Preserve the full utterance.\n- Stereo -> mono by channel mean in memory.\n- Variable-length batches use right padding and waveform masks.\n\n## 6. Safe SSL batching\n\n- WavLM: native padded batch.\n- W2v-BERT 2.0: semantic trim before frontend/native batching.\n- Wav2Vec2: group by exact semantic waveform length, forward each group, then restore original order.\n\n## 7. Split policy\n\n```text\ntrain -> fitting / supervised learning\ndev   -> model selection / threshold calibration\ntest  -> sealed final evaluation\n```\n\n## 8. Final P2 task map\n\n```text\nGSC-35          -> generic standard-speech embedding benchmark\nMDSC-Core30     -> primary dysarthria phrase representation benchmark\nMDSC-Command20  -> command-only diagnostic\nMDSC-WWS10      -> auxiliary 1-shot personalization benchmark\nMDSC long-tail  -> open-set / rejection evaluation\n```\n\nAfter P2-10, P2 is reopened only for a documented data bug.\n'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gsc-raw-audit", type=Path, default=Path("artifacts/p2_02/gsc_v2_raw_audit.json"))
    p.add_argument("--mdsc-raw-audit", type=Path, default=Path("artifacts/p2_02/mdsc_raw_audit.json"))
    p.add_argument("--phrase-inventory", type=Path, default=Path("artifacts/p2_07/phrase_inventory/mdsc_phrase_inventory_summary.json"))
    p.add_argument("--mdsc-policy", type=Path, default=Path("artifacts/p2_07/mdsc_policy_v2/mdsc_task_policy_v2_summary.json"))
    p.add_argument("--audit-dir", type=Path, default=Path("artifacts/data_audit"))
    p.add_argument("--protocol", type=Path, default=Path("docs/DATA_PROTOCOL.md"))
    p.add_argument("--unit-tests-passed", type=int, required=True)
    args = p.parse_args()

    required = [args.gsc_raw_audit, args.mdsc_raw_audit, args.phrase_inventory, args.mdsc_policy]
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        print("Missing required frozen P2 artifacts:")
        for x in missing: print(" -", x)
        return 2

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    args.protocol.parent.mkdir(parents=True, exist_ok=True)

    gsc_dst = args.audit_dir / "gsc_raw_audit.json"
    mdsc_dst = args.audit_dir / "mdsc_raw_audit.json"
    phrase_dst = args.audit_dir / "mdsc_phrase_inventory_summary.json"
    policy_dst = args.audit_dir / "mdsc_task_policy_v2_summary.json"

    gsc = copy_json(args.gsc_raw_audit, gsc_dst)
    mdsc = copy_json(args.mdsc_raw_audit, mdsc_dst)
    phrase = copy_json(args.phrase_inventory, phrase_dst)
    policy = copy_json(args.mdsc_policy, policy_dst)

    if policy.get("gate", {}).get("overall") != "PASS":
        raise RuntimeError("MDSC task policy gate is not PASS")
    if policy.get("policy", {}).get("raw_manifest_rewrite") is not False:
        raise RuntimeError("raw_manifest_rewrite must remain false")
    if policy.get("policy", {}).get("semantic_merging") is not False:
        raise RuntimeError("semantic_merging must remain false")

    args.protocol.write_text(render_protocol(phrase, policy), encoding="utf-8")

    closure = {
        "schema": "papr_ssl.p2_final_closure.v1",
        "phase": "P2",
        "status": "PASS",
        "closed": True,
        "unit_tests": {"passed": args.unit_tests_passed, "failed": 0},
        "mdsc_policy_schema": policy["schema"],
        "mdsc_policy_gate": policy["gate"]["overall"],
        "raw_manifest_rewrite": False,
        "semantic_merging": False,
        "test_policy": "sealed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    closure_path = args.audit_dir / "p2_final_closure.json"
    write(closure_path, closure)

    core = policy["views"]["mdsc_core30_exact_phrase"]
    cmd = policy["views"]["mdsc_command20_exact_phrase"]
    opn = policy["views"]["mdsc_core30_open_set_eval"]
    summary = {
        "schema": "papr_ssl.data_audit_summary.v1",
        "phase": "P2",
        "status": "PASS_CLOSED",
        "datasets": {
            "gsc_v2": {
                "task_view": "gsc_all_35_closed_set",
                "role": "generic_standard_speech_embedding_benchmark",
                "raw_audit_file": gsc_dst.as_posix(),
            },
            "mdsc": {
                "role": "multi_use_mandarin_dysarthria_resource",
                "normalized_phrase_count": phrase["inventory"]["phrase_count"],
                "cross_domain_phrase_count": phrase["inventory"]["cross_domain_phrase_count"],
                "task_views": {
                    "core30": {"count": core["count"], "class_count": core["class_count"], "splits": core["splits"], "domains": core["domains"]},
                    "command20": {"count": cmd["count"], "class_count": cmd["class_count"], "splits": cmd["splits"], "domains": cmd["domains"]},
                    "wws10": {"role": "auxiliary_1shot_personalization_benchmark"},
                    "open_set": {"count": opn["count"], "splits": opn["splits"], "domains": opn["domains"], "use_as_scaf_training_class": False},
                },
                "raw_audit_file": mdsc_dst.as_posix(),
            },
        },
        "frozen_rules": {
            "raw_manifest_rewrite": False,
            "semantic_merging": False,
            "open_set_as_scaf_training_class": False,
            "test_split": "sealed",
        },
        "artifacts": {
            "protocol": args.protocol.as_posix(),
            "p2_final_closure": closure_path.as_posix(),
        },
    }
    summary_path = args.audit_dir / "data_audit_summary.json"
    write(summary_path, summary)

    manifest = {"schema": "papr_ssl.data_audit_manifest.v1", "files": {}}
    for path in [gsc_dst, mdsc_dst, phrase_dst, policy_dst, closure_path, summary_path, args.protocol]:
        manifest["files"][path.as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest_path = args.audit_dir / "manifest.json"
    write(manifest_path, manifest)

    print("=" * 92)
    print("PAPR-SSL P2-10 FINAL DATA PROTOCOL EXPORT")
    print("=" * 92)
    print("DATA_PROTOCOL:", args.protocol)
    print("Audit directory:", args.audit_dir)
    print("Unit tests passed:", args.unit_tests_passed)
    print("MDSC policy gate:", policy["gate"]["overall"])
    print("-" * 92)
    print("P2-10 STATUS: PASS")
    print("P2 remains CLOSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
