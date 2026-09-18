from __future__ import annotations
import unittest

from papr_ssl.data.manifest_schema import AudioManifestRecord
from papr_ssl.data.phrase_inventory_audit import (
    build_mdsc_phrase_inventory,
    summarize_phrase_inventory,
)

def rec(utt, speaker, text, domain, split="train", role="train"):
    return AudioManifestRecord(
        utt_id=utt,
        dataset="mdsc",
        audio_relpath=f"{utt}.wav",
        speaker_id=speaker,
        label=text,
        transcript=text,
        language="zh-CN",
        split=split,
        domain=domain,
        role=role,
        record_type="speech",
        sample_rate_hz=16000,
        num_channels=1,
        num_frames=16000,
        duration_sec=1.0,
        source_meta={},
    )

class PhraseInventoryAuditTest(unittest.TestCase):
    def test_pause_variant_collapses(self):
        rows = [
            rec("a", "C1", "小度小度", "control"),
            rec("b", "D1", "小度<p>小度", "dysarthria"),
        ]
        inv = build_mdsc_phrase_inventory(rows)
        self.assertEqual(len(inv), 1)
        self.assertEqual(inv[0].raw_variant_count, 2)

    def test_semantic_variants_do_not_merge(self):
        rows = [
            rec("a", "C1", "打开空调", "control"),
            rec("b", "D1", "帮我打开空调", "dysarthria"),
        ]
        self.assertEqual(len(build_mdsc_phrase_inventory(rows)), 2)

    def test_train_support_ignores_dev_test(self):
        rows = [
            rec("a", "C1", "打开空调", "control"),
            rec("b", "D1", "打开空调", "dysarthria", "dev", "eval"),
        ]
        row = build_mdsc_phrase_inventory(rows)[0]
        self.assertEqual(row.control_train_speaker_count, 1)
        self.assertEqual(row.dysarthria_train_speaker_count, 0)
        self.assertEqual(row.dysarthria_speaker_count, 1)

    def test_threshold_sweep(self):
        rows = [
            rec("a", "C1", "开灯", "control"),
            rec("b", "D1", "开灯", "dysarthria"),
            rec("c", "C2", "关灯", "control"),
        ]
        inv = build_mdsc_phrase_inventory(rows)
        s = summarize_phrase_inventory(inv, thresholds=((1, 1),))
        self.assertEqual(s["threshold_sweep"][0]["phrase_count"], 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
