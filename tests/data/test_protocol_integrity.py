from __future__ import annotations

import unittest

from papr_ssl.data.manifest_schema import AudioManifestRecord
from papr_ssl.data.protocol_integrity import (
    analyze_mdsc_personalized_protocol,
)
from papr_ssl.data.task_views import MDSC_WAKE_WORDS


def rec(*, utt, speaker, split, role, label):
    return AudioManifestRecord(
        utt_id=utt,
        dataset="mdsc",
        audio_relpath=f"{utt}.wav",
        speaker_id=speaker,
        label=label,
        transcript=label,
        language="zh-CN",
        split=split,
        domain="dysarthria",
        role=role,
        record_type="speech",
        sample_rate_hz=16000,
        num_channels=1,
        num_frames=16000,
        duration_sec=1.0,
        source_meta={},
    )


def full_speaker(speaker: str, split: str):
    rows = []
    for i, wake in enumerate(MDSC_WAKE_WORDS):
        rows.append(
            rec(
                utt=f"{speaker}_e_{i}",
                speaker=speaker,
                split=split,
                role="enrollment",
                label=wake,
            )
        )
        rows.append(
            rec(
                utt=f"{speaker}_v_{i}",
                speaker=speaker,
                split=split,
                role="eval",
                label=wake,
            )
        )
    rows.append(
        rec(
            utt=f"{speaker}_neg",
            speaker=speaker,
            split=split,
            role="eval",
            label="打开空调",
        )
    )
    return rows


class ProtocolIntegrityTest(unittest.TestCase):
    def test_full_wake_coverage_is_detected(self):
        rows = full_speaker("D1", "dev") + full_speaker("D2", "test")
        out = analyze_mdsc_personalized_protocol(rows)

        self.assertTrue(
            out["gate"]["all_speakers_have_10_wakes_in_enrollment"]
        )
        self.assertTrue(
            out["gate"]["all_speakers_have_10_wakes_in_eval"]
        )
        self.assertTrue(
            out["gate"]["dev_test_speakers_disjoint"]
        )

    def test_missing_enrollment_wake_fails_gate(self):
        rows = full_speaker("D1", "dev")
        rows = [
            r for r in rows
            if not (
                r.role == "enrollment"
                and r.label == MDSC_WAKE_WORDS[0]
            )
        ]
        out = analyze_mdsc_personalized_protocol(rows)
        self.assertFalse(
            out["gate"]["all_speakers_have_10_wakes_in_enrollment"]
        )

    def test_same_speaker_in_dev_and_test_is_detected(self):
        rows = full_speaker("D1", "dev")
        rows.append(
            rec(
                utt="leak",
                speaker="D1",
                split="test",
                role="eval",
                label="小爱同学",
            )
        )
        out = analyze_mdsc_personalized_protocol(rows)
        self.assertFalse(out["gate"]["one_split_per_speaker"])
        self.assertFalse(out["gate"]["dev_test_speakers_disjoint"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
