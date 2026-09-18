from __future__ import annotations

import unittest

from papr_ssl.data.manifest_schema import (
    AudioManifestRecord,
    MANIFEST_SCHEMA_VERSION,
    record_from_dict,
)


class AudioManifestRecordTest(unittest.TestCase):
    def test_valid_gsc_speech_record(self) -> None:
        record = AudioManifestRecord(
            utt_id="gsc_v2:backward/0165e0e8_nohash_0.wav",
            dataset="gsc_v2",
            audio_relpath="backward/0165e0e8_nohash_0.wav",
            speaker_id="0165e0e8",
            label="backward",
            transcript="backward",
            language="en",
            split="train",
            domain="standard",
            role="train",
            sample_rate_hz=16000,
            num_channels=1,
            num_frames=16000,
            duration_sec=1.0,
            source_meta={"official_split": True},
        )
        payload = record.to_dict()
        self.assertEqual(payload["schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertEqual(payload["label"], "backward")

    def test_valid_mdsc_eval_record(self) -> None:
        record = AudioManifestRecord(
            utt_id="mdsc:Uncontrol/dev/eval/wav/DF0014/x.wav",
            dataset="mdsc",
            audio_relpath="Uncontrol/dev/eval/wav/DF0014/x.wav",
            speaker_id="DF0014",
            label="示例短语",
            transcript="示例短语",
            language="zh-CN",
            split="dev",
            domain="dysarthria",
            role="eval",
            sample_rate_hz=16000,
            num_channels=1,
            num_frames=48000,
            duration_sec=3.0,
            source_meta={"source_group": "Uncontrol"},
        )
        record.validate()

    def test_absolute_path_rejected(self) -> None:
        record = AudioManifestRecord(
            utt_id="bad",
            dataset="gsc_v2",
            audio_relpath=r"D:\datasets\speech.wav",
            speaker_id="spk",
            label="yes",
            transcript="yes",
            language="en",
            split="train",
            domain="standard",
            role="train",
            sample_rate_hz=16000,
            num_channels=1,
            num_frames=16000,
            duration_sec=1.0,
        )
        with self.assertRaises(ValueError):
            record.validate()

    def test_speech_requires_identity_and_text(self) -> None:
        record = AudioManifestRecord(
            utt_id="bad2",
            dataset="mdsc",
            audio_relpath="Uncontrol/train/wav/x.wav",
            speaker_id=None,
            label=None,
            transcript=None,
            language="zh-CN",
            split="train",
            domain="dysarthria",
            role="train",
            sample_rate_hz=16000,
            num_channels=1,
            num_frames=32000,
            duration_sec=2.0,
        )
        with self.assertRaises(ValueError):
            record.validate()

    def test_round_trip(self) -> None:
        payload = {
            "utt_id": "gsc_v2:yes/a_nohash_0.wav",
            "dataset": "gsc_v2",
            "audio_relpath": "yes/a_nohash_0.wav",
            "speaker_id": "a",
            "label": "yes",
            "transcript": "yes",
            "language": "en",
            "split": "test",
            "domain": "standard",
            "role": "train",
            "record_type": "speech",
            "sample_rate_hz": 16000,
            "num_channels": 1,
            "num_frames": 16000,
            "duration_sec": 1.0,
            "source_meta": {},
            "schema_version": MANIFEST_SCHEMA_VERSION,
        }
        restored = record_from_dict(payload)
        self.assertEqual(restored.to_dict(), payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
