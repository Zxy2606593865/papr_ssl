from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from papr_ssl.data.gsc_adapter import GSCAdapter
from papr_ssl.data.mdsc_adapter import MDSCAdapter


def write_wav(path: Path, seconds: float = 1.0, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(16000 * seconds)
    if channels == 1:
        data = np.zeros((n,), dtype=np.float32)
    else:
        data = np.zeros((n, channels), dtype=np.float32)
    sf.write(path, data, 16000, subtype="PCM_16")


class GSCAdapterTest(unittest.TestCase):
    def test_official_split_and_speaker_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_wav(root / "yes" / "abc_nohash_0.wav")
            write_wav(root / "no" / "def_nohash_0.wav")
            write_wav(root / "stop" / "ghi_nohash_0.wav")
            (root / "validation_list.txt").write_text(
                "no/def_nohash_0.wav\n", encoding="utf-8"
            )
            (root / "testing_list.txt").write_text(
                "stop/ghi_nohash_0.wav\n", encoding="utf-8"
            )

            records = list(GSCAdapter(root).iter_records())
            by_label = {r.label: r for r in records}

            self.assertEqual(len(records), 3)
            self.assertEqual(by_label["yes"].split, "train")
            self.assertEqual(by_label["no"].split, "dev")
            self.assertEqual(by_label["stop"].split, "test")
            self.assertEqual(by_label["yes"].speaker_id, "abc")

    def test_background_noise_excluded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_wav(root / "yes" / "abc_nohash_0.wav")
            write_wav(root / "_background_noise_" / "white_noise.wav", seconds=2)
            (root / "validation_list.txt").write_text("", encoding="utf-8")
            (root / "testing_list.txt").write_text("", encoding="utf-8")

            records = list(GSCAdapter(root).iter_records())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].record_type, "speech")


class MDSCAdapterTest(unittest.TestCase):
    def test_control_train_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wav = root / "Control" / "train" / "wav" / "CM0002" / "CM0002_0001.wav"
            label = root / "Control" / "train" / "transcript" / "label.txt"
            write_wav(wav, seconds=2.0)
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(
                "\ufeffCM0002_0001 小度小度\n", encoding="utf-8"
            )

            records = list(MDSCAdapter(root).iter_records())
            self.assertEqual(len(records), 1)
            r = records[0]
            self.assertEqual(r.speaker_id, "CM0002")
            self.assertEqual(r.domain, "control")
            self.assertEqual(r.split, "train")
            self.assertEqual(r.role, "train")
            self.assertEqual(r.label, "小度小度")

    def test_uncontrol_enrollment_mapping_and_stereo_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wav = (
                root
                / "Uncontrol"
                / "dev"
                / "enrollment"
                / "wav"
                / "DF0014"
                / "DF0014_0002.wav"
            )
            label = (
                root
                / "Uncontrol"
                / "dev"
                / "enrollment"
                / "transcript"
                / "DF0014"
                / "label.txt"
            )
            write_wav(wav, channels=2)
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(
                "\ufeffDF0014_0002 小度小度\n", encoding="utf-8"
            )

            records = list(MDSCAdapter(root).iter_records())
            r = records[0]
            self.assertEqual(r.domain, "dysarthria")
            self.assertEqual(r.split, "dev")
            self.assertEqual(r.role, "enrollment")
            self.assertEqual(r.num_channels, 2)

    def test_uncontrol_eval_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wav = (
                root
                / "Uncontrol"
                / "test"
                / "eval"
                / "wav"
                / "DM0012"
                / "DM0012_0001.wav"
            )
            label = (
                root
                / "Uncontrol"
                / "test"
                / "eval"
                / "transcript"
                / "DM0012"
                / "label.txt"
            )
            write_wav(wav)
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(
                "\ufeffDM0012_0001 小度小度\n", encoding="utf-8"
            )

            r = list(MDSCAdapter(root).iter_records())[0]
            self.assertEqual(r.role, "eval")
            self.assertEqual(r.split, "test")
            self.assertEqual(r.speaker_id, "DM0012")


if __name__ == "__main__":
    unittest.main(verbosity=2)
