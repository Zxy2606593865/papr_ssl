from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from papr_ssl.data.audio_pipeline import (
    GSC_FIXED_NUM_SAMPLES,
    collate_audio_examples,
    load_audio_example,
)
from papr_ssl.data.manifest_schema import AudioManifestRecord


def write_wav(
    path: Path,
    *,
    frames: int,
    channels: int = 1,
    value: float = 0.25,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 1:
        audio = np.full((frames,), value, dtype=np.float32)
    else:
        audio = np.stack(
            [
                np.full((frames,), value, dtype=np.float32),
                np.full((frames,), -value, dtype=np.float32),
            ],
            axis=1,
        )
    sf.write(path, audio, 16000, subtype="PCM_16")


def record(
    *,
    dataset: str,
    rel: str,
    frames: int,
    channels: int,
    split: str = "train",
    domain: str,
    role: str,
    speaker: str = "spk",
    label: str = "label",
) -> AudioManifestRecord:
    return AudioManifestRecord(
        utt_id=f"{dataset}:{rel}",
        dataset=dataset,
        audio_relpath=rel,
        speaker_id=speaker,
        label=label,
        transcript=label,
        language="en" if dataset == "gsc_v2" else "zh-CN",
        split=split,
        domain=domain,
        role=role,
        record_type="speech",
        sample_rate_hz=16000,
        num_channels=channels,
        num_frames=frames,
        duration_sec=frames / 16000.0,
        source_meta={},
    )


class AudioPipelineTest(unittest.TestCase):
    def test_gsc_short_audio_right_pads_but_keeps_valid_length(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rel = "yes/a_nohash_0.wav"
            write_wav(root / rel, frames=12000)
            r = record(
                dataset="gsc_v2",
                rel=rel,
                frames=12000,
                channels=1,
                domain="standard",
                role="train",
            )

            x = load_audio_example(r, {"gsc_v2": root})
            self.assertEqual(x.waveform.shape, (GSC_FIXED_NUM_SAMPLES,))
            self.assertEqual(x.valid_num_samples, 12000)
            self.assertTrue(torch.all(x.waveform[12000:] == 0))

    def test_gsc_longer_than_one_second_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rel = "yes/a_nohash_0.wav"
            write_wav(root / rel, frames=16001)
            r = record(
                dataset="gsc_v2",
                rel=rel,
                frames=16001,
                channels=1,
                domain="standard",
                role="train",
            )

            with self.assertRaises(ValueError):
                load_audio_example(r, {"gsc_v2": root})

    def test_mdsc_full_utterance_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rel = "Control/train/wav/CM0002/CM0002_0001.wav"
            write_wav(root / rel, frames=48000)
            r = record(
                dataset="mdsc",
                rel=rel,
                frames=48000,
                channels=1,
                domain="control",
                role="train",
                speaker="CM0002",
                label="小度小度",
            )

            x = load_audio_example(r, {"mdsc": root})
            self.assertEqual(x.waveform.shape, (48000,))
            self.assertEqual(x.valid_num_samples, 48000)

    def test_mdsc_stereo_is_meaned_to_mono(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rel = "Uncontrol/dev/eval/wav/DF0014/DF0014_0001.wav"
            write_wav(root / rel, frames=32000, channels=2, value=0.25)
            r = record(
                dataset="mdsc",
                rel=rel,
                frames=32000,
                channels=2,
                split="dev",
                domain="dysarthria",
                role="eval",
                speaker="DF0014",
                label="小爱同学",
            )

            x = load_audio_example(r, {"mdsc": root})
            self.assertEqual(x.waveform.ndim, 1)
            self.assertEqual(x.waveform.numel(), 32000)
            # +0.25 and -0.25 channels -> mean approximately zero.
            self.assertLess(float(x.waveform.abs().max()), 1e-3)

    def test_collate_emits_true_lengths_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            rel1 = "Control/train/wav/CM0001/CM0001_0001.wav"
            rel2 = "Control/train/wav/CM0002/CM0002_0001.wav"
            write_wav(root / rel1, frames=16000)
            write_wav(root / rel2, frames=40000)

            r1 = record(
                dataset="mdsc",
                rel=rel1,
                frames=16000,
                channels=1,
                domain="control",
                role="train",
                speaker="CM0001",
            )
            r2 = record(
                dataset="mdsc",
                rel=rel2,
                frames=40000,
                channels=1,
                domain="control",
                role="train",
                speaker="CM0002",
            )

            e1 = load_audio_example(r1, {"mdsc": root})
            e2 = load_audio_example(r2, {"mdsc": root})
            batch = collate_audio_examples([e1, e2])

            self.assertEqual(batch.waveforms.shape, (2, 40000))
            self.assertEqual(batch.waveform_mask.shape, (2, 40000))
            self.assertEqual(batch.lengths.tolist(), [16000, 40000])
            self.assertEqual(int(batch.waveform_mask[0].sum()), 16000)
            self.assertEqual(int(batch.waveform_mask[1].sum()), 40000)
            self.assertTrue(torch.all(batch.waveforms[0, 16000:] == 0))

    def test_collate_gsc_masks_internal_container_padding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rel = "yes/a_nohash_0.wav"
            write_wav(root / rel, frames=10000)
            r = record(
                dataset="gsc_v2",
                rel=rel,
                frames=10000,
                channels=1,
                domain="standard",
                role="train",
            )
            ex = load_audio_example(r, {"gsc_v2": root})
            batch = collate_audio_examples([ex])

            self.assertEqual(batch.waveforms.shape, (1, 16000))
            self.assertEqual(batch.lengths.tolist(), [10000])
            self.assertEqual(int(batch.waveform_mask.sum()), 10000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
