from __future__ import annotations

import unittest

from papr_ssl.data.manifest_schema import AudioManifestRecord
from papr_ssl.data.task_views import (
    GSC_ALL_35,
    MDSC_WAKE_WORDS,
    canonical_mdsc_wake_word,
    is_mdsc_wake_word,
    mdsc_personalized_wws_index,
    normalize_mdsc_task_text,
)


def mdsc_record(
    *,
    utt,
    label,
    split="dev",
    role="eval",
    domain="dysarthria",
    speaker="DF0001",
):
    return AudioManifestRecord(
        utt_id=utt,
        dataset="mdsc",
        audio_relpath=f"{utt}.wav",
        speaker_id=speaker,
        label=label,
        transcript=label,
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


class FrozenTaskViewTest(unittest.TestCase):
    def test_frozen_inventory_sizes(self):
        self.assertEqual(len(GSC_ALL_35), 35)
        self.assertEqual(len(MDSC_WAKE_WORDS), 10)
        self.assertEqual(len(set(MDSC_WAKE_WORDS)), 10)

    def test_pause_marker_variant_maps_to_same_wake_word(self):
        self.assertEqual(
            canonical_mdsc_wake_word("小度<p>小度"),
            "小度小度",
        )
        self.assertEqual(
            canonical_mdsc_wake_word("你<p>好<p>小<p>布"),
            "你好小布",
        )
        self.assertEqual(
            canonical_mdsc_wake_word("小<p>德<p>小<p>德"),
            "小德小德",
        )

    def test_latin_whitespace_normalization(self):
        self.assertEqual(
            canonical_mdsc_wake_word("Hey   Siri"),
            "Hey Siri",
        )
        self.assertEqual(
            canonical_mdsc_wake_word("HEY SIRI"),
            "Hey Siri",
        )

    def test_nonwake_command_is_not_target(self):
        self.assertFalse(is_mdsc_wake_word("打开空调"))
        self.assertFalse(is_mdsc_wake_word("音量调大"))

    def test_personalized_view_keeps_only_dysarthria_dev_test_enroll_eval(self):
        rows = [
            mdsc_record(
                utt="wake",
                label="小爱同学",
                role="enrollment",
            ),
            mdsc_record(
                utt="neg",
                label="打开空调",
                role="eval",
            ),
            mdsc_record(
                utt="train",
                label="小爱同学",
                split="train",
                role="train",
            ),
            mdsc_record(
                utt="control",
                label="小爱同学",
                domain="control",
                role="eval",
            ),
        ]

        view = mdsc_personalized_wws_index(rows)

        self.assertEqual([x.utt_id for x in view], ["wake", "neg"])
        self.assertTrue(view[0].is_target)
        self.assertEqual(view[0].task_label, "小爱同学")
        self.assertFalse(view[1].is_target)
        self.assertEqual(view[1].task_label, "__NON_WAKE__")


if __name__ == "__main__":
    unittest.main(verbosity=2)
