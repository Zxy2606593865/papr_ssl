from __future__ import annotations

import unittest

from papr_ssl.data.manifest_schema import AudioManifestRecord
from papr_ssl.data.mdsc_task_policy_v2 import (
    MDSC_COMMAND_20,
    OPEN_SET_LABEL,
    mdsc_command20_index,
    mdsc_core30_index,
    mdsc_core30_open_set_eval_index,
)
from papr_ssl.data.task_views import MDSC_COMMON_30, MDSC_WAKE_WORDS


def rec(utt, text, split="train", domain="control", role="train"):
    return AudioManifestRecord(
        utt_id=utt,
        dataset="mdsc",
        audio_relpath=f"{utt}.wav",
        speaker_id="S1",
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


class RevisedMDSCTaskPolicyTest(unittest.TestCase):
    def test_command20_is_core30_without_wake10(self):
        self.assertEqual(len(MDSC_COMMON_30), 30)
        self.assertEqual(len(MDSC_WAKE_WORDS), 10)
        self.assertEqual(len(MDSC_COMMAND_20), 20)
        self.assertTrue(set(MDSC_COMMAND_20).isdisjoint(MDSC_WAKE_WORDS))

    def test_core30_keeps_surface_normalization_only(self):
        rows = [
            rec("a", "小度<p>小度"),
            rec("b", "打开空调"),
            rec("c", "帮我打开空调"),
        ]
        out = mdsc_core30_index(rows)
        self.assertEqual({x.utt_id for x in out}, {"a", "b"})

    def test_command20_excludes_wake_words(self):
        rows = [
            rec("a", "小爱同学"),
            rec("b", "关闭空调"),
        ]
        out = mdsc_command20_index(rows)
        self.assertEqual([x.utt_id for x in out], ["b"])

    def test_open_set_uses_dev_test_only(self):
        rows = [
            rec("train_other", "帮我打开空调", "train"),
            rec("dev_other", "帮我打开空调", "dev", role="eval"),
            rec("test_core", "打开空调", "test", role="eval"),
        ]
        out = mdsc_core30_open_set_eval_index(rows)
        self.assertEqual([x.utt_id for x in out], ["dev_other"])
        self.assertEqual(out[0].task_label, OPEN_SET_LABEL)
        self.assertTrue(out[0].is_open_set)


if __name__ == "__main__":
    unittest.main(verbosity=2)
