from __future__ import annotations

import unittest

from papr_ssl.data.manifest_schema import AudioManifestRecord
from papr_ssl.data.task_view_audit import (
    build_label_stats,
    build_mdsc_candidate_views,
)


def rec(
    *,
    utt,
    label,
    speaker,
    role,
    domain,
    split="train",
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


class TaskViewAuditTest(unittest.TestCase):
    def test_label_stats_counts_roles_domains_and_speakers(self):
        rows = [
            rec(
                utt="a1", label="A", speaker="s1",
                role="train", domain="control"
            ),
            rec(
                utt="a2", label="A", speaker="s2",
                role="enrollment", domain="dysarthria"
            ),
            rec(
                utt="a3", label="A", speaker="s2",
                role="eval", domain="dysarthria", split="test"
            ),
        ]
        stats = build_label_stats(rows)
        self.assertEqual(len(stats), 1)
        s = stats[0]
        self.assertEqual(s.count, 3)
        self.assertEqual(s.speaker_count, 2)
        self.assertEqual(s.train_count, 1)
        self.assertEqual(s.enrollment_count, 1)
        self.assertEqual(s.eval_count, 1)
        self.assertEqual(s.control_count, 1)
        self.assertEqual(s.dysarthria_count, 2)

    def test_enrollment_eval_overlap(self):
        rows = [
            rec(
                utt="a1", label="A", speaker="s1",
                role="enrollment", domain="dysarthria", split="dev"
            ),
            rec(
                utt="a2", label="A", speaker="s1",
                role="eval", domain="dysarthria", split="dev"
            ),
            rec(
                utt="b1", label="B", speaker="s1",
                role="eval", domain="dysarthria", split="dev"
            ),
        ]
        stats = build_label_stats(rows)
        views = build_mdsc_candidate_views(stats, total_speakers=1)
        self.assertEqual(views["enrollment_and_eval_overlap"], ["A"])
        self.assertEqual(
            views["eval_only_relative_to_enrollment"],
            ["B"],
        )

    def test_seen_in_all_speakers(self):
        rows = [
            rec(
                utt="a1", label="A", speaker="s1",
                role="train", domain="control"
            ),
            rec(
                utt="a2", label="A", speaker="s2",
                role="train", domain="dysarthria"
            ),
            rec(
                utt="b1", label="B", speaker="s1",
                role="train", domain="control"
            ),
        ]
        stats = build_label_stats(rows)
        views = build_mdsc_candidate_views(stats, total_speakers=2)
        self.assertEqual(views["seen_in_all_speakers"], ["A"])

    def test_train_enrollment_eval_intersection(self):
        rows = [
            rec(
                utt="a1", label="A", speaker="s1",
                role="train", domain="control"
            ),
            rec(
                utt="a2", label="A", speaker="s2",
                role="enrollment", domain="dysarthria", split="dev"
            ),
            rec(
                utt="a3", label="A", speaker="s2",
                role="eval", domain="dysarthria", split="dev"
            ),
        ]
        stats = build_label_stats(rows)
        views = build_mdsc_candidate_views(stats, total_speakers=2)
        self.assertEqual(
            views["train_enrollment_eval_overlap"],
            ["A"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
