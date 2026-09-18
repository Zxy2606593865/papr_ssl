"""Frozen semantic task views for PAPR-SSL public datasets.

P2-07B separates:
- raw transcript identity stored in the immutable manifest;
- task-semantic canonical labels used by experiments.

MDSC wake-word normalization is deliberately conservative:
- remove the explicit <p> pause marker;
- remove whitespace;
- casefold for Latin text matching;
- otherwise do not rewrite, synonym-normalize, or ASR-correct transcripts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from papr_ssl.data.manifest_schema import AudioManifestRecord


GSC_ALL_35 = (
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven",
    "sheila", "six", "stop", "three", "tree", "two", "up", "visual",
    "wow", "yes", "zero",
)

MDSC_WAKE_WORDS = (
    "Hey Siri",
    "你好小布",
    "天猫精灵",
    "小冰小冰",
    "小度小度",
    "小德小德",
    "小溪你好",
    "小爱同学",
    "小艺小艺",
    "灵犀灵犀",
)

# Descriptive 30-phrase cross-speaker view found in P2-07A.
# It is retained only as an optional representation diagnostic view.
MDSC_COMMON_30 = (
    "Hey Siri",
    "你好小布",
    "全部关闭",
    "全部打开",
    "关灯",
    "关闭空调",
    "关闭窗帘",
    "天猫精灵",
    "小冰小冰",
    "小度小度",
    "小德小德",
    "小溪你好",
    "小爱同学",
    "小艺小艺",
    "帮忙暂停",
    "开启省电",
    "开灯",
    "打开上一集",
    "打开后一集",
    "打开空调",
    "打开蓝牙",
    "拉开窗帘",
    "播放上一首",
    "播放下一首",
    "替我取消暂停",
    "灵犀灵犀",
    "调到下一频道",
    "重新开始",
    "降低音量",
    "音量调大",
)


def normalize_mdsc_task_text(text: str) -> str:
    """Conservative task-only normalization, never written back to raw manifest."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    value = text.replace("<p>", "")
    value = re.sub(r"\s+", "", value)
    return value.casefold()


_WAKE_BY_NORMALIZED = {
    normalize_mdsc_task_text(label): label
    for label in MDSC_WAKE_WORDS
}


def canonical_mdsc_wake_word(text: str) -> Optional[str]:
    """Return canonical wake word or None for non-wake transcript."""
    return _WAKE_BY_NORMALIZED.get(normalize_mdsc_task_text(text))


def is_mdsc_wake_word(text: str) -> bool:
    return canonical_mdsc_wake_word(text) is not None


@dataclass(frozen=True)
class TaskIndexRecord:
    utt_id: str
    task_view: str
    task_label: str
    is_target: bool
    canonical_target: Optional[str]
    split: str
    domain: str
    role: str
    speaker_id: Optional[str]

    def to_dict(self) -> dict:
        return {
            "utt_id": self.utt_id,
            "task_view": self.task_view,
            "task_label": self.task_label,
            "is_target": self.is_target,
            "canonical_target": self.canonical_target,
            "split": self.split,
            "domain": self.domain,
            "role": self.role,
            "speaker_id": self.speaker_id,
        }


def gsc_closed_set_index(
    records: Iterable[AudioManifestRecord],
) -> list[TaskIndexRecord]:
    allowed = set(GSC_ALL_35)
    out = []
    for r in records:
        if r.dataset != "gsc_v2" or r.record_type != "speech":
            continue
        if r.label not in allowed:
            raise ValueError(f"Unexpected GSC label: {r.label!r}")
        out.append(
            TaskIndexRecord(
                utt_id=r.utt_id,
                task_view="gsc_all_35_closed_set",
                task_label=r.label,
                is_target=True,
                canonical_target=r.label,
                split=r.split,
                domain=r.domain,
                role=r.role,
                speaker_id=r.speaker_id,
            )
        )
    return out


def mdsc_personalized_wws_index(
    records: Iterable[AudioManifestRecord],
    *,
    include_splits: tuple[str, ...] = ("dev", "test"),
) -> list[TaskIndexRecord]:
    """Build speaker-dependent dysarthria enrollment/eval WWS task index.

    Only dysarthria-domain development/test enrollment/eval records belong to
    this personalized evaluation view.
    """
    allowed_splits = set(include_splits)
    out = []

    for r in records:
        if r.dataset != "mdsc" or r.record_type != "speech":
            continue
        if r.domain != "dysarthria":
            continue
        if r.split not in allowed_splits:
            continue
        if r.role not in {"enrollment", "eval"}:
            continue

        canonical = canonical_mdsc_wake_word(r.label)
        is_target = canonical is not None

        out.append(
            TaskIndexRecord(
                utt_id=r.utt_id,
                task_view="mdsc_personalized_wws_10",
                task_label=canonical if is_target else "__NON_WAKE__",
                is_target=is_target,
                canonical_target=canonical,
                split=r.split,
                domain=r.domain,
                role=r.role,
                speaker_id=r.speaker_id,
            )
        )

    return out


def mdsc_common30_index(
    records: Iterable[AudioManifestRecord],
) -> list[TaskIndexRecord]:
    """Optional 30-class phrase view for representation diagnostics."""
    allowed = set(MDSC_COMMON_30)
    out = []

    for r in records:
        if r.dataset != "mdsc" or r.record_type != "speech":
            continue

        canonical = canonical_mdsc_wake_word(r.label)
        task_label = canonical if canonical is not None else r.label

        if task_label not in allowed:
            continue

        out.append(
            TaskIndexRecord(
                utt_id=r.utt_id,
                task_view="mdsc_common_30_closed_set",
                task_label=task_label,
                is_target=task_label in set(MDSC_WAKE_WORDS),
                canonical_target=(
                    task_label if task_label in set(MDSC_WAKE_WORDS) else None
                ),
                split=r.split,
                domain=r.domain,
                role=r.role,
                speaker_id=r.speaker_id,
            )
        )

    return out
