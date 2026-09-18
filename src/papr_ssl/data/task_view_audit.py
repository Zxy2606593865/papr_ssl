"""P2-07A task-view and label-inventory audit utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable, Any

from papr_ssl.data.manifest_schema import AudioManifestRecord


@dataclass(frozen=True)
class LabelStats:
    label: str
    count: int
    speaker_count: int
    train_count: int
    enrollment_count: int
    eval_count: int
    control_count: int
    dysarthria_count: int
    train_speaker_count: int
    enrollment_speaker_count: int
    eval_speaker_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_label_stats(
    records: Iterable[AudioManifestRecord],
) -> list[LabelStats]:
    rows = list(records)

    counts = Counter()
    speakers = defaultdict(set)
    role_counts = defaultdict(Counter)
    domain_counts = defaultdict(Counter)
    role_speakers = defaultdict(lambda: defaultdict(set))

    for r in rows:
        if r.record_type != "speech" or r.label is None:
            continue
        label = r.label
        counts[label] += 1

        if r.speaker_id is not None:
            speakers[label].add(r.speaker_id)
            role_speakers[label][r.role].add(r.speaker_id)

        role_counts[label][r.role] += 1
        domain_counts[label][r.domain] += 1

    result = []
    for label in sorted(counts):
        result.append(
            LabelStats(
                label=label,
                count=counts[label],
                speaker_count=len(speakers[label]),
                train_count=role_counts[label]["train"],
                enrollment_count=role_counts[label]["enrollment"],
                eval_count=role_counts[label]["eval"],
                control_count=domain_counts[label]["control"],
                dysarthria_count=domain_counts[label]["dysarthria"],
                train_speaker_count=len(role_speakers[label]["train"]),
                enrollment_speaker_count=len(role_speakers[label]["enrollment"]),
                eval_speaker_count=len(role_speakers[label]["eval"]),
            )
        )
    return result


def build_mdsc_candidate_views(
    stats: list[LabelStats],
    *,
    total_speakers: int,
) -> dict[str, list[str]]:
    """Create descriptive candidate inventories; no semantic target claim."""
    def labels_where(pred):
        return sorted(s.label for s in stats if pred(s))

    return {
        "all_transcripts": labels_where(lambda s: True),
        "seen_in_all_speakers": labels_where(
            lambda s: s.speaker_count == total_speakers
        ),
        "control_and_dysarthria_overlap": labels_where(
            lambda s: s.control_count > 0 and s.dysarthria_count > 0
        ),
        "train_and_eval_overlap": labels_where(
            lambda s: s.train_count > 0 and s.eval_count > 0
        ),
        "enrollment_and_eval_overlap": labels_where(
            lambda s: s.enrollment_count > 0 and s.eval_count > 0
        ),
        "train_enrollment_eval_overlap": labels_where(
            lambda s: (
                s.train_count > 0
                and s.enrollment_count > 0
                and s.eval_count > 0
            )
        ),
        "eval_only_relative_to_enrollment": labels_where(
            lambda s: s.eval_count > 0 and s.enrollment_count == 0
        ),
        "enrollment_only_relative_to_eval": labels_where(
            lambda s: s.enrollment_count > 0 and s.eval_count == 0
        ),
    }


def summarize_views(views: dict[str, list[str]]) -> dict[str, int]:
    return {name: len(labels) for name, labels in views.items()}
