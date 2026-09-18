"""P2-07C integrity checks for the personalized MDSC WWS protocol."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable, Any

from papr_ssl.data.manifest_schema import AudioManifestRecord
from papr_ssl.data.task_views import (
    MDSC_WAKE_WORDS,
    canonical_mdsc_wake_word,
)


@dataclass(frozen=True)
class SpeakerProtocolStats:
    speaker_id: str
    split: str
    target_enrollment_count: int
    target_eval_count: int
    nonwake_enrollment_count: int
    nonwake_eval_count: int
    enrollment_wake_coverage: int
    eval_wake_coverage: int
    missing_enrollment_wakes: tuple[str, ...]
    missing_eval_wakes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["missing_enrollment_wakes"] = list(self.missing_enrollment_wakes)
        row["missing_eval_wakes"] = list(self.missing_eval_wakes)
        return row


def analyze_mdsc_personalized_protocol(
    records: Iterable[AudioManifestRecord],
) -> dict[str, Any]:
    wakes = set(MDSC_WAKE_WORDS)

    scoped = [
        r for r in records
        if (
            r.dataset == "mdsc"
            and r.record_type == "speech"
            and r.domain == "dysarthria"
            and r.split in {"dev", "test"}
            and r.role in {"enrollment", "eval"}
        )
    ]

    speaker_splits: dict[str, set[str]] = defaultdict(set)
    target_by_speaker_role: dict[
        tuple[str, str], Counter[str]
    ] = defaultdict(Counter)
    nonwake_by_speaker_role: Counter[tuple[str, str]] = Counter()

    for r in scoped:
        if r.speaker_id is None:
            raise ValueError(f"Missing speaker_id for {r.utt_id}")

        speaker_splits[r.speaker_id].add(r.split)
        canonical = canonical_mdsc_wake_word(r.label)

        if canonical is None:
            nonwake_by_speaker_role[(r.speaker_id, r.role)] += 1
        else:
            target_by_speaker_role[(r.speaker_id, r.role)][canonical] += 1

    multi_split_speakers = sorted(
        spk for spk, splits in speaker_splits.items()
        if len(splits) != 1
    )

    # IMPORTANT:
    # Derive split membership directly from speaker_splits, not from the
    # per-speaker summary row. A speaker that appears in both dev and test must
    # belong to BOTH sets so the leakage intersection is visible.
    dev_speakers = {
        spk for spk, splits in speaker_splits.items()
        if "dev" in splits
    }
    test_speakers = {
        spk for spk, splits in speaker_splits.items()
        if "test" in splits
    }

    speaker_rows: list[SpeakerProtocolStats] = []
    for speaker_id in sorted(speaker_splits):
        splits = sorted(speaker_splits[speaker_id])

        # In the valid protocol this is exactly one value.
        # If leakage exists, preserve that fact in the diagnostic summary
        # instead of silently selecting only dev or only test.
        split = splits[0] if len(splits) == 1 else "+".join(splits)

        enroll = target_by_speaker_role[(speaker_id, "enrollment")]
        eval_ = target_by_speaker_role[(speaker_id, "eval")]

        missing_enroll = tuple(sorted(wakes - set(enroll)))
        missing_eval = tuple(sorted(wakes - set(eval_)))

        speaker_rows.append(
            SpeakerProtocolStats(
                speaker_id=speaker_id,
                split=split,
                target_enrollment_count=sum(enroll.values()),
                target_eval_count=sum(eval_.values()),
                nonwake_enrollment_count=nonwake_by_speaker_role[
                    (speaker_id, "enrollment")
                ],
                nonwake_eval_count=nonwake_by_speaker_role[
                    (speaker_id, "eval")
                ],
                enrollment_wake_coverage=len(set(enroll) & wakes),
                eval_wake_coverage=len(set(eval_) & wakes),
                missing_enrollment_wakes=missing_enroll,
                missing_eval_wakes=missing_eval,
            )
        )

    all_have_10_enrollment = all(
        s.enrollment_wake_coverage == 10
        for s in speaker_rows
    )
    all_have_10_eval = all(
        s.eval_wake_coverage == 10
        for s in speaker_rows
    )
    all_have_nonwake_eval = all(
        s.nonwake_eval_count > 0
        for s in speaker_rows
    )

    # Exact target repetition distributions are reported but not hard-coded
    # into the feasibility gate. This keeps the integrity check about protocol
    # support rather than assuming an undocumented repetition count.
    enroll_per_wake_values = sorted({
        count
        for key, counter in target_by_speaker_role.items()
        if key[1] == "enrollment"
        for count in counter.values()
    })
    eval_per_wake_values = sorted({
        count
        for key, counter in target_by_speaker_role.items()
        if key[1] == "eval"
        for count in counter.values()
    })

    overlap = sorted(dev_speakers & test_speakers)

    gate = {
        "speaker_count_is_8": len(speaker_rows) == 8,
        "dev_test_speakers_disjoint": not bool(overlap),
        "one_split_per_speaker": not multi_split_speakers,
        "all_speakers_have_10_wakes_in_enrollment": all_have_10_enrollment,
        "all_speakers_have_10_wakes_in_eval": all_have_10_eval,
        "all_speakers_have_nonwake_eval_trials": all_have_nonwake_eval,
    }
    gate["overall"] = (
        "PASS"
        if all(v is True for k, v in gate.items() if k != "overall")
        else "FAIL"
    )

    return {
        "schema": "papr_ssl.mdsc_personalized_protocol_integrity.v1",
        "phase": "P2-07C",
        "scope_record_count": len(scoped),
        "speaker_count": len(speaker_rows),
        "split_speaker_counts": {
            "dev": len(dev_speakers),
            "test": len(test_speakers),
        },
        "dev_test_speaker_overlap": overlap,
        "multi_split_speakers": multi_split_speakers,
        "target_repetition_distribution": {
            "enrollment_per_wake_observed_values": enroll_per_wake_values,
            "eval_per_wake_observed_values": eval_per_wake_values,
        },
        "speakers": [s.to_dict() for s in speaker_rows],
        "gate": gate,
    }
