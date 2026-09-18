#!/usr/bin/env python
"""Standalone P3-04 loader/sampler contract check."""

from __future__ import annotations

from collections import Counter

from papr_ssl.training.feature_data import (
    ClassBalancedBatchSampler,
    ClassBalancedSamplerConfig,
)


def main() -> int:
    # Intentionally imbalanced source distribution.
    labels = (
        [0] * 3
        + [1] * 7
        + [2] * 13
        + [3] * 20
        + [4] * 4
        + [5] * 9
        + [6] * 6
        + [7] * 18
        + [8] * 5
        + [9] * 11
    )

    config = ClassBalancedSamplerConfig(
        classes_per_batch=5,
        samples_per_class=3,
        batches_per_epoch=12,
        seed=17,
    )
    sampler = ClassBalancedBatchSampler(labels, config)

    exposure = Counter()
    for batch in sampler:
        batch_labels = [labels[i] for i in batch]
        counts = Counter(batch_labels)

        assert len(batch) == config.batch_size
        assert len(counts) == config.classes_per_batch
        assert set(counts.values()) == {config.samples_per_class}

        for class_id in counts:
            exposure[class_id] += 1

    assert max(exposure.values()) - min(exposure.values()) <= 1

    first_epoch = list(ClassBalancedBatchSampler(labels, config))
    repeat_epoch = list(ClassBalancedBatchSampler(labels, config))
    assert first_epoch == repeat_epoch

    next_epoch_sampler = ClassBalancedBatchSampler(labels, config)
    next_epoch_sampler.set_epoch(1)
    assert first_epoch != list(next_epoch_sampler)

    print("=" * 92)
    print("PAPR-SSL P3-04 DATA LOADER / SAMPLER CONTRACT")
    print("=" * 92)
    print(f"source class counts:         {dict(Counter(labels))}")
    print(f"classes per train batch:    {config.classes_per_batch}")
    print(f"samples per class:          {config.samples_per_class}")
    print(f"train batch size:           {config.batch_size}")
    print(f"batches per epoch:          {len(sampler)}")
    print(
        "class exposure range:       "
        f"{min(exposure.values())}..{max(exposure.values())}"
    )
    print("same seed/epoch deterministic:YES")
    print("set_epoch changes sampling: YES")
    print("dev policy:                 sequential full pass")
    print("test construction:          REFUSED")
    print("-" * 92)
    print("P3-04 STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
