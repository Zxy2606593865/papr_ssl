from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from papr_ssl.cache.ssl_feature_cache import (
    CacheSample,
    MeanFeatureCacheWriter,
    SSLCacheIdentity,
)
from papr_ssl.training.feature_data import (
    ClassBalancedBatchSampler,
    ClassBalancedSamplerConfig,
    OfflineFeatureTaskDataset,
    build_dev_loader,
    build_train_loader,
)


REV = "0123456789abcdef0123456789abcdef01234567"


def _write_task_index(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_cache(
    root: Path,
    rows: list[dict],
) -> tuple[Path, SSLCacheIdentity]:
    identity = SSLCacheIdentity(
        model_id="facebook/wav2vec2-base",
        model_revision=REV,
        layer=12,
    )
    writer = MeanFeatureCacheWriter(
        root,
        identity,
        shard_size=4,
    )
    for i, row in enumerate(rows):
        writer.add(
            CacheSample(
                utt_id=row["utt_id"],
                dataset=row["dataset"],
                split=row["split"],
                sample_hash=row["sample_hash"],
                frame_count=10 + i,
            ),
            torch.tensor(
                [float(i), float(i + 1), float(i + 2)],
                dtype=torch.float32,
            ),
        )
    return writer.finalize().parent, identity


class ClassBalancedSamplerTest(unittest.TestCase):
    def test_batch_has_exact_c_way_k_shot_composition(self):
        labels = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5
        sampler = ClassBalancedBatchSampler(
            labels,
            ClassBalancedSamplerConfig(
                classes_per_batch=3,
                samples_per_class=2,
                batches_per_epoch=8,
                seed=17,
            ),
        )
        for batch in sampler:
            batch_labels = [labels[i] for i in batch]
            counts = {
                c: batch_labels.count(c)
                for c in set(batch_labels)
            }
            self.assertEqual(len(counts), 3)
            self.assertEqual(sorted(counts.values()), [2, 2, 2])

    def test_class_exposure_is_balanced_over_epoch(self):
        labels = [0] * 3 + [1] * 9 + [2] * 20 + [3] * 7 + [4] * 2
        sampler = ClassBalancedBatchSampler(
            labels,
            ClassBalancedSamplerConfig(
                classes_per_batch=2,
                samples_per_class=2,
                batches_per_epoch=11,
                seed=29,
            ),
        )
        exposures = {c: 0 for c in range(5)}
        for batch in sampler:
            present = {labels[i] for i in batch}
            for c in present:
                exposures[c] += 1
        self.assertLessEqual(
            max(exposures.values()) - min(exposures.values()),
            1,
        )

    def test_same_seed_epoch_is_reproducible_and_next_epoch_changes(self):
        labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
        cfg = ClassBalancedSamplerConfig(
            classes_per_batch=2,
            samples_per_class=2,
            batches_per_epoch=5,
            seed=43,
        )
        a = ClassBalancedBatchSampler(labels, cfg)
        b = ClassBalancedBatchSampler(labels, cfg)
        self.assertEqual(list(a), list(b))

        a.set_epoch(1)
        self.assertNotEqual(list(a), list(b))

    def test_small_class_uses_explicit_replacement(self):
        labels = [0] + [1, 1, 1]
        sampler = ClassBalancedBatchSampler(
            labels,
            ClassBalancedSamplerConfig(
                classes_per_batch=2,
                samples_per_class=3,
                batches_per_epoch=1,
                seed=17,
            ),
        )
        batch = next(iter(sampler))
        class0_indices = [i for i in batch if labels[i] == 0]
        self.assertEqual(class0_indices, [0, 0, 0])


class OfflineFeatureDatasetTest(unittest.TestCase):
    def test_train_class_map_is_reused_by_dev(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                {
                    "dataset": "mdsc",
                    "utt_id": "tr_b",
                    "split": "train",
                    "task_label": "B",
                    "sample_hash": "a" * 64,
                },
                {
                    "dataset": "mdsc",
                    "utt_id": "tr_a",
                    "split": "train",
                    "task_label": "A",
                    "sample_hash": "b" * 64,
                },
                {
                    "dataset": "mdsc",
                    "utt_id": "dv_b",
                    "split": "dev",
                    "task_label": "B",
                    "sample_hash": "c" * 64,
                },
            ]
            task_index = root / "task.jsonl"
            _write_task_index(task_index, rows)
            cache_dir, identity = _make_cache(root / "cache", rows)

            train = OfflineFeatureTaskDataset(
                task_index=task_index,
                cache_dir=cache_dir,
                cache_identity=identity,
                split="train",
            )
            dev = OfflineFeatureTaskDataset(
                task_index=task_index,
                cache_dir=cache_dir,
                cache_identity=identity,
                split="dev",
                class_to_index=train.class_to_index,
            )
            self.assertEqual(train.class_to_index, {"A": 0, "B": 1})
            self.assertEqual(dev[0]["label"], 1)

    def test_task_cache_sample_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_rows = [
                {
                    "dataset": "gsc",
                    "utt_id": "u1",
                    "split": "train",
                    "task_label": "yes",
                    "sample_hash": "a" * 64,
                }
            ]
            task_rows = [
                {
                    **cache_rows[0],
                    "sample_hash": "b" * 64,
                }
            ]
            task_index = root / "task.jsonl"
            _write_task_index(task_index, task_rows)
            cache_dir, identity = _make_cache(root / "cache", cache_rows)

            with self.assertRaises(RuntimeError):
                OfflineFeatureTaskDataset(
                    task_index=task_index,
                    cache_dir=cache_dir,
                    cache_identity=identity,
                    split="train",
                )

    def test_test_split_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                {
                    "dataset": "mdsc",
                    "utt_id": "te1",
                    "split": "test",
                    "task_label": "A",
                    "sample_hash": "a" * 64,
                }
            ]
            task_index = root / "task.jsonl"
            _write_task_index(task_index, rows)
            cache_dir, identity = _make_cache(root / "cache", rows)

            with self.assertRaises(ValueError):
                OfflineFeatureTaskDataset(
                    task_index=task_index,
                    cache_dir=cache_dir,
                    cache_identity=identity,
                    split="test",
                )

    def test_train_loader_is_balanced_and_dev_is_deterministic_full_pass(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows: list[dict] = []
            n = 0
            for split, per_class in [("train", 4), ("dev", 2)]:
                for label in ["A", "B", "C"]:
                    for _ in range(per_class):
                        rows.append(
                            {
                                "dataset": "mdsc",
                                "utt_id": f"u{n}",
                                "split": split,
                                "task_label": label,
                                "sample_hash": f"{n:064x}"[-64:],
                            }
                        )
                        n += 1

            task_index = root / "task.jsonl"
            _write_task_index(task_index, rows)
            cache_dir, identity = _make_cache(root / "cache", rows)

            train = OfflineFeatureTaskDataset(
                task_index=task_index,
                cache_dir=cache_dir,
                cache_identity=identity,
                split="train",
            )
            dev = OfflineFeatureTaskDataset(
                task_index=task_index,
                cache_dir=cache_dir,
                cache_identity=identity,
                split="dev",
                class_to_index=train.class_to_index,
            )

            train_loader, sampler = build_train_loader(
                train,
                sampler_config=ClassBalancedSamplerConfig(
                    classes_per_batch=3,
                    samples_per_class=2,
                    batches_per_epoch=2,
                    seed=17,
                ),
            )
            batch = next(iter(train_loader))
            self.assertEqual(tuple(batch["features"].shape), (6, 3))
            counts = torch.bincount(
                batch["labels"],
                minlength=3,
            ).tolist()
            self.assertEqual(counts, [2, 2, 2])

            dev_loader = build_dev_loader(dev, batch_size=4)
            pass1 = [
                utt
                for b in dev_loader
                for utt in b["utt_ids"]
            ]
            pass2 = [
                utt
                for b in dev_loader
                for utt in b["utt_ids"]
            ]
            expected = [row.utt_id for row in dev.rows]
            self.assertEqual(pass1, expected)
            self.assertEqual(pass2, expected)
            self.assertEqual(len(pass1), len(dev))
            self.assertEqual(len(sampler), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
