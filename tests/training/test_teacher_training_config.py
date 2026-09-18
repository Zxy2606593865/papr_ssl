from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from papr_ssl.training.teacher_training_config import (
    config_signature,
    load_teacher_training_config,
    teacher_training_config_from_dict,
)


CONFIGS = (
    Path("configs/p3/teacher/wav2vec2_base.yaml"),
    Path("configs/p3/teacher/wavlm_large.yaml"),
    Path("configs/p3/teacher/w2v_bert2.yaml"),
)


class TeacherTrainingConfigTest(unittest.TestCase):
    def test_all_three_configs_validate(self):
        configs = [load_teacher_training_config(x) for x in CONFIGS]
        self.assertEqual(
            [x.backbone.name for x in configs],
            ["wav2vec2_base", "wavlm_large", "w2v_bert2"],
        )

    def test_all_configs_share_mean_dr_scaf_contract(self):
        configs = [load_teacher_training_config(x) for x in CONFIGS]
        signatures = {config_signature(x) for x in configs}
        self.assertEqual(len(signatures), 1)

        for cfg in configs:
            self.assertEqual(cfg.head.kind, "mean_dr")
            self.assertEqual(cfg.head.embedding_dim, 64)
            self.assertTrue(cfg.head.l2_normalize)
            self.assertEqual(cfg.scaf.k, 3)
            self.assertAlmostEqual(cfg.scaf.margin, 0.2)
            self.assertAlmostEqual(cfg.scaf.scale, 30.0)
            self.assertTrue(cfg.backbone.frozen)
            self.assertTrue(cfg.backbone.force_eval)
            self.assertTrue(cfg.backbone.use_offline_cache)

    def test_base_seed_is_17(self):
        for path in CONFIGS:
            cfg = load_teacher_training_config(path)
            self.assertEqual(cfg.seed, 17)

    def test_layer_means_exact_hidden_states_index(self):
        cfgs = [load_teacher_training_config(x) for x in CONFIGS]
        self.assertEqual(
            [x.backbone.layer for x in cfgs],
            [12, 24, 24],
        )

    def test_wrong_scaf_contract_fails(self):
        data = yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8"))
        data = copy.deepcopy(data)
        data["scaf"]["k"] = 2
        with self.assertRaises(ValueError):
            teacher_training_config_from_dict(data)

    def test_unfrozen_backbone_fails(self):
        data = yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8"))
        data = copy.deepcopy(data)
        data["backbone"]["frozen"] = False
        with self.assertRaises(ValueError):
            teacher_training_config_from_dict(data)

    def test_wrong_revision_fails(self):
        data = yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8"))
        data = copy.deepcopy(data)
        data["backbone"]["model_revision"] = "0" * 40
        with self.assertRaises(ValueError):
            teacher_training_config_from_dict(data)

    def test_out_of_range_layer_fails(self):
        data = yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8"))
        data = copy.deepcopy(data)
        data["backbone"]["layer"] = 13
        with self.assertRaises(ValueError):
            teacher_training_config_from_dict(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
