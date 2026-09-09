from __future__ import annotations

import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TeacherConfigTest(unittest.TestCase):
    def test_all_backbone_configs_share_head_scaf_and_seed(self) -> None:
        paths = sorted((PROJECT_ROOT / "configs" / "teacher").glob("*_mean.yaml"))
        self.assertEqual(len(paths), 3)
        configs = []
        for path in paths:
            with path.open("r", encoding="utf-8") as file:
                configs.append(yaml.safe_load(file))

        self.assertEqual(
            {config["backbone"]["kind"] for config in configs},
            {"wav2vec2", "wavlm", "w2v_bert2"},
        )
        for config in configs:
            self.assertEqual(config["seed"], 17)
            self.assertEqual(config["head"], {"kind": "mean_dr", "output_dim": 64})
            self.assertEqual(
                config["scaf"],
                {"subcenters": 3, "margin_rad": 0.2, "scale": 30.0},
            )
            self.assertEqual(config["backbone"]["revision"], "main")


if __name__ == "__main__":
    unittest.main(verbosity=2)
