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

        expected_revisions = {
            "wav2vec2": "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
            "wavlm": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
            "w2v_bert2": "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
        }

        for config in configs:
            self.assertEqual(config["seed"], 17)
            self.assertEqual(config["head"], {"kind": "mean_dr", "output_dim": 64})
            self.assertEqual(
                config["scaf"],
                {"subcenters": 3, "margin_rad": 0.2, "scale": 30.0},
            )

            kind = config["backbone"]["kind"]
            revision = config["backbone"]["revision"]

            self.assertEqual(revision, expected_revisions[kind])
            self.assertNotEqual(revision, "main")
            self.assertEqual(len(revision), 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
