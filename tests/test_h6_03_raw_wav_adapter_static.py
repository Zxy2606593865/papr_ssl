from pathlib import Path
import ast
import unittest


class H603StaticTest(unittest.TestCase):
    def test_parse(self):
        for rel in (
            "src/papr_ssl/inference/h6_raw_wav_adapter.py",
            "scripts/run_h6_03_raw_wav_parity.py",
            "scripts/audit_h6_03_raw_wav_parity.py",
            "scripts/predict_h6_raw_wav.py",
        ):
            ast.parse(Path(rel).read_text(encoding="utf-8"))

    def test_frozen_model_identity(self):
        src = Path(
            "src/papr_ssl/inference/h6_raw_wav_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertIn('WAVLM_MODEL_ID = "microsoft/wavlm-large"', src)
        self.assertIn(
            'WAVLM_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"',
            src,
        )
        self.assertIn("HIDDEN_STATE_INDEX = 15", src)

    def test_parity_blocks_promotion(self):
        src = Path(
            "scripts/run_h6_03_raw_wav_parity.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"raw_wav_adapter_promoted": bool(passed)', src)
        self.assertIn('"generic_test": "sealed_not_accessed"', src)

    def test_predict_requires_pass(self):
        src = Path("scripts/predict_h6_raw_wav.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('if parity.get("status") != "PASS"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
