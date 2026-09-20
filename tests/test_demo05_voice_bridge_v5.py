from __future__ import annotations

import hashlib
from pathlib import Path
import unittest


STATIC_ROOT = Path("demo/demo05_web/static")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VoiceBridgeV5ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        cls.css = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
        cls.js = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        cls.presentation = (STATIC_ROOT / "presentation.js").read_text(
            encoding="utf-8"
        )

    def test_reference_visual_files_are_byte_exact(self):
        expected = {
            "styles.css": "a4e2cab6f80ec5f2f88a473c890c2eabddc3f2245c9f984f27bc44c45208bc33",
            "presentation.js": "ad70583b4c7d29bc968817b30dea5235d8e894e6e1abf67574e4a918738029a9",
            "assets/agent-wave-poster.jpg": "53d5c26da25ab7186b9aa609f493add5fe7210636ed6aa6ffcb48ff409635047",
            "assets/agent-wave.mp4": "dbcb3d586673bb3166263ac3dbc5e3440541db61be853ad006906577c5b5af90",
            "assets/paper.svg": "3d71ef3d287cd4643fc6cdcd8d88e05929f633e7ac655836aa37d894c45c96df",
        }
        for relative, digest in expected.items():
            with self.subTest(relative=relative):
                self.assertEqual(sha256(STATIC_ROOT / relative), digest)

    def test_original_motion_and_accessibility_contract(self):
        self.assertIn('id="agentWave" autoplay muted loop playsinline', self.html)
        self.assertIn('const reduced = matchMedia("(prefers-reduced-motion: reduce)")', self.presentation)
        self.assertIn('.pipeline-art{display:none}', self.css)

    def test_continuous_recognition_is_the_only_product_extension(self):
        self.assertIn("长语音识别", self.html)
        self.assertIn("/api/continuous/recognize-demo", self.js)
        self.assertIn("Continuous Phrase Spotter", self.html)
        self.assertNotIn("/api/continuous/recognize-raw", self.js)

    def test_frontend_validates_continuous_timeline(self):
        self.assertIn("长语音接口响应缺少 detections", self.js)
        self.assertIn("发生异常重叠", self.js)
        self.assertIn("超出长语音时长", self.js)
        self.assertIn("120000", self.js)

    def test_continuous_mode_supports_the_same_30_second_presentation(self):
        self.assertIn("const CONTINUOUS_EXPLANATIONS = [", self.js)
        self.assertIn(
            "presentUntilOutcome(token,started,()=>outcome,CONTINUOUS_EXPLANATIONS",
            self.js,
        )
        self.assertIn("长语音流程讲解结束", self.js)
        self.assertNotIn(
            'b.disabled=state.running||state.session==="sequence"', self.js
        )
        self.assertNotIn(
            "if(state.running||state.session==='sequence')return;setMode", self.js
        )

    def test_presentation_wait_is_not_reported_as_backend_latency(self):
        self.assertIn("接口往返", self.js)
        self.assertNotIn("前端等待", self.js)


if __name__ == "__main__":
    unittest.main(verbosity=2)
