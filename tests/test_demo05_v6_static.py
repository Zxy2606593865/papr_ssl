from pathlib import Path
import unittest

@unittest.skip("Legacy Demo-05 v6 contract was superseded by the Voice Bridge v5 reference UI")
class Demo05V6StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = Path("demo/demo05_web/static/index.html").read_text(encoding="utf-8")
        cls.js = Path("demo/demo05_web/static/app.js").read_text(encoding="utf-8")

    def test_removed_modules(self):
        self.assertNotIn("试点说明", self.html)
        self.assertNotIn(">已注册规范表达</b><span>PHRASE LIBRARY", self.html)

    def test_dual_analysis(self):
        self.assertIn("时频音谱图", self.html)
        self.assertIn("实时频谱图", self.html)
        self.assertIn("freqCanvas", self.html)
        self.assertIn("specCanvas", self.html)

    def test_interaction_buttons(self):
        for token in (
            "clearStreamBtn",
            "exportStreamBtn",
            "copyResultBtn",
            "fullscreenBtn",
            "playOriginalBtn",
            "restartBtn",
            "pauseStreamBtn",
        ):
            self.assertIn(token, self.html)

    @unittest.skip("Demo-05 v6 large-stream layout was superseded by v9")
    def test_large_stream(self):
        self.assertIn("event-panel large", self.html)
        self.assertIn("exportStream", self.js)

    def test_real_spectrum_math(self):
        self.assertIn("linearMagnitude", self.js)
        self.assertIn("dominantFreq", self.js)
        self.assertIn("sampleRate", self.js)

if __name__ == "__main__":
    unittest.main(verbosity=2)
