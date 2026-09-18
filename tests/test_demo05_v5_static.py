from pathlib import Path
import unittest

class Demo05V5StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = Path("demo/demo05_web/static/index.html").read_text(encoding="utf-8")
        cls.js = Path("demo/demo05_web/static/app.js").read_text(encoding="utf-8")

    def test_business_loop(self):
        self.assertIn("应用闭环", self.html)
        self.assertIn("智能学伴接入", self.html)

    def test_real_audio_metrics(self):
        self.assertIn("calculateAudioStats", self.js)
        self.assertIn("rmsDb", self.js)
        self.assertIn("peakDb", self.js)

    def test_runtime_requested_late(self):
        self.assertIn("13750", self.js)
        self.assertIn("/api/recognize/", self.js)

    def test_no_internal_intent_display(self):
        self.assertNotIn("intent_id", self.html)

    def test_stream_trace(self):
        self.assertIn("SUB_EVENTS", self.js)
        self.assertIn("queueEvent", self.js)

if __name__ == "__main__":
    unittest.main(verbosity=2)
