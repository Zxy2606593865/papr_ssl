from pathlib import Path
import unittest

@unittest.skip("Demo-05 v7 visual layout was superseded by the v9 contract")
class Demo05V7StyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.css = Path("demo/demo05_web/static/styles.css").read_text(encoding="utf-8")

    def test_one_screen(self):
        self.assertIn("height:100dvh", self.css)
        self.assertIn("grid-template-rows:72px 54px minmax(0,1fr) 24px", self.css)

    def test_overview_weakened(self):
        self.assertIn("opacity:.86", self.css)
        self.assertIn("height:54px", self.css)

    def test_event_stream_large_text(self):
        self.assertIn("font-size:13px", self.css)
        self.assertIn("grid-template-columns:86px 1fr 82px", self.css)

    def test_aligned_columns(self):
        self.assertIn("grid-template-rows:1fr 1.34fr .72fr", self.css)
        self.assertIn("grid-template-rows:minmax(0,1.12fr) minmax(270px,.88fr)", self.css)
        self.assertIn("grid-template-rows:minmax(0,1.24fr) minmax(0,.76fr)", self.css)

if __name__ == "__main__":
    unittest.main(verbosity=2)
