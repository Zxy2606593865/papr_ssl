from pathlib import Path
import unittest

class Demo05V9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = Path("demo/demo05_web/static/index.html").read_text(encoding="utf-8")
        cls.css = Path("demo/demo05_web/static/styles.css").read_text(encoding="utf-8")
        cls.js = Path("demo/demo05_web/static/app.js").read_text(encoding="utf-8")

    def test_light_gov_ui(self):
        self.assertIn("--page:#f2f5f8", self.css)
        self.assertIn("--surface:#ffffff", self.css)
        self.assertIn("--blue:#1f5fa8", self.css)

    def test_no_overview_strip(self):
        self.assertNotIn('class="overview-strip"', self.html)

    def test_dual_charts_preserved(self):
        self.assertIn('id="specCanvas"', self.html)
        self.assertIn('id="freqCanvas"', self.html)

    def test_trace_is_prominent(self):
        self.assertIn("运行处理流水", self.html)
        self.assertIn("trace-table-head", self.html)

    def test_chinese_states(self):
        for s in ("已接收","读取中","分析中","匹配中","校验中","判断中","已完成"):
            self.assertIn(s, self.js)

    def test_runtime_api_preserved(self):
        self.assertIn("/api/recognize/", self.js)
        self.assertIn("13750", self.js)

if __name__ == "__main__":
    unittest.main(verbosity=2)
