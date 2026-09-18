from __future__ import annotations
import ast
import unittest
from pathlib import Path


class P210FinalExportStaticTest(unittest.TestCase):
    def test_export_script_parses(self):
        path = Path("scripts/export_p2_final_data_protocol.py")
        ast.parse(path.read_text(encoding="utf-8"))

    def test_spec_exists(self):
        self.assertTrue(Path("docs/P2_10_FINAL_DATA_PROTOCOL_EXPORT.md").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
