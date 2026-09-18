#!/usr/bin/env python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import papr_ssl
from papr_ssl.inference.h6_personalized_runtime import PersonalizedRuntime, UserMemory

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SRC_ROOT:", SRC_ROOT)
print("papr_ssl loaded from:", papr_ssl.__file__)
print("runtime module import: PASS")
