# -*- coding: utf-8 -*-
"""让测试无论用 pytest 还是 unittest 运行，都能 import 到项目包。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
