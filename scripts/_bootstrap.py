"""让 scripts/ 下的入口可以直接 `python scripts/xxx.py` 运行并 import src。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
