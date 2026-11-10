"""Compatibility entry point for rby1.demos.auto_close_drawer."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rby1._compat import redirect
redirect(__name__, "rby1.demos.auto_close_drawer")
