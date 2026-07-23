"""Make ``forma.*`` (the package) and ``scripts.*`` (the standalone scripts)
importable during tests whether or not the package is pip-installed.

CI installs the package (``pip install -e .``) so ``forma`` resolves anyway; this
belt-and-suspenders keeps `python -m pytest` green from a plain checkout too, and
puts the repo root on the path so tests can import ``scripts.mixture_calibration``.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
