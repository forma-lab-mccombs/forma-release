#!/usr/bin/env python
"""Seeded regeneration of the Random Forest benchmark forecast.

RF weights are unshippable (one forest per (target, horizon) = 78 x 20 = 1,560
forests, tens of GB), so the paper's "seeded regeneration scripts otherwise"
clause applies: this script retrains and writes the forecast parquet from the
shipped config (single seed 62, hyper-parameters CV'd on the 8-year val fold,
refit on train+val).

    python scripts/regen_rf.py

COST/VENUE: intended for a cuML (RAPIDS) GPU path; on CPU sklearn it is very slow.
Budget on the order of tens of GPU-hours. Falls back to sklearn automatically if
cuML is unavailable. Requires a canonical ProForma-20Q build at data/processed
and the `[competitors]` extra installed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cfg = REPO_ROOT / "configs" / "r13_random_forest.yaml"
    if not cfg.exists():
        sys.exit(f"ERROR: config not found: {cfg}")
    print("NOTE: Random Forest regeneration is GPU-oriented (cuML) and can take tens "
          "of hours; a CPU sklearn fallback is used automatically if cuML is absent.")
    cmd = [sys.executable, "-m", "forma.train", "--config", str(cfg),
           "--forecast-splits", "test"]
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
