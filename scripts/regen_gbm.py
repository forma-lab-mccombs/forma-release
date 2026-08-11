#!/usr/bin/env python
"""Seeded regeneration of the Chained-GBM benchmark forecasts (L1 and L2).

Chained-GBM weights (per-(chain-item, horizon) LightGBM boosters) are not
shipped; this regenerates the forecasts from the shipped configs. Two variants:
`glm.yaml` (L1 / MAE objective, Panel B) and `glm_mse.yaml` (L2 / MSE, Panel A).

    python scripts/regen_gbm.py            # both variants
    python scripts/regen_gbm.py --objective l1

BUILD DEPENDENCY (important): the chained GBM consumes the `pf_full_glm` feature
set under the `r13_node_optionD_indfe_val8_regfix` dataset tag. That build is NOT
produced by `proforma20q build` (which builds the standard `pf_full` set); it is
part of the research data pipeline. Absent that build, this script cannot run,
and the original GBM forecast parquets are not distributed either — so outside
the research pipeline the Table 1 GBM rows are neither reproducible from this
script nor available as banked forecasts. Requires the `[competitors]` extra.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANTS = {"l1": "glm.yaml", "l2": "glm_mse.yaml"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", choices=["l1", "l2", "both"], default="both")
    args = ap.parse_args()

    print("NOTE: requires the pf_full_glm / *_regfix build (research data pipeline; "
          "not produced by `proforma20q build`). See module docstring.")
    todo = ["l1", "l2"] if args.objective == "both" else [args.objective]
    for obj in todo:
        cfg = REPO_ROOT / "configs" / VARIANTS[obj]
        if not cfg.exists():
            sys.exit(f"ERROR: config not found: {cfg}")
        cmd = [sys.executable, "-m", "forma.train", "--config", str(cfg),
               "--forecast-splits", "test"]
        print(f"[chained_gbm {obj}] {' '.join(cmd)}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
