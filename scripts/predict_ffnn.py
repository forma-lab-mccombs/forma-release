#!/usr/bin/env python
# Copyright 2026 The University of Texas at Austin.
#
# Licensed under The University of Texas at Austin Research License, Version
# 1.1 ("License"). You MAY NOT use this file except in compliance with the
# License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, AND THE
# UNIVERSITY OF TEXAS AT AUSTIN EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
# CONCERNING THIS SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND
# WARRANTIES OF PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM
# COURSE OF DEALING OR USAGE OF TRADE. NO WARRANTY IS EITHER EXPRESS OR IMPLIED
# WITH RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no
# circumstances shall The University of Texas at Austin be liable for
# incidental, special, indirect, direct or consequential damages or loss of
# profits, interruption of business, or related expenses which may arise from
# use of Software or Documentation, including but not limited to those
# resulting from defects in Software and/or Documentation, or loss or
# inaccuracy of data of any kind.
#
# See the License for the specific language governing permissions and
# limitations under this License.

"""Regenerate the FFNN benchmark forecasts (heteroskedastic beta-NLL, 5 seeds).

FFNN training fuses fit->predict: one `forma.train` run per seed config both
trains and writes the forecast parquet. Trained FFNN weights are NOT shipped
(the paper's "seeded regeneration scripts otherwise" clause); the original
per-seed forecasts that produced Table 1's FFNN rows (Full R^2 0.253 / 0.247)
are published at
https://huggingface.co/datasets/forma-lab-mccombs/proforma-20q-artifacts
(gated; the dataset page presents its licence terms). This script regenerates them from
the shipped configs + seeds.

Run from a directory whose `data/processed` is a canonical ProForma-20Q build
(`proforma20q build` against the frozen canonical regularization stats), or pass
`--configs-dir`/paths accordingly.

    python scripts/predict_ffnn.py --variant linear
    python scripts/predict_ffnn.py --variant large --out results/forecasts

Combine the 5 per-seed forecasts into the mixture (Panel C uses the exact
mixture; Panel A uses the materialized mean-of-means) with:

    python scripts/group_seed_forecasts.py --materialize --arm ffnn_<variant>_b50 ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS = [60, 61, 62, 63, 64]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["linear", "large"], required=True)
    ap.add_argument("--configs-dir", type=Path, default=REPO_ROOT / "configs")
    ap.add_argument("--forecast-splits", default="test")
    args = ap.parse_args()

    print("NOTE: the FFNN `epochs` in each seed config is the value selected by the "
          "CV run (configs/r13_ffnn_betanll_cv.yaml); a GPU is recommended.")
    for seed in SEEDS:
        cfg = args.configs_dir / f"r13_ffnn_{args.variant}_betanll_seed{seed}.yaml"
        if not cfg.exists():
            sys.exit(f"ERROR: config not found: {cfg}")
        cmd = [sys.executable, "-m", "forma.train", "--config", str(cfg),
               "--forecast-splits", args.forecast_splits]
        print(f"[ffnn_{args.variant} seed {seed}] {' '.join(cmd)}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
