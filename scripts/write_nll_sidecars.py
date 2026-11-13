#!/usr/bin/env python3
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

"""Write `{stem}.nll.json` predictive-density sidecars for non-Gaussian forecast arms.

`evaluate.py` resolves each forecast's predictive density (for the NLL and CRPS
likelihood tracks) from an OPTIONAL `{stem}.nll.json` sidecar
(`{"family": "gaussian"|"laplace"|"student_t", "df": <float|null>}`); when absent it
defaults to **Gaussian**. So Laplace / Student-t arms REQUIRE a sidecar, or they are
silently scored as Gaussian — which quietly invalidates any Laplace/Student-t-vs-
Gaussian comparison. This bit the r11 CRPS/NLL pool and R13's Laplace arm: until
the R14 fix, train.py's streaming forecast save (the only one in use for Forma)
wrote no sidecar at all, and filing forecasts into a store carried only the
parquet, so the sidecars were staged by hand.

This idempotent utility scans a forecast directory, infers the family from the model
name embedded in the filename (`*laplace*` or a `_lapNN` arm token such as
`forma_lap05_fgrid` -> laplace, `*student_t*` -> student_t), and writes the missing
sidecar. Gaussian arms need none and are skipped. Re-run it after filing new
non-Gaussian forecasts into a store. When a name says nothing about the density, name
the files with --glob and the family with --family instead of relying on inference.

Usage:
  python scripts/write_nll_sidecars.py <forecasts>/all
  python scripts/write_nll_sidecars.py <dir> [--student-t-df 6.0] [--force] [--dry-run]
  python scripts/write_nll_sidecars.py <dir> --glob "forma_lap05_fgrid__pf_full__*__test__predictions.parquet" --family laplace
"""
import argparse
import json
import re
from pathlib import Path


def infer_family(stem: str) -> str:
    """Infer the predictive-density family from the model name in the filename."""
    s = stem.lower()
    if 'laplace' in s or re.search(r'(^|_)lap[0-9]+(_|$)', s):
        return 'laplace'
    if 'student_t' in s or 'student-t' in s or 'studentt' in s:
        return 'student_t'
    return 'gaussian'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('directory', help='Forecast dir to scan; sidecars are written in place.')
    ap.add_argument('--student-t-df', type=float, default=6.0,
                    help='Degrees of freedom for student_t arms (default 6.0, the value all '
                         'repo student_t configs use). df is NOT inferable from the filename, '
                         'so verify it for any non-r9e5 student_t arm.')
    ap.add_argument('--glob', default='*__predictions.parquet',
                    help='Forecast files to consider, as a pattern relative to DIRECTORY '
                         '(default: every *__predictions.parquet). Only .parquet files count, '
                         'so stray .tmp files and existing sidecars are never labelled.')
    ap.add_argument('--family', choices=('laplace', 'student_t'), default=None,
                    help='Use this family for every --glob match instead of inferring it '
                         'from the filename.')
    ap.add_argument('--force', action='store_true', help='Overwrite existing sidecars.')
    ap.add_argument('--dry-run', action='store_true', help='Report only; write nothing.')
    args = ap.parse_args()

    d = Path(args.directory)
    if not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")

    written = present = gauss = 0
    try:
        matches = sorted(p for p in d.glob(args.glob)
                         if p.is_file() and p.name.endswith('.parquet'))
    except (NotImplementedError, ValueError) as e:
        raise SystemExit(f"--glob must be a pattern relative to {d}: {e}")
    if args.family and not matches:
        raise SystemExit(f"--family {args.family}: no files match {args.glob!r} in {d}")
    for fc in matches:
        fam = args.family or infer_family(fc.stem)
        if fam == 'gaussian':
            gauss += 1
            continue
        side = fc.parent / f"{fc.stem}.nll.json"           # matches evaluate._read_nll_meta
        if side.exists() and not args.force:
            present += 1
            continue
        meta = {'family': fam, 'df': (args.student_t_df if fam == 'student_t' else None)}
        if fam == 'student_t':
            print(f"  [ASSUMING df={args.student_t_df}] {fc.name}")
        if args.dry_run:
            print(f"  DRY: would write {side.name} -> {meta}")
        else:
            side.write_text(json.dumps(meta), encoding='ascii')
            print(f"  wrote {side.name} -> {meta}")
        written += 1

    prefix = 'DRY-RUN: ' if args.dry_run else ''
    print(f"\n{prefix}sidecars written={written}, already-present={present}, gaussian-skipped={gauss}")


if __name__ == '__main__':
    main()
