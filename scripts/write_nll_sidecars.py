#!/usr/bin/env python3
"""Write `{stem}.nll.json` predictive-density sidecars for non-Gaussian forecast arms.

`evaluate.py` resolves each forecast's predictive density (for the NLL and CRPS
likelihood tracks) from an OPTIONAL `{stem}.nll.json` sidecar
(`{"family": "gaussian"|"laplace"|"student_t", "df": <float|null>}`); when absent it
defaults to **Gaussian**. So Laplace / Student-t arms REQUIRE a sidecar, or they are
silently scored as Gaussian — which quietly invalidates any Laplace/Student-t-vs-
Gaussian comparison (this bit the r11 CRPS/NLL pool: the sidecars `train.py` writes
into each run dir were not carried over when forecasts were filed into the canonical
store `<forecasts>/all`).

This idempotent utility scans a forecast directory, infers the family from the model
name embedded in the filename (`*laplace*` -> laplace, `*student_t*` -> student_t),
and writes the missing sidecar. Gaussian arms need none and are skipped. Re-run it
after filing new non-Gaussian forecasts into a store.

Usage:
  python scripts/write_nll_sidecars.py <forecasts>/all
  python scripts/write_nll_sidecars.py <dir> [--student-t-df 6.0] [--force] [--dry-run]
"""
import argparse
import json
from pathlib import Path


def infer_family(stem: str) -> str:
    """Infer the predictive-density family from the model name in the filename."""
    s = stem.lower()
    if 'laplace' in s:
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
    ap.add_argument('--force', action='store_true', help='Overwrite existing sidecars.')
    ap.add_argument('--dry-run', action='store_true', help='Report only; write nothing.')
    args = ap.parse_args()

    d = Path(args.directory)
    if not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")

    written = present = gauss = 0
    for fc in sorted(d.glob('*__predictions.parquet')):
        fam = infer_family(fc.stem)
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
