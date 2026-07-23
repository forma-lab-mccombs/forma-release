#!/usr/bin/env python
"""Assemble a cross-seed mixture pool from existing forecasts (no retraining).

evaluate.py forms a *cross-seed mixture family* (and emits `mixture_nll__*.csv`)
when it discovers >=2 sigma-bearing forecasts that share a `{model}__{feature_set}`
id and differ only by an integer seed token, i.e. files named

    {arm}__{feature_set}__{seed}__{split}__predictions.parquet

The canonical archive (`forecasts/all/`) instead names files per *experiment*
(`{model}-{exp}__{fs}__{split}`), so two seeds of one arm land under DIFFERENT
ids and never group. This script bridges that gap: point it at the seed files of
one arm and it copies them into a pool dir under a shared `arm` token with each
file's seed in the seed slot, so the existing grouping fires with no code change
and no retraining.

Crucially it carries each forecast's `{stem}.nll.json` density sidecar along to
the matching new stem — without it the Laplace arms get mis-scored as Gaussian
(the exact r11 sidecar-drop bug). The seed for each source file is read from the
archive MANIFEST.tsv `seed` column (or an already-seeded filename).

This script is model-agnostic (schema-driven) — Forma and FFNN both work. NOTE the
upstream train.py asymmetry that decides how seeds group natively (this script
overrides the token regardless, so it's only relevant when assembling pools from
raw exp-dir forecasts): FFNN's forecast token is its per-arm config key
(ffnn_current/large/...), so seeds group with just `inference.seed_in_forecast_name:
true`; Forma's token is hardcoded 'forma' for every arm, so a Forma arm also needs
`forecast_name: <arm>` (shared across its seed-configs, distinct across arms) or two
Forma arms in one pool wrongly merge.

Two modes (dry-run by default; pass --apply to write). Each --arm is repeatable;
each value after the arm token is a filename or glob (matched against --src) — list
one per seed or one glob that catches them all.

1. Default (copy) -> a TRUE-mixture pool. Copies each seed under the seed-aware name
   so evaluate.py groups them and emits the exact mixture-of-normals NLL
   (`mixture_nll__*.csv`). Use for the one-off fidelity check.

    python scripts/group_seed_forecasts.py \
        --src  <forecasts>/all \
        --dest <forecasts>/r13_mixture_pool/forecasts \
        --arm forma_gauss_clamp "forma-r11_fp32_r10e8*__pf_full__test__predictions.parquet" \
        --arm forma_laplace     "forma-r11e4?_laplace_fp32_*__pf_full__test__predictions.parquet" \
        --apply

2. --materialize -> the canonical single-model ensemble. Writes ONE moment-matched
   file per arm (prediction = mean of seed means; sigma = law-of-total-variance std),
   a first-class forecast scored under the normal evaluate.py path (R2/MAE exact;
   NLL/CRPS the moment-matched approximation) — drop it into any benchmark pool,
   score repeatedly, pairwise included.

    python scripts/group_seed_forecasts.py \
        --src  <forecasts>/all \
        --dest <forecasts>/paper_current \
        --arm forma "forma-r11_fp32_r10e8*__pf_full__test__predictions.parquet" \
        --materialize --apply
"""
import argparse
import fnmatch
import os
import shutil
import sys
from collections import defaultdict


def _load_manifest_seeds(manifest_path):
    """Return {filename -> seed_str} from a forecasts MANIFEST.tsv (or {} if none)."""
    seeds = {}
    if not manifest_path or not os.path.exists(manifest_path):
        return seeds
    with open(manifest_path, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    if not lines:
        return seeds
    header = lines[0].split('\t')
    try:
        fi, si = header.index('filename'), header.index('seed')
    except ValueError:
        return seeds
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split('\t')
        if len(cols) > max(fi, si):
            seeds[cols[fi]] = cols[si]
    return seeds


def _parse_name(basename):
    """(feature_set, seed, split) from a forecast basename, seed='' if absent.

    Mirrors evaluate.py / archive_forecasts.py length dispatch after stripping
    '__predictions.parquet':
      4 segs -> {model}__{fs}__{seed}__{split}
      3 segs -> {model}__{fs}__{split}
      2 segs -> {model}__{fs}                (split defaults 'test')
    """
    parts = basename.replace('__predictions.parquet', '').split('__')
    if len(parts) == 4:
        return parts[1], parts[2], parts[3]
    if len(parts) == 3:
        return parts[1], '', parts[2]
    if len(parts) == 2:
        return parts[1], '', 'test'
    raise ValueError(
        f"Unrecognized forecast filename shape: {basename!r} "
        f"(expected 2-4 '__'-segments before '__predictions.parquet')")


def _sidecar(name):
    """`{stem}.nll.json` companion name for a `*.parquet` forecast filename."""
    return name[:-len('.parquet')] + '.nll.json' if name.endswith('.parquet') else None


def _resolve_arm(token, patterns, src, manifest_seeds):
    """Resolve one arm spec -> (rows, errors).

    rows: list of dicts {src_name, dst_name, seed, fs, split, has_sidecar}.
    """
    src_files = [f for f in os.listdir(src)
                 if f.endswith('__predictions.parquet')]
    matched = []
    seen = set()
    for pat in patterns:
        hits = fnmatch.filter(src_files, pat) or ([pat] if pat in src_files else [])
        if not hits:
            return [], [f"arm '{token}': no file in {src} matches {pat!r}"]
        for h in hits:
            if h not in seen:
                seen.add(h)
                matched.append(h)

    rows, errors = [], []
    fs_seen, split_seen, seed_seen = set(), set(), {}
    for name in matched:
        fs, name_seed, split = _parse_name(name)
        seed = str(manifest_seeds.get(name, '') or name_seed).strip()
        # Require a non-negative integer: evaluate.py groups on `parts[-3].isdigit()`,
        # which rejects '-5', so accepting a negative seed here would silently emit a
        # name that never groups. Match evaluate's check exactly.
        if not seed.isdigit():
            errors.append(f"arm '{token}': no non-negative integer seed for {name!r} "
                          f"(not in manifest and filename is not seed-aware)")
            continue
        seed = str(int(seed))
        if seed in seed_seen:
            errors.append(f"arm '{token}': duplicate seed {seed} "
                          f"({seed_seen[seed]} and {name}) — seeds must be distinct")
            continue
        seed_seen[seed] = name
        fs_seen.add(fs)
        split_seen.add(split)
        has_sidecar = os.path.exists(os.path.join(src, _sidecar(name)))
        dst_name = f"{token}__{fs}__{seed}__{split}__predictions.parquet"
        rows.append({'src_name': name, 'dst_name': dst_name, 'seed': seed,
                     'fs': fs, 'split': split, 'has_sidecar': has_sidecar})

    if len(fs_seen) > 1:
        errors.append(f"arm '{token}': mixed feature_sets {sorted(fs_seen)} — "
                      f"a mixture family must share one feature_set")
    if len(split_seen) > 1:
        errors.append(f"arm '{token}': mixed splits {sorted(split_seen)}")
    if rows and len(rows) < 2:
        print(f"  WARNING: arm '{token}' has only 1 seed ({rows[0]['seed']}); "
              f"it will NOT form a mixture family (need >=2).")
    return rows, errors


_KEYS = ['firm_id', 'target', 'quarter', 'forecast_horizon']


def _family_of(src, name, has_sidecar):
    """(family, df) for a forecast — from its .nll.json sidecar, else Gaussian.

    Matches evaluate.py's resolver: absent sidecar -> gaussian (the eval default).
    """
    if not has_sidecar:
        return ('gaussian', None)
    import json
    with open(os.path.join(src, _sidecar(name)), encoding='utf-8') as fh:
        meta = json.load(fh)
    return (str(meta.get('family', 'gaussian')).lower(), meta.get('df'))


def _materialize_arm(token, rows, src, dest, apply):
    """Collapse a seed family into ONE moment-matched forecast file (returns dst_name).

    prediction = mean_k(mu_k); sigma = sqrt(mean_k(sigma_k^2) + var_k(mu_k)) — the
    equal-weight mixture's law-of-total-variance projection onto a single density of
    the family (sigma is the predictive std in gaussian AND the r11 laplace
    convention, so the formula is uniform). The result is a standard-schema
    {token}__{fs}__{split}__predictions.parquet (+ matching sidecar iff non-Gaussian):
    a first-class single-model "Forma (seed-ensemble)" forecast that scores under the
    normal evaluate.py path (R2/MAE exact; NLL/CRPS the moment-matched approximation).

    Memory: reads every seed frame resident, so run where evaluate.py runs (a 327M-row
    pf_full pair peaks in the tens of GB). Cells are averaged positionally after a key
    equality check; a key-order mismatch falls back to an index-align (heavier) and a
    differing key SET is a hard error (can't average misaligned cells).
    """
    import json

    import numpy as np
    import pandas as pd

    fam_set = {_family_of(src, r['src_name'], r['has_sidecar']) for r in rows}
    if len({f[0] for f in fam_set}) > 1:
        return None, [f"arm '{token}': mixed densities {sorted({f[0] for f in fam_set})} "
                      f"— can't moment-match across families"]
    family, df = next(iter(fam_set))
    fs, split = rows[0]['fs'], rows[0]['split']
    dst_name = f"{token}__{fs}__{split}__predictions.parquet"
    if not apply:
        return dst_name, []

    frames = []
    for r in sorted(rows, key=lambda x: int(x['seed'])):
        path = os.path.join(src, r['src_name'])
        try:
            f = pd.read_parquet(path, columns=_KEYS + ['prediction', 'sigma'])
        except (ValueError, KeyError):
            f = pd.read_parquet(path, columns=_KEYS + ['prediction'])
            f['sigma'] = np.nan
        frames.append(f)

    base = frames[0]
    n = len(base)
    mus, sigs = [], []
    for f in frames:
        if len(f) != n:
            return None, [f"arm '{token}': seed row-count mismatch ({n} vs {len(f)}) "
                          f"— cannot align cells"]
        aligned = all(f[k].reset_index(drop=True).equals(
            base[k].reset_index(drop=True)) for k in _KEYS)
        if not aligned:
            f = (f.set_index(_KEYS)
                  .reindex(pd.MultiIndex.from_frame(base[_KEYS]))
                  .reset_index())
            if f['prediction'].isna().any():
                return None, [f"arm '{token}': seed key sets differ — cells missing "
                              f"after align; the seeds aren't on the same grid"]
        mus.append(f['prediction'].to_numpy(np.float64))
        sigs.append(f['sigma'].to_numpy(np.float64))

    # Standard forecast schema by construction — mirrors the canonical contract in
    # forecast_io.FORECAST_SCHEMA_COLS / finalize_forecast_frame (the source of truth;
    # keep column order + float32 payload in sync if that changes):
    # firm_id,target,quarter,forecast_horizon,prediction,[sigma],model.
    mu = np.vstack(mus)
    out = base[_KEYS].copy()
    out['prediction'] = mu.mean(axis=0).astype(np.float32)
    sig = np.vstack(sigs)
    if np.isfinite(sig).all():
        # law of total variance: mean of component variances + variance of the means
        var_bar = (sig ** 2).mean(axis=0) + mu.var(axis=0)  # ddof=0 = equal-weight
        out['sigma'] = np.sqrt(var_bar).astype(np.float32)
    else:
        # Any non-finite sigma (incl. a point-only arm with no sigma column) collapses
        # to a point-only forecast — say so rather than silently dropping the density.
        n_bad = int((~np.isfinite(sig)).sum())
        print(f"  NOTE: arm '{token}': sigma dropped -> point-only forecast "
              f"({n_bad:,}/{sig.size:,} non-finite sigma entries across seeds)")
    out['model'] = token

    # Atomic write: a crash mid-write must not leave a truncated parquet that
    # evaluate.py reads as a complete forecast (the guarantee atomic_parquet_writers
    # gives the in-tree writers) — write to .tmp, then os.replace into place.
    os.makedirs(dest, exist_ok=True)
    final = os.path.join(dest, dst_name)
    tmp = final + '.tmp'
    out.to_parquet(tmp, index=False, engine='pyarrow', compression='zstd')
    os.replace(tmp, final)
    if family != 'gaussian':
        side = os.path.join(dest, _sidecar(dst_name))
        side_tmp = side + '.tmp'
        with open(side_tmp, 'w', encoding='utf-8') as fh:
            json.dump({'family': family, 'df': df}, fh)
        os.replace(side_tmp, side)
    return dst_name, []


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True,
                    help='source forecast dir (e.g. forecasts/all)')
    ap.add_argument('--dest', required=True,
                    help='destination pool forecasts dir (created if absent)')
    ap.add_argument('--arm', nargs='+', action='append', metavar='TOKEN_THEN_FILES',
                    required=True,
                    help='shared arm token followed by >=1 filename/glob (in --src); '
                         'repeatable, one per arm')
    ap.add_argument('--manifest', default=None,
                    help='MANIFEST.tsv for seed lookup '
                         '(default: <src>/../MANIFEST.tsv, then <src>/MANIFEST.tsv)')
    ap.add_argument('--materialize', action='store_true',
                    help='instead of copying per-seed files, write ONE moment-matched '
                         'mixture file per arm (prediction=mean mu, sigma=law-of-total-'
                         'variance std): the canonical single-model ensemble forecast, '
                         'scored under the normal evaluate.py path (R2/MAE exact, '
                         'NLL/CRPS moment-matched).')
    ap.add_argument('--apply', action='store_true',
                    help='actually write files (default: dry-run)')
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        ap.error(f"--src not found: {args.src}")

    manifest = args.manifest
    if manifest is None:
        for cand in (os.path.join(os.path.dirname(args.src.rstrip('/\\')), 'MANIFEST.tsv'),
                     os.path.join(args.src, 'MANIFEST.tsv')):
            if os.path.exists(cand):
                manifest = cand
                break
    manifest_seeds = _load_manifest_seeds(manifest)
    print(f"Seed source: {manifest or '(none — relying on seed-aware filenames)'}"
          f"  [{len(manifest_seeds)} rows]")

    # Resolve every arm's seed files up front (shared by both modes).
    arms, all_errors = [], []
    for spec in args.arm:
        if len(spec) < 2:
            all_errors.append(f"--arm needs a token and >=1 file (got {spec})")
            continue
        token, patterns = spec[0], spec[1:]
        rows, errors = _resolve_arm(token, patterns, args.src, manifest_seeds)
        all_errors.extend(errors)
        arms.append((token, rows))

    dst_seen = {}
    if args.materialize:
        # One moment-matched mixture file per arm (the canonical single-model artifact).
        plans = []
        for token, rows in arms:
            if not rows:
                continue
            dst_name, errs = _materialize_arm(token, rows, args.src, args.dest, apply=False)
            all_errors.extend(errs)
            n_sig = sum(1 for r in rows if r['has_sidecar'])
            fam_note = ('Laplace (sidecar)' if n_sig and n_sig == len(rows)
                        else 'Gaussian')
            print(f"\nArm '{token}' -> {dst_name}  ({len(rows)} seed(s), {fam_note})")
            for r in sorted(rows, key=lambda x: int(x['seed'])):
                print(f"    seed {r['seed']:>3}: {r['src_name']}")
            if dst_name in dst_seen:
                all_errors.append(f"destination collision: {dst_name} from arms "
                                  f"'{dst_seen[dst_name]}' and '{token}'")
            dst_seen[dst_name] = token
            plans.append((token, rows))
    else:
        # Per-seed copies under the seed-aware name -> a true-mixture pool for evaluate.py.
        for token, rows in arms:
            model_id = f"{token}__{rows[0]['fs']}" if rows else token
            print(f"\nArm '{token}' -> id '{model_id}'  ({len(rows)} seed(s))")
            for r in sorted(rows, key=lambda x: int(x['seed'])):
                sc = 'with sidecar' if r['has_sidecar'] else 'NO sidecar (-> Gaussian default)'
                print(f"    seed {r['seed']:>3}: {r['src_name']}")
                print(f"             -> {r['dst_name']}  [{sc}]")
                if r['dst_name'] in dst_seen:
                    all_errors.append(f"destination collision: {r['dst_name']} from both "
                                      f"{dst_seen[r['dst_name']]} and {r['src_name']}")
                dst_seen[r['dst_name']] = r['src_name']

    if all_errors:
        print("\nERRORS (nothing written):")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)

    n_arms = sum(1 for _, rows in arms if rows)
    flat_rows = [r for _, rows in arms for r in rows]
    n_side = sum(1 for r in flat_rows if r['has_sidecar'])
    if args.materialize:
        print(f"\n{'APPLY' if args.apply else 'DRY-RUN'}: {n_arms} mixture file(s) "
              f"from {len(flat_rows)} seed(s) -> {args.dest}")
    else:
        print(f"\n{'APPLY' if args.apply else 'DRY-RUN'}: {len(flat_rows)} forecast(s) "
              f"({n_side} sidecar(s)) -> {args.dest}")
    if not args.apply:
        print("  (pass --apply to write)")
        return

    os.makedirs(args.dest, exist_ok=True)
    if args.materialize:
        for token, rows in arms:
            if not rows:
                continue
            dst_name, errs = _materialize_arm(token, rows, args.src, args.dest, apply=True)
            if errs:
                print("\nERRORS during materialize:")
                for e in errs:
                    print(f"  - {e}")
                sys.exit(1)
            print(f"  wrote {dst_name}")
        print(f"Wrote {n_arms} mixture file(s). Drop into any pool and run evaluate.py "
              f"— scores as a single model (R2/MAE/NLL/CRPS), pairwise included.")
    else:
        copied = 0
        for r in flat_rows:
            shutil.copy2(os.path.join(args.src, r['src_name']),
                         os.path.join(args.dest, r['dst_name']))
            copied += 1
            if r['has_sidecar']:
                shutil.copy2(os.path.join(args.src, _sidecar(r['src_name'])),
                             os.path.join(args.dest, _sidecar(r['dst_name'])))
        print(f"Copied {copied} forecast(s) + {n_side} sidecar(s). "
              f"Now run evaluate.py on the pool and read mixture_nll__global.csv.")


if __name__ == '__main__':
    main()
