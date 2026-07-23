"""Mixture-level PIT / coverage / exact-CRPS pass over a seed family.

Publication plan §4.4 (exhibit F1b): per common-sample cell, the K-seed
equal-weight mixture PIT

    u = (1/K) sum_k F_k(y),   F_k = the seed's predictive CDF

drives (a) empirical coverage of nominal central intervals (50/80/90/95%)
by horizon — the "does the 80% interval still cover at h=20" credential —
and (b) a PIT histogram. Two companions from the same streamed pass:

  * total-variance z²:  z² = (y - mu_mix)² / Var_mix,
    Var_mix = mean_k sigma_k² + var_k mu_k  (population var across seeds);
  * the exact mixture CRPS (closed form; see below).

FAMILY AWARENESS (PR: laplace-mixture CRPS)
-------------------------------------------
`sigma` in a forecast file is always the predictive STANDARD DEVIATION; the
predictive DENSITY is a property of the model and travels in the optional
`{stem}.nll.json` sidecar (absent -> gaussian), exactly as in evaluate.py.
This script reads that sidecar and dispatches BOTH the PIT and the CRPS on
it. Getting this wrong is silent: a Gaussian closed form applied to Laplace
(mu, sigma) returns the CRPS of the moment-matched Gaussian mixture, not of
the model's own predictive distribution — which is how T1 Panel C's Laplace
row briefly carried a Gaussian CRPS (0.2986) under an "exact" label.

  gaussian:  F_k = Phi((y-mu_k)/sigma_k)
      CRPS = (1/K) sum_k A(y-mu_k, sigma_k)
           - (1/2K²) sum_k sum_l A(mu_k-mu_l, sqrt(sigma_k²+sigma_l²))
      A(m, s) = m·(2·Phi(m/s) - 1) + 2·s·phi(m/s)          [Grimit et al. 2006]

  laplace:   b_k = sigma_k / sqrt(2) is the Laplace SCALE (std = sqrt(2)·b)
      F_k = Laplace CDF; and via CRPS(F,y) = E|X-y| - ½·E|X-X'|:
      CRPS = (1/K) sum_k [ M_k + b_k·exp(-M_k/b_k) ],  M_k = |y-mu_k|
           - (1/2K²) sum_k sum_l E|L_k - L_l|
      The difference of two independent Laplaces has characteristic function
      1/((1+b_k²t²)(1+b_l²t²)) — a SIGNED mixture of Laplace(0,b_k) and
      Laplace(0,b_l) with weights b_k²/(b_k²-b_l²) and b_l²/(b_l²-b_k²) —
      so with m = mu_k-mu_l, M = |m|, and E|Laplace(0,b) - m| = M + b·e^(-M/b):
          E|L_k - L_l| = M + (b_k³·e^(-M/b_k) - b_l³·e^(-M/b_l))/(b_k² - b_l²)
      That raw form is 0/0 at b_k = b_l and catastrophically cancels near it,
      so `_lap_E_abs_diff` evaluates the algebraically equivalent single-
      exponential rearrangement (see its docstring), exact at b_k = b_l.
      Both reduce, at K=1, to evaluate.py's per-seed b·(z + e^-z - 3/4).

  student_t: no closed form wired (evaluate.py returns NaN) -> hard error.

z² is family-agnostic: sigma is the std in every family, so Var_mix is too.

Everything is computable from the per-seed eval caches alone (residual_k =
mu_k - y, so y - mu_k = -residual_k and mu_k - mu_l = residual_k -
residual_l): no truth join, one stream over (target, horizon) blocks.

Sample discipline: cells where the actual, the change-space baseline, and
ALL K seed residuals+sigmas are finite — the fgrid family's own footprint,
which is the binding constraint of the r13 likelihood pool (expected count
327,244,429; printed for verification).

Usage:
  python scripts/mixture_calibration.py \
      --exp_dir <forecasts>/r13_lik_pool \
      --family forma_fgrid --seeds 60,61,62,63,64 \
      --out results/icaif26_panels/full_sample_likelihood

Output: coverage_by_horizon.csv (+ pooled row: n, coverage at each nominal
level, mean PIT, mixture z², mixture CRPS) and pit_hist_by_horizon.csv
(20-bin normalized PIT histogram per horizon + pooled).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from forma.scoring.eval_cache import (  # noqa: E402
    TruthGrid,
    load_error_cache,
    file_fingerprint,
    PARQUET_ENGINE,
)
from forma.scoring.evaluate import _read_nll_meta  # noqa: E402

NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)
PIT_BINS = 20
POOLED_KEY = 0  # accumulator key for the all-horizon row
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
_SQRT2 = np.sqrt(2.0)
SUPPORTED_FAMILIES = ('gaussian', 'laplace')


def _phi(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _crps_A(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Grimit et al. (2006) helper: E|X| for X ~ N(m, s²) (even in m)."""
    z = m / s
    return m * (2.0 * ndtr(z) - 1.0) + 2.0 * s * _phi(z)


def _lap_E_abs(m: np.ndarray, b: np.ndarray) -> np.ndarray:
    """E|X| for X ~ Laplace(m, b), i.e. E|Laplace(0,b) - m| (even in m).

    = |m| + b·exp(-|m|/b).  At m=0 this is b, the Laplace mean absolute
    deviation; as |m|->inf it tends to |m|.
    """
    M = np.abs(m)
    return M + b * np.exp(-M / b)


def _lap_S(t: np.ndarray) -> np.ndarray:
    """S(t) = (1 - exp(-t))/t on t >= 0, with S(0) = 1 and S(inf) = 0.

    Bounded in (0, 1]; `expm1` keeps it accurate as t -> 0.
    """
    out = np.ones_like(t)
    nz = t > 0.0
    out[nz] = -np.expm1(-t[nz]) / t[nz]
    return out


def _lap_E_abs_diff(m: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """E|L1 - L2| for INDEPENDENT L1~Laplace(mu1,b1), L2~Laplace(mu2,b2), m = mu1-mu2.

    The partial-fraction result (see module docstring)

        M + (a³·e^(-M/a) - c³·e^(-M/c)) / (a² - c²),   M = |m|

    is exact but numerically unusable: it is 0/0 at a = c (the k=l diagonal
    and any two seeds that happen to agree to machine precision) and loses
    most of its significant digits whenever a ≈ c. Split the numerator as
    e^(-M/a)(a³-c³) + c³(e^(-M/a) - e^(-M/c)) and fold the second difference
    through expm1 — with a := max(b1,b2), c := min(b1,b2) so t >= 0 and the
    surviving exponential is the LARGER one — to get the equivalent

        M + e^(-M/a) · [ a·(a² + a·c + c²) + c²·M·S(t) ] / (a·(a + c)),
        t = M·(a - c)/(a·c),   S(t) = (1 - e^(-t))/t.

    One exponential, every other factor bounded: no cancellation, no
    overflow. Checks: a = c gives M + e^(-M/a)(3a + M)/2 (so m = 0, a = c
    gives 3a/2, the classic E|L - L'|); c -> 0 gives M + a·e^(-M/a) = the
    point-mass limit `_lap_E_abs`; M = 0 gives (a²+ac+c²)/(a+c).
    """
    M = np.abs(m)
    a = np.maximum(b1, b2)
    c = np.minimum(b1, b2)
    t = M * (a - c) / (a * c)
    return M + np.exp(-M / a) * (a * (a * a + a * c + c * c) + c * c * M * _lap_S(t)) / (a * (a + c))


def _mixture_pit(res: np.ndarray, sig: np.ndarray, family: str) -> np.ndarray:
    """u = (1/K)·sum_k F_k(y).  res = mu - y, so (y - mu_k)/scale = -res_k/scale."""
    if family == 'gaussian':
        return ndtr(-res / sig).mean(axis=0)
    if family == 'laplace':
        # F(y) = ½·exp(x) for x<=0 else 1 - ½·exp(-x), x = (y-mu)/b. Both
        # branches evaluate exp of a non-positive number -> no overflow.
        x = -res / (sig / _SQRT2)
        e = np.exp(-np.abs(x))
        return np.where(x <= 0.0, 0.5 * e, 1.0 - 0.5 * e).mean(axis=0)
    raise ValueError(f"unsupported family {family!r}")


def _mixture_crps(res: np.ndarray, sig: np.ndarray, family: str) -> np.ndarray:
    """Exact CRPS of the equal-weight K-seed mixture, per cell.

    CRPS(F, y) = E|X - y| - ½·E|X - X'| with X, X' iid ~ F. For the
    equal-weight mixture both expectations decompose over seed pairs, giving
    the (1/K)·sum_k ... - (1/2K²)·sum_k sum_l ... shape below in either family.

    res[k] = mu_k - y  (so y - mu_k = -res[k], mu_k - mu_l = res[k] - res[l]);
    sig[k] = predictive std. Both helper kernels are even in their first
    argument, so residual signs need no special handling.
    """
    K = res.shape[0]
    if family == 'gaussian':
        crps = _crps_A(res, sig).mean(axis=0)
        cross = np.zeros(res.shape[1], dtype=np.float64)
        for k in range(K):
            cross += _crps_A(np.zeros_like(sig[k]), _SQRT2 * sig[k])
            for l in range(k + 1, K):
                cross += 2.0 * _crps_A(res[k] - res[l],
                                       np.sqrt(sig[k] ** 2 + sig[l] ** 2))
    elif family == 'laplace':
        b = sig / _SQRT2
        crps = _lap_E_abs(res, b).mean(axis=0)
        cross = np.zeros(res.shape[1], dtype=np.float64)
        for k in range(K):
            cross += 1.5 * b[k]          # E|L_k - L_k'| = 3·b_k/2 (indep copies)
            for l in range(k + 1, K):
                cross += 2.0 * _lap_E_abs_diff(res[k] - res[l], b[k], b[l])
    else:
        raise ValueError(f"unsupported family {family!r}")
    return crps - 0.5 * cross / (K * K)


def _resolve_family(paths) -> str:
    """Read the density family from the per-seed `.nll.json` sidecars.

    All seeds of one family must agree — a mixture over heterogeneous
    densities is not what any of the closed forms above describe. Raises
    ValueError (not SystemExit) so the check is reusable as a library call;
    main() converts it to a clean CLI exit.
    """
    fams = {}
    for p in paths:
        fam, df = _read_nll_meta(p)
        if fam == 'student_t':
            raise ValueError(
                f"{p.name}: family 'student_t' (df={df}) has no closed-form mixture "
                f"CRPS wired (evaluate.py returns NaN per-entry). Refusing to score "
                f"rather than silently applying a Gaussian form.")
        if fam not in SUPPORTED_FAMILIES:
            raise ValueError(f"{p.name}: unknown density family {fam!r}")
        fams.setdefault(fam, []).append(p.name)
    if len(fams) != 1:
        raise ValueError(f"seeds disagree on predictive density family: {fams}")
    return next(iter(fams))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--exp_dir', required=True,
                    help="pool dir holding <exp_dir>/forecasts + config.yaml (per-seed files)")
    ap.add_argument('--family', default='forma_fgrid')
    ap.add_argument('--feature_set', default='pf_full')
    ap.add_argument('--seeds', default='60,61,62,63,64')
    ap.add_argument('--split', default='test')
    ap.add_argument('--mask_families', default='',
                    help="comma list of ADDITIONAL seed families whose finite-"
                         "RESIDUAL footprint is intersected into the mask "
                         "(their residual values and sigmas are never scored; "
                         "sigma finiteness is not checked — sufficient when the "
                         "mask family's sigma is finite wherever its residual "
                         "is, as verified for forma_fgrid). Needed for "
                         "non-binding families (e.g. FFNN) so their numbers "
                         "share the pool's common sample: the binding family's "
                         "footprint IS the common sample (structural, verified "
                         "for forma_fgrid), so pass --mask_families forma_fgrid "
                         "when scoring FFNN.")
    ap.add_argument('--out', required=True, help="output directory for the CSVs")
    ap.add_argument('--processed_dir', default='data/processed',
                    help="dir holding the tabular_<split>__<fs>__<tag>.parquet truth")
    ap.add_argument('--cache-dir', dest='cache_dir', default=None,
                    help="override root for the error caches (default: "
                         "<exp_dir>/forecasts/eval_cache), matching evaluate.py")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    K = len(seeds)

    dataset_tag = None
    cfg_path = exp_dir / 'config.yaml'
    if cfg_path.exists():
        import yaml
        with open(cfg_path, 'r', encoding='utf-8-sig') as f:
            run_cfg = yaml.safe_load(f) or {}
        dataset_tag = (run_cfg.get('data') or {}).get('dataset_tag')

    from forma.scoring.evaluate import resolve_ground_truth_path  # noqa: E402
    truth_path = resolve_ground_truth_path(Path(args.processed_dir), args.feature_set,
                                           args.split, dataset_tag=dataset_tag)
    print(f"Loading truth: {truth_path}", flush=True)
    truth_df = pd.read_parquet(truth_path, engine=PARQUET_ENGINE)
    grid = TruthGrid(truth_df, fingerprint=file_fingerprint(truth_path))

    seed_paths = [exp_dir / 'forecasts' / (
        f"{args.family}__{args.feature_set}__{s}__{args.split}__predictions.parquet")
        for s in seeds]
    try:
        family = _resolve_family(seed_paths)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"Predictive density family: {family} (from .nll.json sidecars)", flush=True)

    caches = []
    for s, fp in zip(seeds, seed_paths):
        cache = load_error_cache(fp, grid, cache_root=args.cache_dir)
        if cache is None or cache['sigma'] is None:
            raise SystemExit(
                f"no valid (residual, sigma) cache for {fp.name} — run evaluate.py on "
                f"{exp_dir} first so the caches exist and match the current truth")
        caches.append(cache)
        print(f"  cache hit: seed {s}", flush=True)

    mask_caches = []
    for fam in [f.strip() for f in args.mask_families.split(',') if f.strip()]:
        for s in seeds:
            fp = exp_dir / 'forecasts' / (
                f"{fam}__{args.feature_set}__{s}__{args.split}__predictions.parquet")
            cache = load_error_cache(fp, grid, cache_root=args.cache_dir)
            if cache is None:
                raise SystemExit(f"no valid cache for mask family file {fp.name}")
            mask_caches.append(cache)
        print(f"  mask family loaded: {fam} ({len(seeds)} seeds)", flush=True)

    horizons = list(grid.horizons)
    # POOLED_KEY=0 shares the accumulator keyspace with horizons (dm_tests.py
    # convention) — an h=0 truth column would silently collide with pooled.
    assert POOLED_KEY not in horizons, "grid has horizon 0; pooled key collides"
    # NOTE: sigma is used RAW (no floor). Safe for Forma-family caches (min
    # fgrid sigma ~0.014) but would diverge from evaluate.py's --sigma-floor
    # scoring on degenerate-sigma models (e.g. chronos_raw) — don't point this
    # script at those without adding a floor.
    keys = horizons + [POOLED_KEY]
    acc = {k: {'n': 0,
               'cov': np.zeros(len(NOMINAL_LEVELS), dtype=np.int64),
               'pit_hist': np.zeros(PIT_BINS, dtype=np.int64),
               'pit_sum': 0.0,
               'z2_sum': 0.0,
               'crps_sum': 0.0}
           for k in keys}
    lo_hi = [((1.0 - p) / 2.0, (1.0 + p) / 2.0) for p in NOMINAL_LEVELS]

    n_blocks = 0
    t0 = time.time()
    print(f"Streaming {len(grid.targets)} targets x {len(horizons)} horizons, K={K}...",
          flush=True)
    for target, horizon, sl, act, base in grid.iter_blocks():
        mask = np.isfinite(act) & np.isfinite(base)
        for mc in mask_caches:
            mask &= np.isfinite(np.asarray(mc['residual'][sl]))
        res_l, sig_l = [], []
        for c in caches:
            r = np.asarray(c['residual'][sl])
            s = np.asarray(c['sigma'][sl])
            mask &= np.isfinite(r) & np.isfinite(s) & (s > 0)
            res_l.append(r)
            sig_l.append(s)
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        res = np.stack([r[idx].astype(np.float64) for r in res_l])   # (K, n)
        sig = np.stack([s[idx].astype(np.float64) for s in sig_l])   # (K, n)

        # PIT of the K-seed mixture, under the model's own predictive density
        u = _mixture_pit(res, sig, family)

        # total-variance z² (family-agnostic: sigma is the std in every family)
        m_mix = res.mean(axis=0)                       # mu_mix - y
        var_mix = (sig ** 2).mean(axis=0) + res.var(axis=0)
        z2 = (m_mix ** 2) / var_mix

        # exact mixture CRPS, in the model's own family
        crps = _mixture_crps(res, sig, family)

        hist = np.bincount(np.minimum((u * PIT_BINS).astype(np.int64), PIT_BINS - 1),
                           minlength=PIT_BINS)
        cov = np.array([np.count_nonzero((u >= lo) & (u <= hi)) for lo, hi in lo_hi],
                       dtype=np.int64)
        for key in (horizon, POOLED_KEY):
            a = acc[key]
            a['n'] += idx.size
            a['cov'] += cov
            a['pit_hist'] += hist
            a['pit_sum'] += float(u.sum())
            a['z2_sum'] += float(z2.sum())
            a['crps_sum'] += float(crps.sum())

        n_blocks += 1
        if n_blocks % 100 == 0:
            print(f"  {n_blocks} blocks done ({time.time() - t0:.0f}s), "
                  f"pooled n so far = {acc[POOLED_KEY]['n']:,}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, hist_rows = [], []
    for key in keys:
        a = acc[key]
        if a['n'] == 0:
            continue
        label = 'pooled' if key == POOLED_KEY else key
        # `family` makes the CSV self-describing: PIT/coverage/CRPS are all
        # computed under this density, so a reader never has to guess.
        row = {'horizon': label, 'n_cells': a['n'], 'family': family,
               'mean_pit': a['pit_sum'] / a['n'],
               'z2_mixture': a['z2_sum'] / a['n'],
               'crps_mixture': a['crps_sum'] / a['n']}
        for p, c in zip(NOMINAL_LEVELS, a['cov']):
            row[f"cov_{int(p * 100)}"] = c / a['n']
        rows.append(row)
        hist_rows.append({'horizon': label,
                          **{f"bin_{i}": a['pit_hist'][i] / a['n'] * PIT_BINS
                             for i in range(PIT_BINS)}})

    cov_df = pd.DataFrame(rows)
    cov_path = out_dir / 'coverage_by_horizon.csv'
    cov_df.to_csv(cov_path, index=False, encoding='utf-8-sig')
    pd.DataFrame(hist_rows).to_csv(out_dir / 'pit_hist_by_horizon.csv',
                                   index=False, encoding='utf-8-sig')
    print(f"\nWrote {cov_path} ({time.time() - t0:.0f}s total)")
    print(cov_df.to_string(index=False))
    pooled = cov_df[cov_df['horizon'] == 'pooled'].iloc[0]
    print(f"\nPOOLED SUMMARY  family={family}  n={int(pooled['n_cells']):,}  "
          f"cov50={pooled['cov_50']:.3f} cov80={pooled['cov_80']:.3f} "
          f"cov90={pooled['cov_90']:.3f} cov95={pooled['cov_95']:.3f}  "
          f"z2_mix={pooled['z2_mixture']:.3f}  crps_mix={pooled['crps_mixture']:.4f}")


if __name__ == '__main__':
    main()
