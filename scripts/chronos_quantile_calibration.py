"""Quantile-based PIT / coverage / CRPS pass for chronos_raw (issue #246).

T1 Panel C carries mixture-PIT coverage columns for every density model, but
Chronos-2's predictive travels as 21 native quantile levels, not (mu, sigma):
``scripts/mixture_calibration.py`` is parametric and cannot score it, which is
why its two coverage cells were placeholders and its CRPS cell a *Gaussian
surrogate* (the Gaussian closed form applied to chronos's moment-matched
(mu, sigma) — the same error class as the Laplace-row incident documented in
mixture_calibration.py). This script scores chronos's own predictive from the
``...__quantiles.parquet`` sibling written by ``ChronosRawTSModel``
(``save_quantiles: true``):

  * empirical coverage of the nominal central intervals whose endpoints are
    NATIVE grid levels — 50% (0.25/0.75), 80% (0.10/0.90), 90% (0.05/0.95).
    95% needs 0.025/0.975, which are off-grid, so no cov_95 column is written
    (Panel C reports 50/90 only). Intervals are closed: a truth value exactly
    on a knot counts as covered.
  * PIT under the piecewise-linear CDF through the knots (ties get the
    mid-CDF value; beyond the outer knots the PIT is clamped to 0.01/0.99).
  * CRPS of the piecewise-linear quantile function, EXACT: the pinball
    integral CRPS(F, y) = int_0^1 2 rho_tau(y - Q(tau)) d tau is evaluated in
    closed form per linear segment, with Q extended FLAT into the [0, 0.01]
    and [0.99, 1] tails (equivalently: F jumps to 0/1 at the outer knots).
    Benchmarked in issue #246: ~0.35% population error vs a known parametric
    truth, versus 4.05% for the naive GluonTS-style mean-pinball (which
    assumes a uniform tau grid — ours is not: 0.04 spacing at the tails,
    0.05 interior). Do NOT replace this with 2 x mean(pinball).
  * the Gaussian-surrogate CRPS on the SAME cells, for the record: the
    Gaussian closed form on the forecast file's (residual, sigma) with the
    same --sigma-floor policy evaluate.py used for the published number.

Sample discipline: the r13 likelihood pool's common sample is the masked-seed
family's own footprint — cells where the actual, the change-space baseline,
and ALL K mask-family seeds' residuals+sigmas are finite (sigma > 0), exactly
mixture_calibration.py's binding-family mask (expected count 327,244,429 for
forma_fgrid / pf_full; printed for verification). Chronos cells outside that
mask are ignored; mask cells the quantile file does NOT cover are counted and
reported loudly — a nonzero count means the "identical sample" claim fails.

Coverage/CRPS here are floor-insensitive by construction (no sigma in the
quantile path): the ~0.6% degenerate zero-width chronos cells score as
CRPS = |y - q| and (unless y lands exactly on the point mass) as
non-coverage — the honest severe-under-coverage result, no NLL-style blow-up.

Usage:
  python scripts/chronos_quantile_calibration.py \
      --quantiles <dir>/chronos_raw__pf_full__test__quantiles.parquet \
      --exp_dir <forecasts>/r13_lik_pool \
      --mask_family forma_fgrid --mask_seeds 60,61,62,63,64 \
      --out results/icaif26_panels/full_sample_likelihood/chronos_raw_calibration

Output: coverage_by_horizon.csv (+ pooled row: n, cov_50/80/90, mean PIT,
quantile CRPS, Gaussian-surrogate CRPS) and pit_hist_by_horizon.csv, in the
mixture_calibration.py layout so downstream readers share one code path; plus
saturation_split.csv — the pooled scores split into 'saturated' cells
(zero-width knot span OR truth at the ±clip rail, where the clipped-support
predictive can post CRPS 0 / coverage 1 in ways an unbounded-support mixture
cannot) vs 'interior' cells. The split disentangles the two mechanisms behind
the quantile-vs-surrogate CRPS gap (clip-space point masses vs the
surrogate's pre-clip sigma inflation): if the interior-only quantile CRPS
still beats the interior-only surrogate, the gap is not a rail artifact.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SRC = _REPO_ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from forma.scoring.eval_cache import (  # noqa: E402
    TruthGrid,
    load_error_cache,
    file_fingerprint,
    PARQUET_ENGINE,
)
from forma.scoring.evaluate import _crps_per_entry  # noqa: E402
from forma.models.chronos_ts import ChronosRawTSModel  # noqa: E402  (_mean_weights only)

PIT_BINS = 20
POOLED_KEY = 0  # accumulator key for the all-horizon row (mixture_calibration convention)
# Nominal central intervals whose endpoints are native 21-grid levels.
NATIVE_INTERVALS = ((0.50, 0.25, 0.75), (0.80, 0.10, 0.90), (0.90, 0.05, 0.95))


def parse_quantile_columns(columns) -> "list[tuple[str, float]]":
    """[(column, level)] for the q_* columns, sorted by level.

    Inverse of models.chronos_ts.quantile_col ('q_{level:g}'). Raises on a
    malformed q_* column rather than silently dropping a level.
    """
    out = []
    for c in columns:
        if not c.startswith("q_"):
            continue
        try:
            out.append((c, float(c[2:])))
        except ValueError:
            raise ValueError(f"malformed quantile column name {c!r}")
    out.sort(key=lambda kv: kv[1])
    if not out:
        raise ValueError("no q_* quantile columns found")
    levels = [lv for _c, lv in out]
    if any(not (0.0 < lv < 1.0) for lv in levels):
        raise ValueError(f"quantile levels outside (0,1): {levels}")
    return out


def piecewise_linear_crps(q: np.ndarray, taus: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact CRPS of the piecewise-linear quantile function, flat beyond the outer knots.

    CRPS(F, y) = int_0^1 2 rho_tau(y - Q(tau)) d tau, rho_tau(u) = u (tau - 1{u<0}),
    with Q(tau) linear between the knots (taus[j], q[:, j]) and constant on
    [0, taus[0]] and [taus[-1], 1]. The integrand is piecewise quadratic in tau,
    so each of the (n_knots + 1) segments integrates in closed form; segments
    where y - Q changes sign are split at the crossing. Zero-width (tied) knots
    are exact degenerate cases (m = 0 segments), and a fully degenerate row
    (all knots equal c) reduces to CRPS = |y - c| exactly.

    Parameters: q (n, K) NONDECREASING knot values per row, taus (K,) strictly
    increasing levels in (0,1), y (n,). Returns (n,) float64.
    """
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    n, K = q.shape

    # Segment endpoints in tau: [0, t0], [t0, t1], ..., [t_{K-1}, 1].
    a = np.concatenate(([0.0], taus))                    # (K+1,)
    b = np.concatenate((taus, [1.0]))                    # (K+1,)
    qa = np.concatenate((q[:, :1], q), axis=1)           # (n, K+1) Q at segment start
    qb = np.concatenate((q, q[:, -1:]), axis=1)          # (n, K+1) Q at segment end

    width = (b - a)[None, :]                             # (1, K+1) > 0
    m = (qb - qa) / width                                # slope of Q on each segment
    # d(tau) = y - Q(tau) = alpha + beta * tau on the segment.
    beta = -m
    alpha = (y[:, None] - qa) + m * a[None, :]
    da = alpha + beta * a[None, :]                       # d at segment start
    db = alpha + beta * b[None, :]                       # d at segment end

    def P(t):
        """Antiderivative of 2 d(tau) tau (the d >= 0 branch)."""
        return alpha * t * t + (2.0 / 3.0) * beta * t * t * t

    def N(t):
        """Antiderivative of 2 d(tau) (tau - 1) (the d < 0 branch)."""
        return P(t) - t * (2.0 * alpha + beta * t)

    ta = np.broadcast_to(a[None, :], alpha.shape)
    tb = np.broadcast_to(b[None, :], alpha.shape)
    pos_a = da >= 0.0
    pos_b = db >= 0.0

    # Sign-crossing segments split at tau* = -alpha/beta (beta != 0 whenever
    # the sign changes; guard the division anyway and fall back to no split).
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = np.where(beta != 0.0, -alpha / np.where(beta != 0.0, beta, 1.0), ta)
    t_star = np.clip(t_star, ta, tb)

    same_sign = pos_a == pos_b
    seg = np.where(
        same_sign,
        np.where(pos_a, P(tb) - P(ta), N(tb) - N(ta)),
        np.where(
            pos_a,  # d >= 0 on [a, tau*], d < 0 on [tau*, b]
            (P(t_star) - P(ta)) + (N(tb) - N(t_star)),
            (N(t_star) - N(ta)) + (P(tb) - P(t_star)),
        ),
    )
    return seg.sum(axis=1)


def quantile_pit(q: np.ndarray, taus: np.ndarray, y: np.ndarray) -> np.ndarray:
    """PIT under the piecewise-linear CDF through the knots.

    Linear interpolation of tau in y between knots; y beyond the outer knots
    clamps to taus[0]/taus[-1] (the CDF value there is unidentified below
    0.01 / above 0.99, and the clamp keeps the histogram's edge bins honest
    about tail exceedances without inventing tail shape). Ties (y inside a
    zero-width knot run) get the mid-CDF value — the median of the levels
    collapsed onto that point — so a fully degenerate row scores 0.5, not an
    artifact of interpolator tie-breaking.
    """
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, K = q.shape
    yc = y[:, None]
    i_lo = (q < yc).sum(axis=1)    # first knot index >= y
    i_hi = (q <= yc).sum(axis=1)   # first knot index > y

    u = np.empty(len(y), dtype=np.float64)

    tied = i_hi > i_lo             # y sits exactly on >= 1 knot
    if tied.any():
        u[tied] = 0.5 * (taus[i_lo[tied]] + taus[i_hi[tied] - 1])

    free = ~tied
    k = i_lo[free]
    below = k == 0
    above = k == K
    interior = ~(below | above)
    uf = np.empty(free.sum(), dtype=np.float64)
    uf[below] = taus[0]
    uf[above] = taus[-1]
    if interior.any():
        ki = k[interior]
        rows = np.flatnonzero(free)[interior]
        q_lo = q[rows, ki - 1]
        q_hi = q[rows, ki]
        frac = (y[rows] - q_lo) / (q_hi - q_lo)   # q_hi > q_lo (no tie here)
        uf[interior] = taus[ki - 1] + (taus[ki] - taus[ki - 1]) * frac
    u[free] = uf
    return u


def build_common_mask(grid: TruthGrid, mask_caches) -> np.ndarray:
    """The binding family's common-sample footprint as a flat n_cells bool array.

    Exactly mixture_calibration.py's per-block mask: finite actual, finite
    change-space baseline, and finite residual + finite positive sigma for
    every mask-family seed.
    """
    mask = np.zeros(grid.n_cells, dtype=bool)
    for _target, _h, sl, act, base in grid.iter_blocks():
        m = np.isfinite(act) & np.isfinite(base)
        for c in mask_caches:
            r = np.asarray(c["residual"][sl])
            s = np.asarray(c["sigma"][sl])
            m &= np.isfinite(r) & np.isfinite(s) & (s > 0)
        mask[sl] = m
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--quantiles', required=True,
                    help="path to the ...__quantiles.parquet sibling written by "
                         "ChronosRawTSModel with save_quantiles: true")
    ap.add_argument('--exp_dir', required=True,
                    help="pool dir holding <exp_dir>/forecasts + config.yaml "
                         "(mask-family seed files + the chronos predictions file)")
    ap.add_argument('--forecasts_dir', default=None,
                    help="override the forecasts directory (default: "
                         "<exp_dir>/forecasts). Lets the Box paper_current "
                         "mirror be scored directly — it is a flat dir, not a "
                         "pool — while --exp_dir still supplies config.yaml.")
    ap.add_argument('--model', default='chronos_raw',
                    help="model name of the predictions file scored for the "
                         "Gaussian-surrogate column. Must be a MEAN-point file "
                         "(point_stat: 'mean'): the same-run gate reproduces "
                         "prediction as the trapezoid quantile integral, so a "
                         "median-point file (e.g. chronos_raw_med) trips the "
                         "gate with a misleading mixed-provenance diagnosis.")
    ap.add_argument('--mask_family', default='forma_fgrid',
                    help="seed family whose footprint IS the common sample")
    ap.add_argument('--mask_seeds', default='60,61,62,63,64')
    ap.add_argument('--feature_set', default='pf_full')
    ap.add_argument('--split', default='test')
    ap.add_argument('--sigma-floor', dest='sigma_floor', type=float, default=1e-4,
                    help="floor for the Gaussian-surrogate column only (match "
                         "evaluate.py's published policy; default 1e-4). The "
                         "quantile-based columns never touch sigma.")
    ap.add_argument('--out', required=True, help="output directory for the CSVs")
    ap.add_argument('--processed_dir', default='data/processed',
                    help="dir holding the tabular_<split>__<fs>__<tag>.parquet truth")
    ap.add_argument('--cache-dir', dest='cache_dir', default=None,
                    help="override root for the error caches (default: "
                         "<exp_dir>/forecasts/eval_cache), matching evaluate.py")
    ap.add_argument('--expected_n', type=int, default=None,
                    help="assert the pooled scored cell count equals this "
                         "(e.g. 327244429 for the r13 pf_full likelihood pool)")
    ap.add_argument('--allow-uncovered', dest='allow_uncovered', action='store_true',
                    help="write CSVs even when some common-sample cells have no "
                         "quantile row (default: hard-fail — a partial sample "
                         "breaks the identical-sample claim)")
    ap.add_argument('--clip-rail', dest='clip_rail', type=float, default=6.0,
                    help="|truth| at-or-beyond this value counts as rail-saturated "
                         "in saturation_split.csv. Must match the build's label "
                         "clip (configs' max_abs_zscore; 6.0 for every r13 build).")
    args = ap.parse_args()

    t0 = time.time()
    exp_dir = Path(args.exp_dir)
    q_path = Path(args.quantiles)
    seeds = [s.strip() for s in args.mask_seeds.split(',') if s.strip()]

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

    fdir = Path(args.forecasts_dir) if args.forecasts_dir else exp_dir / 'forecasts'
    mask_caches = []
    for s in seeds:
        fp = fdir / (
            f"{args.mask_family}__{args.feature_set}__{s}__{args.split}__predictions.parquet")
        cache = load_error_cache(fp, grid, cache_root=args.cache_dir)
        if cache is None or cache['sigma'] is None:
            raise SystemExit(
                f"no valid (residual, sigma) cache for {fp.name} — run evaluate.py on "
                f"{exp_dir} first so the caches exist and match the current truth")
        mask_caches.append(cache)
        print(f"  mask cache hit: {args.mask_family} seed {s}", flush=True)

    surro_fp = fdir / (
        f"{args.model}__{args.feature_set}__{args.split}__predictions.parquet")
    surro = load_error_cache(surro_fp, grid, cache_root=args.cache_dir)
    if surro is None or surro['sigma'] is None:
        raise SystemExit(
            f"no valid (residual, sigma) cache for {surro_fp.name} — run evaluate.py "
            f"first (needed for the Gaussian-surrogate CRPS column)")
    print(f"  surrogate cache hit: {args.model}", flush=True)

    print("Building the common-sample mask...", flush=True)
    mask = build_common_mask(grid, mask_caches)
    n_mask = int(mask.sum())
    print(f"  common sample: {n_mask:,} cells "
          f"({args.mask_family} footprint x finite actual/baseline)", flush=True)

    horizons = list(grid.horizons)
    assert POOLED_KEY not in horizons, "grid has horizon 0; pooled key collides"
    keys = horizons + [POOLED_KEY]
    acc = {k: {'n': 0,
               'cov': np.zeros(len(NATIVE_INTERVALS), dtype=np.int64),
               'pit_hist': np.zeros(PIT_BINS, dtype=np.int64),
               'pit_sum': 0.0,
               'crps_q_sum': 0.0,
               'crps_g_sum': 0.0,
               'n_g': 0}       # finite surrogate entries (evaluate.py convention)
           for k in keys}
    # Saturation decomposition (pooled only): 'saturated' = zero-width knot
    # span OR truth exactly at the ±clip rail; 'interior' = everything else.
    sat_acc = {g: {'n': 0, 'cov': np.zeros(len(NATIVE_INTERVALS), dtype=np.int64),
                   'crps_q_sum': 0.0, 'crps_g_sum': 0.0, 'n_g': 0}
               for g in ('saturated', 'interior')}
    n_degen = n_rail = 0
    clip_rail = args.clip_rail  # label clip (max_abs_zscore); truth saturates exactly here

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(q_path))
    qcols_lv = parse_quantile_columns(pf.schema_arrow.names)
    qcols = [c for c, _lv in qcols_lv]
    taus = np.array([lv for _c, lv in qcols_lv], dtype=np.float64)
    print(f"Quantile file: {q_path.name} — {len(qcols)} levels "
          f"[{taus[0]:g} .. {taus[-1]:g}], {pf.metadata.num_rows:,} rows", flush=True)
    iv_idx = []
    for nominal, lo, hi in NATIVE_INTERVALS:
        try:
            iv_idx.append((nominal, int(np.flatnonzero(np.isclose(taus, lo))[0]),
                           int(np.flatnonzero(np.isclose(taus, hi))[0])))
        except IndexError:
            raise SystemExit(f"interval {nominal:.0%} needs levels {lo}/{hi}, "
                             f"not on the file's grid {taus.tolist()}")

    # Trapezoid weights over the file's own levels: used to re-derive the
    # predictive mean from the persisted quantiles and cross-check it against
    # the predictions file (prediction = residual + actual). A max deviation
    # beyond float32 round-off means the quantile sibling and the predictions
    # file are NOT from the same run -- refuse to publish mixed provenance.
    grid_w = ChronosRawTSModel._mean_weights(taus.tolist())
    MEAN_DEV_TOL = 1e-3
    mean_dev = 0.0

    key_cols = ['firm_id', 'quarter', 'target', 'forecast_horizon']
    n_rows = n_dropped = n_crossed = 0
    n_chunks = 0
    for batch in pf.iter_batches(batch_size=2_000_000, columns=key_cols + qcols):
        chunk = batch.to_pandas()
        n_rows += len(chunk)
        n_chunks += 1
        row_id, valid = grid.map_forecast_rows(chunk)
        take = valid & mask[np.where(valid, row_id, 0)]
        n_dropped += int((~take).sum())
        if not take.any():
            continue
        rid = row_id[take]
        qmat = chunk.loc[take, qcols].to_numpy(np.float64)
        # Dedup keep='first' across the whole file, matching the error caches:
        # within-chunk via duplicated(), cross-chunk via the consumed mask bits
        # (a cell already scored has mask False and never passes `take` again).
        dup = pd.Series(rid).duplicated().to_numpy()
        if dup.any():
            n_dropped += int(dup.sum())
            rid = rid[~dup]
            qmat = qmat[~dup]
        bad = ~np.isfinite(qmat).all(axis=1)
        if bad.any():
            # A NaN knot would silently score as non-coverage and PIT 0.01;
            # and a common-sample cell that cannot be scored breaks the
            # identical-sample claim either way. Refuse to continue.
            raise SystemExit(
                f"{int(bad.sum())} common-sample quantile rows carry non-finite "
                f"knots (first rid {int(rid[bad][0])}) — cannot score the "
                f"identical sample; investigate the quantile file.")
        # Consume each cell's mask bit; at the end, the un-consumed remainder =
        # common-sample cells the quantile file failed to cover.
        mask[rid] = False

        # Per (target, horizon) block: fetch y from the truth column.
        blk = rid // grid.n_wide
        wide_row = rid - blk * grid.n_wide
        order = np.argsort(blk, kind='stable')
        blk_s = blk[order]
        wr_s = wide_row[order]
        rid_s = rid[order]
        q_s = qmat[order]
        bounds = np.flatnonzero(np.r_[True, blk_s[1:] != blk_s[:-1], True])
        for i in range(len(bounds) - 1):
            lo_i, hi_i = bounds[i], bounds[i + 1]
            b_id = int(blk_s[lo_i])
            act_col = grid.actual_for_block(b_id)
            y = act_col[wr_s[lo_i:hi_i]]
            qb_raw = q_s[lo_i:hi_i]
            h = horizons[b_id % grid.n_h]

            # Chronos-2 emits mild quantile crossing on a nontrivial share of
            # rows (~8.5% at r13 pf_full scale). The predictive-CDF scoring
            # needs nondecreasing knots -> sort those rows. The same-run
            # cross-check below must use the RAW order: the writer integrated
            # the mean over the levels as emitted, and interior grid weights
            # are non-uniform at the tails, so sorting shifts the weighted sum.
            row_crossed = (qb_raw[:, 1:] < qb_raw[:, :-1]).any(axis=1)
            if row_crossed.any():
                n_crossed += int(row_crossed.sum())
                qb = np.sort(qb_raw, axis=1)
            else:
                qb = qb_raw

            u = quantile_pit(qb, taus, y)
            crps_q = piecewise_linear_crps(qb, taus, y)
            res = np.asarray(surro['residual'][rid_s[lo_i:hi_i]], dtype=np.float64)
            sig = np.maximum(np.asarray(surro['sigma'][rid_s[lo_i:hi_i]], dtype=np.float64),
                             args.sigma_floor)
            crps_g = _crps_per_entry(res, sig, 'gaussian', None)
            good_g = np.isfinite(crps_g)

            # Same-run cross-check: quantile-integrated mean (RAW emit order)
            # vs the predictions file's point (prediction = residual + actual).
            fin = np.isfinite(res)
            if fin.any():
                dev = np.abs((qb_raw @ grid_w)[fin] - (res[fin] + y[fin]))
                mean_dev = max(mean_dev, float(dev.max()))

            hist = np.bincount(np.minimum((u * PIT_BINS).astype(np.int64), PIT_BINS - 1),
                               minlength=PIT_BINS)
            cov = np.array([np.count_nonzero((y >= qb[:, i_lo]) & (y <= qb[:, i_hi]))
                            for _nom, i_lo, i_hi in iv_idx], dtype=np.int64)
            for key in (h, POOLED_KEY):
                a = acc[key]
                a['n'] += len(y)
                a['cov'] += cov
                a['pit_hist'] += hist
                a['pit_sum'] += float(u.sum())
                a['crps_q_sum'] += float(crps_q.sum())
                a['crps_g_sum'] += float(crps_g[good_g].sum())
                a['n_g'] += int(good_g.sum())

            degen = (qb[:, -1] - qb[:, 0]) == 0.0
            rail = np.abs(y) >= clip_rail
            n_degen += int(degen.sum())
            n_rail += int(rail.sum())
            sat = degen | rail
            for gname, gm in (('saturated', sat), ('interior', ~sat)):
                if not gm.any():
                    continue
                a = sat_acc[gname]
                a['n'] += int(gm.sum())
                a['cov'] += np.array(
                    [np.count_nonzero((y[gm] >= qb[gm][:, i_lo]) & (y[gm] <= qb[gm][:, i_hi]))
                     for _nom, i_lo, i_hi in iv_idx], dtype=np.int64)
                a['crps_q_sum'] += float(crps_q[gm].sum())
                gg = good_g & gm
                a['crps_g_sum'] += float(crps_g[gg].sum())
                a['n_g'] += int(gg.sum())
        if n_chunks % 20 == 0:
            print(f"  {n_chunks} chunks done ({time.time() - t0:.0f}s), "
                  f"pooled n so far = {acc[POOLED_KEY]['n']:,}", flush=True)

    n_uncovered = int(mask.sum())
    n_scored = acc[POOLED_KEY]['n']
    n_g = acc[POOLED_KEY]['n_g']
    print(f"\nScored {n_scored:,} cells from {n_rows:,} file rows "
          f"({n_dropped:,} outside the common sample / duplicates).", flush=True)
    print(f"  same-run cross-check: max |quantile-integrated mean - prediction| = "
          f"{mean_dev:.2e} (tolerance {MEAN_DEV_TOL:g})", flush=True)
    if n_crossed:
        print(f"  NOTE: {n_crossed:,} rows had quantile crossing (sorted before scoring).")
    if n_g != n_scored:
        print(f"  WARNING: Gaussian-surrogate column covers {n_g:,}/{n_scored:,} cells "
              f"(non-finite residual/sigma elsewhere) — its mean is over its own count.")
    if n_uncovered:
        msg = (f"{n_uncovered:,} common-sample cells have NO quantile row — "
               f"the 'identical sample' claim FAILS.")
        if args.allow_uncovered:
            print(f"  WARNING: {msg} Writing CSVs anyway (--allow-uncovered).")
        else:
            raise SystemExit(msg + " Refusing to write CSVs "
                             "(pass --allow-uncovered to override).")
    if mean_dev > MEAN_DEV_TOL:
        raise SystemExit(
            f"quantile file and predictions file disagree on the predictive mean "
            f"(max dev {mean_dev:.2e} > {MEAN_DEV_TOL:g}): mixed-run provenance — "
            f"refusing to write CSVs. (Also raised when --model points at a "
            f"MEDIAN-point file, e.g. chronos_raw_med: the gate reconstructs the "
            f"trapezoid-integrated MEAN, so check the file's point_stat before "
            f"chasing a provenance ghost.)")
    if args.expected_n is not None and n_scored != args.expected_n:
        raise SystemExit(f"scored n = {n_scored:,} != expected {args.expected_n:,}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, hist_rows = [], []
    for key in keys:
        a = acc[key]
        if a['n'] == 0:
            continue
        label = 'pooled' if key == POOLED_KEY else key
        row = {'horizon': label, 'n_cells': a['n'],
               # self-describing provenance: these columns are computed from the
               # model's own 21-level quantile grid, not a parametric density.
               'family': 'quantile21',
               'mean_pit': a['pit_sum'] / a['n'],
               'crps_quantile': a['crps_q_sum'] / a['n'],
               'crps_gaussian_surrogate': (a['crps_g_sum'] / a['n_g']
                                           if a['n_g'] else float('nan'))}
        for (nominal, _lo, _hi), c in zip(NATIVE_INTERVALS, a['cov']):
            row[f"cov_{int(nominal * 100)}"] = c / a['n']
        rows.append(row)
        hist_rows.append({'horizon': label,
                          **{f"bin_{i}": a['pit_hist'][i] / a['n'] * PIT_BINS
                             for i in range(PIT_BINS)}})

    sat_rows = []
    for gname in ('saturated', 'interior'):
        a = sat_acc[gname]
        if a['n'] == 0:
            continue
        row = {'group': gname, 'n_cells': a['n'],
               'share': a['n'] / n_scored if n_scored else float('nan'),
               'crps_quantile': a['crps_q_sum'] / a['n'],
               'crps_gaussian_surrogate': (a['crps_g_sum'] / a['n_g']
                                           if a['n_g'] else float('nan'))}
        for (nominal, _lo, _hi), c in zip(NATIVE_INTERVALS, a['cov']):
            row[f"cov_{int(nominal * 100)}"] = c / a['n']
        sat_rows.append(row)
    sat_df = pd.DataFrame(sat_rows)
    sat_df.to_csv(out_dir / 'saturation_split.csv', index=False, encoding='utf-8-sig')
    print(f"\nSaturation decomposition (degenerate knot span: {n_degen:,}; "
          f"truth at ±{clip_rail:g} rail: {n_rail:,}; union = 'saturated'):")
    print(sat_df.to_string(index=False))

    cov_df = pd.DataFrame(rows)
    cov_path = out_dir / 'coverage_by_horizon.csv'
    cov_df.to_csv(cov_path, index=False, encoding='utf-8-sig')
    pd.DataFrame(hist_rows).to_csv(out_dir / 'pit_hist_by_horizon.csv',
                                   index=False, encoding='utf-8-sig')
    print(f"\nWrote {cov_path} ({time.time() - t0:.0f}s total)")
    print(cov_df.to_string(index=False))
    pooled = cov_df[cov_df['horizon'] == 'pooled'].iloc[0]
    print(f"\nPOOLED SUMMARY  n={int(pooled['n_cells']):,}  "
          f"cov50={pooled['cov_50']:.3f} cov80={pooled['cov_80']:.3f} "
          f"cov90={pooled['cov_90']:.3f}  crps_quantile={pooled['crps_quantile']:.4f}  "
          f"crps_gaussian_surrogate={pooled['crps_gaussian_surrogate']:.4f}")


if __name__ == '__main__':
    main()
