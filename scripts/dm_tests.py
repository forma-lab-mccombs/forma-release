"""Diebold-Mariano tests vs a reference model, on the pooled-eval common sample.

Publication plan §4.5: per model-pair × horizon, collapse per-cell loss
differentials to CALENDAR-QUARTER means (each origin quarter counts once —
this absorbs all cross-sectional dependence, so the effective sample is the
~60 test quarters, not the 300M+ cells), then a Newey-West (Bartlett) variance
with bandwidth h-1: h-quarter-ahead forecasts from overlapping origins have
MA(h-1) forecast errors. The pooled (all-horizon) test uses the max-overlap
bandwidth max(h)-1.

Sample discipline: differentials are accumulated over EXACTLY the same
common sample as evaluate.py's pooled tables (finite actual, finite baseline,
finite residual in EVERY model in the pool), reusing the same per-forecast
error caches (src/eval_cache.py) — so a star in T1 tests precisely the two
numbers printed in that panel, on that panel's footprint.

Two loss differentials per pair, matching T1's two columns:
    d_sq  = e_ref^2 - e_other^2   (squared loss — the R² column's ranking;
                                   on a common sample R² ordering == MSE ordering)
    d_abs = |e_ref| - |e_other|   (absolute loss — the MAE column)
Negative mean d => the reference (Forma) is better. Markers are two-sided and
direction-coded: * / ** / *** when the reference wins at 10/5/1%,
† / †† / ††† when the comparator wins — the star regime must be able to
decorate a comparator win (e.g. the LLM MAE cells) or the table reads rigged.

Small-sample inference: the plain DM statistic over-rejects at T≈65 quarters
(synthetic MA(1) check: ~10% at nominal 5%), so the reported p-values apply the
Harvey-Leybourne-Newbold (1997) correction — DM* = DM·sqrt((T+1-2h+h(h-1)/T)/T)
against Student-t(T-1); the pooled row uses h = max horizon (conservative).
Uncorrected t is kept in the output for reference. Quarter series with interior
gaps (a quarter with no common cells at that horizon) are treated as
consecutive for the NW kernel — flagged in the output when it occurs.

Usage (pool dirs follow evaluate.py's --exp_dir convention):
  python scripts/dm_tests.py --exp_dir <forecasts>/r13_point_pool
  python scripts/dm_tests.py --exp_dir <forecasts>/llm_panelC --reference forma_fgrid__pf_full

Output: <exp_dir>/metrics/<feature_set>/<split>/dm/dm_vs_<reference>.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from forma.scoring.evaluate import (  # noqa: E402
    discover_forecast_files,
    discover_feature_sets,
    resolve_ground_truth_path,
)
from forma.scoring.eval_cache import (  # noqa: E402
    TruthGrid,
    get_error_cache,
    file_fingerprint,
    canon_quarters,
    PARQUET_ENGINE,
)

POOLED_KEY = 0  # horizon key for the all-horizon (pooled) accumulator row


def newey_west_tstat(d: np.ndarray, bandwidth: int) -> tuple[float, float, float]:
    """DM t-stat for mean(d)=0 with a Bartlett-kernel long-run variance.

    d is the chronologically ordered per-quarter mean differential series.
    Returns (mean, se, t). se uses max(bandwidth, 0) lags with weights
    1 - l/(L+1); the l=0 term is the plain variance. A non-positive long-run
    variance (possible in small samples with negative autocovariances) falls
    back to the l=0 variance so the t-stat stays defined.
    """
    T = len(d)
    if T < 2:
        return float(d.mean()) if T else np.nan, np.nan, np.nan
    dc = d - d.mean()
    L = min(max(int(bandwidth), 0), T - 1)
    lrv = float(dc @ dc) / T
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        lrv += 2.0 * w * float(dc[lag:] @ dc[:-lag]) / T
    if lrv <= 0.0:
        lrv = float(dc @ dc) / T
    se = np.sqrt(lrv / T)
    mean = float(d.mean())
    if se > 0.0:
        t = mean / se
    else:
        # Constant differential series (variance exactly 0): one model is
        # identically better every quarter. That is maximal evidence, not
        # missing evidence — signed inf (p=0, full marker), not NaN.
        t = 0.0 if mean == 0.0 else float(np.sign(mean)) * float('inf')
    return mean, float(se), t


def hln_corrected_p(t: float, T: int, h: int) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold small-sample DM correction.

    Returns (t_hln, p) with t_hln = t * sqrt((T+1-2h+h(h-1)/T)/T) referred to
    Student-t(T-1). Falls back to the uncorrected t when the correction factor
    is non-positive (h large relative to T)."""
    from scipy.stats import t as t_dist
    if np.isnan(t) or T < 2:
        return np.nan, np.nan
    # Signed inf (constant differential series) passes through: k2 scaling
    # keeps it inf and t_dist.sf(inf) = 0, i.e. p = 0 with correct direction.
    k2 = (T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T
    t_hln = t * np.sqrt(k2) if k2 > 0 else t
    return float(t_hln), float(2.0 * t_dist.sf(abs(t_hln), df=T - 1))


def marker(t: float, p: float) -> str:
    """Direction-coded significance marker: stars = reference better (d<0),
    daggers = comparator better (d>0)."""
    if not np.isfinite(p):
        return ''
    sym = '*' if t < 0 else '†'
    for thresh, k in ((0.01, 3), (0.05, 2), (0.10, 1)):
        if p <= thresh:
            return sym * k
    return ''


def run_split(forecast_files, truth_path, reference: str, out_dir: Path,
              split_name: str, cache_root=None) -> None:
    truth_df = pd.read_parquet(truth_path, engine=PARQUET_ENGINE)
    grid = TruthGrid(truth_df, fingerprint=file_fingerprint(truth_path))

    # Chronological order of the grid's quarter codes (codes are assigned in
    # order of appearance in the truth frame, not calendar order).
    q_uniques = pd.Index(canon_quarters(truth_df['quarter']).unique())
    chron = np.argsort(np.argsort(q_uniques.to_numpy()))  # code -> chron rank
    n_q = len(q_uniques)

    caches: dict[str, dict] = {}
    for file_path, model_name, fs_name, _split, seed in forecast_files:
        display_id = (f"{model_name}__{fs_name}__seed{seed}" if seed is not None
                      else f"{model_name}__{fs_name}")
        cache, hit = get_error_cache(Path(file_path), grid, cache_root=cache_root,
                                     display_id=display_id)
        if cache is None:
            raise RuntimeError(
                f"unreadable forecast {file_path} — DM must run on the full pool "
                f"(a dropped model re-bases the common sample vs the tables)")
        print(f"  {'cache hit' if hit else 'cache built'}: {display_id}")
        caches[display_id] = cache

    if reference not in caches:
        raise SystemExit(f"reference {reference!r} not in pool: {sorted(caches)}")
    others = [d for d in sorted(caches) if d != reference]
    horizons = list(grid.horizons)

    # Accumulators: per (other, horizon-or-POOLED) -> per-quarter-code sums.
    def _new():
        return {'sq': np.zeros(n_q), 'abs': np.zeros(n_q), 'n': np.zeros(n_q, dtype=np.int64)}
    acc = {(o, h): _new() for o in others for h in list(horizons) + [POOLED_KEY]}

    print(f"Streaming {len(grid.targets)} targets × {len(horizons)} horizons "
          f"({len(caches)} models, reference={reference})...")
    for target, horizon, sl, act, base in grid.iter_blocks():
        common = ~np.isnan(act) & np.isfinite(base)
        res = {}
        for did, c in caches.items():
            r = np.asarray(c['residual'][sl])
            common &= np.isfinite(r)
            res[did] = r
        if not common.any():
            continue
        idx = np.flatnonzero(common)
        qc = grid.quarter_codes[idx]
        r_ref = res[reference][idx].astype(np.float64)
        ref_sq = r_ref ** 2
        ref_abs = np.abs(r_ref)
        for o in others:
            r_o = res[o][idx].astype(np.float64)
            d_sq = ref_sq - r_o ** 2
            d_abs = ref_abs - np.abs(r_o)
            for key in (horizon, POOLED_KEY):
                a = acc[(o, key)]
                a['sq'] += np.bincount(qc, weights=d_sq, minlength=n_q)
                a['abs'] += np.bincount(qc, weights=d_abs, minlength=n_q)
                a['n'] += np.bincount(qc, minlength=n_q)

    max_bw = max(horizons) - 1
    rows = []
    for o in others:
        for key in list(horizons) + [POOLED_KEY]:
            a = acc[(o, key)]
            has = a['n'] > 0
            if not has.any():
                continue
            order = np.argsort(chron[np.flatnonzero(has)])
            codes = np.flatnonzero(has)[order]
            ranks = np.sort(chron[np.flatnonzero(has)])
            gapped = bool(((ranks[1:] - ranks[:-1]) > 1).any()) if len(ranks) > 1 else False
            d_sq_q = a['sq'][codes] / a['n'][codes]
            d_abs_q = a['abs'][codes] / a['n'][codes]
            bw = max_bw if key == POOLED_KEY else key - 1
            h_eff = max(horizons) if key == POOLED_KEY else key
            T = int(has.sum())
            m_sq, se_sq, t_sq = newey_west_tstat(d_sq_q, bw)
            m_abs, se_abs, t_abs = newey_west_tstat(d_abs_q, bw)
            t_sq_hln, p_sq = hln_corrected_p(t_sq, T, h_eff)
            t_abs_hln, p_abs = hln_corrected_p(t_abs, T, h_eff)
            rows.append({
                'model': o, 'reference': reference,
                'horizon': 'pooled' if key == POOLED_KEY else key,
                'n_quarters': T, 'nw_bandwidth': bw,
                'gapped_quarters': gapped,
                'mean_d_sq': m_sq, 'se_sq': se_sq, 't_sq': t_sq,
                't_sq_hln': t_sq_hln, 'p_sq': p_sq,
                'marker_sq': marker(t_sq, p_sq),
                'mean_d_abs': m_abs, 'se_abs': se_abs, 't_abs': t_abs,
                't_abs_hln': t_abs_hln, 'p_abs': p_abs,
                'marker_abs': marker(t_abs, p_abs),
            })

    out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_slug = reference.replace('__', '-')
    out_path = out_dir / f"dm_vs_{ref_slug}__{split_name}.csv"
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Wrote {len(out)} rows -> {out_path}")
    pooled = out[out['horizon'] == 'pooled']
    if len(pooled):
        print(pooled[['model', 'n_quarters', 'mean_d_sq', 't_sq', 'marker_sq',
                      'mean_d_abs', 't_abs', 'marker_abs']].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--exp_dir', required=True,
                    help="pool directory (evaluate.py convention: <exp_dir>/forecasts + config.yaml)")
    ap.add_argument('--reference', default=None,
                    help="reference display id (model__feature_set); default: the unique "
                         "model starting with 'forma'")
    ap.add_argument('--feature_set', default=None)
    ap.add_argument('--splits', default='test')
    ap.add_argument('--cache-dir', default=None,
                    help="override cache root (default <forecasts dir>/eval_cache — "
                         "shared with evaluate.py)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    forecasts_dir = exp_dir / 'forecasts'
    processed_dir = Path('data/processed')

    dataset_tag = None
    cfg_path = exp_dir / 'config.yaml'
    if cfg_path.exists():
        import yaml
        with open(cfg_path, 'r', encoding='utf-8-sig') as f:
            run_cfg = yaml.safe_load(f) or {}
        dataset_tag = (run_cfg.get('data') or {}).get('dataset_tag')
        if dataset_tag:
            print(f"Using dataset_tag='{dataset_tag}' from pool config.")

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    feature_sets = discover_feature_sets(forecasts_dir, args.feature_set, splits)
    if not feature_sets:
        raise SystemExit(f"no forecasts discovered under {forecasts_dir}")

    for fs in feature_sets:
        for split_name in splits:
            forecast_files = discover_forecast_files(forecasts_dir, feature_set=fs,
                                                     split=split_name)
            if not forecast_files:
                continue
            truth_path = resolve_ground_truth_path(processed_dir, fs, split_name,
                                                   dataset_tag=dataset_tag)
            if not truth_path.exists():
                print(f"[{fs}/{split_name}] no ground truth at {truth_path}, skipping")
                continue
            reference = args.reference
            if reference is None:
                formas = [f"{m}__{f}" for _p, m, f, _s, seed in forecast_files
                          if m.startswith('forma') and seed is None]
                if len(formas) != 1:
                    raise SystemExit(
                        f"--reference required (found {formas or 'no'} forma candidates)")
                reference = formas[0]
            out_dir = exp_dir / 'metrics' / fs / split_name / 'dm'
            print(f"\n=== DM [{fs}/{split_name}] reference={reference} ===")
            run_split(forecast_files, truth_path, reference, out_dir, split_name,
                      cache_root=args.cache_dir)


if __name__ == '__main__':
    main()
