"""
Master evaluation script for comparing all models across one or many feature sets.

This script scans results/forecasts/ for *predictions.parquet files with the
naming pattern: {model_name}__{feature_set}__{split}__predictions.parquet
(seed-aware: {model}__{feature_set}__{seed}__{split}__predictions.parquet).

For every feature_set with both forecast files and a matching ground truth file
it computes comparative metrics (R^2, MAE, avg obs) per
(model, target, forecast_horizon). Forecasts that additionally carry a `sigma`
column (predictive std, likelihood track) are also scored on mean NLL and mean
CRPS (a proper score in TARGET units, far less tail-sensitive than the log score
— a tail-robust cross-check on NLL) — both computed HERE against the single shared
ground truth under the forecast's own density family, never trusted from the generator.

Grid error-cache design (issues #145 + #84; replaces the pre-melt + two-pass
forecast-read accumulator):
  Every scoreable cell has a fixed address on the canonical truth grid
  (see src/eval_cache.py): row_id = (target_id * n_h + h_idx) * n_wide +
  wide_row. Each forecast file gets a persistent per-cell float32 residual
  cache (residual = prediction - actual == change-space residual; baseline
  cancels), built ONCE by this evaluation code from the forecast + shared
  ground truth and validated by file fingerprints — re-evaluating the same
  forecast against a different comparison pool reuses the cache.

  Phase 1 (caches): load or build the error cache for every forecast. A file
  that fails to read is excluded — which would silently re-base the all-models
  common sample — so by default that ABORTS (--allow-missing overrides),
  exactly as before.

  Phase 2 (scoring): stream block-by-block over the grid's contiguous
  (target, horizon) slices. Per block: the common metric sample is the cells
  with a FINITE cached residual in EVERY readable model AND a finite baseline
  (the strict all-models key intersection — truth ⋂ every forecast). Truth
  stats (n_complete, Σy, Σy² of actual_change) come straight from the wide
  truth columns under that mask, so the R² denominator is model- and
  order-independent. Per model: ss_res/sae over the common cells, n_obs
  (avg_obs) over the model's FULL coverage, NLL over common cells with a
  valid sigma. Every (target, horizon) update propagates to all 4 AGG_LEVELS
  as POOLED ratios of sums.

  Finalize: R^2 = 1 - ss_res / (sum_y^2 - (sum_y)^2/n + 1e-12),
            MAE = sae / n_complete, avg_obs = n_obs / |union(quarters)|.
  Matches r6 @ 9242e46 formulas (incl. +1e-12 numerical guard). A post-loop
  self-check asserts every model's per-(target, horizon) n_metric equals the
  group's n_complete (the coverage-identity invariant) and warns loudly on
  any violation.

Precision: residuals are cached float32 and accumulated in float64; metrics
match the previous all-float64 pipeline to ~1e-6 (tables report 3 decimals).

Peak memory ≈ the wide truth frame (~7 GB at pf_full scale) + one cache build
(dense float32 cells + seen bitmask, ~3 GB) + per-block work (a few MB);
scoring reads the caches via memmap. Comfortable on a 64 GB box — the ~130 GB
pre-melt floor (issue #145) is gone. Warm runs skip the forecast parquet reads
entirely.

Usage examples (from repo root):
  python -m forma.scoring.evaluate                 # evaluate all feature sets discovered
  python -m forma.scoring.evaluate --feature_set core   # only evaluate 'core'
  python -m forma.scoring.evaluate --rebuild-cache      # ignore + rebuild error caches
"""

import argparse
import json
import math
import pandas as pd
import numpy as np
from scipy.special import ndtr  # standard-normal CDF (vectorized) for Gaussian CRPS
from pathlib import Path
import sys
import gc
from collections import defaultdict

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from forma.scoring.eval_cache import (  # noqa: E402
    TruthGrid,
    get_error_cache,
    file_fingerprint,
    PARQUET_ENGINE,
    SIGMA_COL,  # noqa: F401  (re-exported: was an evaluate.py constant pre-refactor)
)

MERGE_COLS = ['firm_id', 'quarter', 'target', 'forecast_horizon']

# Aggregation levels (group_keys); empty list = global pool.
AGG_LEVELS = {
    'by_target_horizon': ['target', 'forecast_horizon'],
    'by_target': ['target'],
    'by_horizon': ['forecast_horizon'],
    'global': [],
}

# ---------------------------- Likelihood (NLL) track ---------------------------- #
# Forecasts may carry an OPTIONAL predictive scale alongside the point forecast.
# The forecast file stays "dumb" — it reports only mu (`prediction`) and, if the
# model has a variance head, sigma (`SIGMA_COL`, the predictive standard deviation
# in the SAME regularized target space as `prediction`/`actual`). evaluate owns
# the single ground truth and computes the NLL itself (never trusts a generator
# to report its own likelihood). The predictive *density* (Gaussian vs Student-t
# and its df) is a property of the model, not the data, so it travels in an
# OPTIONAL `{stem}.nll.json` sidecar; absent → Gaussian (the r10 default).


def _read_nll_meta(file_path) -> tuple[str, float | None]:
    """Resolve the predictive density (family, df) for a forecast.

    Reads an OPTIONAL `{stem}.nll.json` sidecar ({"family":
    "gaussian"|"laplace"|"student_t", "df": <float|null>}). The density is a
    property of the MODEL (which loss it was
    fit under), not of the ground truth, so it has to travel with the forecast — but
    we keep it out of the parquet rows. Absent/unparseable → ('gaussian', None), the
    r10 likelihood-track default.
    """
    side = Path(file_path).parent / f"{Path(file_path).stem}.nll.json"
    if side.exists():
        try:
            with open(side, encoding="utf-8") as fh:
                meta = json.load(fh)
            fam = str(meta.get('family', 'gaussian')).lower()
            df = meta.get('df', None)
            return fam, (float(df) if df is not None else None)
        except Exception as e:
            print(f"  Warning: could not parse NLL sidecar {side.name}: {e}; "
                  f"defaulting to gaussian")
    return 'gaussian', None


def _nll_per_entry(res: np.ndarray, sigma: np.ndarray,
                   family: str, df: float | None) -> np.ndarray:
    """Per-observation negative log-likelihood of `actual` under the model's
    predictive density centered at the forecast.

    res = pred - actual (the predictive error; baseline cancels in the change
    metric). `sigma` is the predictive std in the SAME space as `res`, so the
    density is N(0, sigma^2) over the error (Student-t analog for fat tails).
    The FULL normalizing constant is included so the result is a proper mean NLL
    ("mean likelihood of the test data given model mu/sigma"); within a family this
    ranks identically to Forma/FFNN's constant-dropped training loss.
    """
    sigma_sq = sigma * sigma
    log_sigma_sq = np.log(sigma_sq)
    if family == 'gaussian':
        return 0.5 * math.log(2.0 * math.pi) + 0.5 * log_sigma_sq + 0.5 * res * res / sigma_sq
    if family == 'laplace':
        # `sigma` is the predictive STD; the Laplace scale is b = sigma / sqrt(2)
        # (predictive std = sqrt(2)*b). NLL = log(2b) + |res|/b. Mirrors the
        # 'laplace' training loss in forma.py so train and eval share the density.
        b = sigma / math.sqrt(2.0)
        return np.log(2.0 * b) + np.abs(res) / b
    if family == 'student_t':
        if df is None:
            raise ValueError(
                "student_t NLL requires 'df' in the forecast's .nll.json sidecar")
        nu = float(df)
        # log normalizing constant of the standardized Student-t density
        c = math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(nu * math.pi)
        return -c + 0.5 * log_sigma_sq + 0.5 * (nu + 1.0) * np.log1p(res * res / (nu * sigma_sq))
    raise ValueError(
        f"Unknown NLL family '{family}'. Expected 'gaussian', 'laplace', or 'student_t'.")


def _crps_per_entry(res: np.ndarray, sigma: np.ndarray,
                    family: str, df: float | None) -> np.ndarray:
    """Per-observation Continuous Ranked Probability Score under the model's
    predictive density centered at the forecast.

    A PROPER scoring rule like NLL, but reported in the UNITS of the target (not
    log-density) and far less tail-sensitive: an outlier contributes ~linearly in
    distance instead of NLL's -log(density) blow-up, and CRPS reduces to MAE for a
    point forecast. Used to test whether a heavy-tailed family's NLL win (e.g.
    Laplace << Gaussian) reflects genuinely better predictive distributions or is a
    log-score tail artifact. Closed forms: Gneiting & Raftery (2007), Jordan et
    al. (2019, scoringRules).

    res = pred - actual; `sigma` is the predictive STD in the same space as res.
    Gaussian and Laplace are both symmetric about the forecast, so each closed form
    is EVEN in res — the sign cancels and res/scale is used directly. Families
    without a wired closed form (student_t) return NaN per entry, so they drop out
    of the accumulator cleanly rather than being mis-scored.
    """
    if family == 'gaussian':
        # CRPS(N(mu,sigma^2), y) = sigma * [ w(2*Phi(w)-1) + 2*phi(w) - 1/sqrt(pi) ],
        # w = (y-mu)/sigma. The bracket is even in w, so w = res/sigma is fine.
        w = res / sigma
        phi = np.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
        return sigma * (w * (2.0 * ndtr(w) - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    if family == 'laplace':
        # `sigma` is the predictive STD; the Laplace scale is b = sigma / sqrt(2).
        # CRPS(Laplace(mu,b), y) = b * [ |z| + exp(-|z|) - 3/4 ], z = (y-mu)/b (even).
        b = sigma / math.sqrt(2.0)
        z = np.abs(res) / b
        return b * (z + np.exp(-z) - 0.75)
    if family == 'student_t':
        # A closed-form Student-t CRPS exists (Jordan et al. 2019) but is not wired
        # here; the r11 likelihood arms are gaussian/laplace. NaN -> dropped.
        return np.full(np.shape(res), np.nan, dtype=np.float64)
    raise ValueError(
        f"Unknown CRPS family '{family}'. Expected 'gaussian', 'laplace', or 'student_t'.")


def _mixture_nll_per_cell(nll_stack: np.ndarray) -> np.ndarray:
    """Per-cell NLL of the equal-weight seed MIXTURE, given per-seed NLLs.

    nll_stack[k] is seed k's per-cell NLL (so its predictive density is
    exp(-nll_k)). The K-seed mixture density is the mean of the seed densities, so

        NLL_mix = -log( (1/K) * sum_k exp(-nll_k) )
                = log K - logsumexp_k(-nll_k).

    This is the TRUE mixture (a proper predictive density over the seeds), NOT the
    mean of the per-seed NLLs (a Jensen upper bound) and NOT a moment-matched
    single Gaussian — the r11 plan wants the honest mixture (~0.06 nats below
    moment-matching). Computed log-sum-exp-stable.
    """
    neg = -nll_stack
    m = np.max(neg, axis=0)
    lse = m + np.log(np.sum(np.exp(neg - m), axis=0))
    return math.log(nll_stack.shape[0]) - lse

# ---------------------------- Forecast Loading ---------------------------- #

def discover_forecast_files(forecasts_dir: Path, feature_set: str | None = None, split: str | None = None) -> list[tuple[Path, str, str, str, int | None]]:
    """Discover forecast files matching filters without loading data.

    Filename grammar (`__` is the segment separator, `_` is allowed inside
    any segment):
      * 3-segment legacy: ``{model}__{feature_set}__predictions.parquet`` →
        split defaults to "test".
      * 4-segment current: ``{model}__{feature_set}__{split}__predictions.parquet``.
      * 5-segment seeded: ``{model}__{feature_set}__{seed}__{split}__predictions.parquet``
        (seed must be an integer token).
    The model token may itself contain ``__`` (e.g. the LLM benchmark emits
    ``llm_{prompt_version}__{sanitized_model}__core__test__predictions.parquet``);
    in that case ``model_name`` is reconstructed by joining all leading
    segments with ``__``.

    Parsing is anchored at the end of the filename (predictions ← split ←
    [seed] ← feature_set ← model) so adding more underscores in the model
    token doesn't break discovery.

    Args:
        forecasts_dir: Directory containing prediction files
        feature_set: Optional feature set name to filter on
        split: Optional split name to filter on

    Returns:
        List of tuples: (file_path, model_name, feature_set_name, split_name, seed)
        seed is None for files without a seed token.
    """
    results = []
    unmatched: list[str] = []  # files that look like forecasts but match no naming pattern
    for file_path in forecasts_dir.glob('*predictions.parquet'):
        parts = file_path.stem.split('__')
        if len(parts) < 3 or parts[-1] != 'predictions':
            unmatched.append(file_path.name)
            continue

        seed = None
        # 3-segment legacy: {model}__{feature_set}__predictions  (assume test)
        if len(parts) == 3:
            model_name = parts[0]
            feature_set_name = parts[1]
            split_name = 'test'
        # 5+ segment seeded: {model...}__{feature_set}__{seed}__{split}__predictions
        elif len(parts) >= 5 and parts[-3].isdigit():
            seed = int(parts[-3])
            model_name = '__'.join(parts[:-4])
            feature_set_name = parts[-4]
            split_name = parts[-2]
        # 4+ segment current: {model...}__{feature_set}__{split}__predictions
        else:
            model_name = '__'.join(parts[:-3])
            feature_set_name = parts[-3]
            split_name = parts[-2]

        # feature_set / split are intentional caller filters, not naming failures
        if feature_set is not None and feature_set != feature_set_name:
            continue
        if split is not None and split != split_name:
            continue

        results.append((file_path, model_name, feature_set_name, split_name, seed))

    if unmatched:
        print(
            f"Warning: {len(unmatched)} *predictions.parquet file(s) did not match "
            f"any known naming pattern and were skipped: {sorted(unmatched)}"
        )

    return results


def discover_feature_sets(forecasts_dir: Path, feature_set_filter: str | None, splits: list[str]) -> list[str]:
    """Discover feature sets from forecast filenames without loading full data."""
    feature_sets: set[str] = set()
    for split in splits:
        for _, _, fs_name, _, _ in discover_forecast_files(forecasts_dir, feature_set=feature_set_filter, split=split):
            feature_sets.add(fs_name)
    return sorted(feature_sets)

# ---------------------------- Evaluation Core ---------------------------- #

def _new_truth_entry():
    """Model-independent sufficient stats per (level, group_vals)."""
    return {
        'quarters': set(),
        'n_complete': 0,
        'sum_y': 0.0,
        'sum_y2': 0.0,
    }


def _new_model_entry():
    """Model-dependent sufficient stats per (level, group_vals, display_id).

    n_metric is the per-model count of metric-eligible cells (in the common
    sample AND finite residual). Under the coverage-identity invariant it equals
    the group's truth-side n_complete for every model; the finalize step asserts
    this and warns loudly otherwise (the Gap-2 self-check).

    sum_nll/n_nll accumulate the likelihood track: the summed per-cell NLL and its
    count over the common-sample cells that also carry a finite positive sigma. They
    stay 0 for point-only forecasts (no sigma), which finalize surfaces as NaN nll.

    sum_z2/n_z2 accumulate the calibration track: summed squared standardized
    residual ((pred-actual)/sigma)^2 over the same valid-sigma cells. z2 = sum/n is
    the mean z^2 — the per-seed calibration diagnostic. The "-> 1 when calibrated"
    reading is GAUSSIAN-ONLY (for student_t/laplace sigma is a scale, not the std),
    so z^2 is accumulated ONLY for the gaussian family and left blank (NaN) for the
    others. Same 0-default for point-only forecasts.
    """
    return {'n_obs': 0, 'n_metric': 0, 'ss_res': 0.0, 'sae': 0.0,
            'sum_nll': 0.0, 'n_nll': 0, 'sum_z2': 0.0, 'n_z2': 0,
            'sum_crps': 0.0, 'n_crps': 0}


def level_group_vals(level_name: str, target: str, horizon: int) -> tuple:
    """Materialize the group_vals tuple a (target, horizon) lands in at each level."""
    if level_name == 'global':
        return ()
    if level_name == 'by_target':
        return (target,)
    if level_name == 'by_horizon':
        return (horizon,)
    if level_name == 'by_target_horizon':
        return (target, horizon)
    raise ValueError(f"unknown level: {level_name!r}")


def evaluate_models(forecast_files: list[tuple[Path, str, str, str, int | None]],
                    truth_df: pd.DataFrame, output_dir: Path, split_name: str = "test",
                    allow_missing: bool = False, truth_fingerprint: dict | None = None,
                    cache_root=None, rebuild_cache: bool = False,
                    sigma_floor: float = 1e-4):
    """Pooled evaluation on the all-models common sample via grid error caches.

    Phase 1 ensures every forecast has a valid per-cell residual cache
    (see src/eval_cache.py — built here from forecast + shared ground truth,
    fingerprint-validated, reused across runs and comparison pools). Phase 2
    streams the grid's contiguous (target, horizon) blocks once: the common
    metric sample per block is the cells finitely predicted by EVERY readable
    model with a finite baseline (the strict all-models key intersection), so
    every model is scored on the IDENTICAL rows with the IDENTICAL
    model-independent R² denominator — correct, file-order-independent, and
    globally comparable (the PR #91 coverage-identity guarantees, preserved).
    A target with no all-models cells drops from ALL levels.

    Args:
        forecast_files: list of (file_path, model_name, feature_set_name,
                        split_name, seed) tuples. seed is None for 4-part filenames.
        truth_df: wide-format truth ({target}_t{h} + {target}_level_0 cols).
        output_dir: directory to save metrics CSVs (split subdir is created).
        split_name: data-split label for file naming (test/train/val).
        allow_missing: score only the readable forecasts when some fail to
                       read, instead of aborting (re-bases the common sample!).
        truth_fingerprint: identity of the truth artifact for cache validation
                       (file_fingerprint(truth_path) when the caller knows the
                       path; content-hashed from the frame otherwise).
        cache_root: override directory for error caches (default:
                       <forecast dir>/eval_cache/).
        rebuild_cache: ignore existing caches and rebuild from the parquet.
    """
    print(f"Preparing data for evaluation [{split_name}] (grid error-cache accumulation)...")

    grid = TruthGrid(truth_df, fingerprint=truth_fingerprint)
    if not grid.targets:
        print("No valid targets found in ground truth. Skipping.")
        return

    forecast_meta: list[tuple[Path, str, str, int | None, str]] = []
    for file_path, model_name, fs_name, _split, seed in forecast_files:
        if not Path(file_path).exists():
            print(f"  Warning: missing forecast {file_path}")
            continue
        model_id = f"{model_name}__{fs_name}"
        display_id = f"{model_id}__seed{seed}" if seed is not None else model_id
        forecast_meta.append((Path(file_path), model_id, fs_name, seed, display_id))
    if not forecast_meta:
        print("No valid forecast files. Skipping.")
        return
    all_display_ids = [m[4] for m in forecast_meta]

    print(f"Found {len(grid.targets)} targets, {len(forecast_meta)} forecast files")
    print("Computing metrics on CHANGE from baseline (actual - baseline)")
    print("Pooled R^2/MAE at each AGG_LEVEL via sufficient-stats accumulator.")

    # ===================== Phase 1: per-forecast error caches =====================
    print("Phase 1/2: loading/building per-forecast error caches...")
    caches: dict[str, dict] = {}
    nll_density: dict[str, tuple[str, float | None]] = {}
    ok_display_ids: list[str] = []
    failed_reads: list[str] = []
    for fi, (file_path, _mid, _fs, _seed, display_id) in enumerate(forecast_meta):
        cache, was_hit = get_error_cache(file_path, grid, cache_root=cache_root,
                                         rebuild=rebuild_cache, display_id=display_id)
        if cache is None:
            failed_reads.append(f"{display_id} ({file_path.name})")
            continue
        print(f"  [{fi+1}/{len(forecast_meta)}] {'cache hit' if was_hit else 'cache built'}: {file_path.name}")
        # Build-time warnings scroll away across runs; re-surface dup counts on hits.
        n_dup = cache['meta'].get('n_duplicate', 0)
        if was_hit and n_dup > 0:
            print(f"  Warning: {display_id}: {n_dup} duplicate {MERGE_COLS} row(s) "
                  f"(recorded at cache build; first of each kept)")
        caches[display_id] = cache
        ok_display_ids.append(display_id)
        if cache['sigma'] is not None:
            fam, dfv = _read_nll_meta(file_path)
            nll_density[display_id] = (fam, dfv)
            print(f"  {display_id}: sigma present — scoring NLL ({fam}"
                  + (f", df={dfv}" if fam == 'student_t' else "") + ")")

    # A forecast that fails to read is silently EXCLUDED from the all-models
    # intersection -- which re-bases the common sample and shifts EVERY surviving
    # model's pooled metric. That is the one way metrics move without a hard
    # failure, so make it loud: abort unless --allow-missing is set.
    if failed_reads:
        msg = (f"{len(failed_reads)}/{len(forecast_meta)} forecast(s) failed to read and "
               f"would be dropped from the all-models common sample (re-basing it and "
               f"shifting every surviving model's metrics): " + "; ".join(failed_reads))
        if not allow_missing:
            raise RuntimeError(
                msg + ". Refusing to emit silently-degraded metrics. Fix the file(s) "
                "(see the per-file warning above; pyarrow-written files -> "
                "scripts/convert_forecast_to_fastparquet.py) or pass --allow-missing "
                "to score only the readable models on purpose.")
        print(f"  WARNING (--allow-missing): {msg}")

    if not ok_display_ids:
        print("No readable forecasts. Skipping.")
        return

    if nll_density:
        print(f"  Likelihood-track sigma floor: {sigma_floor:g} "
              f"({'exact-zero sigmas are dropped' if sigma_floor == 0 else 'sub-floor sigmas scored at the floor'}).")

    # Per-model truth coverage (cells each model finitely predicts AND truth holds).
    # A model whose join keys drift (firm_id/quarter/target/horizon representation)
    # shows anomalously low coverage here -- visible before any metric is computed.
    print(f"  Per-model truth coverage ({len(ok_display_ids)} readable model(s)):")
    for did in ok_display_ids:
        print(f"    {caches[did]['meta'].get('n_finite_residual', 0):>14,} rows  {did}")

    # ===================== Phase 2: block-streamed scoring =====================
    truth_stats: dict[tuple[str, tuple], dict] = defaultdict(_new_truth_entry)
    model_stats: dict[tuple[str, tuple], dict[str, dict]] = defaultdict(dict)

    # Seed-mixture NLL families: group the sigma-bearing, seeded forecasts by their
    # base model id (display_id = "{model_id}__seed{seed}"). A family with >=2 seeds
    # gets a TRUE equal-weight mixture NLL (computed per cell over the common sample,
    # below) — the headline likelihood number per arm. Non-seeded / single-seed /
    # point-only models simply don't form a family, so this is zero-overhead for them.
    _did_to_mid = {m[4]: m[1] for m in forecast_meta}
    _did_to_seed = {m[4]: m[3] for m in forecast_meta}
    seed_families: dict[str, list] = defaultdict(list)
    for did in ok_display_ids:
        if did in nll_density and _did_to_seed.get(did) is not None:
            seed_families[_did_to_mid[did]].append(did)
    mixture_families = {mid: dids for mid, dids in seed_families.items() if len(dids) >= 2}
    # mixture_stats[(level_name, gv, mid)] -> {'sum_mix': float, 'n_mix': int}
    mixture_stats: dict[tuple[str, tuple, str], dict] = defaultdict(
        lambda: {'sum_mix': 0.0, 'n_mix': 0})
    if mixture_families:
        print(f"  Seed-mixture NLL will be scored for {len(mixture_families)} model "
              f"family(ies): " + ", ".join(f"{mid}({len(d)} seeds)"
                                           for mid, d in mixture_families.items()))

    print("Phase 2/2: scoring all models block-by-block on the common sample...")
    n_ok = len(ok_display_ids)
    n_common_total = 0
    targets_seen: set[str] = set()
    targets_with_common: set[str] = set()
    targets_with_obs: dict[str, set] = {did: set() for did in ok_display_ids}

    for target, horizon, sl, act, base in grid.iter_blocks():
        # Old melt semantics: rows with NaN actual are dropped; everything else stays.
        act_fin = ~np.isnan(act)
        if not act_fin.any():
            continue
        targets_seen.add(target)

        res_blocks = {did: np.asarray(caches[did]['residual'][sl]) for did in ok_display_ids}

        # Common metric sample: finite residual in EVERY model AND finite baseline
        # (finite residual already implies finite prediction and non-NaN actual).
        common = act_fin & np.isfinite(base)
        for r in res_blocks.values():
            common &= np.isfinite(r)
        n_c = int(common.sum())
        n_common_total += n_c
        if n_c > 0:
            targets_with_common.add(target)

        # ---- model-INDEPENDENT truth stats (computed once, no file ordering) ----
        ac_common = (act - base)[common]
        sy = float(ac_common.sum())
        sy2 = float((ac_common ** 2).sum())
        # quarters over the FULL slice (avg_obs is a coverage diagnostic, not the
        # metric): distinct quarter codes among non-NaN-actual rows.
        q_present = np.flatnonzero(
            np.bincount(grid.quarter_codes[act_fin], minlength=grid.n_quarter_codes))
        qvals = q_present.tolist()
        for level_name in AGG_LEVELS:
            gv = level_group_vals(level_name, target, horizon)
            entry = truth_stats[(level_name, gv)]
            entry['quarters'].update(qvals)
            entry['n_complete'] += n_c
            entry['sum_y'] += sy
            entry['sum_y2'] += sy2

        # ---- per-model stats ----
        for did in ok_display_ids:
            r = res_blocks[did]
            # avg_obs counts non-null predictions over full coverage (r6 semantic);
            # NaN residual = pred NaN, actual NaN, or key outside the forecast.
            n_obs = int(np.count_nonzero(~np.isnan(r)))
            if n_obs > 0:
                targets_with_obs[did].add(target)

            r_c = r[common].astype(np.float64)
            finite_c = np.isfinite(r_c)
            if not finite_c.all():  # cannot happen if common is correct; stay safe
                r_c = r_c[finite_c]
            n_metric = int(finite_c.sum())
            if n_metric > 0:
                ss_res = float((r_c ** 2).sum())
                sae = float(np.abs(r_c).sum())
            else:
                ss_res = sae = 0.0

            if did in nll_density:
                fam, dfv = nll_density[did]
                # Likelihood-track sigma floor: an evaluation-time POLICY, not a
                # property of the forecast file. Raw-sigma producers (chronos_raw)
                # write zeros for degenerate-certainty cells; the floor decides
                # their treatment: floor>0 scores them at sigma=floor (default
                # 1e-4 reproduces the historical model-side floor and is a no-op
                # for models that never emit sigma below it); floor=0 drops exact
                # zeros via the sig>0 validity mask. np.maximum propagates NaN.
                sig = np.maximum(np.asarray(caches[did]['sigma'][sl], dtype=np.float64), sigma_floor)
                nll_valid = common & np.isfinite(sig) & (sig > 0.0)
                if nll_valid.any():
                    res_v = r[nll_valid].astype(np.float64)
                    sig_v = sig[nll_valid]
                    nll_row = _nll_per_entry(res_v, sig_v, fam, dfv)
                    good = np.isfinite(nll_row)
                    n_nll = int(good.sum())
                    sum_nll = float(nll_row[good].sum()) if n_nll > 0 else 0.0
                    # CRPS over the same valid-sigma common cells, scored under the
                    # forecast's OWN family — a tail-robust proper-score complement
                    # to NLL (units of the target; NaN for unwired families).
                    crps_row = _crps_per_entry(res_v, sig_v, fam, dfv)
                    good_c = np.isfinite(crps_row)
                    n_crps = int(good_c.sum())
                    sum_crps = float(crps_row[good_c].sum()) if n_crps > 0 else 0.0
                    # z^2 = ((pred-actual)/sigma)^2 — calibration diagnostic, but the
                    # "-> 1 when calibrated" reading is GAUSSIAN-ONLY. For
                    # student_t/laplace `sigma` is a scale, not the std (e.g.
                    # E[z^2] = nu/(nu-2) for student_t), so emitting z^2 there would
                    # read as miscalibrated. Leave it blank (NaN) for non-Gaussian
                    # families. (The canonical r11 likelihood track is Gaussian.)
                    if fam == 'gaussian':
                        z2_row = (res_v / sig_v) ** 2
                        good_z = np.isfinite(z2_row)
                        n_z2 = int(good_z.sum())
                        sum_z2 = float(z2_row[good_z].sum()) if n_z2 > 0 else 0.0
                    else:
                        n_z2, sum_z2 = 0, 0.0
                else:
                    n_nll, sum_nll, n_z2, sum_z2 = 0, 0.0, 0, 0.0
                    n_crps, sum_crps = 0, 0.0
            else:
                n_nll, sum_nll, n_z2, sum_z2 = 0, 0.0, 0, 0.0
                n_crps, sum_crps = 0, 0.0

            for level_name in AGG_LEVELS:
                gv = level_group_vals(level_name, target, horizon)
                acc = model_stats[(level_name, gv)].setdefault(did, _new_model_entry())
                acc['n_obs'] += n_obs
                acc['n_metric'] += n_metric
                acc['ss_res'] += ss_res
                acc['sae'] += sae
                acc['sum_nll'] += sum_nll
                acc['n_nll'] += n_nll
                acc['sum_z2'] += sum_z2
                acc['n_z2'] += n_z2
                acc['sum_crps'] += sum_crps
                acc['n_crps'] += n_crps

        # ---- seed-mixture NLL (per family) on this block's common sample ----
        for mid, dids in mixture_families.items():
            # Cells valid for the mixture: common AND every seed has a usable sigma.
            mix_valid = common.copy()
            for fdid in dids:
                fsig = np.maximum(np.asarray(caches[fdid]['sigma'][sl], dtype=np.float64), sigma_floor)
                mix_valid &= np.isfinite(fsig) & (fsig > 0.0)
            if not mix_valid.any():
                continue
            nll_rows = []
            for fdid in dids:
                ffam, fdfv = nll_density[fdid]
                fres = res_blocks[fdid][mix_valid].astype(np.float64)
                fsig = np.maximum(np.asarray(caches[fdid]['sigma'][sl], dtype=np.float64)[mix_valid], sigma_floor)
                nll_rows.append(_nll_per_entry(fres, fsig, ffam, fdfv))
            mix = _mixture_nll_per_cell(np.vstack(nll_rows))
            good_m = np.isfinite(mix)
            if not good_m.any():
                continue
            s_mix = float(mix[good_m].sum())
            n_mix = int(good_m.sum())
            for level_name in AGG_LEVELS:
                gv = level_group_vals(level_name, target, horizon)
                ent = mixture_stats[(level_name, gv, mid)]
                ent['sum_mix'] += s_mix
                ent['n_mix'] += n_mix

    dropped_targets = sorted(targets_seen - targets_with_common)
    if dropped_targets:
        print(
            f"  NOTE: {len(dropped_targets)} target(s) dropped from ALL metric levels — "
            f"no cell is present-and-finite in every one of the {n_ok} readable models, so "
            f"they have no all-models common sample: "
            + ", ".join(dropped_targets[:12])
            + (" ..." if len(dropped_targets) > 12 else "")
        )

    # Total common-sample size: the denominator backbone for every pooled metric.
    # Track this run-over-run — a sudden change means coverage shifted.
    print(f"  Common metric sample: {n_common_total:,} truth row(s) present-and-finite in ALL "
          f"{n_ok} model(s), over {len(targets_with_common)} target(s). "
          f"Track this run-over-run — a sudden change means coverage shifted.")
    for did in ok_display_ids:
        print(f"  {did}: held rows in {len(targets_with_obs[did])} truth target(s); "
              f"scored on the common sample")

    gc.collect()

    # ---- Self-check: every scored model's n_metric must equal n_complete ----
    # The common sample is BY CONSTRUCTION the cells every readable model finitely
    # predicts, so a violation signals a bug in the coverage mask.
    scored_display_ids = ok_display_ids
    invariant_violations = []
    for (ln, gv), tentry in truth_stats.items():
        if ln != 'by_target_horizon':
            continue
        n_complete = tentry['n_complete']
        mbucket = model_stats.get((ln, gv), {})
        for display_id in scored_display_ids:
            m = mbucket.get(display_id)
            n_metric = m['n_metric'] if m is not None else 0
            if n_metric != n_complete:
                invariant_violations.append((gv, display_id, n_metric, n_complete))
    if invariant_violations:
        print(
            f"\n  *** BUG: common-sample invariant VIOLATED in "
            f"{len(invariant_violations)} (target, horizon, model) cells — pooled "
            f"metrics are NOT trustworthy; investigate the coverage mask. "
            f"First few offending cells:"
        )
        for gv, did, nm, nc in invariant_violations[:10]:
            print(f"        {gv} model={did}: n_metric={nm} != n_complete={nc} (delta={nm - nc})")
    else:
        print("  Common-sample invariant holds (every model scored on the identical intersection).")

    # ---- Finalize per (level, gv, model) metrics ----
    print(f"Finalizing metrics from {len(truth_stats)} accumulated stat groups...")
    all_level_metrics: dict[str, pd.DataFrame] = {}
    for level_name, group_keys in AGG_LEVELS.items():
        rows = []
        for (ln, gv), tentry in truth_stats.items():
            if ln != level_name:
                continue
            n = tentry['n_complete']
            ss_total = (tentry['sum_y2'] - (tentry['sum_y'] ** 2) / n) if n > 0 else 0.0
            n_quarters = len(tentry['quarters'])
            mbucket = model_stats.get((ln, gv), {})

            # all_display_ids here (not scored_display_ids): a model that failed to read
            # still gets a row with NaN r2/mae (m is None) — surfaced, not silently hidden.
            for display_id in all_display_ids:
                m = mbucket.get(display_id)
                n_obs = m['n_obs'] if m is not None else 0
                avg_obs = n_obs / n_quarters if n_quarters > 0 else np.nan
                if n > 0 and m is not None:
                    mae = m['sae'] / n
                    # +1e-12 / >1e-15 guards match r6 (stable on near-constant targets).
                    r2 = (1.0 - m['ss_res'] / (ss_total + 1e-12)) if ss_total > 1e-15 else np.nan
                else:
                    mae = np.nan
                    r2 = np.nan
                # Mean NLL over the common rows the model carried a valid sigma for.
                # NaN for point-only forecasts (no sigma) so they leave the nll CSV blank.
                n_nll = m['n_nll'] if m is not None else 0
                nll = (m['sum_nll'] / n_nll) if (m is not None and n_nll > 0) else np.nan
                # Mean z^2 (calibration) over the same valid-sigma cells; NaN for
                # point-only forecasts so they leave the z2 table blank.
                n_z2 = m['n_z2'] if m is not None else 0
                z2 = (m['sum_z2'] / n_z2) if (m is not None and n_z2 > 0) else np.nan
                # Mean CRPS over the same valid-sigma common cells; NaN for
                # point-only forecasts and unwired families (e.g. student_t).
                n_crps = m['n_crps'] if m is not None else 0
                crps = (m['sum_crps'] / n_crps) if (m is not None and n_crps > 0) else np.nan
                row = {
                    'model': display_id,
                    'r2': r2,
                    'mae': mae,
                    'nll': nll,
                    'crps': crps,
                    'z2': z2,
                    'avg_obs': avg_obs,
                    'n_complete_rows': n,
                }
                for key_name, val in zip(group_keys, gv):
                    row[key_name] = val
                rows.append(row)
        df = pd.DataFrame(rows)
        if 'forecast_horizon' in df.columns and not df.empty:
            df['forecast_horizon'] = df['forecast_horizon'].astype(int)
        all_level_metrics[level_name] = df

    del truth_stats, model_stats
    gc.collect()

    if all(df.empty for df in all_level_metrics.values()):
        print("No metrics could be computed. Exiting.")
        return

    bth_df = all_level_metrics.get('by_target_horizon', pd.DataFrame())
    if not bth_df.empty:
        print(f"Computed metrics for {len(bth_df)} model-target-horizon combinations")

    output_dir = output_dir / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_metric(metric: str, metrics_df: pd.DataFrame, suffix: str, index_cols: list[str]):
        if metrics_df.empty:
            return None
        if not index_cols:
            # Single pooled row per model; aggfunc='first' is correct HERE (one
            # row per model in the input frame). The buggy historical use was
            # on per-(model, target, horizon) frames where 'first' picked an
            # arbitrary cell.
            pivot = metrics_df.pivot_table(columns='model', values=metric, aggfunc='first')
        else:
            pivot = metrics_df.pivot_table(index=index_cols, columns='model', values=metric)
        # pivot_table drops all-NaN model columns; an empty pivot means no model
        # reported this metric (e.g. `nll` when no forecast carried a sigma). Skip
        # the degenerate file rather than write an unparseable empty CSV.
        if pivot.empty:
            return None
        filename = f"{metric}_scores__{suffix}.csv" if metric != 'avg_obs' else f"avg_obs__{suffix}.csv"
        out_path = output_dir / filename
        pivot.to_csv(out_path)
        return metric, pivot, out_path

    outputs = []
    saved = {}
    for level_name, group_keys in AGG_LEVELS.items():
        level_df = all_level_metrics.get(level_name, pd.DataFrame())
        for metric in ('r2', 'mae', 'nll', 'crps', 'z2', 'avg_obs'):
            result = save_metric(metric, level_df, level_name, group_keys)
            outputs.append(result)
            saved[(metric, level_name)] = result

    global_df = all_level_metrics.get('global', pd.DataFrame())
    if not global_df.empty:
        n_path = output_dir / "n_complete_rows__global.csv"
        global_df.pivot_table(columns='model', values='n_complete_rows', aggfunc='first').to_csv(n_path)
        outputs.append(('n_complete_rows', None, n_path))

    seeded = [m for m in forecast_meta if m[3] is not None]
    bth_for_seed = all_level_metrics.get('by_target_horizon', pd.DataFrame())
    if seeded and not bth_for_seed.empty:
        display_to_base = {m[4]: m[1] for m in forecast_meta}
        display_to_seed = {m[4]: m[3] for m in forecast_meta}
        seed_df = bth_for_seed.copy()
        seed_df['base_model'] = seed_df['model'].map(display_to_base)
        seed_df['seed'] = seed_df['model'].map(display_to_seed)
        agg = seed_df.groupby(['base_model', 'target', 'forecast_horizon']).agg(
            r2_mean=('r2', 'mean'),
            r2_std=('r2', 'std'),
            mae_mean=('mae', 'mean'),
            mae_std=('mae', 'std'),
            nll_mean=('nll', 'mean'),
            nll_std=('nll', 'std'),
            crps_mean=('crps', 'mean'),
            crps_std=('crps', 'std'),
            z2_mean=('z2', 'mean'),
            z2_std=('z2', 'std'),
            avg_obs_mean=('avg_obs', 'mean'),
            n_seeds=('seed', 'nunique'),
        ).reset_index()
        agg_path = output_dir / "seed_agg__by_target_horizon.csv"
        agg.to_csv(agg_path, index=False)
        outputs.append(('seed_agg', None, agg_path))
        print(f"Seed-aggregated metrics saved to {agg_path}")
        for metric in ('r2_mean', 'mae_mean', 'nll_mean', 'crps_mean'):
            pivot = agg.pivot_table(index=['target', 'forecast_horizon'], columns='base_model', values=metric)
            # Same guard as save_metric: an all-NaN metric (e.g. nll_mean when no
            # seeded forecast carried sigma) drops every column -> skip the degenerate
            # empty CSV rather than write an unparseable one.
            if pivot.empty:
                continue
            p = output_dir / f"seed_agg__{metric}__by_target_horizon.csv"
            pivot.to_csv(p)
            outputs.append((metric, None, p))

    # ---- Seed-mixture NLL output (per base model family, per level) ----
    # The honest likelihood headline per arm: a proper equal-weight mixture over
    # seeds (see _mixture_nll_per_cell), distinct from seed_agg's nll_mean (the
    # average of per-seed NLLs, a Jensen upper bound).
    if mixture_stats:
        for level_name, group_keys in AGG_LEVELS.items():
            rows = []
            for (ln, gv, mid), ent in mixture_stats.items():
                if ln != level_name or ent['n_mix'] <= 0:
                    continue
                row = {'model': mid, 'mixture_nll': ent['sum_mix'] / ent['n_mix']}
                for key_name, val in zip(group_keys, gv):
                    row[key_name] = val
                rows.append(row)
            if not rows:
                continue
            mdf = pd.DataFrame(rows)
            if group_keys:
                pivot = mdf.pivot_table(index=group_keys, columns='model', values='mixture_nll')
            else:
                pivot = mdf.pivot_table(columns='model', values='mixture_nll', aggfunc='first')
            if pivot.empty:
                continue
            mp = output_dir / f"mixture_nll__{level_name}.csv"
            pivot.to_csv(mp)
            outputs.append(('mixture_nll', None, mp))
        print(f"Seed-mixture NLL saved for {len({k[2] for k in mixture_stats})} family(ies).")

    print(f"\nEvaluation complete [{split_name}]. Results saved to:")
    for item in outputs:
        if item is None:
            continue
        _, _, p = item
        print(f"- {p}")


# ---------------------------- Main Entry Point ---------------------------- #

def resolve_ground_truth_path(
    processed_dir: Path,
    feature_set: str,
    split: str = "test",
    dataset_tag: str | None = None,
) -> Path:
    """Resolve the processed ground truth path for a feature set and split.

    Delegates tagged/untagged tabular path resolution to resolve_dataset_path,
    and keeps a legacy {split}__{feature_set}.parquet fallback for older runs.
    """
    from forma.data.config_utils import resolve_dataset_path

    preferred = resolve_dataset_path(
        processed_dir, f"tabular_{split}", feature_set, dataset_tag
    )
    if preferred.exists():
        return preferred
    legacy = processed_dir / f"{split}__{feature_set}.parquet"
    if legacy.exists():
        return legacy
    return preferred

def main():
    parser = argparse.ArgumentParser(description='Evaluate forecasting models across feature sets')
    parser.add_argument('--feature_set', type=str, required=False, help='Optional: restrict evaluation to a single feature set')
    parser.add_argument('--skip_target_date_calc', action='store_true', help='Deprecated no-op (kept for CLI compatibility)')
    parser.add_argument('--exp_dir', type=str, help='Path to an experiment directory (e.g., results/EXP_...)')
    parser.add_argument('--comment', type=str, help='Evaluation-time comment to append to exp_summary.txt')
    parser.add_argument('--allow-missing', action='store_true',
                        help='Score only the readable forecasts when some fail to read, instead '
                             'of aborting. WARNING: dropping a model re-bases the all-models common '
                             'sample and shifts every surviving model\'s pooled metric.')
    parser.add_argument('--splits', type=str, default='test,train,val',
                        help='Comma-separated list of splits to evaluate (default: test,train,val).')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Ignore existing error caches and rebuild them from the forecast '
                             'parquets (e.g. after changing eval-cache code).')
    parser.add_argument('--sigma-floor', type=float, default=1e-4,
                        help='Likelihood-track sigma floor applied at scoring time (default 1e-4). '
                             'Raw-sigma forecasts (e.g. chronos_raw) may carry sigma=0 for '
                             'degenerate-certainty cells; floor>0 scores them at sigma=floor '
                             '(1e-4 reproduces the historical model-side floor; no-op for models '
                             'that never emit smaller sigma), floor=0 drops exact-zero cells from '
                             'NLL/CRPS instead. Point track (R2/MAE) is unaffected.')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Root directory for per-forecast error caches (default: '
                             '<forecasts dir>/eval_cache/). Point at a big drive if needed.')
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir) if args.exp_dir else None

    if exp_dir and not exp_dir.exists():
        print(f"Error: Experiment directory {exp_dir} does not exist.")
        return 1

    forecasts_dir = exp_dir / 'forecasts' if exp_dir else Path('results/forecasts')
    metrics_base_dir = exp_dir / 'metrics' if exp_dir else Path('results/metrics')

    # Log comment if provided
    if args.comment and exp_dir:
        summary_path = exp_dir / "exp_summary.txt"
        with open(summary_path, "a") as f:
            f.write(f"\nEvaluation Comment: {args.comment}\n")
            f.write("-" * 50 + "\n")
    processed_dir = Path('data/processed')
    raw_dir = Path('data/raw')

    # Pull dataset_tag from the run's saved config so ground truth matches the
    # split the run was trained against. Falls back to legacy untagged naming.
    dataset_tag: str | None = None
    if exp_dir and (exp_dir / "config.yaml").exists():
        import yaml
        with open(exp_dir / "config.yaml", "r", encoding="utf-8-sig") as f:
            run_cfg = yaml.safe_load(f) or {}
        dataset_tag = (run_cfg.get("data") or {}).get("dataset_tag")
        if dataset_tag:
            print(f"Using dataset_tag='{dataset_tag}' from run config.")

    # Splits to evaluate: test is primary, train and val are for diagnostics
    splits_to_evaluate = [s.strip() for s in args.splits.split(',') if s.strip()]

    feature_sets = discover_feature_sets(forecasts_dir, args.feature_set, splits_to_evaluate)
    if not feature_sets:
        print("No valid forecast files found. Exiting.")
        return 1

    # Optional filtering (defensive check if filters above returned no matching set)
    if args.feature_set:
        if args.feature_set not in feature_sets:
            print(f"No forecasts found for feature_set={args.feature_set}. Exiting.")
            return 1
        feature_sets = [args.feature_set]

    # make a dictionary from feature_set -> ground truth path (for test split, used for filtering)
    ground_truth_paths = {
        fs: resolve_ground_truth_path(processed_dir, fs, 'test', dataset_tag=dataset_tag)
        for fs in feature_sets
    }

    # Filter only those with available test ground truth
    evaluable_feature_sets = [fs for fs in feature_sets if ground_truth_paths[fs].exists()]
    if not evaluable_feature_sets:
        print("No feature sets have both forecasts and ground truth. Exiting.")
        if feature_sets:
            example_fs = feature_sets[0]
            expected = processed_dir / f"tabular_test__{example_fs}.parquet"
            print(f"Expected ground truth like: {expected}")
        return 1

    # Track per-split eval failures across every feature_set/split so the process
    # can exit non-zero at the end. The per-split try/except below deliberately
    # continues (so every split's traceback prints), which means the fail-loud
    # RuntimeError from evaluate_models protects the *output* (no degraded metrics
    # file is written) but does NOT by itself set the exit status -- a CI/pipeline
    # caller grepping the exit code would otherwise see 0. Collect failures here
    # and sys.exit(1) once the loops finish.
    eval_failures: list[str] = []

    for fs in evaluable_feature_sets:
        print(f"\n=== Evaluating feature_set: {fs} ===")

        output_dir = metrics_base_dir / fs

        # Evaluate on all splits (test, train, val)
        for split_name in splits_to_evaluate:
            # Discover forecast files for this feature set and split (without loading data)
            forecast_files = discover_forecast_files(forecasts_dir, feature_set=fs, split=split_name)

            if not forecast_files:
                print(f"  [{split_name}] No forecasts found for this split, skipping...")
                continue

            print(f"  [{split_name}] Found {len(forecast_files)} forecast files")

            truth_path = resolve_ground_truth_path(processed_dir, fs, split_name, dataset_tag=dataset_tag)

            if not truth_path.exists():
                print(f"  [{split_name}] Ground truth not found at {truth_path}, skipping...")
                continue

            try:
                truth_df = pd.read_parquet(truth_path, engine=PARQUET_ENGINE)
            except Exception as e:
                print(f"  [{split_name}] Failed to load ground truth from {truth_path}: {e}")
                continue
            # Key canonicalization (quarter -> Period[Q], firm_id -> lossless
            # Int64) happens inside TruthGrid/eval_cache — no pre-normalization.
            # Truth identity for error-cache validation: tied to the on-disk file.
            truth_fp = file_fingerprint(truth_path)

            print(f"\n--- Evaluating {split_name} split ---")
            try:
                evaluate_models(forecast_files, truth_df, output_dir, split_name=split_name,
                                allow_missing=args.allow_missing,
                                truth_fingerprint=truth_fp,
                                cache_root=args.cache_dir,
                                rebuild_cache=args.rebuild_cache,
                                sigma_floor=args.sigma_floor)
            except Exception as e:
                print(f"  [{split_name}] Error evaluating: {e}")
                import traceback
                traceback.print_exc()
                eval_failures.append(f"{fs}/{split_name}: {e}")
            finally:
                del truth_df
                gc.collect()

    if eval_failures:
        print(f"\n{len(eval_failures)} split(s) failed to evaluate cleanly "
              f"(no metrics file written for these):")
        for f in eval_failures:
            print(f"  - {f}")
        print("Exiting non-zero so pipeline callers see the failure.")
        sys.exit(1)

    print("\nAll evaluations complete.")

if __name__ == '__main__':
    raise SystemExit(main())
