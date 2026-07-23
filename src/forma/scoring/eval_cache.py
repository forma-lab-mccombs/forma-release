"""Per-forecast prediction-error caches aligned to a canonical truth grid.

Motivation (issues #145 / #84): evaluate.py and pairwise_evaluate.py spent
their wall time and RAM re-joining hundreds of millions of long-form,
string-keyed forecast rows against melted truth slices — the held slice dict
alone peaked ~130 GB on a pf_full pool, and every (target, model) pair paid a
pandas merge on object firm_id keys. Both costs are paid again on every re-run,
even though the (forecast file, truth file) pair hasn't changed.

Canonical grid
--------------
The wide truth frame already enumerates every scoreable (firm_id, quarter) row
exactly once (n_wide rows, unique keys). With the discovered target list and
the global sorted horizon list H, every scoreable cell has a fixed address:

    row_id = (target_id * len(H) + h_idx) * n_wide + wide_row

so one forecast's prediction errors fit in ONE dense float32 array of length
n_cells = n_targets * len(H) * n_wide — NaN wherever the forecast holds no
matching row (or the truth actual is NaN). Each (target, horizon) block is a
contiguous slice of length n_wide, so evaluators can stream block-by-block from
a memmap without ever materializing a long-form join.

    residual = prediction - actual

which equals the change-space residual (prediction - baseline) - actual_change
(the baseline cancels): the cache depends ONLY on the forecast file and the
truth file, never on the comparison set — so the same cache serves pooled
evaluation, pairwise evaluation, and any future comparison pool.

Division of labor (the "models only output forecasts" rule)
-----------------------------------------------------------
Caches are built BY EVALUATION CODE from the forecast file plus the single
shared ground truth. Models never write them; the cache just memoizes
evaluate's first step (key-join + residual). Likelihood (NLL) is still
computed at evaluation time from the cached residual + sigma and the
`{stem}.nll.json` sidecar — the cache stores only arithmetic facts.

Validation / invalidation
-------------------------
`meta.json` records fingerprints (name, size, mtime_ns) of the forecast file,
a fingerprint of the truth (file fingerprint when the caller knows the path;
content hash otherwise), and the grid signature (n_wide, targets, horizons).
Any mismatch -> the cache is ignored and rebuilt. meta.json is written LAST,
so an interrupted build never leaves a loadable cache.

Precision: residuals are stored float32 and accumulated in float64 by the
evaluators. Metrics agree with the previous float64 pipeline to ~1e-6
(relative) — well within the 3-decimal precision reported in result tables.
Storage is n_cells * 4 bytes per forecast (~2.2 GB at pf_full scale).

Dedup semantics: duplicate (firm_id, quarter, target, forecast_horizon) keys
keep the FIRST occurrence in file order, as before. One deliberate difference:
dedup now applies only to grid-mappable rows, so a duplicate whose first
occurrence is out-of-truth no longer shadows a later scoreable row.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET_ENGINE = 'fastparquet'
SIGMA_COL = 'sigma'
# v2: key canonicalization changed from datetime64[us] truncation to
# Period[Q] quarters + lossless-Int64 firm_ids (PR #143 semantics) — v1
# caches are refused and rebuilt.
CACHE_FORMAT_VERSION = 2
FORECAST_KEY_COLS = ['firm_id', 'quarter', 'target', 'forecast_horizon']
FORECAST_COLS = FORECAST_KEY_COLS + ['prediction']


def canon_firm_ids(s: pd.Series) -> pd.Series:
    """Canonicalize firm_id for cross-source key matching (PR #143 semantics).

    Truth uses zero-padded strings ('001004'); LLM benchmark forecasts use
    ints. Cast to nullable Int64 — but only when LOSSLESS for the whole
    series (no new nulls introduced), so genuinely non-numeric ids (e.g. the
    test fixtures' 'F1' / 'firm_0000') pass through unchanged. Applied
    independently to the truth frame and to each forecast chunk; both sides
    of a numeric universe land on Int64, both sides of a non-numeric one
    stay strings.
    """
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().sum() == s.notna().sum():
        return num.astype('Int64')
    return s


def canon_quarters(s: pd.Series) -> pd.Series:
    """Canonicalize quarter to Period[Q] — the calendar quarter IS the key.

    Truth carries 23:59:59.999999999 [ns] EOQ stamps, Forma forecasts
    23:59:59.999999 [us], LLM benchmark forecasts midnight EOQ; Period[Q]
    collapses all of them to the same calendar quarter (supersedes the old
    [ns]→[us] truncation, which handled the first two but not the third).
    """
    return pd.to_datetime(s).dt.to_period('Q')


def _discover_targets(truth_df: pd.DataFrame) -> tuple[list[str], dict[str, list[tuple[str, int]]]]:
    """Find (target, [(horizon_col, h)]) pairs with a `{target}_level_0` baseline column."""
    horizon_pattern = re.compile(r'(.+)_t(\d+)$')
    per_target: dict[str, list[tuple[str, int]]] = {}
    for col in truth_df.columns:
        m = horizon_pattern.match(col)
        if not m:
            continue
        per_target.setdefault(m.group(1), []).append((col, int(m.group(2))))
    targets = sorted(t for t in per_target if f"{t}_level_0" in truth_df.columns)
    for t in per_target:
        per_target[t].sort(key=lambda kv: kv[1])
    return targets, per_target


def file_fingerprint(path) -> dict:
    """Cheap identity fingerprint (name, size, mtime_ns) for cache validation."""
    p = Path(path)
    st = p.stat()
    return {'name': p.name, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def truth_content_fingerprint(truth_df: pd.DataFrame) -> dict:
    """Content-based truth fingerprint for callers holding only a DataFrame.

    Hashes shape, column names, the key columns, and per-column sums — enough
    to invalidate on any realistic truth change without a full-value hash.
    Callers that know the truth FILE should pass file_fingerprint() instead
    (cheaper and strictly tied to the on-disk artifact).
    """
    import hashlib
    h = hashlib.sha1()
    h.update(f"{truth_df.shape}".encode())
    h.update("\x00".join(map(str, truth_df.columns)).encode())
    h.update(pd.util.hash_pandas_object(truth_df['firm_id'], index=False).to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(truth_df['quarter'], index=False).to_numpy().tobytes())
    sums = truth_df.sum(numeric_only=True).to_numpy(np.float64)
    h.update(np.nan_to_num(sums).tobytes())
    return {'truth_sha1': h.hexdigest()}


class TruthGrid:
    """Canonical dense cell grid over a wide truth frame.

    Cell address: row_id = (target_id * n_h + h_idx) * n_wide + wide_row, where
    wide_row is the row's position in the wide truth frame, target_id indexes
    the sorted discovered targets, and h_idx indexes the global sorted horizon
    list. Each (target, horizon) block is one contiguous slice of n_wide cells.

    Holds a reference to the truth frame (left unmodified — canonical keys
    are computed on the side); `actual` and `baseline` are zero-copy column
    views, never melted.
    """

    def __init__(self, truth_df: pd.DataFrame, fingerprint: dict | None = None):
        firm_c = canon_firm_ids(truth_df['firm_id'])
        qtr_c = canon_quarters(truth_df['quarter'])
        # Uniqueness checked on the CANONICAL keys: two truth rows for the
        # same firm in the same calendar quarter (even at different
        # timestamps) would collide on the grid — fail loudly rather than
        # silently overwrite (the old melt+merge would have fanned out
        # forecast rows here instead).
        if pd.DataFrame({'f': firm_c, 'q': qtr_c}).duplicated().any():
            raise ValueError(
                "ground truth has duplicate (firm_id, calendar-quarter) rows; "
                "the grid requires unique canonical keys")
        self.truth_df = truth_df
        self.fingerprint = (fingerprint if fingerprint is not None
                            else truth_content_fingerprint(truth_df))

        targets, per_target = _discover_targets(truth_df)
        self.targets = targets
        self.target_to_id = {t: i for i, t in enumerate(targets)}
        horizons = sorted({h for t in targets for _c, h in per_target[t]})
        self.horizons = horizons
        self.n_h = len(horizons)
        self.n_wide = len(truth_df)
        self.n_cells = len(targets) * self.n_h * self.n_wide

        # horizon value -> h_idx lookup (dense; horizons are small ints)
        self._h_lut = np.full((horizons[-1] + 1) if horizons else 1, -1, dtype=np.int64)
        for i, h in enumerate(horizons):
            self._h_lut[h] = i

        # (firm_id, quarter) -> wide_row via two categorical encodings of the
        # CANONICAL keys (lossless-Int64 firm, Period[Q] quarter) + a dense
        # (n_firms x n_quarters) lookup matrix (-1 = key not in truth).
        self._firm_uniques = pd.Index(firm_c.unique())
        self._q_uniques = pd.Index(qtr_c.unique())
        fc = pd.Categorical(firm_c, categories=self._firm_uniques).codes.astype(np.int64)
        qc = pd.Categorical(qtr_c, categories=self._q_uniques).codes.astype(np.int64)
        self._wide_lut = np.full((len(self._firm_uniques), len(self._q_uniques)), -1, dtype=np.int64)
        self._wide_lut[fc, qc] = np.arange(self.n_wide, dtype=np.int64)
        # per-wide-row quarter code (int32): evaluators count distinct quarters
        # per block with bincount instead of np.unique over datetime values.
        self.quarter_codes = qc.astype(np.int32)
        self.n_quarter_codes = len(self._q_uniques)

        # (target_id, h_idx) -> truth column name; missing pairs have no truth
        # column (their cells stay NaN in every cache and are skipped by
        # iter_blocks).
        self._col_for: dict[tuple[int, int], str] = {}
        for t in targets:
            for col, h in per_target[t]:
                self._col_for[(self.target_to_id[t], int(self._h_lut[h]))] = col
        self._act_cache: dict[int, np.ndarray | None] = {}

    def grid_signature(self) -> dict:
        """Layout identity stored in cache meta; mismatch invalidates the cache."""
        return {
            'n_wide': int(self.n_wide),
            'targets': list(self.targets),
            'horizons': [int(h) for h in self.horizons],
        }

    def map_forecast_rows(self, chunk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized (firm_id, quarter, target, horizon) -> row_id for one chunk.

        Returns (row_id int64 array, valid bool mask). Rows whose key does not
        exist in the grid (unknown firm/quarter pair, target without a
        `{t}_level_0` column, horizon outside the truth's horizon list) are
        invalid — exactly the rows the old inner merge dropped.

        Keys are canonicalized per chunk (lossless-Int64 firm_id, Period[Q]
        quarter) so cross-source representation differences — int vs
        zero-padded-string firm ids, midnight vs EOQ-nanosecond quarter
        stamps — land on the same grid cells.
        """
        fcodes = self._canon_codes(chunk['firm_id'], canon_firm_ids, self._firm_uniques)
        qcodes = self._canon_codes(chunk['quarter'], canon_quarters, self._q_uniques)
        tcodes = pd.Categorical(chunk['target'], categories=self.targets).codes.astype(np.int64)
        h = chunk['forecast_horizon'].to_numpy(np.int64, copy=False)
        h_in_range = (h >= 0) & (h < len(self._h_lut))
        h_idx = np.where(h_in_range, self._h_lut[np.clip(h, 0, len(self._h_lut) - 1)], -1)

        key_ok = (fcodes >= 0) & (qcodes >= 0)
        wide_row = np.full(len(chunk), -1, dtype=np.int64)
        if key_ok.any():
            wide_row[key_ok] = self._wide_lut[fcodes[key_ok], qcodes[key_ok]]

        valid = (wide_row >= 0) & (tcodes >= 0) & (h_idx >= 0)
        row_id = np.where(valid, (tcodes * self.n_h + h_idx) * self.n_wide + wide_row, -1)
        return row_id, valid

    @staticmethod
    def _canon_codes(values: pd.Series, canon, categories: pd.Index) -> np.ndarray:
        """Categorical codes of canon(values) against `categories`, but with the
        (expensive) canonicalization applied only to the chunk's UNIQUE values.

        Key columns repeat heavily within a chunk (~13k firms / ~60 quarters per
        ~1M rows), and to_numeric/to_period cost per ELEMENT — canonicalizing
        uniques and broadcasting through factorize codes cut the pf_full cache
        build roughly in half. Equivalent to canon(full series): factorize drops
        NaN (code -1, mapped to -1 here), and canon_firm_ids' lossless-cast
        check over non-null uniques is the same condition as over all non-null
        values.
        """
        codes, uniq = pd.factorize(values)
        mapped = pd.Categorical(canon(pd.Series(uniq)),
                                categories=categories).codes.astype(np.int64)
        out = np.full(len(codes), -1, dtype=np.int64)
        pos = codes >= 0
        out[pos] = mapped[codes[pos]]
        return out

    def actual_for_block(self, block_id: int) -> np.ndarray | None:
        """float64 view of the actual column for block_id = target_id*n_h + h_idx.

        None when the truth has no `{target}_t{h}` column for this pair.
        """
        if block_id in self._act_cache:
            return self._act_cache[block_id]
        col = self._col_for.get((block_id // self.n_h, block_id % self.n_h))
        arr = (self.truth_df[col].to_numpy(np.float64, copy=False)
               if col is not None else None)
        self._act_cache[block_id] = arr
        return arr

    def iter_blocks(self):
        """Yield (target, horizon, cell_slice, actual_f64, baseline_f64) per block.

        Skips (target, horizon) pairs with no truth column — the same pairs the
        old melt produced no rows for. actual/baseline are zero-copy column
        views of length n_wide; cell_slice addresses the block in any cache
        array.
        """
        for t_id, target in enumerate(self.targets):
            baseline = self.truth_df[f"{target}_level_0"].to_numpy(np.float64, copy=False)
            for h_idx, h in enumerate(self.horizons):
                act = self.actual_for_block(t_id * self.n_h + h_idx)
                if act is None:
                    continue
                start = (t_id * self.n_h + h_idx) * self.n_wide
                yield target, int(h), slice(start, start + self.n_wide), act, baseline


# ---------------------------- cache I/O ---------------------------- #

def cache_dir_for(forecast_path, cache_root=None) -> Path:
    """Cache directory for one forecast file.

    Default root is `<forecast dir>/eval_cache/` (subdirectories don't match
    the `*predictions.parquet` discovery glob); `cache_root` overrides it,
    e.g. to point at a big slow drive.
    """
    forecast_path = Path(forecast_path)
    root = Path(cache_root) if cache_root else forecast_path.parent / 'eval_cache'
    return root / forecast_path.stem


def load_error_cache(forecast_path, grid: TruthGrid, cache_root=None) -> dict | None:
    """Load a cache as memmapped arrays iff every fingerprint matches.

    Returns {'residual': memmap, 'sigma': memmap|None, 'meta': dict} or None
    (missing, stale, or built against a different truth/grid).
    """
    d = cache_dir_for(forecast_path, cache_root)
    meta_path = d / 'meta.json'
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if meta.get('format_version') != CACHE_FORMAT_VERSION:
        return None
    if meta.get('forecast') != file_fingerprint(forecast_path):
        return None
    if meta.get('truth') != grid.fingerprint or meta.get('grid') != grid.grid_signature():
        return None
    try:
        residual = np.load(d / 'residual.npy', mmap_mode='r')
        sigma = np.load(d / 'sigma.npy', mmap_mode='r') if meta.get('has_sigma') else None
    except Exception:
        return None
    if residual.shape != (grid.n_cells,):
        return None
    return {'residual': residual, 'sigma': sigma, 'meta': meta}


def _atomic_save(arr: np.ndarray, path: Path) -> None:
    tmp = path.with_name(path.stem + '.tmp.npy')
    np.save(tmp, arr)
    os.replace(tmp, path)


def build_error_cache(forecast_path, grid: TruthGrid, cache_root=None,
                      display_id: str | None = None) -> dict | None:
    """Build (and persist) the error cache for one forecast file.

    Streams the forecast row-group by row-group (memory stays bounded by one
    row group + the dense cache arrays), maps each row to its grid cell, and
    scatters residual = prediction - actual (float32; NaN where the prediction
    is NaN, the actual is NaN, or the key is outside the grid). Duplicate keys
    keep the first occurrence in file order, warned loudly as before.

    Returns the loaded (memmapped) cache dict, or None if the file is
    unreadable / missing required columns — callers treat that exactly like
    the old per-file read failure.

    Windows note: rebuilding over an existing cache requires that no live
    memmap handle is open on its .npy files (os.replace fails loudly there;
    release/del prior cache objects first).
    """
    from fastparquet import ParquetFile

    forecast_path = Path(forecast_path)
    name = display_id or forecast_path.name
    try:
        forecast_fp = file_fingerprint(forecast_path)
        pf = ParquetFile(str(forecast_path))
        file_cols = set(pf.columns)
    except Exception as e:
        print(
            f"  Warning: fastparquet read failed for {forecast_path.name}: {e}\n"
            f"           If the file was written with pyarrow, run "
            f"`python scripts/convert_forecast_to_fastparquet.py {forecast_path}` "
            f"and re-run. Skipping."
        )
        return None
    missing = set(FORECAST_COLS) - file_cols
    if missing:
        print(f"  Warning: {forecast_path.name} missing required columns {missing}; skipping forecast")
        return None

    has_sigma = SIGMA_COL in file_cols
    cols = FORECAST_COLS + ([SIGMA_COL] if has_sigma else [])

    residual = np.full(grid.n_cells, np.nan, dtype=np.float32)
    sigma_arr = np.full(grid.n_cells, np.nan, dtype=np.float32) if has_sigma else None
    seen = np.zeros(grid.n_cells, dtype=bool)
    n_rows = 0
    n_unmatched = 0
    n_dup = 0

    try:
        for chunk in pf.iter_row_groups(columns=cols):
            n_rows += len(chunk)
            row_id, valid = grid.map_forecast_rows(chunk)
            n_unmatched += int((~valid).sum())
            if not valid.any():
                continue
            rid = row_id[valid]
            pred = chunk['prediction'].to_numpy(np.float64, copy=False)[valid]
            sig = (chunk[SIGMA_COL].to_numpy(np.float64, copy=False)[valid]
                   if has_sigma else None)

            # keep='first' across the whole file: within-chunk duplicates via
            # hash-based duplicated(), cross-chunk via the seen bitmask.
            dup = pd.Series(rid).duplicated().to_numpy() | seen[rid]
            if dup.any():
                n_dup += int(dup.sum())
                keep = ~dup
                rid = rid[keep]
                pred = pred[keep]
                if sig is not None:
                    sig = sig[keep]
                if rid.size == 0:
                    continue  # whole chunk was duplicates of earlier rows
            seen[rid] = True

            # Per (target, horizon) block: residual against that block's actual
            # column. Sort by block so each column is fetched once per chunk.
            blk = rid // grid.n_wide
            wide_row = rid - blk * grid.n_wide
            order = np.argsort(blk, kind='stable')
            rid_s = rid[order]
            blk_s = blk[order]
            wr_s = wide_row[order]
            pred_s = pred[order]
            sig_s = sig[order] if sig is not None else None
            bounds = np.flatnonzero(np.r_[True, blk_s[1:] != blk_s[:-1], True])
            for i in range(len(bounds) - 1):
                lo, hi = bounds[i], bounds[i + 1]
                act_col = grid.actual_for_block(int(blk_s[lo]))
                if act_col is None:
                    continue  # no truth column: cells stay NaN (never scoreable)
                residual[rid_s[lo:hi]] = pred_s[lo:hi] - act_col[wr_s[lo:hi]]
                if sig_s is not None:
                    sigma_arr[rid_s[lo:hi]] = sig_s[lo:hi]
    except Exception as e:
        print(f"  Warning: failed reading {forecast_path.name} during cache build: {e}; skipping.")
        return None

    if n_dup > 0:
        print(f"  Warning: {name}: {n_dup} duplicate {FORECAST_KEY_COLS} row(s); keeping first of each")

    n_finite = int(np.isfinite(residual).sum())
    d = cache_dir_for(forecast_path, cache_root)
    d.mkdir(parents=True, exist_ok=True)
    _atomic_save(residual, d / 'residual.npy')
    del residual, seen
    if sigma_arr is not None:
        _atomic_save(sigma_arr, d / 'sigma.npy')
        del sigma_arr
    meta = {
        'format_version': CACHE_FORMAT_VERSION,
        'forecast': forecast_fp,
        'truth': grid.fingerprint,
        'grid': grid.grid_signature(),
        'has_sigma': has_sigma,
        'n_file_rows': n_rows,
        'n_unmatched': n_unmatched,
        'n_duplicate': n_dup,
        'n_finite_residual': n_finite,
    }
    # meta.json LAST: an interrupted build leaves no loadable cache.
    (d / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')

    cache = load_error_cache(forecast_path, grid, cache_root)
    if cache is None:  # e.g. file replaced mid-build; treat as unreadable
        print(f"  Warning: {name}: cache failed to validate immediately after build; skipping.")
    return cache


def get_error_cache(forecast_path, grid: TruthGrid, cache_root=None,
                    rebuild: bool = False, display_id: str | None = None
                    ) -> tuple[dict | None, bool]:
    """Load a valid cache or build one. Returns (cache_or_None, was_cache_hit)."""
    if not rebuild:
        cache = load_error_cache(forecast_path, grid, cache_root)
        if cache is not None:
            return cache, True
    return build_error_cache(forecast_path, grid, cache_root, display_id=display_id), False
