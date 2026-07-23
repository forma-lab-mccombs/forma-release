#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm_tuple_source.py — source LLM-benchmark inputs from Forma's stored tuples.

Phase B of docs/llm_forecast_generation_plan.md. Instead of re-deriving history
windows from raw Compustat (YTD→quarterly CF, aociq reconstruction, its own
origin enumeration), the LLM benchmark reads the *exact same* tuples Forma
trains/tests on:

    data/processed/tuple_test__pf_full__<dataset>.parquet   (RAW values)
    data/processed/account_id_map.csv                        (account_id↔code)

This module is pure data-shaping — it knows nothing about prompts, primitives,
or regularization. It hands back:

  * an eligible-origin pool (firm_id, center_q, max_horizon), enumerated with
    the same rule Forma's FormaWindowDataset uses (scale account present at the
    center, ≥ min_lookback grid history, ≥ 1 future quarter), restricted to the
    test window [test_start_q, test_end_q];
  * per-origin DENSE window DataFrames (one row per integer quarter in
    [center − (max_lookback−1), center + max_horizon], Compustat-code columns,
    NaN where a cell is absent) so the benchmark's existing finish_origin() can
    score actuals/baseline by positional row index exactly as on the raw path.

Tuple values are raw (post-YTD-conversion, post-derived-features baked in at
dataset-creation time) — the benchmark must NOT redo any of that, and this
module deliberately does not. Regularization stats + scale come from Forma's
own artifacts (the scale account, id-mapped here; Forma's reg-stats parquet).

Quarter integers are offsets from a base period (default 1969Q4 == 0), so
q161 == 2010Q1. See src/data/dataset_creation.py and src/data/forma_data.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Quarter <-> integer-offset mapping (matches Forma's base period) ───────────

class QuarterMap:
    """Bidirectional integer-quarter ↔ calendar mapping, base = `qbase` (q=0)."""

    def __init__(self, qbase: str = "1969Q4"):
        self.base = pd.Period(qbase, freq="Q")

    def to_period(self, q: int) -> pd.Period:
        return self.base + int(q)

    def to_qend_ts(self, q: int) -> pd.Timestamp:
        """Normalized quarter-END Timestamp (midnight) for integer quarter q.

        Matches the raw-Compustat path's `Timestamp + QuarterEnd(0)` so the
        prediction CSV's `quarter` column and custom_ids are identical in
        either sourcing mode."""
        return (self.base + int(q)).to_timestamp(how="end").normalize()

    def ts_to_q(self, ts) -> int:
        return (pd.Timestamp(ts).to_period("Q") - self.base).n

    @staticmethod
    def rel_label(offset: int) -> str:
        """Relative history label: 0 → 'Q0', −11 → 'Q-11', +3 → 'Q3'."""
        return "Q0" if offset == 0 else f"Q{offset}"


def history_labels(max_lookback: int) -> list[str]:
    """['Q-11', ..., 'Q-1', 'Q0'] for max_lookback=12."""
    return [QuarterMap.rel_label(off) for off in range(-(max_lookback - 1), 1)]


# ── Account map ────────────────────────────────────────────────────────────────

def load_account_map(path) -> tuple[dict[int, str], dict[str, int]]:
    amap = pd.read_csv(path)
    id_to_name = {int(i): str(n) for i, n in
                  zip(amap["account_id"], amap["account_name"])}
    name_to_id = {n: i for i, n in id_to_name.items()}
    return id_to_name, name_to_id


def load_firm_map(path) -> tuple[dict[int, int], dict[int, int]]:
    """firm_id_map.csv: firm_id (gvkey) ↔ firm_id_int (the dense id the tuples
    are keyed by). Returns (int_to_gvkey, gvkey_to_int).

    The benchmark must EMIT the gvkey (Forma's prediction parquet and the
    sklearn baselines key on the gvkey, not firm_id_int — the pooled evaluator
    and pairwise join normalize firm_id with pd.to_numeric), while DATA ACCESS
    into the tuples is by firm_id_int. This map bridges the two."""
    fmap = pd.read_csv(path)
    int_to_gvkey = {int(i): int(g) for g, i in
                    zip(fmap["firm_id"], fmap["firm_id_int"])}
    gvkey_to_int = {g: i for i, g in int_to_gvkey.items()}
    return int_to_gvkey, gvkey_to_int


# ── Tuple source ───────────────────────────────────────────────────────────────

class TupleSource:
    """Loads Forma's stored test tuples and serves history/origin windows.

    Per-firm data is stored as Forma stores it (sorted parallel arrays of
    quarter/account_id/value) so window slicing is a binary-search + mask,
    matching FormaWindowDataset's access pattern.
    """

    def __init__(self, tuple_parquet, account_map_path, *,
                 firm_map_path=None,
                 scale_name: str = "scale",
                 max_lookback: int = 12, max_horizon: int = 20,
                 min_lookback: int = 4,
                 test_start_q: int = 161, test_end_q: int = 220,
                 qbase: str = "1969Q4"):
        self.qmap = QuarterMap(qbase)
        self.max_lookback = int(max_lookback)
        self.max_horizon = int(max_horizon)
        self.min_lookback = int(min_lookback)
        self.test_start_q = int(test_start_q)
        self.test_end_q = int(test_end_q)

        self.id_to_name, self.name_to_id = load_account_map(account_map_path)
        if scale_name not in self.name_to_id:
            raise KeyError(f"scale account '{scale_name}' not in {account_map_path}")
        self.scale_id = self.name_to_id[scale_name]

        # firm_id_int ↔ gvkey. Identity map when no firm_id_map is supplied
        # (synthetic tests treat the tuple firm_id as already a gvkey).
        if firm_map_path is not None:
            self.int_to_gvkey, self.gvkey_to_int = load_firm_map(firm_map_path)
        else:
            self.int_to_gvkey, self.gvkey_to_int = None, None

        # `industry_id` (FF48 bucket, per-firm constant) rides along when the
        # tuple file carries it — Forma consumes it via the industry node
        # (industry_mode: 'node'), so the LLM benchmark surfaces it in the
        # prompt for input parity. Older tuple files predate the column; degrade
        # gracefully to "no industry signal".
        try:
            df = pd.read_parquet(
                tuple_parquet,
                columns=["firm_id", "account_id", "quarter", "value", "industry_id"])
            has_industry = True
        except (KeyError, ValueError):
            df = pd.read_parquet(
                tuple_parquet,
                columns=["firm_id", "account_id", "quarter", "value"])
            has_industry = False
        # Stable sort by (firm, quarter); account order within a quarter is
        # irrelevant since we look up by code.
        df = df.sort_values(["firm_id", "quarter"], kind="stable").reset_index(drop=True)

        self.lookup: dict[int, dict] = {}
        self.firm_industry_id: dict[int, int] = {}  # firm_id_int -> FF48 id
        for fid, g in df.groupby("firm_id", sort=False):
            self.lookup[int(fid)] = {
                "quarter": g["quarter"].to_numpy(np.int64),
                "account_id": g["account_id"].to_numpy(np.int64),
                "value": g["value"].to_numpy(np.float64),
            }
            if has_industry:
                iid = g["industry_id"].iloc[0]
                if pd.notna(iid):
                    self.firm_industry_id[int(fid)] = int(iid)

    # ── gvkey ↔ firm_id_int ───────────────────────────────────────────────────

    def gvkey_of(self, fid_int: int) -> int:
        if self.int_to_gvkey is None:
            return int(fid_int)
        return int(self.int_to_gvkey.get(int(fid_int), int(fid_int)))

    def int_of(self, gvkey: int) -> int:
        if self.gvkey_to_int is None:
            return int(gvkey)
        return int(self.gvkey_to_int.get(int(gvkey), int(gvkey)))

    def industry_id_of(self, gvkey: int) -> int | None:
        """FF48 industry id for an external gvkey, or None if the tuple file
        carried no industry_id (or the firm is absent / unmapped)."""
        return self.firm_industry_id.get(self.int_of(gvkey))

    # ── origin enumeration ────────────────────────────────────────────────────

    def build_origin_pool(self) -> pd.DataFrame:
        """Eligible (firm_id, center_q, max_horizon) origins.

        center_q is eligible iff:
          * test_start_q ≤ center_q < test_end_q
          * the scale account is present at center_q
          * center_q ≥ firm_min_q + (min_lookback − 1)   (grid history available)
          * firm_max_q > center_q                          (firm has data past c)
        max_horizon = min(self.max_horizon, test_end_q − center_q): the requested
        horizon is bounded by the END OF THE SAMPLE, NOT by the firm's last
        in-file quarter. This mirrors Forma's future-grid setup
        (use_future_grid: true, future_grid_sample_overhang: false): every origin
        is asked for the same delisting-blind horizon, and finish_origin scores
        only the quarters that actually have data (the rest are the LLM analogue
        of Forma's masked synthetic future nodes). Truncating at firm_max_q
        instead would leak delisting through the requested-quarter count — a
        short ask ⇒ the firm is about to disappear — which is exactly the
        survivorship/look-ahead channel the closed-overhang config removes.

        The upper bound is STRICT (center_q < test_end_q): an origin exactly at
        test_end_q has no future quarter inside the sample, so test_end_q − c
        would be 0 and it would be dispatched as a paid call with an empty
        (malformed) horizon list. The strict bound guarantees max_horizon ≥ 1.
        `firm_max_q > center_q` additionally drops firms with no data at all past
        c. Both are set-level filters, invisible per-prompt, carrying no
        delisting signal.
        """
        rows = []
        for fid, d in self.lookup.items():
            qs = d["quarter"]
            accs = d["account_id"]
            if qs.size == 0:
                continue
            fmin = int(qs[0])
            fmax = int(qs[-1])
            scale_qs = qs[accs == self.scale_id]
            if scale_qs.size == 0:
                continue
            lo = max(self.test_start_q, fmin + self.min_lookback - 1)
            for c in scale_qs:
                c = int(c)
                # Strict upper bound: c == test_end_q has no in-sample future
                # (test_end_q − c == 0 ⇒ max_h == 0, a degenerate paid call with
                # an empty horizon list), so require c < test_end_q.
                if c < lo or c >= self.test_end_q:
                    continue
                if fmax <= c:
                    continue
                # End-of-sample bound, NOT fmax — delisting-blind, matching
                # Forma's closed-overhang future grid (see docstring).
                max_h = min(self.max_horizon, self.test_end_q - c)
                rows.append((self.gvkey_of(fid), c, int(max_h)))
        # firm_id column holds the GVKEY (Forma's join key), not firm_id_int.
        pool = pd.DataFrame(rows, columns=["firm_id", "center_q", "max_horizon"])
        return pool.sort_values(["firm_id", "center_q"]).reset_index(drop=True)

    # ── per-origin window ─────────────────────────────────────────────────────

    def _byquarter(self, gvkey: int, q_lo: int, q_hi: int) -> dict[int, dict[str, float]]:
        """{quarter_int: {code: value}} for quarters in [q_lo, q_hi] (inclusive).

        `gvkey` is the external firm id; mapped to firm_id_int for array access."""
        d = self.lookup[self.int_of(gvkey)]
        qs = d["quarter"]
        start = int(np.searchsorted(qs, q_lo, side="left"))
        end = int(np.searchsorted(qs, q_hi, side="right"))
        sq = qs[start:end]
        sa = d["account_id"][start:end]
        sv = d["value"][start:end]
        out: dict[int, dict[str, float]] = {}
        id_to_name = self.id_to_name
        for q, a, v in zip(sq.tolist(), sa.tolist(), sv.tolist()):
            name = id_to_name.get(a)
            if name is None:
                continue
            out.setdefault(q, {})[name] = float(v)
        return out

    def build_window(self, gvkey: int, center_q: int, max_horizon: int):
        """Dense window for one origin (`gvkey` = external firm id).

        Returns (window_df, scale_q0, byq):
          * window_df: rows for q in [center−(max_lookback−1), center+max_horizon],
            columns = every Compustat code that appears anywhere in the slice +
            'datadate'; NaN for absent cells. Row position center − q_lo == q0_idx.
          * scale_q0: scale-account value at center (None ⇒ origin ineligible).
          * byq: the {quarter_int: {code: value}} dict (raw), for history building.
        q0_idx is always max_lookback − 1.
        """
        q_lo = center_q - (self.max_lookback - 1)
        q_hi = center_q + max_horizon
        byq = self._byquarter(gvkey, q_lo, q_hi)

        scale_q0 = byq.get(center_q, {}).get("scale")

        codes = sorted({c for qd in byq.values() for c in qd})
        grid_qs = list(range(q_lo, q_hi + 1))
        data = {"datadate": [self.qmap.to_qend_ts(q) for q in grid_qs]}
        for code in codes:
            data[code] = [byq.get(q, {}).get(code, np.nan) for q in grid_qs]
        window = pd.DataFrame(data)
        return window, scale_q0, byq


# ── Variable-length firm-block sampler (Phase C) ───────────────────────────────

def _runs_for_firm(centers: list[int]) -> list[list[int]]:
    """Split a firm's sorted eligible centers into consecutive-quarter runs."""
    runs: list[list[int]] = []
    run: list[int] = []
    for q in centers:
        if run and q == run[-1] + 1:
            run.append(q)
        else:
            if run:
                runs.append(run)
            run = [q]
    if run:
        runs.append(run)
    return runs


def variable_block_sample_origins(pool: pd.DataFrame, n_firms: int, seed: int, *,
                                  min_block: int = 4, max_block: int = 20,
                                  calendar_balance: bool = True,
                                  qbase: str = "1969Q4") -> pd.DataFrame:
    """Sample one variable-length block of consecutive origins per firm.

    For each chosen firm: pick one of its consecutive-eligible runs, then a
    START position with ≥ `min_block` eligible centers remaining in that run
    (uniform among valid starts); the block extends from the start to the end
    of the run, capped at `max_block`. So block length =
    min(max_block, run_end − start + 1), always ≥ min_block.

    `calendar_balance=True` spreads the *block-start* years across the test
    window by round-robining over years when choosing firms — this flattens the
    origin-year distribution (the old fixed-length design was tent-shaped).

    Deterministic in (pool, n_firms, seed, min_block, max_block,
    calendar_balance). Returns columns [firm_id, center_q, max_horizon].
    """
    qmap = QuarterMap(qbase)
    pool = pool.sort_values(["firm_id", "center_q"]).reset_index(drop=True)

    # max_horizon lookup keyed by (firm, center) so emitted blocks carry it.
    mh = {(int(r.firm_id), int(r.center_q)): int(r.max_horizon)
          for r in pool.itertuples(index=False)}

    # Enumerate candidate (firm, start_center, block_len, start_year) entries:
    # one per valid start position across all runs of all firms.
    cand_by_firm: dict[int, list[tuple[int, int, int]]] = {}
    for fid, g in pool.groupby("firm_id", sort=True):
        centers = g["center_q"].tolist()
        entries: list[tuple[int, int, int]] = []
        for run in _runs_for_firm(centers):
            L = len(run)
            if L < min_block:
                continue
            for i in range(0, L - min_block + 1):
                start = run[i]
                block_len = min(max_block, L - i)
                start_year = qmap.to_period(start).year
                entries.append((start, block_len, start_year))
        if entries:
            cand_by_firm[int(fid)] = entries

    eligible_firms = sorted(cand_by_firm.keys())
    if not eligible_firms:
        raise ValueError(f"No firm has a run of ≥ {min_block} consecutive "
                         f"eligible origins; lower --min-block.")

    rng = np.random.RandomState(seed)
    n_target = min(n_firms, len(eligible_firms))

    chosen: list[tuple[int, int, int]] = []  # (firm, start_center, block_len)

    if not calendar_balance:
        firms = rng.choice(eligible_firms, size=n_target, replace=False)
        for fid in sorted(int(f) for f in firms):
            entries = cand_by_firm[fid]
            start, block_len, _ = entries[rng.randint(len(entries))]
            chosen.append((fid, start, block_len))
    else:
        # Round-robin over start-years; at each step pick a still-unused firm
        # that has a candidate start in that year, and one such start. This
        # equalizes block-start years subject to availability.
        firms_by_year: dict[int, set[int]] = {}
        for fid, entries in cand_by_firm.items():
            for (_s, _bl, yr) in entries:
                firms_by_year.setdefault(yr, set()).add(fid)
        firms_by_year = {yr: sorted(fids) for yr, fids in firms_by_year.items()}
        years = sorted(firms_by_year.keys())
        used: set[int] = set()
        yi = 0
        guard = 0
        max_guard = len(years) * (n_target + 5) + 10
        while len(chosen) < n_target and guard < max_guard:
            guard += 1
            yr = years[yi % len(years)]
            yi += 1
            avail = [f for f in firms_by_year[yr] if f not in used]
            if not avail:
                continue
            fid = int(avail[rng.randint(len(avail))])
            used.add(fid)
            year_entries = [e for e in cand_by_firm[fid] if e[2] == yr]
            start, block_len, _ = year_entries[rng.randint(len(year_entries))]
            chosen.append((fid, start, block_len))
        # If years exhausted before n_target (small pools), top up uniformly.
        if len(chosen) < n_target:
            remaining = [f for f in eligible_firms if f not in used]
            rng.shuffle(remaining)
            for fid in remaining[: n_target - len(chosen)]:
                entries = cand_by_firm[fid]
                start, block_len, _ = entries[rng.randint(len(entries))]
                chosen.append((int(fid), start, block_len))

    chosen.sort()
    rows = []
    for fid, start, block_len in chosen:
        for c in range(start, start + block_len):
            rows.append((fid, c, mh[(fid, c)]))
    return pd.DataFrame(rows, columns=["firm_id", "center_q", "max_horizon"])


def select_noise_subset(sample: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Pick n origins uniformly at random for K-replicate noise probing.

    SIMPLE RANDOM — no horizon stratification. Noise is measured per forecast
    horizon h, and the model is delisting/calendar-blind (the prompt always asks
    Q1..Q20 and history carries no dates), so cross-call dispersion at horizon h
    does not depend on an origin's total available horizon. A uniform draw over
    the production sample is therefore both representative (it inherits the
    sample's calendar/lifespan mix) and unbiased per horizon: every origin that
    reaches horizon h contributes a replicate observation there. Stratifying by
    `max_horizon` (its former behaviour) would instead over-sample the low-
    available-horizon tail — i.e. late-sample and short-lived/delisting firms —
    injecting a survivorship/calendar skew the noise estimate doesn't want.

    Deterministic in (sample, n, seed). Returns exactly min(n, len(sample)) rows,
    in their original sample order."""
    sample = sample.reset_index(drop=True)
    if len(sample) <= n:
        return sample.copy()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(sample), size=n, replace=False)
    return sample.iloc[sorted(idx.tolist())].reset_index(drop=True)
