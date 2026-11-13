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

"""Tuple split panels: the natural key and the overlap-aware union.

The tuple splits written by `create_time_based_splits_int_quarter` overlap by
design -- val and test carry `pre_quarters_included` quarters of lead-in
history so their first origins get a full lookback window (#161). Anything
that pools two splits into one panel must therefore deduplicate on the tuple
key, or every overlap tuple lands twice (#279). This module is the one place
that logic lives; `src/forma/data/forma_data.py` (the `no_cv` training union)
imports it.

Deliberately pandas/numpy only: nothing here needs torch, so the logic stays
usable from plain scripts and pandas-only environments (`forma.data` imports
nothing eagerly, so `from forma.data.tuple_splits import ...` stays torch-free).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Natural key of a tuple row. A training panel must be unique on it: a
# duplicated key becomes two nodes in the same graph (double loss weight plus
# a copy-the-answer shortcut when only one twin is masked).
TUPLE_KEY = ["firm_id", "account_id", "quarter"]


def _validate_key_columns(df: pd.DataFrame, key: Sequence[str], which: str) -> None:
    """Key columns must be integer-typed (numpy or nullable Int) with no nulls.

    Checked over the whole frame, before any banding, so the contract holds
    whether or not the frames overlap. A float key would be truncated by the
    int64 cast and could merge distinct keys; an object key (e.g. a gvkey
    string) cannot be packed at all -- each gets its own message.
    """
    for c in key:
        s = df[c]
        if not pd.api.types.is_integer_dtype(s.dtype):
            raise ValueError(f"tuple key column {c!r} in {which} has non-integer dtype "
                             f"{s.dtype}; refusing to pack it for dedup")
        n_null = int(s.isna().sum())
        if n_null:
            raise ValueError(f"tuple key column {c!r} in {which} contains {n_null:,} null(s); "
                             f"refusing to union")
        if s.dtype.kind == "u" and len(s) and int(s.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"tuple key column {c!r} in {which} holds unsigned values above "
                             f"int64.max, which would wrap negative on the packing cast; "
                             f"refusing to union")


def _key_widths(frames_rows: Sequence[tuple[pd.DataFrame, np.ndarray]], key: Sequence[str]) -> list[int]:
    """Bit width per key column, sized from the maximum over ALL frames' selected rows.

    The widths must be shared by every frame that is packed for one comparison:
    sizing them per frame would let two frames with different column maxima
    pack the same key to different codes -- and distinct keys to the same code.
    """
    widths = []
    for c in key:
        m = 0
        for df, rows in frames_rows:
            arr = df[c].to_numpy(dtype=np.int64)[rows]
            if arr.size:
                if arr.min() < 0:
                    raise ValueError(f"tuple key column {c!r} contains negative ids; refusing to union")
                m = max(m, int(arr.max()))
        widths.append(max(m.bit_length(), 1))
    if sum(widths) > 63:
        raise ValueError(f"tuple key {list(key)} needs {sum(widths)} bits; cannot pack for dedup")
    return widths


def _packed_keys(df: pd.DataFrame, rows: np.ndarray, key: Sequence[str],
                 widths: Sequence[int]) -> np.ndarray:
    """One int64 per selected row encoding the key columns, for cheap membership tests.

    `widths` must come from `_key_widths` over every frame taking part in the
    comparison, so the packing is identical across frames and, with the 63-bit
    guard there, injective on distinct keys.
    """
    n = int(rows.sum())
    packed = np.zeros(n, dtype=np.int64)
    if not n:
        return packed
    shift = 0
    for c, width in zip(reversed(list(key)), reversed(list(widths))):
        arr = df[c].to_numpy(dtype=np.int64)[rows]
        if int(arr.max()) >= (1 << width):
            raise ValueError(f"tuple key column {c!r} exceeds its packing width; widths must "
                             f"be computed over every frame being compared")
        packed |= arr << shift
        shift += width
    return packed


def _values_agree(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise equality that treats a missing value on both sides as agreement.

    Missingness is decided first with `pd.isna` (NaN, NaT, None and `pd.NA`),
    and `==` is evaluated only where both sides are present -- comparing first
    would put `pd.NA` into the result for masked `boolean` / `string` columns
    and fail on the cast to bool.
    """
    a_na = np.asarray(pd.isna(a), dtype=bool)
    b_na = np.asarray(pd.isna(b), dtype=bool)
    eq = a_na & b_na
    sel = ~(a_na | b_na)
    if sel.any():
        eq[sel] = np.asarray(a[sel] == b[sel], dtype=bool)
    return eq


def union_tuple_splits(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    key: Sequence[str] = TUPLE_KEY,
    label: str = "train+val",
    verify_values: bool = True,
) -> pd.DataFrame:
    """Union two tuple split frames into one panel with no key shared across them.

    The result is unique on `key` provided each split is (per-split uniqueness
    is enforced only within the shared quarter band -- see below).

    Rows of `second` whose key already appears in `first` are dropped *before*
    the concat, so the concat produces the final panel and there is no second
    full copy of the union. Measured on the 57.1 M-row canonical train+val
    union (peak RSS over the resident inputs): plain `pd.concat` +2.11 GB;
    `concat` + `duplicated()` + boolean take +9.40 GB; this form +3.05 GB. Only
    rows in the quarter band both frames share can collide, so the key work is
    confined to that band (11 quarters, ~5.7 M rows of each frame on the
    canonical build).

    Semantics are the same as `drop_duplicates(key, keep='first')` on the
    concat, provided each split is itself unique on the key -- which the
    splitters guarantee (quarter/firm masks over one `tuple_df`). That
    precondition is enforced within the shared quarter band (a repeated key
    there raises); a split that repeats a key entirely outside the band is
    passed through as-is, since nothing this function does could touch it. The
    dedup is by key, not by quarter boundary: under the firm-based splitter
    train and val share the whole quarter range with disjoint firms, so the
    boundary filter would silently empty val while this is a harmless no-op.

    With `verify_values` (default) the dropped rows are checked against their
    `first` partners on every shared non-key column and a disagreement raises
    -- "assert, don't silently pick a side". Under the canonical splitter the copies are
    byte-identical; a per-split transform applied to only one file would fail
    loudly here instead of training on whichever copy sorted first.

    Only for unions. Single-split readers must not dedup or drop lead-in rows:
    they are the history of the first origins in the split, so trimming the test
    frame would truncate those windows. (The test origin index itself is floored
    separately, by `FormaDataModule.first_test_origin_q`.)
    """
    key = list(key)
    missing = [c for c in key if c not in first.columns or c not in second.columns]
    if missing:
        raise KeyError(f"tuple key column(s) {missing} missing from the split frames")
    _validate_key_columns(first, key, "first")
    _validate_key_columns(second, key, "second")

    n_first, n_second = len(first), len(second)
    drop = np.zeros(n_second, dtype=bool)
    n_dropped = 0

    if n_first and n_second:
        if "quarter" in key:
            fq = first["quarter"].to_numpy()
            sq = second["quarter"].to_numpy()
            lo, hi = max(fq.min(), sq.min()), min(fq.max(), sq.max())
            f_band = (fq >= lo) & (fq <= hi)
            s_band = (sq >= lo) & (sq <= hi)
        else:
            f_band = np.ones(n_first, dtype=bool)
            s_band = np.ones(n_second, dtype=bool)

        if f_band.any() and s_band.any():
            widths = _key_widths([(first, f_band), (second, s_band)], key)
            k_first = _packed_keys(first, f_band, key, widths)
            k_second = _packed_keys(second, s_band, key, widths)
            # Each split must itself be unique on the key, or "drop second's
            # rows already in first" cannot make the union unique. Check it
            # where it is cheap (the shared band; ~1.8 s on the canonical 5.7 M
            # rows) rather than certify a property that may not hold.
            for which, k in (("first", k_first), ("second", k_second)):
                n_dup = k.size - np.unique(k).size
                if n_dup:
                    raise ValueError(
                        f"[{label} union] the {which} split carries {n_dup:,} rows that repeat "
                        f"a {key} key within the shared quarter band; the union cannot make "
                        f"it unique -- fix the split file")
            hit = np.isin(k_second, k_first)
            s_rows = np.flatnonzero(s_band)[hit]
            drop[s_rows] = True
            n_dropped = int(hit.sum())

            if n_dropped and verify_values:
                order = np.argsort(k_first, kind="stable")
                k_sorted = k_first[order]
                pos = np.searchsorted(k_sorted, k_second[hit])
                f_rows = np.flatnonzero(f_band)[order[pos]]
                for col in first.columns:
                    if col in key or col not in second.columns:
                        continue
                    a = first[col].to_numpy()[f_rows]
                    b = second[col].to_numpy()[s_rows]
                    n_bad = int((~_values_agree(a, b)).sum())
                    if n_bad:
                        raise ValueError(
                            f"[{label} union] {n_bad:,} of {n_dropped:,} overlapping tuples "
                            f"disagree on {col!r} between the two splits; the split files "
                            f"are not copies of one panel -- refusing to pick a side")

    if n_dropped:
        union = pd.concat([first, second.loc[~drop]], ignore_index=True)
        dq = second["quarter"].to_numpy()[drop] if "quarter" in key else None
        span = f" (quarters {int(dq.min())}..{int(dq.max())})" if dq is not None else ""
        n_before = n_first + n_second
        print(f"[{label} union] dropped {n_dropped:,} duplicate tuple rows "
              f"({n_dropped / n_before:.2%} of {n_before:,}) from the split overlap{span}; "
              f"{len(union):,} rows remain")
    else:
        union = pd.concat([first, second], ignore_index=True)
        print(f"[{label} union] {len(union):,} tuples, no overlap between splits")
    return union
