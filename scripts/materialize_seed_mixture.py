#!/usr/bin/env python
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

"""Keyed, row-order-safe seed-mixture materializer (Gaussian and Laplace arms).

Moment-matches an arm's N per-seed forecasts into ONE single-file forecast, the
form the point / GBM / LLM pools and the reconciliation pass key on:

    prediction = mean_k(mu_k)
    sigma      = sqrt( mean_k(sigma_k^2) + var_k(mu_k) )   # law of total variance

For the Gaussian and Laplace families `sigma` is the predictive standard deviation
(evaluate.py derives the Laplace scale as b = sigma / sqrt(2)), so this is the
exact mean and standard deviation of the equal-weight seed mixture for those arms.
For student_t, `sigma` is the t SCALE, not the std (std = scale * sqrt(df/(df-2))),
so the formula would be wrong: any family but gaussian and laplace is refused.
It is the formula of the research repo's earlier positional (chunked)
materializer and of R13's Laplace "chunked keyed moment-match". On row-aligned
Gaussian seeds this script's values are cell-for-cell identical, on the keys, to
the positional script's (the rows come out in another order; tested on synthetic
seeds). That the R13 Laplace mixture averaged in the same order is
NOT verified, hence the f32-rounding gate below.

Point metrics (R^2 / MAE) scored on the materialized file are exact: `prediction`
is the mixture mean. NLL / CRPS scored on it are a moment-matched single-density
approximation (one Gaussian / Laplace with the mixture's mean and std); the exact
mixture NLL / CRPS comes from the per-seed pool (evaluate.py's seed families,
scripts/mixture_calibration.py).

Why keyed: the positional materializer averages cells POSITIONALLY and hard-fails
when any batch's key columns differ across seeds. R13's Laplace seed 60 was
written in a different row order from seeds 61-64, so a positional average
cannot align it. Here every seed is first hash-partitioned by firm_id into
on-disk buckets (one streaming pass per seed, low memory). Each bucket is then
sorted on the key columns (firm_id, target, quarter, forecast_horizon) in every
seed, and the sorted keys must be identical across seeds: same key SET, no
duplicate keys, no null keys. Any difference is a hard error, and nothing is
written. Key TYPES may differ across seeds (R13's Gaussian seed 61 has string
firm_id and a timestamp[ns] quarter, its siblings large_string and
timestamp[us]): every key column is cast on read to one canonical type (firm_id,
target -> large_string, forecast_horizon -> int64), and a cast that would lose
information (a fractional horizon, say) is an error. The quarter key is its
calendar quarter, as in eval: an end-of-quarter stamp in any unit (train.py's
23:59:59.999999999 [ns] under pandas < 3, the forecasts' 23:59:59.999999 [us])
becomes the one canonical timestamp[us] 23:59:59.999999, and a quarter value that
is not an end-of-quarter stamp is an error.

The predictive-density sidecar travels along: if the seeds carry a
`{stem}.nll.json` (e.g. {"family": "laplace", "df": null}), every seed must
carry the same one, and the mixture gets a copy. Gaussian seeds have none, and
neither does their mixture (train.py's convention; absent means gaussian).
--expect-family {gaussian,laplace} fails an arm whose seeds resolve to another
density: a Laplace arm staged without its sidecars would otherwise mix, and then
score, as Gaussian.

Output: `<dest>/<per-seed name with the __<seed>__ token removed>`, e.g.
forma_lap05_fgrid__pf_full__60__test__predictions.parquet ->
forma_lap05_fgrid__pf_full__test__predictions.parquet. Columns firm_id, target,
quarter, forecast_horizon (the canonical key types), prediction (f32), sigma (f32),
model (the arm id, the name's first `__` token). Rows are grouped by bucket and
sorted by key within a bucket, NOT in the seeds' row order; every consumer joins
on the keys. Atomic writes (.tmp -> os.replace), the sidecar's BEFORE the
parquet's, so a new mixture never sits next to a missing or stale sidecar; a
Gaussian mixture deletes any stale `<out stem>.nll.json` at the destination. If
the parquet's replace then fails, the prior sidecar (or its absence) is put back,
so the old parquet is not left beside the new sidecar either.

Gate (--check-against FILE): compare the written mixture with an existing one
on the keys (the reference's key types are canonicalized the same way): the key
sets must match, and so must each value column's NaN pattern (NaN == NaN; a NaN
facing a number fails under any tolerance) and its infinities (an infinity facing
a number or the opposite infinity, or a finite reference value too large for
f32, fails the same way). Over the remaining rows it reports, per value column,
max |d| (in f64) and max_f32_ulp (the ulp distance between the output's f32 and
the reference ROUNDED to f32), plus the rows identical in f64 / in f32 and the
NaN / infinity mismatch counts. It passes iff both columns are within --max-f32-ulp N
ulps when that is given, else iff max |d| <= --tol (default 0: bit-identical).
The reference must also resolve to the same density as the output (its
.nll.json sidecar; both absent is a match), so a Laplace reference copied
without its sidecar fails. Use it to reproduce a published mixture from its
per-seed files before trusting the script on a new round. R13's Laplace mixture
(forma_lap05_fgrid__pf_full__test) is f64, written by a DuckDB GROUP BY with
string keys, and only a small share of its values is exactly f32-representable,
so its gate is f32-rounding equivalence (--max-f32-ulp 1: the two f64 means may
straddle an f32 rounding boundary), not bit identity.

Usage:
  python scripts/materialize_seed_mixture.py --src <store> --dest <store> \\
      --arm "forma_lap05_fgrid__pf_full__*__test__predictions.parquet" \\
      --arm "forma_fgrid__pf_full__*__test__predictions.parquet" --expect-seeds 5

  # R13 reproduction gate (Laplace; the reference is an f64 DuckDB mixture)
  python scripts/materialize_seed_mixture.py --src <r13 per-seed dir> --dest <scratch> \\
      --arm "forma_lap05_fgrid__pf_full__*__test__predictions.parquet" --expect-seeds 5 \\
      --expect-family laplace --max-f32-ulp 1 \\
      --check-against <r13 dir>/forma_lap05_fgrid__pf_full__test__predictions.parquet
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

KEYS = ["firm_id", "target", "quarter", "forecast_horizon"]
KEY_TYPES = {"firm_id": pa.large_string(), "target": pa.large_string(),
             "quarter": pa.timestamp("us"), "forecast_horizon": pa.int64()}
VALUES = ["prediction", "sigma"]
FAMILIES = ("gaussian", "laplace")
SEED_RE = re.compile(r"__(?:seed)?(\d+)__")


class MixtureError(RuntimeError):
    """Inputs cannot be mixed safely (misaligned keys, mixed families, ...)."""


def seed_of(name: str) -> int:
    m = SEED_RE.search(name)
    if not m:
        raise MixtureError(f"no __<seed>__ token in {name}")
    return int(m.group(1))


def mixture_name(name: str) -> str:
    """Per-seed file name -> mixture name (the first __<seed>__ token removed)."""
    return SEED_RE.sub("__", name, count=1)


def output_name(files: list[Path]) -> str:
    """The arm's mixture file name, which every seed file must map to."""
    names = {mixture_name(p.name) for p in files}
    if len(names) != 1:
        raise MixtureError(f"seed files do not share one arm name: {sorted(names)}")
    return names.pop()


def same_file(a: Path, b: Path) -> bool:
    """a and b name one file: os.path.samefile when both exist (catches hard links and
    case-folded Windows paths), else the normalized resolved paths."""
    if a.exists() and b.exists():
        return os.path.samefile(a, b)
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.nll.json")


def read_sidecar(path: Path) -> dict | None:
    side = sidecar_path(path)
    if not side.exists():
        return None
    with open(side, encoding="utf-8") as fh:
        return json.load(fh)


def density_of(side: dict | None) -> tuple[str, float | None]:
    """(family, df) as evaluate.py's _read_nll_meta resolves a sidecar: absent -> gaussian."""
    side = side or {}
    df = side.get("df")
    return str(side.get("family", "gaussian")).lower(), (float(df) if df is not None else None)


def canonical_quarter(col: pa.ChunkedArray | pa.Array, label: str) -> pa.ChunkedArray | pa.Array:
    """Quarter key -> its calendar quarter's one canonical end-of-quarter stamp, the
    timestamp[us] 23:59:59.999999 of the quarter's last day. The calendar quarter IS
    the key, as in eval (src/eval_cache.py canon_quarters): train.py's
    PeriodIndex.end_time writes 23:59:59.999999999 [ns] under pandas < 3 (R13's FFNN
    seeds and Gaussian seed 61), other forecasts 23:59:59.999999 [us]. A value is
    accepted iff it is the quarter's last tick in its own unit; anything else (a
    mid-quarter date, midnight, 23:59:59 in a sub-second unit) is an error."""
    if not pa.types.is_timestamp(col.type) or col.type.tz is not None:
        raise MixtureError(f"{label}: quarter must be a tz-naive timestamp, not {col.type}")
    unit = col.type.unit
    uniq = pc.unique(col)
    t = uniq.cast(pa.int64()).to_numpy()
    months = t.view(f"datetime64[{unit}]").astype("datetime64[M]").astype(np.int64)
    next_start = (months - months % 3 + 3).astype("datetime64[M]")
    bad = t + 1 != next_start.astype(f"datetime64[{unit}]").astype(np.int64)
    if bad.any():
        bad_values = uniq.filter(pa.array(bad))
        n_rows = pc.sum(pc.is_in(col, value_set=bad_values)).as_py()
        raise MixtureError(f"{label}: {n_rows:,} quarter value(s) are not end-of-quarter stamps (the "
                           f"quarter's last {unit} tick), e.g. {bad_values[0]}")
    eoq = pa.array(next_start.astype("datetime64[us]").astype(np.int64) - 1).cast(pa.timestamp("us"))
    return pc.take(eoq, pc.index_in(col, value_set=uniq))


def canonicalize_keys(tbl: pa.Table, label: str) -> pa.Table:
    """Cast the key columns to KEY_TYPES, so seeds whose keys are equal but typed
    differently (R13's Gaussian seed 61 has string firm_id and timestamp[ns] quarter,
    seeds 60/62-64 large_string and timestamp[us]) hash, sort and compare alike. The
    quarter goes to its canonical end-of-quarter stamp (canonical_quarter). The other
    casts are safe: a fractional horizon, say, is an error, never a silent truncation.
    A null key is an error too: the duplicate check and the key comparison cannot see
    through nulls."""
    for k, typ in KEY_TYPES.items():
        i = tbl.schema.get_field_index(k)
        col = tbl.column(i)
        if col.null_count:
            raise MixtureError(f"{label}: {col.null_count:,} null {k} key(s)")
        if k == "quarter":
            tbl = tbl.set_column(i, k, canonical_quarter(col, label))
            continue
        if col.type == typ:
            continue
        try:
            cast = col.cast(typ, safe=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
            raise MixtureError(f"{label}: cannot cast {k} from {col.type} to {typ} losslessly: {e}") from e
        tbl = tbl.set_column(i, k, cast)
    return tbl


def restore_file(path: Path, content: bytes | None) -> None:
    """Put a small file back as it was before: `content`, or absent when None."""
    if content is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(content)


def remove_tree(path: Path) -> None:
    """Delete a scratch dir without masking the exception (if any) that is unwinding."""
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        print(f"  WARNING: could not remove scratch dir {path}", file=sys.stderr, flush=True)


def bucket_ids(firm_id: pa.ChunkedArray | pa.Array, n_buckets: int) -> np.ndarray:
    """Deterministic firm_id -> bucket. pandas' hash_array uses a fixed key, so
    the same firm lands in the same bucket in every seed (and in --check-against)."""
    values = firm_id.to_pandas() if hasattr(firm_id, "to_pandas") else pd.Series(firm_id)
    return (pd.util.hash_array(np.asarray(values.astype(str))) % np.uint64(n_buckets)).astype(np.int64)


def partition(path: Path, out_dir: Path, n_buckets: int, batch_rows: int, columns: list[str]) -> int:
    """Stream one forecast into <out_dir>/b<k>.parquet buckets, key types made canonical
    before the firm hash. Returns rows written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(path)
    missing = [c for c in columns if c not in pf.schema_arrow.names]
    if missing:
        raise MixtureError(f"{path.name}: missing columns {missing}")
    writers: dict[int, pq.ParquetWriter] = {}
    rows = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            tbl = canonicalize_keys(pa.Table.from_batches([batch]), path.name)
            b = bucket_ids(tbl.column("firm_id"), n_buckets)
            order = np.argsort(b, kind="stable")
            tbl = tbl.take(pa.array(order))
            counts = np.bincount(b, minlength=n_buckets)
            start = 0
            for k in np.flatnonzero(counts):
                part = tbl.slice(start, int(counts[k]))
                start += int(counts[k])
                if k not in writers:
                    writers[k] = pq.ParquetWriter(out_dir / f"b{k}.parquet", part.schema)
                writers[k].write_table(part)
            rows += tbl.num_rows
    finally:
        for w in writers.values():
            w.close()
    return rows


def read_bucket(path: Path) -> pa.Table | None:
    """One bucket file (key types canonical), or None when that seed had no rows in the bucket."""
    return canonicalize_keys(pq.read_table(path), str(path)) if path.exists() else None


def sort_by_keys(tbl: pa.Table) -> pa.Table:
    return tbl.take(pc.sort_indices(tbl, sort_keys=[(k, "ascending") for k in KEYS]))


def assert_unique_keys(tbl: pa.Table, label: str) -> None:
    """On a key-sorted table, duplicates are adjacent: compare each row's keys to the next."""
    n = tbl.num_rows
    if n < 2:
        return
    same = None
    for k in KEYS:
        col = tbl.column(k)
        eq = pc.equal(col.slice(0, n - 1), col.slice(1, n - 1))
        same = eq if same is None else pc.and_(same, eq)
    if pc.any(same).as_py():
        raise MixtureError(f"{label}: duplicate (firm_id, target, quarter, forecast_horizon) keys")


def keys_equal(a: pa.Table, b: pa.Table) -> bool:
    return a.num_rows == b.num_rows and all(a.column(k).equals(b.column(k)) for k in KEYS)


def materialize_arm(files: list[Path], dest: Path, tmp_root: Path, n_buckets: int,
                    batch_rows: int, expect_seeds: int | None, expect_family: str | None = None) -> Path:
    files = sorted(files, key=lambda p: seed_of(p.name))
    seeds = [seed_of(p.name) for p in files]
    if len(set(seeds)) != len(seeds):
        raise MixtureError(f"duplicate seeds {seeds}")
    if len(files) < 2:
        raise MixtureError(f"need >= 2 seeds, got {seeds}")
    if expect_seeds is not None and len(files) != expect_seeds:
        raise MixtureError(f"expected {expect_seeds} seeds, got {len(files)}: {seeds}")
    out_name = output_name(files)
    arm_id = out_name.split("__", 1)[0]

    sides = [read_sidecar(p) for p in files]
    if any(s != sides[0] for s in sides):
        raise MixtureError(f"seeds disagree on the .nll.json density sidecar: {dict(zip(seeds, sides))}")
    family = density_of(sides[0])[0]
    if family not in FAMILIES:
        raise MixtureError(f"{arm_id}: density family {family!r} is not supported ({', '.join(FAMILIES)} "
                           f"only): the moment match needs sigma to be the predictive std, and a "
                           f"student_t sigma is the t scale")
    if expect_family is not None and family != expect_family:
        raise MixtureError(f"{arm_id}: the seeds' density is {family} "
                           f"({'per their .nll.json' if sides[0] else 'no .nll.json sidecar'}), "
                           f"expected {expect_family}")

    counts = [pq.ParquetFile(p).metadata.num_rows for p in files]
    if len(set(counts)) != 1:
        raise MixtureError(f"seed row counts differ: {dict(zip(seeds, counts))}")
    rows = counts[0]
    if rows == 0:
        raise MixtureError(f"seeds {seeds} have no rows")

    out = dest / out_name
    tmp_out = out.with_name(out.name + ".tmp")
    side = sidecar_path(out)
    tmp_side = side.with_name(side.name + ".tmp")
    work = tmp_root / f"_mixture_tmp__{out_name[:-len('.parquet')]}"
    if work.exists():
        shutil.rmtree(work)
    print(f"{arm_id}: seeds {seeds}, {rows:,} rows each, family {family}, {n_buckets} buckets -> {out}",
          flush=True)
    try:
        for s, p in zip(seeds, files):
            n = partition(p, work / f"seed{s}", n_buckets, batch_rows, KEYS + VALUES)
            print(f"  partitioned seed {s}: {n:,} rows", flush=True)

        writer = None
        done = 0
        try:
            for k in range(n_buckets):
                tables = [read_bucket(work / f"seed{s}" / f"b{k}.parquet") for s in seeds]
                if all(t is None for t in tables):
                    continue
                if any(t is None for t in tables):
                    raise MixtureError(f"bucket {k}: has rows in some seeds only")
                tables = [sort_by_keys(t) for t in tables]
                assert_unique_keys(tables[0], f"seed {seeds[0]} bucket {k}")
                for s, t in zip(seeds[1:], tables[1:]):
                    if not keys_equal(tables[0], t):
                        raise MixtureError(f"bucket {k}: seed {s}'s keys differ from seed {seeds[0]}'s "
                                           f"({t.num_rows:,} vs {tables[0].num_rows:,} rows)")
                mus = np.vstack([t.column("prediction").to_numpy().astype(np.float64) for t in tables])
                sig = np.vstack([t.column("sigma").to_numpy().astype(np.float64) for t in tables])
                pred = mus.mean(axis=0).astype(np.float32)
                sigma = np.sqrt((sig ** 2).mean(axis=0) + mus.var(axis=0)).astype(np.float32)
                base = tables[0]
                out_tbl = pa.table({
                    **{key: base.column(key) for key in KEYS},
                    "prediction": pa.array(pred),
                    "sigma": pa.array(sigma),
                    "model": pa.array([arm_id] * base.num_rows, type=pa.string()),
                })
                if writer is None:
                    writer = pq.ParquetWriter(tmp_out, out_tbl.schema, compression="zstd")
                writer.write_table(out_tbl)
                done += base.num_rows
        finally:
            if writer is not None:
                writer.close()
        if done != rows:
            raise MixtureError(f"wrote {done:,} rows != {rows:,} per seed")
        prior_side = side.read_bytes() if side.exists() else None
        try:
            if sides[0] is not None:
                with open(tmp_side, "w", encoding="utf-8") as fh:
                    json.dump(sides[0], fh)
                os.replace(tmp_side, side)
            else:
                side.unlink(missing_ok=True)
            os.replace(tmp_out, out)
        except BaseException:
            restore_file(side, prior_side)
            raise
    except BaseException:
        tmp_out.unlink(missing_ok=True)
        tmp_side.unlink(missing_ok=True)
        raise
    finally:
        remove_tree(work)
    print(f"  WROTE {out} ({done:,} rows){' + ' + side.name if sides[0] else ''}", flush=True)
    return out


def f32_ordinal(x: np.ndarray) -> np.ndarray:
    """float32 -> int64, monotonic in the value, so |ord(a) - ord(b)| is the number of
    f32 steps (ulps) from a to b. -0.0 and +0.0 both map to 0."""
    i = np.ascontiguousarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, np.int64(-2 ** 31) - i, i)


def column_diff(x: np.ndarray, y: np.ndarray) -> tuple[int, int, float, int, np.ndarray, np.ndarray]:
    """Output column x vs reference column y (any float dtypes). Returns
    - the number of rows NaN on one side only;
    - the number of infinity mismatches: rows (NaN on neither side) whose f64 values
      differ and where an infinity appears once both are rounded to f32 -- an infinity
      facing a number or the opposite infinity, or a finite f64 value too large for
      f32 (its f32 rounding is inf, which must not pass as 1 ulp from FLT_MAX);
    - over the remaining rows, max |x - y| in f64 and the max f32 ulp distance between
      x as f32 and y ROUNDED to f32 (equal infinities differ by 0);
    - the per-row masks "equal in f64" and "equal once both are f32", where NaN == NaN
      and an infinity mismatch is never equal."""
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        x32, y32 = x64.astype(np.float32), y64.astype(np.float32)
    nx, ny = np.isnan(x64), np.isnan(y64)
    ok = ~nx & ~ny
    inf_bad = ok & (np.isinf(x32) | np.isinf(y32)) & (x64 != y64)
    cmp = ok & ~inf_bad
    with np.errstate(invalid="ignore"):
        d = np.where(x64 == y64, 0.0, np.abs(x64 - y64))[cmp]
    ulp = np.abs(f32_ordinal(x32[cmp]) - f32_ordinal(y32[cmp]))
    return (int((nx != ny).sum()), int(inf_bad.sum()), float(d.max()) if d.size else 0.0,
            int(ulp.max()) if ulp.size else 0, (x64 == y64) | (nx & ny), ((x32 == y32) & ~inf_bad) | (nx & ny))


def check_against(out: Path, ref: Path, tmp_root: Path, n_buckets: int, batch_rows: int,
                  tol: float = 0.0, max_f32_ulp: int | None = None) -> dict:
    """Keyed comparison of a written mixture with a reference one (key types canonicalized
    on both sides, values compared in f64 and after rounding to f32). Passes iff the key
    sets match and, per value column, the NaN patterns match (NaN == NaN), no row has an
    infinity mismatch (column_diff) and, over the remaining rows, max f32 ulp <=
    max_f32_ulp when that is given, else
    max |d| <= tol; the two files must also carry the same density (.nll.json sidecar,
    resolved as evaluate.py does, so both absent is a match). Raises MixtureError otherwise."""
    side_ref = read_sidecar(ref)
    dens_out, dens_ref = density_of(read_sidecar(out)), density_of(side_ref)
    if dens_out != dens_ref:
        raise MixtureError(f"check-against: output density {dens_out} != reference {dens_ref} (from "
                           f"{sidecar_path(ref).name if side_ref else 'no sidecar: read as gaussian'})")
    n_out, n_ref = (pq.ParquetFile(p).metadata.num_rows for p in (out, ref))
    if n_out != n_ref:
        raise MixtureError(f"check-against: {n_out:,} rows vs reference {n_ref:,}")
    work = tmp_root / f"_mixture_check__{out.stem}"
    if work.exists():
        shutil.rmtree(work)
    worst = {c: 0.0 for c in VALUES}
    worst_ulp = {c: 0 for c in VALUES}
    nan_mismatch = {c: 0 for c in VALUES}
    inf_mismatch = {c: 0 for c in VALUES}
    exact = f32_exact = 0
    try:
        for name, path in (("out", out), ("ref", ref)):
            partition(path, work / name, n_buckets, batch_rows, KEYS + VALUES)
        for k in range(n_buckets):
            a = read_bucket(work / "out" / f"b{k}.parquet")
            b = read_bucket(work / "ref" / f"b{k}.parquet")
            if a is None and b is None:
                continue
            if a is None or b is None:
                raise MixtureError(f"check-against: bucket {k} present on one side only")
            a, b = sort_by_keys(a), sort_by_keys(b)
            if not keys_equal(a, b):
                raise MixtureError(f"check-against: bucket {k} key sets differ")
            same = np.ones(a.num_rows, dtype=bool)
            same32 = np.ones(a.num_rows, dtype=bool)
            for c in VALUES:
                n_nan, n_inf, max_d, max_ulp, eq, eq32 = column_diff(a.column(c).to_numpy(),
                                                                     b.column(c).to_numpy())
                nan_mismatch[c] += n_nan
                inf_mismatch[c] += n_inf
                worst[c] = max(worst[c], max_d)
                worst_ulp[c] = max(worst_ulp[c], max_ulp)
                same &= eq
                same32 &= eq32
            exact += int(same.sum())
            f32_exact += int(same32.sum())
    finally:
        remove_tree(work)
    report = {"rows": n_out,
              **{f"max_abs_d_{c}": worst[c] for c in VALUES},
              **{f"max_f32_ulp_{c}": worst_ulp[c] for c in VALUES},
              **{f"nan_mismatch_{c}": nan_mismatch[c] for c in VALUES},
              **{f"inf_mismatch_{c}": inf_mismatch[c] for c in VALUES},
              "rows_bit_identical": exact, "rows_f32_identical": f32_exact}
    print(f"  check-against {ref.name}: {report}", flush=True)
    problems = [f"{c}: {n:,} rows NaN on one side only" for c, n in nan_mismatch.items() if n]
    problems += [f"{c}: {n:,} rows with an infinity mismatch" for c, n in inf_mismatch.items() if n]
    if max_f32_ulp is not None:
        problems += [f"{c}: max f32 ulp {worst_ulp[c]} > --max-f32-ulp {max_f32_ulp}"
                     for c in VALUES if worst_ulp[c] > max_f32_ulp]
    else:
        problems += [f"{c}: max |d| {worst[c]:.3g} > --tol {tol:g}" for c in VALUES if worst[c] > tol]
    if problems:
        raise MixtureError("check-against: " + "; ".join(problems))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="dir holding the per-seed forecasts")
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True, metavar="GLOB",
                    help="per-seed file glob for one arm (repeat per arm)")
    ap.add_argument("--expect-seeds", type=int, default=None, help="fail unless each arm has exactly N seeds")
    ap.add_argument("--expect-family", choices=FAMILIES, default=None,
                    help="fail unless each arm's seeds carry this density (no .nll.json sidecar = gaussian)")
    ap.add_argument("--buckets", type=int, default=64, help="firm hash buckets (more = less RAM per bucket)")
    ap.add_argument("--batch-rows", type=int, default=8_000_000)
    ap.add_argument("--tmp-dir", type=Path, default=None, help="scratch for the buckets (default: --dest)")
    ap.add_argument("--check-against", type=Path, default=None,
                    help="existing mixture to compare with (only with a single --arm)")
    ap.add_argument("--tol", type=float, default=None,
                    help="max |d| (f64) allowed by --check-against (default 0: bit-identical)")
    ap.add_argument("--max-f32-ulp", type=int, default=None, metavar="N",
                    help="--check-against passes iff both value columns are within N f32 ulps of "
                         "the reference rounded to f32 (replaces --tol; for an f64 reference)")
    args = ap.parse_args(argv)
    if args.check_against is not None and len(args.arm) != 1:
        ap.error("--check-against needs exactly one --arm")
    if args.check_against is not None and not args.check_against.is_file():
        ap.error(f"--check-against {args.check_against}: no such file")
    if args.tol is not None and args.max_f32_ulp is not None:
        ap.error("--tol and --max-f32-ulp are alternative pass criteria; give one")
    if args.check_against is None and (args.tol is not None or args.max_f32_ulp is not None):
        ap.error("--tol / --max-f32-ulp only apply with --check-against")
    if args.tol is not None and not (math.isfinite(args.tol) and args.tol >= 0):
        ap.error(f"--tol must be finite and >= 0, not {args.tol}")
    if args.max_f32_ulp is not None and args.max_f32_ulp < 0:
        ap.error(f"--max-f32-ulp must be >= 0, not {args.max_f32_ulp}")

    args.dest.mkdir(parents=True, exist_ok=True)
    tmp_root = args.tmp_dir or args.dest
    src_files = sorted(p.name for p in args.src.iterdir() if p.suffix == ".parquet")
    try:
        for pat in args.arm:
            files = [args.src / f for f in fnmatch.filter(src_files, pat)]
            if not files:
                raise MixtureError(f"no files matching {pat!r} in {args.src}")
            if args.check_against is not None and same_file(args.check_against, args.dest / output_name(files)):
                raise MixtureError(f"--check-against {args.check_against} is the file this arm writes: it "
                                   f"would be overwritten, then compared with itself")
            out = materialize_arm(files, args.dest, tmp_root, args.buckets, args.batch_rows,
                                  args.expect_seeds, args.expect_family)
            if args.check_against is not None:
                check_against(out, args.check_against, tmp_root, args.buckets, args.batch_rows,
                              tol=args.tol or 0.0, max_f32_ulp=args.max_f32_ulp)
    except MixtureError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
