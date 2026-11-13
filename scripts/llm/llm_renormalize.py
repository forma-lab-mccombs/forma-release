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

"""Re-normalize archived LLM-benchmark forecasts onto another dataset build.

The LLM benchmark (scripts/llm/llm_benchmark.py) shows the model RAW USD
history and gets RAW USD forecasts back. The regularized `prediction` that
evaluate.py scores is computed afterwards, in `finish_origin`, as

    prediction = reg_value(pred_raw, scale_q0, k, mu, sigma)

where `scale_q0` is the tuple `scale` account at the origin quarter and
`(k, mu, sigma)` are Forma's regularization stats for the target at the origin
quarter. A rebuilt dataset (new k target, new reg stats, more rows) therefore
needs no new API calls: re-apply that same map, with the new build's artifacts,
to the archived `pred_raw`.

Do NOT use `llm_benchmark.py --resume --dataset <new tag>` for this. Resume
reloads the CSV's already-normalized `prediction` column and re-emits it
unchanged, so it costs $0 and silently keeps the old build's numbers.

Fidelity:
  * The math is the benchmark's own: `reg_value`, `RegStatLookup` (including
    its nearest-quarter fallback) and `TupleSource` for `scale_q0`, imported
    from scripts/llm/, not re-implemented.
  * The archived `*_predictions.csv` is the input. The tuple source builds a
    DENSE quarter grid, so the CSV already holds a row for every
    (origin, target, horizon <= the origin's max_horizon) that `finish_origin`
    scored, gaps included. Horizons the LLM returned past the pinned
    max_horizon are not in the CSV, so this is complete only while the target
    build's sample ends at TEST_END_Q (2024Q4), which holds for R14. Nothing is
    re-parsed from the completions.
  * Every CSV row is carried over. A row is dropped only when the target build
    cannot normalize it (origin absent or without a valid scale, target without
    reg stats, non-finite `pred_raw`), and every drop is counted by reason in
    the report.
  * A gvkey missing from the build's firm_id_map is treated as missing. The
    benchmark's `TupleSource.int_of` would fall back to reading the gvkey as a
    dense firm id, i.e. another firm's data.

Acceptance gate: run against the build the CSVs were produced on (R13) with
`--check-against <published parquets> --require-match`. The output must
reproduce the published parquets key for key, within `--tol`, and reproduce
the CSV's own `prediction` column. Only then point it at the new build.

    python scripts/llm/llm_renormalize.py \
        --csv <llm archive>/*_predictions.csv \
        --processed-dir data/processed --dataset-tag r13_node_optionD_indfe_val8 \
        --out-dir <scratch>/renorm_r13 \
        --check-against <llm archive> --require-match

and then, for the camera-ready LLM column, the same CSVs onto the R14 build:

    python scripts/llm/llm_renormalize.py \
        --csv <llm archive>/*_predictions.csv \
        --processed-dir data/processed --dataset-tag r14_node_optionD_indfe_val8 \
        --out-dir <forecasts>/llm_r14

Outputs, per CSV `llm_{tag}_predictions.csv`:
`<out-dir>/llm_{tag}__{feature_set}__test__predictions.parquet` (the published
schema: firm_id, target, quarter, forecast_horizon, prediction), plus one JSON
report for the whole run (`--report`, default `<out-dir>/renormalize_report.json`).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The harness modules are siblings in scripts/llm/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_benchmark as bench  # noqa: E402
from llm_tuple_source import TupleSource  # noqa: E402

KEY = ["firm_id", "target", "quarter", "forecast_horizon"]
OUT_COLS = KEY + ["prediction"]
CSV_SUFFIX = "_predictions.csv"
DROP_REASONS = ("firm_not_in_build", "no_valid_scale", "no_reg_stats", "pred_raw_nonfinite")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_name_for(csv_path: Path, feature_set: str) -> str:
    """llm_{tag}_predictions.csv -> llm_{tag}__{feature_set}__test__predictions.parquet,
    the name llm_benchmark.output_paths gives the same run."""
    name = csv_path.name
    if not (name.startswith("llm_") and name.endswith(CSV_SUFFIX)):
        raise ValueError(f"{name}: expected an archived 'llm_<tag>{CSV_SUFFIX}' file")
    tag = name[: -len(CSV_SUFFIX)]
    return f"{tag}__{feature_set}__test__predictions.parquet"


class Build:
    """The artifacts of one dataset build that the normalization reads."""

    def __init__(self, processed_dir: Path, feature_set: str, dataset_tag: str):
        pdir = Path(processed_dir)
        self.paths = {
            "reg_stats": pdir / f"regularization_stats__{feature_set}__{dataset_tag}.parquet",
            "tuple_test": pdir / f"tuple_test__{feature_set}__{dataset_tag}.parquet",
            "account_id_map": pdir / "account_id_map.csv",
            "firm_id_map": pdir / "firm_id_map.csv",
        }
        missing = [str(p) for p in self.paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("build artifacts not found: " + ", ".join(missing))
        self.dataset_tag = dataset_tag
        self.feature_set = feature_set

        stats = pd.read_parquet(self.paths["reg_stats"], engine="fastparquet")
        self.reg = bench.RegStatLookup(stats)
        # (feature, quarter Period) pairs present verbatim; anything else is served
        # by RegStatLookup's nearest-quarter fallback, which the report counts.
        self._exact_stats = set(zip(
            stats["feature"].astype(str),
            pd.PeriodIndex(pd.to_datetime(stats["quarter"]), freq="Q")))
        self.tsrc = TupleSource(
            self.paths["tuple_test"], self.paths["account_id_map"],
            firm_map_path=self.paths["firm_id_map"],
            max_lookback=bench.TUPLE_HISTORY_QS, max_horizon=bench.MAX_HORIZON,
            min_lookback=bench.MIN_LOOKBACK,
            test_start_q=bench.TEST_START_Q, test_end_q=bench.TEST_END_Q,
        )
        self._scale_cache: dict[tuple[int, pd.Timestamp], tuple[str | None, float | None]] = {}
        self._reg_cache: dict[tuple[str, pd.Timestamp], tuple | None] = {}

    def provenance(self) -> dict:
        return {"dataset_tag": self.dataset_tag, "feature_set": self.feature_set,
                **{k: {"path": str(p), "md5": md5sum(p)} for k, p in self.paths.items()}}

    def scale_at(self, gvkey: int, q0: pd.Timestamp) -> tuple[str | None, float | None]:
        """(drop_reason, scale_q0) for an origin, as prepare_origin reads it."""
        key = (gvkey, q0)
        if key not in self._scale_cache:
            ts = self.tsrc
            if ts.gvkey_to_int is None or gvkey not in ts.gvkey_to_int \
                    or ts.gvkey_to_int[gvkey] not in ts.lookup:
                self._scale_cache[key] = ("firm_not_in_build", None)
            else:
                center_q = ts.qmap.ts_to_q(q0)
                # max_horizon does not affect scale_q0 (it is read at center_q).
                _, scale_q0, _ = ts.build_window(gvkey, center_q, 0)
                if scale_q0 is None or not np.isfinite(scale_q0) or scale_q0 <= 0:
                    self._scale_cache[key] = ("no_valid_scale", None)
                else:
                    self._scale_cache[key] = (None, float(scale_q0))
        return self._scale_cache[key]

    def stats_for(self, target: str, q0: pd.Timestamp):
        key = (target, q0)
        if key not in self._reg_cache:
            self._reg_cache[key] = self.reg.get(target, q0)
        return self._reg_cache[key]

    def is_exact_stat(self, target: str, q0: pd.Timestamp) -> bool:
        return (target, q0.to_period("Q")) in self._exact_stats


def read_archived_csv(path: Path) -> pd.DataFrame:
    """The archived predictions CSV, floats parsed round-trip exact (the benchmark
    writes them with pandas' shortest-repr formatter)."""
    df = pd.read_csv(
        path,
        usecols=KEY + ["prediction", "pred_raw"],
        dtype={"firm_id": "int64", "target": str, "forecast_horizon": "int64",
               "prediction": "float64", "pred_raw": "float64"},
        parse_dates=["quarter"],
        float_precision="round_trip",
    )
    dup = int(df.duplicated(KEY).sum())
    if dup:
        raise ValueError(f"{path.name}: {dup:,} duplicated {KEY} rows; refusing to guess which to keep")
    return df


def renormalize(df: pd.DataFrame, build: Build) -> tuple[pd.DataFrame, dict]:
    """Recompute `prediction` from `pred_raw` with the build's artifacts.

    Returns (rows kept, with prediction_new / prediction_csv), stats."""
    n = len(df)
    new = np.full(n, np.nan)
    reason = np.full(n, None, dtype=object)
    fallback = np.zeros(n, dtype=bool)
    firm = df["firm_id"].to_numpy()
    tgt = df["target"].to_numpy()
    q0s = df["quarter"].to_numpy()
    raw = df["pred_raw"].to_numpy()
    for i in range(n):
        gv, q0 = int(firm[i]), pd.Timestamp(q0s[i])
        why, scale = build.scale_at(gv, q0)
        if why is not None:
            reason[i] = why
            continue
        rs = build.stats_for(tgt[i], q0)
        if rs is None:
            reason[i] = "no_reg_stats"
            continue
        v = float(raw[i])
        if not math.isfinite(v):
            reason[i] = "pred_raw_nonfinite"
            continue
        k, mu, sigma = rs
        new[i] = bench.reg_value(v, scale, k, mu, sigma)
        fallback[i] = not build.is_exact_stat(tgt[i], q0)
    keep = pd.isna(reason)
    origins = df[["firm_id", "quarter"]].drop_duplicates()
    out = df.loc[keep, KEY].copy()
    out["prediction"] = new[keep]
    out["prediction_csv"] = df.loc[keep, "prediction"].to_numpy()
    if out["prediction"].isna().any():
        raise AssertionError("reg_value returned NaN on a row that passed every guard")
    stats = {
        "rows_in": n,
        "rows_out": int(keep.sum()),
        "origins_in": int(len(origins)),
        "origins_out": int(len(out[["firm_id", "quarter"]].drop_duplicates())),
        "dropped": {r: int((reason == r).sum()) for r in DROP_REASONS},
        "rows_reg_stats_nearest_quarter_fallback": int(fallback[keep].sum()),
        "rows_clipped_at_max_abs_zscore": int((np.abs(new[keep]) >= bench.MAX_ABS_ZSCORE).sum()),
    }
    return out, stats


def diff_summary(a: np.ndarray, b: np.ndarray) -> dict:
    """|a - b| over the rows where neither side is NaN; NaN rows are counted in
    `n_nan` (a NaN max would never exceed a tolerance, so it must not hide there)."""
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    nan = np.isnan(a) | np.isnan(b)
    a_ok, b_ok = a[~nan], b[~nan]
    d = np.where(a_ok == b_ok, 0.0, np.abs(a_ok - b_ok))
    q = np.quantile(d, [0.5, 0.99, 0.999]) if len(d) else [np.nan] * 3
    return {
        "n_compared": int(len(a)),
        "n_nan": int(nan.sum()),
        "n_exact": int((a == b).sum()),
        "max_abs": float(d.max()) if len(d) else 0.0,
        "p50_abs": float(q[0]), "p99_abs": float(q[1]), "p999_abs": float(q[2]),
    }


def compare_to_reference(out: pd.DataFrame, ref_path: Path) -> dict:
    ref = pd.read_parquet(ref_path, engine="fastparquet")
    ref = ref.astype({"firm_id": "int64", "forecast_horizon": "int64"})
    ref["quarter"] = pd.to_datetime(ref["quarter"])
    dup = int(ref.duplicated(KEY).sum())
    m = out[KEY + ["prediction"]].merge(
        ref[KEY + ["prediction"]].rename(columns={"prediction": "prediction_ref"}),
        on=KEY, how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    res = {
        "path": str(ref_path), "md5": md5sum(ref_path),
        "rows_reference": int(len(ref)), "reference_duplicate_keys": dup,
        "only_in_output": int((m["_merge"] == "left_only").sum()),
        "only_in_reference": int((m["_merge"] == "right_only").sum()),
    }
    res.update(diff_summary(both["prediction"].to_numpy(), both["prediction_ref"].to_numpy()))
    return res


def write_parquet(out: pd.DataFrame, path: Path) -> None:
    frame = out[OUT_COLS].reset_index(drop=True).astype(
        {"firm_id": "int64", "target": object, "forecast_horizon": "int64", "prediction": "float64"})
    frame["quarter"] = pd.to_datetime(frame["quarter"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(tmp, index=False, engine="fastparquet")
    back = pd.read_parquet(tmp, engine="fastparquet")
    if len(back) != len(frame) or not np.array_equal(
            back["prediction"].to_numpy(), frame["prediction"].to_numpy()):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"read-back of {tmp} does not match what was written")
    tmp.replace(path)


def gate_failures(cell: dict, tol: float) -> list[str]:
    fails = []
    if cell["rows_out"] != cell["rows_in"]:
        fails.append(f"dropped {cell['rows_in'] - cell['rows_out']:,} rows {cell['dropped']}")
    if cell["vs_archived_csv"]["n_nan"]:
        fails.append(f"{cell['vs_archived_csv']['n_nan']:,} row(s) NaN in new or csv prediction")
    if cell["vs_archived_csv"]["max_abs"] > tol:
        fails.append(f"max |new - csv prediction| = {cell['vs_archived_csv']['max_abs']:.3g} > tol {tol:g}")
    ref = cell.get("vs_reference")
    if ref is None:
        fails.append("no reference parquet to check against")
    else:
        if ref["only_in_output"] or ref["only_in_reference"] or ref["reference_duplicate_keys"]:
            fails.append(f"key sets differ (only_in_output={ref['only_in_output']:,}, "
                         f"only_in_reference={ref['only_in_reference']:,}, "
                         f"reference_duplicate_keys={ref['reference_duplicate_keys']:,})")
        if ref["n_nan"]:
            fails.append(f"{ref['n_nan']:,} row(s) NaN in new or reference prediction")
        if ref["max_abs"] > tol:
            fails.append(f"max |new - reference| = {ref['max_abs']:.3g} > tol {tol:g}")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                    help="archived llm_<tag>_predictions.csv files (globs allowed)")
    ap.add_argument("--processed-dir", required=True,
                    help="dir holding the build's tuple_test / reg stats / id maps")
    ap.add_argument("--dataset-tag", required=True)
    ap.add_argument("--feature-set", default="pf_full")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check-against", default=None,
                    help="dir of parquets to compare with (same file names), e.g. the published ones")
    ap.add_argument("--require-match", action="store_true",
                    help="exit 1 unless every row is kept and matches the CSV prediction and "
                         "the --check-against parquet within --tol (the same-build gate)")
    ap.add_argument("--tol", type=float, default=1e-12,
                    help="max |delta| allowed by --require-match (default 1e-12)")
    ap.add_argument("--report", default=None, help="JSON report path (default <out-dir>/renormalize_report.json)")
    args = ap.parse_args(argv)
    if args.require_match and not args.check_against:
        ap.error("--require-match needs --check-against")

    csvs = sorted({Path(p) for pat in args.csv for p in (glob.glob(pat) or [pat])})
    for p in csvs:
        if not p.exists():
            ap.error(f"not found: {p}")
    out_dir = Path(args.out_dir)
    report_path = Path(args.report) if args.report else out_dir / "renormalize_report.json"

    print(f"Loading build {args.dataset_tag} from {args.processed_dir} ...")
    build = Build(Path(args.processed_dir), args.feature_set, args.dataset_tag)
    report = {"build": build.provenance(), "tol": args.tol,
              "require_match": args.require_match, "cells": []}

    all_fails = {}
    for csv_path in csvs:
        name = parquet_name_for(csv_path, args.feature_set)
        print(f"\n[{csv_path.name}]")
        df = read_archived_csv(csv_path)
        out, stats = renormalize(df, build)
        cell = {"csv": {"path": str(csv_path), "md5": md5sum(csv_path)}, **stats}
        cell["vs_archived_csv"] = diff_summary(out["prediction"].to_numpy(), out["prediction_csv"].to_numpy())
        if args.check_against:
            ref_path = Path(args.check_against) / name
            if ref_path.exists():
                cell["vs_reference"] = compare_to_reference(out, ref_path)
            else:
                print(f"  WARN: no reference {ref_path}")
        out_path = out_dir / name
        write_parquet(out, out_path)
        cell["out"] = {"path": str(out_path), "md5": md5sum(out_path)}
        report["cells"].append(cell)

        print(f"  rows {stats['rows_in']:,} -> {stats['rows_out']:,}; origins "
              f"{stats['origins_in']:,} -> {stats['origins_out']:,}; dropped {stats['dropped']}")
        v = cell["vs_archived_csv"]
        print(f"  vs archived CSV prediction: max|d|={v['max_abs']:.3g}, "
              f"exact {v['n_exact']:,}/{v['n_compared']:,}, NaN {v['n_nan']:,}")
        if "vs_reference" in cell:
            r = cell["vs_reference"]
            print(f"  vs reference parquet: max|d|={r['max_abs']:.3g}, exact {r['n_exact']:,}/{r['n_compared']:,}, "
                  f"NaN {r['n_nan']:,}, only_in_output={r['only_in_output']:,}, only_in_reference={r['only_in_reference']:,}")
        print(f"  wrote {out_path}")
        if args.require_match:
            fails = gate_failures(cell, args.tol)
            if fails:
                all_fails[csv_path.name] = fails

    report["gate"] = ({"passed": not all_fails, "failures": all_fails}
                      if args.require_match else None)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {report_path}")
    if args.require_match:
        if all_fails:
            print("GATE FAIL:")
            for cell_name, fails in all_fails.items():
                for f in fails:
                    print(f"  {cell_name}: {f}")
            return 1
        print(f"GATE PASS: {len(csvs)} cell(s) reproduce the reference within tol {args.tol:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
