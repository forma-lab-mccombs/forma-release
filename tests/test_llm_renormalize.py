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

"""Tests for scripts/llm/llm_renormalize.py.

The script moves the paid LLM forecasts onto a rebuilt dataset without new API
calls (it produced the camera-ready LLM column: the archived R13-prompt pred_raw,
re-normalized onto the R14 build), so the failure modes worth pinning are silent
ones: a re-normalization that quietly reuses the old build's values, reads
another firm's scale, or drops rows without saying so. Everything runs on a tiny
synthetic build written to tmp; no data, network or credentials. Adapted from the
research repo's test_llm_renormalize.py (its prompt-drift-audit half is not
ported: that script is not part of this release).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "llm"))  # the harness lives under scripts/llm

# The harness imports httpx at module level, which ships in the [llm] extra.
pytest.importorskip("httpx", reason="LLM harness tests need the [llm] extra")

import llm_benchmark as bench  # noqa: E402
from llm_tuple_source import QuarterMap  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "llm" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


renorm = _load("llm_renormalize")

QMAP = QuarterMap()
TAG = "synth"
ACCOUNTS = {"revtq": 0, "cogsq": 1, "niq": 2, "scale": 3}
# gvkey -> dense firm id. gvkey 1 is deliberately NOT in the map while dense id 1
# exists (gvkey 2000): the benchmark's TupleSource.int_of would silently read
# that firm's data.
FIRMS = {1000: 0, 2000: 1}
ORIGIN_Q = [161, 162]          # 2010Q1, 2010Q2


def _qend(q):
    return QMAP.to_qend_ts(q)


def write_build(d: Path, *, k=0.5, mu_shift=0.0, scale_mult=1.0, revtq_bump=0.0,
                industry=(5, 5), drop_scale_for=None, drop_niq_stats_q=None, drop_cells=(),
                scale_override=None):
    """A minimal processed dir: tuple_test, reg stats, account/firm/industry maps.

    `drop_cells`: (gvkey, account, quarter) tuples left out of tuple_test.
    `scale_override`: {(gvkey, quarter): scale value} written in place of the default."""
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"account_name": list(ACCOUNTS), "account_id": list(ACCOUNTS.values())}) \
        .to_csv(d / "account_id_map.csv", index=False)
    pd.DataFrame({"firm_id": list(FIRMS), "firm_id_int": list(FIRMS.values())}) \
        .to_csv(d / "firm_id_map.csv", index=False)
    pd.DataFrame({"industry_name": ["Agric", "Food", "Soda", "Beer", "Smoke", "Toys"],
                  "industry_id": [0, 1, 2, 3, 4, 5]}).to_csv(d / "industry_id_map.csv", index=False)
    rows = []
    for gv, fid in FIRMS.items():
        for q in range(150, 183):
            base = 100.0 * (fid + 1) + q
            vals = {"revtq": base + revtq_bump, "cogsq": 0.6 * base, "niq": 0.1 * base,
                    "scale": (scale_override or {}).get((gv, q), (50.0 * (fid + 1) + q) * scale_mult)}
            for acct, v in vals.items():
                if (acct == "scale" and drop_scale_for == (gv, q)) or (gv, acct, q) in drop_cells:
                    continue
                rows.append((fid, ACCOUNTS[acct], q, float(v), industry[fid]))
    pd.DataFrame(rows, columns=["firm_id", "account_id", "quarter", "value", "industry_id"]) \
        .to_parquet(d / f"tuple_test__pf_full__{TAG}.parquet", index=False)
    stats = []
    for feat, kf in (("revtq", k), ("cogsq", k), ("niq", 2 * k)):
        for q in range(150, 183):
            if feat == "niq" and drop_niq_stats_q == q:
                continue
            stats.append({"feature": feat, "quarter": _qend(q) + pd.Timedelta("23:59:59.999"),
                          "mu": 0.01 * q + mu_shift, "sigma": 0.5 + 0.001 * q, "k": kf})
    pd.DataFrame(stats).to_parquet(d / f"regularization_stats__pf_full__{TAG}.parquet", index=False)
    return d


def expected_prediction(build_dir: Path, gv, q, target, pred_raw):
    """reg_value with the build's artifacts, computed independently of the script."""
    stats = pd.read_parquet(build_dir / f"regularization_stats__pf_full__{TAG}.parquet")
    row = stats[(stats.feature == target)
                & (pd.PeriodIndex(pd.to_datetime(stats.quarter), freq="Q") == _qend(q).to_period("Q"))]
    tup = pd.read_parquet(build_dir / f"tuple_test__pf_full__{TAG}.parquet")
    scale = tup[(tup.firm_id == FIRMS[gv]) & (tup.account_id == ACCOUNTS["scale"]) & (tup.quarter == q)].value
    k, mu, sg = float(row.k.iloc[0]), float(row.mu.iloc[0]), float(row.sigma.iloc[0])
    return bench.reg_value(pred_raw, float(scale.iloc[0]), k, mu, sg)


def write_archive(d: Path, build_dir: Path, rows=None, name="llm_v8_q0__synthmodel"):
    """An archived predictions CSV whose `prediction` was produced on build_dir, and
    the matching published parquet."""
    d.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = [(gv, q, t, h, 1000.0 * (h + 1) + 7.0 * FIRMS[gv])
                for gv in FIRMS for q in ORIGIN_Q for t in ("revtq", "cogsq", "niq") for h in (1, 2, 3)]
    recs = []
    for gv, q, t, h, raw in rows:
        recs.append({"firm_id": gv, "target": t, "quarter": _qend(q), "forecast_horizon": h,
                     "prediction": expected_prediction(build_dir, gv, q, t, raw) if gv in FIRMS else 0.0,
                     "actual": 0.0, "baseline": 0.0, "pred_raw": raw, "actual_raw": 1.0,
                     "baseline_raw": 1.0, "target_datadate": _qend(q + h)})
    df = pd.DataFrame(recs)
    csv = d / f"{name}_predictions.csv"
    df.to_csv(csv, index=False)
    df[renorm.OUT_COLS].to_parquet(d / f"{name}__pf_full__test__predictions.parquet", index=False)
    return csv


def run_renorm(csv, build_dir, out_dir, *extra):
    return renorm.main(["--csv", str(csv), "--processed-dir", str(build_dir), "--dataset-tag", TAG,
                        "--out-dir", str(out_dir), *extra])


def read_report(out_dir):
    return json.loads((out_dir / "renormalize_report.json").read_text(encoding="utf-8"))


# ── llm_renormalize ─────────────────────────────────────────────────────────────

def test_same_build_round_trip_passes_gate(tmp_path):
    b = write_build(tmp_path / "b13")
    csv = write_archive(tmp_path / "arch", b)
    out = tmp_path / "out"
    assert run_renorm(csv, b, out, "--check-against", str(tmp_path / "arch"), "--require-match") == 0
    rep = read_report(out)["cells"][0]
    assert rep["rows_out"] == rep["rows_in"] == 36
    assert rep["vs_reference"]["n_exact"] == 36 and rep["vs_reference"]["max_abs"] == 0.0
    got = pd.read_parquet(out / "llm_v8_q0__synthmodel__pf_full__test__predictions.parquet", engine="fastparquet")
    assert list(got.columns) == renorm.OUT_COLS
    assert got.dtypes["firm_id"] == np.int64 and str(got.dtypes["quarter"]).startswith("datetime64")
    # The written file itself, frame for frame against the published parquet.
    want = pd.read_parquet(tmp_path / "arch" / "llm_v8_q0__synthmodel__pf_full__test__predictions.parquet",
                           engine="fastparquet")
    pd.testing.assert_frame_equal(got.sort_values(renorm.KEY).reset_index(drop=True),
                                  want.sort_values(renorm.KEY).reset_index(drop=True), check_dtype=True)


CELL = "llm_v8_q0__synthmodel"
OUT_NAME = f"{CELL}__pf_full__test__predictions.parquet"


def gate_setup(tmp_path):
    """A same-build archive that passes the gate untouched: (build, archive dir, csv, reference parquet)."""
    b = write_build(tmp_path / "b13")
    arch = tmp_path / "arch"
    csv = write_archive(arch, b)
    return b, arch, csv, arch / OUT_NAME


def run_gate(csv, b, arch, out):
    """(exit code, the gate's failure messages for the one cell)."""
    rc = run_renorm(csv, b, out, "--check-against", str(arch), "--require-match")
    return rc, read_report(out)["gate"]["failures"].get(csv.name, [])


def rewrite_reference(ref, edit):
    frame = pd.read_parquet(ref, engine="fastparquet")
    edit(frame).to_parquet(ref, index=False, engine="fastparquet")


def test_gate_fails_on_nan_in_reference(tmp_path):
    b, arch, csv, ref = gate_setup(tmp_path)

    def nan_one(f):
        f.loc[3, "prediction"] = np.nan
        return f
    rewrite_reference(ref, nan_one)
    rc, fails = run_gate(csv, b, arch, tmp_path / "out")
    assert rc == 1
    assert len(fails) == 1 and "NaN in new or reference" in fails[0]
    assert read_report(tmp_path / "out")["cells"][0]["vs_reference"]["n_nan"] == 1


def test_gate_fails_on_blank_csv_prediction(tmp_path):
    b, arch, csv, _ = gate_setup(tmp_path)
    df = pd.read_csv(csv, float_precision="round_trip")
    df["prediction"] = df["prediction"].astype(object)
    df.loc[3, "prediction"] = ""
    df.to_csv(csv, index=False)
    rc, fails = run_gate(csv, b, arch, tmp_path / "out")
    assert rc == 1
    assert len(fails) == 1 and "NaN in new or csv" in fails[0]
    assert read_report(tmp_path / "out")["cells"][0]["vs_archived_csv"]["n_nan"] == 1


def test_gate_fails_on_extra_reference_key(tmp_path):
    """One reference row the output lacks; every shared value identical."""
    b, arch, csv, ref = gate_setup(tmp_path)

    def add_row(f):
        extra = f.iloc[[0]].assign(forecast_horizon=f["forecast_horizon"].max() + 1)
        return pd.concat([f, extra], ignore_index=True)
    rewrite_reference(ref, add_row)
    rc, fails = run_gate(csv, b, arch, tmp_path / "out")
    assert rc == 1
    assert len(fails) == 1 and "key sets differ" in fails[0] and "only_in_reference=1" in fails[0]


def test_gate_fails_on_missing_reference_key(tmp_path):
    b, arch, csv, ref = gate_setup(tmp_path)
    rewrite_reference(ref, lambda f: f.drop(index=5).reset_index(drop=True))
    rc, fails = run_gate(csv, b, arch, tmp_path / "out")
    assert rc == 1
    assert len(fails) == 1 and "key sets differ" in fails[0] and "only_in_output=1" in fails[0]


def test_gate_fails_on_duplicate_reference_key(tmp_path):
    """One reference row repeated: the key sets agree, so only the duplicate check can catch it."""
    b, arch, csv, ref = gate_setup(tmp_path)
    rewrite_reference(ref, lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True))
    rc, fails = run_gate(csv, b, arch, tmp_path / "out")
    assert rc == 1
    assert len(fails) == 1 and "reference_duplicate_keys=1" in fails[0]
    assert read_report(tmp_path / "out")["cells"][0]["vs_reference"]["only_in_reference"] == 0


def test_gate_fails_on_reference_value_off_by_1e_9(tmp_path):
    """The CSV matches; only the reference parquet is off."""
    b, arch, csv, ref = gate_setup(tmp_path)

    def nudge(f):
        f.loc[7, "prediction"] += 1e-9
        return f
    rewrite_reference(ref, nudge)
    rc, fails = run_gate(csv, b, arch, tmp_path / "out")
    assert rc == 1
    assert len(fails) == 1 and fails[0].startswith("max |new - reference|")


def test_gate_fails_on_csv_prediction_off_by_1e_9(tmp_path):
    """The reference matches; only the archived CSV's own prediction is off."""
    b, arch, csv, _ = gate_setup(tmp_path)
    df = pd.read_csv(csv, float_precision="round_trip")
    df.loc[7, "prediction"] += 1e-9
    df.to_csv(csv, index=False)
    rc, fails = run_gate(csv, b, arch, tmp_path / "out")
    assert rc == 1
    assert len(fails) == 1 and fails[0].startswith("max |new - csv prediction|")


def test_gate_fails_when_reference_parquet_is_absent(tmp_path):
    b, arch, csv, ref = gate_setup(tmp_path)
    ref.unlink()
    rc, fails = run_gate(csv, b, arch, tmp_path / "out")
    assert rc == 1
    assert fails == ["no reference parquet to check against"]


def test_new_build_uses_new_artifacts_not_the_csv_prediction(tmp_path):
    b13 = write_build(tmp_path / "b13")
    b14 = write_build(tmp_path / "b14", k=0.8, mu_shift=0.05, scale_mult=1.1)
    csv = write_archive(tmp_path / "arch", b13)
    out = tmp_path / "out"
    assert run_renorm(csv, b14, out) == 0
    got = pd.read_parquet(out / "llm_v8_q0__synthmodel__pf_full__test__predictions.parquet", engine="fastparquet")
    src = pd.read_csv(csv, parse_dates=["quarter"])
    m = got.merge(src, on=renorm.KEY, suffixes=("_new", "_old"))
    assert len(m) == len(got) == len(src) == 36   # the loop below must not run on an empty merge
    for _, r in m.iterrows():
        q = QMAP.ts_to_q(r.quarter)
        assert r.prediction_new == expected_prediction(b14, r.firm_id, q, r.target, r.pred_raw)
    assert (m.prediction_new != m.prediction_old).all()   # the resume trap would keep these equal
    # The same-build gate must refuse this run.
    assert run_renorm(csv, b14, tmp_path / "out2", "--check-against", str(tmp_path / "arch"),
                      "--require-match") == 1


def test_drops_are_counted_by_reason_and_fail_the_gate(tmp_path):
    b13 = write_build(tmp_path / "b13")
    b14 = write_build(tmp_path / "b14", drop_scale_for=(2000, 162))
    rows = [(1000, 161, "revtq", 1, 500.0), (2000, 162, "revtq", 1, 600.0),
            (1000, 161, "dvq", 1, 5.0), (1000, 161, "niq", 2, float("nan"))]
    csv = write_archive(tmp_path / "arch", b13, rows=[r for r in rows if r[2] != "dvq" and np.isfinite(r[4])])
    df = pd.read_csv(csv)
    extra = pd.DataFrame([
        {**df.iloc[0].to_dict(), "target": "dvq"},
        {**df.iloc[0].to_dict(), "target": "niq", "forecast_horizon": 2, "pred_raw": float("nan")},
    ])
    pd.concat([df, extra]).to_csv(csv, index=False)
    out = tmp_path / "out"
    assert run_renorm(csv, b14, out) == 0
    rep = read_report(out)["cells"][0]
    assert rep["dropped"] == {"firm_not_in_build": 0, "no_valid_scale": 1,
                              "no_reg_stats": 1, "pred_raw_nonfinite": 1}
    assert rep["rows_out"] == 1
    assert run_renorm(csv, b13, tmp_path / "o2", "--check-against", str(tmp_path / "arch"),
                      "--require-match") == 1


def test_csv_floats_are_parsed_round_trip_exact(tmp_path):
    """Non-dyadic pred_raw values that pandas' default CSV parser reads one ulp off."""
    b = write_build(tmp_path / "b13")
    keys = [(gv, q, t, h) for gv in FIRMS for q in ORIGIN_Q for t in ("revtq", "cogsq", "niq") for h in (1, 2, 3)]
    rows = [(*k, 1234.5678901234567 + i * 111.11111111111111) for i, k in enumerate(keys)]
    csv = write_archive(tmp_path / "arch", b, rows=rows)
    raw_default = pd.read_csv(csv)["pred_raw"].to_numpy()
    assert (raw_default != np.array([r[4] for r in rows])).any()   # the fixture really exercises the parser
    out = tmp_path / "out"
    assert run_renorm(csv, b, out, "--check-against", str(tmp_path / "arch"), "--require-match") == 0
    rep = read_report(out)["cells"][0]
    assert rep["vs_archived_csv"]["max_abs"] == 0.0 and rep["vs_archived_csv"]["n_exact"] == 36
    assert rep["vs_reference"]["max_abs"] == 0.0 and rep["vs_reference"]["n_exact"] == 36
    assert rep["rows_clipped_at_max_abs_zscore"] == 0
    got = pd.read_parquet(out / OUT_NAME, engine="fastparquet")
    assert got.dtypes["forecast_horizon"] == np.int64


def test_zero_and_negative_scale_count_as_no_valid_scale(tmp_path):
    b13 = write_build(tmp_path / "b13")
    b14 = write_build(tmp_path / "b14", scale_override={(1000, 161): 0.0, (2000, 162): -5.0})
    csv = write_archive(tmp_path / "arch", b13)
    out = tmp_path / "out"
    assert run_renorm(csv, b14, out) == 0
    rep = read_report(out)["cells"][0]
    # 2 origins x 3 targets x 3 horizons.
    assert rep["dropped"] == {"firm_not_in_build": 0, "no_valid_scale": 18,
                              "no_reg_stats": 0, "pred_raw_nonfinite": 0}
    assert rep["rows_out"] == 18 and rep["origins_out"] == 2


def test_huge_pred_raw_is_clipped_and_counted(tmp_path):
    b = write_build(tmp_path / "b13")
    rows = [(1000, 161, "revtq", 1, 1e300), (1000, 161, "revtq", 2, -1e300), (1000, 161, "revtq", 3, 500.0)]
    csv = write_archive(tmp_path / "arch", b, rows=rows)
    out = tmp_path / "out"
    assert run_renorm(csv, b, out) == 0
    rep = read_report(out)["cells"][0]
    assert rep["rows_clipped_at_max_abs_zscore"] == 2
    got = pd.read_parquet(out / OUT_NAME, engine="fastparquet").sort_values("forecast_horizon")
    assert got["prediction"].tolist()[:2] == [bench.MAX_ABS_ZSCORE, -bench.MAX_ABS_ZSCORE]


def test_gvkey_missing_from_firm_map_never_reads_another_firm(tmp_path):
    """gvkey 1 is not in firm_id_map, but dense firm id 1 exists (gvkey 2000)."""
    b = write_build(tmp_path / "b")
    csv = tmp_path / "arch" / "llm_v8_q0__synthmodel_predictions.csv"
    csv.parent.mkdir(parents=True)
    pd.DataFrame([{"firm_id": 1, "target": "revtq", "quarter": _qend(161), "forecast_horizon": 1,
                   "prediction": 0.0, "actual": 0.0, "baseline": 0.0, "pred_raw": 10.0,
                   "actual_raw": 1.0, "baseline_raw": 1.0, "target_datadate": _qend(162)}]).to_csv(csv, index=False)
    out = tmp_path / "out"
    assert run_renorm(csv, b, out) == 0
    rep = read_report(out)["cells"][0]
    assert rep["dropped"]["firm_not_in_build"] == 1 and rep["rows_out"] == 0


def test_nearest_quarter_fallback_is_counted(tmp_path):
    b13 = write_build(tmp_path / "b13")
    b14 = write_build(tmp_path / "b14", drop_niq_stats_q=161)
    csv = write_archive(tmp_path / "arch", b13)
    out = tmp_path / "out"
    assert run_renorm(csv, b14, out) == 0
    rep = read_report(out)["cells"][0]
    # niq rows at origin 2010Q1: 2 firms x 3 horizons, served by the nearest quarter.
    assert rep["rows_reg_stats_nearest_quarter_fallback"] == 6


def test_duplicate_keys_refused(tmp_path):
    b = write_build(tmp_path / "b")
    csv = write_archive(tmp_path / "arch", b)
    df = pd.read_csv(csv)
    pd.concat([df, df.iloc[:1]]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="duplicated"):
        run_renorm(csv, b, tmp_path / "out")


def test_parquet_name_matches_benchmark_output_paths():
    tag = "llm_v8s_q0__gpt_5_5"
    got = renorm.parquet_name_for(Path(f"{tag}_predictions.csv"), "pf_full")
    want = bench.output_paths(True, "gpt-5.5", prompt_version="v8s", feature_set="pf_full")["pred_parquet"].name
    assert got == want
    with pytest.raises(ValueError):
        renorm.parquet_name_for(Path("forma_fgrid__pf_full__test__predictions.parquet"), "pf_full")
