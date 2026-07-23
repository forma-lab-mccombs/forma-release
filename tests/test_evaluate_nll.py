"""Tests for the likelihood (NLL) track in evaluate.evaluate_models.

Covers:
  * The per-entry NLL helper (`_nll_per_entry`) against closed forms / scipy.
  * End-to-end: a forecast carrying a `sigma` column is scored on mean NLL
    (computed by evaluate against the shared truth) and the value matches a
    hand-computed pooled Gaussian NLL on the all-models common sample.
  * A point-only forecast (no sigma) leaves the NLL table blank (column dropped)
    while still being scored on r2/mae — i.e. NLL is purely additive.
  * The `{stem}.nll.json` sidecar selects the Student-t density + df.

End-to-end NLL assertions use abs=1e-6: residual/sigma are cached float32
(accumulation float64), so values match the float64 reference to ~1e-7.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from forma.scoring.evaluate import evaluate_models, _nll_per_entry, _mixture_nll_per_cell

FIRMS = ["F1", "F2", "F3", "F4"]
QUARTERS = pd.to_datetime(["2010-03-31", "2010-06-30", "2010-09-30"])
TARGETS = ["niq", "atq"]
HORIZONS = [1, 2]


def _make_truth(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in FIRMS:
        for q in QUARTERS:
            row = {"firm_id": f, "quarter": q}
            for t in TARGETS:
                row[f"{t}_level_0"] = float(rng.normal())
                for h in HORIZONS:
                    row[f"{t}_t{h}"] = float(rng.normal(scale=2.0))
            rows.append(row)
    return pd.DataFrame(rows)


def _make_forecast(truth, seed, with_sigma):
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in truth.iterrows():
        for t in TARGETS:
            for h in HORIZONS:
                actual = r[f"{t}_t{h}"]
                pred = actual + rng.normal(scale=1.0)
                row = {
                    "firm_id": r["firm_id"], "quarter": r["quarter"],
                    "target": t, "forecast_horizon": h, "prediction": float(pred),
                }
                if with_sigma:
                    # strictly positive predictive std
                    row["sigma"] = float(0.5 + abs(rng.normal(scale=1.0)))
                rows.append(row)
    return pd.DataFrame(rows)


def _write(df, path):
    df.to_parquet(path, engine="fastparquet", index=False)


def _expected_global_nll(truth, fdf, family="gaussian", df=None):
    """Pooled mean NLL of a sigma-bearing forecast over its full (here = common) sample."""
    long = []
    for _, r in truth.iterrows():
        for t in TARGETS:
            for h in HORIZONS:
                long.append((r["firm_id"], r["quarter"], t, h, r[f"{t}_t{h}"]))
    tl = pd.DataFrame(long, columns=["firm_id", "quarter", "target", "forecast_horizon", "actual"])
    m = tl.merge(fdf, on=["firm_id", "quarter", "target", "forecast_horizon"], how="inner")
    res = m["prediction"].to_numpy(float) - m["actual"].to_numpy(float)
    sigma = m["sigma"].to_numpy(float)
    per = _nll_per_entry(res, sigma, family, df)
    return float(per.mean()), len(m)


def _run(tmp_path, forecasts_named, truth, sidecars=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = []
    for name, fdf in forecasts_named:
        p = tmp_path / f"{name}__pf__test__predictions.parquet"
        _write(fdf, p)
        if sidecars and name in sidecars:
            side = tmp_path / f"{name}__pf__test__predictions.nll.json"
            side.write_text(json.dumps(sidecars[name]))
        files.append((p, name, "pf", "test", None))
    out = tmp_path / "metrics"
    evaluate_models(files, truth.copy(), out, split_name="test")
    return out / "test"


# ----------------------------- helper unit tests -----------------------------

def test_nll_per_entry_gaussian_closed_form():
    # standard normal at the mean: 0.5*log(2π)
    assert _nll_per_entry(np.array([0.0]), np.array([1.0]), "gaussian", None)[0] == pytest.approx(
        0.5 * math.log(2 * math.pi))
    # general point matches the explicit formula
    res, sig = np.array([1.5]), np.array([2.0])
    expect = 0.5 * math.log(2 * math.pi) + 0.5 * math.log(4.0) + 0.5 * (1.5 ** 2) / 4.0
    assert _nll_per_entry(res, sig, "gaussian", None)[0] == pytest.approx(expect)


def test_nll_per_entry_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    res = rng.normal(size=50)
    sig = 0.5 + np.abs(rng.normal(size=50))
    g = _nll_per_entry(res, sig, "gaussian", None)
    assert np.allclose(g, -scipy_stats.norm.logpdf(res, loc=0.0, scale=sig))
    t = _nll_per_entry(res, sig, "student_t", 6.0)
    assert np.allclose(t, -scipy_stats.t.logpdf(res, df=6.0, loc=0.0, scale=sig))


def test_nll_per_entry_student_t_requires_df():
    with pytest.raises(ValueError):
        _nll_per_entry(np.array([0.0]), np.array([1.0]), "student_t", None)


# ------------------------------ end-to-end tests -----------------------------

def test_nll_scored_for_sigma_forecast(tmp_path):
    truth = _make_truth(seed=1)
    sig_model = _make_forecast(truth, seed=2, with_sigma=True)
    point_model = _make_forecast(truth, seed=3, with_sigma=False)

    out = _run(tmp_path, [("probnet", sig_model), ("pointnet", point_model)], truth)

    nll = pd.read_csv(out / "nll_scores__global.csv", index_col=0)
    exp_nll, n = _expected_global_nll(truth, sig_model)
    assert nll.loc["nll", "probnet__pf"] == pytest.approx(exp_nll, abs=1e-6)
    # point-only model carries no sigma → dropped from the NLL table entirely
    assert "pointnet__pf" not in nll.columns

    # ...but both models are still scored on the point metrics (NLL is additive).
    r2 = pd.read_csv(out / "r2_scores__global.csv", index_col=0)
    assert {"probnet__pf", "pointnet__pf"}.issubset(set(r2.columns))


def test_nll_student_t_sidecar(tmp_path):
    truth = _make_truth(seed=4)
    sig_model = _make_forecast(truth, seed=5, with_sigma=True)

    out = _run(
        tmp_path, [("studentnet", sig_model)], truth,
        sidecars={"studentnet": {"family": "student_t", "df": 6.0}},
    )

    nll = pd.read_csv(out / "nll_scores__global.csv", index_col=0)
    exp_nll, _ = _expected_global_nll(truth, sig_model, family="student_t", df=6.0)
    assert nll.loc["nll", "studentnet__pf"] == pytest.approx(exp_nll, abs=1e-6)


def test_nll_absent_when_no_sigma_forecasts(tmp_path):
    """With only point forecasts (no sigma anywhere), no nll CSV is produced."""
    truth = _make_truth(seed=6)
    a = _make_forecast(truth, seed=7, with_sigma=False)
    b = _make_forecast(truth, seed=8, with_sigma=False)
    out = _run(tmp_path, [("a", a), ("b", b)], truth)
    # Point metrics still computed; the all-NaN nll pivot is skipped, not written.
    assert (out / "r2_scores__global.csv").exists()
    assert not (out / "nll_scores__global.csv").exists()


def _run_seeded(tmp_path, forecasts_seeded, truth):
    """Like _run but assigns integer seeds -> 5-part filenames -> the seed_agg path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = []
    for name, fdf, sd in forecasts_seeded:
        p = tmp_path / f"{name}__pf__{sd}__test__predictions.parquet"
        _write(fdf, p)
        files.append((p, name, "pf", "test", sd))
    out = tmp_path / "metrics"
    evaluate_models(files, truth.copy(), out, split_name="test")
    return out / "test"


def test_seed_agg_nll_skipped_when_no_sigma(tmp_path):
    # Seeded point-only forecasts exercise the seed_agg branch with an all-NaN
    # nll_mean: the guard must skip the degenerate CSV (the same failure save_metric
    # avoids), while r2_mean is still written.
    truth = _make_truth(seed=20)
    a = _make_forecast(truth, seed=21, with_sigma=False)
    b = _make_forecast(truth, seed=22, with_sigma=False)
    od = _run_seeded(tmp_path, [("m", a, 1), ("m", b, 2)], truth)
    assert (od / "seed_agg__r2_mean__by_target_horizon.csv").exists()
    assert not (od / "seed_agg__nll_mean__by_target_horizon.csv").exists()


def test_seed_agg_nll_written_when_sigma_present(tmp_path):
    # With seeded sigma-bearing forecasts, the seed-aggregated NLL IS produced and
    # carries finite values.
    truth = _make_truth(seed=30)
    a = _make_forecast(truth, seed=31, with_sigma=True)
    b = _make_forecast(truth, seed=32, with_sigma=True)
    od = _run_seeded(tmp_path, [("m", a, 1), ("m", b, 2)], truth)
    nll_path = od / "seed_agg__nll_mean__by_target_horizon.csv"
    assert nll_path.exists()
    df = pd.read_csv(nll_path)
    assert df["m__pf"].notna().any()


# ----------------------------- z^2 (calibration) ----------------------------- #

def _expected_global(truth, fdf, fn):
    """Apply fn(res, sigma) -> per-cell array; return its mean over the merged sample."""
    long = []
    for _, r in truth.iterrows():
        for t in TARGETS:
            for h in HORIZONS:
                long.append((r["firm_id"], r["quarter"], t, h, r[f"{t}_t{h}"]))
    tl = pd.DataFrame(long, columns=["firm_id", "quarter", "target", "forecast_horizon", "actual"])
    m = tl.merge(fdf, on=["firm_id", "quarter", "target", "forecast_horizon"], how="inner")
    res = m["prediction"].to_numpy(float) - m["actual"].to_numpy(float)
    sigma = m["sigma"].to_numpy(float)
    return float(fn(res, sigma).mean()), m


def test_z2_column_written_and_correct(tmp_path):
    truth = _make_truth(seed=40)
    sig_model = _make_forecast(truth, seed=41, with_sigma=True)
    point_model = _make_forecast(truth, seed=42, with_sigma=False)
    out = _run(tmp_path, [("probnet", sig_model), ("pointnet", point_model)], truth)

    z2 = pd.read_csv(out / "z2_scores__global.csv", index_col=0)
    exp_z2, _ = _expected_global(truth, sig_model, lambda res, sig: (res / sig) ** 2)
    assert z2.loc["z2", "probnet__pf"] == pytest.approx(exp_z2, abs=1e-6)
    # point-only model has no sigma -> dropped from the z2 table.
    assert "pointnet__pf" not in z2.columns


def test_z2_absent_when_no_sigma(tmp_path):
    truth = _make_truth(seed=43)
    a = _make_forecast(truth, seed=44, with_sigma=False)
    b = _make_forecast(truth, seed=45, with_sigma=False)
    out = _run(tmp_path, [("a", a), ("b", b)], truth)
    assert (out / "r2_scores__global.csv").exists()
    assert not (out / "z2_scores__global.csv").exists()


def test_z2_blank_for_non_gaussian_family(tmp_path):
    """z^2's '-> 1' reading is Gaussian-only, so it's left blank for student_t.
    A Gaussian sibling still gets z^2; both are still scored on NLL."""
    truth = _make_truth(seed=46)
    gnet = _make_forecast(truth, seed=47, with_sigma=True)
    tnet = _make_forecast(truth, seed=48, with_sigma=True)
    out = _run(
        tmp_path, [("gnet", gnet), ("tnet", tnet)], truth,
        sidecars={"tnet": {"family": "student_t", "df": 6.0}},
    )
    z2 = pd.read_csv(out / "z2_scores__global.csv", index_col=0)
    assert "gnet__pf" in z2.columns           # Gaussian gets z^2
    assert "tnet__pf" not in z2.columns        # student_t left blank -> column dropped
    # both are still scored on NLL (with their respective densities)
    nll = pd.read_csv(out / "nll_scores__global.csv", index_col=0)
    assert {"gnet__pf", "tnet__pf"}.issubset(set(nll.columns))


# ----------------------------- seed-mixture NLL ----------------------------- #

def test_mixture_helper_identity_and_two_seed():
    # K identical seeds -> mixture NLL == the single-seed NLL.
    nll = np.array([[0.3, 1.1, 2.0]])
    same = np.vstack([nll, nll, nll])
    assert np.allclose(_mixture_nll_per_cell(same), nll[0])
    # Two distinct seeds -> -log(0.5*(e^-a + e^-b)).
    a = np.array([0.5, 2.0]); b = np.array([1.5, 0.25])
    got = _mixture_nll_per_cell(np.vstack([a, b]))
    expect = -np.log(0.5 * (np.exp(-a) + np.exp(-b)))
    assert np.allclose(got, expect)
    # Mixture is <= the mean of per-seed NLLs (Jensen).
    assert np.all(got <= 0.5 * (a + b) + 1e-12)


def test_seed_mixture_nll_written_and_correct(tmp_path):
    truth = _make_truth(seed=50)
    a = _make_forecast(truth, seed=51, with_sigma=True)
    b = _make_forecast(truth, seed=52, with_sigma=True)
    od = _run_seeded(tmp_path, [("m", a, 1), ("m", b, 2)], truth)

    mp = od / "mixture_nll__global.csv"
    assert mp.exists()
    mix = pd.read_csv(mp, index_col=0)

    # Hand-compute the global mixture NLL on the merged cells.
    _, ma = _expected_global(truth, a, lambda res, sig: res)  # just to get the merged frame
    _, mb = _expected_global(truth, b, lambda res, sig: res)
    key = ["firm_id", "quarter", "target", "forecast_horizon"]
    j = ma.merge(mb, on=key, suffixes=("_a", "_b"))
    res_a = j["prediction_a"].to_numpy(float) - j["actual_a"].to_numpy(float)
    res_b = j["prediction_b"].to_numpy(float) - j["actual_b"].to_numpy(float)
    nll_a = _nll_per_entry(res_a, j["sigma_a"].to_numpy(float), "gaussian", None)
    nll_b = _nll_per_entry(res_b, j["sigma_b"].to_numpy(float), "gaussian", None)
    expect = float(_mixture_nll_per_cell(np.vstack([nll_a, nll_b])).mean())
    assert mix.loc["mixture_nll", "m__pf"] == pytest.approx(expect, abs=1e-6)

    # And the mixture beats the mean-of-seed-NLLs (seed_agg nll_mean), per Jensen.
    seed_nll_mean = 0.5 * (float(nll_a.mean()) + float(nll_b.mean()))
    assert mix.loc["mixture_nll", "m__pf"] <= seed_nll_mean + 1e-9
