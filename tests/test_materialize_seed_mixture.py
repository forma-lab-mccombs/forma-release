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

"""Tests for scripts/materialize_seed_mixture.py (the keyed seed-mixture materializer
that produced the camera-ready Panel A/B mixtures). Data-free: synthetic seeds only."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
KEYS = ["firm_id", "target", "quarter", "forecast_horizon"]
VALUE_COLS = ["prediction", "sigma"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def msm():
    return _load("materialize_seed_mixture", "scripts/materialize_seed_mixture.py")


def _keys(n_firms=30, targets=("atq", "revtq", "niq"), horizons=4):
    firms = [f"{1000 + 7 * i:06d}" for i in range(n_firms)]
    quarters = pd.PeriodIndex(["2010Q1", "2010Q2"], freq="Q").end_time
    rows = [(f, t, q, h) for f in firms for t in targets for q in quarters for h in range(1, horizons + 1)]
    return pd.DataFrame(rows, columns=KEYS)


def _seed_frame(keys, seed):
    rng = np.random.default_rng(seed)
    df = keys.copy()
    df["prediction"] = rng.normal(size=len(df)).astype(np.float32)
    df["sigma"] = rng.uniform(0.1, 2.0, size=len(df)).astype(np.float32)
    df["model"] = "forma_lap05_fgrid"
    return df


def _write(df, path, str_type=pa.string(), ts_unit="us", horizon_type=pa.int64()):
    """ts_unit truncates the end-of-quarter stamp the way a coarser writer stores it
    (ns .999999999 -> us .999999, the R13 forecasts' form)."""
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    for col, typ in (("firm_id", str_type), ("target", str_type),
                     ("quarter", pa.timestamp(ts_unit)), ("forecast_horizon", horizon_type)):
        tbl = tbl.set_column(tbl.schema.get_field_index(col), col,
                             tbl.column(col).cast(typ, safe=col != "quarter"))
    pq.write_table(tbl, path)


def _make_arm(tmp, arm="forma_lap05_fgrid", seeds=(60, 61, 62, 63, 64), sidecar=None, shuffle=()):
    src = tmp / "src"
    src.mkdir(exist_ok=True)
    keys = _keys()
    frames = {}
    for s in seeds:
        df = _seed_frame(keys, s)
        frames[s] = df
        out = df.sample(frac=1.0, random_state=s) if s in shuffle else df
        p = src / f"{arm}__pf_full__{s}__test__predictions.parquet"
        _write(out, p)
        if sidecar is not None:
            (src / f"{p.stem}.nll.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return src, keys, frames


def _expected(frames, seeds):
    mus = np.vstack([frames[s]["prediction"].to_numpy().astype(np.float64) for s in seeds])
    sig = np.vstack([frames[s]["sigma"].to_numpy().astype(np.float64) for s in seeds])
    return mus.mean(axis=0).astype(np.float32), np.sqrt((sig ** 2).mean(axis=0) + mus.var(axis=0)).astype(np.float32)


def _read_sorted(path):
    return pd.read_parquet(path).sort_values(KEYS).reset_index(drop=True)


def test_keyed_moment_match_with_shuffled_seed(msm, tmp_path):
    seeds = (60, 61, 62, 63, 64)
    src, keys, frames = _make_arm(tmp_path, sidecar={"family": "laplace", "df": None}, shuffle=(60,))
    rc = msm.main(["--src", str(src), "--dest", str(tmp_path / "out"), "--buckets", "7",
                   "--batch-rows", "100", "--expect-seeds", "5",
                   "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"])
    assert rc == 0
    out = tmp_path / "out" / "forma_lap05_fgrid__pf_full__test__predictions.parquet"
    got = _read_sorted(out)
    order = keys.sort_values(KEYS).index.to_numpy()
    pred, sigma = _expected(frames, seeds)
    assert len(got) == len(keys)
    np.testing.assert_array_equal(got["prediction"].to_numpy(), pred[order])
    np.testing.assert_array_equal(got["sigma"].to_numpy(), sigma[order])
    assert set(got["model"]) == {"forma_lap05_fgrid"}
    assert got["prediction"].dtype == np.float32 and got["sigma"].dtype == np.float32
    assert pq.read_schema(out).field("quarter").type == pa.timestamp("us")
    side = out.with_name(f"{out.stem}.nll.json")
    assert json.loads(side.read_text(encoding="utf-8")) == {"family": "laplace", "df": None}
    assert not any(p.name.startswith("_mixture_tmp") for p in (tmp_path / "out").iterdir())


def test_mixes_seeds_whose_key_types_differ(msm, tmp_path):
    """R13's Gaussian seed 61 has string firm_id + timestamp[ns] quarter, its siblings
    large_string + timestamp[us]: same keys, so they must mix, onto the canonical types."""
    seeds = (60, 61, 62)
    src, keys, frames = _make_arm(tmp_path, arm="forma_fgrid", seeds=seeds)
    for s in seeds:
        p = src / f"forma_fgrid__pf_full__{s}__test__predictions.parquet"
        if s == 61:
            _write(frames[s], p, str_type=pa.string(), ts_unit="ns")
        else:
            _write(frames[s], p, str_type=pa.large_string(),
                   horizon_type=pa.int32() if s == 62 else pa.int64())
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "out"), "--buckets", "5",
                     "--batch-rows", "100", "--arm", "forma_fgrid__pf_full__*__test__predictions.parquet"]) == 0
    out = tmp_path / "out" / "forma_fgrid__pf_full__test__predictions.parquet"
    got = _read_sorted(out)
    order = keys.sort_values(KEYS).index.to_numpy()
    pred, sigma = _expected(frames, seeds)
    np.testing.assert_array_equal(got["prediction"].to_numpy(), pred[order])
    np.testing.assert_array_equal(got["sigma"].to_numpy(), sigma[order])
    schema = pq.read_schema(out)
    assert {k: schema.field(k).type for k in KEYS} == {
        "firm_id": pa.large_string(), "target": pa.large_string(),
        "quarter": pa.timestamp("us"), "forecast_horizon": pa.int64()}


def test_seeds_written_by_the_train_py_path_mix(msm, tmp_path):
    """train.py stamps quarters with qoff_to_quarter_end (PeriodIndex.end_time): under
    this env's pandas < 3 that is 23:59:59.999999999 [ns], which a safe [ns] -> [us]
    cast refuses. Every R13 FFNN seed is written that way."""
    from forma.data.quarters import qoff_to_quarter_end
    seeds = (60, 61, 62)
    qoff = np.array([161, 162, 177])
    eoq = qoff_to_quarter_end(qoff)
    assert eoq.dtype == "datetime64[ns]" and (eoq.nanosecond == 999).all()
    src = tmp_path / "src"
    src.mkdir()
    frames = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        df = pd.DataFrame([(f"{1000 + 7 * i:06d}", t, q, h) for i in range(20) for t in ("atq", "niq")
                           for q in qoff for h in (1, 2)], columns=KEYS)
        df["quarter"] = qoff_to_quarter_end(df["quarter"].values.astype("int64"))
        df["prediction"] = rng.normal(size=len(df)).astype(np.float32)
        df["sigma"] = rng.uniform(0.1, 2.0, size=len(df)).astype(np.float32)
        df["model"] = "ffnn_large_b50"
        frames[s] = df
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       src / f"ffnn_large_b50__pf_full__{s}__test__predictions.parquet")
    assert pq.read_schema(src / "ffnn_large_b50__pf_full__60__test__predictions.parquet").field(
        "quarter").type == pa.timestamp("ns")
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "out"), "--buckets", "3",
                     "--arm", "ffnn_large_b50__pf_full__*__test__predictions.parquet"]) == 0
    out = tmp_path / "out" / "ffnn_large_b50__pf_full__test__predictions.parquet"
    assert pq.read_schema(out).field("quarter").type == pa.timestamp("us")
    got = _read_sorted(out)
    keys = frames[60][KEYS]
    order = keys.sort_values(KEYS).index.to_numpy()
    pred, sigma = _expected(frames, seeds)
    np.testing.assert_array_equal(got["prediction"].to_numpy(), pred[order])
    np.testing.assert_array_equal(got["sigma"].to_numpy(), sigma[order])
    canon = pd.Series(keys["quarter"].to_numpy()[order]).dt.floor("us")
    np.testing.assert_array_equal(got["quarter"].to_numpy().astype("datetime64[us]"),
                                  canon.to_numpy().astype("datetime64[us]"))
    assert set(pd.Series(got["quarter"]).dt.strftime("%H:%M:%S.%f")) == {"23:59:59.999999"}


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_accepts_end_of_quarter_stamps_in_any_unit(msm, tmp_path, unit):
    """The quarter's last tick in the column's own unit is an end-of-quarter stamp
    (parquet has no [s] unit: it stores timestamp[s] as [ms], i.e. 23:59:59.000)."""
    seeds = (60, 61)
    src, keys, frames = _make_arm(tmp_path, arm="forma_fgrid", seeds=seeds)
    _write(frames[61], src / "forma_fgrid__pf_full__61__test__predictions.parquet", ts_unit=unit)
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "out"),
                     "--arm", "forma_fgrid__pf_full__*__test__predictions.parquet"]) == 0
    got = _read_sorted(tmp_path / "out" / "forma_fgrid__pf_full__test__predictions.parquet")
    np.testing.assert_array_equal(got["prediction"].to_numpy(),
                                  _expected(frames, seeds)[0][keys.sort_values(KEYS).index.to_numpy()])


@pytest.mark.parametrize("unit, stamp", [
    ("ns", "2010-03-31 23:59:59.999999998"),
    ("us", "2010-03-31 23:59:59"),
    ("us", "2010-03-31 00:00:00"),
    ("us", "2010-04-01 00:00:00"),
    ("us", "2010-02-15 12:00:00"),
])
def test_rejects_non_end_of_quarter_stamps(msm, tmp_path, capsys, unit, stamp):
    src, _, frames = _make_arm(tmp_path, arm="forma_fgrid", seeds=(60, 61))
    df = frames[61].copy()
    df.loc[5, "quarter"] = pd.Timestamp(stamp)
    _write(df, src / "forma_fgrid__pf_full__61__test__predictions.parquet", ts_unit=unit)
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "out"),
                     "--arm", "forma_fgrid__pf_full__*__test__predictions.parquet"]) == 1
    assert "1 quarter value(s) are not end-of-quarter stamps" in capsys.readouterr().err
    assert not (tmp_path / "out" / "forma_fgrid__pf_full__test__predictions.parquet").exists()


def test_gaussian_mixture_removes_a_stale_sidecar(msm, tmp_path):
    """A leftover laplace sidecar at the output path would make evaluate.py score the new
    Gaussian mixture as Laplace."""
    src, _, _ = _make_arm(tmp_path, arm="forma_fgrid", seeds=(60, 61))
    dest = tmp_path / "store"
    dest.mkdir()
    stale = dest / "forma_fgrid__pf_full__test__predictions.nll.json"
    stale.write_text(json.dumps({"family": "laplace", "df": None}), encoding="utf-8")
    assert msm.main(["--src", str(src), "--dest", str(dest),
                     "--arm", "forma_fgrid__pf_full__*__test__predictions.parquet"]) == 0
    assert not stale.exists()
    assert (dest / "forma_fgrid__pf_full__test__predictions.parquet").exists()


def test_sidecar_is_replaced_before_the_parquet(msm, tmp_path, monkeypatch):
    src, _, _ = _make_arm(tmp_path, seeds=(60, 61), sidecar={"family": "laplace", "df": None})
    dest = tmp_path / "store"
    dest.mkdir()
    (dest / "forma_lap05_fgrid__pf_full__test__predictions.nll.json").write_text("{}", encoding="utf-8")
    real, order = msm.os.replace, []

    def recording_replace(a, b):
        order.append((Path(a).name, Path(b).name))
        real(a, b)
    monkeypatch.setattr(msm.os, "replace", recording_replace)
    assert msm.main(["--src", str(src), "--dest", str(dest),
                     "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"]) == 0
    assert order == [("forma_lap05_fgrid__pf_full__test__predictions.nll.json.tmp",
                      "forma_lap05_fgrid__pf_full__test__predictions.nll.json"),
                     ("forma_lap05_fgrid__pf_full__test__predictions.parquet.tmp",
                      "forma_lap05_fgrid__pf_full__test__predictions.parquet")]
    side = json.loads((dest / "forma_lap05_fgrid__pf_full__test__predictions.nll.json").read_text(encoding="utf-8"))
    assert side == {"family": "laplace", "df": None}


@pytest.mark.parametrize("arm_sidecar, prior", [
    ({"family": "laplace", "df": None}, {"family": "gaussian", "df": None}),
    ({"family": "laplace", "df": None}, None),
    (None, {"family": "laplace", "df": None}),
])
def test_failed_parquet_replace_restores_the_prior_sidecar(msm, tmp_path, monkeypatch, arm_sidecar, prior):
    """The sidecar lands first; if the parquet's os.replace then fails, the old parquet
    must not be left beside the new sidecar (or without its own)."""
    src, _, _ = _make_arm(tmp_path, seeds=(60, 61), sidecar=arm_sidecar)
    dest = tmp_path / "store"
    dest.mkdir()
    old = dest / "forma_lap05_fgrid__pf_full__test__predictions.parquet"
    old.write_bytes(b"old parquet")
    side = dest / "forma_lap05_fgrid__pf_full__test__predictions.nll.json"
    if prior is not None:
        side.write_text(json.dumps(prior), encoding="utf-8")
    before = side.read_bytes() if prior is not None else None
    real = msm.os.replace

    def failing_replace(a, b):
        if Path(b).suffix == ".parquet":
            raise OSError("simulated failure replacing the parquet")
        real(a, b)
    monkeypatch.setattr(msm.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated"):
        msm.main(["--src", str(src), "--dest", str(dest),
                  "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"])
    assert old.read_bytes() == b"old parquet"
    assert (side.read_bytes() if side.exists() else None) == before
    assert _scratch(dest) == []


def test_rejects_different_key_sets(msm, tmp_path):
    src, keys, _ = _make_arm(tmp_path, seeds=(60, 61))
    p = src / "forma_lap05_fgrid__pf_full__61__test__predictions.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "forecast_horizon"] = 9
    _write(df, p)
    rc = msm.main(["--src", str(src), "--dest", str(tmp_path / "out"), "--buckets", "3",
                   "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"])
    assert rc == 1
    assert not (tmp_path / "out" / "forma_lap05_fgrid__pf_full__test__predictions.parquet").exists()


def test_rejects_a_bucket_present_in_some_seeds_only(msm, tmp_path, capsys):
    """Equal per-seed row counts, but seed 61 swaps one firm for another that hashes to a
    bucket no seed-60 firm uses: that bucket exists in seed 61 only."""
    n_buckets = 997
    src, keys, frames = _make_arm(tmp_path, seeds=(60, 61))
    firms = sorted(keys["firm_id"].unique())
    b = dict(zip(firms, msm.bucket_ids(pa.array(firms), n_buckets)))
    counts = pd.Series(b).value_counts()
    alone = next(f for f in firms if counts[b[f]] == 1)
    new = next(f"{9000 + i:06d}" for i in range(1000)
               if msm.bucket_ids(pa.array([f"{9000 + i:06d}"]), n_buckets)[0] not in counts.index)
    df = frames[61].copy()
    df.loc[df["firm_id"] == alone, "firm_id"] = new
    _write(df, src / "forma_lap05_fgrid__pf_full__61__test__predictions.parquet")
    assert len(df) == len(frames[60])
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "out"), "--buckets", str(n_buckets),
                     "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"]) == 1
    assert "has rows in some seeds only" in capsys.readouterr().err
    assert not (tmp_path / "out" / "forma_lap05_fgrid__pf_full__test__predictions.parquet").exists()


def test_rejects_differing_seed_row_counts(msm, tmp_path, capsys):
    src, _, frames = _make_arm(tmp_path, seeds=(60, 61, 62))
    _write(frames[62].iloc[:-1], src / "forma_lap05_fgrid__pf_full__62__test__predictions.parquet")
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "out"),
                     "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"]) == 1
    assert "seed row counts differ" in capsys.readouterr().err
    assert not (tmp_path / "out" / "forma_lap05_fgrid__pf_full__test__predictions.parquet").exists()


def test_rejects_duplicate_keys(msm, tmp_path):
    src, keys, _ = _make_arm(tmp_path, seeds=(60, 61))
    for s in (60, 61):
        p = src / f"forma_lap05_fgrid__pf_full__{s}__test__predictions.parquet"
        df = pd.read_parquet(p)
        df = pd.concat([df.iloc[1:], df.iloc[[2]]])
        _write(df, p)
    rc = msm.main(["--src", str(src), "--dest", str(tmp_path / "out"), "--buckets", "3",
                   "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"])
    assert rc == 1


@pytest.mark.parametrize("col", KEYS)
def test_rejects_null_keys(msm, tmp_path, capsys, col):
    """A null key in the SAME row of every seed slips past the key comparison (.equals
    matches null with null) and the duplicate check (null == null is null, not True)."""
    seeds = (60, 61)
    src, _, frames = _make_arm(tmp_path, seeds=seeds)
    for s in seeds:
        df = frames[s].copy()
        df[col] = df[col].astype(object)
        df.loc[3, col] = None
        _write(df, src / f"forma_lap05_fgrid__pf_full__{s}__test__predictions.parquet")
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "out"),
                     "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"]) == 1
    err = capsys.readouterr().err
    assert f"null {col} key" in err and "forma_lap05_fgrid__pf_full__60__test__predictions.parquet" in err
    assert not (tmp_path / "out" / "forma_lap05_fgrid__pf_full__test__predictions.parquet").exists()


def test_rejects_mixed_sidecars_and_seed_count(msm, tmp_path):
    src, _, _ = _make_arm(tmp_path, seeds=(60, 61, 62), sidecar={"family": "laplace", "df": None})
    pattern = "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o1"), "--expect-seeds", "5",
                     "--arm", pattern]) == 1
    (src / "forma_lap05_fgrid__pf_full__62__test__predictions.nll.json").unlink()
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o2"), "--arm", pattern]) == 1


def test_expect_family(msm, tmp_path, capsys):
    pattern = "{}__pf_full__*__test__predictions.parquet"
    (tmp_path / "lap").mkdir()
    lap, _, _ = _make_arm(tmp_path / "lap", seeds=(60, 61), sidecar={"family": "laplace", "df": None})
    (tmp_path / "gau").mkdir()
    gau, _, _ = _make_arm(tmp_path / "gau", arm="forma_fgrid", seeds=(60, 61))
    cases = [(lap, "forma_lap05_fgrid", "laplace", None), (lap, "forma_lap05_fgrid", "gaussian", "per their"),
             (gau, "forma_fgrid", "gaussian", None), (gau, "forma_fgrid", "laplace", "no .nll.json sidecar")]
    for i, (src, arm, family, why) in enumerate(cases):
        capsys.readouterr()
        rc = msm.main(["--src", str(src), "--dest", str(tmp_path / f"o{i}"), "--arm", pattern.format(arm),
                       "--expect-family", family])
        if why is None:
            assert rc == 0
        else:
            err = capsys.readouterr().err
            assert rc == 1 and f"expected {family}" in err and why in err


@pytest.mark.parametrize("sidecar", [{"family": "student_t", "df": 5.0}, {"family": "cauchy", "df": None}])
def test_rejects_unsupported_families(msm, tmp_path, capsys, sidecar):
    """student_t's sigma is the t scale, not the std, so mean sigma^2 + var mu is not the
    mixture variance there: refuse rather than write a wrong sigma."""
    src, _, _ = _make_arm(tmp_path, seeds=(60, 61), sidecar=sidecar)
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "out"),
                     "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"]) == 1
    assert f"density family '{sidecar['family']}' is not supported" in capsys.readouterr().err
    assert not (tmp_path / "out" / "forma_lap05_fgrid__pf_full__test__predictions.parquet").exists()


def test_check_against_requires_the_same_density(msm, tmp_path, capsys):
    """A Laplace mixture checked against a reference copied without its sidecar (which
    evaluate.py would score as Gaussian) must fail, and vice versa."""
    pattern = "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"
    src, _, _ = _make_arm(tmp_path, seeds=(60, 61), sidecar={"family": "laplace", "df": None})
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "ref"), "--arm", pattern]) == 0
    ref = tmp_path / "ref" / "forma_lap05_fgrid__pf_full__test__predictions.parquet"
    side = ref.with_name(f"{ref.stem}.nll.json")
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o1"), "--arm", pattern,
                     "--check-against", str(ref)]) == 0
    side.unlink()
    capsys.readouterr()
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o2"), "--arm", pattern,
                     "--check-against", str(ref)]) == 1
    assert "no sidecar: read as gaussian" in capsys.readouterr().err

    (tmp_path / "g").mkdir()
    gsrc, _, _ = _make_arm(tmp_path / "g", seeds=(60, 61))
    side.write_text(json.dumps({"family": "laplace", "df": None}), encoding="utf-8")
    capsys.readouterr()
    assert msm.main(["--src", str(gsrc), "--dest", str(tmp_path / "o3"), "--arm", pattern,
                     "--check-against", str(ref), "--tol", "1e30"]) == 1
    assert "output density ('gaussian', None) != reference ('laplace', None)" in capsys.readouterr().err


def test_check_against(msm, tmp_path):
    src, _, _ = _make_arm(tmp_path, seeds=(60, 61, 62))
    pattern = "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"
    ref_dir = tmp_path / "ref"
    assert msm.main(["--src", str(src), "--dest", str(ref_dir), "--arm", pattern]) == 0
    ref = ref_dir / "forma_lap05_fgrid__pf_full__test__predictions.parquet"
    shuffled = pd.read_parquet(ref).sample(frac=1.0, random_state=1)
    _write(shuffled, tmp_path / "ref_shuffled.parquet")
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o1"), "--buckets", "4",
                     "--arm", pattern, "--check-against", str(tmp_path / "ref_shuffled.parquet")]) == 0
    shuffled.iloc[3, shuffled.columns.get_loc("sigma")] += np.float32(1e-3)
    _write(shuffled, tmp_path / "ref_bad.parquet")
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o2"),
                     "--arm", pattern, "--check-against", str(tmp_path / "ref_bad.parquet")]) == 1
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o3"), "--tol", "1e-2",
                     "--arm", pattern, "--check-against", str(tmp_path / "ref_bad.parquet")]) == 0


def _check_rc(msm, tmp_path, src, ref_df, name, *extra):
    """Write ref_df as a reference and run the gate against it; returns the exit code."""
    _write(ref_df, tmp_path / f"{name}.parquet")
    return msm.main(["--src", str(src), "--dest", str(tmp_path / f"o_{name}"), "--buckets", "4",
                     "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet",
                     "--check-against", str(tmp_path / f"{name}.parquet"), *extra])


def test_check_against_rejects_key_and_row_and_prediction_changes(msm, tmp_path, capsys):
    src, _, _ = _make_arm(tmp_path, seeds=(60, 61, 62))
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "ref"),
                     "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"]) == 0
    good = pd.read_parquet(tmp_path / "ref" / "forma_lap05_fgrid__pf_full__test__predictions.parquet")
    capsys.readouterr()

    one_key = good.copy()
    one_key.loc[5, "forecast_horizon"] = 9
    assert len(one_key) == len(good)
    assert _check_rc(msm, tmp_path, src, one_key, "one_key", "--tol", "1e30") == 1
    assert "key sets differ" in capsys.readouterr().err

    assert _check_rc(msm, tmp_path, src, good.drop(index=5), "row_dropped", "--tol", "1e30") == 1
    assert "rows vs reference" in capsys.readouterr().err

    pred_only = good.copy()
    pred_only.loc[5, "prediction"] += np.float32(1e-3)
    assert _check_rc(msm, tmp_path, src, pred_only, "pred_only") == 1
    err = capsys.readouterr().err
    assert "prediction: max |d|" in err and "sigma:" not in err


def test_check_against_rejects_a_bucket_on_one_side_only(msm, tmp_path, capsys):
    """Same row count, but the reference renames a firm that is alone in its bucket to one
    hashing to a bucket the output leaves empty: each of those buckets exists on one side."""
    n_buckets = 997
    src, keys, _ = _make_arm(tmp_path, seeds=(60, 61))
    pattern = "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "ref"), "--arm", pattern]) == 0
    good = pd.read_parquet(tmp_path / "ref" / "forma_lap05_fgrid__pf_full__test__predictions.parquet")
    firms = sorted(keys["firm_id"].unique())
    b = dict(zip(firms, msm.bucket_ids(pa.array(firms), n_buckets)))
    counts = pd.Series(b).value_counts()
    alone = next(f for f in firms if counts[b[f]] == 1)
    new = next(f"{9000 + i:06d}" for i in range(1000)
               if msm.bucket_ids(pa.array([f"{9000 + i:06d}"]), n_buckets)[0] not in counts.index)
    assert counts[b[alone]] == 1 and msm.bucket_ids(pa.array([new]), n_buckets)[0] not in counts.index
    bad = good.copy()
    bad.loc[bad["firm_id"] == alone, "firm_id"] = new
    assert len(bad) == len(good) and (bad["firm_id"] == new).sum() > 0
    _write(bad, tmp_path / "ref_moved_firm.parquet")
    capsys.readouterr()
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "out"), "--buckets", str(n_buckets),
                     "--arm", pattern, "--check-against", str(tmp_path / "ref_moved_firm.parquet"),
                     "--tol", "1e30"]) == 1
    assert "present on one side only" in capsys.readouterr().err


def _f64_reference(keys, frames, seeds):
    """The true mixture as R13's DuckDB GROUP BY wrote it: f64 values (another summation
    order), string keys -- the shape of the published forma_lap05_fgrid__pf_full__test."""
    mus = np.vstack([frames[s]["prediction"].to_numpy().astype(np.float64) for s in reversed(seeds)])
    sig = np.vstack([frames[s]["sigma"].to_numpy().astype(np.float64) for s in reversed(seeds)])
    pred = mus.sum(axis=0) / len(seeds)
    ref = keys.copy()
    ref["prediction"] = pred
    ref["sigma"] = np.sqrt((sig ** 2).sum(axis=0) / len(seeds) + ((mus - pred) ** 2).sum(axis=0) / len(seeds))
    ref["model"] = "forma_lap05_fgrid"
    return ref


def test_check_against_f64_string_keyed_reference(msm, tmp_path, capsys):
    seeds = (60, 61, 62, 63, 64)
    src, keys, frames = _make_arm(tmp_path, seeds=seeds)
    pattern = "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"
    ref = _f64_reference(keys, frames, seeds).sample(frac=1.0, random_state=3)
    ref_path = tmp_path / "ref64.parquet"
    _write(ref, ref_path)
    schema = pq.read_schema(ref_path)
    assert schema.field("firm_id").type == pa.string() and schema.field("sigma").type == pa.float64()

    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o1"), "--buckets", "4", "--arm", pattern,
                     "--check-against", str(ref_path), "--max-f32-ulp", "1"]) == 0
    assert "max_f32_ulp_prediction" in capsys.readouterr().out
    out = tmp_path / "o1" / "forma_lap05_fgrid__pf_full__test__predictions.parquet"
    report = msm.check_against(out, ref_path, tmp_path / "chk", 4, 1000, max_f32_ulp=1)
    assert report["rows"] == len(keys)
    assert report["max_f32_ulp_prediction"] <= 1 and report["max_f32_ulp_sigma"] <= 1
    assert report["rows_f32_identical"] > report["rows_bit_identical"]
    # the f32 output is not the f64 reference, so the bit-identity default must fail
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o2"), "--arm", pattern,
                     "--check-against", str(ref_path)]) == 1

    for i, col in enumerate(VALUE_COLS):
        bad = ref.copy()
        bad.iloc[7, bad.columns.get_loc(col)] += 1e-4
        _write(bad, tmp_path / f"ref_bad_{col}.parquet")
        capsys.readouterr()
        assert msm.main(["--src", str(src), "--dest", str(tmp_path / f"bad{i}"), "--arm", pattern,
                         "--check-against", str(tmp_path / f"ref_bad_{col}.parquet"), "--max-f32-ulp", "1"]) == 1
        assert f"{col}: max f32 ulp" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        msm.main(["--src", str(src), "--dest", str(tmp_path / "o3"), "--arm", pattern,
                  "--check-against", str(ref_path), "--tol", "1", "--max-f32-ulp", "1"])
    with pytest.raises(SystemExit):
        msm.main(["--src", str(src), "--dest", str(tmp_path / "o4"), "--arm", pattern,
                  "--check-against", str(tmp_path / "missing.parquet"), "--max-f32-ulp", "1"])
    assert not (tmp_path / "o4" / "forma_lap05_fgrid__pf_full__test__predictions.parquet").exists()


def test_max_f32_ulp_boundary(msm, tmp_path, capsys):
    """A reference cell exactly 1 or 2 f32 ulps from the output (np.nextafter on the f32
    value, stored as f64): within N ulps passes, N + 1 fails."""
    src, _, _ = _make_arm(tmp_path, seeds=(60, 61))
    pattern = "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "ref"), "--arm", pattern]) == 0
    good = pd.read_parquet(tmp_path / "ref" / "forma_lap05_fgrid__pf_full__test__predictions.parquet")
    for col in VALUE_COLS:
        good[col] = good[col].astype(np.float64)
    for col in VALUE_COLS:
        for n_ulps, limit, rc in ((1, 0, 1), (1, 1, 0), (2, 1, 1), (2, 2, 0)):
            v = np.float32(good.loc[7, col])
            for _ in range(n_ulps):
                v = np.nextafter(v, np.float32(np.inf))
            bad = good.copy()
            bad.loc[7, col] = np.float64(v)
            name = f"ref_{col}_{n_ulps}_{limit}"
            _write(bad, tmp_path / f"{name}.parquet")
            capsys.readouterr()
            assert msm.main(["--src", str(src), "--dest", str(tmp_path / f"o_{name}"), "--arm", pattern,
                             "--check-against", str(tmp_path / f"{name}.parquet"),
                             "--max-f32-ulp", str(limit)]) == rc, (col, n_ulps, limit)
            if rc:
                assert f"{col}: max f32 ulp {n_ulps} > --max-f32-ulp {limit}" in capsys.readouterr().err


@pytest.mark.parametrize("flag, value", [("--tol", "-1"), ("--tol", "nan"), ("--tol", "inf"),
                                         ("--tol", "-inf"), ("--max-f32-ulp", "-1")])
def test_rejects_invalid_gate_thresholds(msm, tmp_path, capsys, flag, value):
    """A negative, NaN or infinite --tol (NaN passes every `>` test) or a negative
    --max-f32-ulp is refused before anything is materialized."""
    src, _, _ = _make_arm(tmp_path, seeds=(60, 61))
    ref = tmp_path / "ref.parquet"
    ref.write_bytes(b"")
    with pytest.raises(SystemExit):
        msm.main(["--src", str(src), "--dest", str(tmp_path / "out"),
                  "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet",
                  "--check-against", str(ref), f"{flag}={value}"])
    assert f"{flag} must be" in capsys.readouterr().err
    assert not (tmp_path / "out" / "forma_lap05_fgrid__pf_full__test__predictions.parquet").exists()


def test_check_against_nan_patterns(msm, tmp_path, capsys):
    """np.nanmax would hide a NaN facing a number; NaN facing NaN is a match."""
    src, _, frames = _make_arm(tmp_path, seeds=(60, 61))
    df = frames[60].copy()
    df.loc[4, "prediction"] = np.nan
    _write(df, src / "forma_lap05_fgrid__pf_full__60__test__predictions.parquet")
    pattern = "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "ref"), "--arm", pattern]) == 0
    ref = pd.read_parquet(tmp_path / "ref" / "forma_lap05_fgrid__pf_full__test__predictions.parquet")
    assert ref["prediction"].isna().sum() == 1 and ref["sigma"].isna().sum() == 1
    _write(ref.sample(frac=1.0, random_state=2), tmp_path / "ref_same.parquet")
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o0"), "--arm", pattern,
                     "--check-against", str(tmp_path / "ref_same.parquet")]) == 0
    report = msm.check_against(tmp_path / "o0" / "forma_lap05_fgrid__pf_full__test__predictions.parquet",
                               tmp_path / "ref_same.parquet", tmp_path / "chk", 4, 1000)
    assert report["rows_bit_identical"] == report["rows_f32_identical"] == report["rows"] == len(ref)

    for i, col in enumerate(VALUE_COLS):
        for j, fill in enumerate(("nan_where_finite", "finite_where_nan")):
            bad = ref.copy()
            if fill == "nan_where_finite":
                bad.loc[int(np.flatnonzero(bad[col].notna())[0]), col] = np.nan
            else:
                bad[col] = bad[col].fillna(0.5)
            path = tmp_path / f"ref_{col}_{fill}.parquet"
            _write(bad, path)
            for tol in ("0", "1e30"):
                capsys.readouterr()
                assert msm.main(["--src", str(src), "--dest", str(tmp_path / f"o{i}{j}"), "--arm", pattern,
                                 "--check-against", str(path), "--tol", tol]) == 1
                assert f"{col}: 1 rows NaN on one side only" in capsys.readouterr().err


def test_check_against_refuses_its_own_output(msm, tmp_path, capsys):
    """--check-against naming the file the arm writes would overwrite the reference and
    then compare the new file with itself (a vacuous pass)."""
    src, _, _ = _make_arm(tmp_path, seeds=(60, 61))
    pattern = "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"
    dest = tmp_path / "store"
    assert msm.main(["--src", str(src), "--dest", str(dest), "--arm", pattern]) == 0
    ref = dest / "forma_lap05_fgrid__pf_full__test__predictions.parquet"
    df = pd.read_parquet(ref)
    df.loc[0, "prediction"] += np.float32(1.0)
    _write(df, ref)
    before = ref.read_bytes()
    (dest / "sub").mkdir()
    for spelling in (ref, dest / "sub" / ".." / ref.name):
        capsys.readouterr()
        assert msm.main(["--src", str(src), "--dest", str(dest), "--arm", pattern,
                         "--check-against", str(spelling)]) == 1
        assert "is the file this arm writes" in capsys.readouterr().err
    assert ref.read_bytes() == before


def _scratch(dest):
    return sorted(p.name for p in dest.iterdir()
                  if p.name.startswith(("_mixture_tmp", "_mixture_check")) or p.suffix == ".tmp")


def test_rejected_runs_leave_no_scratch(msm, tmp_path):
    pattern = "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"
    (tmp_path / "a").mkdir()
    src, _, frames = _make_arm(tmp_path / "a", seeds=(60, 61))
    df = frames[61].copy()
    df.loc[len(df) - 1, "forecast_horizon"] = 9
    _write(df, src / "forma_lap05_fgrid__pf_full__61__test__predictions.parquet")
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o1"), "--buckets", "3", "--arm", pattern]) == 1
    assert _scratch(tmp_path / "o1") == []

    (tmp_path / "b").mkdir()
    src, _, frames = _make_arm(tmp_path / "b", seeds=(60, 61))
    _write(frames[60].drop(columns=["sigma"]), tmp_path / "ref_no_sigma.parquet")
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "o2"), "--arm", pattern,
                     "--check-against", str(tmp_path / "ref_no_sigma.parquet")]) == 1
    assert _scratch(tmp_path / "o2") == []


def test_row_count_mismatch_fails_before_partitioning(msm, tmp_path, capsys, monkeypatch):
    src, _, frames = _make_arm(tmp_path, seeds=(60, 61))
    _write(frames[61].iloc[1:], src / "forma_lap05_fgrid__pf_full__61__test__predictions.parquet")

    def boom(*a, **k):
        raise AssertionError("partitioned before the row-count check")
    monkeypatch.setattr(msm, "partition", boom)
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "out"),
                     "--arm", "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"]) == 1
    assert "seed row counts differ" in capsys.readouterr().err
    assert _scratch(tmp_path / "out") == []


def test_column_diff_infinity_mismatches(msm):
    """inf vs FLT_MAX and FLT_MAX vs an f64 value above the f32 range are 1 f32 ulp apart
    by ordinal, but they are infinity mismatches, not rounding."""
    fmax = np.finfo(np.float32).max
    x = np.array([fmax, np.inf, np.inf, -np.inf, 1.0, np.inf, np.nan, np.inf], dtype=np.float32)
    y = np.array([1e39, fmax, -np.inf, -np.inf, 1.0, 1e39, np.nan, np.inf], dtype=np.float64)
    n_nan, n_inf, max_d, max_ulp, eq, eq32 = msm.column_diff(x, y)
    assert (n_nan, n_inf, max_d, max_ulp) == (0, 4, 0.0, 0)
    np.testing.assert_array_equal(eq, [False, False, False, True, True, False, True, True])
    np.testing.assert_array_equal(eq32, [False, False, False, True, True, False, True, True])


def test_check_against_fails_on_infinity_mismatches(msm, tmp_path, capsys):
    src, _, frames = _make_arm(tmp_path, seeds=(60, 61))
    fmax = np.finfo(np.float32).max
    for s in (60, 61):
        df = frames[s].copy()
        df.loc[4, "prediction"] = fmax
        _write(df, src / f"forma_lap05_fgrid__pf_full__{s}__test__predictions.parquet")
    pattern = "forma_lap05_fgrid__pf_full__*__test__predictions.parquet"
    assert msm.main(["--src", str(src), "--dest", str(tmp_path / "ref"), "--arm", pattern]) == 0
    good = pd.read_parquet(tmp_path / "ref" / "forma_lap05_fgrid__pf_full__test__predictions.parquet")
    assert (good["prediction"] == fmax).sum() == 1
    good["prediction"] = good["prediction"].astype(np.float64)
    for i, bad_value in enumerate((np.inf, 1e39, -np.inf)):
        bad = good.copy()
        bad.loc[bad["prediction"] == fmax, "prediction"] = bad_value
        _write(bad, tmp_path / f"ref_{i}.parquet")
        capsys.readouterr()
        assert msm.main(["--src", str(src), "--dest", str(tmp_path / f"o{i}"), "--arm", pattern,
                         "--check-against", str(tmp_path / f"ref_{i}.parquet"), "--max-f32-ulp", "1"]) == 1
        assert "prediction: 1 rows with an infinity mismatch" in capsys.readouterr().err


def test_f32_ordinal_counts_ulps(msm):
    one = np.float32(1.0)
    up = np.nextafter(one, np.float32(2.0))
    x = np.array([one, up, np.float32(-0.0), np.float32(0.0)], dtype=np.float32)
    y = np.array([up, one, np.float32(0.0), -np.nextafter(np.float32(0.0), np.float32(1.0))], dtype=np.float32)
    np.testing.assert_array_equal(np.abs(msm.f32_ordinal(x) - msm.f32_ordinal(y)), [1, 1, 0, 1])


def test_mixture_name_and_seed(msm):
    assert msm.mixture_name("ffnn_large_b50__pf_full__63__test__predictions.parquet") == \
        "ffnn_large_b50__pf_full__test__predictions.parquet"
    assert msm.seed_of("forma_lap05_fgrid__pf_full__60__test__predictions.parquet") == 60
    with pytest.raises(msm.MixtureError):
        msm.seed_of("forma_lap05_fgrid__pf_full__test__predictions.parquet")
