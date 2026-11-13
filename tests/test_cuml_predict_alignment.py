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

"""cuML-style estimator outputs must line up with their rows by position.

cuML 25.12, given a pandas frame, returns predictions as a pd.Series with a fresh
0..n-1 index. MultiHorizonBaselineModel.predict() builds its wide frame with
``index=X_test_clean.index``, and pandas aligns a Series on labels. So before the
fix, a test frame whose index was not 0..n-1 broke silently. An offset index came
out all-NaN. A gappy one (dropped NaN-feature rows) came out shifted onto the
wrong firm-quarters and partly blanked: the rows past the last kept position got
NaN. A permuted index came out scrambled. None of these raised. sklearn returns
numpy and never hit this.

These tests wrap fitted sklearn models so that their predict() returns exactly
what cuML does. They then require the forecast, both in memory and streamed, to
match predictions taken straight from the estimators and keyed by firm_id. A
cuML-gated variant repeats this against a real cuML RF where cuML is installed. They also pin
the cuML seed fix: a constructor that takes ``**kwargs`` but names neither
``seed`` nor ``random_state`` must still receive ``random_state``, and a dropped
seed must warn.
"""
import numpy as np
import pandas as pd
import pytest

# The baselines need the [competitors] extra (scikit-learn); CI's core job omits it.
pytest.importorskip("sklearn")

import forma.models.forecast_io as fio  # noqa: E402
from forma.models.baselines import (  # noqa: E402
    SklearnBaselineModel,
    _positional_predictions,
    _set_cuml_random_forest_seed_param,
)

TARGETS, HORIZONS = ["niq", "revtq"], list(range(1, 13))
SORT_KEYS = ["target", "forecast_horizon", "firm_id", "quarter"]


def _make_df(n, seed, index=None):
    rng = np.random.RandomState(seed)
    data = {"f1": rng.randn(n).astype("float32"), "f2": rng.randn(n).astype("float32")}
    for t in TARGETS:
        for h in HORIZONS:
            data[f"{t}_t{h}"] = (rng.randn(n) + 2.0 * data["f1"] - data["f2"] + 0.1 * h).astype("float32")
    df = pd.DataFrame(data, index=index)
    df["firm_id"] = np.arange(n, dtype="int64")          # unique keys: no sort ties
    df["quarter"] = (np.arange(n) % 20).astype("int64")
    return df


class _CumlLikeOutput:
    """Wraps a fitted estimator so predict() returns a reset-index pd.Series, as
    cuML 25.12 does for pandas input."""

    def __init__(self, inner):
        self.inner = inner

    def predict(self, X):
        return pd.Series(np.asarray(self.inner.predict(X)))   # fresh RangeIndex


def _fitted():
    m = SklearnBaselineModel(model_type="ridge")
    m.fit(_make_df(300, 0), _make_df(150, 1))
    return m


def _sorted(df):
    return df.sort_values(SORT_KEYS).reset_index(drop=True)


def _forecast(model, test, tmp_path=None):
    """In-memory forecast, plus (when tmp_path is given) the streamed one."""
    out = [_sorted(fio.finalize_forecast_frame(model.predict(test)))]
    if tmp_path is not None:
        p = tmp_path / "streamed.parquet"
        assert model.predict(test, stream_save_paths=[p], stream_group_size=1) is None
        out.append(_sorted(pd.read_parquet(p, engine="fastparquet")))
    return out


def _expected(model, test):
    """Absolute expectation, independent of predict(): each fitted estimator called
    directly on the kept rows (no NaN feature), keyed by those rows' firm_id."""
    X = test[model.feature_columns]
    keep = ~X.isna().any(axis=1)
    Xc = X[keep].reset_index(drop=True)
    firms = test.loc[keep, "firm_id"].to_numpy()
    parts = []
    for col, est in model.models_by_horizon.items():
        target, h = col.rsplit("_t", 1)
        parts.append(pd.DataFrame({
            "target": target, "forecast_horizon": int(h), "firm_id": firms,
            "expected": np.asarray(est.predict(Xc), dtype="float64").astype("float32"),
        }))
    return pd.concat(parts, ignore_index=True)


def _assert_matches_expected(got, expected):
    m = got.merge(expected, on=["target", "forecast_horizon", "firm_id"], how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), m["_merge"].value_counts().to_dict()
    np.testing.assert_allclose(m["prediction"].to_numpy(), m["expected"].to_numpy(), rtol=0, atol=1e-6)


def _test_frames():
    n = 120
    offset = _make_df(n, 2, index=pd.RangeIndex(1000, 1000 + n))   # e.g. a filtered slice
    gappy = _make_df(n, 3)
    gappy.loc[[5, 17, 60], "f1"] = np.nan                          # predict() drops these rows
    shuffled = _make_df(n, 4, index=np.random.RandomState(9).permutation(n))
    return {"offset_index": offset, "dropped_nan_rows": gappy, "permuted_index": shuffled}


@pytest.mark.parametrize("case", ["offset_index", "dropped_nan_rows", "permuted_index"])
def test_cuml_like_output_aligns_by_position(case, tmp_path):
    test = _test_frames()[case]
    model = _fitted()
    expected = _expected(model, test)                     # straight from the estimators
    (reference,) = _forecast(model, test)                 # sklearn numpy output
    _assert_matches_expected(reference, expected)

    model.models_by_horizon = {k: _CumlLikeOutput(v) for k, v in model.models_by_horizon.items()}
    in_mem, streamed = _forecast(model, test, tmp_path)
    for got in (in_mem, streamed):
        assert got["prediction"].notna().all()
        _assert_matches_expected(got, expected)
        pd.testing.assert_frame_equal(got, reference)

    if case == "dropped_nan_rows":
        assert set(reference["firm_id"]) == set(range(120)) - {5, 17, 60}


@pytest.mark.parametrize("case", ["offset_index", "dropped_nan_rows", "permuted_index"])
def test_real_cuml_rf_aligns_by_position(case, tmp_path):
    """Same contract against a real cuML RandomForestRegressor (cuML hosts only;
    skipped wherever cuML is not installed). The forests are fit on pandas, as
    MultiHorizonBaselineModel.fit does. The test then asserts that cuML's
    pandas-input output really carries a fresh 0..n-1 index, so the behaviour the
    fix guards against is observed, not assumed. If a future cuML stops doing that,
    the final assertion fails loudly instead of letting the test pass trivially."""
    cuml = pytest.importorskip("cuml")
    from cuml.ensemble import RandomForestRegressor as CumlRF

    train = _make_df(300, 0)
    model = _fitted()                         # supplies feature/target columns + predict()
    feats = model.feature_columns
    model.models_by_horizon = {
        col: CumlRF(n_estimators=5, max_depth=4, random_state=0).fit(train[feats], train[col])
        for col in model.models_by_horizon
    }
    test = _test_frames()[case]
    X = test[feats]
    keep = ~X.isna().any(axis=1)
    firms = test.loc[keep, "firm_id"].to_numpy()
    expected = pd.concat([
        pd.DataFrame({"target": col.rsplit("_t", 1)[0], "forecast_horizon": int(col.rsplit("_t", 1)[1]),
                      "firm_id": firms,
                      # numpy input -> numpy output: positional by construction
                      "expected": np.asarray(est.predict(X[keep].to_numpy(np.float32))).reshape(-1).astype("float32")})
        for col, est in model.models_by_horizon.items()
    ], ignore_index=True)
    for got in _forecast(model, test, tmp_path):
        _assert_matches_expected(got, expected)

    raw = next(iter(model.models_by_horizon.values())).predict(X[keep])
    observed = f"cuML {cuml.__version__}: pandas-input predict -> {type(raw).__name__}, index={getattr(raw, 'index', None)!r}"
    print(observed)
    assert isinstance(raw, pd.Series) and raw.index.equals(pd.RangeIndex(int(keep.sum()))), (
        f"cuML output for pandas input changed ({observed}); this test no longer exercises "
        "the reset-index path. Revisit _positional_predictions' docstring and this test.")


def test_positional_predictions_shapes():
    s = pd.Series([1.0, 2.0, 3.0], index=[7, 8, 9])
    np.testing.assert_array_equal(_positional_predictions(s, 3), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_positional_predictions(np.array([[1.0], [2.0]]), 2), [1.0, 2.0])
    with pytest.raises(ValueError):
        _positional_predictions(np.zeros(4), 3)


class _FakeCupy:          # device array: no to_numpy, np.asarray refuses it, .get() copies to host
    def __init__(self, a):
        self._a = np.asarray(a)

    def get(self):
        return self._a

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed")


class _FakeCudfSeries:    # cudf.Series: to_numpy() to host, carrying its own index
    def __init__(self, a):
        self._a, self.index = np.asarray(a), pd.RangeIndex(500, 500 + len(a))

    def to_numpy(self):
        return self._a


def test_positional_predictions_device_outputs():
    np.testing.assert_array_equal(_positional_predictions(_FakeCupy([4.0, 5.0]), 2), [4.0, 5.0])
    np.testing.assert_array_equal(_positional_predictions(_FakeCudfSeries([[6.0], [7.0]]), 2), [6.0, 7.0])


class _KwargsOnly:            # cuML 25.12: RandomForestRegressor(self, **kwargs)
    def __init__(self, **kwargs):
        pass


class _NamedSeed:             # older RAPIDS
    def __init__(self, n_estimators=100, seed=None):
        pass


class _NamedRandomState:
    def __init__(self, n_estimators=100, random_state=None, **kwargs):
        pass


class _NamedPlusKwargs:       # named non-seed params AND **kwargs (no seed name)
    def __init__(self, *, split_criterion=2, handle=None, verbose=False, **kwargs):
        pass


@pytest.mark.parametrize("cls, key", [(_KwargsOnly, "random_state"),
                                      (_NamedSeed, "seed"),
                                      (_NamedRandomState, "random_state"),
                                      (_NamedPlusKwargs, "random_state")])
def test_cuml_seed_param_reaches_constructor(cls, key):
    params = {"n_estimators": 10}
    _set_cuml_random_forest_seed_param(cls, params, 62)
    assert params.get(key) == 62
    cls(**params)   # the chosen name is accepted


class _Recording:             # (**kwargs) like cuML 25.12; records what it was built with
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _RejectsSeed:           # a RAPIDS build that refuses seed-like kwargs
    def __init__(self, **kwargs):
        if "random_state" in kwargs or "seed" in kwargs:
            raise TypeError("unexpected keyword argument 'random_state'")
        self.kwargs = kwargs


class _NoSeedNames:           # named params only, neither seed nor random_state
    def __init__(self, n_estimators=100, max_depth=16):
        self.kwargs = {"n_estimators": n_estimators, "max_depth": max_depth}


def _rf_via_factory(monkeypatch, estimator_cls):
    """The real _create_single_model path, with cuML replaced by a stand-in."""
    import forma.models.baselines as baselines
    monkeypatch.setattr(baselines, "CUML_AVAILABLE", True)
    monkeypatch.setattr(baselines, "cuMLRandomForestRegressor", estimator_cls)
    m = SklearnBaselineModel(model_type="random_forest", random_state=62, n_jobs=-1, n_estimators=7)
    return m._create_single_model()


def test_factory_passes_seed_to_kwargs_only_cuml(monkeypatch):
    est = _rf_via_factory(monkeypatch, _Recording)
    assert est.kwargs == {"n_estimators": 7, "random_state": 62}   # n_jobs stripped, seed kept


def test_factory_warns_when_cuml_rejects_seed(monkeypatch):
    with pytest.warns(RuntimeWarning, match="UNSEEDED"):
        est = _rf_via_factory(monkeypatch, _RejectsSeed)
    assert est.kwargs == {"n_estimators": 7}


class _RejectsEverything:     # a TypeError that has nothing to do with the seed
    def __init__(self, **kwargs):
        raise TypeError("bad max_features")


def test_factory_non_seed_typeerror_raises_without_seed_warning(monkeypatch, recwarn):
    with pytest.raises(TypeError, match="bad max_features"):
        _rf_via_factory(monkeypatch, _RejectsEverything)
    assert not [w for w in recwarn if "UNSEEDED" in str(w.message)]


def test_factory_warns_when_cuml_names_no_seed(monkeypatch):
    with pytest.warns(RuntimeWarning, match="UNSEEDED"):
        est = _rf_via_factory(monkeypatch, _NoSeedNames)
    assert est.kwargs == {"n_estimators": 7, "max_depth": 16}


def test_cuml_seed_param_none_is_noop():
    params = {}
    _set_cuml_random_forest_seed_param(_KwargsOnly, params, None)
    assert params == {}
