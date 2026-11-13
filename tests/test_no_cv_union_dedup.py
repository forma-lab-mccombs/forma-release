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

"""`no_cv: true` must union train+val into a panel unique on (firm_id, account_id, quarter).

The tuple splits overlap by design: `create_time_based_splits_int_quarter` gives
val `pre_quarters_included` quarters of lead-in history that are also the tail of
train (#161). FormaDataModule.setup() used to concatenate the two as though they
were disjoint, so every overlap tuple became two nodes in the same training graph
-- double loss weight plus a copy-the-answer shortcut whenever only one twin was
masked (#279).

Tested at the consumer, not the producer: the pooled frame is *supposed* to have
duplicates, so asserting the splits are disjoint would be wrong before and after
the fix. What must hold is that no firm's `train_lookup` carries a repeated
(account_id, quarter) once setup() has run.

Built on synthetic tuples written to a tmp dir (fastparquet), with the metadata
that setup() would otherwise read off a processed build injected directly -- the
same pattern as the research repo's test_deterministic_inference_mask.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forma.data.forma_data import FormaDataModule, IdentitySchema
from forma.data.tuple_splits import TUPLE_KEY, union_tuple_splits

fastparquet = pytest.importorskip("fastparquet")
ENGINE = "fastparquet"

SCALE, REVT, COGS = 0, 1, 2
ACCOUNTS = (SCALE, REVT, COGS)
FEATURE_SET, TAG = "pf_full", "synthetic_overlap"
TRAIN_END, VAL_END = 2001, 2009
# The configured value (max_lookback 12 -> 11 quarters of lead-in), not the
# make_dataset fallback of 4.
PRE_Q = 11
BASE = pd.Timestamp("1969-12-31").to_period("Q")


# The dataset builder (research repo: src/data/dataset_creation.py) is not part of
# this release. These two reproduce its tuple splitters' boundary rules exactly,
# so the overlap the union must dedup is the one real builds have.
def create_time_based_splits_int_quarter(df, train_end_year, val_end_year,
                                         pre_quarters_included=0):
    train_q_max = (pd.Period(f"{train_end_year}Q4") - BASE).n
    val_q_max = (pd.Period(f"{val_end_year}Q4") - BASE).n
    q = df["quarter"]
    train_df = df[q <= train_q_max].copy()
    val_df = df[(q > train_q_max - pre_quarters_included) & (q <= val_q_max)].copy()
    test_df = df[q > val_q_max - pre_quarters_included].copy()
    return train_df, val_df, test_df


def create_firm_based_splits_int_quarter(df, val_end_year, holdout_firms,
                                         pre_quarters_included=0):
    val_q_max = (pd.Period(f"{val_end_year}Q4") - BASE).n
    pre_test = df["quarter"] <= val_q_max
    in_holdout = df["firm_id"].isin(holdout_firms)
    return (df[pre_test & ~in_holdout], df[pre_test & in_holdout],
            df[df["quarter"] > val_q_max - pre_quarters_included])


def _qi(year: int, q: int) -> int:
    return (pd.Period(f"{year}Q{q}") - BASE).n


def _panel(firms=(1, 2), years=range(1995, 2016)) -> pd.DataFrame:
    rows = [(f, a, _qi(y, q), float(f * 1000 + a * 100 + _qi(y, q)))
            for f in firms for y in years for q in range(1, 5) for a in ACCOUNTS]
    df = pd.DataFrame(rows, columns=["firm_id", "account_id", "quarter", "value"])
    df["firm_id"] = df["firm_id"].astype("int64")
    df["account_id"] = df["account_id"].astype("int64")
    df["quarter"] = df["quarter"].astype("int64")
    return df


def _time_splits(panel: pd.DataFrame):
    return create_time_based_splits_int_quarter(
        panel, train_end_year=TRAIN_END, val_end_year=VAL_END, pre_quarters_included=PRE_Q)


def _overlap_quarters() -> set[int]:
    """The 11 quarters (1999Q2..2001Q4) that appear in both train and val."""
    return set(range(_qi(TRAIN_END, 4) - PRE_Q + 1, _qi(TRAIN_END, 4) + 1))


# --------------------------------------------------------------------------- #
# The helper on its own                                                        #
# --------------------------------------------------------------------------- #

def test_time_splits_really_overlap():
    # Premise of everything below: the producer overlap is real and by design.
    tr, va, _ = _time_splits(_panel())
    comb = pd.concat([tr, va], ignore_index=True)
    dup_q = set(comb.loc[comb.duplicated(TUPLE_KEY, keep=False), "quarter"].tolist())
    assert dup_q == _overlap_quarters()
    assert comb.groupby(TUPLE_KEY).size().max() == 2


def test_union_is_unique_on_tuple_key_and_lossless():
    panel = _panel()
    tr, va, _ = _time_splits(panel)
    union = union_tuple_splits(tr, va)

    assert not union.duplicated(TUPLE_KEY).any()
    # Nothing but the duplicates is dropped: the union is exactly the panel
    # restricted to the train+val quarter range.
    expected = panel[panel["quarter"] <= _qi(VAL_END, 4)]
    left = union.sort_values(TUPLE_KEY).reset_index(drop=True)
    right = expected.sort_values(TUPLE_KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_like=True)
    # Lead-in rows are kept once, not removed: the union still reaches back to
    # the panel's first quarter.
    assert int(union["quarter"].min()) == int(panel["quarter"].min())


def test_union_is_a_noop_for_firm_disjoint_splits():
    # The firm-based splitter puts train and val on the SAME quarter range with
    # disjoint firms. Deduping by key must keep every row -- the quarter-boundary
    # filter the issue warns against would empty val here.
    panel = _panel(firms=tuple(range(1, 11)))
    tr, va, _ = create_firm_based_splits_int_quarter(
        panel, val_end_year=VAL_END, holdout_firms={2, 5, 9},
        pre_quarters_included=PRE_Q)
    assert set(tr["firm_id"]).isdisjoint(set(va["firm_id"]))
    union = union_tuple_splits(tr, va)
    assert len(union) == len(tr) + len(va)
    assert set(union["firm_id"]) == set(tr["firm_id"]) | set(va["firm_id"])


def test_union_raises_when_overlap_copies_disagree():
    # The helper assumes the two splits are masks of one panel and checks it:
    # a per-split transform applied to only one file must fail loudly, not
    # train on whichever copy sorted first (forecast_to_usd.py precedent).
    tr, va, _ = _time_splits(_panel())
    va = va.copy()
    q_over = min(_overlap_quarters())
    bad = (va["quarter"] == q_over) & (va["account_id"] == REVT)
    assert bad.sum() == 2
    va.loc[bad, "value"] += 1.0
    with pytest.raises(ValueError, match="2 of .* disagree on 'value'"):
        union_tuple_splits(tr, va)
    # ... and is skippable for a caller that has already verified.
    out = union_tuple_splits(tr, va, verify_values=False)
    assert not out.duplicated(TUPLE_KEY).any()


def test_union_value_check_treats_missing_as_agreement_for_every_dtype():
    # Identical copies carrying NaN / NaT / None in a non-key column are copies
    # of one panel and must not trip the disagreement check (round-2 review:
    # the float-only NaN branch raised a false alarm on object and datetime).
    tr, va, _ = _time_splits(_panel())
    over = va["quarter"].isin(_overlap_quarters())
    for frame in (tr, va):
        frame["value"] = frame["value"].where(frame["account_id"] != COGS, np.nan)
        frame["note"] = np.where(frame["account_id"] == REVT, "x", None)
        frame["dt"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        # pandas masked extension dtypes: to_numpy() yields object arrays holding
        # pd.NA, whose `==` result cannot be cast to bool (round-3 review).
        frame["flag"] = pd.array(np.where(frame["account_id"] == SCALE, True, None),
                                 dtype="boolean")
        frame["tag"] = pd.array(np.where(frame["account_id"] == COGS, "c", None),
                                dtype="string")
    assert over.any()
    out = union_tuple_splits(tr, va)
    assert not out.duplicated(TUPLE_KEY).any()
    assert len(out) == len(tr) + len(va) - int(over.sum())
    # A real disagreement on an object column still raises.
    va2 = va.copy()
    va2.loc[over & (va2["account_id"] == REVT), "note"] = "y"
    with pytest.raises(ValueError, match="disagree on 'note'"):
        union_tuple_splits(tr, va2)


def test_union_rejects_a_split_that_repeats_a_key_within_the_band():
    # The precondition behind "same as drop_duplicates": each split is itself
    # unique on the key. Enforced where it is cheap -- the shared quarter band.
    first = pd.DataFrame({"firm_id": [1, 1, 1], "account_id": [0, 0, 0],
                          "quarter": [10, 10, 11], "value": [1.0, 1.0, 2.0]})
    second = pd.DataFrame({"firm_id": [1, 1], "account_id": [0, 0],
                           "quarter": [10, 12], "value": [1.0, 3.0]})
    with pytest.raises(ValueError, match="first split carries 1 rows that repeat"):
        union_tuple_splits(first, second)
    with pytest.raises(ValueError, match="second split carries 1 rows that repeat"):
        union_tuple_splits(second.iloc[[0]], first.iloc[[0, 1]])
    # A repeat entirely outside the shared band is out of reach and passes through.
    outside = pd.DataFrame({"firm_id": [1, 1], "account_id": [0, 0],
                            "quarter": [11, 11], "value": [2.0, 2.0]})
    out = union_tuple_splits(second.iloc[[0]], outside)  # band empty: 10 vs 11
    assert len(out) == 3


def test_union_rejects_non_integer_or_null_key_columns():
    # Guards run over the whole frame, before banding, so they hold whether or
    # not the frames overlap. Each cause gets its own message.
    base = pd.DataFrame({"firm_id": [1, 2], "account_id": [0, 0], "quarter": [10, 11],
                         "value": [1.0, 2.0]})
    far = base.assign(quarter=[500, 501])  # no shared quarter band with `base`

    as_str = base.assign(firm_id=["001000", "001001"])
    with pytest.raises(ValueError, match="'firm_id' in second has non-integer dtype object"):
        union_tuple_splits(base, as_str)
    with pytest.raises(ValueError, match="non-integer dtype object"):
        union_tuple_splits(as_str, far)  # still raised with no overlap band

    as_float = base.assign(firm_id=[1.0, 1.4])  # would truncate to the same key
    with pytest.raises(ValueError, match="'firm_id' in second has non-integer dtype float64"):
        union_tuple_splits(base, as_float)

    with_null = base.assign(firm_id=pd.array([1, None], dtype="Int64"))
    with pytest.raises(ValueError, match="'firm_id' in first contains 1 null"):
        union_tuple_splits(with_null, far)

    # uint64 above int64.max would wrap negative on the packing cast; say so,
    # instead of the misleading "contains negative ids".
    huge = base.assign(firm_id=np.array([1, 2 ** 63 + 5], dtype=np.uint64))
    with pytest.raises(ValueError, match="'firm_id' in second holds unsigned values above"):
        union_tuple_splits(base, huge)
    small_u = base.assign(firm_id=np.array([1, 2], dtype=np.uint64))  # in range: fine
    assert len(union_tuple_splits(base, small_u)) == 2

    # Nullable Int64 WITHOUT nulls is fine, and matches the numpy-int result.
    nullable = base.astype({"firm_id": "Int64", "account_id": "Int64", "quarter": "Int64"})
    # second's first row is a copy of base's second row (same key, same value).
    shifted = base.assign(firm_id=[2, 3], quarter=[11, 12], value=[2.0, 3.0])
    out = union_tuple_splits(nullable, shifted)
    assert len(out) == 3
    assert out["firm_id"].tolist() == [1, 2, 3]


def test_union_packs_keys_with_widths_shared_across_frames():
    # Round-2 review regression: sizing the packing widths per frame let two
    # DISTINCT keys with different per-frame column maxima pack to the same
    # code -- (firm 3, acct 1) vs (firm 1, acct 3) both became 122 -- and a
    # real row was silently dropped. Widths must be shared across frames.
    first = pd.DataFrame({"firm_id": [3], "account_id": [1], "quarter": [10], "value": [5.0]})
    second = pd.DataFrame({"firm_id": [1], "account_id": [3], "quarter": [10], "value": [5.0]})
    out = union_tuple_splits(first, second)
    assert len(out) == 2
    # And the general statement over a panel whose two frames have different
    # per-column maxima: identical to drop_duplicates for many random pairings.
    rng = np.random.default_rng(0)
    for _ in range(50):
        n1, n2 = rng.integers(1, 40, size=2)
        mk = lambda n: pd.DataFrame({
            "firm_id": rng.integers(0, 2 ** rng.integers(1, 12), size=n),
            "account_id": rng.integers(0, 2 ** rng.integers(1, 6), size=n),
            "quarter": rng.integers(100, 100 + 2 ** rng.integers(1, 5), size=n),
            "value": 1.0}).drop_duplicates(TUPLE_KEY, ignore_index=True)
        a, b = mk(n1), mk(n2)
        want = pd.concat([a, b], ignore_index=True).drop_duplicates(TUPLE_KEY, keep="first",
                                                                    ignore_index=True)
        pd.testing.assert_frame_equal(union_tuple_splits(a, b), want)


def test_union_matches_drop_duplicates_and_filters_second_frame():
    # Semantics pin: identical rows out to concat + drop_duplicates(keep='first'),
    # in the same order. The memory story is that `second` is filtered before
    # the concat, so `first` must come through untouched at the top.
    tr, va, _ = _time_splits(_panel())
    got = union_tuple_splits(tr, va)
    want = pd.concat([tr, va], ignore_index=True).drop_duplicates(
        TUPLE_KEY, keep="first", ignore_index=True)
    pd.testing.assert_frame_equal(got, want)
    pd.testing.assert_frame_equal(got.iloc[: len(tr)].reset_index(drop=True),
                                  tr.reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Through FormaDataModule.setup('fit')                                         #
# --------------------------------------------------------------------------- #

def _account_map() -> pd.DataFrame:
    return pd.DataFrame({"account_name": ["scale", "revtq", "cogsq"],
                         "account_id": list(ACCOUNTS)})


def _write_splits(data_dir: Path, tr: pd.DataFrame, va: pd.DataFrame) -> None:
    for kind, frame in (("tuple_train", tr), ("tuple_val", va)):
        frame.to_parquet(data_dir / f"{kind}__{FEATURE_SET}__{TAG}.parquet",
                         engine=ENGINE, index=False)


def _datamodule(data_dir: Path, no_cv: bool) -> FormaDataModule:
    """A FormaDataModule pointed at tmp tuple splits, with the small metadata that
    setup() would read off a processed build injected so only the split path is
    exercised. Re-check the injected block if setup()'s preamble changes."""
    amap = _account_map()
    amap_path = data_dir / "account_id_map.csv"
    amap.to_csv(amap_path, index=False)

    data_config = {
        "feature_set": FEATURE_SET,
        "dataset_tag": TAG,
        "no_cv": no_cv,
        "batch_size": 4,
        "num_workers": 0,
        "use_future_grid": True,
        "future_grid_sample_overhang": False,
        "seed": 60,
        "time_splits": {"train_end_year": TRAIN_END, "val_end_year": VAL_END},
        "curriculum": {"max_lookback": 12, "max_horizon": 20,
                       "min_lookback": 1, "min_horizon": 1},
    }
    dm = FormaDataModule(
        data_dir_path=data_dir,
        account_map_path=amap_path,
        firm_id_map_path=data_dir / "firm_id_map.csv",
        data_config=data_config,
    )
    # Skip the reg-stats parquet / identities json / industry map reads.
    dm.reg_stats_by_q = {}
    dm.scale_account_id = SCALE
    dm.identity_schema = IdentitySchema([], amap)
    dm.firm_industry_lookup = {}
    return dm


def _lookup_has_repeated_nodes(lookup: dict) -> dict[int, int]:
    """firm_id -> number of repeated (account_id, quarter) pairs in its arrays."""
    out = {}
    for firm_id, arrs in lookup.items():
        pairs = pd.DataFrame({"account_id": arrs["account_id"], "quarter": arrs["quarter"]})
        n = int(pairs.duplicated().sum())
        if n:
            out[int(firm_id)] = n
    return out


def test_setup_no_cv_train_lookup_has_no_duplicate_nodes(tmp_path):
    panel = _panel()
    tr, va, _ = _time_splits(panel)
    _write_splits(tmp_path, tr, va)

    dm = _datamodule(tmp_path, no_cv=True)
    dm.setup("fit")

    assert dm.val_lookup == {} and dm.val_index == []
    assert _lookup_has_repeated_nodes(dm.train_lookup) == {}

    # Every firm's arrays hold exactly the panel's train+val-range tuples, once.
    expected = panel[panel["quarter"] <= _qi(VAL_END, 4)]
    for firm_id, grp in expected.groupby("firm_id"):
        arrs = dm.train_lookup[firm_id]
        assert len(arrs["quarter"]) == len(grp)
        got = pd.DataFrame({"account_id": arrs["account_id"], "quarter": arrs["quarter"],
                            "value": arrs["value"]}).sort_values(["quarter", "account_id"])
        want = grp[["account_id", "quarter", "value"]].sort_values(["quarter", "account_id"])
        np.testing.assert_array_equal(got["quarter"].values, want["quarter"].values)
        np.testing.assert_array_equal(got["account_id"].values, want["account_id"].values)
        np.testing.assert_allclose(got["value"].values, want["value"].values, rtol=0, atol=0)

    # The union spans one boundary (2009Q4), not two: origins run from the
    # panel's first eligible quarter to the end of val, once per firm-quarter.
    assert dm.train_global_max_q == _qi(VAL_END, 4)
    q_min, q_max = int(expected["quarter"].min()), int(expected["quarter"].max())
    n_origins_per_firm = (q_max - dm.min_horizon) - (q_min + dm.min_lookback - 1) + 1
    assert len(dm.train_index) == n_origins_per_firm * expected["firm_id"].nunique()
    assert len(set(dm.train_index)) == len(dm.train_index)


def test_setup_no_cv_sample_window_has_no_duplicate_nodes(tmp_path):
    # The consumer-facing statement of the bug: a window centred inside the
    # overlap used to carry each overlap tuple twice.
    tr, va, _ = _time_splits(_panel())
    _write_splits(tmp_path, tr, va)
    dm = _datamodule(tmp_path, no_cv=True)
    dm.setup("fit")

    firm, center = 1, _qi(2000, 3)  # 6 overlap quarters in history, 5 in the horizon
    assert (firm, center) in dm.train_slice_cache
    lo, hi = dm.train_slice_cache[(firm, center)]
    assert hi > lo, "window must not be empty -- the assertions below would be vacuous"
    arrs = dm.train_lookup[firm]
    window = pd.DataFrame({"account_id": arrs["account_id"][lo:hi],
                           "quarter": arrs["quarter"][lo:hi]})
    assert not window.duplicated().any()
    # Full window: max_lookback history quarters (incl. the origin) + max_horizon
    # future quarters, every account present exactly once in each.
    n_q_expected = dm.max_lookback + dm.max_horizon
    assert int(window["quarter"].nunique()) == n_q_expected
    assert len(window) == n_q_expected * len(ACCOUNTS)


def test_setup_cv_path_is_untouched(tmp_path):
    # With no_cv off the two splits are built separately and each keeps its
    # lead-in rows -- the dedup must not reach the single-split readers. (The
    # val reader's frame minimum sets its first origin; test origins are floored.)
    tr, va, _ = _time_splits(_panel())
    _write_splits(tmp_path, tr, va)
    dm = _datamodule(tmp_path, no_cv=False)
    dm.setup("fit")

    assert dm.train_lookup and dm.val_lookup
    for firm_id in dm.val_lookup:
        assert len(dm.val_lookup[firm_id]["quarter"]) == int((va["firm_id"] == firm_id).sum())
        assert int(dm.val_lookup[firm_id]["quarter"].min()) == int(va["quarter"].min())
    assert dm.val_global_max_q == _qi(VAL_END, 4)
    assert dm.train_global_max_q == _qi(TRAIN_END, 4)


def test_setup_test_split_keeps_lead_in_and_first_origin(tmp_path):
    # The single-split reader must never dedup or trim: the lead-in rows are the
    # history of the first test origins. The origin INDEX is floored at the first
    # test quarter, though -- the lead-in quarters are history, not origins.
    panel = _panel()
    tr, va, te = _time_splits(panel)
    _write_splits(tmp_path, tr, va)
    te.to_parquet(tmp_path / f"tuple_test__{FEATURE_SET}__{TAG}.parquet",
                  engine=ENGINE, index=False)
    dm = _datamodule(tmp_path, no_cv=True)
    dm.setup("test")

    te_min = int(te["quarter"].min())
    assert te_min == _qi(VAL_END, 4) - PRE_Q + 1  # lead-in present in the file ...
    for firm_id, grp in te.groupby("firm_id"):
        arrs = dm.test_lookup[firm_id]
        assert len(arrs["quarter"]) == len(grp)          # ... and every row kept
        assert int(arrs["quarter"].min()) == te_min
    first_center = min(q for _, q in dm.test_index)
    assert first_center == _qi(VAL_END + 1, 1)
    # Without the floor the lead-in would have produced origins from here on.
    assert te_min + dm.min_lookback - 1 < first_center
    # Every firm's first test origin still gets its full max_lookback window.
    for firm_id in te["firm_id"].unique():
        s, e = dm.test_slice_cache[(firm_id, first_center)]
        qs = dm.test_lookup[firm_id]["quarter"][s:e]
        assert int(qs.min()) == first_center - dm.max_lookback + 1
    assert dm.test_global_max_q == int(te["quarter"].max())


def test_setup_test_notes_a_floor_past_the_files_lead_in(tmp_path, capsys):
    # A config val_end_year later than the build's moves the floor past the
    # file's lead-in, silently dropping real test origins; setup('test') must
    # run the check that says so.
    panel = _panel()
    tr, va, te = _time_splits(panel)
    _write_splits(tmp_path, tr, va)
    te.to_parquet(tmp_path / f"tuple_test__{FEATURE_SET}__{TAG}.parquet",
                  engine=ENGINE, index=False)
    dm = _datamodule(tmp_path, no_cv=True)
    dm.data_config["time_splits"] = {"train_end_year": TRAIN_END, "val_end_year": VAL_END + 2}
    dm.setup("test")

    assert "dropping real test origins" in capsys.readouterr().out
    assert min(q for _, q in dm.test_index) == _qi(VAL_END + 3, 1)
