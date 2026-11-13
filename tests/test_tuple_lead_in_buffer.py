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

"""Test-origin floor and lead-in runtime guard on the tuple splits (#161, #289).

The tuple val/test split files carry ``max_lookback - 1`` quarters of lead-in
history before their split boundary so the boundary origin gets its full window.
Two runtime behaviours of ``FormaDataModule`` depend on it:

1. ``setup('test')`` floors the test origin index at the first test-period quarter
   (``first_test_origin_q``): left unfloored, every lead-in quarter past the first
   ``min_lookback - 1`` would itself be enumerated as a forecast origin, on a
   truncated window. The lead-in rows stay in the lookup as history.
2. ``_warn_if_lead_in_too_short`` warns when a config's ``max_lookback`` outruns
   the lead-in a reused dataset was built with -- the case a build-time check
   structurally cannot see.

Adapted from the research repo's test_tuple_lead_in_buffer.py. The build-side
buffer sizing it also pins lives in the dataset builder, which this release does
not ship, so the splits here come from a local copy of the builder's boundary
rule with the buffer the builder would choose (``max_lookback - 1``).
"""
from __future__ import annotations

import pandas as pd
import pytest

from forma.data.quarters import qoff_to_period_index

BASE = pd.Timestamp("1969-12-31").to_period("Q")


def _pre_q(max_lookback: int) -> int:
    """The lead-in the builder sizes from curriculum.max_lookback (#161)."""
    return max(int(max_lookback) - 1, 0)


def create_time_based_splits_int_quarter(df, train_end_year, val_end_year,
                                         pre_quarters_included=0):
    """The builder's time-based tuple splitter (boundary rule only)."""
    train_q_max = (pd.Period(f"{train_end_year}Q4") - BASE).n
    val_q_max = (pd.Period(f"{val_end_year}Q4") - BASE).n
    q = df["quarter"]
    return (df[q <= train_q_max].copy(),
            df[(q > train_q_max - pre_quarters_included) & (q <= val_q_max)].copy(),
            df[q > val_q_max - pre_quarters_included].copy())


def create_firm_based_splits_int_quarter(df, val_end_year, holdout_firms,
                                         pre_quarters_included=0):
    """The builder's firm-based tuple splitter (boundary rule only)."""
    val_q_max = (pd.Period(f"{val_end_year}Q4") - BASE).n
    pre_test = df["quarter"] <= val_q_max
    in_holdout = df["firm_id"].isin(holdout_firms)
    return (df[pre_test & ~in_holdout], df[pre_test & in_holdout],
            df[df["quarter"] > val_q_max - pre_quarters_included])


def _dense_panel(firm=1, years=range(1995, 2016)) -> pd.DataFrame:
    return pd.DataFrame(
        [(firm, 0, (pd.Period(f"{y}Q{q}") - BASE).n, 1.0)
         for y in years for q in range(1, 5)],
        columns=["firm_id", "account_id", "quarter", "value"],
    )


@pytest.mark.parametrize("max_lookback,first_origin_label",
                         [(4, "2010Q1"), (12, "2008Q1"), (20, "2006Q1")])
def test_buffer_also_moves_the_first_emitted_forecast_origin(max_lookback,
                                                             first_origin_label):
    """Unfloored, the lead-in quarters become origins too.

    Driven through the real `FormaDataModule._build_lookup_and_index` rather than
    re-deriving `start_center` here -- a test that recomputes the formula it is
    pinning would still pass if that formula changed, which is the whole thing it
    is supposed to catch.

    The 12-quarter column landing on 2008Q1 is what forecasts made before the
    floor existed carry. `setup('test')` passes a floor; see the next test.
    """
    pre_q = _pre_q(max_lookback)
    _, _, test_df = create_time_based_splits_int_quarter(
        _dense_panel(), train_end_year=2001, val_end_year=2009,
        pre_quarters_included=pre_q)

    dm = _stub_datamodule(max_lookback)
    _, index, _ = dm._build_lookup_and_index(test_df)

    first_origin = min(center_q for _, center_q in index)
    assert str(qoff_to_period_index([first_origin])[0]) == first_origin_label


@pytest.mark.parametrize("max_lookback", [4, 12, 20])
def test_test_origin_floor_pins_first_origin_to_the_test_period(max_lookback):
    """With the floor `setup('test')` passes, test origins start at 2010Q1 for any
    buffer, and the lead-in rows remain in the lookup as the 2010Q1 window's history.

    At max_lookback == min_lookback (4) the unfloored enumeration already starts at
    2010Q1, so that case is a no-regression check; the floor binds at 12 and 20."""
    pre_q = _pre_q(max_lookback)
    _, _, test_df = create_time_based_splits_int_quarter(
        _dense_panel(), train_end_year=2001, val_end_year=2009,
        pre_quarters_included=pre_q)

    dm = _stub_datamodule(max_lookback)
    floor = dm.first_test_origin_q()
    assert str(qoff_to_period_index([floor])[0]) == "2010Q1"
    lookup, index, cache = dm._build_lookup_and_index(test_df, first_origin_q=floor)
    _, raw_index, _ = dm._build_lookup_and_index(test_df)
    raw_first = min(center_q for _, center_q in raw_index)
    if max_lookback > dm.min_lookback:
        assert raw_first < floor          # the floor actually binds here
    else:
        assert raw_first == floor

    assert min(center_q for _, center_q in index) == floor
    for firm_id, grp in test_df.groupby("firm_id"):
        assert len(lookup[firm_id]["quarter"]) == len(grp)
        s, e = cache[(firm_id, floor)]
        assert int(lookup[firm_id]["quarter"][s:e].min()) == floor - max_lookback + 1


@pytest.mark.parametrize("first_label,expected_label", [
    ("2009Q4", "2010Q3"),   # lists inside the lead-in: min_lookback binds, not the floor
    ("2012Q1", "2012Q4"),   # lists after the floor: min_lookback binds
    ("2007Q2", "2010Q1"),   # full lead-in: the floor binds
])
def test_test_origin_floor_never_overrides_min_lookback(first_label, expected_label):
    """The floor is a lower bound, combined with `max`: a firm that lists late still
    needs `min_lookback` quarters of its own history before its first origin."""
    first_q = (pd.Period(first_label) - BASE).n
    panel = _dense_panel(years=range(1995, 2016))
    test_df = panel[panel["quarter"] >= first_q]

    dm = _stub_datamodule(12)
    _, index, _ = dm._build_lookup_and_index(test_df, first_origin_q=dm.first_test_origin_q())

    first_origin = min(center_q for _, center_q in index)
    assert str(qoff_to_period_index([first_origin])[0]) == expected_label


@pytest.mark.parametrize("build_val_end,config_val_end,ok", [
    (2009, 2009, True),    # canonical: lead-in exactly max_lookback - 1
    (2009, 2011, False),   # config boundary later than the build's: floor drops real origins
])
def test_floor_check_flags_a_build_with_an_earlier_boundary(build_val_end, config_val_end,
                                                            ok, capsys):
    pre_q = _pre_q(12)
    _, _, test_df = create_time_based_splits_int_quarter(
        _dense_panel(), train_end_year=2001, val_end_year=build_val_end,
        pre_quarters_included=pre_q)
    dm = _stub_datamodule(12, val_end_year=config_val_end)
    assert dm._check_test_floor_against_file(test_df, dm.first_test_origin_q()) is ok
    assert ("dropping real test origins" in capsys.readouterr().out) is (not ok)


def test_test_origin_floor_skipped_without_time_splits():
    dm = _stub_datamodule(12)
    dm.data_config = {}
    assert dm.first_test_origin_q() is None


def _stub_datamodule(max_lookback, train_end_year=2001, val_end_year=2009,
                     min_lookback=4):
    """A FormaDataModule shell carrying only what the methods under test read.

    Constructing a real one needs a processed build on disk, so bypass __init__ and
    supply the handful of attributes `_warn_if_lead_in_too_short` and
    `_build_lookup_and_index` actually touch. `min_lookback` defaults to 4, the
    value every rN config uses.
    """
    from forma.data.forma_data import FormaDataModule
    dm = FormaDataModule.__new__(FormaDataModule)
    dm.max_lookback = max_lookback
    dm.min_lookback = min_lookback
    dm.min_horizon = 0
    dm.max_horizon = 20
    dm.scale_account_id = None
    dm.industry_mode = 'none'
    dm.dataset_tag = 'synthetic'
    dm.data_config = {"time_splits": {"train_end_year": train_end_year,
                                      "val_end_year": val_end_year}}
    return dm


def _split_built_for(max_lookback):
    _, _, test_df = create_time_based_splits_int_quarter(
        _dense_panel(), train_end_year=2001, val_end_year=2009,
        pre_quarters_included=_pre_q(max_lookback))
    return test_df


def test_runtime_guard_passes_when_dataset_matches_the_config():
    assert _stub_datamodule(12)._warn_if_lead_in_too_short(_split_built_for(12), "test")


def test_runtime_guard_fires_when_a_config_outruns_a_reused_dataset(capsys):
    """Bump max_lookback, reuse the r13 tag, never rebuild -- the real #161 failure.

    The dataset builder does not run in this path, so a build-time check cannot see it.
    """
    dm = _stub_datamodule(20)
    assert dm._warn_if_lead_in_too_short(_split_built_for(12), "test") is False
    out = capsys.readouterr().out
    assert "WARNING" in out and "max_lookback=20" in out and "#161" in out


def test_runtime_guard_checks_val_against_the_train_boundary():
    """val's lead-in sits at train_end_year, not val_end_year."""
    _, val_df, _ = create_time_based_splits_int_quarter(
        _dense_panel(), train_end_year=2001, val_end_year=2009,
        pre_quarters_included=_pre_q(12))
    assert _stub_datamodule(12)._warn_if_lead_in_too_short(val_df, "val")
    assert _stub_datamodule(20)._warn_if_lead_in_too_short(val_df, "val") is False


def test_runtime_guard_is_silent_without_time_splits(capsys):
    """No boundary configured -> nothing to check; never warn on a guess."""
    dm = _stub_datamodule(20)
    dm.data_config = {}
    assert dm._warn_if_lead_in_too_short(_split_built_for(12), "test")
    assert capsys.readouterr().out == ""


def test_runtime_guard_reports_a_dataset_mismatch_as_one(capsys):
    """A file starting after its own boundary is a wrong-build error, not a short buffer.

    `have` goes negative there. Reporting "-17 lead-in quarters" and advising a
    lower max_lookback would send the reader after the wrong thing entirely.
    """
    dm = _stub_datamodule(12, train_end_year=1994)
    _, val_df, _ = create_time_based_splits_int_quarter(
        _dense_panel(years=range(2005, 2016)), train_end_year=2008, val_end_year=2012,
        pre_quarters_included=11)
    assert dm._warn_if_lead_in_too_short(val_df, "val") is False
    out = capsys.readouterr().out
    assert "not built with these time_splits" in out
    # The normal short-buffer message must not also fire -- that is the one that
    # would report a negative count and advise a max_lookback change.
    assert "carries only" not in out


def test_runtime_guard_does_not_false_alarm_on_a_firm_split_val():
    """Firm-split val spans the whole panel, so it can never be short -- and must not warn.

    `create_firm_based_splits_int_quarter` applies the lead-in to test only; val is
    a firm partition over every quarter <= val_end_year. Its minimum is the panel
    minimum, so the val branch passes vacuously. Pinned so a future "tighten the
    guard" change does not start warning on every firm-split run.
    """
    # Span the real panel's range (1971Q1+), not the 1995Q1 default: the docstring
    # claims firm-split val can never be short, and a fixture with only 28 quarters
    # of slack would flip to a warning at max_lookback 30.
    years = range(1971, 2016)
    panel = pd.concat([_dense_panel(firm=1, years=years),
                       _dense_panel(firm=2, years=years)], ignore_index=True)
    _, val_df, test_df = create_firm_based_splits_int_quarter(
        panel, val_end_year=2009, holdout_firms={2},
        pre_quarters_included=_pre_q(12))
    assert not val_df.empty and not test_df.empty
    for max_lookback in (12, 20, 30, 120):
        assert _stub_datamodule(max_lookback)._warn_if_lead_in_too_short(val_df, "val")
    # The test branch still does the real work on a firm-split build.
    assert _stub_datamodule(12)._warn_if_lead_in_too_short(test_df, "test")
    assert _stub_datamodule(20)._warn_if_lead_in_too_short(test_df, "test") is False
