"""The firm-map resolver must fail with a route forward, never a dead end.

`proforma20q build` computes the int->gvkey map and discards it (upstream #13),
so a reviewer following the README hits this path first. It has to say what is
missing, why, and what to do next.
"""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "predict_forma", Path(__file__).resolve().parents[1] / "scripts" / "predict_forma.py")
pf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pf)


def test_uses_the_build_copy_when_present(tmp_path):
    m = tmp_path / "firm_id_map.csv"
    m.write_text("firm_id_int,firm_id\n0,001004\n")
    assert pf._resolve_firm_map(tmp_path, None, None) == m


def test_explicit_map_wins(tmp_path):
    explicit = tmp_path / "elsewhere.csv"
    explicit.write_text("firm_id_int,firm_id\n0,001004\n")
    (tmp_path / "firm_id_map.csv").write_text("firm_id_int,firm_id\n0,999999\n")
    assert pf._resolve_firm_map(tmp_path, explicit, None) == explicit


def test_explicit_map_missing_is_reported(tmp_path):
    with pytest.raises(SystemExit) as ei:
        pf._resolve_firm_map(tmp_path, tmp_path / "nope.csv", None)
    assert "--firm-map not found" in str(ei.value)


def test_missing_map_explains_cause_and_both_remedies(tmp_path):
    with pytest.raises(SystemExit) as ei:
        pf._resolve_firm_map(tmp_path, None, None)
    msg = str(ei.value)
    assert "firm_id_map.csv" in msg
    assert "issue #13" in msg                      # why it is missing
    assert "--firm-map" in msg                     # remedy 1
    assert "--derive-firm-map-from-raw" in msg     # remedy 2


def test_derivation_refuses_a_mismatched_raw_panel(tmp_path):
    """A raw panel from a different build must not silently mint wrong gvkeys."""
    pd = pytest.importorskip("pandas")
    # Raw panel with 2 firms -> ids {0, 1}.
    raw = tmp_path / "raw.parquet"
    pd.DataFrame({"gvkey": ["001004", "001013"]}).to_parquet(raw, engine="fastparquet")
    # Tuple view referencing id 7 -> inconsistent with a 2-firm map.
    pd.DataFrame({
        "firm_id": [0, 7], "account_id": [0, 0], "quarter": [0, 0],
        "value": [0.0, 0.0], "industry_id": [0, 0],
    }).to_parquet(tmp_path / "tuple_test__pf_full__tag.parquet", engine="fastparquet")

    with pytest.raises(SystemExit) as ei:
        pf._resolve_firm_map(tmp_path, None, raw)
    assert "does not match this build" in str(ei.value)
    assert not (tmp_path / "firm_id_map.csv").exists()  # nothing written on refusal


def test_derivation_writes_a_verified_map(tmp_path):
    pd = pytest.importorskip("pandas")
    raw = tmp_path / "raw.parquet"
    pd.DataFrame({"gvkey": ["001013", "001004", "001004"]}).to_parquet(
        raw, engine="fastparquet")
    pd.DataFrame({
        "firm_id": [0, 1], "account_id": [0, 0], "quarter": [0, 0],
        "value": [0.0, 0.0], "industry_id": [0, 0],
    }).to_parquet(tmp_path / "tuple_test__pf_full__tag.parquet", engine="fastparquet")

    out = pf._resolve_firm_map(tmp_path, None, raw)
    got = pd.read_csv(out, dtype={"firm_id": str})
    # build's rule: sorted unique gvkeys, enumerated from 0.
    assert list(got["firm_id"]) == ["001004", "001013"]
    assert list(got["firm_id_int"]) == [0, 1]
