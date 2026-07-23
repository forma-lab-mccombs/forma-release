"""End-to-end CI smoke test for the Forma inference pipeline (acceptance gate #3).

Proves the shipped inference path works *shape-wise* and produces a
schema-valid ProForma-20Q forecast, WITHOUT WRDS, a GPU, large downloads, or the
shipped checkpoints. Everything is generated in ``tmp_path`` and runs on CPU in a
few seconds.

Pipeline exercised (data -> build -> infer -> validate):

  1. Synthetic raw panel  -- a tiny Compustat-shaped frame (adapted from the
     companion's ``proforma-20q/tests/fixtures.py::synthetic_raw``; inlined so the
     test is self-contained and does not import the companion's non-packaged test
     module). Carries the YTD source columns (``oancfy``/``capxy``) + a handful of
     pf_full statement items + ``gvkey``/``datadate``/``fyearq``/``fqtr``/``sich``.

  2. Build -- the companion's public ``proforma20q.build.build`` produces the
     ``tuple`` view + ``regularization_stats`` (which='tuple' only; Forma never
     needs the tabular view). ``build()`` derives but does NOT persist the
     categorical->index id maps, so we capture them by wrapping the public
     ``build_tuple`` (its documented ``(df, firm_id_map, account_id_map)`` return)
     and write ``account_id_map.csv`` / ``firm_id_map.csv`` / ``industry_id_map.csv``
     in exactly the layout ``FormaDataModule`` + ``generate_predictions`` read.

  3. Forma inference -- deliberately uses UNTRAINED weights sized to THIS synthetic
     build (num_account_types / num_industries read from the maps just written), not
     a shipped checkpoint: the synthetic build has far fewer accounts (~11 incl.
     'scale') than the shipped 82, which would mismatch a shipped checkpoint's
     embedding dims. We reuse the proven write path
     ``forma.train.generate_predictions`` over a ``FormaDataModule`` on the synth
     ``tuple_test`` split, exactly as ``scripts/predict_forma.py`` drives it. The
     point is pipeline shape + a valid forecast file, NOT numeric accuracy.

  4. Validate -- assert the forecast passes the ProForma-20Q submission schema, both
     in-process (``proforma20q.schema.validate_forecast``) and via the shipping
     acceptance path (``python -m proforma20q.cli validate`` as a subprocess).

Config is based on ``configs/r13_forma_fgrid_seed60.yaml`` (industry_mode=node,
use_future_grid, present_in_q0, inference.fp32, identities.json), with the tuple
window shrunk (max_horizon/max_lookback) and num_workers=0 for CI speed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Synthetic raw panel -- adapted from proforma-20q/tests/fixtures.py::synthetic_raw
# (inlined for self-containment). A tiny Compustat-like panel with the YTD
# cash-flow sources, a few pf_full items, and industry codes. Sized DOWN to
# ~36 firms for CI speed (the research audit used ~150 firms / ~10 min).
# ---------------------------------------------------------------------------
def _synthetic_raw(n_firms: int = 36, start: str = "1996Q1", end: str = "2013Q4",
                   seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firms = [f"{i:05d}" for i in range(n_firms)]
    quarters = pd.period_range(start, end, freq="Q")
    rows = []
    for f in firms:
        base = rng.lognormal(4, 1)
        oancf_ytd = 0.0
        for q in quarters:
            fqtr = q.quarter
            if fqtr == 1:
                oancf_ytd = 0.0
            rev = base * rng.lognormal(0, 0.15) * (1 + 0.01 * (q.year - 1996))
            cogs = rev * rng.uniform(0.5, 0.8)
            ni = rev - cogs - rev * rng.uniform(0.05, 0.2)
            oancf_ytd += ni * rng.uniform(0.7, 1.3)
            rows.append(dict(
                gvkey=f, datadate=q.to_timestamp(how="end").normalize(),
                fyearq=q.year, fqtr=fqtr,
                sich=float(rng.choice([2000, 3500, 7370, 1200])),
                naicsh=float(rng.choice([311111, 334111])),
                revtq=rev, cogsq=cogs, niq=ni,
                atq=base * rng.uniform(2, 4), ltq=base * rng.uniform(1, 2),
                seqq=base * rng.uniform(0.5, 1.5),
                oancfy=oancf_ytd, capxy=(base * 0.05 * fqtr) * rng.uniform(0.8, 1.2),
            ))
    return pd.DataFrame(rows)


def _build_synth_tuple(tmp: Path):
    """Run the companion build on the synthetic raw; return (out_dir, account_map,
    firm_map, dataset_tag) plus the written-artifact paths.

    Captures the id maps by wrapping ``proforma20q.build.build_tuple`` (which
    ``build()`` calls internally but does not persist) so the maps written to disk
    are byte-for-byte the ones the tuple ids were encoded with -- the checkpoint/
    id-map contract predict_forma.py enforces.
    """
    import proforma20q.build as pb
    from proforma20q.build import build

    raw = _synthetic_raw()
    raw_path = tmp / "compustat_with_permno.parquet"
    raw.to_parquet(raw_path, engine="fastparquet")

    captured: dict = {}
    orig_build_tuple = pb.build_tuple

    def _capturing_build_tuple(*args, **kwargs):
        out, firm_map, account_map = orig_build_tuple(*args, **kwargs)
        captured["firm_map"] = firm_map
        captured["account_map"] = account_map
        return out, firm_map, account_map

    pb.build_tuple = _capturing_build_tuple
    try:
        written = build(raw_path, out_dir=tmp, which=("tuple",), verbose=False)
    finally:
        pb.build_tuple = orig_build_tuple

    assert "account_map" in captured, "build() did not call build_tuple as expected"
    return written, captured["account_map"], captured["firm_map"]


def _write_id_maps(tmp: Path, account_map: dict, firm_map: dict,
                   tuple_test_path: Path) -> int:
    """Write the three id-map CSVs FormaDataModule / generate_predictions read.
    Returns num_industries (max industry id present + 1)."""
    from proforma20q.build import _ff48_flat

    pd.DataFrame({"account_name": list(account_map.keys()),
                  "account_id": list(account_map.values())}
                 ).to_csv(tmp / "account_id_map.csv", index=False)
    # generate_predictions maps internal int -> gvkey via columns firm_id_int/firm_id.
    pd.DataFrame({"firm_id": list(firm_map.keys()),
                  "firm_id_int": list(firm_map.values())}
                 ).to_csv(tmp / "firm_id_map.csv", index=False)

    _ranges, _unknown, id_to_name = _ff48_flat()
    max_ind = int(pd.read_parquet(tuple_test_path)["industry_id"].max())
    num_industries = max_ind + 1
    pd.DataFrame([{"industry_name": id_to_name.get(i, f"ind{i}"), "industry_id": i}
                  for i in range(num_industries)]
                 ).to_csv(tmp / "industry_id_map.csv", index=False)
    return num_industries


def test_synthetic_smoke(tmp_path):
    # Imported lazily so a plain `pytest --collect-only` doesn't pay torch import.
    import lightning as L
    from forma.train import create_data_module, generate_predictions, load_config
    from forma.models.forma import FormaModel
    from proforma20q.schema import (REQUIRED_COLS, normalize_columns,
                                    read_forecast, validate_forecast)

    L.seed_everything(60, workers=True)

    # -- 1 & 2. Synthetic raw -> companion build (tuple + reg-stats) + id maps ----
    written, account_map, firm_map = _build_synth_tuple(tmp_path)
    for key in ("tuple_train", "tuple_val", "tuple_test", "reg_stats"):
        assert Path(written[key]).exists(), f"build did not write {key}"

    num_account_types = len(account_map)          # ~11 incl. 'scale' (vs shipped 82)
    assert "scale" in account_map, "tuple build must expose a 'scale' account"
    num_industries = _write_id_maps(tmp_path, account_map, firm_map,
                                    Path(written["tuple_test"]))

    # -- 3. Config: base on the canonical fgrid seed-60 config, override for CI ----
    cfg = load_config(str(REPO_ROOT / "configs" / "r13_forma_fgrid_seed60.yaml"))
    d = cfg["data"]
    d["processed_dir"] = str(tmp_path)
    d["feature_set"] = "pf_full"
    d["dataset_tag"] = "r13_node_optionD_indfe_val8"   # matches build() default suffix
    d["identities_path"] = str(REPO_ROOT / "configs" / "identities.json")
    d["account_map_path"] = str(tmp_path / "account_id_map.csv")
    d["firm_id_map_path"] = str(tmp_path / "firm_id_map.csv")
    d["industry_id_map_path"] = str(tmp_path / "industry_id_map.csv")
    d["industry_mode"] = "node"
    d["num_workers"] = 0                                 # CPU / Windows / determinism
    d["batch_size"] = 32
    d["predict_batch_size"] = 64
    d["seed"] = 60
    # Keep the tuple window small for CI speed (test uses max_lookback / max_horizon).
    d["curriculum"]["max_horizon"] = 8
    d["curriculum"]["max_lookback"] = 8
    d["curriculum"]["min_lookback"] = 4
    d["curriculum"]["min_horizon"] = 0
    cfg["forecast_splits"] = ["test"]
    out_dir = tmp_path / "forecasts"
    cfg["results_dir"] = str(out_dir)

    # -- 3. Untrained FormaModel sized to THIS synthetic build --------------------
    params = dict(cfg["models"]["forma"]["parameters"])
    params["lr"] = float(params["lr"])
    params["dropout"] = float(params["dropout"])
    params["num_account_types"] = num_account_types
    params["num_industries"] = num_industries
    params["industry_mode"] = "node"
    model = FormaModel(**params)
    model.model_name = cfg["models"]["forma"]["forecast_name"]   # 'forma_fgrid'

    # FormaDataModule over the synth tuple; one deterministic predict pass writes
    # the forecast parquet via the proven generate_predictions path (exp_dir=None ->
    # single file straight into results_dir).
    dm = create_data_module("forma", d)
    generate_predictions(model, dm, cfg)

    # -- 4. Locate + validate the produced forecast ------------------------------
    produced = sorted(out_dir.glob(f"{model.model_name}__*__test__predictions.parquet"))
    assert produced, f"no forecast parquet produced under {out_dir}"
    forecast_path = produced[-1]

    # In-process schema validation (raises SubmissionError on any problem).
    df = read_forecast(forecast_path, validate=False)
    df = normalize_columns(df)
    problems = validate_forecast(df, strict=False)
    assert not problems, f"forecast failed submission schema: {problems}"

    # Shape / content assertions: required columns, non-empty, sane targets/horizons,
    # and the synthetic firms are represented.
    for col in REQUIRED_COLS:
        assert col in df.columns, f"forecast missing required column {col!r}"
    assert len(df) > 0, "forecast is empty"

    from proforma20q.config import pf_full_targets
    valid_targets = set(pf_full_targets())
    got_targets = set(df["target"].astype(str))
    assert got_targets, "forecast has no targets"
    assert got_targets <= valid_targets, (
        f"forecast targets not in pf_full: {sorted(got_targets - valid_targets)}")

    horizons = pd.to_numeric(df["horizon"])
    assert horizons.min() >= 1 and horizons.max() <= 20, "horizons out of [1,20]"

    n_firms_out = df["firm"].nunique()
    assert n_firms_out >= 2, f"expected multiple synthetic firms, got {n_firms_out}"
    assert df["prediction"].notna().any(), "all predictions are null"

    # -- 4b. Shipping acceptance path: `proforma20q validate` as a subprocess -----
    result = subprocess.run(
        [sys.executable, "-m", "proforma20q.cli", "validate", str(forecast_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"`proforma20q validate` failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}")
