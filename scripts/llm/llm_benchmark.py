#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_benchmark.py — LLM vs Forma single-shot benchmark, aligned with pf_full.

Drives the prompt + derivations from the validated pf_full feature set
(configs/feature_sets.yaml) and the validated identities
(configs/identities.json):

  * LLM forecasts the 55 pf_full primitives using raw Compustat codes
    (revtq, cogsq, aoq, loq, req, sstkq, ...).
  * All derived items (atq, ltq, seqq, piq, niq, ivncfq, fincfq, etc.) are
    computed mechanically from identities.json, not by the LLM.
  * xsgaq includes xrdq (no double-count), piq includes spiq, niq uses
    miiq + xidoq, accounting equation includes mibtq, liability decomposition
    uses raw loq (which already contains drltq), cash CF formulas include
    sivq/sppeq/ivstchq/dlcchq/txbcofq, cash rollforward includes exreq.
  * aociq is reconstructed from its 4 disclosure sub-components (which ARE
    in the parquet) rather than sourced as a top-level column.
  * Interruption-safe: per-firm CSV append, --resume flag.

R² scoring is change-from-baseline in regularized (asinh/scale/z-score) space.

Two production prompt arms ship, selected via --prompt-version:
  * 'unstructured' — free-form steering (prompts/system_prompt_unstructured.txt)
  * 'structured'   — explicit driver-hierarchy / roll-forward projection method
                     (prompts/system_prompt_structured.txt)

Usage:
    python llm_benchmark.py --env-path .env --n-origins 500
    python llm_benchmark.py --env-path .env --n-origins 5 --dry-run
    python llm_benchmark.py --env-path .env --resume

    # Production-sample design: 1000 firms x 4 consecutive origins (enables
    # forecast-revision bias tests), via the Anthropic Batch API:
    python llm_benchmark.py --env-path .env --sample-mode firm-block \
        --n-firms 1000 --block-len 4 --batch

    # Same sample via OpenAI's native Batch API:
    python llm_benchmark.py --env-path .env --provider openai \
        --model gpt-5.5 --sample-mode firm-block --n-firms 1000 --batch

Every live run also archives the verbatim completions (+ usage + prompt
shas; + full prompts with --log-prompts) to a *__responses.jsonl sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # llm_benchmark/ — sibling modules (llm_tuple_source, forma_llm)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root — lazy config/data-relative imports in batch-eval paths
# forma_llm imported lazily inside functions that need it (load_env, call_llm)
# to keep the module importable without the anthropic SDK (e.g. for unit tests).
from llm_tuple_source import (  # noqa: E402  (after sys.path tweak)
    TupleSource, variable_block_sample_origins, select_noise_subset,
    history_labels,
)

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path("data")  # contains compustat_df.parquet + processed/
FORECASTS_DIR = Path("results/forecasts")
CONFIGS_DIR   = Path("configs")
DEFAULT_FEATURE_SET = "pf_full"  # which regularization_stats__{X}.parquet to read

def metrics_dir_for(feature_set: str) -> Path:
    """LLM benchmark outputs land under results/metrics/<feature_set>/test/
    so the path matches the feature set actually used (matches Forma's own
    convention of segregating outputs by feature set)."""
    return Path(f"results/metrics/{feature_set}/test")

HISTORY_QS      = 8                  # raw-Compustat path window (legacy)
TUPLE_HISTORY_QS = 12                 # tuple path = Forma's max_lookback (input parity)
DEFAULT_DATASET = "r9_node_optionD_indfe"  # Forma's canonical production tuple/reg-stats tag
MIN_LOOKBACK    = 4                   # Forma curriculum min_lookback (origin eligibility)
TEST_START_Q    = 161                 # 2010Q1 in Forma's integer-quarter encoding
TEST_END_Q      = 220                 # 2024Q4
ORIGIN_SEED     = 20260615            # permanent shared origins seed (model-agnostic)
MAX_HORIZON     = 20                  # match Forma's pf_full curriculum max
MIN_FUTURE_QS   = 1                   # need at least 1 future actual to score
MAX_GAP_DAYS    = 120
API_DELAY_S     = 1.0
MAX_ABS_ZSCORE  = 6.0
SCALE_CONSTANT  = 1e-3
TRAIN_CUTOFF    = pd.Timestamp("1994-12-31")
TEST_START      = pd.Timestamp("2010-01-01")
TEST_END        = pd.Timestamp("2024-12-31")
FLUSH_EVERY     = 25

DEFAULT_PROMPT_VERSION = "unstructured"
PROMPTS_DIR = Path(__file__).parent / "prompts"

# The two production prompt arms and the system-prompt file each resolves to.
PROMPT_FILE_BY_VERSION: dict[str, str] = {
    "unstructured": "system_prompt_unstructured.txt",
    "structured":   "system_prompt_structured.txt",
}

def system_prompt_path_for(version: str) -> Path:
    try:
        return PROMPTS_DIR / PROMPT_FILE_BY_VERSION[version]
    except KeyError:
        raise SystemExit(
            f"ERROR: prompt version '{version}' is not available in this "
            f"release. Choose from {sorted(PROMPT_FILE_BY_VERSION)}.")

SYSTEM_PROMPT_PATH = system_prompt_path_for(DEFAULT_PROMPT_VERSION)


def origins_csv_path(present_in_q0: bool, seed: int, n_origins: int,
                     sample_mode: str = "iid",
                     n_firms: int | None = None,
                     block_len: int | None = None,
                     source: str = "compustat",
                     min_block: int | None = None,
                     max_block: int | None = None) -> Path:
    """Origin sample is keyed by (scope, sampling design, seed, size) —
    model-agnostic so multiple model ablations share the same firm-quarter
    sample, maximizing pairwise overlap. NOT tagged by model.

    iid mode keeps the original filename (back-compat with cached samples);
    firm-block / var-block modes get their own tags so designs never collide.
    The tuple var-block design (the production sample) is tagged `tup`.
    """
    # Arm-agnostic tag: the origins sample is shared across both prompt arms
    # and all models, so it is NOT branded by prompt version — only by scope.
    scope_tag = "shared_q0" if present_in_q0 else "shared"
    src_tag = "tup__" if source == "tuples" else ""
    if sample_mode == "var_block":
        return FORECASTS_DIR / (f"origins__{src_tag}{scope_tag}__varblk"
                                f"{min_block}_{max_block}__seed{seed}__f{n_firms}.csv")
    if sample_mode == "firm_block":
        return FORECASTS_DIR / (f"origins__{src_tag}{scope_tag}__blk{block_len}"
                                f"__seed{seed}__f{n_firms}.csv")
    return FORECASTS_DIR / f"origins__{src_tag}{scope_tag}__seed{seed}__n{n_origins}.csv"

def origins_fingerprint(sample: pd.DataFrame) -> str:
    """Stable content hash of an origins set: sha256 over the sorted
    (gvkey, q0_date, max_horizon) rows. Order-independent — two runs that
    sampled the same firm-quarters fingerprint identically regardless of row
    order or dtype quirks. Used to *prove* every model in the benchmark ran the
    identical shared origins set (see --expect-origins-sha): a silently
    divergent sample (e.g. one model run with a different --n-firms) would break
    the pairwise / bias comparison, and the cache filename alone doesn't catch
    it once a param drifts."""
    cols = ["gvkey", "q0_date", "max_horizon"]
    df = sample[cols].copy()
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df["q0_date"] = pd.to_datetime(df["q0_date"]).dt.strftime("%Y-%m-%d")
    df["max_horizon"] = df["max_horizon"].astype(int).astype(str)
    rows = df.sort_values(cols).itertuples(index=False, name=None)
    return _sha256("\n".join("\t".join(r) for r in rows))

def _sanitize(s: str) -> str:
    """Filename-safe slug — keep alphanumerics, fold dots/dashes/underscores."""
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def output_paths(present_in_q0: bool, model_name: str,
                 prompt_version: str = DEFAULT_PROMPT_VERSION,
                 feature_set: str = DEFAULT_FEATURE_SET,
                 ablation_suffix: str = "") -> dict[str, Path]:
    """Distinct output filenames per (scope, prompt_version, model,
    feature_set [, ablation_suffix]) so ablations don't clobber each other.
    The `prompt_version` becomes the scope tag stem ("unstructured" /
    "structured"); the `feature_set` lands in the parquet name so the
    predictions file declares the regularization space its values live in
    (matches Forma's '{model}__{feature_set}__{split}__predictions.parquet'
    convention, letting an external scorer discover and score it against the
    matching truth file). `ablation_suffix` (e.g. "_noindustry", "_r0") gets
    appended to the scope tag for ablation / replicate runs so they don't
    overwrite the headline outputs."""
    scope_tag = f"{prompt_version}_q0" if present_in_q0 else prompt_version
    if ablation_suffix:
        scope_tag = f"{scope_tag}{ablation_suffix}"
    model_tag = _sanitize(model_name)
    tag = f"{scope_tag}__{model_tag}"
    return {
        "pred_parquet":   FORECASTS_DIR / f"llm_{tag}__{feature_set}__test__predictions.parquet",
        "pred_csv":       FORECASTS_DIR / f"llm_{tag}_predictions.csv",
        "adherence_json": FORECASTS_DIR / f"llm_{tag}_identity_adherence.json",
        "batch_sidecar":  FORECASTS_DIR / f"llm_{tag}__{feature_set}__test__batch.json",
        "batch_meta":     FORECASTS_DIR / f"llm_{tag}__{feature_set}__test__batch_meta.pkl",
        "raw_jsonl":      FORECASTS_DIR / f"llm_{tag}__{feature_set}__test__responses.jsonl",
        "metric_prefix":  f"llm_{tag}",
    }

FEATURE_SETS_YAML = CONFIGS_DIR / "feature_sets.yaml"
IDENTITIES_JSON   = CONFIGS_DIR / "identities.json"

DEFAULT_FORMA_PARQUET = Path("results/forecasts/forma__pf_full__test__predictions.parquet")

# Targets scored in R² (match configs/core.yaml)
CORE_TARGETS = ["revtq", "cogsq", "gpq", "oiadpq", "niq", "dpq",
                "cheq",  "ppentq", "atq",  "ltq",   "seqq", "dlttq"]

# ── pf_full schema: primitives LLM forecasts, derivations computed in Python ──

# Primitives (LLM forecasts these). Order is deliberate — groups follow the
# prompt's glossary. Items must never appear as LHS in identities.json.
PRIMITIVES_IS = [
    "revtq", "cogsq", "xsgaq", "xrdq", "dpq", "stkcoq",
    "xintq", "nopiq", "spiq", "txtq", "miiq", "xidoq",
]
PRIMITIVES_BS_ASSETS = [
    "cheq", "rectq", "invtq", "acoq",
    "ppentq", "dpactq",
    "gdwlq", "intanoq", "aoq",
]
PRIMITIVES_BS_LIAB = [
    "apq", "dlcq", "txpq", "drcq", "lcoq",
    "drltq", "dlttq", "txditcq", "loq",
]
PRIMITIVES_BS_EQUITY = [
    "cstkq", "capsq", "req", "acomincq", "tstkq", "pstkq", "mibtq",
]
PRIMITIVES_CF = [
    "oancfq", "fopoq",
    "capxq", "ivchq", "aqcq", "sivq", "sppeq", "ivstchq", "ivacoq",
    "sstkq", "prstkcq", "dltisq", "dltrq", "dlcchq", "dvq", "txbcofq", "fiaoq",
    "exreq",
]
PRIMITIVES = (PRIMITIVES_IS + PRIMITIVES_BS_ASSETS + PRIMITIVES_BS_LIAB
              + PRIMITIVES_BS_EQUITY + PRIMITIVES_CF)

# The two shipped prompt arms forecast the identical primitive set in the same
# order; they differ only in the SYSTEM prompt:
#   * 'unstructured' — free-form steering paragraph.
#   * 'structured'   — explicit driver-hierarchy / roll-forward projection
#                      method. Registered as its own version purely so its
#                      outputs get a distinct tag and don't clobber the
#                      unstructured run in the head-to-head.
PRIMITIVES_BY_VERSION: dict[str, list] = {
    "unstructured": list(PRIMITIVES),
    "structured":   list(PRIMITIVES),
}

# AOCI 4-component sub-features (used to reconstruct aociq if needed)
AOCI_SUBS = ["aociderglq", "aociotherq", "aocipenq", "aocisecglq"]

# YTD cash-flow columns in the parquet (renamed to `*q` after conversion).
CF_YTD_COLS = {
    "oancfy":  "oancfq",  "fopoy":   "fopoq",
    "capxy":   "capxq",   "ivchy":   "ivchq",   "aqcy":    "aqcq",
    "sivy":    "sivq",    "sppey":   "sppeq",   "ivstchy": "ivstchq",
    "ivacoy":  "ivacoq",  "ivncfy":  "ivncfq",
    "sstky":   "sstkq",   "prstkcy": "prstkcq", "dltisy":  "dltisq",
    "dltry":   "dltrq",   "dlcchy":  "dlcchq",  "dvy":     "dvq",
    "txbcofy": "txbcofq", "fiaoy":   "fiaoq",   "fincfy":  "fincfq",
    "exrey":   "exreq",
}

# Raw columns we need to SELECT from the parquet (primitives + AOCI subs +
# stored aggregates for history display + scale + quarterly derived from YTD).
RAW_NEEDED_COLS = (
    ["gvkey", "datadate", "fyearq", "fqtr", "sich"]
    + PRIMITIVES_IS + PRIMITIVES_BS_ASSETS + PRIMITIVES_BS_LIAB
    + PRIMITIVES_BS_EQUITY + AOCI_SUBS
    # BS/IS aggregates used for history display AND as "actuals" for scoring
    + ["atq", "actq", "ancq", "lctq", "ltq", "ceqq", "seqq",
       "oibdpq", "oiadpq", "piq", "ibq", "niq", "xoprq",
       "intanq", "ppegtq"]
    # YTD cash-flow cols (converted to quarterly post-load)
    + list(CF_YTD_COLS.keys())
)
RAW_NEEDED_COLS = list(dict.fromkeys(RAW_NEEDED_COLS))  # dedupe

# History display groups (for the user message)
HISTORY_GROUPS = [
    ("INCOME STATEMENT", [
        "revtq", "cogsq", "xsgaq", "xrdq", "dpq", "stkcoq",
        "gpq", "oibdpq", "oiadpq", "xintq", "nopiq", "spiq",
        "piq", "txtq", "miiq", "ibq", "xidoq", "niq", "xoprq",
    ]),
    ("BALANCE SHEET — ASSETS", [
        "cheq", "rectq", "invtq", "acoq", "actq",
        "ppentq", "dpactq", "ppegtq",
        "gdwlq", "intanoq", "intanq",
        "aoq", "ancq", "atq",
    ]),
    ("BALANCE SHEET — LIABILITIES", [
        "apq", "dlcq", "txpq", "drcq", "lcoq", "lctq",
        "drltq", "dlttq", "txditcq", "loq", "ltq",
    ]),
    ("BALANCE SHEET — EQUITY", [
        "cstkq", "capsq", "req", "acomincq", "tstkq", "ceqq",
        "pstkq", "seqq", "mibtq",
    ]),
    ("CASH FLOW (quarterly; YTD already converted)", [
        "oancfq", "fopoq",
        "capxq", "ivchq", "aqcq", "sivq", "sppeq", "ivstchq", "ivacoq", "ivncfq",
        "sstkq", "prstkcq", "dltisq", "dltrq", "dlcchq", "dvq", "txbcofq", "fiaoq", "fincfq",
        "exreq",
    ]),
]

# ── Identity-driven derivation (mirrors identities.json formulas) ─────────────
#
# Each derivation computes a single item from primitives (and previously
# computed derived items). Order matters — dependencies must precede dependents.
# Missing primitive ⇒ treated as 0.0 with a warning flag.

def _g(d: dict, k: str, missing: list) -> float:
    v = d.get(k)
    if v is None or not np.isfinite(v):
        missing.append(k)
        return 0.0
    return float(v)

def derive_all(primitives: dict) -> tuple[dict, list]:
    """Given {compustat_code: value} primitives, compute all derived items.

    Returns (full_dict, missing_primitives) where full_dict contains the union
    of primitives + all derived/computed items.
    """
    d = dict(primitives)
    missing: list = []
    def g(k): return _g(d, k, missing)

    # ── Income statement ──────────────────────────────────────────────────
    d["gpq"]          = g("revtq") - g("cogsq")
    d["xoprq"]        = g("cogsq") + g("xsgaq")
    d["xsgaq_ex_rd"]  = g("xsgaq") - g("xrdq")
    d["oibdpq"]       = d["gpq"] - g("xsgaq")             # xsgaq already incl. xrdq
    d["oiadpq"]       = d["oibdpq"] - g("dpq")
    d["piq"]          = d["oiadpq"] - g("xintq") + g("nopiq") + g("spiq")
    d["ibq"]          = d["piq"] - g("txtq") - g("miiq")
    d["niq"]          = d["ibq"] + g("xidoq")

    # ── Balance sheet ─────────────────────────────────────────────────────
    d["intanq"]        = g("gdwlq") + g("intanoq")
    d["aoq_ex_intanq"] = g("aoq") - d["intanq"]
    d["ppegtq"]        = g("ppentq") + g("dpactq")
    d["actq"]          = g("cheq") + g("rectq") + g("invtq") + g("acoq")
    d["ancq"]          = g("ppentq") + g("aoq")
    d["atq"]           = d["actq"] + d["ancq"]
    d["lctq"]          = g("apq") + g("dlcq") + g("txpq") + g("lcoq")
    d["loq_ex_dr"]     = g("loq") - g("drltq")
    d["ltq"]           = d["lctq"] + g("dlttq") + g("txditcq") + g("loq")
    d["ceqq"]          = g("cstkq") + g("capsq") + g("req") - g("tstkq")
    d["seqq"]          = d["ceqq"] + g("pstkq")
    d["wcapq"]         = d["actq"] - d["lctq"]

    # ── Cash flow ─────────────────────────────────────────────────────────
    d["ivncfq"] = (-g("capxq") - g("ivchq") - g("aqcq")
                   + g("sivq") + g("sppeq") + g("ivstchq") + g("ivacoq"))
    d["fincfq"] = (g("sstkq") - g("prstkcq") + g("dltisq") - g("dltrq")
                   + g("dlcchq") - g("dvq") + g("txbcofq") + g("fiaoq"))
    d["fcfq"]   = g("oancfq") - g("capxq")

    return d, missing


def derive_all_strict(primitives: dict) -> dict:
    """NaN-propagating mirror of derive_all.

    derive_all() substitutes 0.0 for missing primitives — that's correct for
    LLM predictions (the model provides every primitive it intends to forecast,
    and reconciliation is by-construction consistent). It is wrong for ground-
    truth actuals/baselines, where a missing input primitive should propagate
    as NaN rather than silently fabricate a zero.

    Used to fill actuals/baselines for derived-only targets (gpq, xsgaq_ex_rd,
    aoq_ex_intanq, loq_ex_dr, wcapq, ivncfq, fincfq, fcfq) and as a fallback
    when Compustat's stored aggregate is null.
    """
    def g(k):
        v = primitives.get(k)
        if v is None or not np.isfinite(v):
            return float("nan")
        return float(v)

    d = {k: float(v) for k, v in primitives.items()
         if v is not None and np.isfinite(v)}

    d["gpq"]          = g("revtq") - g("cogsq")
    d["xoprq"]        = g("cogsq") + g("xsgaq")
    d["xsgaq_ex_rd"]  = g("xsgaq") - g("xrdq")
    d["oibdpq"]       = d["gpq"] - g("xsgaq")
    d["oiadpq"]       = d["oibdpq"] - g("dpq")
    d["piq"]          = d["oiadpq"] - g("xintq") + g("nopiq") + g("spiq")
    d["ibq"]          = d["piq"] - g("txtq") - g("miiq")
    d["niq"]          = d["ibq"] + g("xidoq")

    d["intanq"]        = g("gdwlq") + g("intanoq")
    d["aoq_ex_intanq"] = g("aoq") - d["intanq"]
    d["ppegtq"]        = g("ppentq") + g("dpactq")
    d["actq"]          = g("cheq") + g("rectq") + g("invtq") + g("acoq")
    d["ancq"]          = g("ppentq") + g("aoq")
    d["atq"]           = d["actq"] + d["ancq"]
    d["lctq"]          = g("apq") + g("dlcq") + g("txpq") + g("lcoq")
    d["loq_ex_dr"]     = g("loq") - g("drltq")
    d["ltq"]           = d["lctq"] + g("dlttq") + g("txditcq") + g("loq")
    d["ceqq"]          = g("cstkq") + g("capsq") + g("req") - g("tstkq")
    d["seqq"]          = d["ceqq"] + g("pstkq")
    d["wcapq"]         = d["actq"] - d["lctq"]

    d["ivncfq"] = (-g("capxq") - g("ivchq") - g("aqcq")
                   + g("sivq") + g("sppeq") + g("ivstchq") + g("ivacoq"))
    d["fincfq"] = (g("sstkq") - g("prstkcq") + g("dltisq") - g("dltrq")
                   + g("dlcchq") - g("dvq") + g("txbcofq") + g("fiaoq"))
    d["fcfq"]   = g("oancfq") - g("capxq")

    return d


# ── Regularization ────────────────────────────────────────────────────────────

def transform(v, k):
    return np.arcsinh(np.asarray(v, dtype=np.float64) * k)

def compute_scale(row: dict) -> float | None:
    ltq, seqq, atq = row.get("ltq"), row.get("seqq"), row.get("atq")
    primary = None
    if ltq is not None and seqq is not None and np.isfinite(ltq) and np.isfinite(seqq):
        primary = abs(ltq) + abs(seqq) + SCALE_CONSTANT
    if primary is not None and primary > 0 and np.isfinite(primary):
        return primary
    if atq is not None and np.isfinite(atq):
        cand = abs(atq) + SCALE_CONSTANT
        if cand > 0 and np.isfinite(cand):
            return cand
    return None


def estimate_k(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    if len(vals) < 30:
        return 1.0
    k_grid = np.logspace(-2, 3, num=250)
    best_k, best_diff = 1.0, float("inf")
    for k in k_grid:
        t = transform(vals, k)
        n = len(t)
        if n < 4:
            continue
        m = t.mean()
        d = t - m
        m2 = (d ** 2).mean()
        m4 = (d ** 4).mean()
        if m2 <= 0:
            continue
        kurt = m4 / (m2 ** 2) - 3.0
        diff = abs(kurt - 3.0)
        if diff < best_diff:
            best_diff = diff
            best_k = k
    return best_k


def build_regularization_stats(conn, parquet: str, save_path: Path) -> pd.DataFrame:
    """Compute per-quarter (k, mu, sigma) for the 12 core targets.
    Cached to save_path — re-used verbatim across prompt arms (identical scaling)."""
    if save_path.exists():
        print(f"  Loading regularization stats from {save_path}")
        print(f"  (reusing Forma's saved stats — no rebuild needed)")
        return pd.read_parquet(save_path)

    print(f"  No cached stats at {save_path}; rebuilding from {parquet} (~3-5 min)...")
    print(f"  TIP: if Forma has already produced stats elsewhere, point "
          f"--data-dir at that location and rerun.")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    needed = list(dict.fromkeys(["gvkey", "datadate", "ltq", "seqq", "atq"]
                                + [c for c in CORE_TARGETS if c != "gpq"]
                                + ["revtq", "cogsq"]))
    col_sql = ", ".join(needed)
    print(f"    Loading {len(needed)} columns ...")
    df = conn.execute(f"""
        SELECT {col_sql}
        FROM '{parquet}'
        WHERE NOT (sich >= 6000 AND sich <= 6999)
    """).df()
    df["datadate"] = pd.to_datetime(df["datadate"])
    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    df["quarter"] = df["datadate"] + pd.offsets.QuarterEnd(0)

    ltq, seqq, atq = (df[c].to_numpy(dtype=np.float64) for c in ("ltq", "seqq", "atq"))
    primary = np.abs(ltq) + np.abs(seqq) + SCALE_CONSTANT
    valid_primary = np.isfinite(primary) & (primary > 0)
    fallback = np.abs(atq) + SCALE_CONSTANT
    valid_fallback = np.isfinite(fallback) & (fallback > 0)
    scale = np.where(valid_primary, primary, np.where(valid_fallback, fallback, np.nan))
    df["scale"] = scale
    df = df[df["scale"].notna() & np.isfinite(df["scale"])].reset_index(drop=True)
    df["gpq"] = df["revtq"] - df["cogsq"]

    train_mask = df["datadate"] <= TRAIN_CUTOFF
    all_stats: list[pd.DataFrame] = []
    for acct in CORE_TARGETS:
        print(f"    [{acct}] ", end="", flush=True)
        lead = df.groupby("gvkey")[acct].shift(-4)
        lag  = df.groupby("gvkey")[acct].shift(4)
        scaled_lead = (lead / df["scale"]).replace([np.inf, -np.inf], np.nan)
        scaled_lag  = (lag  / df["scale"]).replace([np.inf, -np.inf], np.nan)
        train_pool = pd.concat(
            [scaled_lead[train_mask].dropna(), scaled_lag[train_mask].dropna()],
            ignore_index=True,
        ).to_numpy(dtype=np.float64)
        if train_pool.size < 30:
            train_pool = pd.concat(
                [scaled_lead.dropna(), scaled_lag.dropna()], ignore_index=True
            ).to_numpy(dtype=np.float64)
        k_val = estimate_k(train_pool)
        print(f"k={k_val:.3f} ", end="", flush=True)
        t_lead = transform(scaled_lead.fillna(np.nan), k_val)
        t_lag  = transform(scaled_lag.fillna(np.nan),  k_val)
        lead_df = pd.DataFrame({"quarter": df["quarter"], "t": t_lead}).dropna()
        lag_df  = pd.DataFrame({"quarter": df["quarter"], "t": t_lag }).dropna()
        lead_agg = lead_df.groupby("quarter")["t"].agg(["mean", "std"]).rename(
            columns={"mean": "mu_lead", "std": "sigma_lead"})
        lag_agg  = lag_df.groupby("quarter")["t"].agg(["mean", "std"]).rename(
            columns={"mean": "mu_lag",  "std": "sigma_lag"})
        qstats = lead_agg.join(lag_agg, how="outer").reset_index()
        qstats["mu_raw"]    = qstats[["mu_lead", "mu_lag"]].mean(axis=1, skipna=True).fillna(0.0)
        qstats["sigma_raw"] = qstats[["sigma_lead", "sigma_lag"]].mean(axis=1, skipna=True)
        qstats["sigma_raw"] = qstats["sigma_raw"].replace(0, np.nan).fillna(1.0) + 1e-8
        qstats = qstats.sort_values("quarter")
        qstats["mu"]    = qstats["mu_raw"].rolling(window=4, min_periods=1).mean()
        qstats["sigma"] = qstats["sigma_raw"].rolling(window=4, min_periods=1).mean() + 1e-8
        out = qstats[["quarter", "mu", "sigma"]].copy()
        out["feature"] = acct
        out["k"]       = k_val
        all_stats.append(out)
        print(f"{len(out)} quarters")

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df.to_parquet(save_path)
    print(f"  Saved to {save_path}  ({len(stats_df):,} rows)")
    return stats_df


class RegStatLookup:
    """Per-(feature, quarter) (k, mu, sigma) lookup.

    Keyed by pandas quarter Period (freq='Q') rather than raw Timestamp so a
    clean quarter-end (2010-03-31 00:00:00) matches Forma's stored reg-stats
    timestamps (which carry a 23:59:59.999… quarter-end tail) exactly, instead
    of relying on the nearest-quarter fallback every call. Benefits both the
    tuple-sourced and raw-Compustat paths."""

    def __init__(self, stats_df: pd.DataFrame):
        self._k: dict[str, float] = {}
        self._mu: dict[str, dict[pd.Period, float]] = {}
        self._sigma: dict[str, dict[pd.Period, float]] = {}
        for feat, grp in stats_df.groupby("feature"):
            self._k[feat] = float(grp["k"].iloc[0])
            grp = grp.sort_values("quarter")
            periods = pd.PeriodIndex(pd.to_datetime(grp["quarter"]), freq="Q")
            self._mu[feat]    = dict(zip(periods, grp["mu"]))
            self._sigma[feat] = dict(zip(periods, grp["sigma"]))

    def get(self, feat: str, quarter) -> tuple[float, float, float] | None:
        if feat not in self._k:
            return None
        p = pd.Timestamp(quarter).to_period("Q")
        mu = self._mu[feat].get(p)
        sg = self._sigma[feat].get(p)
        if mu is None or sg is None:
            available = sorted(self._mu[feat].keys())
            if not available:
                return None
            idx = min(range(len(available)), key=lambda i: abs((available[i] - p).n))
            mu = self._mu[feat][available[idx]]
            sg = self._sigma[feat][available[idx]]
        return self._k[feat], float(mu), float(sg)


def reg_value(v: float, scale: float, k: float, mu: float, sigma: float) -> float:
    if v is None or not np.isfinite(v) or scale is None or not np.isfinite(scale) or scale <= 0:
        return float("nan")
    r = (math.asinh(k * v / scale) - mu) / (sigma + 1e-8)
    return float(np.clip(r, -MAX_ABS_ZSCORE, MAX_ABS_ZSCORE))


# ── Data loading ──────────────────────────────────────────────────────────────

def open_db(parquet: Path):
    import duckdb
    conn = duckdb.connect()
    probe = conn.execute(f"SELECT * FROM '{parquet}' LIMIT 0").df()
    avail = set(probe.columns)
    cols = [c for c in RAW_NEEDED_COLS if c in avail]
    col_sql = ", ".join(cols)
    print(f"  {len(cols)}/{len(RAW_NEEDED_COLS)} columns available")
    missing = [c for c in RAW_NEEDED_COLS if c not in avail]
    if missing:
        print(f"  Missing columns (filled as NaN downstream): {missing}")
    return conn, cols, col_sql


def load_firm_rows(conn, parquet: str, gvkey: str, col_sql: str) -> pd.DataFrame:
    df = conn.execute(
        f"""
        SELECT {col_sql}
        FROM '{parquet}'
        WHERE gvkey::VARCHAR = ?
          AND year(datadate) BETWEEN 2010 AND 2024
        ORDER BY datadate
        """,
        [str(gvkey)],
    ).df()
    df["datadate"] = pd.to_datetime(df["datadate"])
    return df


def compute_quarterly_cf(df: pd.DataFrame) -> pd.DataFrame:
    """Convert YTD CF columns (*y) to quarterly (*q). Renames columns in place."""
    if "fyearq" not in df.columns or "fqtr" not in df.columns:
        return df
    df = df.sort_values(["gvkey", "fyearq", "fqtr"]).copy()
    for ytd_col, q_col in CF_YTD_COLS.items():
        if ytd_col not in df.columns:
            continue
        prior = df.groupby(["gvkey", "fyearq"])[ytd_col].shift(1)
        # Within-year gap guard: only valid if prior row is same fyearq and fqtr - 1
        prior_fqtr = df.groupby(["gvkey", "fyearq"])["fqtr"].shift(1)
        valid_prior = (prior_fqtr == df["fqtr"] - 1)
        quarterly = df[ytd_col].where(df["fqtr"] == 1,
                                      (df[ytd_col] - prior).where(valid_prior, np.nan))
        df[q_col] = quarterly
    return df


def reconstruct_aociq(row: pd.Series) -> float | None:
    """aociq = sum of 4 disclosure subcomponents (all nullable, skip if all null)."""
    vals = [row.get(c) for c in AOCI_SUBS]
    vals = [v for v in vals if v is not None and pd.notna(v)]
    if not vals:
        return None
    return float(sum(vals))


def build_window_around_q0(firm_df: pd.DataFrame, q0_date: pd.Timestamp,
                           horizon: int) -> pd.DataFrame | None:
    """Slice an (8-history + horizon-future) window centered on q0_date.

    Caller has already verified eligibility (build_origin_pool); this is a
    pure index lookup. Returns None if q0_date isn't in firm_df or the
    surrounding window doesn't fit (defensive — shouldn't happen given a
    well-formed origin pool).
    """
    firm_df = firm_df.sort_values("datadate").reset_index(drop=True)
    matches = firm_df.index[firm_df["datadate"] == q0_date].tolist()
    if not matches:
        return None
    q0_idx = matches[0]
    start = q0_idx - HISTORY_QS + 1
    end   = q0_idx + horizon + 1
    if start < 0 or end > len(firm_df):
        return None
    return firm_df.iloc[start:end].reset_index(drop=True)


# ── History formatting & prompt ───────────────────────────────────────────────

def build_history_dict(window: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """{compustat_code: {Q-7..Q0: value or None}}. Includes aociq reconstructed."""
    labels = [f"Q-{7-i}" if i < 7 else "Q0" for i in range(HISTORY_QS)]
    # All items we want to show — primitives + stored aggregates.
    all_items = set()
    for _, codes in HISTORY_GROUPS:
        all_items.update(codes)
    all_items.add("aociq")

    history: dict[str, dict[str, float | None]] = {c: {} for c in all_items}
    for i, label in enumerate(labels):
        row = window.iloc[i]
        for c in all_items:
            if c == "aociq":
                v = reconstruct_aociq(row)
            else:
                v = row.get(c)
                v = float(v) if (v is not None and pd.notna(v)) else None
            history[c][label] = v
    return history


def _has_any_history(history: dict, code: str, quarters: list[str]) -> bool:
    """True if `code` is non-null in at least one of the supplied history quarters."""
    series = history.get(code, {})
    return any(series.get(q) is not None for q in quarters)


def active_primitives(history: dict, quarters: list[str], present_in_q0: bool,
                      prompt_version: str = DEFAULT_PROMPT_VERSION) -> list[str]:
    """Per-firm filter on which primitives the LLM should forecast.

    `prompt_version` selects which registered primitive set / ordering to
    iterate; both shipped arms use the same IS→BS→CF glossary order.

    Mirrors Forma's pf_full inference behavior:
      * Default (Forma `present_in_q0: false`): primitives present anywhere
        in the 8-quarter history window — matches the dropna(subset=["value"])
        tuple-level filter used in dataset creation.
      * --present-in-q0 (Forma `present_in_q0: true`): primitives non-null
        at Q0 specifically.
    """
    order = PRIMITIVES_BY_VERSION[prompt_version]
    if present_in_q0:
        return [c for c in order if history.get(c, {}).get("Q0") is not None]
    return [c for c in order if _has_any_history(history, c, quarters)]


# ── Fama-French 48 industry labels ───────────────────────────────────────────
# The tuples Forma trains on carry a per-firm `industry_id` (FF48 bucket, 0-47;
# 48 = Unknown) assigned from the firm's modal SIC code
# (see ff48.py:sic_to_ff48). Forma consumes it via an industry
# node embedding (industry_mode: 'node' in the canonical configs); the tabular
# baselines via FF48 one-hot dummies. To match that information in the LLM
# prompt we surface the firm's industry by name. industry_id_map.csv is the
# authority for id↔short-name; FF48_LONG_NAMES expands the terse Ken-French
# abbreviations into plain English so the model can apply a sensible prior.
FF48_LONG_NAMES = {
    "Agric": "Agriculture",
    "Food": "Food Products",
    "Soda": "Candy & Soda",
    "Beer": "Beer & Liquor",
    "Smoke": "Tobacco Products",
    "Toys": "Recreation",
    "Fun": "Entertainment",
    "Books": "Printing & Publishing",
    "Hshld": "Consumer Goods",
    "Clths": "Apparel",
    "Hlth": "Healthcare",
    "MedEq": "Medical Equipment",
    "Drugs": "Pharmaceutical Products",
    "Chems": "Chemicals",
    "Rubbr": "Rubber & Plastic Products",
    "Txtls": "Textiles",
    "BldMt": "Construction Materials",
    "Cnstr": "Construction",
    "Steel": "Steel Works",
    "FabPr": "Fabricated Products",
    "Mach": "Machinery",
    "ElcEq": "Electrical Equipment",
    "Autos": "Automobiles & Trucks",
    "Aero": "Aircraft",
    "Ships": "Shipbuilding & Railroad Equipment",
    "Guns": "Defense",
    "Gold": "Precious Metals",
    "Mines": "Non-Metallic & Industrial Metal Mining",
    "Coal": "Coal",
    "Oil": "Petroleum & Natural Gas",
    "Util": "Utilities",
    "Telcm": "Communication",
    "PerSv": "Personal Services",
    "BusSv": "Business Services",
    "Comps": "Computers",
    "Chips": "Electronic Equipment",
    "LabEq": "Measuring & Control Equipment",
    "Paper": "Business Supplies",
    "Boxes": "Shipping Containers",
    "Trans": "Transportation",
    "Whlsl": "Wholesale",
    "Rtail": "Retail",
    "Meals": "Restaurants, Hotels & Motels",
    "Banks": "Banking",
    "Insur": "Insurance",
    "RlEst": "Real Estate",
    "Fin": "Trading",
    "Other": "Other / Conglomerate",
    "Unknown": "Unknown (unclassified)",
}


def load_industry_labels(map_path: Path) -> dict[int, str]:
    """industry_id → plain-English FF48 label, sourced from industry_id_map.csv
    (the same file Forma reads). Falls back to the raw short name if a code is
    missing from FF48_LONG_NAMES."""
    df = pd.read_csv(map_path)
    out: dict[int, str] = {}
    for short, iid in zip(df["industry_name"], df["industry_id"]):
        out[int(iid)] = FF48_LONG_NAMES.get(str(short), str(short))
    return out


def format_history(history: dict, quarters: list[str]) -> str:
    """CSV-like sections by statement, using raw Compustat codes.

    Suppresses rows where the item is null in ALL history quarters (matches
    Forma's tuple-level dropna). aociq is included in the equity section
    (reconstructed from 4 subcomponents)."""
    lines = []
    for title, items in HISTORY_GROUPS:
        rows = []
        for item in items:
            if item not in history or not _has_any_history(history, item, quarters):
                continue
            vals = ",".join(
                f"{history[item][q]:.2f}" if history[item].get(q) is not None else ""
                for q in quarters
            )
            rows.append(f"{item},{vals}")
        if (title.startswith("BALANCE SHEET — EQUITY")
                and "aociq" in history
                and _has_any_history(history, "aociq", quarters)):
            vals = ",".join(
                f"{history['aociq'][q]:.2f}" if history['aociq'].get(q) is not None else ""
                for q in quarters
            )
            rows.append(f"aociq,{vals}")
        if not rows:
            continue
        lines.append(f"=== {title} ===")
        lines.append("Account," + ",".join(quarters))
        lines.extend(rows)
        lines.append("")
    return "\n".join(lines)


def load_system_prompt(path: Path = SYSTEM_PROMPT_PATH, *,
                       lookback_qs: int = TUPLE_HISTORY_QS,
                       max_horizon: int = MAX_HORIZON) -> str:
    """Load the system prompt, substituting the lookback/horizon placeholders
    from the run's actual config so the prompt header can never drift from the
    data window we feed (the old hard-coded "8 quarters" bug). The prompt file
    carries literal `{lookback_qs}` / `{lookback_first}` / `{max_horizon}`
    tokens; we str.replace() them (not .format(), so stray braces are safe)."""
    txt = path.read_text(encoding="utf-8").rstrip()
    return (txt
            .replace("{lookback_qs}", str(lookback_qs))
            .replace("{lookback_first}", str(lookback_qs - 1))
            .replace("{max_horizon}", str(max_horizon)))


# ── LLM call & response parsing ───────────────────────────────────────────────

import re
_QUARTER_LINE = re.compile(r'^Q(\d+):\s*(.+)$')
# Bare numeric fragment between commas — the tail of a thousands-separated
# number ("revtq=1,234.5" splits into "revtq=1" + "234.5"; "revtq=1,234e5"
# into "revtq=1" + "234e5"). Used to detect and drop the corrupted preceding
# value instead of keeping the wrong leading digits.
_NUM_FRAGMENT = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')


DEFAULT_MAX_OUTPUT_TOKENS = 64000  # gpt-5.x reasoning (esp. on the structured
                                   # prompt) can exhaust a smaller cap on
                                   # hard origins — at 32K, 2/20 gpt-5.4 pilot
                                   # origins spent the WHOLE budget on reasoning
                                   # and returned no answer ("no text content").
                                   # 64K gives headroom for reasoning + the ~9K
                                   # visible answer. Anthropic finishes ~10-16K
                                   # (Sonnet budget-capped, Opus self-limits) so
                                   # they never touch the extra — billed only if
                                   # used. Sonnet max=64K, Opus max=128K, gpt ample.

# Reasoning level — kept at an equivalent "medium" tier across vendors so the
# Anthropic and OpenAI models reason comparably (not byte-identical compute, but
# each vendor's "medium"). Passed as raw request-body fields (batch) / extra_body
# (sync SDK) so they don't depend on the installed SDK typing the kwargs.
#
# Anthropic thinking is model-aware (see _anthropic_thinking_params). The DEFAULT
# is adaptive — the forward-safe choice, since new frontier models are adaptive-
# ONLY and 400 on budget_tokens; only an enumerated set of older models uses an
# explicit budget_tokens cap. Through Portkey's batch passthrough, output_config
# .effort is NOT forwarded (confirmed empirically), so adaptive models run at the
# default (high) effort — but they stay bounded here (Opus ~4-6K, Sonnet 5 ~12-17K
# thinking; no runaway, no truncation under the 64K cap). The budget_tokens cap is
# for (a) Sonnet 4.6, whose adaptive thinking runs away on this prompt, and
# (b) pre-4.6 models that don't support adaptive at all.
MEDIUM_THINKING_BUDGET  = 6000   # Sonnet thinking-token cap ~ "medium" (uses ~2K)
OPENAI_REASONING_EFFORT = "medium"

# OpenAI wire protocol for the openai provider. "chat" = /v1/chat/completions
# (default; returns NO reasoning summary for gpt-5.x — the archived `thinking`
# field is null). "responses" = /v1/responses with reasoning.summary="detailed",
# which surfaces a reasoning SUMMARY (summary_text) we can capture into
# `thinking`, at the same billed-token cost. Set via --openai-api and stashed on
# os.environ so the module-level batch/sync helpers can read it without threading
# a flag through every call site (mirrors PORTKEY_OPENAI_PROVIDER).
OPENAI_REASONING_SUMMARY = "detailed"


def _openai_api_mode() -> str:
    m = os.environ.get("OPENAI_API_MODE", "chat")
    return m if m in ("chat", "responses") else "chat"


def _anthropic_thinking_params(model: str) -> dict:
    """Request-body fragment for medium Anthropic thinking, model-aware.

    DEFAULT is adaptive — the forward-safe path. Every adaptive-capable model
    (Opus 4.6/4.7/4.8, Sonnet 5, Fable/Mythos, and any newer frontier model)
    takes it, and the newest are adaptive-ONLY (they 400 on budget_tokens), so an
    unknown/new/re-slugged model must default to adaptive rather than silently
    hitting the budget_tokens branch and 400-ing every request. Only the
    explicitly-enumerated older models are routed to a fixed budget_tokens cap:
      * Sonnet 4.6 — supports adaptive but its adaptive thinking RUNS AWAY on this
        prompt (consumes the whole max_tokens budget, blanking the answer).
      * Adaptive-incapable frontier models (Haiku 4.5, Sonnet 4.5, Opus 4.5/4.1) —
        do not support adaptive at all and require budget_tokens.
    Adaptive thinking is supported only on Opus 4.6/4.7/4.8, Sonnet 4.6, Sonnet 5,
    and Fable/Mythos (and presumably newer); those (minus Sonnet 4.6) take the
    default adaptive path.

    NOTE: older adaptive-incapable models (Claude 3.x, Sonnet/Opus 4.0, pre-4.5
    Haiku) are intentionally NOT enumerated — their ids use a different, reversed
    convention ("claude-3-5-haiku") that this new-format substring set wouldn't
    match cleanly, and they're outside this benchmark's model universe. If one is
    ever run it will 400 LOUDLY on the adaptive default ("use enabled/budget_tokens")
    rather than corrupt a run silently — add its id substring here if needed.
    """
    m = (model or "").lower()
    # Substrings below match both the bare alias (claude-sonnet-4-5) and the dated
    # id (claude-sonnet-4-5-20250929), and the Portkey-slugged forms of each.
    BUDGET_TOKENS_MODELS = (
        "sonnet-4-6",                                       # adaptive runs away -> bound it
        "haiku-4-5", "sonnet-4-5", "opus-4-5", "opus-4-1",  # adaptive-incapable frontier
    )
    if any(t in m for t in BUDGET_TOKENS_MODELS):
        # display defaults to "summarized" on these (Sonnet 4.6 / pre-4.6), but
        # set it explicitly so the reasoning summary is always archived.
        return {"thinking": {"type": "enabled",
                             "budget_tokens": MEDIUM_THINKING_BUDGET,
                             "display": "summarized"}}
    # Adaptive default (adaptive-only frontier models + anything newer).
    # display: "summarized" is REQUIRED here — on Sonnet 5 / Opus 4.8/4.7 (and
    # Fable/Mythos 5) thinking.display defaults to "omitted", which returns
    # thinking blocks with an EMPTY thinking field (signature only), so the
    # reasoning-capture archive would be blank. "summarized" returns the readable
    # summary at no extra cost (full thinking tokens are billed either way). See
    # https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking.
    # output_config.effort is a no-op via Portkey today (not forwarded), so these
    # run at default (high) adaptive, which stays bounded (no runaway/truncation).
    return {"thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "medium"}}


_ANTHROPIC_CLIENT = None
_ANTHROPIC_CLIENT_LOCK = None  # lazy threading.Lock; only allocated if needed

def _get_anthropic_client():
    """Lazy singleton of anthropic.Anthropic. Honors auto-derived ANTHROPIC_*
    env vars (set in main() when PORTKEY_API_KEY is present), so a Portkey
    user just sets PORTKEY_API_KEY + PORTKEY_MODEL and the SDK transparently
    routes through Portkey's gateway."""
    global _ANTHROPIC_CLIENT, _ANTHROPIC_CLIENT_LOCK
    if _ANTHROPIC_CLIENT is not None:
        return _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT_LOCK is None:
        import threading
        _ANTHROPIC_CLIENT_LOCK = threading.Lock()
    with _ANTHROPIC_CLIENT_LOCK:
        if _ANTHROPIC_CLIENT is not None:
            return _ANTHROPIC_CLIENT
        import anthropic
        extra_headers: dict[str, str] = {}
        for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines():
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                extra_headers[k.strip()] = v.strip()
        _ANTHROPIC_CLIENT = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN", "dummy"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
            default_headers=extra_headers if extra_headers else None,
            # 180s per-request timeout (the SDK's default is 10 min, which is
            # too long for a hung Portkey response — concurrent workers would
            # block as_completed forever). Retry up to 3 times on 5xx /
            # connection errors with exponential backoff (SDK default = 2).
            timeout=180.0,
            max_retries=3,
        )
    return _ANTHROPIC_CLIENT


def _build_system_block(system_prompt: str) -> list:
    """Wrap the system prompt in a single TextBlockParam with ephemeral
    cache_control. Lets Anthropic cache the ~3K-token system prompt across
    the run (5-min ephemeral TTL refreshed by every call), saving ~90% on
    that portion of input tokens after the first hit."""
    return [{"type": "text", "text": system_prompt,
             "cache_control": {"type": "ephemeral"}}]


def call_llm_v8(system: str, user: str,
                max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> tuple[str, dict]:
    """One Claude call via the Anthropic SDK (Portkey transparently when
    PORTKEY_API_KEY is set in env). Uses prompt caching on the system block.

    Returns (text, usage_dict) where usage_dict has:
        input_tokens, output_tokens,
        cache_creation_input_tokens, cache_read_input_tokens
    """
    # Bare model name (slug stripped): provider routing rides in the
    # x-portkey-provider header (set in main), matching the batch path. Sending
    # the @slug/model form here 401s on Portkey's /v1/messages endpoint.
    model = _portkey_batch_model()
    client = _get_anthropic_client()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_build_system_block(system),
        messages=[{"role": "user", "content": user}],
        # Medium thinking (bounded, model-aware). extra_body so it works
        # regardless of the installed SDK version typing these fields.
        extra_body=_anthropic_thinking_params(model),
    )
    # With thinking enabled the first block(s) are thinking blocks; pull the
    # text block rather than assuming content[0] is the answer.
    text = next((b.text for b in resp.content
                 if getattr(b, "type", None) == "text"), None)
    if text is None:
        raise RuntimeError("no text block in response (only thinking?)")
    u = resp.usage
    usage = {
        "input_tokens":                getattr(u, "input_tokens", 0) or 0,
        "output_tokens":               getattr(u, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens":     getattr(u, "cache_read_input_tokens", 0) or 0,
    }
    return text, usage


# ── Raw-response archive ──────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class RawResponseLog:
    """Append-only JSONL archive of every raw LLM completion.

    The prediction CSV/parquet only keep the *parsed* numbers; the completion
    text itself was previously discarded, which made a run impossible to
    re-analyze — re-parse with a better parser, audit tail blowups, or measure
    textual bias (anchoring/overreaction language). One JSON object per line:

        {ts, custom_id, gvkey, q0, model, prompt_version, system_prompt_sha,
         user_msg_sha, user_msg?, completion, thinking, usage, error}

    `user_msg` is included only when log_prompts=True (it is deterministically
    regenerable from data + code + flags; the sha pins which prompt was sent).
    `completion` is the verbatim model output; `error` is set instead of
    `completion` for API/batch failures so failure modes are archived too.
    `thinking` is the model's reasoning summary (Anthropic thinking blocks) when
    extended/adaptive thinking is on — billed for and otherwise unrecoverable,
    so it is archived here; None when the provider returns no reasoning.
    """

    def __init__(self, path: Path, model_name: str, prompt_version: str,
                 system_prompt: str, resume: bool, log_prompts: bool = False):
        self.path = path
        self.model = model_name
        self.prompt_version = prompt_version
        self.system_prompt_sha = _sha256(system_prompt)
        self.log_prompts = log_prompts
        append = resume and path.exists()
        self._fh = open(path, "a" if append else "w",
                        encoding="utf-8", buffering=1)
        if append:
            print(f"  Raw-response log (append): {path.name}")
        else:
            print(f"  Raw-response log: {path.name}")

    def log(self, gvkey: int, q0_iso: str, completion: str | None,
            usage: dict | None, error: str | None = None,
            user_msg: str | None = None, user_msg_sha: str | None = None,
            custom_id: str | None = None, thinking: str | None = None) -> None:
        if user_msg_sha is None and user_msg is not None:
            user_msg_sha = _sha256(user_msg)
        rec = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "custom_id": custom_id or f"{int(gvkey):06d}_{q0_iso.replace('-', '')}",
            "gvkey": f"{int(gvkey):06d}",
            "q0": q0_iso,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "system_prompt_sha": self.system_prompt_sha,
            "user_msg_sha": user_msg_sha,
            "completion": completion,
            "thinking": thinking,
            "usage": usage,
            "error": error,
        }
        if self.log_prompts and user_msg is not None:
            rec["user_msg"] = user_msg
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._fh.close()


# ── Portkey Batch API helpers ─────────────────────────────────────────────────
#
# Portkey exposes Anthropic batch processing via its own /v1/batches endpoint
# (NOT the Anthropic-native /v1/messages/batches, which the Anthropic SDK's
# client.messages.batches.* targets). Per
# https://portkey.ai/docs/integrations/llms/anthropic/batches the call shape
# is:
#   POST   /v1/batches              create
#   GET    /v1/batches/{id}         retrieve (status + request_counts)
#   GET    /v1/batches/{id}/output  stream JSONL results (after status=ended)
#   POST   /v1/batches/{id}/cancel  cancel
# All four require headers
#   x-portkey-api-key: <PORTKEY_API_KEY>
#   x-portkey-provider: @<integration-slug>
# where <integration-slug> is your Model Catalog slug (e.g. @your-portkey-slug).
# The slug is parsed out of PORTKEY_MODEL ("@<slug>/<model>" → slug + bare).
#
# Body envelope is Anthropic-native: {"requests": [{"custom_id", "params": {
# "model", "max_tokens", "system", "messages"}}]}. Result JSONL lines are
# also Anthropic-native: {"custom_id", "result": {"type", "message": {
# "content", "usage"}}}. So the request-building and result-parsing logic
# is identical to a direct-to-Anthropic flow — only the transport layer and
# polling shape are Portkey-specific.
#
# We use raw httpx (already a transitive dep of the anthropic SDK) rather
# than the portkey_ai SDK to avoid a hard new dependency and to keep the
# call shape transparent.

PORTKEY_BATCHES_BASE = "https://api.portkey.ai/v1/batches"
BATCH_HTTP_TIMEOUT_S = 120.0
# HTTP statuses we retry instead of aborting the batch (after potentially
# hours of server-side processing). 429 is rate-limit; 5xx is Portkey or
# upstream Anthropic transient error.
_BATCH_RETRY_STATUSES = (429, 500, 502, 503, 504)


def _http_get_with_retry(url: str, headers: dict, retry_secs: int,
                         label: str) -> httpx.Response:
    """GET with retry on transient failures (network errors + retriable
    HTTP statuses). Hard 4xx (other than 429) raises RuntimeError. Loops
    until either a 2xx is received or a non-retriable error fires.

    Used by both poll_batch and the output fetch to avoid losing a long-
    running batch to a transient 503 from Portkey or upstream."""
    while True:
        try:
            r = httpx.get(url, headers=headers, timeout=BATCH_HTTP_TIMEOUT_S)
        except httpx.HTTPError as e:
            print(f"    {label} transient error: {type(e).__name__}: {e}; "
                  f"sleeping {retry_secs}s")
            time.sleep(retry_secs)
            continue
        if r.status_code in _BATCH_RETRY_STATUSES:
            print(f"    [{label} HTTP {r.status_code}; sleeping {retry_secs}s]")
            time.sleep(retry_secs)
            continue
        if r.status_code != 200:
            raise RuntimeError(
                f"{label} failed: HTTP {r.status_code} — {r.text[:500]}"
            )
        return r


def _portkey_batch_provider_value() -> str:
    """Derive the x-portkey-provider header value from PORTKEY_MODEL.

    PORTKEY_MODEL is typically '@<slug>/<model>' (Model Catalog
    routing); strip the @ + slash and re-prefix with @. Falls back to
    '@anthropic' if no slug is present (works on Portkey workspaces with a
    plain Anthropic integration registered)."""
    m = os.environ.get("PORTKEY_MODEL", "")
    if m.startswith("@") and "/" in m:
        slug = m.split("/", 1)[0].lstrip("@")
        return f"@{slug}"
    return "@anthropic"


def _portkey_batch_model() -> str:
    """Bare Anthropic model name (slug stripped) for use in the request body."""
    m = os.environ.get("PORTKEY_MODEL",
                       os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6"))
    if m.startswith("@") and "/" in m:
        return m.split("/", 1)[1]
    return m


def _portkey_batch_headers() -> dict:
    """Auth + provider headers for /v1/batches requests."""
    key = os.environ.get("PORTKEY_API_KEY", "")
    if not key:
        raise RuntimeError("PORTKEY_API_KEY not set; --batch requires Portkey")
    return {
        "x-portkey-api-key":  key,
        "x-portkey-provider": _portkey_batch_provider_value(),
        "Content-Type":       "application/json",
    }


def build_batch_request(custom_id: str, system_prompt: str, user_msg: str,
                        model: str,
                        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> dict:
    """Build one batch request envelope (a plain dict). Mirrors the sync
    call_llm_v8 shape: same model, same max_tokens, same cached system
    block, same single user-turn message."""
    return {
        "custom_id": custom_id,
        "params": {
            "model":         model,
            "max_tokens":    max_tokens,
            "system":        _build_system_block(system_prompt),
            "messages":      [{"role": "user", "content": user_msg}],
            **_anthropic_thinking_params(model),  # medium thinking (bounded)
        },
    }


def submit_batch(batch_requests: list, sidecar_path: Path,
                 meta_path: Path, meta_by_custom_id: dict,
                 model: str) -> str:
    """Phase B: POST /v1/batches with the given requests. Persists
    meta_by_custom_id (pickle) and a JSON sidecar with the batch_id +
    submission metadata so a future --resume can pick up from Phase C."""
    n = len(batch_requests)
    if n == 0:
        raise ValueError("submit_batch called with zero requests")
    print(f"  Submitting batch of {n:,} requests to POST {PORTKEY_BATCHES_BASE} ...")
    r = httpx.post(PORTKEY_BATCHES_BASE,
                   headers=_portkey_batch_headers(),
                   json={"requests": batch_requests},
                   timeout=BATCH_HTTP_TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(
            f"batches.create failed: HTTP {r.status_code} — {r.text[:500]}"
        )
    batch_id = r.json().get("id")
    if not batch_id:
        raise RuntimeError(f"batches.create missing id in response: {r.text[:500]}")

    with open(meta_path, "wb") as f:
        pickle.dump(meta_by_custom_id, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(sidecar_path, "w") as f:
        json.dump({
            "batch_id":     batch_id,
            "provider":     "anthropic",
            "submitted_at": pd.Timestamp.utcnow().isoformat(),
            "n_requests":   n,
            "model":        model,
            "meta_path":    str(meta_path),
        }, f, indent=2)

    print(f"  Batch submitted: id={batch_id}  n={n:,}")
    print(f"    sidecar -> {sidecar_path.name}")
    print(f"    meta    -> {meta_path.name}")
    return batch_id


def poll_batch(batch_id: str, poll_secs: int) -> dict:
    """Phase C: GET /v1/batches/{id} until terminal. Portkey returns
        { "status": "in_progress" | "completed" | "failed" | ..., (varies)
          "request_counts": {"total": N, "completed": K, "failed": E} }
    where status is OpenAI-batch-shaped. We detect terminal via the simpler
    invariant `completed + failed >= total` once total is known, since
    Portkey's `status` field is sometimes None mid-flight while counts
    correctly tick. Returns the last retrieve response dict.

    Retries on network errors and HTTP 429/5xx (via _http_get_with_retry)
    so a transient gateway hiccup doesn't abort a batch that's been
    processing for half an hour."""
    print(f"  Polling batch {batch_id} every {poll_secs}s ...")
    last_counts_str = None
    headers = {k: v for k, v in _portkey_batch_headers().items()
               if k != "Content-Type"}
    url = f"{PORTKEY_BATCHES_BASE}/{batch_id}"
    while True:
        r = _http_get_with_retry(url, headers, poll_secs, "retrieve")
        j = r.json()
        status = j.get("status")
        rc = j.get("request_counts") or {}
        total     = rc.get("total", 0)
        completed = rc.get("completed", 0)
        failed    = rc.get("failed", 0)
        counts_str = f"total={total} completed={completed} failed={failed}"
        terminal = (status in ("completed", "ended", "failed", "expired", "cancelled")
                    or (total > 0 and completed + failed >= total))
        if counts_str != last_counts_str or terminal:
            ts = time.strftime("%H:%M:%S")
            print(f"    [{ts}] status={status!r}  {counts_str}")
            last_counts_str = counts_str
        if terminal:
            return j
        time.sleep(poll_secs)


def _parse_batch_result_line(obj: dict) -> tuple[str | None, str | None, dict | None, str | None, str | None]:
    """Parse one JSONL line from /v1/batches/{id}/output. Returns
        (custom_id, text_or_None, usage_or_None, error_msg_or_None, thinking_or_None).

    Result line shape (Anthropic-native; Portkey passes through):
        {"custom_id": "...", "result": {
            "type": "succeeded" | "errored" | "expired" | "canceled",
            "message": {"content": [{"text": "..."}], "usage": {...}},  # if succeeded
            "error":   {"type": "...", "message": "..."},                # if not
        }}
    With extended/adaptive thinking enabled, `content` leads with one or more
    thinking blocks ({"type": "thinking", "thinking": "..."}). We return the
    concatenated human-readable reasoning as the 5th element so the raw-response
    archive can retain it: those tokens are billed and the summary is otherwise
    unrecoverable after the run. `redacted_thinking` blocks are encrypted and
    not captured (nothing readable to keep). Factored out of iter_batch_results
    so the parser can be unit-tested without an HTTP round-trip."""
    custom_id = obj.get("custom_id")
    result = obj.get("result") or {}
    rtype = result.get("type")
    if rtype == "succeeded":
        msg = result.get("message") or {}
        content = msg.get("content") or []
        # With thinking enabled, content may lead with thinking block(s); take
        # the text block rather than content[0].
        text = next((b.get("text") for b in content
                     if b.get("type") == "text"), None)
        thinking_parts = [b.get("thinking") for b in content
                          if b.get("type") == "thinking" and b.get("thinking")]
        thinking = "\n".join(thinking_parts) if thinking_parts else None
        u = msg.get("usage") or {}
        usage = {
            "input_tokens":                u.get("input_tokens", 0) or 0,
            "output_tokens":               u.get("output_tokens", 0) or 0,
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens":     u.get("cache_read_input_tokens", 0) or 0,
        }
        if text is None:
            return custom_id, None, usage, "succeeded but no text content", thinking
        return custom_id, text, usage, None, thinking
    err = result.get("error") or {}
    err_msg = (f"{rtype}: {err.get('message', err)}"
               if err else f"{rtype or 'unknown'}")
    return custom_id, None, None, err_msg, None


def iter_batch_results(batch_id: str, retry_secs: int = 30):
    """Phase D: stream GET /v1/batches/{id}/output line-by-line via
    httpx.stream(), parse each JSONL line, yield (custom_id, text_or_None,
    usage_or_None, error_msg_or_None, thinking_or_None).

    Streaming (vs r.text.splitlines()) keeps memory flat regardless of n —
    a 200-origin batch is ~3MB but a 5000-origin Opus batch could be
    ~80MB. The fetch is wrapped in transient-error retry just like poll_batch.
    The output endpoint isn't resumable, so a mid-body error re-opens the
    stream from the top — `yielded` dedupes by custom_id across restarts so
    already-yielded results are never emitted twice (a restart 80% through a
    2,000-request fetch would otherwise double-write every earlier origin)."""
    headers = {k: v for k, v in _portkey_batch_headers().items()
               if k != "Content-Type"}
    url = f"{PORTKEY_BATCHES_BASE}/{batch_id}/output"
    yielded: set[str] = set()
    while True:
        try:
            with httpx.stream("GET", url, headers=headers,
                              timeout=BATCH_HTTP_TIMEOUT_S) as r:
                if r.status_code in _BATCH_RETRY_STATUSES:
                    r.read()  # surface body for the log
                    print(f"    [output HTTP {r.status_code}; sleeping {retry_secs}s]")
                    time.sleep(retry_secs)
                    continue
                if r.status_code != 200:
                    raise RuntimeError(
                        f"batches.output failed: HTTP {r.status_code} — {r.text[:500]}"
                    )
                for line in r.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  WARN: non-JSON output line skipped: {e}: {line[:200]}")
                        continue
                    parsed = _parse_batch_result_line(obj)
                    cid = parsed[0]
                    if cid is not None:
                        if cid in yielded:
                            continue  # re-streamed after a mid-body retry
                        yielded.add(cid)
                    yield parsed
                return
        except httpx.HTTPError as e:
            print(f"    output transient error: {type(e).__name__}: {e}; "
                  f"sleeping {retry_secs}s (already-fetched results are "
                  f"deduped on restart)")
            time.sleep(retry_secs)
            continue


# ── OpenAI driver (sync + native Batch API) ──────────────────────────────────
#
# The second provider allowed by the study design. Unlike the Anthropic path
# (SDK via Portkey), this talks to api.openai.com directly with raw httpx:
#   sync:   POST /v1/chat/completions
#   batch:  POST /v1/files (purpose=batch, JSONL upload)
#           POST /v1/batches {input_file_id, endpoint, completion_window:24h}
#           GET  /v1/batches/{id}            poll (status + request_counts)
#           GET  /v1/files/{output_file_id}/content   result JSONL
# Batch is 50% off all tokens, same discount shape as Anthropic's.
#
# Request bodies use max_completion_tokens (GPT-5-era reasoning models reject
# the legacy max_tokens) and set no temperature (rejected by reasoning
# models; Anthropic-side calls don't set one either, so the designs match).
# Usage is normalized to the Anthropic-shaped dict the rest of the script
# expects; OpenAI's prompt_tokens INCLUDES cached tokens, so cached tokens
# are subtracted out of input_tokens to keep the end-of-run token report's
# "total input = input + cache_read + cache_write" identity intact.

OPENAI_BASE_DIRECT  = "https://api.openai.com/v1"
PORTKEY_OPENAI_BASE = "https://api.portkey.ai/v1"
OPENAI_SYNC_TIMEOUT_S = 600.0  # reasoning models can think for minutes


def _openai_via_portkey() -> bool:
    """Route the OpenAI provider through Portkey's unified gateway (sync +
    Batches API) instead of api.openai.com directly. True when no direct
    OPENAI_API_KEY is set but a PORTKEY_API_KEY is — the team's standard
    setup, where OpenAI credentials live in the Portkey Model Catalog and are
    referenced by a provider slug (x-portkey-provider), so no per-request
    OpenAI key is needed. Set OPENAI_FORCE_DIRECT=1 to opt back into the
    direct api.openai.com path even when a Portkey key is present."""
    if os.environ.get("OPENAI_FORCE_DIRECT") == "1":
        return False
    return (not os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("PORTKEY_API_KEY"))


def _openai_base() -> str:
    return PORTKEY_OPENAI_BASE if _openai_via_portkey() else OPENAI_BASE_DIRECT


def _openai_headers() -> dict:
    """Auth headers for the OpenAI-compatible REST endpoints (chat/completions,
    files, batches). Two modes:
      * via Portkey: x-portkey-api-key + x-portkey-provider:<slug>. Portkey's
        unified API mirrors OpenAI's file+batch shape, so the same request
        bodies route to OpenAI through the gateway using the Portkey key only.
      * direct: Authorization: Bearer OPENAI_API_KEY (api.openai.com)."""
    if _openai_via_portkey():
        pk = os.environ.get("PORTKEY_API_KEY", "")
        provider = os.environ.get("PORTKEY_OPENAI_PROVIDER", "@openai")
        return {"x-portkey-api-key": pk, "x-portkey-provider": provider}
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set; required for --provider openai (or set "
            "PORTKEY_API_KEY + PORTKEY_OPENAI_PROVIDER / "
            "--openai-portkey-provider to route through Portkey)")
    return {"Authorization": f"Bearer {key}"}


def _normalize_openai_usage(u: dict) -> dict:
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0
    prompt = u.get("prompt_tokens", 0) or 0
    return {
        "input_tokens":                max(prompt - cached, 0),
        "output_tokens":               u.get("completion_tokens", 0) or 0,
        "cache_creation_input_tokens": 0,  # OpenAI caching is automatic; no write premium
        "cache_read_input_tokens":     cached,
    }


def _normalize_openai_usage_responses(u: dict) -> dict:
    """Same normalization for the Responses API usage block, which names its
    fields input_tokens / output_tokens (vs chat's prompt_tokens /
    completion_tokens) and nests cached under input_tokens_details. As in chat,
    output_tokens ALREADY INCLUDES reasoning_tokens, so cost accounting is
    directly comparable to /v1/chat/completions."""
    cached = ((u.get("input_tokens_details") or {}).get("cached_tokens", 0)) or 0
    inp = u.get("input_tokens", 0) or 0
    return {
        "input_tokens":                max(inp - cached, 0),
        "output_tokens":               u.get("output_tokens", 0) or 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens":     cached,
    }


def call_llm_openai(system: str, user: str, model: str,
                    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> tuple[str, dict]:
    """One OpenAI chat completion. Mirrors call_llm_v8's (text, usage) contract.
    Retries transient statuses with backoff (httpx has no built-in retry).
    In --openai-api responses mode, talks to /v1/responses instead and returns
    only the forecast text (the reasoning summary is captured on the batch path;
    sync callers only consume text)."""
    responses_mode = _openai_api_mode() == "responses"
    if responses_mode:
        path = "/responses"
        body = {
            "model": model,
            "input": [{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            "reasoning": {"effort": OPENAI_REASONING_EFFORT,
                          "summary": OPENAI_REASONING_SUMMARY},
            "max_output_tokens": max_tokens,
        }
    else:
        path = "/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
            "reasoning_effort": OPENAI_REASONING_EFFORT,  # medium (GPT-5 default; explicit)
        }
    delay = 5.0
    for attempt in range(4):
        try:
            r = httpx.post(f"{_openai_base()}{path}",
                           headers=_openai_headers(), json=body,
                           timeout=OPENAI_SYNC_TIMEOUT_S)
        except httpx.HTTPError as e:
            if attempt == 3:
                raise
            print(f"    openai transient error: {type(e).__name__}; retry in {delay:.0f}s")
            time.sleep(delay); delay *= 2
            continue
        if r.status_code in _BATCH_RETRY_STATUSES and attempt < 3:
            print(f"    openai HTTP {r.status_code}; retry in {delay:.0f}s")
            time.sleep(delay); delay *= 2
            continue
        if r.status_code != 200:
            raise RuntimeError(f"openai {path} failed: HTTP {r.status_code} "
                               f"— {r.text[:500]}")
        j = r.json()
        if responses_mode:
            text, usage, _err, _think = _extract_openai_responses_body(j)
            return (text or ""), usage
        choice = (j.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        return text, _normalize_openai_usage(j.get("usage") or {})
    raise RuntimeError("unreachable")


def build_openai_batch_line(custom_id: str, system_prompt: str, user_msg: str,
                            model: str,
                            max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> dict:
    """One line of the OpenAI batch input JSONL. Mirrors call_llm_openai.
    Two wire protocols selected by --openai-api (see _openai_api_mode):
      * chat (default): /v1/chat/completions — no reasoning summary returned.
      * responses: /v1/responses with reasoning.summary — captures the summary."""
    if _openai_api_mode() == "responses":
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "input": [{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_msg}],
                "reasoning": {"effort": OPENAI_REASONING_EFFORT,
                              "summary": OPENAI_REASONING_SUMMARY},
                "max_output_tokens": max_tokens,
            },
        }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_msg}],
            "max_completion_tokens": max_tokens,
            "reasoning_effort": OPENAI_REASONING_EFFORT,  # medium (GPT-5 default; explicit)
        },
    }


def submit_openai_batch(batch_requests: list, sidecar_path: Path,
                        meta_path: Path, meta_by_custom_id: dict,
                        model: str) -> str:
    """Phase B (OpenAI): upload the JSONL to /v1/files, create the batch,
    persist sidecar + meta exactly like the Portkey path so --resume works."""
    n = len(batch_requests)
    if n == 0:
        raise ValueError("submit_openai_batch called with zero requests")
    payload = "\n".join(json.dumps(req, ensure_ascii=False)
                        for req in batch_requests) + "\n"
    print(f"  Uploading batch input file ({n:,} requests, "
          f"{len(payload)/1e6:.1f} MB) to {_openai_base()}/files ...")
    r = httpx.post(f"{_openai_base()}/files", headers=_openai_headers(),
                   data={"purpose": "batch"},
                   files={"file": ("batch_input.jsonl",
                                   payload.encode("utf-8"),
                                   "application/jsonl")},
                   timeout=BATCH_HTTP_TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(f"files.create failed: HTTP {r.status_code} — {r.text[:500]}")
    input_file_id = r.json().get("id")
    if not input_file_id:
        raise RuntimeError(f"files.create missing id: {r.text[:500]}")

    endpoint = ("/v1/responses" if _openai_api_mode() == "responses"
                else "/v1/chat/completions")
    print(f"  Creating batch (input_file_id={input_file_id}, endpoint={endpoint}) ...")
    r = httpx.post(f"{_openai_base()}/batches", headers=_openai_headers(),
                   json={"input_file_id": input_file_id,
                         "endpoint": endpoint,
                         "completion_window": "24h"},
                   timeout=BATCH_HTTP_TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(f"batches.create failed: HTTP {r.status_code} — {r.text[:500]}")
    batch_id = r.json().get("id")
    if not batch_id:
        raise RuntimeError(f"batches.create missing id: {r.text[:500]}")

    with open(meta_path, "wb") as f:
        pickle.dump(meta_by_custom_id, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(sidecar_path, "w") as f:
        json.dump({
            "batch_id":        batch_id,
            "provider":        "openai",
            # The wire mode this batch was submitted with. Persisted so a later
            # --resume parses the output with the SAME mode: a Responses batch
            # parsed under chat (or vice-versa) reads the wrong body shape and
            # drops every forecast to a soft "no text content" error.
            "openai_api_mode": _openai_api_mode(),
            "input_file_id":   input_file_id,
            "submitted_at":    pd.Timestamp.utcnow().isoformat(),
            "n_requests":      n,
            "model":           model,
            "meta_path":       str(meta_path),
        }, f, indent=2)
    print(f"  Batch submitted: id={batch_id}  n={n:,}")
    print(f"    sidecar -> {sidecar_path.name}")
    print(f"    meta    -> {meta_path.name}")
    return batch_id


def poll_openai_batch(batch_id: str, poll_secs: int) -> dict:
    """Phase C (OpenAI): GET /v1/batches/{id} until a terminal status.
    Returns the final retrieve response (carries output_file_id /
    error_file_id, which Phase D needs)."""
    print(f"  Polling batch {batch_id} every {poll_secs}s ...")
    url = f"{_openai_base()}/batches/{batch_id}"
    last_counts_str = None
    terminal_statuses = ("completed", "failed", "expired", "cancelled")
    while True:
        r = _http_get_with_retry(url, _openai_headers(), poll_secs, "retrieve")
        j = r.json()
        status = j.get("status")
        rc = j.get("request_counts") or {}
        counts_str = (f"total={rc.get('total', 0)} "
                      f"completed={rc.get('completed', 0)} "
                      f"failed={rc.get('failed', 0)}")
        if counts_str != last_counts_str or status in terminal_statuses:
            ts = time.strftime("%H:%M:%S")
            print(f"    [{ts}] status={status!r}  {counts_str}")
            last_counts_str = counts_str
        if status in terminal_statuses:
            return j
        time.sleep(poll_secs)


def _parse_openai_batch_result_line(obj: dict) -> tuple[str | None, str | None, dict | None, str | None, str | None]:
    """Parse one OpenAI batch output/error JSONL line into the same
    (custom_id, text, usage, error, thinking) tuple shape as the Portkey
    parser. OpenAI's /v1/chat/completions does NOT return a reasoning summary,
    so `thinking` is only whatever a gateway happens to surface on the message
    (`reasoning_content` / `reasoning`), else None — capturing a summary from
    OpenAI reasoning models would require the Responses API + reasoning.summary."""
    custom_id = obj.get("custom_id")
    err = obj.get("error")
    if err:
        return custom_id, None, None, f"error: {err.get('message', err)}", None
    resp = obj.get("response") or {}
    if resp.get("status_code") != 200:
        body = resp.get("body") or {}
        msg = (body.get("error") or {}).get("message", "")
        return custom_id, None, None, f"HTTP {resp.get('status_code')}: {msg}", None
    body = resp.get("body") or {}
    if _openai_api_mode() == "responses":
        return (custom_id,) + _extract_openai_responses_body(body)
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content")
    thinking = message.get("reasoning_content") or message.get("reasoning") or None
    usage = _normalize_openai_usage(body.get("usage") or {})
    if not text:
        return custom_id, None, usage, "succeeded but no text content", thinking
    return custom_id, text, usage, None, thinking


def _extract_openai_responses_body(body: dict) -> tuple[str | None, dict | None, str | None, str | None]:
    """Pull (text, usage, error, thinking) out of a /v1/responses response body.
    The `output` array interleaves items: type=="reasoning" carries a
    `summary` list of {type:"summary_text", text} blocks (the captured
    reasoning summary -> `thinking`); type=="message" carries a `content` list
    of {type:"output_text", text} blocks (the forecast text). Both are
    concatenated across all matching items to be robust to multi-block output.
    A response can also surface status=="incomplete" (e.g. max_output_tokens
    hit before the message) — treated as an error so it's counted, not silently
    dropped."""
    out_items = body.get("output") or []
    text_parts, think_parts = [], []
    for item in out_items:
        itype = item.get("type")
        if itype == "message":
            for blk in item.get("content") or []:
                if blk.get("type") == "output_text" and blk.get("text"):
                    text_parts.append(blk["text"])
        elif itype == "reasoning":
            for blk in item.get("summary") or []:
                if blk.get("type") == "summary_text" and blk.get("text"):
                    think_parts.append(blk["text"])
    text = "".join(text_parts) or None
    thinking = "\n".join(think_parts) or None
    usage = _normalize_openai_usage_responses(body.get("usage") or {})
    status = body.get("status")
    if not text:
        detail = (body.get("incomplete_details") or {}).get("reason") or status or "no output_text"
        return None, usage, f"succeeded but no text content ({detail})", thinking
    return text, usage, None, thinking


def iter_openai_batch_results(final_status: dict, retry_secs: int = 30):
    """Phase D (OpenAI): stream the output file (succeeded requests) then the
    error file (failed requests), yielding the same 5-tuples as
    iter_batch_results. Both files are plain JSONL behind
    GET /v1/files/{id}/content. A mid-body error re-opens the stream from the
    top; `yielded` dedupes by custom_id across restarts (see
    iter_batch_results)."""
    yielded: set[str] = set()
    for key in ("output_file_id", "error_file_id"):
        file_id = final_status.get(key)
        if not file_id:
            continue
        url = f"{_openai_base()}/files/{file_id}/content"
        while True:
            try:
                with httpx.stream("GET", url, headers=_openai_headers(),
                                  timeout=BATCH_HTTP_TIMEOUT_S) as r:
                    if r.status_code in _BATCH_RETRY_STATUSES:
                        r.read()
                        print(f"    [{key} HTTP {r.status_code}; sleeping {retry_secs}s]")
                        time.sleep(retry_secs)
                        continue
                    if r.status_code != 200:
                        raise RuntimeError(f"files.content({key}) failed: "
                                           f"HTTP {r.status_code} — {r.text[:500]}")
                    for line in r.iter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"  WARN: non-JSON output line skipped: {e}: {line[:200]}")
                            continue
                        parsed = _parse_openai_batch_result_line(obj)
                        cid = parsed[0]
                        if cid is not None:
                            if cid in yielded:
                                continue  # re-streamed after a mid-body retry
                            yielded.add(cid)
                        yield parsed
                    break
            except httpx.HTTPError as e:
                print(f"    {key} transient error: {type(e).__name__}: {e}; "
                      f"sleeping {retry_secs}s (already-fetched results are "
                      f"deduped on restart)")
                time.sleep(retry_secs)
                continue


def parse_response(text: str) -> dict[str, dict[str, float]]:
    """Parse 'Q1: revtq=X, cogsq=Y, ...' → {Qn: {code: val}}."""
    out = {}
    for line in text.strip().splitlines():
        m = _QUARTER_LINE.match(line.strip())
        if not m:
            continue
        # Normalize the label: a zero-padded "Q01:" must land on the same key
        # finish_origin looks up ("Q1") — verbatim storage would miss every
        # lookup while still passing the parse-coverage gate.
        qn = f"Q{int(m.group(1))}"
        pairs = {}
        last_key = None
        for kv in m.group(2).split(","):
            kv = kv.strip()
            if "=" not in kv:
                # A bare numeric fragment is the tail of a thousands-separated
                # value split on "," — the previous key's stored value is the
                # truncated leading digits (silently WRONG, e.g. 1234.5 → 1.0).
                # Drop it: a masked gap is recoverable, a wrong value is not.
                if last_key is not None and _NUM_FRAGMENT.fullmatch(kv):
                    pairs.pop(last_key, None)
                last_key = None
                continue
            k, v = kv.split("=", 1)
            try:
                pairs[k.strip()] = float(v.strip())
                last_key = k.strip()
            except ValueError:
                last_key = None
        out[qn] = pairs
    return out


def forecast_firm(history: dict, hist_quarters: list[str], system_prompt: str,
                  active_list: list[str], horizon: int,
                  dry_run: bool = False) -> dict | None:
    """Returns {Qn: {code: value}} with primitives + derived items, or None.

    `active_list` is the per-firm primitive whitelist (Forma `present_in_q0`
    scope). `horizon` is the number of future quarters this origin needs
    (1 ≤ horizon ≤ 20). Primitives outside `active_list` are treated as
    structurally absent (0) when deriving aggregates.
    """
    n_active = len(active_list)
    # Full horizon set, no availability hint — see prepare_origin note.
    user_msg = (
        f"Forecast horizons: Q1 through Q{MAX_HORIZON}.\n\n"
        f"This firm reports {n_active} primitive line items. Forecast ONLY "
        f"these items — do not output any other primitives:\n\n"
        f"{', '.join(active_list)}\n\n"
        f"Items not in this list are structurally absent for this firm "
        f"(consistently null in history); they will be treated as 0.\n\n"
        f"Historical data:\n\n"
        + format_history(history, hist_quarters)
    )

    if dry_run:
        # Naïve: repeat Q0 values, but only for active primitives.
        q0_prims = {c: history[c].get("Q0") for c in active_list
                    if history.get(c, {}).get("Q0") is not None}
        raw = {f"Q{i}": dict(q0_prims) for i in range(1, horizon + 1)}
    else:
        try:
            llm_text, _ = call_llm_v8(system_prompt, user_msg)
        except Exception as e:
            print(f"    API error: {e}")
            return None
        raw = parse_response(llm_text)
        # Tolerate partial output: as long as we got at least horizon // 2 + 1
        # horizons we proceed; missing horizons are simply not scored.
        min_required = max(1, horizon // 2)
        if not raw or len(raw) < min_required:
            print(f"    Parse error: only {len(raw)} horizons (needed ≥{min_required})")
            return None

    active_set = set(active_list)
    out = {}
    for h in range(1, horizon + 1):
        q_label = f"Q{h}"
        prims = raw.get(q_label, {})
        # Strip hallucinations on structurally-absent items.
        prims = {k: v for k, v in prims.items() if k in active_set}
        full, _ = derive_all(prims)
        out[q_label] = full
        out[f"{q_label}_active_returned"] = sorted(set(prims) & active_set)
        out[f"{q_label}_active_omitted"]  = sorted(active_set - set(prims))
    return out


# ── R² accumulator (copied from v6) ───────────────────────────────────────────

class R2Accumulator:
    def __init__(self):
        self._stats: dict[tuple, dict] = {}

    def add(self, key: tuple, pred: float, actual: float, baseline: float):
        if not (np.isfinite(pred) and np.isfinite(actual) and np.isfinite(baseline)):
            return
        ac = actual - baseline
        res = pred - actual
        e = self._stats.setdefault(key, {"sum_y": 0.0, "sum_y2": 0.0, "sum_res2": 0.0, "n": 0})
        e["sum_y"]    += ac
        e["sum_y2"]   += ac * ac
        e["sum_res2"] += res * res
        e["n"]        += 1


def aggregate_r2(acc: R2Accumulator, group_keys: list[str]) -> pd.DataFrame:
    rows = []
    col_idx = {"target": 0, "forecast_horizon": 1}
    pool: dict[tuple, dict] = {}
    for key, e in acc._stats.items():
        gv = tuple(key[col_idx[g]] for g in group_keys)
        p = pool.setdefault(gv, {"sum_y": 0.0, "sum_y2": 0.0, "sum_res2": 0.0, "n": 0})
        for k in ("sum_y", "sum_y2", "sum_res2", "n"):
            p[k] += e[k]
    for gv, e in pool.items():
        n = e["n"]
        if n == 0:
            r2 = float("nan"); rmse = float("nan")
        else:
            ss_total = e["sum_y2"] - (e["sum_y"] ** 2) / n
            r2 = 1.0 - e["sum_res2"] / ss_total if ss_total > 0 else float("nan")
            rmse = math.sqrt(e["sum_res2"] / n)
        row = {"r2": r2, "rmse": rmse, "n": n}
        for k, v in zip(group_keys, gv):
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ── Stratified sample (copied from v6) ────────────────────────────────────────

def build_origin_pool(conn, parquet: str) -> pd.DataFrame:
    """Enumerate every eligible (gvkey, q0_date, max_horizon) origin.

    Eligibility mirrors Forma's pf_full inference behaviour as closely as
    possible without reusing Forma's pipeline:
      * Non-financial sector (SIC ∉ 6000-6999).
      * `revtq` and `atq` non-null at q0.
      * 8 prior consecutive quarters (every gap ≤ 120 days within the firm).
      * ≥ 1 future consecutive quarter (so we have at least one actual to
        score against, matches Forma `min_horizon: 0` at inference).
      * q0 ∈ [2010-01-01, 2024-12-31] (Forma test window).
      * `max_horizon` = min(20, future_consecutive_quarters_available).

    Returns DataFrame: gvkey, q0_date, max_horizon (one row per eligible origin).
    """
    df = conn.execute(f"""
        SELECT gvkey::VARCHAR AS gvkey, datadate,
               COALESCE(sich, 0) AS sich, atq, revtq
        FROM '{parquet}'
        WHERE NOT (sich BETWEEN 6000 AND 6999)
          AND atq   IS NOT NULL
          AND revtq IS NOT NULL
        ORDER BY gvkey, datadate
    """).df()
    df["datadate"] = pd.to_datetime(df["datadate"])

    # Per-firm "run id": increments at each >120-day gap or new firm.
    diff_days = df.groupby("gvkey")["datadate"].diff().dt.days
    gap_break = (diff_days > MAX_GAP_DAYS).fillna(False)
    df["run_id"] = gap_break.groupby(df["gvkey"]).cumsum().astype(int)

    # Position within run + run length.
    df["run_idx"]  = df.groupby(["gvkey", "run_id"]).cumcount()
    df["run_size"] = df.groupby(["gvkey", "run_id"])["datadate"].transform("count")

    # Eligibility:
    #   run_idx ≥ HISTORY_QS - 1     (8q history available behind q0)
    #   run_size - run_idx > MIN_FUTURE_QS  (≥1 future quarter ahead)
    #   q0 in test window
    history_ok = df["run_idx"] >= (HISTORY_QS - 1)
    future_ok  = (df["run_size"] - df["run_idx"] - 1) >= MIN_FUTURE_QS
    date_ok    = (df["datadate"] >= TEST_START) & (df["datadate"] <= TEST_END)
    eligible_mask = history_ok & future_ok & date_ok

    elig = df.loc[eligible_mask, ["gvkey", "datadate", "run_idx", "run_size"]].copy()
    future_avail = elig["run_size"] - elig["run_idx"] - 1
    elig["max_horizon"] = np.minimum(MAX_HORIZON, future_avail).astype(int)
    elig = elig.rename(columns={"datadate": "q0_date"})
    return elig[["gvkey", "q0_date", "max_horizon"]].reset_index(drop=True)


def random_sample_origins(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Uniform random sample of n origins from the eligible pool.

    Deterministic given (pool contents, n, seed). The pool is sorted by
    (gvkey, q0_date) before sampling so row-order from upstream sources
    can't perturb which rows are selected.
    """
    pool = pool.sort_values(["gvkey", "q0_date"]).reset_index(drop=True)
    if len(pool) <= n:
        return pool.copy()
    return pool.sample(n=n, random_state=seed).reset_index(drop=True)


def firm_block_sample_origins(pool: pd.DataFrame, n_firms: int,
                              block_len: int, seed: int) -> pd.DataFrame:
    """Sample n_firms firms, then a block of `block_len` CONSECUTIVE eligible
    origin quarters per firm.

    Rationale: expectations-bias tests (Coibion-Gorodnichenko, Bordalo et al.)
    regress forecast errors on forecast REVISIONS, which need the same
    (firm, target-quarter, variable) forecast from two adjacent origins.
    An i.i.d. firm-quarter sample (random_sample_origins) almost never yields
    adjacent origins for the same firm; this design guarantees block_len - 1
    revisions per firm for every target quarter the horizons overlap on.

    Mechanics: eligible origins within one coverage run are contiguous
    calendar quarters (the eligibility filters in build_origin_pool each
    preserve contiguity within a run), so consecutive-origin segments are
    recovered by splitting each firm's eligible q0s at gaps > MAX_GAP_DAYS.
    A block start is drawn uniformly over ALL valid start positions for the
    firm (across its segments), so longer coverage runs are proportionally
    more likely — the same weighting an i.i.d. draw over starts would give.

    Deterministic given (pool contents, n_firms, block_len, seed): the pool
    is sorted, firms are drawn with one seeded RandomState, and per-firm
    starts are drawn in sorted-firm order from the same RandomState.
    """
    pool = pool.sort_values(["gvkey", "q0_date"]).reset_index(drop=True)

    # Segment id: new segment at each firm change or gap > MAX_GAP_DAYS
    # between consecutive eligible q0s.
    gap_days = pool.groupby("gvkey")["q0_date"].diff().dt.days
    seg_break = gap_days.isna() | (gap_days > MAX_GAP_DAYS)
    pool = pool.assign(seg_id=seg_break.cumsum())
    seg_len = pool.groupby("seg_id")["q0_date"].transform("count")
    pool = pool.assign(seg_pos=pool.groupby("seg_id").cumcount(),
                       seg_len=seg_len)

    # Valid block starts: positions with >= block_len rows left in the segment.
    starts = pool[pool["seg_pos"] + block_len <= pool["seg_len"]]
    eligible_firms = np.sort(starts["gvkey"].unique())
    if len(eligible_firms) == 0:
        raise ValueError(f"No firm has {block_len} consecutive eligible "
                         f"origins; lower --block-len.")

    rng = np.random.RandomState(seed)
    if len(eligible_firms) > n_firms:
        chosen = rng.choice(eligible_firms, size=n_firms, replace=False)
    else:
        print(f"  WARN: only {len(eligible_firms):,} firms have a "
              f"{block_len}-quarter block (asked for {n_firms:,}); using all.")
        chosen = eligible_firms
    chosen = np.sort(chosen)  # draw per-firm starts in deterministic order

    blocks = []
    starts_by_firm = {gv: grp for gv, grp in starts.groupby("gvkey")}
    for gv in chosen:
        firm_starts = starts_by_firm[gv]
        start_row = firm_starts.iloc[rng.randint(len(firm_starts))]
        i0 = start_row.name  # positional == label here (reset_index above)
        blocks.append(pool.iloc[i0:i0 + block_len])

    out = pd.concat(blocks, ignore_index=True)
    return out[["gvkey", "q0_date", "max_horizon"]]


# ── Per-firm processing ───────────────────────────────────────────────────────

def _history_from_byq(byq: dict, center_q: int, max_lookback: int
                      ) -> tuple[dict, list[str]]:
    """Build {code: {Q-11..Q0: value}} + the relative-label list from a
    tuple-sourced {quarter_int: {code: value}} dict (no aociq reconstruction —
    aociq isn't in pf_full; acomincq is the equity primitive)."""
    hist_quarters = history_labels(max_lookback)
    history: dict[str, dict[str, float | None]] = {}
    for off in range(-(max_lookback - 1), 1):
        label = "Q0" if off == 0 else f"Q{off}"
        for code, val in byq.get(center_q + off, {}).items():
            history.setdefault(code, {})[label] = val
    return history, hist_quarters


def prepare_origin(conn, parquet: str, gvkey, q0_date: pd.Timestamp,
                   max_horizon: int, col_sql: str,
                   present_in_q0: bool = False,
                   prompt_version: str = DEFAULT_PROMPT_VERSION,
                   tsrc: "TupleSource | None" = None,
                   firm_industry: "dict[int, str] | None" = None,
                   ) -> tuple[dict | None, str | None]:
    """Phase 1 of origin processing — pure data shaping, no LLM call.

    Returns (meta, user_msg) or (None, None) if the origin can't be processed
    (no window, bad scale, no active primitives). Splitting prepare/finish
    lets us run LLM calls in a thread pool (or future batch API) without
    parallelising DB / pandas work.

    Two sourcing modes share the same downstream shaping:
      * `tsrc is None` (raw Compustat): history/window re-derived from
        compustat_df.parquet (YTD→quarterly CF, aociq reconstruction, an
        8-quarter window).
      * `tsrc` set (tuples): history/window/scale read from Forma's stored
        tuples (12-quarter window = max_lookback, scale = the tuple's scale
        account, no re-derivation). `gvkey` is the integer firm_id and
        `q0_date` the center quarter-end (mapped back to the integer quarter).

    `meta` carries everything `finish_origin` needs to score the response,
    including the `prompt_version` used for finish-side dispatch.
    """
    if tsrc is not None:
        gvkey_i = int(gvkey)  # gvkey (the tuple-pool firm_id column is the gvkey)
        center_q = tsrc.qmap.ts_to_q(q0_date)
        window, scale_q0, byq = tsrc.build_window(gvkey_i, center_q, max_horizon)
        if scale_q0 is None or not np.isfinite(scale_q0) or scale_q0 <= 0:
            return None, None
        history, hist_quarters = _history_from_byq(byq, center_q,
                                                   tsrc.max_lookback)
        q0_idx = tsrc.max_lookback - 1
        q0_row = window.iloc[q0_idx]
        q0_qend = tsrc.qmap.to_qend_ts(center_q)
    else:
        firm_df = load_firm_rows(conn, parquet, gvkey, col_sql)
        firm_df = compute_quarterly_cf(firm_df)
        window = build_window_around_q0(firm_df, q0_date, max_horizon)
        del firm_df
        if window is None:
            return None, None

        hist_quarters = [f"Q-{7-i}" if i < 7 else "Q0" for i in range(HISTORY_QS)]
        history = build_history_dict(window.iloc[:HISTORY_QS].reset_index(drop=True))

        q0_idx = HISTORY_QS - 1
        q0_row = window.iloc[q0_idx]
        q0_qend = pd.Timestamp(q0_row["datadate"]) + pd.offsets.QuarterEnd(0)

        row_dict = {c: (q0_row[c] if pd.notna(q0_row.get(c)) else None)
                    for c in ("ltq", "seqq", "atq")}
        scale_q0 = compute_scale(row_dict)
        if scale_q0 is None:
            return None, None

    active_list = active_primitives(history, hist_quarters, present_in_q0,
                                    prompt_version=prompt_version)
    if not active_list:
        return None, None

    # Firm industry (FF48) — input parity with Forma's industry node and the
    # tabular baselines' FF48 dummies. None when --industry off or the firm is
    # unmapped (no line emitted in that case).
    industry_label = (firm_industry or {}).get(int(gvkey))

    # Always request the FULL horizon set (MAX_HORIZON), never the per-origin
    # count of in-file future quarters. Revealing "this origin has N future
    # quarters" leaks that the firm delists / data ends before MAX_HORIZON —
    # a look-ahead signal. Forma forecasts every horizon and masks the
    # missing targets at scoring time; finish_origin already scores only the
    # horizons that have an actual, so the model stays blind to availability.
    user_msg = (
        f"Forecast horizons: Q1 through Q{MAX_HORIZON}.\n\n"
        + (f"Firm industry (Fama-French 48): {industry_label}.\n\n"
           if industry_label else "")
        + f"This firm reports {len(active_list)} primitive line items. Forecast "
        f"ONLY these items — do not output any other primitives:\n\n"
        f"{', '.join(active_list)}\n\n"
        f"Items not in this list are structurally absent for this firm "
        f"(consistently null in history); they will be treated as 0.\n\n"
        f"Historical data:\n\n"
        + format_history(history, hist_quarters)
    )

    meta = {
        "gvkey":          int(gvkey),
        "q0_qend":        q0_qend,
        "max_horizon":    max_horizon,
        "scale_q0":       scale_q0,
        "active_list":    active_list,
        "history":        history,         # used by dry-run fallback
        "window":         window,          # used by finish for actuals
        "q0_idx":         q0_idx,
        "q0_row":         q0_row,
        "prompt_version": prompt_version,  # finish-side dispatch
    }
    return meta, user_msg


def finish_origin(meta: dict, llm_text: str | None, reg: RegStatLookup,
                  ) -> tuple[list[dict], dict | None]:
    """Phase 2 of origin processing — parse LLM response, derive aggregates,
    score against actuals, build prediction rows. Pure CPU/pandas work."""
    active_list = meta["active_list"]
    max_horizon = meta["max_horizon"]
    active_set = set(active_list)

    if llm_text is None:
        # Dry-run / API failure: synthesize a naive Q0-repeat forecast.
        history = meta["history"]
        q0_prims = {c: history[c].get("Q0") for c in active_list
                    if history.get(c, {}).get("Q0") is not None}
        raw = {f"Q{i}": dict(q0_prims) for i in range(1, max_horizon + 1)}
    else:
        raw = parse_response(llm_text)
        min_required = max(1, max_horizon // 2)
        if not raw or len(raw) < min_required:
            return [], None

    forecasts: dict = {}
    for h in range(1, max_horizon + 1):
        q_label = f"Q{h}"
        prims = {k: v for k, v in raw.get(q_label, {}).items() if k in active_set}
        if not prims:
            # Horizon absent from the response (max_tokens truncation, label
            # drift, refusal tail). Do NOT derive: derive_all({}) fabricates
            # all-zero aggregates (niq=atq=ltq=...=0.0) that would be scored
            # as real predictions and silently poison R². Leave the horizon
            # unscored — the masked-gap treatment Forma applies to missing
            # targets. (Partially-emitted horizons keep the documented
            # missing-primitive⇒0 catch-all: the prompt tells the model
            # structurally-absent items are treated as 0.)
            continue
        full, _ = derive_all(prims)
        forecasts[q_label] = full
        forecasts[f"{q_label}_active_returned"] = sorted(set(prims) & active_set)
        forecasts[f"{q_label}_active_omitted"]  = sorted(active_set - set(prims))

    window   = meta["window"]
    q0_idx   = meta["q0_idx"]
    q0_qend  = meta["q0_qend"]
    q0_row   = meta["q0_row"]
    scale_q0 = meta["scale_q0"]
    gvkey    = meta["gvkey"]

    def primitives_in_row(row) -> dict:
        return {p: float(row[p]) for p in PRIMITIVES
                if p in row.index and pd.notna(row[p])}

    q0_full_strict = derive_all_strict(primitives_in_row(q0_row))

    rows = []
    for h in range(1, max_horizon + 1):
        future_idx = q0_idx + h
        if future_idx >= len(window):
            break
        future_row = window.iloc[future_idx]
        future_date = pd.Timestamp(future_row["datadate"]) + pd.offsets.QuarterEnd(0)
        q_fc = forecasts.get(f"Q{h}", {})
        future_full_strict = derive_all_strict(primitives_in_row(future_row))

        # Score every target the LLM emitted (primitives + derived items
        # reconstructed via derive_all). For each, prefer Compustat's stored
        # aggregate when available; else fall back to strict re-derivation
        # from the same row's primitives (NaN-propagating, no fabrication).
        for target in q_fc.keys():
            pred_raw = q_fc.get(target)
            if pred_raw is None or not np.isfinite(pred_raw):
                continue

            stored_actual = future_row.get(target) if target in future_row.index else None
            if stored_actual is not None and pd.notna(stored_actual):
                actual_raw = float(stored_actual)
            else:
                v = future_full_strict.get(target)
                actual_raw = float(v) if v is not None and np.isfinite(v) else None

            stored_baseline = q0_row.get(target) if target in q0_row.index else None
            if stored_baseline is not None and pd.notna(stored_baseline):
                baseline_raw = float(stored_baseline)
            else:
                v0 = q0_full_strict.get(target)
                baseline_raw = float(v0) if v0 is not None and np.isfinite(v0) else None

            rs = reg.get(target, q0_qend)
            if rs is None:
                continue
            k_val, mu_val, sigma_val = rs

            pred_reg     = reg_value(pred_raw,     scale_q0, k_val, mu_val, sigma_val)
            actual_reg   = (reg_value(actual_raw,   scale_q0, k_val, mu_val, sigma_val)
                            if actual_raw is not None else float("nan"))
            baseline_reg = (reg_value(baseline_raw, scale_q0, k_val, mu_val, sigma_val)
                            if baseline_raw is not None else float("nan"))

            rows.append({
                "firm_id":          int(gvkey),
                "target":           target,
                "quarter":          q0_qend,
                "forecast_horizon": h,
                "prediction":       pred_reg,
                "actual":           actual_reg,
                "baseline":         baseline_reg,
                "pred_raw":         pred_raw,
                "actual_raw":       actual_raw,
                "baseline_raw":     baseline_raw,
                "target_datadate":  future_date,
            })

    n_returned = sum(len(forecasts.get(f"Q{h}_active_returned", []))
                     for h in range(1, max_horizon + 1)
                     if forecasts.get(f"Q{h}") is not None)
    n_active_total = len(active_list) * max_horizon
    return rows, {
        "n_pass":              n_returned,
        "n_total":             n_active_total,
        "n_active_per_horizon": len(active_list),
        "max_horizon":         max_horizon,
    }


def process_origin(conn, parquet: str, gvkey, q0_date: pd.Timestamp,
                   max_horizon: int, col_sql: str,
                   reg: RegStatLookup, system_prompt: str, dry_run: bool,
                   present_in_q0: bool = False,
                   prompt_version: str = DEFAULT_PROMPT_VERSION,
                   llm_call=None,
                   raw_log: "RawResponseLog | None" = None,
                   tsrc: "TupleSource | None" = None,
                   firm_industry: "dict[int, str] | None" = None,
                   ) -> tuple[list[dict], dict | None, dict | None]:
    """Serial path: prepare → call LLM → finish. Returns (rows, adh, usage)."""
    meta, user_msg = prepare_origin(conn, parquet, gvkey, q0_date,
                                    max_horizon, col_sql, present_in_q0,
                                    prompt_version=prompt_version,
                                    tsrc=tsrc,
                                    firm_industry=firm_industry)
    if meta is None:
        return [], None, None

    if dry_run:
        rows, adh = finish_origin(meta, llm_text=None, reg=reg)
        return rows, adh, None

    if llm_call is None:
        llm_call = call_llm_v8
    # Quarter-end ISO, matching the prediction CSV's `quarter` column so raw
    # log records join 1:1 against scored rows.
    q0_iso = pd.Timestamp(meta["q0_qend"]).strftime("%Y-%m-%d")
    try:
        llm_text, usage = llm_call(system_prompt, user_msg)
    except Exception as e:
        print(f"    API error: {e}")
        if raw_log is not None:
            raw_log.log(int(gvkey), q0_iso, None, None,
                        error=f"{type(e).__name__}: {e}", user_msg=user_msg)
        return [], None, None

    if raw_log is not None:
        raw_log.log(int(gvkey), q0_iso, llm_text, usage, user_msg=user_msg)
    rows, adh = finish_origin(meta, llm_text, reg)
    return rows, adh, usage


# ── Final output ──────────────────────────────────────────────────────────────

def pairwise_vs_forma(llm_df: pd.DataFrame, llm_model_id: str,
                      forma_parquet: Path, output_dir: Path) -> None:
    """In-harness pairwise LLM-vs-Forma scoring is deferred in this release.

    The research-only comparison that used to live here depended on an internal
    scoring module that is not shipped. Score the saved LLM prediction parquet
    against Forma with `proforma20q evaluate` instead.
    """
    print("  In-harness pairwise LLM-vs-Forma comparison is not available in "
          "this release; score the saved predictions with `proforma20q "
          "evaluate` instead.")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Force UTF-8 stdout so progress banners with R^2 / Delta / box-drawing
    # chars don't crash on Windows cp1252 consoles partway through the run.
    # `errors='replace'` keeps any truly unrenderable codepoint from aborting.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass  # non-tty or older Python — fall back to default encoding
    parser = argparse.ArgumentParser(description="LLM vs Forma benchmark (pf_full-aligned)")
    parser.add_argument("--env-path", type=str, default=None)
    parser.add_argument("--source", type=str, default="tuples",
                        choices=["tuples", "compustat"],
                        help="Where history/origins/scale come from. 'tuples' "
                             "(default): Forma's stored "
                             "tuple_test__{feature_set}__{dataset}.parquet — exact "
                             "input parity with Forma (12q window, scale account, "
                             "Forma's reg-stats; no YTD/aociq re-derivation). "
                             "'compustat': the legacy path that re-derives history "
                             "from compustat_df.parquet (8q window). Production runs "
                             "use 'tuples'.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"(tuples) Forma dataset tag selecting "
                             f"tuple_test__{{feature_set}}__{{dataset}}.parquet and "
                             f"regularization_stats__{{feature_set}}__{{dataset}}."
                             f"parquet. Default '{DEFAULT_DATASET}' (Forma's "
                             f"canonical production run).")
    parser.add_argument("--n-origins", type=int, default=500,
                        help="Number of (firm, q0) origins to sample uniformly "
                             "from the eligible pool (default 500). A given firm "
                             "may contribute multiple origins.")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Random seed for origin sampling. Default: "
                             f"{ORIGIN_SEED} for the tuple var-block design (the "
                             "permanent shared seed inherited by every model), "
                             "else 42. With the same (seed, design, scope) the "
                             "exact same origin set is selected — so different "
                             "--model runs share an identical, maximally-"
                             "overlapping sample.")
    parser.add_argument("--sample-mode", type=str, default=None,
                        choices=["iid", "firm-block", "var-block"],
                        help="Origin sampling design. Default: 'var-block' when "
                             "--source tuples, else 'iid'. 'iid': uniform random "
                             "firm-quarters. 'firm-block': --n-firms firms x "
                             "--block-len FIXED consecutive origins. 'var-block' "
                             "(tuple design): --n-firms firms, one VARIABLE-length "
                             "block each (length = min(--max-block, run remaining "
                             "from a uniform start with >= --min-block left), "
                             "calendar-balanced start-years) — the production "
                             "design for revision-panel bias tests. Origin-cache "
                             "filename is tagged by the design, so all coexist.")
    parser.add_argument("--n-firms", type=int, default=1000,
                        help="(firm-block / var-block) Number of firms to sample. "
                             "Default 1000. var-block total origins ~= n_firms x "
                             "E[block] (measured; ~15 on the canonical tuples).")
    parser.add_argument("--block-len", type=int, default=4,
                        help="(firm-block only) FIXED consecutive origin quarters "
                             "per firm. Default 4 (= 3 revisions per firm "
                             "per overlapping target quarter).")
    parser.add_argument("--min-block", type=int, default=4,
                        help="(var-block) Minimum block length / minimum eligible "
                             "origins remaining at a valid start. Default 4.")
    parser.add_argument("--max-block", type=int, default=20,
                        help="(var-block) Maximum block length (cap). Default 20 "
                             "(= Forma max_horizon).")
    parser.add_argument("--noise-subset-size", type=int, default=0,
                        help="If >0, ALSO write a random noise-subset "
                             "origins file (this many origins) alongside the main "
                             "sample — the subset re-queried K times/model at "
                             "default temp to measure cross-API-call dispersion. "
                             "Selection only (writing the file); run the "
                             "K-replicate measurement by invoking K times with "
                             "--origins-file <that file> --replicate-id 0..K-1. "
                             "Default 0 (off).")
    parser.add_argument("--origins-file", type=str, default=None,
                        help="Run on this exact origins CSV (columns gvkey, "
                             "q0_date, max_horizon) instead of building/sampling a "
                             "pool. Use it to re-run the noise subset, or any "
                             "fixed origin set. Bypasses --sample-mode/--seed/"
                             "--n-firms.")
    parser.add_argument("--build-origins-only", action="store_true",
                        help="Build (and cache) the shared origins file + the "
                             "optional --noise-subset-size subset, print the "
                             "origins fingerprint, then EXIT before any LLM "
                             "calls. No model / API keys / spend needed — the "
                             "clean way to produce the canonical shared sample "
                             "ONCE, then point every model run at it via "
                             "--origins-file + --expect-origins-sha. Always "
                             "rebuilds deterministically (ignores an existing "
                             "cache) so the artifact + fingerprint are fresh.")
    parser.add_argument("--expect-origins-sha", type=str, default=None,
                        help="Abort unless the resolved origins set hashes to "
                             "this sha256 (see origins_fingerprint). Pin every "
                             "model's production run to the canonical shared "
                             "sample so a divergent origin set (a drifted "
                             "--n-firms/seed/scope, or a regenerated cache) "
                             "fails fast BEFORE spending, instead of silently "
                             "breaking the pairwise/bias comparison. Accepts a "
                             "short prefix (>=8 chars).")
    parser.add_argument("--replicate-id", type=int, default=-1,
                        help="Noise-replicate index. When >= 0, appends '_r{N}' "
                             "to the output tag so K repeat runs of the SAME "
                             "origins (same model, default temp) archive to "
                             "distinct prediction/response files instead of "
                             "clobbering. Default -1 (single, untagged run).")
    parser.add_argument("--provider", type=str, default="anthropic",
                        choices=["anthropic", "openai"],
                        help="LLM provider. 'anthropic' (default): Anthropic "
                             "SDK via Portkey (PORTKEY_API_KEY + PORTKEY_MODEL). "
                             "'openai': OpenAI models (--model required, e.g. "
                             "'gpt-5.5') — via Portkey's unified gateway when "
                             "PORTKEY_API_KEY is set and no direct OPENAI_API_KEY "
                             "is (see --openai-portkey-provider), else "
                             "api.openai.com directly with OPENAI_API_KEY. Both "
                             "support sync, --concurrency, and --batch.")
    parser.add_argument("--openai-api", type=str, default="chat",
                        choices=["chat", "responses"],
                        help="OpenAI wire protocol for --provider openai. "
                             "'chat' (default): /v1/chat/completions — no "
                             "reasoning summary is returned (gpt-5.x `thinking` "
                             "stays null). 'responses': /v1/responses with "
                             "reasoning.summary='detailed', which surfaces a "
                             "reasoning SUMMARY captured into `thinking` at the "
                             "same billed-token cost. Batch + sync both honor it.")
    parser.add_argument("--openai-portkey-provider", type=str, default=None,
                        help="Portkey Model Catalog provider slug for OpenAI "
                             "(e.g. '@openai' or '@your-portkey-slug'). When "
                             "set — or whenever OPENAI_API_KEY is absent but "
                             "PORTKEY_API_KEY is present — '--provider openai' "
                             "routes through Portkey's unified gateway/Batches "
                             "API instead of api.openai.com, authenticating with "
                             "the Portkey key only (OpenAI creds live in the "
                             "Model Catalog). Defaults to '@openai' if routing "
                             "via Portkey without an explicit slug.")
    parser.add_argument("--log-prompts", dest="log_prompts",
                        action="store_true", default=True,
                        help="Store the full user message in the raw-response "
                             "JSONL. DEFAULT ON — for the irreversible "
                             "production run the archive should be self-"
                             "contained (prompts are only otherwise regenerable "
                             "while data+code+flags don't drift). Adds "
                             "~10-15KB/origin. Pass --no-log-prompts for a lean "
                             "sha-only archive (pilots/ablations).")
    parser.add_argument("--no-log-prompts", dest="log_prompts",
                        action="store_false",
                        help="Store only the prompt's sha256, not the full "
                             "text — leaner archive for pilots/ablations where "
                             "the prompt is cheaply regenerable.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Use naive Q0-repeat forecast, no API calls.")
    parser.add_argument("--resume",   action="store_true",
                        help="Load existing CSV, skip processed origins, append new.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly discard existing artifacts for this "
                             "output tag (prediction CSV, raw-response archive, "
                             "batch sidecar) and start fresh. Without it, a run "
                             "that finds existing artifacts and no --resume "
                             "ABORTS — a forgotten --resume would otherwise "
                             "truncate paid-for results and resubmit the batch "
                             "(double spend).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Submit at most N NEW origins this invocation "
                             "(applied after the --resume skip, preserving "
                             "origins-file order). Enables the staged rollout: "
                             "run the full pinned origins file with --limit 20, "
                             "verify, then --resume --limit 200, verify, then "
                             "--resume with no limit for the remainder — one "
                             "fingerprint pin throughout, completed origins are "
                             "never re-billed.")
    parser.add_argument("--prompt-version", type=str,
                        default=DEFAULT_PROMPT_VERSION,
                        choices=sorted(PRIMITIVES_BY_VERSION.keys()),
                        help=f"Which production prompt arm to run. "
                             f"'unstructured' uses a free-form steering system "
                             f"prompt; 'structured' uses an explicit driver-"
                             f"hierarchy / roll-forward projection method. Both "
                             f"forecast the same primitive set; they differ only "
                             f"in the system prompt, and each gets its own output-"
                             f"file scope tag so the two arms can coexist on the "
                             f"same model. Default '{DEFAULT_PROMPT_VERSION}'.")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Path to system prompt .txt. Default: derived from "
                             "--prompt-version (prompts/system_prompt_{version}.txt). "
                             "Pass an explicit path to override.")
    parser.add_argument("--present-in-q0", action="store_true",
                        help="Restrict the per-firm primitive forecast set to items "
                             "non-null at Q0 specifically. REQUIRED FOR PARITY with "
                             "the canonical Forma, which sets `present_in_q0: true` "
                             "(r7e1b / r10e8a and the closed-overhang config) — pass "
                             "this on every production run. Default (off) scopes to "
                             "items present anywhere in the history window, which does "
                             "NOT match canonical Forma; use it only for the "
                             "any-history ablation.")
    parser.add_argument("--model", type=str, default=None,
                        help="Portkey model string (e.g. 'claude-opus-4-6', "
                             "'claude-sonnet-4-7', 'claude-haiku-4-5'). Overrides "
                             "PORTKEY_MODEL from the env file. Output paths are "
                             "tagged with the model so ablations don't clobber "
                             "each other.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent LLM requests (default 1, "
                             "serial). Use 4-16 for live runs to cut wall time "
                             "by ~Nx. Each origin still streams its rows to CSV "
                             "as soon as the response lands. Prompt caching "
                             "still works (1st call writes cache; remaining "
                             "calls hit it once it lands ~5s later).")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help=f"Forma's data directory. Must contain "
                             f"compustat_df.parquet and processed/ subdir. "
                             f"Default '{DEFAULT_DATA_DIR}' (relative to cwd). "
                             f"From a worktree pass an absolute path, e.g. "
                             f"/path/to/forma/data.")
    parser.add_argument("--feature-set", type=str, default=DEFAULT_FEATURE_SET,
                        help=f"Which regularization_stats__{{X}}.parquet to read "
                             f"from {{data_dir}}/processed/. Default "
                             f"'{DEFAULT_FEATURE_SET}' (matches the pf_full "
                             f"primitive set). The 12 core targets we score "
                             f"against are present in core, pro_forma, and "
                             f"pf_full stats files identically.")
    parser.add_argument("--forma-parquet", type=str,
                        default=str(DEFAULT_FORMA_PARQUET),
                        help=f"Path to a Forma forecast parquet to compare "
                             f"against in Step 4 (pairwise comparison on "
                             f"mutual overlap, in regularized change-from-"
                             f"baseline space). Default "
                             f"'{DEFAULT_FORMA_PARQUET}'. Set to empty string "
                             f"to skip the pairwise step.")

    parser.add_argument("--industry", type=str, default="on",
                        choices=["on", "off"],
                        help="Whether to state the firm's Fama-French 48 "
                             "industry in the prompt — input parity with "
                             "Forma's industry node (industry_mode: 'node') and "
                             "the tabular baselines' FF48 dummies. Default on. "
                             "'off' appends '_noindustry' to the output tag.")

    # ── Batch API ──────────────────────────────────────────────────────────
    parser.add_argument("--batch", action="store_true",
                        help="Submit all LLM calls as a single batch (50%% "
                             "input+output discount, async) — Anthropic via "
                             "Portkey /v1/batches, or OpenAI's native Batch "
                             "API with --provider openai. Mutually exclusive "
                             "with --concurrency > 1 and --dry-run. With "
                             "--resume + an existing batch sidecar, skip "
                             "submission and pick up from polling/result-"
                             "streaming. Output files are drop-in identical "
                             "to a sync run.")
    parser.add_argument("--batch-poll-secs", type=int, default=30,
                        help="How often (seconds) to poll batch status when "
                             "--batch is set. Default 30.")
    args = parser.parse_args()

    if args.batch and args.concurrency > 1:
        sys.exit("ERROR: --batch is mutually exclusive with --concurrency > 1 "
                 "(server-side batch IS the parallelism). Drop --concurrency "
                 "or drop --batch.")
    if args.batch and args.dry_run:
        sys.exit("ERROR: --batch makes no sense with --dry-run "
                 "(dry-run issues no API calls, so there is nothing to batch).")
    # Validate batch credentials up front. --env-path hasn't been loaded yet,
    # but for --batch the user MUST be using Portkey (we POST to api.portkey.ai)
    # so PORTKEY_API_KEY needs to be reachable somehow — either already in env
    # or in the file pointed at by --env-path. Defer the actual presence check
    # to right after env-loading (below); fail fast there before Phase A's
    # serial DB prep, which can take several minutes for n>=200.

    # Resolve sample-mode + seed defaults by source (tuples ⇒ var-block design,
    # permanent shared seed).
    if args.sample_mode is None:
        args.sample_mode = "var-block" if args.source == "tuples" else "iid"
    if args.seed is None:
        args.seed = ORIGIN_SEED if args.sample_mode == "var-block" else 42
    if args.source == "compustat" and args.sample_mode == "var-block":
        sys.exit("ERROR: --sample-mode var-block requires --source tuples "
                 "(it enumerates origins from the tuple grid).")

    DATA_DIR = Path(args.data_dir)
    PARQUET_PATH   = DATA_DIR / "compustat_df.parquet"
    PROCESSED_DIR  = DATA_DIR / "processed"
    if args.source == "tuples":
        # Forma's stored artifacts carry the dataset tag.
        REG_STATS_PATH = (PROCESSED_DIR /
                          f"regularization_stats__{args.feature_set}__{args.dataset}.parquet")
        TUPLE_PATH = (PROCESSED_DIR /
                      f"tuple_test__{args.feature_set}__{args.dataset}.parquet")
        ACCOUNT_MAP_PATH = PROCESSED_DIR / "account_id_map.csv"
        FIRM_MAP_PATH = PROCESSED_DIR / "firm_id_map.csv"
        for p in (REG_STATS_PATH, TUPLE_PATH, ACCOUNT_MAP_PATH, FIRM_MAP_PATH):
            if not p.exists():
                sys.exit(f"ERROR: required tuple-source file not found: {p}. "
                         f"Check --data-dir / --feature-set / --dataset.")
    else:
        REG_STATS_PATH = PROCESSED_DIR / f"regularization_stats__{args.feature_set}.parquet"
        TUPLE_PATH = ACCOUNT_MAP_PATH = FIRM_MAP_PATH = None
        if not PARQUET_PATH.exists():
            sys.exit(f"ERROR: parquet not found at {PARQUET_PATH}. Pass --data-dir "
                     f"to point at Forma's data root (e.g. /path/to/forma/data).")
    if not PROCESSED_DIR.exists():
        sys.exit(f"ERROR: processed dir not found at {PROCESSED_DIR}. Check --data-dir.")

    if args.env_path:
        import forma_llm as flm
        flm.load_env(Path(args.env_path))

    if args.provider == "openai":
        # OpenAI path: --model is the literal OpenAI model id (bare name; the
        # Portkey provider slug, if any, rides in the x-portkey-provider header,
        # not in the model string).
        if not args.model and not args.build_origins_only:
            sys.exit("ERROR: --provider openai requires --model "
                     "(e.g. --model gpt-5.5).")
        # Stash the explicit Portkey OpenAI slug (if given) so _openai_headers
        # can read it. Routing mode (Portkey vs direct) is decided by
        # _openai_via_portkey(): Portkey when a PORTKEY_API_KEY is present and
        # no direct OPENAI_API_KEY is.
        if args.openai_portkey_provider:
            os.environ["PORTKEY_OPENAI_PROVIDER"] = args.openai_portkey_provider
        # Stash the wire-protocol choice so the module-level OpenAI helpers
        # (build_openai_batch_line / submit_openai_batch / parser / sync) read
        # it without threading a flag through every call site.
        os.environ["OPENAI_API_MODE"] = args.openai_api
        if (not os.environ.get("OPENAI_API_KEY")
                and not os.environ.get("PORTKEY_API_KEY")
                and not args.dry_run and not args.build_origins_only):
            sys.exit("ERROR: --provider openai needs either OPENAI_API_KEY "
                     "(direct api.openai.com) or PORTKEY_API_KEY "
                     "(+ --openai-portkey-provider / PORTKEY_OPENAI_PROVIDER) to "
                     "route through Portkey. Found neither in the environment or "
                     "--env-path. Failing now to avoid wasting Phase A's serial "
                     "DB prep on a doomed run.")
        model_name = args.model
    elif args.model:
        # Preserve any @-slug routing prefix from the env-loaded PORTKEY_MODEL
        # so swapping models doesn't accidentally drop the Model Catalog
        # provider slug. e.g. PORTKEY_MODEL="@your-portkey-slug/claude-sonnet-4-6"
        # + --model claude-sonnet-4-7 -> "@your-portkey-slug/claude-sonnet-4-7".
        existing = os.environ.get("PORTKEY_MODEL", "")
        if (existing.startswith("@") and "/" in existing
                and not args.model.startswith("@")):
            slug = existing.split("/", 1)[0]
            os.environ["PORTKEY_MODEL"] = f"{slug}/{args.model}"
        else:
            os.environ["PORTKEY_MODEL"] = args.model
    if args.provider != "openai":
        model_name = os.environ.get("PORTKEY_MODEL", "claude-opus-4-6")

    # Auto-derive the Anthropic SDK env vars from PORTKEY_API_KEY when the
    # user's env file only has the Portkey-style vars set (the minimal case).
    # Lets a 2-line portkey.env (PORTKEY_API_KEY + PORTKEY_MODEL) Just Work
    # without forcing every consumer to also set ANTHROPIC_BASE_URL etc.
    portkey_key = os.environ.get("PORTKEY_API_KEY")
    if (args.batch and args.provider == "anthropic" and not portkey_key
            and not args.build_origins_only):
        sys.exit(
            "ERROR: --batch with --provider anthropic requires "
            "PORTKEY_API_KEY in the environment (or in --env-path); the "
            "Portkey /v1/batches endpoint is the Anthropic batch transport. "
            "Failing now to avoid wasting Phase A's serial DB prep on a "
            "doomed run."
        )
    if args.provider == "openai":
        pass  # no Anthropic env derivation needed
    elif portkey_key:
        # FORCE these (hard set, NOT setdefault). A Claude Code / CI / pod shell
        # frequently pre-sets ANTHROPIC_BASE_URL (e.g. to api.anthropic.com) and
        # possibly ANTHROPIC_AUTH_TOKEN. With setdefault the stale value wins, so
        # the sync SDK silently POSTs to Anthropic directly with our Portkey
        # headers + a dummy key → 401 "invalid x-api-key". When PORTKEY_API_KEY
        # is set the user wants Portkey, so we override unconditionally. (The
        # batch path is immune — it uses raw httpx against an explicit Portkey
        # URL — which is why /v1/batches authed fine while sync 401'd.)
        os.environ["ANTHROPIC_BASE_URL"] = "https://api.portkey.ai"
        os.environ["ANTHROPIC_AUTH_TOKEN"] = "dummy"  # ignored; auth via header
        # Route the sync SDK exactly like the batch path: x-portkey-api-key +
        # x-portkey-provider header, with the BARE model name (slug stripped) in
        # the request body (call_llm_v8 uses _portkey_batch_model()). The batch
        # path proves this header shape works for this workspace.
        # _portkey_batch_provider_value() yields '@<slug>' (or '@anthropic').
        os.environ["ANTHROPIC_CUSTOM_HEADERS"] = (
            f"x-portkey-api-key: {portkey_key}\n"
            f"x-portkey-provider: {_portkey_batch_provider_value()}"
        )

    # Provider-dispatched sync caller (used by the serial + concurrent paths;
    # the batch path has its own provider branch further down).
    if args.provider == "openai":
        def llm_call(system, user, _m=model_name):
            return call_llm_openai(system, user, _m)
    else:
        llm_call = call_llm_v8

    # Diagnostic: show the routing config (with the API key masked) so 401s
    # are easier to debug.
    if not args.dry_run and args.provider == "openai":
        if _openai_via_portkey():
            key = os.environ.get("PORTKEY_API_KEY", "")
            masked = (key[:6] + "..." + key[-4:]) if len(key) > 12 else "***"
            provider = os.environ.get("PORTKEY_OPENAI_PROVIDER", "@openai")
            print(f"  Provider: openai via Portkey ({_openai_base()})  "
                  f"x-portkey-provider={provider}  x-portkey-api-key={masked}")
        else:
            key = os.environ.get("OPENAI_API_KEY", "")
            masked = (key[:6] + "..." + key[-4:]) if len(key) > 12 else "***"
            print(f"  Provider: openai (direct {_openai_base()})  "
                  f"OPENAI_API_KEY={masked}")
    if not args.dry_run and args.provider != "openai":
        masked_headers = []
        for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines():
            line = line.strip()
            if line.lower().startswith("x-portkey-api-key:"):
                k = line.split(":", 1)[1].strip()
                masked = (k[:6] + "..." + k[-4:]) if len(k) > 12 else "***"
                masked_headers.append(f"x-portkey-api-key: {masked}")
            elif line:
                masked_headers.append(line)
        print(f"  ANTHROPIC_BASE_URL={os.environ.get('ANTHROPIC_BASE_URL')}")
        print(f"  ANTHROPIC_CUSTOM_HEADERS:")
        for h in masked_headers:
            print(f"    {h}")

    system_prompt_file = (Path(args.system_prompt) if args.system_prompt
                          else system_prompt_path_for(args.prompt_version))
    if not system_prompt_file.exists():
        sys.exit(f"ERROR: system prompt not found at {system_prompt_file}. "
                 f"Check --prompt-version / --system-prompt.")
    lookback_qs = TUPLE_HISTORY_QS if args.source == "tuples" else HISTORY_QS
    system_prompt = load_system_prompt(system_prompt_file,
                                       lookback_qs=lookback_qs,
                                       max_horizon=MAX_HORIZON)
    print(f"Loaded system prompt from {system_prompt_file} ({len(system_prompt):,} chars; "
          f"lookback={lookback_qs}q, max_horizon={MAX_HORIZON})")
    print(f"Prompt version: {args.prompt_version}")
    print(f"Model: {model_name}")
    scope_label = "present_in_q0=True (Q0-only)" if args.present_in_q0 \
                  else "present_in_q0=False (any history quarter, Forma pf_full default)"
    print(f"Per-firm primitive scope: {scope_label}")

    # Output-file scope suffix: the industry ablation and per-replicate noise
    # draws each get a tag so they don't clobber the headline run.
    industry_on = (args.industry == "on")
    ablation_bits: list[str] = []
    if not industry_on:
        ablation_bits.append("noindustry")
    if args.replicate_id >= 0:
        ablation_bits.append(f"r{args.replicate_id}")
    ablation_suffix = ("_" + "_".join(ablation_bits)) if ablation_bits else ""

    paths = output_paths(args.present_in_q0, model_name,
                         prompt_version=args.prompt_version,
                         feature_set=args.feature_set,
                         ablation_suffix=ablation_suffix)
    PRED_PARQUET   = paths["pred_parquet"]
    PRED_CSV       = paths["pred_csv"]
    ADHERENCE_JSON = paths["adherence_json"]
    BATCH_SIDECAR  = paths["batch_sidecar"]
    BATCH_META     = paths["batch_meta"]
    RAW_JSONL      = paths["raw_jsonl"]
    METRIC_PREFIX  = paths["metric_prefix"]
    # Run-meta sidecar: records prompt-affecting settings (notably the industry
    # flag) that the headline filename does NOT encode when on, so --resume can
    # refuse to append onto a CSV written under a different setting (PR #186).
    RUN_META_PATH  = PRED_CSV.with_name(PRED_CSV.name + ".runmeta.json")
    print(f"Output parquet: {PRED_PARQUET}")
    print(f"Output CSV:     {PRED_CSV}")
    if args.batch:
        print(f"Batch sidecar:  {BATCH_SIDECAR}")

    METRICS_DIR = metrics_dir_for(args.feature_set)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Metrics dir:    {METRICS_DIR}")

    # Data source: 'tuples' uses Forma's stored tuples (no DuckDB / compustat
    # parquet); 'compustat' opens the raw parquet via DuckDB (legacy path).
    tsrc = None
    conn = None
    col_sql = None
    parquet = None
    if args.source == "tuples":
        print(f"Loading tuple source {TUPLE_PATH.name} (12q window, "
              f"dataset={args.dataset}) ...")
        tsrc = TupleSource(
            TUPLE_PATH, ACCOUNT_MAP_PATH, firm_map_path=FIRM_MAP_PATH,
            max_lookback=TUPLE_HISTORY_QS, max_horizon=MAX_HORIZON,
            min_lookback=MIN_LOOKBACK,
            test_start_q=TEST_START_Q, test_end_q=TEST_END_Q,
        )
        print(f"  {len(tsrc.lookup):,} firms loaded; scale account_id="
              f"{tsrc.scale_id}; firm_id output = gvkey (Forma join key)")
    else:
        parquet = str(PARQUET_PATH)
        print(f"Opening {PARQUET_PATH} via DuckDB ...")
        conn, avail_cols, col_sql = open_db(PARQUET_PATH)

    # Firm → FF48 industry label (input parity with Forma's industry node /
    # the tabular baselines' FF48 dummies). Tuples: read the per-firm
    # industry_id Forma actually consumed straight from the tuple file. Compustat
    # (legacy): re-derive from the modal SIC per firm, exactly as make_dataset
    # does. Keyed by int gvkey to match prepare_origin's int(gvkey) lookup.
    firm_industry: dict[int, str] = {}
    if industry_on:
        INDUSTRY_MAP_PATH = PROCESSED_DIR / "industry_id_map.csv"
        if not INDUSTRY_MAP_PATH.exists():
            print(f"  WARN: {INDUSTRY_MAP_PATH} not found — industry signal "
                  f"disabled for this run.")
        else:
            id_to_label = load_industry_labels(INDUSTRY_MAP_PATH)
            if args.source == "tuples":
                for fid_int, iid in tsrc.firm_industry_id.items():
                    firm_industry[tsrc.gvkey_of(fid_int)] = id_to_label.get(int(iid))
                if not firm_industry:
                    print(f"  WARN: tuple file {TUPLE_PATH.name} carried no "
                          f"industry_id column — industry signal unavailable.")
            else:
                from ff48 import load_ff48_mapping, sic_to_ff48
                sic_ranges, unknown_id, _ = load_ff48_mapping(
                    str(CONFIGS_DIR / "ff48_sic_ranges.json"))
                sic_df = pd.read_parquet(PARQUET_PATH, columns=["gvkey", "sich"])
                modal = sic_df.groupby("gvkey")["sich"].agg(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
                ff48 = sic_to_ff48(modal, sic_ranges, unknown_id)
                for gv, iid in ff48.items():
                    firm_industry[int(gv)] = id_to_label.get(int(iid))
            firm_industry = {g: lbl for g, lbl in firm_industry.items() if lbl}
            print(f"  Industry signal ON: {len(firm_industry):,} firms mapped "
                  f"to FF48 labels")

    # Step 1 — regularization stats. Tuples: load Forma's own reg-stats verbatim
    # (exact scoring parity). Compustat: build/cache (legacy).
    print("\n[Step 1/4] Regularization stats ...")
    if args.source == "tuples":
        print(f"  Loading Forma reg-stats {REG_STATS_PATH.name}")
        stats_df = pd.read_parquet(REG_STATS_PATH)
    else:
        stats_df = build_regularization_stats(conn, parquet, REG_STATS_PATH)
    reg = RegStatLookup(stats_df)

    # Step 2 — origin pool (cached by sampling design + seed + scope so
    # multiple model runs reuse the same firm-quarter sample, maximizing
    # pairwise overlap).
    sample_mode = args.sample_mode.replace("-", "_")
    ORIGINS_CSV = origins_csv_path(args.present_in_q0, args.seed, args.n_origins,
                                   sample_mode=sample_mode,
                                   n_firms=args.n_firms,
                                   block_len=args.block_len,
                                   source=args.source,
                                   min_block=args.min_block,
                                   max_block=args.max_block)
    if args.origins_file:
        # Fixed origin set (e.g. the noise subset) — bypass pool/sampler.
        ORIGINS_CSV = Path(args.origins_file)
        if not ORIGINS_CSV.exists():
            sys.exit(f"ERROR: --origins-file not found: {ORIGINS_CSV}")
        sample = pd.read_csv(ORIGINS_CSV, parse_dates=["q0_date"])
        sample["gvkey"] = sample["gvkey"].astype(str).str.zfill(6)
        print(f"\n[Step 2/4] Loaded fixed origins ({len(sample):,}) from "
              f"--origins-file {ORIGINS_CSV.name}"
              + (f" (replicate r{args.replicate_id})" if args.replicate_id >= 0 else ""))
    elif ORIGINS_CSV.exists() and not args.build_origins_only:
        sample = pd.read_csv(ORIGINS_CSV, parse_dates=["q0_date"])
        sample["gvkey"] = sample["gvkey"].astype(str).str.zfill(6)
        print(f"\n[Step 2/4] Loaded cached origins ({len(sample):,}) from {ORIGINS_CSV.name}")
        print(f"  (scope={'q0-only' if args.present_in_q0 else 'any-history'}, "
              f"mode={sample_mode}, seed={args.seed})")
    else:
        print(f"\n[Step 2/4] Building origin pool (every eligible firm-quarter in 2010-2024) ...")
        raw_pool = None
        if args.source == "tuples":
            raw_pool = tsrc.build_origin_pool()  # firm_id(gvkey), center_q, max_h
            # gvkey/q0_date view so iid/firm-block samplers work on tuples too.
            pool = raw_pool.rename(columns={"firm_id": "gvkey"}).copy()
            pool["q0_date"] = pool["center_q"].map(tsrc.qmap.to_qend_ts)
            pool = pool[["gvkey", "q0_date", "max_horizon"]]
        else:
            pool = build_origin_pool(conn, parquet)
        print(f"  Eligible origins in pool: {len(pool):,} "
              f"(spanning {pool['gvkey'].nunique():,} unique firms)")
        if sample_mode == "var_block":
            vb = variable_block_sample_origins(
                raw_pool, args.n_firms, args.seed,
                min_block=args.min_block, max_block=args.max_block,
                calendar_balance=True)
            vb["gvkey"] = vb["firm_id"]
            vb["q0_date"] = vb["center_q"].map(tsrc.qmap.to_qend_ts)
            sample = vb[["gvkey", "q0_date", "max_horizon"]]
            print(f"  var-block design: {sample['gvkey'].nunique():,} firms x "
                  f"variable blocks (min={args.min_block}, max={args.max_block}); "
                  f"E[block]={len(sample)/max(1, sample['gvkey'].nunique()):.1f}")
        elif sample_mode == "firm_block":
            sample = firm_block_sample_origins(pool, args.n_firms,
                                               args.block_len, args.seed)
            print(f"  firm-block design: {sample['gvkey'].nunique():,} firms "
                  f"x {args.block_len} consecutive origins")
        else:
            sample = random_sample_origins(pool, args.n_origins, args.seed)
        sample = sample.copy()
        sample["gvkey"] = sample["gvkey"].astype(str).str.zfill(6)
        sample.to_csv(ORIGINS_CSV, index=False)
        print(f"  Sampled {len(sample):,} origins (seed={args.seed}) -> {ORIGINS_CSV.name}")
        # Show the time / horizon distribution of the sample
        sample["q0_year"] = pd.to_datetime(sample["q0_date"]).dt.year
        print(f"  Year distribution: {sample['q0_year'].value_counts().sort_index().to_dict()}")
        print(f"  max_horizon distribution:")
        print(f"    mean={sample['max_horizon'].mean():.1f}  "
              f"median={sample['max_horizon'].median():.0f}  "
              f"min={sample['max_horizon'].min()}  max={sample['max_horizon'].max()}")
        sample = sample.drop(columns=["q0_year"])

        # Optional random noise subset (selection only — the K-replicate run is
        # a separate invocation pointed at this file).
        if args.noise_subset_size > 0:
            noise = select_noise_subset(sample, args.noise_subset_size,
                                        seed=args.seed + 1)
            noise_path = ORIGINS_CSV.with_name(
                ORIGINS_CSV.stem + f"__noise{args.noise_subset_size}.csv")
            noise.to_csv(noise_path, index=False)
            noise_sha = origins_fingerprint(noise)
            print(f"  Noise subset ({len(noise):,} origins, random) "
                  f"-> {noise_path.name}")
            print(f"  Noise fingerprint (sha256): {noise_sha}")
            print(f"  Pin every noise-replicate run to it with:")
            print(f"    --origins-file {noise_path} "
                  f"--expect-origins-sha {noise_sha[:16]} --replicate-id <N>")

    # ── Origins fingerprint — prove the shared sample is byte-identical across
    # every model. The cache filename encodes the design params, but once a
    # single flag drifts (e.g. one model run with a different --n-firms) it
    # silently points at a different file; the fingerprint + --expect-origins-sha
    # catch that BEFORE any spend, since a divergent sample breaks the
    # cross-model pairwise / bias comparison.
    origins_sha = origins_fingerprint(sample)
    print(f"  Origins fingerprint (sha256): {origins_sha}")
    if args.expect_origins_sha:
        want = args.expect_origins_sha.strip().lower()
        if len(want) < 8:
            sys.exit("ERROR: --expect-origins-sha must be >= 8 hex chars "
                     "(pass the full sha or a >=8-char prefix).")
        if not origins_sha.startswith(want):
            sys.exit(
                f"ERROR: origins fingerprint mismatch — this run's sample hashes "
                f"to {origins_sha}, but --expect-origins-sha requires {want}*. "
                f"The origin set diverged from the canonical shared sample "
                f"(check --n-firms / --seed / --present-in-q0 / --dataset, or a "
                f"regenerated cache). Refusing to spend on a non-comparable run.")
        print(f"  Origins fingerprint matches --expect-origins-sha ({want}*)  [OK]")

    if args.build_origins_only:
        print(f"\n[build-origins-only] Shared origins artifact ready; skipping "
              f"LLM calls (no spend).")
        print(f"  Origins file: {ORIGINS_CSV}")
        print(f"  Fingerprint:  {origins_sha}")
        print(f"  Pin every model run to it with:")
        print(f"    --origins-file {ORIGINS_CSV} "
              f"--expect-origins-sha {origins_sha[:16]}")
        return

    # Step 3 — per-origin LLM forecasts
    print(f"\n[Step 3/4] LLM forecasts ({len(sample)} origins, "
          f"{'DRY RUN' if args.dry_run else 'live calls via ' + model_name}) ...")

    acc = R2Accumulator()
    all_rows: list[dict] = []
    n_success = n_no_window = n_batch_error = 0
    cov_pass = cov_total = 0
    seen_origins: set[tuple[int, str]] = set()  # (gvkey, q0_date_iso)

    if args.resume and PRED_CSV.exists():
        # Setting-mismatch guards. The headline filename is industry-agnostic
        # when --industry on (only 'off' tags the name), so an industry-on
        # resume of a pre-PR / industry-off CSV would silently mix industry-free
        # and industry-bearing rows — seen_origins skips by (gvkey, quarter)
        # only and never re-forecasts the stale rows (PR #186 review). The same
        # logic covers present_in_q0 (belt-and-suspenders: the scope also tags
        # the filename) and the origins sha — resuming a CSV written from a
        # DIFFERENT origin set appends a silently mixed sample that
        # --expect-origins-sha (which only pins the CURRENT run's file) can
        # never catch post-hoc.
        cur_ind = "on" if industry_on else "off"
        if RUN_META_PATH.exists():
            prev_meta: dict = {}
            try:
                with open(RUN_META_PATH) as f:
                    prev_meta = json.load(f)
            except Exception as e:
                print(f"  (could not read run-meta {RUN_META_PATH.name}: {e})")
            prev_ind = prev_meta.get("industry")
            if prev_ind is not None and prev_ind != cur_ind:
                sys.exit(
                    f"ERROR: --resume target {PRED_CSV.name} was written with "
                    f"--industry {prev_ind}, but this run is --industry {cur_ind}. "
                    f"Resuming would mix industry-free and industry-bearing rows. "
                    f"Start a fresh run (delete {PRED_CSV.name}) or rerun with "
                    f"--industry {prev_ind}.")
            prev_q0 = prev_meta.get("present_in_q0")
            if prev_q0 is not None and bool(prev_q0) != bool(args.present_in_q0):
                sys.exit(
                    f"ERROR: --resume target {PRED_CSV.name} was written with "
                    f"present_in_q0={bool(prev_q0)}, but this run has "
                    f"present_in_q0={bool(args.present_in_q0)}. Resuming would "
                    f"mix primitive scopes. Rerun with the matching "
                    f"--present-in-q0 setting (canonical production runs "
                    f"require it).")
            prev_sha = prev_meta.get("origins_sha")
            if prev_sha is not None and prev_sha != origins_sha:
                sys.exit(
                    f"ERROR: --resume target {PRED_CSV.name} was written from "
                    f"origins sha {prev_sha}, but this run's origins hash to "
                    f"{origins_sha}. Resuming would append rows from a "
                    f"different origin set — a silently mixed sample. Point "
                    f"--origins-file at the original canonical file, or start "
                    f"a fresh output tag.")
        elif PRED_CSV.stat().st_size > 0 and industry_on:
            # No sidecar + non-empty CSV ⇒ predates the guard (likely a pre-PR,
            # industry-free file). Only safe to append under --industry off.
            sys.exit(
                f"ERROR: --resume target {PRED_CSV.name} has no run-meta sidecar "
                f"— it predates the industry-parity guard and is probably "
                f"industry-free, but this run is --industry on. Resuming would "
                f"silently mix prompts. Start a fresh run (delete {PRED_CSV.name}) "
                f"or, to continue the legacy file, rerun with --industry off.")
        print(f"  --resume: loading existing {PRED_CSV} ...")
        prev = pd.read_csv(PRED_CSV, parse_dates=["quarter", "target_datadate"])
        if len(prev) > 0:
            seen_origins = set(
                (int(r["firm_id"]), pd.Timestamp(r["quarter"]).strftime("%Y-%m-%d"))
                for _, r in prev[["firm_id", "quarter"]].drop_duplicates().iterrows()
            )
            for _, r in prev.iterrows():
                d = r.to_dict()
                d["quarter"] = pd.Timestamp(d["quarter"])
                all_rows.append(d)
                acc.add((r["target"], int(r["forecast_horizon"])),
                        float(r["prediction"]), float(r["actual"]), float(r["baseline"]))
            n_success = len(seen_origins)
            print(f"  resumed from {n_success:,} origins, {len(prev):,} prior rows")
        if ADHERENCE_JSON.exists():
            try:
                with open(ADHERENCE_JSON) as f:
                    prev_adh = json.load(f)
                cov_pass  = int(prev_adh.get("pass", 0))
                cov_total = int(prev_adh.get("total", 0))
                print(f"  restored coverage counters: {cov_pass}/{cov_total}")
            except Exception as e:
                print(f"  (could not restore adherence JSON: {e})")

    # Overwrite guard — re-running a command WITHOUT --resume truncates the
    # prediction CSV, wipes the paid-for (unrecoverable) raw-response archive,
    # ignores any submitted-batch sidecar, and resubmits the whole batch: a
    # full double spend. Starting fresh over existing artifacts must be an
    # explicit decision (--overwrite), never a forgotten flag.
    if not args.resume and not args.overwrite:
        blockers = []
        if PRED_CSV.exists() and PRED_CSV.stat().st_size > 0:
            blockers.append(PRED_CSV.name)
        if RAW_JSONL.exists() and RAW_JSONL.stat().st_size > 0 \
                and not args.dry_run:
            blockers.append(f"{RAW_JSONL.name} (paid-for response archive)")
        if BATCH_SIDECAR.exists():
            blockers.append(f"{BATCH_SIDECAR.name} (SUBMITTED batch — a fresh "
                            f"run would resubmit it and double-bill)")
        if blockers:
            sys.exit(
                "ERROR: existing run artifacts found for this output tag:\n"
                + "".join(f"  - {b}\n" for b in blockers)
                + "Re-running without --resume would truncate/overwrite them "
                  "(and resubmit any in-flight batch — a double spend).\n"
                  "Pass --resume to continue this run, or --overwrite to "
                  "explicitly discard the artifacts and start fresh.")

    csv_exists = PRED_CSV.exists() and args.resume
    csv_fh = open(PRED_CSV, "a" if csv_exists else "w", buffering=1)
    csv_writer = None

    # Stamp the run-meta sidecar so a later --resume can detect a setting change
    # (the guard above reads this). Written for every mode (batch/serial/dry).
    try:
        with open(RUN_META_PATH, "w") as f:
            json.dump({"industry": "on" if industry_on else "off",
                       "prompt_version": args.prompt_version,
                       "present_in_q0": bool(args.present_in_q0),
                       "origins_sha": origins_sha}, f, indent=2)
    except Exception as e:
        print(f"  (could not write run-meta {RUN_META_PATH.name}: {e})")

    # Raw-completion archive (live runs only — dry-run makes no API calls).
    raw_log: RawResponseLog | None = None
    if not args.dry_run:
        raw_log = RawResponseLog(RAW_JSONL, model_name, args.prompt_version,
                                 system_prompt, resume=args.resume,
                                 log_prompts=args.log_prompts)

    # Cumulative usage tallies for the cache-hit report at the end.
    tot_in = tot_out = tot_cache_write = tot_cache_read = 0

    # Pre-skip already-seen origins so concurrency planning sees only new work.
    pending = []
    for idx, (_, s_row) in enumerate(sample.iterrows()):
        gvkey = int(s_row["gvkey"])
        q0_date = pd.Timestamp(s_row["q0_date"])
        max_h = int(s_row["max_horizon"])
        if (gvkey, q0_date.strftime("%Y-%m-%d")) in seen_origins:
            continue
        pending.append((idx + 1, gvkey, q0_date, max_h))

    # Staged rollout: cap how many NEW origins this invocation submits. The
    # full pinned origins file is still loaded (so --expect-origins-sha holds);
    # only the work list is truncated, in stable file order.
    if args.limit is not None:
        if args.limit < 0:
            sys.exit("ERROR: --limit must be >= 0.")
        if len(pending) > args.limit:
            print(f"  --limit {args.limit}: submitting the first "
                  f"{args.limit:,} of {len(pending):,} pending origins "
                  f"(staged rollout; --resume later for the rest).")
            pending = pending[:args.limit]

    print(f"  Pending origins: {len(pending):,} "
          f"(skipped {len(sample) - len(pending):,} already in CSV) "
          f"| concurrency = {args.concurrency}")

    def _emit(label_idx: int, gvkey: int, rows: list, adh: dict | None,
              usage: dict | None) -> bool:
        """Persist one origin's results. Returns True if rows were written."""
        nonlocal csv_writer, n_success, n_no_window, cov_pass, cov_total
        nonlocal tot_in, tot_out, tot_cache_write, tot_cache_read
        if not rows:
            print(f"  [{label_idx:>4}/{len(sample)}] gvkey={gvkey:06d}: "
                  f"no window / bad scale / parse fail")
            n_no_window += 1
            return False
        n_success += 1
        msg = f"  [{label_idx:>4}/{len(sample)}] gvkey={gvkey:06d}: OK ({len(rows)} rows)"
        if usage is not None:
            tot_in          += usage.get("input_tokens", 0)
            tot_out         += usage.get("output_tokens", 0)
            tot_cache_write += usage.get("cache_creation_input_tokens", 0)
            tot_cache_read  += usage.get("cache_read_input_tokens", 0)
            cr = usage.get("cache_read_input_tokens", 0)
            cw = usage.get("cache_creation_input_tokens", 0)
            msg += f" [in={usage.get('input_tokens',0)} out={usage.get('output_tokens',0)}"
            if cr:  msg += f" cache_read={cr}"
            if cw:  msg += f" cache_write={cw}"
            msg += "]"
        print(msg)

        firm_df = pd.DataFrame(rows)
        if csv_writer is None:
            csv_writer = list(firm_df.columns)
            if not csv_exists:
                csv_fh.write(",".join(csv_writer) + "\n")
        firm_df.to_csv(csv_fh, header=False, index=False)
        csv_fh.flush()
        for r in rows:
            all_rows.append(r)
            acc.add((r["target"], r["forecast_horizon"]),
                    r["prediction"], r["actual"], r["baseline"])
        if adh is not None:
            cov_pass  += adh["n_pass"]
            cov_total += adh["n_total"]
            with open(ADHERENCE_JSON, "w") as f:
                json.dump({"pass": cov_pass, "total": cov_total,
                           "pct": (cov_pass/cov_total*100) if cov_total else 0.0,
                           "metric": "primitive_coverage"}, f, indent=2)
        # Record the REAL dedupe key — the Phase-D resume skip and the pre-skip
        # loop both match on (gvkey, q0-quarter ISO). A junk marker here would
        # let a re-streamed or re-processed result double-write silently.
        seen_origins.add(
            (gvkey, pd.Timestamp(rows[0]["quarter"]).strftime("%Y-%m-%d")))
        if n_success % FLUSH_EVERY == 0:
            pd.DataFrame(all_rows)[["firm_id","target","quarter","forecast_horizon","prediction"]] \
                .to_parquet(PRED_PARQUET, index=False)
        return True

    t0 = time.time()
    if args.batch:
        # ── Batch API path (Portkey /v1/batches or OpenAI native) ──────────
        # Phase A: serial prepare → build meta_by_custom_id + batch_requests
        # Phase B: submit + write sidecar
        # Phase C: poll until terminal
        # Phase D: stream results, finish_origin + _emit
        if args.provider == "openai":
            batch_model = model_name
        else:
            batch_model = _portkey_batch_model()
            if batch_model != model_name:
                print(f"  Batch model (slug stripped): {batch_model}  "
                      f"(provider header: {_portkey_batch_provider_value()})")

        meta_by_custom_id: dict[str, dict] = {}
        batch_requests: list = []
        label_by_custom_id: dict[str, int] = {}
        gvkey_by_custom_id: dict[str, int] = {}

        existing_batch_id: str | None = None
        if args.resume and BATCH_SIDECAR.exists() and BATCH_META.exists():
            try:
                with open(BATCH_SIDECAR) as f:
                    sidecar = json.load(f)
                sidecar_provider = sidecar.get("provider", "anthropic")
                if sidecar_provider != args.provider:
                    sys.exit(f"ERROR: batch sidecar {BATCH_SIDECAR.name} was "
                             f"submitted via provider '{sidecar_provider}' "
                             f"but this run is --provider {args.provider}. "
                             f"Rerun with the matching provider (or delete "
                             f"the sidecar to resubmit).")
                # The submitted batch's wire mode is authoritative for parsing
                # (like batch_id/provider): the output body shape is fixed at
                # submit time, not by whatever --openai-api this resume passed.
                # Restore it so a forgotten --openai-api responses on resume
                # can't silently parse a Responses batch under the chat branch
                # and drop every forecast. Absent (pre-this-change sidecars) ⇒
                # chat, which is the historical default.
                sidecar_mode = sidecar.get("openai_api_mode")
                if (args.provider == "openai" and sidecar_mode
                        and sidecar_mode != _openai_api_mode()):
                    print(f"  WARN: batch sidecar was submitted with "
                          f"--openai-api {sidecar_mode!r} but this resume "
                          f"passed {_openai_api_mode()!r}; using the sidecar's "
                          f"mode to parse the batch (a mismatch would drop "
                          f"every forecast).")
                    os.environ["OPENAI_API_MODE"] = sidecar_mode
                existing_batch_id = sidecar.get("batch_id")
                with open(BATCH_META, "rb") as f:
                    meta_by_custom_id = pickle.load(f)
                # Rebuild label/gvkey lookups from persisted meta. We don't
                # have the original sample-row label_idx after a restart, so
                # use the order of meta_by_custom_id as a stand-in (matches
                # submission order, since dicts preserve insertion order).
                for i, (cid, m) in enumerate(meta_by_custom_id.items(), 1):
                    label_by_custom_id[cid] = i
                    gvkey_by_custom_id[cid] = int(m["gvkey"])
                # Sanity: sidecar should contain ≤ pending count (some
                # pending origins fail prepare_origin and are filtered out
                # before submission, so equality isn't guaranteed; but the
                # sidecar holding *more* than pending implies sidecar/CSV
                # drift — typically a stale sidecar paired with a partially
                # rebuilt CSV). Warn but don't abort, since the user may
                # have intentionally pruned the CSV.
                if len(meta_by_custom_id) > len(pending) + len(seen_origins):
                    print(f"  WARN: sidecar holds {len(meta_by_custom_id):,} "
                          f"requests but sample shows {len(pending) + len(seen_origins):,} "
                          f"origins (pending+seen). Sidecar may be stale; "
                          f"results will still process but consider "
                          f"deleting {BATCH_SIDECAR.name} and rerunning.")
                print(f"  --resume + batch sidecar found: skipping Phase A/B, "
                      f"resuming poll on batch_id={existing_batch_id} "
                      f"({len(meta_by_custom_id):,} requests in sidecar).")
            except Exception as e:
                print(f"  WARN: could not load batch sidecar ({e}); "
                      f"falling through to full submission.")
                existing_batch_id = None
                meta_by_custom_id = {}
                label_by_custom_id = {}
                gvkey_by_custom_id = {}

        if existing_batch_id is None:
            # Phase A: serial prepare
            print(f"  [Phase A] Preparing {len(pending):,} origins serially ...")
            for label_idx, gvkey, q0_date, max_h in pending:
                meta, user_msg = prepare_origin(
                    conn, parquet, str(gvkey).zfill(6), q0_date, max_h,
                    col_sql, present_in_q0=args.present_in_q0,
                    prompt_version=args.prompt_version,
                    tsrc=tsrc,
                    firm_industry=firm_industry,
                )
                if meta is None:
                    _emit(label_idx, gvkey, [], None, None)
                    continue
                custom_id = f"{int(gvkey):06d}_{q0_date.strftime('%Y%m%d')}"
                if custom_id in meta_by_custom_id:
                    print(f"  WARN: duplicate custom_id {custom_id}; skipping later origin.")
                    continue
                # Stash the prompt fingerprint (and optionally the prompt) so
                # Phase D can write a complete raw-response record — by then
                # the request payloads are long gone.
                meta["_user_msg_sha"] = _sha256(user_msg)
                if args.log_prompts:
                    meta["_user_msg"] = user_msg
                meta_by_custom_id[custom_id] = meta
                label_by_custom_id[custom_id] = label_idx
                gvkey_by_custom_id[custom_id] = int(gvkey)
                if args.provider == "openai":
                    batch_requests.append(build_openai_batch_line(
                        custom_id, system_prompt, user_msg, batch_model,
                    ))
                else:
                    batch_requests.append(build_batch_request(
                        custom_id, system_prompt, user_msg, batch_model,
                    ))

            if not batch_requests:
                print("  No eligible origins to submit; skipping batch.")
            else:
                # Phase B: submit + persist sidecar
                print(f"  [Phase B] Submitting batch ...")
                submit_fn = (submit_openai_batch if args.provider == "openai"
                             else submit_batch)
                existing_batch_id = submit_fn(
                    batch_requests, BATCH_SIDECAR, BATCH_META,
                    meta_by_custom_id, batch_model,
                )
                # Free request payloads from RAM — we no longer need them.
                batch_requests.clear()

        if existing_batch_id is not None:
            # Phase C: poll
            print(f"  [Phase C] Polling batch ...")
            if args.provider == "openai":
                final_status = poll_openai_batch(existing_batch_id,
                                                 args.batch_poll_secs)
                results_iter = iter_openai_batch_results(final_status)
            else:
                poll_batch(existing_batch_id, args.batch_poll_secs)
                results_iter = iter_batch_results(existing_batch_id)

            # Phase D: stream + process
            print(f"  [Phase D] Processing batch results ...")
            n_processed = 0
            for cid, text, usage, err, thinking in results_iter:
                meta = meta_by_custom_id.get(cid)
                if meta is None:
                    print(f"  WARN: result for unknown custom_id {cid}; skipping.")
                    continue
                gvkey = gvkey_by_custom_id.get(cid, int(meta["gvkey"]))
                label_idx = label_by_custom_id.get(cid, 0)

                # Honor --resume: skip origins already in CSV.
                q0_iso = pd.Timestamp(meta["q0_qend"]).strftime("%Y-%m-%d")
                if (gvkey, q0_iso) in seen_origins:
                    n_processed += 1
                    continue

                if raw_log is not None:
                    raw_log.log(gvkey, q0_iso, text, usage, error=err,
                                user_msg=meta.get("_user_msg"),
                                user_msg_sha=meta.get("_user_msg_sha"),
                                custom_id=cid, thinking=thinking)

                if err is not None:
                    # Per-result errored/expired/canceled — DISTINCT from
                    # "no window / bad scale", which is a Phase-A skip
                    # (origin never made it into the batch). Track them
                    # separately so the end-of-run summary doesn't conflate
                    # data-eligibility filters with API failures.
                    print(f"  [{label_idx:>4}/{len(sample)}] gvkey={gvkey:06d} "
                          f"({cid}): batch error: {err}")
                    n_batch_error += 1
                    n_processed += 1
                    continue

                rows, adh = finish_origin(meta, text, reg)
                _emit(label_idx, gvkey, rows, adh, usage)
                n_processed += 1

            print(f"  [Phase D] Processed {n_processed:,} results.")

            # Batch fully fetched & processed — retire the sidecar so the NEXT
            # --resume invocation proceeds to Phase A and submits any remaining
            # pending origins (staged rollouts, error gap-fill) instead of
            # re-polling this finished batch forever. Errored origins are not
            # in the CSV, so a later --resume run resubmits exactly those.
            # A crash before this point leaves the sidecar in place (re-poll +
            # re-fetch are idempotent: results are deduped by custom_id).
            _bid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(existing_batch_id))
            for _p in (BATCH_SIDECAR, BATCH_META):
                try:
                    if _p.exists():
                        _done = _p.with_name(f"{_p.name}.done.{_bid}")
                        if _done.exists():
                            _done.unlink()
                        _p.rename(_done)
                except OSError as e:
                    print(f"  WARN: could not archive {_p.name}: {e}")
            print(f"  Batch sidecar archived (*.done.{_bid}) — a later "
                  f"--resume run submits any remaining pending origins.")
    elif args.concurrency <= 1 or args.dry_run:
        # Serial path (also used for dry-run since there's no API call to parallelise).
        for label_idx, gvkey, q0_date, max_h in pending:
            rows, adh, usage = process_origin(
                conn, parquet, str(gvkey).zfill(6), q0_date, max_h,
                col_sql, reg, system_prompt, args.dry_run,
                present_in_q0=args.present_in_q0,
                prompt_version=args.prompt_version,
                llm_call=llm_call,
                raw_log=raw_log,
                tsrc=tsrc,
                firm_industry=firm_industry,
            )
            _emit(label_idx, gvkey, rows, adh, usage)
            if not args.dry_run:
                time.sleep(API_DELAY_S)
    else:
        # Concurrent path: serial DB prep, parallel LLM calls.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        prepared = []
        for label_idx, gvkey, q0_date, max_h in pending:
            meta, user_msg = prepare_origin(
                conn, parquet, str(gvkey).zfill(6), q0_date, max_h,
                col_sql, present_in_q0=args.present_in_q0,
                prompt_version=args.prompt_version,
                tsrc=tsrc,
                firm_industry=firm_industry,
            )
            if meta is None:
                _emit(label_idx, gvkey, [], None, None)
                continue
            prepared.append((label_idx, gvkey, meta, user_msg))
        print(f"  Pre-prepared {len(prepared):,} origins; dispatching to "
              f"{args.concurrency} concurrent LLM workers ...")

        # Per-future deadline = SDK timeout (180s) + buffer for retries. If
        # something does hang past this, we want to skip it instead of
        # blocking the pool indefinitely.
        FUTURE_DEADLINE_S = 360.0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            future_to_meta = {
                ex.submit(llm_call, system_prompt, user_msg):
                    (label_idx, gvkey, meta, user_msg)
                for label_idx, gvkey, meta, user_msg in prepared
            }
            for fut in as_completed(future_to_meta, timeout=None):
                label_idx, gvkey, meta, user_msg = future_to_meta[fut]
                q0_iso = pd.Timestamp(meta["q0_qend"]).strftime("%Y-%m-%d")
                try:
                    llm_text, usage = fut.result(timeout=FUTURE_DEADLINE_S)
                except Exception as e:
                    print(f"  [{label_idx:>4}/{len(sample)}] gvkey={gvkey:06d}: "
                          f"API error: {type(e).__name__}: {e}")
                    if raw_log is not None:
                        raw_log.log(gvkey, q0_iso, None, None,
                                    error=f"{type(e).__name__}: {e}",
                                    user_msg=user_msg)
                    n_no_window += 1
                    continue
                if raw_log is not None:
                    raw_log.log(gvkey, q0_iso, llm_text, usage,
                                user_msg=user_msg)
                rows, adh = finish_origin(meta, llm_text, reg)
                _emit(label_idx, gvkey, rows, adh, usage)

    # Final parquet flush after the loop in either path.
    if all_rows:
        pd.DataFrame(all_rows)[["firm_id","target","quarter","forecast_horizon","prediction"]] \
            .to_parquet(PRED_PARQUET, index=False)

    csv_fh.close()
    if raw_log is not None:
        raw_log.close()
        print(f"  Raw responses archived -> {RAW_JSONL}")
    if conn is not None:
        conn.close()
    elapsed = time.time() - t0
    summary = f"Success: {n_success} | No window: {n_no_window}"
    if n_batch_error:
        summary += f" | Batch errors: {n_batch_error}"
    summary += f" | Elapsed: {elapsed:.0f}s"
    print(f"\n  {summary}")

    # Cache hit / token report (live mode only)
    if not args.dry_run and (tot_in + tot_cache_read + tot_cache_write) > 0:
        total_input = tot_in + tot_cache_read + tot_cache_write
        cache_hit_pct = (tot_cache_read / total_input * 100) if total_input else 0.0
        print(f"  Token totals: input={tot_in:,}  output={tot_out:,}  "
              f"cache_write={tot_cache_write:,}  cache_read={tot_cache_read:,}")
        print(f"  Cache hit rate: {cache_hit_pct:.1f}% of total input tokens "
              f"served from cache (saves ~90% on those tokens).")

    if not all_rows:
        sys.exit("ERROR: No successful forecasts.")

    df_pred = pd.DataFrame(all_rows)
    forma_cols = ["firm_id", "target", "quarter", "forecast_horizon", "prediction"]
    df_pred[forma_cols].to_parquet(PRED_PARQUET, index=False)
    print(f"  Saved forecast parquet ({len(df_pred):,} rows) -> {PRED_PARQUET}")
    print(f"  CSV written incrementally -> {PRED_CSV}")

    coverage_pct = (cov_pass / cov_total * 100) if cov_total > 0 else float("nan")

    # Step 4 — metrics
    print(f"\n[Step 4/4] Computing R² in regularized change-from-baseline space ...")
    by_th     = aggregate_r2(acc, ["target", "forecast_horizon"])
    by_target = aggregate_r2(acc, ["target"])
    by_hzn    = aggregate_r2(acc, ["forecast_horizon"])
    glb       = aggregate_r2(acc, [])

    prefix = METRIC_PREFIX
    by_th.to_csv(METRICS_DIR / f"r2_scores__{prefix}__by_target_horizon.csv", index=False)
    by_target.to_csv(METRICS_DIR / f"r2_scores__{prefix}__by_target.csv", index=False)
    by_hzn.to_csv(METRICS_DIR / f"r2_scores__{prefix}__by_horizon.csv", index=False)
    glb.to_csv(METRICS_DIR / f"r2_scores__{prefix}__global.csv", index=False)

    r2_global   = float(glb["r2"].iloc[0]) if len(glb) > 0 else float("nan")
    rmse_global = float(glb["rmse"].iloc[0]) if len(glb) > 0 else float("nan")
    n_pairs     = int(glb["n"].iloc[0]) if len(glb) > 0 else 0

    print(f"\n  R² by target:")
    print(by_target.sort_values("r2", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  R² by horizon:")
    print(by_hzn.sort_values("forecast_horizon").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  Overall: R²={r2_global:.4f}, RMSE={rmse_global:.4f}, n_obs={n_pairs:,}")
    print(f"  Primitive coverage: {coverage_pct:.1f}% ({cov_pass}/{cov_total})")

    print(f"\n{'='*120}")
    print(f"  LLM {args.prompt_version} self-report (pf_full, regularized Δ space)")
    print(f"{'='*120}")
    r2_pct = r2_global * 100 if not math.isnan(r2_global) else float("nan")
    r2_str = f"{r2_pct:.1f}%" if not math.isnan(r2_pct) else "N/A"
    rmse_str = f"{rmse_global:.4f}" if not math.isnan(rmse_global) else "N/A"
    print(f"  Sample: {n_success} origins ({n_pairs:,} pairs scored)")
    print(f"  R² = {r2_str}   RMSE = {rmse_str}   "
          f"primitive coverage = {coverage_pct:.1f}%")

    forma_parquet = Path(args.forma_parquet) if args.forma_parquet else None
    if forma_parquet:
        pairwise_dir = METRICS_DIR / "pairwise" / METRIC_PREFIX
        pairwise_vs_forma(df_pred, METRIC_PREFIX, forma_parquet, pairwise_dir)
    else:
        print("  --forma-parquet was empty; skipping pairwise comparison vs Forma.")


if __name__ == "__main__":
    main()
