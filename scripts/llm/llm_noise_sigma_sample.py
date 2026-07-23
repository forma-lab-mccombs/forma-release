"""Preliminary noise-replicate dispersion (sigma_sample) for the LLM benchmark.

Measures the wobble of the forecasts THEMSELVES across the K=5 decoding draws
(main run + replicates r0..r3), in the regularized z-space, WITHOUT referencing
ground truth. For each (firm_id, target, quarter, forecast_horizon) cell present
in all K draws, sigma_sample = std of those predictions. Reports the determinism
fraction (how often the draws are near-identical, i.e. sigma < 1e-9) and how
sigma_sample scales with forecast horizon.

Run from the repo root:  python scripts/llm/llm_noise_sigma_sample.py
"""
import glob
import numpy as np
import pandas as pd

FORECASTS = "results/forecasts"
# Replicate parquets get moved to a subdir (eval hygiene: keeps them out of
# evaluate.py's non-recursive results/forecasts/*predictions.parquet glob).
SEARCH_DIRS = [FORECASTS, f"{FORECASTS}/noise_replicates"]
KEYS = ["firm_id", "target", "quarter", "forecast_horizon"]

CELLS = [
    ("unstructured sonnet", "llm_unstructured", "claude_sonnet_5"),
    ("unstructured opus",   "llm_unstructured", "claude_opus_4_8"),
    ("unstructured gpt",    "llm_unstructured", "gpt_5_5"),
    ("structured sonnet",   "llm_structured",   "claude_sonnet_5"),
    ("structured opus",     "llm_structured",   "claude_opus_4_8"),
    ("structured gpt",      "llm_structured",   "gpt_5_5"),
]


def _one(name_glob):
    """First match of name_glob across SEARCH_DIRS (main dir + moved-out replicate subdir)."""
    for d in SEARCH_DIRS:
        hits = glob.glob(f"{d}/{name_glob}")
        if hits:
            return hits[0]
    return None


def load_draws(prefix, infix):
    """Return list of (label, DataFrame[keys+prediction]) for main + r0..r3."""
    draws = []
    main = _one(f"{prefix}_q0__{infix}__*predictions.parquet")
    if main:
        draws.append(("main", main))
    for n in range(4):
        rep = _one(f"{prefix}_q0_r{n}__{infix}__*predictions.parquet")
        if rep:
            draws.append((f"r{n}", rep))
    frames = []
    for label, path in draws:
        df = pd.read_parquet(path, engine="fastparquet", columns=KEYS + ["prediction"])
        frames.append(df.rename(columns={"prediction": label}))
    return [lbl for lbl, _ in draws], frames


def analyze(prefix, infix):
    labels, frames = load_draws(prefix, infix)
    if len(frames) < 2:
        return None
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=KEYS, how="inner")
    n_joined = len(merged)
    merged = merged.dropna(subset=labels)   # a NaN prediction in any draw would poison std + miscount %ident
    n_dropped = n_joined - len(merged)
    pred = merged[labels].to_numpy(np.float64)
    sig = pred.std(axis=1, ddof=1)
    merged["sigma_sample"] = sig
    zero_frac = float(np.mean(sig < 1e-9))
    by_h = merged.groupby("forecast_horizon")["sigma_sample"].mean()
    return {
        "n_draws": len(labels),
        "n_cells": len(merged),
        "n_dropped": n_dropped,
        "mean": float(sig.mean()),
        "median": float(np.median(sig)),
        "p90": float(np.quantile(sig, 0.90)),
        "zero_frac": zero_frac,
        "by_h": by_h,
    }


def main():
    hs = [1, 4, 8, 12, 16, 20]
    print("sigma_sample = std of the K decoding draws (main + r0..r3; per-cell count in the "
          "'draws' column) in regularized z-space (pop std ~= 1). Ground-truth-free.\n")
    header = (f"{'cell':20} {'draws':>5} {'ncells':>8} {'mean':>7} {'median':>7} "
              f"{'p90':>6} {'%ident':>7}   " + "  ".join(f"h{h:<2}" for h in hs))
    print(header)
    print("-" * len(header))
    results = {}
    for label, prefix, infix in CELLS:
        r = analyze(prefix, infix)
        if r is None:
            print(f"{label:20}  (files missing)")
            continue
        results[label] = r
        hvals = "  ".join(f"{r['by_h'].get(h, float('nan')):.3f}" for h in hs)
        print(f"{label:20} {r['n_draws']:5} {r['n_cells']:8,} {r['mean']:7.4f} "
              f"{r['median']:7.4f} {r['p90']:6.3f} {100*r['zero_frac']:6.1f}%   {hvals}")
    dropped = sum(r["n_dropped"] for r in results.values())
    if dropped:
        print(f"\nNote: dropped {dropped} cell-row(s) with a NaN prediction in some draw.")
    print("\nColumns h1..h20 = mean sigma_sample at that forecast horizon.")
    print("%ident = fraction of cells with near-zero dispersion (sigma < 1e-9) across draws.")


if __name__ == "__main__":
    main()
