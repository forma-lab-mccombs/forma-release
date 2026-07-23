"""Torch-free Fama-French 48 industry mapping helpers (SIC-range lookup)."""

import json
import pandas as pd
from typing import Dict, List, Tuple


def load_ff48_mapping(config_path: str = "configs/ff48_sic_ranges.json") -> Tuple[List[Tuple[int, int, int]], int, Dict[int, str]]:
    """Load Fama-French 48 industry classification from JSON config.

    Returns:
        sic_ranges: List of (sic_start, sic_end, industry_id) tuples, sorted by start.
        unknown_id: Integer ID for firms with missing/unmatched SIC codes.
        id_to_name: Dict mapping industry_id -> industry name.
    """
    with open(config_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    sic_ranges = []
    id_to_name = {}
    for ind in data["industries"]:
        ind_id = ind["id"]
        id_to_name[ind_id] = ind["name"]
        for rng in ind["sic_ranges"]:
            sic_ranges.append((rng[0], rng[1], ind_id))

    unknown_id = data["unknown_id"]
    id_to_name[unknown_id] = data["unknown_name"]

    sic_ranges.sort(key=lambda x: x[0])
    return sic_ranges, unknown_id, id_to_name


def sic_to_ff48(sich_series: pd.Series, sic_ranges: List[Tuple[int, int, int]], unknown_id: int) -> pd.Series:
    """Map SIC codes to Fama-French 48 industry IDs (vectorized).

    Args:
        sich_series: Series of SIC codes (may contain NaN).
        sic_ranges: Output of load_ff48_mapping.
        unknown_id: ID for unmatched/missing SIC.

    Returns:
        Series of int64 FF48 industry IDs.
    """
    result = pd.Series(unknown_id, index=sich_series.index, dtype="int64")
    valid = sich_series.notna()
    sic_vals = sich_series[valid].astype(int)

    # First-match-wins across ranges: only assign where the slot is still unknown_id.
    # FF48 ranges happen to be disjoint, so this is currently a no-op safety net,
    # but it preserves the documented intent if a future ranges file overlaps.
    for start, end, ind_id in sic_ranges:
        mask = sic_vals.between(start, end)
        unassigned = result.loc[sic_vals.index] == unknown_id
        matched = mask & unassigned
        result.loc[sic_vals[matched].index] = ind_id

    return result
