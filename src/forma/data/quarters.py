"""Canonical quarter-offset <-> calendar-quarter conversion.

The tuple pipeline encodes calendar quarters as integer offsets with base
1969Q4 = 0 (see FormaWindowDataset._get_reg_stats). This is the single shared
implementation of the inverse map; train.py's predict loop,
extract_embeddings.py, and scripts/scenario_pilot.py all use it. The test
suite keeps its own literal copies deliberately, as independent oracles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_YEAR = 1969  # offset 0 == 1969Q4


def qoff_to_period_index(qoff) -> pd.PeriodIndex:
    """Integer quarter offsets (base 1969Q4 = 0) -> quarterly PeriodIndex."""
    qoff = np.asarray(qoff, dtype=np.int64)
    years = BASE_YEAR + (3 + qoff) // 4
    quarters = ((3 + qoff) % 4) + 1
    return pd.PeriodIndex.from_fields(year=years, quarter=quarters, freq='Q')


def qoff_to_quarter_end(qoff) -> pd.DatetimeIndex:
    """Integer quarter offsets -> end-of-quarter timestamps (forecast-file convention)."""
    return qoff_to_period_index(qoff).end_time
