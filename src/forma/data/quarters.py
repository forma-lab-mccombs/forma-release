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
