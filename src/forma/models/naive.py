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

"""
Seasonal random walk baselines (with and without drift) for financial forecasting.

Both models are SEASONAL: the base for horizon h is the most recent observation
falling in the SAME fiscal quarter as the target, not the last observed value.
The base lag is (4 - h) % 4, so h=1 reads q-3, h=2 reads q-2, h=3 reads q-1, and
h=4 reads q; the base therefore sits 4, 8, 12, 16, or 20 quarters before the
target. At h in {4, 8, 12, 16, 20} the lag is 0 and the seasonal random walk
coincides with a plain no-change forecast, but at every other horizon it does
not. This is the standard naive expectation for quarterly accounting series
(Foster 1977; Bernard and Thomas 1990), which are seasonally differenced.

The seasonal base is assigned upstream, in
FormaNaiveDataModule._create_naive_prediction_data (src/forma/data/tabular_data.py),
which emits one row per (target, lag, horizon) with the chosen lag's level in
`current_value`. This module only consumes `current_value`.

- NaiveModel (`naive`), the paper's baseline:
      value_i,q+h = value_i,q-((4-h) % 4)
- NaiveDriftModel (`naive_drift`), not used in the ICAIF'26 paper:
      value_i,q+h = value_i,q-lag + (h + lag) * avg_growth_rate_i / 4
  where the effective horizon (h + lag) spans base to target, and
  avg_growth_rate_i is the mean of the last 4 year-over-year changes:
      mean( value_yoy_0, value_yoy_1, value_yoy_2, value_yoy_3 )
"""

import pandas as pd
from typing import Optional
import os

from .base import BaseModel

def _predict_common(
    test_data: pd.DataFrame,
    model_name: str,
    use_drift: bool,
    stream_save_paths=None,
) -> pd.DataFrame:
    """Shared prediction + denormalization + reporting logic for naive models.
    
    Args:
        test_data: Input dataframe with required columns including 'lag_level', 'current_value', and 'forecast_horizon'
        save_path: Output parquet path
        model_name: Name of the calling model
        use_drift: Whether to apply drift term (avg growth * effective horizon / 4)
    """
    print(f"Processing {len(test_data):,} prediction requests{' with drift' if use_drift else ' (no drift)'}...")
    forecast_df = test_data.copy()

    # Compute raw predictions in (z-scored symlog) space using cyclical pattern
    if use_drift:
        if 'avg_growth_rate' not in forecast_df.columns:
            raise ValueError("avg_growth_rate column required for drift model but not found")
        
        # Calculate effective horizon for drift calculation
        # For cyclical pattern, we need to account for the lag when calculating drift
        # If we're using q-k to predict q+h, the effective time difference is h+k quarters
        effective_horizon = forecast_df['forecast_horizon'] + forecast_df['lag_level']
        
        raw_predictions = (
            forecast_df['current_value'] +
            effective_horizon * forecast_df['avg_growth_rate'] / 4.0
        )
    else:
        # No drift - just use the lagged value directly
        raw_predictions = forecast_df['current_value']
        
    forecast_df['prediction'] = raw_predictions

    forecast_df['model'] = model_name

    # Emit only the standard forecast schema shared with the sklearn baselines.
    # The intermediate columns (current_value, lag_level, avg_growth_rate,
    # actual_value, ...) are computation/diagnostic-only: evaluate.py derives its
    # own actuals from the truth file and joins on
    # ['firm_id', 'quarter', 'target', 'forecast_horizon'], so carrying them just
    # bloats the parquet (naive files were ~5x the baselines'). Drop them here.
    cols = ['firm_id', 'target', 'quarter', 'forecast_horizon', 'prediction', 'model']
    forecast_df = forecast_df[cols]

    # Streaming save: train.py:1171 always passes stream_save_paths (per-exp path +
    # global "latest"); honor the same save contract as the other models. The naive
    # forecast is already tall and fully materialized (no wide->tall melt to chunk),
    # so write it in one atomic shot via the shared crash-safe writer and return None.
    if stream_save_paths:
        import pyarrow as pa
        from .forecast_io import finalize_forecast_frame, atomic_parquet_writers
        tall = finalize_forecast_frame(forecast_df)
        tbl = pa.Table.from_pandas(tall, preserve_index=False)
        with atomic_parquet_writers(stream_save_paths) as write_table:
            write_table(tbl)
        return None

    return forecast_df

class NaiveDriftModel(BaseModel):
    """Seasonal random walk WITH drift; shared core prediction logic.

    Base is the same-fiscal-quarter lag (see module docstring), not the last
    observed value. Not used in the ICAIF'26 paper.
    """
    def __init__(self, growth_periods: int = 4, min_history_required: int = 8, **kwargs):
        super().__init__("naive_drift", **kwargs)
        self.growth_periods = growth_periods
        self.min_history_required = min_history_required

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> None:
        print(f"Initializing {self.model_name} model...")
        print("  No training required - using pre-computed growth rates from data")
        self.is_fitted = True
        print(f"Completed initialization for {self.model_name}")

    def predict(self, test_data: pd.DataFrame, stream_save_paths=None,
                stream_group_size=4) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return _predict_common(test_data, self.model_name, use_drift=True,
                               stream_save_paths=stream_save_paths)

    def save(self, folder_path: str) -> None:
        # just save growth_periods and min_history_required as a text file
        os.makedirs(folder_path, exist_ok=True)
        params_path = os.path.join(folder_path, "model_params.txt")
        with open(params_path, "w") as f:
            f.write(f"growth_periods={self.growth_periods}\n")
            f.write(f"min_history_required={self.min_history_required}\n")

    def load(self, folder_path: str) -> None:
        params_path = os.path.join(folder_path, "model_params.txt")
        with open(params_path, "r") as f:
            for line in f:
                key, value = line.strip().split("=")
                if key == "growth_periods":
                    self.growth_periods = int(value)
                elif key == "min_history_required":
                    self.min_history_required = int(value)
        self.is_fitted = True


class NaiveModel(BaseModel):
    """Seasonal random walk, no drift; the paper's `naive` baseline.

    Predicts the same-fiscal-quarter base unchanged (see module docstring):
    value_i,q+h = value_i,q-((4-h) % 4). This is NOT a plain no-change
    forecast except at h in {4, 8, 12, 16, 20}.
    """
    def __init__(self, **kwargs):
        super().__init__("naive", **kwargs)

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> None:
        print(f"Initializing {self.model_name} model (no drift)...")
        print("  No training required - using current value only")
        self.is_fitted = True
        print(f"Completed initialization for {self.model_name}")

    def predict(self, test_data: pd.DataFrame, stream_save_paths=None,
                stream_group_size=4) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return _predict_common(test_data, self.model_name, use_drift=False,
                               stream_save_paths=stream_save_paths)

    def save(self, folder_path: str) -> None:
        # nothing to save here
        pass

    def load(self, folder_path: str) -> None:
        # nothing to load here
        self.is_fitted = True


