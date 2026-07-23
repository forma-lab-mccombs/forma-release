"""
Naive Random Walk with Drift model for financial forecasting.

This model implements the simplest baseline approach described in docs/architecture/EXPERIMENTAL_DESIGN.md:
- Uses the last observed value as the base
- Adds the historical average quarterly growth rate for the firm in question
- Forecasts: value_i,q+h = value_i,q=0 + h * avg_growth_rate_i

avg_growth_rate_i is computed as the mean year-over-year changes in "value" over the last 4 quarters:
avg_growth_rate = mean( value_yoy_0 + value_yoy_1 + value_yoy_2 + value_yoy_3 ) )"""

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
    """Naive Random Walk with Drift model with shared core prediction logic."""
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
    """Naive (no drift) model reusing shared prediction logic."""
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


