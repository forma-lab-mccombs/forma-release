import pandas as pd
from typing import Dict, Optional
from pathlib import Path

from forma.data.config_utils import resolve_dataset_path


class TabularDataModule:
    """
    DataModule

    This DataModule handles tabular data for baseline models like
    sklearn and XGBoost.
    """

    def __init__(self, data_dir: str = "data/processed", batch_size: int = 32, data_config: Dict = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_config = data_config or {}
        self.batch_size = batch_size
        self.feature_set = self.data_config.get('feature_set', 'core')
        self.dataset_tag = self.data_config.get('dataset_tag')

        # Get target variables from config - will be finalized in setup() when we have column metadata
        # Store the config settings for later use
        targets_config = self.data_config.get('targets', {})
        self._targets_all = targets_config.get('all', False)
        self._targets_variables = targets_config.get('variables', ['niq'])
        self.target_variables = None  # Will be set in setup()

        # Will be populated during setup
        self.feature_columns = []
        self.target_columns = []
        self.metadata_columns = []
        self.reg_stats_df = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _split_path(self, kind: str, ext: str = "parquet") -> Path:
        return resolve_dataset_path(
            self.data_dir, kind, self.feature_set, self.dataset_tag, ext=ext
        )

    def prepare_data(self):
        """Check if processed tabular data exists."""
        required_paths = [
            self._split_path('tabular_train'),
            self._split_path('tabular_val'),
            self._split_path('tabular_test'),
            self._split_path('regularization_stats'),
            self.data_dir / f'column_metadata__{self.feature_set}.json',
        ]

        missing_files = [str(p) for p in required_paths if not p.exists()]

        if missing_files:
            print(f"Missing processed data files: {missing_files}")
            print("Please build the ProForma-20Q dataset first with the companion package: proforma20q build")
            raise FileNotFoundError(f"Required processed data files not found: {missing_files}")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""

        # Load column metadata
        import json
        metadata_path = self.data_dir / f'column_metadata__{self.feature_set}.json'
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        self.feature_columns = column_metadata['feature_columns']
        self.target_columns = column_metadata['target_columns']
        self.metadata_columns = column_metadata['metadata_columns']

        # Determine target variables based on config
        if self._targets_all:
            # targets.all = True: use all account types from feature columns as targets
            # Extract unique base feature names from feature columns (e.g., 'niq' from 'niq_level_0')
            base_features = set()
            for col in self.feature_columns:
                # Feature columns are like 'niq_level_0', 'niq_yoy_1', etc.
                if '_level_' in col:
                    base_name = col.rsplit('_level_', 1)[0]
                    base_features.add(base_name)
                elif '_yoy_' in col:
                    base_name = col.rsplit('_yoy_', 1)[0]
                    base_features.add(base_name)
            self.target_variables = sorted(base_features)
            print(f"targets.all=True: Using all {len(self.target_variables)} features as targets")
        else:
            # targets.all = False: use specified variables from config, or fall back to metadata
            if self._targets_variables:
                self.target_variables = self._targets_variables
            else:
                # Fall back to what was saved during make_dataset
                self.target_variables = column_metadata.get('target_variables', ['niq'])
            print(f"Using {len(self.target_variables)} specified target variables: {self.target_variables}")

        print(f"Loaded column metadata from {metadata_path}")
        print(f"  Features: {len(self.feature_columns)} columns")
        print(f"  Targets: {len(self.target_columns)} columns")
        print(f"  Metadata: {len(self.metadata_columns)} columns")

        # Load regularization statistics for z-score to symlog conversion
        self.reg_stats_df = pd.read_parquet(self._split_path('regularization_stats'))

        if stage == "fit" or stage is None:
            # Load training and validation data
            self.train_data = pd.read_parquet(self._split_path('tabular_train'))
            self.val_data = pd.read_parquet(self._split_path('tabular_val'))

            # If no_cv mode, combine train and validation data
            if self.data_config.get('no_cv', False):
                print("NO-CV MODE: Combining train and validation data for training...")
                self.train_data = pd.concat([self.train_data, self.val_data], ignore_index=True)
                # Keep val_data as None to indicate no separate validation
                self.val_data = None

        if stage == "test" or stage is None:
            self.test_data = pd.read_parquet(self._split_path('tabular_test'))

    def get_train_data(self) -> pd.DataFrame:
        """Get training data for sklearn/XGBoost models."""
        return self.train_data

    def get_val_data(self) -> pd.DataFrame:
        """Get validation data for sklearn/XGBoost models."""
        return self.val_data

    def get_test_data(self) -> pd.DataFrame:
        """Get test data for sklearn/XGBoost models."""
        return self.test_data

    def get_regularization_stats(self) -> pd.DataFrame:
        """Get regularization statistics for z-score to symlog conversion."""
        return self.reg_stats_df


class NaiveDriftDataModule:
    """
    Data module for the Naive Drift model.

    This module handles the special data format needed for the Naive Drift model,
    which uses YoY features from the tabular dataset to compute growth rates.
    """

    def __init__(self, data_dir: str = "data/processed", data_config: Dict = None):
        self.data_dir = Path(data_dir)
        self.data_config = data_config or {}
        # Get target config settings - will be finalized in setup() when we have data
        targets_config = self.data_config.get('targets', {})
        self._targets_all = targets_config.get('all', False)
        self._targets_variables = targets_config.get('variables', ['niq', 'fcfq'])
        self.target_variables = None  # Will be set in setup()
        self.feature_set = self.data_config.get('feature_set', 'core')
        self.dataset_tag = self.data_config.get('dataset_tag')
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _split_path(self, kind: str, ext: str = "parquet") -> Path:
        return resolve_dataset_path(
            self.data_dir, kind, self.feature_set, self.dataset_tag, ext=ext
        )

    def prepare_data(self):
        """Check if processed data exists."""
        required_paths = [
            self._split_path('tabular_train'),
            self._split_path('tabular_test'),
        ]

        missing_files = [str(p) for p in required_paths if not p.exists()]

        if missing_files:
            print(f"Missing processed data files: {missing_files}")
            print("Please build the ProForma-20Q dataset first with the companion package: proforma20q build")
            raise FileNotFoundError(f"Required processed data files not found: {missing_files}")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""

        # Load column metadata to determine target variables
        import json
        metadata_path = self.data_dir / f'column_metadata__{self.feature_set}.json'
        with open(metadata_path, 'r') as f:
            column_metadata = json.load(f)

        feature_columns = column_metadata['feature_columns']

        # Determine target variables based on config
        if self._targets_all:
            # targets.all = True: use all account types from feature columns as targets
            base_features = set()
            for col in feature_columns:
                if '_level_' in col:
                    base_name = col.rsplit('_level_', 1)[0]
                    base_features.add(base_name)
                elif '_yoy_' in col:
                    base_name = col.rsplit('_yoy_', 1)[0]
                    base_features.add(base_name)
            self.target_variables = sorted(base_features)
            print(f"targets.all=True: Using all {len(self.target_variables)} features as targets")
        else:
            # targets.all = False: use specified variables from config, or fall back to metadata
            if self._targets_variables:
                self.target_variables = self._targets_variables
            else:
                self.target_variables = column_metadata.get('target_variables', ['niq', 'fcfq'])
            print(f"Using {len(self.target_variables)} specified target variables: {self.target_variables}")

        if stage == "fit" or stage is None:
            # Load processed tabular data
            train_df = pd.read_parquet(self._split_path('tabular_train'))

            # If no_cv mode, also load and combine validation data
            if self.data_config.get('no_cv', False):
                val_path = self._split_path('tabular_val')
                if val_path.exists():
                    print("NO-CV MODE: Combining train and validation data for training...")
                    val_df = pd.read_parquet(val_path)
                    train_df = pd.concat([train_df, val_df], ignore_index=True)

            # Convert to naive prediction format (same format as test data)
            print("Loading train data for naive prediction format...")
            self.train_data = self._create_naive_prediction_data(train_df, split_name='train')

            # Validation data is not used for naive drift, but we'll create it for consistency
            self.val_data = None

        if stage == "test" or stage is None:
            # Load processed tabular data
            test_df = pd.read_parquet(self._split_path('tabular_test'))

            # Convert to naive prediction format
            self.test_data = self._create_naive_prediction_data(test_df, split_name='test')

    def _create_naive_prediction_data(self, tabular_df: pd.DataFrame, split_name: str = 'test') -> pd.DataFrame:
        """
        Create prediction data for Naive/Naive Drift models using cyclical pattern.

        Uses cyclical prediction pattern:
        - q-3 values (level_3) predict q+1, q+5, q+9, etc. (horizons 1, 5, 9, ...)
        - q-2 values (level_2) predict q+2, q+6, q+10, etc. (horizons 2, 6, 10, ...)
        - q-1 values (level_1) predict q+3, q+7, q+11, etc. (horizons 3, 7, 11, ...)
        - q values (level_0) predict q+4, q+8, q+12, etc. (horizons 4, 8, 12, ...)

        Creates separate prediction requests for each target, lag level, and lead.

        Args:
            tabular_df: Input dataframe with feature columns like {target}_level_{lag}
            split_name: Name of the split (train/val/test) for logging
        """
        forecast_horizon = self.data_config.get('feature_engineering', {}).get('forecast_horizon', 20)

        # Collect all test rows for all targets at once
        all_test_rows = []

        print(f"Processing {len(self.target_variables)} targets for Naive {split_name} data with cyclical pattern...")

        # For each target variable
        for target_idx, target in enumerate(self.target_variables):
            # Check which level columns are available (level_0, level_1, level_2, level_3)
            available_level_cols = []
            for lag in range(4):  # Check level_0 through level_3
                level_col = f'{target}_level_{lag}'
                if level_col in tabular_df.columns:
                    available_level_cols.append((lag, level_col))

            if not available_level_cols:
                print(f"Warning: No level columns found for target {target}. Skipping.")
                continue

            # Get YoY columns for drift calculation
            yoy_cols = [f'{target}_yoy_{i}' for i in range(4)]
            available_yoy_cols = [col for col in yoy_cols if col in tabular_df.columns]

            target_df = tabular_df.copy()

            if len(target_df) == 0:
                print(f"Warning: No data for target {target}. Skipping.")
                continue

            # Vectorized computation of average growth rates for drift
            # This will be NaN if no YoY values are available for a row
            if available_yoy_cols:
                yoy_data = target_df[available_yoy_cols]
                avg_growth_rates = yoy_data.mean(axis=1, skipna=True)
            else:
                print(f"Warning: No YoY columns for {target}, drift will be zero")
                avg_growth_rates = pd.Series(0.0, index=target_df.index)

            # Create expanded data for cyclical prediction pattern
            expanded_rows = []

            # For each available level (lag)
            for lag, level_col in available_level_cols:
                # Calculate which forecast horizons this lag should predict
                # lag 0 (q) -> horizons 4, 8, 12, 16, 20...
                # lag 1 (q-1) -> horizons 3, 7, 11, 15, 19...
                # lag 2 (q-2) -> horizons 2, 6, 10, 14, 18...
                # lag 3 (q-3) -> horizons 1, 5, 9, 13, 17...

                # Starting horizon for this lag
                start_horizon = 4 - lag

                # Generate horizons for this lag
                lag_horizons = []
                horizon = start_horizon
                while horizon <= forecast_horizon:
                    lag_horizons.append(horizon)
                    horizon += 4

                if not lag_horizons:
                    continue

                print(f"  {target} level_{lag} -> horizons {lag_horizons}")

                # Create base data for this target-lag combination
                base_data = pd.DataFrame({
                    'firm_id': target_df['firm_id'],
                    'target': target,
                    'quarter': target_df['quarter'],
                    'lag_level': lag,
                    'current_value': target_df[level_col],
                    'avg_growth_rate': avg_growth_rates
                })

                # Create expanded data for all forecast horizons for this lag
                for lead in lag_horizons:
                    # Look for the actual target with lead suffix
                    target_col = f'{target}_t{lead}'

                    if target_col in target_df.columns:
                        lead_data = base_data.copy()
                        lead_data['forecast_horizon'] = lead
                        lead_data['actual_value'] = target_df[target_col].values
                        expanded_rows.append(lead_data)
                    else:
                        # If target column doesn't exist for this horizon, skip it
                        print(f"    Skipping horizon {lead} for {target} - no target data available")
                        continue

            # Concatenate all lags and leads for this target
            if expanded_rows:
                target_test_data = pd.concat(expanded_rows, ignore_index=True)
                all_test_rows.append(target_test_data)

        if not all_test_rows:
            print("Warning: No valid test data created for any target.")
            return pd.DataFrame()

        # Concatenate all targets
        print("Combining data for all targets...")
        final_df = pd.concat(all_test_rows, ignore_index=True)

        print(f"Created {len(final_df):,} {split_name} prediction requests with cyclical pattern")

        return final_df

    def get_train_data(self) -> pd.DataFrame:
        """Get training data for Naive Drift model."""
        return self.train_data

    def get_val_data(self) -> Optional[pd.DataFrame]:
        """Get validation data (not used for Naive Drift)."""
        return self.val_data

    def get_test_data(self) -> pd.DataFrame:
        """Get test data for Naive Drift model."""
        return self.test_data


