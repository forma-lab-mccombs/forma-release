"""
Chained Gradient Boosting Machine (GBM) model following Geertsema, Lu, and Ma (2026).

Forecasts financial statement items sequentially in a 14-step chain,
conditioning downstream items on upstream predictions. Uses teacher forcing
during training and cascade during prediction.

Naming convention (repo-wide): "GBM" refers to THIS model -- the chained
gradient-boosting estimator (ChainedGBMModel / model_type ``chained_gbm``).
"GLM" -- the authors' initials, Geertsema/Lu/Ma -- refers to the paper-side
artifacts: the ``glm`` feature set and chain-target set, ``configs/glm.yaml``,
and the benchmark itself. The model is never called "GLM"; the dataset,
feature set, and benchmark are never called "GBM".
"""

from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from .base import BaseModel


DEFAULT_CHAIN_ORDER: List[List[str]] = [
    ['revtq'],                                  # S1:  Revenue
    ['cogsq', 'rectq'],                         # S2:  COGS, Receivables
    ['invtq', 'apq'],                           # S3:  Inventory, Payables
    ['xsgaq'],                                  # S4:  SG&A
    ['acoq', 'lcoq'],                           # S5:  Other current A/L
    ['ppentq', 'intanq'],                       # S6:  PP&E, Intangibles
    ['dpq'],                                    # S7:  Depreciation
    ['dlcq', 'dlttq', 'loq'],                   # S8:  Debt, Other LT liab
    ['xintq'],                                  # S9:  Interest expense
    ['aoq', 'neiq', 'mibtq'],                   # S10: Other LT assets, equity iss, NCI
    ['nopiq', 'spiq', 'xidoq'],                 # S11: Non-op, special, extraordinary
    ['txtq', 'txpq', 'txditcq'],                # S12: Tax items
    ['miiq', 'dvpq', 'dvcq'],                   # S13: Minority interest, preferred/common divs
    ['acomincq'],                               # S14: OCI
]


_WORKER_DATA: Dict[str, Any] = {}


def _init_fit_worker(X_train_base, X_val_base, train_targets, val_targets, available_targets):
    """ProcessPoolExecutor initializer: stash the large shared inputs in a
    module-level global so they are pickled once per worker at pool startup,
    not once per horizon task. Without this, the full ~N x 421 float32 train and
    val matrices (plus the target dict) are re-serialized for every one of the
    20 horizon tasks, ballooning peak host RAM."""
    _WORKER_DATA['X_train_base'] = X_train_base
    _WORKER_DATA['X_val_base'] = X_val_base
    _WORKER_DATA['train_targets'] = train_targets
    _WORKER_DATA['val_targets'] = val_targets
    _WORKER_DATA['available_targets'] = available_targets


def _fit_horizon(
    h: int,
    chain_order: List[List[str]],
    objective: str,
    n_estimators: int,
    num_leaves_options: List[int],
    early_stopping_patience: int,
    random_state: int,
    verbose: int,
    n_jobs_per_model: int,
) -> Tuple[int, Dict[str, Any], Dict[str, Dict], Dict[str, int], Dict[str, int]]:
    """
    Fit all chain steps for a single horizon. The large shared inputs (feature
    matrices and target dicts) are read from the per-worker ``_WORKER_DATA``
    global populated by ``_init_fit_worker``, so they are not re-pickled per task.

    Returns:
        (horizon, models_dict, best_params_dict, best_n_estimators_dict,
         train_rows_dict) -- train_rows_dict maps item -> surviving train rows
         (diagnostic for upstream-NaN attrition in late chain steps).
    """
    X_train_base = _WORKER_DATA['X_train_base']
    X_val_base = _WORKER_DATA['X_val_base']
    train_targets = _WORKER_DATA['train_targets']
    val_targets = _WORKER_DATA['val_targets']
    available_targets = _WORKER_DATA['available_targets']

    models: Dict[str, Any] = {}
    best_params: Dict[str, Dict] = {}
    best_n_est_out: Dict[str, int] = {}
    train_rows_out: Dict[str, int] = {}

    # Early-stopping / model-selection metric follows the objective so an L2
    # objective is not silently ranked on L1. (best_score_ is keyed by metric
    # name, so this key must match eval_metric.) Defaults to L1, matching the
    # GLM-paper MAE objective.
    eval_metric = "l2" if objective in ("regression", "regression_l2", "mse", "l2") else "l1"

    upstream_actuals_train: Dict[str, np.ndarray] = {}
    upstream_actuals_val: Dict[str, np.ndarray] = {}

    def _build_aug(X_base, upstream_vals, step_idx):
        upstream_items = []
        for s in range(step_idx):
            upstream_items.extend(chain_order[s])
        if not upstream_items:
            return X_base
        chain_cols = {}
        for item in upstream_items:
            if item in upstream_vals:
                chain_cols[f"chain_{item}"] = upstream_vals[item]
        if not chain_cols:
            return X_base
        # Fill missing upstream actuals with 0 (asinh(0)=0, i.e. an unreported item
        # treated as zero) so a NaN upstream value does NOT drop the row from the
        # downstream training set. Items like spiq, xidoq, miiq, mibtq and acomincq
        # are frequently NaN in quarterly Compustat; without this, late chain steps
        # would train only on rows where every upstream actual is simultaneously
        # present -- a tiny, heavily-selected subsample -- and would never see the
        # "missing upstream" pattern that the predict-time cascade can still produce.
        chain_df = pd.DataFrame(chain_cols, index=X_base.index, dtype="float32").fillna(0.0)
        return pd.concat([X_base, chain_df], axis=1)

    for step_idx, step_items in enumerate(chain_order):
        X_train_aug = _build_aug(X_train_base, upstream_actuals_train, step_idx)
        X_val_aug = (
            _build_aug(X_val_base, upstream_actuals_val, step_idx)
            if X_val_base is not None else None
        )

        for item in step_items:
            target_col = f"{item}_t{h}"
            if target_col not in available_targets:
                continue

            y_train = pd.Series(train_targets[target_col], index=X_train_base.index)
            y_val = (
                pd.Series(val_targets[target_col], index=X_val_base.index)
                if X_val_base is not None and target_col in val_targets
                else None
            )

            # Chain columns are NaN-filled in _build_aug, so this mask now drops
            # rows only for missing base features or a missing target -- not for a
            # missing upstream actual.
            train_mask = ~(X_train_aug.isna().any(axis=1) | y_train.isna())
            X_tr = X_train_aug[train_mask]
            y_tr = y_train[train_mask]
            train_rows_out[item] = int(len(X_tr))

            if X_val_aug is not None and y_val is not None:
                val_mask = ~(X_val_aug.isna().any(axis=1) | y_val.isna())
                X_v = X_val_aug[val_mask]
                y_v = y_val[val_mask]
            else:
                X_v, y_v = None, None

            if len(X_tr) == 0:
                continue

            best_val_score = np.inf
            best_nl = num_leaves_options[0]
            best_n_est = n_estimators

            for nl in num_leaves_options:
                model = lgb.LGBMRegressor(
                    objective=objective,
                    num_leaves=nl,
                    n_estimators=n_estimators,
                    random_state=random_state,
                    verbose=verbose,
                    n_jobs=n_jobs_per_model,
                )
                if X_v is not None and len(X_v) > 0:
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_v, y_v)],
                        eval_metric=eval_metric,
                        callbacks=[
                            lgb.early_stopping(early_stopping_patience, verbose=False),
                            lgb.log_evaluation(period=0),
                        ],
                    )
                    val_score = model.best_score_["valid_0"][eval_metric]
                    n_est = model.best_iteration_
                else:
                    model.fit(X_tr, y_tr)
                    val_score = np.inf
                    n_est = n_estimators

                if val_score < best_val_score:
                    best_val_score = val_score
                    best_nl = nl
                    best_n_est = n_est

            if X_v is not None and len(X_v) > 0:
                X_combined = pd.concat([X_tr, X_v], axis=0)
                y_combined = pd.concat([y_tr, y_v], axis=0)
            else:
                X_combined = X_tr
                y_combined = y_tr

            final_model = lgb.LGBMRegressor(
                objective=objective,
                num_leaves=best_nl,
                n_estimators=best_n_est,
                random_state=random_state,
                verbose=verbose,
                n_jobs=n_jobs_per_model,
            )
            final_model.fit(X_combined, y_combined)

            models[item] = final_model
            best_params[item] = {"num_leaves": best_nl}
            best_n_est_out[item] = best_n_est

        # Expose this step's items as upstream chain inputs for later steps --
        # but ONLY for items that actually got a fitted model. An item skipped
        # for lack of training rows has no model, so the predict-time cascade
        # never produces its chain_<item> column; adding it to the teacher-forcing
        # actuals here would train downstream models on a feature that is then
        # absent at inference, triggering a LightGBM feature-name mismatch.
        for item in step_items:
            if item not in models:
                continue
            target_col = f"{item}_t{h}"
            if target_col in train_targets:
                upstream_actuals_train[item] = train_targets[target_col]
            if target_col in val_targets:
                upstream_actuals_val[item] = val_targets[target_col]

    return h, models, best_params, best_n_est_out, train_rows_out


class ChainedGBMModel(BaseModel):
    """
    Chained LightGBM model for financial statement forecasting.

    Trains a chain of LightGBM models per forecast horizon, where each
    downstream item is conditioned on upstream predictions. During training
    uses teacher forcing (actual upstream values); during prediction uses
    the cascade (predicted upstream values).
    """

    def __init__(
        self,
        model_name: str = "chained_gbm",
        chain_order: Optional[List[List[str]]] = None,
        hyperparameter_grid: Optional[Dict] = None,
        early_stopping_patience: int = 10,
        n_estimators: int = 500,
        objective: str = "regression_l1",
        verbose: int = -1,
        random_state: int = 42,
        max_workers: Optional[int] = None,
        **kwargs,
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is required for ChainedGBMModel. Install with: pip install lightgbm")

        super().__init__(model_name, **kwargs)
        self.chain_order = chain_order or DEFAULT_CHAIN_ORDER
        self.core_items: List[str] = [item for step in self.chain_order for item in step]
        self.hyperparameter_grid = hyperparameter_grid or {"num_leaves": [15, 31]}
        self.early_stopping_patience = early_stopping_patience
        self.n_estimators = n_estimators
        self.objective = objective
        self.verbose = verbose
        self.random_state = random_state
        # Cap on horizon-parallel worker processes. Each worker holds its own copy
        # of the train+val feature matrices, so peak host RAM ~ n_workers x matrix
        # size; cap this on memory-constrained hosts. None -> min(n_horizons, cpu//2).
        self.max_workers = max_workers

        self.models: Dict[int, Dict[str, Any]] = {}
        self.best_params: Dict[int, Dict[str, Dict]] = {}
        self.best_n_estimators: Dict[int, Dict[str, int]] = {}
        self.train_rows_: Dict[int, Dict[str, int]] = {}
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.is_fitted = False

    def _get_upstream_items(self, step_idx: int) -> List[str]:
        """Return flat list of all items from chain steps before step_idx."""
        upstream = []
        for s in range(step_idx):
            upstream.extend(self.chain_order[s])
        return upstream

    def _build_augmented_features(
        self,
        X_base: pd.DataFrame,
        upstream_values: Dict[str, np.ndarray],
        step_idx: int,
    ) -> pd.DataFrame:
        """
        Augment base features with upstream item values for a given chain step.

        Args:
            X_base: Base feature DataFrame (level/yoy columns).
            upstream_values: Dict mapping item name -> array of values
                (actuals during training, predictions during test).
            step_idx: Current chain step index.

        Returns:
            DataFrame with additional chain_* columns appended.
        """
        upstream_items = self._get_upstream_items(step_idx)
        if not upstream_items:
            return X_base

        chain_cols = {}
        for item in upstream_items:
            if item in upstream_values:
                chain_cols[f"chain_{item}"] = upstream_values[item]

        if not chain_cols:
            return X_base

        # Match the training-time fill (see _build_aug): cascade predictions are
        # normally non-NaN, but fill defensively so train/predict stay consistent.
        chain_df = pd.DataFrame(chain_cols, index=X_base.index, dtype="float32").fillna(0.0)
        return pd.concat([X_base, chain_df], axis=1)

    def fit(self, train_data, validation_data=None) -> None:
        """
        Train the chained GBM model.

        Horizons are trained in parallel via ProcessPoolExecutor (each
        horizon's 14-step chain is independent). Within a horizon, steps
        run sequentially to honour the chaining dependency.

        Args:
            train_data: TabularDataModule or DataFrame.
            validation_data: Optional separate validation DataFrame.
        """
        if hasattr(train_data, 'get_train_data'):
            train_df = train_data.get_train_data()
            val_df = train_data.get_val_data() if validation_data is None else validation_data
            feature_cols = getattr(train_data, 'feature_columns', [])
            target_cols = getattr(train_data, 'target_columns', [])
            if feature_cols and target_cols:
                self.feature_columns = feature_cols
                self.target_columns = target_cols
            else:
                self.target_columns = [c for c in train_df.columns if '_t' in c and c.split('_t')[-1].isdigit()]
                self.feature_columns = [c for c in train_df.columns
                                        if c not in self.target_columns
                                        and c not in ('firm_id', 'quarter')]
        else:
            train_df = train_data
            val_df = validation_data
            self.target_columns = [c for c in train_df.columns if '_t' in c and c.split('_t')[-1].isdigit()]
            self.feature_columns = [c for c in train_df.columns
                                    if c not in self.target_columns
                                    and c not in ('firm_id', 'quarter')]

        horizons = sorted({int(c.rsplit('_t', 1)[1]) for c in self.target_columns})
        available_targets = set(self.target_columns)

        # Guard against chain_order / target-list drift. Every core chain item
        # must appear as a target column (for at least one horizon); a chain item
        # absent from the resolved targets would otherwise be silently skipped --
        # its forecast never produced and its teacher-forcing signal missing from
        # downstream steps -- rather than failing loudly. This checks column
        # *presence*, not row coverage: an all-NaN target like acomincq still has
        # columns and passes here (it is handled by the per-item zero-train-row
        # skip in _fit_horizon).
        target_bases = {c.rsplit('_t', 1)[0] for c in self.target_columns}
        missing_chain_items = [item for item in self.core_items if item not in target_bases]
        if missing_chain_items:
            raise ValueError(
                f"{self.model_name}: chain_order items missing from the resolved "
                f"target set: {missing_chain_items}. The model's chain_order and "
                f"the config's targets have drifted -- every chain item must have a "
                f"corresponding target column. Reconcile chain_order in "
                f"chained_gbm.py with targets.variables in the config."
            )

        X_train_base = train_df[self.feature_columns].astype("float32")
        X_val_base = val_df[self.feature_columns].astype("float32") if val_df is not None else None

        total_items = sum(
            1 for h in horizons
            for step in self.chain_order
            for item in step
            if f"{item}_t{h}" in available_targets
        )

        num_leaves_options = self.hyperparameter_grid.get("num_leaves", [31])

        cpu_count = os.cpu_count() or 2  # os.cpu_count() can return None
        n_workers = min(len(horizons), max(1, cpu_count // 2))
        if self.max_workers is not None:
            n_workers = max(1, min(n_workers, self.max_workers))
        n_jobs_per_model = max(1, cpu_count // n_workers)

        print(f"\n{self.model_name}: Training {total_items} models "
              f"({len(self.core_items)} items x {len(horizons)} horizons) "
              f"using {n_workers} parallel workers, "
              f"{n_jobs_per_model} threads each")
        print(f"  Note: each worker holds its own copy of the train+val feature "
              f"matrices (~{X_train_base.shape[0]:,} x {X_train_base.shape[1]} "
              f"float32) plus the train+val target dicts ({len(self.target_columns)} "
              f"columns' values); peak host RAM scales with n_workers.")

        train_targets = {
            col: train_df[col].values for col in self.target_columns
        }
        val_targets = {}
        if val_df is not None:
            val_targets = {
                col: val_df[col].values
                for col in self.target_columns
                if col in val_df.columns
            }

        pbar = tqdm(total=len(horizons), desc=f"{self.model_name}", unit="horizon")

        # Pass the large feature matrices / target dicts once per worker via the
        # pool initializer (serialized n_workers times) rather than once per
        # horizon task (n_horizons times).
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_fit_worker,
            initargs=(X_train_base, X_val_base, train_targets, val_targets, available_targets),
        ) as executor:
            futures = {
                executor.submit(
                    _fit_horizon,
                    h,
                    self.chain_order,
                    self.objective,
                    self.n_estimators,
                    num_leaves_options,
                    self.early_stopping_patience,
                    self.random_state,
                    self.verbose,
                    n_jobs_per_model,
                ): h
                for h in horizons
            }

            for future in as_completed(futures):
                h, h_models, h_params, h_n_est, h_rows = future.result()
                self.models[h] = h_models
                self.best_params[h] = h_params
                self.best_n_estimators[h] = h_n_est
                self.train_rows_[h] = h_rows
                pbar.update(1)

        pbar.close()
        self.is_fitted = True

        row_counts = [n for hr in self.train_rows_.values() for n in hr.values()]
        if row_counts:
            print(f"  Surviving train rows per (item, horizon): "
                  f"min={min(row_counts)}, median={int(np.median(row_counts))}, "
                  f"max={max(row_counts)} (of {len(X_train_base):,} train rows; a "
                  f"low min flags heavy upstream-NaN attrition in late chain steps)")
        print(f"{self.model_name}: Training completed — "
              f"{sum(len(v) for v in self.models.values())} models fitted")

    def predict(self, test_data, stream_save_paths=None, stream_group_size=4) -> pd.DataFrame:
        """
        Generate predictions using the cascade (no teacher forcing).

        For each horizon, runs the chain sequentially, feeding predicted
        upstream values into downstream models.

        Args:
            test_data: TabularDataModule or DataFrame.

        Returns:
            DataFrame in tall format: firm_id, target, quarter,
            forecast_horizon, prediction, model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if hasattr(test_data, 'get_test_data'):
            test_df = test_data.get_test_data()
        else:
            test_df = test_data

        X_test_base = test_df[self.feature_columns].astype("float32")
        valid_mask = ~X_test_base.isna().any(axis=1)
        X_test_base = X_test_base[valid_mask]
        test_df_clean = test_df.loc[valid_mask]

        n_orig = len(test_df)
        n_clean = len(test_df_clean)
        print(f"\n{self.model_name}: Predicting on {n_clean}/{n_orig} test samples")

        if n_clean == 0:
            raise ValueError("No valid test samples after dropping NaN")

        horizons = sorted(self.models.keys())
        predictions_dict: Dict[str, np.ndarray] = {}

        for h in horizons:
            cascade_preds: Dict[str, np.ndarray] = {}

            for step_idx, step_items in enumerate(self.chain_order):
                X_test_aug = self._build_augmented_features(
                    X_test_base, cascade_preds, step_idx
                )

                for item in step_items:
                    if item not in self.models.get(h, {}):
                        continue

                    model = self.models[h][item]
                    preds = model.predict(X_test_aug)
                    cascade_preds[item] = preds
                    predictions_dict[f"{item}_t{h}"] = preds

        forecast_wide = pd.DataFrame(predictions_dict, index=X_test_base.index)
        forecast_wide["firm_id"] = test_df_clean["firm_id"].values
        forecast_wide["quarter"] = test_df_clean["quarter"].values
        forecast_wide["model"] = self.model_name

        pred_cols = [c for c in forecast_wide.columns if c not in ("firm_id", "quarter", "model")]

        # wide->tall via the shared helper (identical melt + `_t` split that was
        # previously inlined here). In streaming mode (train.py:1171 always passes
        # stream_save_paths -- per-exp path + global "latest") write the tall
        # forecast in column groups so peak memory stays bounded by ~one group
        # rather than melting the whole wide frame at once. Mirrors
        # MultiHorizonBaselineModel / FeedForwardModel.
        from .baselines import wide_forecasts_to_tall
        if stream_save_paths:
            import pyarrow as pa
            from .forecast_io import finalize_forecast_frame, atomic_parquet_writers
            # group_cols is a pure memory-sizing chunk width, NOT a target-aligned
            # grouping: unlike baselines/ffnn (target-major columns), pred_cols here
            # is built HORIZON-major (`{item}_t{h}` generated in an h-outer loop), so
            # a slice is an arbitrary mix of (item, horizon), not "N targets x 20
            # horizons". That's fine -- each column melts independently, so any
            # chunking yields the same row-set. The `*20` is just a reasonable width;
            # don't "fix" it to align with targets.
            group_cols = max(1, stream_group_size) * 20
            with atomic_parquet_writers(stream_save_paths) as write_table:
                for start in range(0, len(pred_cols), group_cols):
                    group = pred_cols[start:start + group_cols]
                    tall = finalize_forecast_frame(wide_forecasts_to_tall(forecast_wide, group))
                    tbl = pa.Table.from_pandas(tall, preserve_index=False)
                    write_table(tbl)
                    del tall, tbl
                    print(f"  {self.model_name}: streamed cols {start + 1}-{start + len(group)}/{len(pred_cols)}")
            return None

        return wide_forecasts_to_tall(forecast_wide, pred_cols)

    def save(self, folder_path: str) -> None:
        """Save all fitted LightGBM models and metadata."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        os.makedirs(folder_path, exist_ok=True)

        for h, item_models in self.models.items():
            for item, model in item_models.items():
                model_path = os.path.join(folder_path, f"{self.model_name}_{item}_t{h}.txt")
                model.booster_.save_model(model_path)

        metadata = {
            "model_name": self.model_name,
            "chain_order": self.chain_order,
            "core_items": self.core_items,
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "best_params": {
                str(h): {item: params for item, params in items.items()}
                for h, items in self.best_params.items()
            },
            "best_n_estimators": {
                str(h): {item: n for item, n in items.items()}
                for h, items in self.best_n_estimators.items()
            },
            "horizons": sorted(self.models.keys()),
            "hyperparameter_grid": self.hyperparameter_grid,
            "objective": self.objective,
            "early_stopping_patience": self.early_stopping_patience,
        }
        meta_path = os.path.join(folder_path, f"{self.model_name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"{self.model_name}: Saved {sum(len(v) for v in self.models.values())} models to {folder_path}")

    def load(self, folder_path: str) -> None:
        """Load previously saved LightGBM models and metadata.

        Note: loaded models are raw ``lgb.Booster`` objects (from the saved
        ``.txt`` boosters), not the ``lgb.LGBMRegressor`` instances held after
        ``fit()``. ``.predict()`` is identical on both, which is all predict()
        relies on -- but code reaching for sklearn-wrapper attributes (e.g.
        ``.booster_``, ``.best_iteration_``) must not assume a fitted-vs-loaded
        model are the same type.
        """
        meta_path = os.path.join(folder_path, f"{self.model_name}_metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        self.chain_order = metadata["chain_order"]
        self.core_items = metadata["core_items"]
        self.feature_columns = metadata["feature_columns"]
        self.target_columns = metadata["target_columns"]
        self.hyperparameter_grid = metadata.get("hyperparameter_grid", {})
        self.objective = metadata.get("objective", "regression_l1")

        self.best_params = {
            int(h): items for h, items in metadata["best_params"].items()
        }
        self.best_n_estimators = {
            int(h): {item: n for item, n in items.items()}
            for h, items in metadata.get("best_n_estimators", {}).items()
        }

        self.models = {}
        for h in metadata["horizons"]:
            self.models[h] = {}
            for step in self.chain_order:
                for item in step:
                    model_path = os.path.join(folder_path, f"{self.model_name}_{item}_t{h}.txt")
                    if os.path.exists(model_path):
                        booster = lgb.Booster(model_file=model_path)
                        self.models[h][item] = booster

        self.is_fitted = True
        print(f"{self.model_name}: Loaded {sum(len(v) for v in self.models.values())} models from {folder_path}")
