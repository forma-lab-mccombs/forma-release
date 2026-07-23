"""
Baseline model implementations using sklearn and XGBoost.

This module provides wrapper classes for traditional ML models that conform
to the BaseModel interface for consistent benchmarking.
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import os
import joblib
import sys
import json
import hashlib
from contextlib import contextmanager
from pathlib import Path
from tqdm.auto import tqdm
import inspect

# threadpoolctl ships with scikit-learn; we use it to cap BLAS/LAPACK threads to
# one *per worker* while the outer refit loop is parallel, so linear/ridge arms
# (whose solvers call into multithreaded LAPACK/BLAS) don't oversubscribe cores
# as outer_n_jobs * BLAS_threads. Import defensively so a missing dependency
# degrades to "no limiting" rather than crashing.
try:
    from threadpoolctl import threadpool_limits
    THREADPOOLCTL_AVAILABLE = True
except ImportError:
    THREADPOOLCTL_AVAILABLE = False

    @contextmanager
    def threadpool_limits(*args, **kwargs):
        yield

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

# Try to import cuML for GPU-accelerated Random Forest. Broad `except` because
# the RAPIDS stack can fail in ways beyond ImportError — e.g. a cupy↔cuda-pathfinder
# version mismatch surfaces as AttributeError on cupy's first internal import,
# which would otherwise propagate up and crash train.py at module-load time
# (observed on some GPU cluster stacks). The intent of this try/except
# is to fall back to sklearn cleanly whenever RAPIDS isn't usable, regardless of
# how the import fails. Diagnostic message names the cause so it isn't silent.
try:
    from cuml.ensemble import RandomForestRegressor as cuMLRandomForestRegressor
    CUML_AVAILABLE = True
    print("cuML detected - GPU-accelerated Random Forest available")
except Exception as _cuml_err:
    CUML_AVAILABLE = False
    cuMLRandomForestRegressor = None
    print(f"cuML unavailable, falling back to sklearn RandomForest ({type(_cuml_err).__name__}: {_cuml_err})")

from .base import BaseModel


def wide_forecasts_to_tall(forecast_wide: pd.DataFrame, target_columns: List[str],
                           sigma_wide: pd.DataFrame = None) -> pd.DataFrame:
    """Melt a wide forecast frame into the standardized tall forecast schema.

    Shared by every parameter-shared, vector-output forecaster
    (MultiHorizonBaselineModel and FeedForwardModel) so the wide->tall conversion
    lives in exactly one place.

    Args:
        forecast_wide: DataFrame with one column per target column (e.g. 'niq_t1',
            'niq_t2', ...) holding predictions, plus the metadata columns
            'firm_id', 'quarter', and 'model'.
        target_columns: The value columns to melt (e.g. ['niq_t1', ..., 'ltq_t20']).
        sigma_wide: Optional DataFrame with the SAME row index and the same
            `target_columns`, holding the per-(target, horizon) predictive std
            (likelihood track). When given, the output gains a `sigma` column. Both
            frames melt in identical row order (same index, same value_vars), so the
            columns align positionally.

    Returns:
        Tall DataFrame with columns 'firm_id', 'target', 'quarter',
        'forecast_horizon', 'prediction', 'model' (plus 'sigma' when sigma_wide
        is provided).
    """
    forecast_df = forecast_wide.melt(
        id_vars=['firm_id', 'quarter', 'model'],
        value_vars=target_columns,
        var_name='target_col',
        value_name='prediction',
    )

    # Parse target_col to extract target name and forecast horizon
    # e.g., 'niq_t1' -> target='niq', forecast_horizon=1
    target_horizon_split = forecast_df['target_col'].str.rsplit('_t', n=1, expand=True)
    forecast_df['target'] = target_horizon_split[0]
    forecast_df['forecast_horizon'] = target_horizon_split[1].astype(int)

    out_cols = ['firm_id', 'target', 'quarter', 'forecast_horizon', 'prediction', 'model']
    if sigma_wide is not None:
        sigma_long = sigma_wide.melt(value_vars=target_columns,
                                     var_name='target_col', value_name='sigma')
        if len(sigma_long) != len(forecast_df):
            raise ValueError(
                "sigma_wide melt length does not match prediction melt; sigma_wide "
                "must share forecast_wide's index and target_columns.")
        forecast_df['sigma'] = sigma_long['sigma'].to_numpy()
        out_cols = ['firm_id', 'target', 'quarter', 'forecast_horizon',
                    'prediction', 'sigma', 'model']

    return forecast_df[out_cols]


def _callable_supported_kwargs(callable_obj, kwargs: Dict) -> Dict:
    """Best-effort filter of kwargs to those accepted by callable_obj.

    RAPIDS/cuML estimators can vary across versions; this keeps us compatible.
    If signature introspection fails (e.g., some cython bindings), we return
    kwargs unchanged and rely on retry logic at construction time.
    """
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return dict(kwargs)

    # If callable accepts **kwargs, don't filter.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    allowed = set(params.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _set_cuml_random_forest_seed_param(estimator_cls, params: Dict, seed_value) -> None:
    """Set seed/random_state param on cuML RF depending on what's supported."""
    if seed_value is None:
        return

    supported = None
    try:
        supported = set(inspect.signature(estimator_cls).parameters.keys())
    except (TypeError, ValueError):
        supported = None

    if supported is None:
        # Unknown signature; prefer sklearn-style name, fall back to cuML-style.
        params.setdefault('random_state', seed_value)
    elif 'random_state' in supported:
        params['random_state'] = seed_value
    elif 'seed' in supported:
        params['seed'] = seed_value
    # else: neither supported; do nothing

class MultiHorizonBaselineModel(BaseModel):
    """
    Base class for multi-horizon forecasting models.

    This class implements the common logic for training separate models
    for each forecast horizon with optional hyperparameter tuning.
    Subclasses only need to implement _create_single_model().
    """
    
    def __init__(self, model_name: str, hyperparameter_grid: Dict = None,
                 shared_horizon_hparams: bool = False,
                 hparam_reference_target: str = "niq",
                 outer_n_jobs: int = -1, outer_backend: str = "threading", **kwargs):
        """
        Initialize multi-horizon baseline model.

        Args:
            model_name: Name identifier for this model
            hyperparameter_grid: Grid of hyperparameters for grid search (optional)
            shared_horizon_hparams: If True, select one hyperparameter set per
                forecast horizon h using cross-validation on a single reference
                target, then reuse horizon h's params for that horizon across ALL
                targets (so the grid is searched H times, not |targets|*H times).
                If False (default), search independently per (target, horizon).
            hparam_reference_target: Base target variable (e.g. "niq") whose
                horizons drive the shared hyperparameter search. Only used when
                shared_horizon_hparams is True.
            outer_n_jobs: Parallelism for the per-(target, horizon) refit loop.
                -1 uses all cores (default), 1 disables outer parallelism. Only
                takes effect for the linear family (see _supports_outer_parallel);
                RandomForest / GPU / multithreaded estimators stay serial to avoid
                oversubscription. When the outer loop runs in parallel, any inner
                GridSearchCV is forced to n_jobs=1 and BLAS/LAPACK is capped to one
                thread per worker (via threadpoolctl) so no level oversubscribes.
            outer_backend: joblib backend for the outer loop ("threading" or
                "loky"). "threading" (default) avoids copying the feature matrices
                per worker; the linear-family solvers release the GIL during their
                (BLAS-capped) numeric work, so threads still run concurrently. Use
                "loky" (process-based) if a future estimator does not release it.
            **kwargs: Model-specific parameters (used if no grid search)
        """
        super().__init__(model_name, **kwargs)
        self.hyperparameter_grid = hyperparameter_grid
        self.shared_horizon_hparams = shared_horizon_hparams
        self.hparam_reference_target = hparam_reference_target
        self.outer_n_jobs = outer_n_jobs
        self.outer_backend = outer_backend
        self.models_by_horizon = {}  # Store one model per horizon
        self.best_params = {}  # Store best hyperparameters from validation
        self.params_by_horizon = {}  # Shared params keyed by horizon h (shared mode)
        self.target_columns = []  # Will be populated during fit
        self.feature_columns = []  # Will be populated during fit

        # --- Incremental / resumable persistence ---------------------------------
        # When set (by train.py, before fit), each (target, horizon) model is
        # dumped to disk and freed as soon as it is trained, instead of all being
        # accumulated in self.models_by_horizon. This keeps peak memory bounded by
        # ONE model — essential for the 78-target x 20-horizon Random Forest sweep
        # (1560 models, ~30-40h) that previously OOM'd because every forest stayed
        # resident and nothing was flushed until the very end. It also makes the
        # run resumable: a re-launch skips any (target, horizon) already on disk.
        # When None (tests / direct DataFrame use), behaviour is unchanged: models
        # are kept in memory and only persisted by an explicit save().
        self.incremental_dir = None
        # Directory a lazy load() reads models from (set by load()). predict()/save()
        # fall back to loading one model at a time from here when not in memory.
        self._model_dir = None
        # Hash of the fit-defining config (grid, params, feature/target columns),
        # stamped into each sidecar so a resume that points --resume-dir at a dir
        # trained with a different grid/dataset re-trains instead of silently
        # reusing stale models. Computed in fit() once columns are known.
        self._fit_signature_hash = None

    # --- Incremental persistence helpers -------------------------------------
    def _active_model_dir(self) -> Optional[str]:
        """Directory to lazily load per-horizon models from when they are not
        held in memory: the incremental-save dir during/after a streaming fit, or
        the folder a lazy load() was pointed at."""
        return self.incremental_dir or self._model_dir

    def _model_pkl_path(self, folder: str, target_col: str) -> str:
        """On-disk path for one (target, horizon) model. Matches the naming used
        by save()/load() so a streaming fit writes the exact files save() expects."""
        return os.path.join(folder, f"{self.model_name}_{target_col}.pkl")

    def _params_sidecar_path(self, folder: str, target_col: str) -> str:
        """Path of the small JSON sidecar for one model. It holds the chosen
        params, the fit-signature hash, and the pre-computed metadata/coef-summary
        record. Its presence (alongside the .pkl) is the per-target "done" marker
        used to skip already-trained pairs on resume, and it lets save() rebuild
        the full metadata without unpickling a single model."""
        return os.path.join(folder, f"{self.model_name}_{target_col}.params.json")

    def _fit_signature(self) -> str:
        """Short hash of everything that defines what a fitted model *is*: the
        grid, base params, the shared-hparam mode, and the feature/target columns
        (which capture the design-matrix schema / dataset). Two runs with the same
        signature are interchangeable; a mismatch means on-disk models are stale."""
        sig = {
            'model_name': self.model_name,
            'hyperparameter_grid': self.hyperparameter_grid,
            'hyperparameters': self.hyperparameters,
            'shared_horizon_hparams': self.shared_horizon_hparams,
            'hparam_reference_target': self.hparam_reference_target,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
        }
        blob = json.dumps(sig, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _read_sidecar(self, folder: str, target_col: str) -> Optional[Dict]:
        """Load a model's sidecar dict, or None if absent/unreadable."""
        path = self._params_sidecar_path(folder, target_col)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _completed_pair(self, folder: str, target_col: str, warn: bool = False) -> Optional[Dict]:
        """Return the sidecar dict for a (target, horizon) that is fully persisted
        AND matches the current fit signature, else None (so the pair re-trains).

        Returns None when:
        - the pkl is missing (never trained), or
        - the sidecar is missing/corrupt — e.g. a crash *between* the pkl write and
          the sidecar write; the both-required rule makes that pair re-train rather
          than resume from a model whose params were never recorded, or
        - the stored fit-signature hash differs from this run's (different grid /
          features / dataset) — resuming those would silently reuse stale models.
        """
        if not os.path.exists(self._model_pkl_path(folder, target_col)):
            return None
        sc = self._read_sidecar(folder, target_col)
        if sc is None:
            return None
        stored = sc.get('config_hash')
        if (stored is not None and self._fit_signature_hash is not None
                and stored != self._fit_signature_hash):
            if warn:
                tqdm.write(f"{self.model_name} - {target_col}: on-disk model was trained with a "
                           f"different config/grid/features (hash {stored} != "
                           f"{self._fit_signature_hash}); re-training this pair.")
            return None
        return sc

    @staticmethod
    def _atomic_dump(dump_fn, path: str) -> None:
        """Write via a temp file + os.replace so a crash mid-write never leaves a
        half-written file that a later resume would mistake for complete. The
        temp name is pid-tagged so concurrent (outer-parallel) writers don't clash."""
        tmp = f"{path}.tmp.{os.getpid()}"
        dump_fn(tmp)
        os.replace(tmp, path)

    @staticmethod
    def _write_json(obj, path: str) -> None:
        """Write JSON and close the handle before returning, so the subsequent
        os.replace in _atomic_dump never races an open handle (Windows would
        otherwise refuse to replace a still-open file)."""
        with open(path, 'w') as f:
            json.dump(obj, f)

    def _persist_one(self, folder: str, target_col: str, model, best_params) -> None:
        """Atomically write a single trained model + its sidecar. The pkl is
        written first, then the sidecar, so a crash between the two leaves a pkl
        with no sidecar — which resume treats as not-done and re-trains (cheap,
        idempotent). The sidecar also caches the per-model metadata/coef-summary
        record computed here while the model is in memory, so save() never has to
        re-load the forest just to extract feature_importances/coefficients."""
        os.makedirs(folder, exist_ok=True)
        sidecar = {
            'best_params': best_params,
            'config_hash': self._fit_signature_hash,
            'model_param_dict': self._model_param_dict(model, target_col, best_params),
            'coef_summary_row': self._coef_summary_row(model, target_col, best_params),
        }
        self._atomic_dump(lambda p: joblib.dump(model, p),
                          self._model_pkl_path(folder, target_col))
        self._atomic_dump(lambda p: self._write_json(sidecar, p),
                          self._params_sidecar_path(folder, target_col))

    def _completed_targets(self, folder: str) -> set:
        """Targets already fully persisted in `folder` and matching this run's fit
        signature (so the resume count reflects what will actually be skipped)."""
        return {tc for tc in self.target_columns
                if self._completed_pair(folder, tc) is not None}

    def _get_model(self, target_col: str):
        """Return (model, loaded_from_disk) for one horizon.

        Uses the in-memory model when present (non-incremental path); otherwise
        loads it on demand from the active model dir. Callers free the model after
        use when loaded_from_disk is True, so only one model is resident at a time.
        """
        model = self.models_by_horizon.get(target_col)
        if model is not None:
            return model, False
        folder = self._active_model_dir()
        if folder is None:
            raise KeyError(
                f"No in-memory model for '{target_col}' and no model directory to "
                f"load it from. Was the model fitted or loaded?")
        path = self._model_pkl_path(folder, target_col)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path), True

    def _uses_inner_jobs(self) -> bool:
        """
        Returns True if the model's internal training uses parallelization (e.g., n_jobs).

        Subclasses should override this if their models use internal parallelization
        (like RandomForest with n_jobs or cuML GPU models). When True, GridSearchCV
        will use n_jobs=1 to avoid nested parallelization.

        Returns:
            bool: True if model uses internal parallelization, False otherwise.
        """
        return False

    def _supports_outer_parallel(self) -> bool:
        """Whether the outer (target, horizon) refit loop may be parallelized.

        Only safe for estimators whose single .fit() is effectively
        single-threaded; parallelizing the loop for internally-parallel
        estimators (RandomForest, GPU models, multithreaded XGBoost) would
        oversubscribe cores/GPU. Subclasses override to opt specific model types
        in. Defaults to False so the base behaviour is unchanged.
        """
        return False

    def _effective_outer_n_jobs(self) -> int:
        """Resolve the outer-loop job count, gated by estimator eligibility.

        Returns 1 (serial) when outer parallelism is disabled (outer_n_jobs == 1)
        or unsafe for this estimator; otherwise returns the configured
        outer_n_jobs.
        """
        if self.outer_n_jobs == 1:
            return 1
        if not self._supports_outer_parallel():
            return 1
        return self.outer_n_jobs

    def _create_single_model(self, **params):
        """
        Create a single model instance with given parameters.

        Must be implemented by subclasses.

        Args:
            **params: Model parameters

        Returns:
            Model instance
        """
        raise NotImplementedError("Subclasses must implement _create_single_model()")

    @staticmethod
    def _extract_target_columns(data_df: pd.DataFrame, target_variables: List[str] = None) -> List[str]:
        """
        Extract target columns from DataFrame.

        Args:
            data_df: DataFrame with features and target columns
            target_variables: List of target variable names (e.g., ['niq', 'fcfq'])

        Returns:
            List of target column names (e.g., ['niq_t1', 'niq_t2', ...])
        """
        if target_variables is None:
            # Auto-detect: find all columns ending with _t{digit}
            target_cols = [c for c in data_df.columns if '_t' in c and c.split('_t')[-1].replace('-', '').isdigit()]
        else:
            # Find all columns for specified target variables
            target_cols = []
            for target_var in target_variables:
                target_cols.extend([c for c in data_df.columns if c.startswith(f"{target_var}_t")])

        return sorted(target_cols)

    @staticmethod
    def _parse_horizon(target_col: str) -> int:
        """Extract the integer forecast horizon h from a column like 'niq_t12' -> 12."""
        return int(target_col.rsplit('_t', 1)[1])

    @staticmethod
    def _parse_base_var(target_col: str) -> str:
        """Extract the base target variable from a column like 'niq_t12' -> 'niq'."""
        return target_col.rsplit('_t', 1)[0]

    def _prepare_xy(self, target_col, X_train_base, X_val_base, train_df, val_df):
        """Slice features/target for one horizon and drop rows with NaN in X or y.

        Returns (X_train, y_train, X_val, y_val); X_val/y_val are None when no
        validation frame is available.
        """
        y_train_base = train_df[target_col]
        y_val_base = val_df[target_col] if val_df is not None else None

        train_valid_mask = ~(X_train_base.isna().any(axis=1) | y_train_base.isna())
        X_train = X_train_base[train_valid_mask]
        y_train = y_train_base[train_valid_mask]

        if X_val_base is not None and y_val_base is not None:
            val_valid_mask = ~(X_val_base.isna().any(axis=1) | y_val_base.isna())
            X_val = X_val_base[val_valid_mask]
            y_val = y_val_base[val_valid_mask]
        else:
            X_val = None
            y_val = None
        return X_train, y_train, X_val, y_val

    def _search_hparams(self, X_train, y_train, X_val, y_val, inner_n_jobs=None):
        """Grid-search hyperparameters using a PredefinedSplit validation fold.

        Combines train+val into one matrix and tells GridSearchCV to score only on
        the validation rows (test_fold==0). Returns (best_params, best_val_mse).

        inner_n_jobs: when set, overrides the GridSearchCV job count. The outer
        refit loop passes 1 when it is itself running in parallel, so the inner
        search and outer loop never nest into core oversubscription.
        """
        X_combined_for_cv = pd.concat([X_train, X_val], axis=0)
        y_combined_for_cv = pd.concat([y_train, y_val], axis=0)

        # PredefinedSplit: -1 = train (never in test fold), 0 = validation fold
        test_fold = np.concatenate([
            np.full(len(X_train), -1),
            np.full(len(X_val), 0),
        ])
        ps = PredefinedSplit(test_fold)

        # If the model parallelizes internally, don't also parallelize CV. When
        # the outer (target, horizon) loop is already parallel, the caller passes
        # inner_n_jobs=1 to prevent nested oversubscription.
        if inner_n_jobs is not None:
            cv_n_jobs = inner_n_jobs
        else:
            cv_n_jobs = 1 if self._uses_inner_jobs() else -1

        grid_search = GridSearchCV(
            estimator=self._create_single_model(),
            param_grid=self.hyperparameter_grid,
            cv=ps,
            scoring='neg_mean_squared_error',
            n_jobs=cv_n_jobs,
            refit=False,  # We'll refit manually on combined data
            verbose=0,
        )
        grid_search.fit(X_combined_for_cv, y_combined_for_cv)
        return grid_search.best_params_, -grid_search.best_score_

    def fit(self, train_data, validation_data: Optional[pd.DataFrame] = None) -> None:
        """
        Train the model with hyperparameter tuning.

        This method:
        1. Extracts features and targets from data
        2. Uses train/validation split for hyperparameter search
        3. Refits on combined train+validation with best hyperparameters

        Args:
            train_data: DataFrame with features and target columns, or data module
            validation_data: Optional validation DataFrame
        """

        # Handle data module vs DataFrame input
        if hasattr(train_data, 'get_train_data'):
            # It's a data module
            train_df = train_data.get_train_data()
            val_df = train_data.get_val_data() if validation_data is None else validation_data
            target_variables = getattr(train_data, 'target_variables', None)

            # Load feature and target column metadata from the data module
            # Check that the lists are populated, not just that the attributes exist
            feature_cols = getattr(train_data, 'feature_columns', [])
            target_cols = getattr(train_data, 'target_columns', [])
            if feature_cols and target_cols:
                self.feature_columns = feature_cols
                self.target_columns = target_cols
            else:
                # Fallback to auto-detection if metadata not available
                self.target_columns = self._extract_target_columns(train_df, target_variables)
                self.feature_columns = [c for c in train_df.columns if c not in self.target_columns]
                print(f"Warning: Using auto-detection for feature/target columns. Metadata columns may be included as features.")
        else:
            # It's a DataFrame
            train_df = train_data
            val_df = validation_data
            target_variables = None

            # Fallback to auto-detection for direct DataFrame input
            self.target_columns = self._extract_target_columns(train_df, target_variables)
            self.feature_columns = [c for c in train_df.columns if c not in self.target_columns]
            print(f"Warning: Using auto-detection for feature/target columns. Metadata columns may be included as features.")

        if len(self.target_columns) == 0:
            raise ValueError("No target columns found in training data. Expected columns like 'niq_t1', 'niq_t2', etc.")

        print(f"\n{self.model_name}: Training on {len(self.feature_columns)} features for {len(self.target_columns)} target horizons")

        # Prepare base training and validation feature sets
        X_train_base = train_df[self.feature_columns].copy()
        X_val_base = val_df[self.feature_columns].copy() if val_df is not None else None

        # Validate and convert feature dtypes to numeric (required for sklearn/cuML)
        non_numeric_cols = X_train_base.columns.difference(
            X_train_base.select_dtypes(include=[np.number]).columns
        ).tolist()
        if non_numeric_cols:
            preview_cols = non_numeric_cols[:5]
            print(f"Warning: Converting {len(non_numeric_cols)} non-numeric columns to float: {preview_cols}{'...' if len(non_numeric_cols) > 5 else ''}")
            for col in non_numeric_cols:
                X_train_base[col] = pd.to_numeric(X_train_base[col], errors='coerce')
                if X_val_base is not None:
                    X_val_base[col] = pd.to_numeric(X_val_base[col], errors='coerce')

        # Ensure all columns are float32 for GPU compatibility
        X_train_base = X_train_base.astype('float32')
        if X_val_base is not None:
            X_val_base = X_val_base.astype('float32')

        # Stamp the fit signature now that feature/target columns are finalized, so
        # incremental persistence + resume can detect a config/grid/dataset change.
        self._fit_signature_hash = self._fit_signature()

        # --- Optional: shared hyperparameters per horizon ---
        # When enabled, search the grid only on the reference target's horizons,
        # producing one param set per horizon h. Every (target, h) pair then reuses
        # horizon h's params, so the grid is searched H times instead of |targets|*H
        # times. When disabled, behaviour is unchanged: search per (target, horizon).
        shared_mode = self.shared_horizon_hparams and self.hyperparameter_grid is not None
        self.params_by_horizon = {}
        if shared_mode:
            ref = self.hparam_reference_target
            ref_cols = [c for c in self.target_columns if self._parse_base_var(c) == ref]
            if not ref_cols:
                available = sorted({self._parse_base_var(c) for c in self.target_columns})
                raise ValueError(
                    f"shared_horizon_hparams=True with reference target '{ref}', but no "
                    f"target columns match '{ref}_t*'. Available base targets: {available}"
                )
            print(f"\n{self.model_name}: Selecting shared hyperparameters per horizon "
                  f"using reference target '{ref}' ({len(ref_cols)} horizons)")
            for target_col in sorted(ref_cols, key=self._parse_horizon):
                h = self._parse_horizon(target_col)

                # Resume: the per-horizon search trains many models on the
                # reference target — expensive to redo on every restart. If this
                # reference (target, horizon) was already persisted (and matches
                # this run's signature), recover its chosen params from the sidecar
                # instead of re-searching.
                if self.incremental_dir is not None:
                    sc = self._completed_pair(self.incremental_dir, target_col)
                    if sc is not None:
                        self.params_by_horizon[h] = sc['best_params']
                        tqdm.write(f"    [ref {ref}] h={h}: reusing params from disk (resume)")
                        continue

                X_train, y_train, X_val, y_val = self._prepare_xy(
                    target_col, X_train_base, X_val_base, train_df, val_df)
                if X_val is not None and len(X_val) > 0:
                    best_params, best_score = self._search_hparams(X_train, y_train, X_val, y_val)
                    self.params_by_horizon[h] = best_params
                    tqdm.write(f"    [ref {ref}] h={h}: Best params = {best_params}, Val MSE = {best_score:.6f}")
                else:
                    self.params_by_horizon[h] = self.hyperparameters
                    tqdm.write(f"    [ref {ref}] h={h}: No validation data; using default params")

        # Train a separate model for each (target, forecast horizon). These fits
        # are mutually independent, so we fan them out across CPUs. The outer loop
        # is only parallelized for estimators whose single .fit() is itself
        # single-threaded (the linear family — see _supports_outer_parallel); for
        # RandomForest / GPU / multithreaded estimators _effective_outer_n_jobs
        # returns 1 so behaviour is unchanged. When the loop runs in parallel we
        # force any inner GridSearchCV to n_jobs=1 so the two levels never nest.
        effective_outer = self._effective_outer_n_jobs()
        outer_parallel = effective_outer != 1
        inner_n_jobs = 1 if outer_parallel else None

        # Incremental/resumable mode: stream each model to disk and free it as it
        # is trained (bounded memory), and skip pairs already on disk (resume).
        if self.incremental_dir is not None:
            os.makedirs(self.incremental_dir, exist_ok=True)
            already_done = self._completed_targets(self.incremental_dir)
            if already_done:
                tqdm.write(f"{self.model_name}: incremental save -> {self.incremental_dir}; "
                           f"resuming, {len(already_done)}/{len(self.target_columns)} "
                           f"(target, horizon) models already on disk will be skipped")
            else:
                tqdm.write(f"{self.model_name}: incremental save -> {self.incremental_dir}; "
                           f"each model is flushed to disk and freed as it finishes")

        common_args = (X_train_base, X_val_base, train_df, val_df, shared_mode, inner_n_jobs)
        if outer_parallel:
            blas_note = "" if THREADPOOLCTL_AVAILABLE else " [threadpoolctl missing: BLAS not capped]"
            tqdm.write(f"{self.model_name}: Refitting {len(self.target_columns)} (target, horizon) "
                       f"models with outer_n_jobs={effective_outer}, backend='{self.outer_backend}' "
                       f"(inner search forced to n_jobs=1){blas_note}")
            # Cap BLAS/LAPACK to 1 thread per worker for the duration of the loop:
            # linear/ridge solvers are multithreaded internally, so without this the
            # outer threads would oversubscribe as outer_n_jobs * BLAS_threads.
            # No-op for the coordinate-descent solvers (elasticnet/lasso).
            with threadpool_limits(limits=1):
                results = joblib.Parallel(n_jobs=effective_outer, backend=self.outer_backend)(
                    joblib.delayed(self._fit_one_target)(horizon_idx, target_col, *common_args)
                    for horizon_idx, target_col in enumerate(self.target_columns, start=1)
                )
        else:
            results = [
                self._fit_one_target(horizon_idx, target_col, *common_args)
                for horizon_idx, target_col in enumerate(self.target_columns, start=1)
            ]

        # Collect results into self after the (possibly concurrent) fits complete.
        # In incremental mode the worker has already flushed the model to disk and
        # returns final_model=None, so nothing heavy accumulates here — only the
        # tiny best_params dict. predict()/save() lazily load each model on demand.
        for target_col, final_model, best_params in results:
            self.best_params[target_col] = best_params
            if final_model is not None:
                self.models_by_horizon[target_col] = final_model

        self.is_fitted = True
        print(f"{self.model_name}: Training completed for all horizons")

    def _fit_one_target(self, horizon_idx, target_col, X_train_base, X_val_base,
                        train_df, val_df, shared_mode, inner_n_jobs):
        """Fit one (target, horizon) model.

        Does not mutate self (reads shared, read-only state only), so it is safe
        to call concurrently across horizons. Returns
        (target_col, fitted_model_or_None, best_params); the caller writes the
        results into self.models_by_horizon / self.best_params once all fits
        finish.

        In incremental mode (self.incremental_dir set) the trained model is
        flushed to disk and dropped here — returned as None — so peak memory stays
        bounded by a single model regardless of how many (target, horizon) pairs
        there are. Pairs already on disk are skipped (resume).
        """
        incremental = self.incremental_dir is not None
        h = self._parse_horizon(target_col)

        # Resume: a fully-persisted pair (pkl + sidecar, matching this run's fit
        # signature) is skipped without re-reading the data or re-fitting. A pkl
        # with no sidecar (crash between the two writes) or a signature mismatch
        # is treated as not-done and re-trained — see _completed_pair.
        if incremental:
            sc = self._completed_pair(self.incremental_dir, target_col, warn=True)
            if sc is not None:
                tqdm.write(f"{self.model_name} - Horizon {horizon_idx}/{len(self.target_columns)}: "
                           f"{target_col}: already trained, skipping (resume)")
                return target_col, None, sc['best_params']

        X_train, y_train, X_val, y_val = self._prepare_xy(
            target_col, X_train_base, X_val_base, train_df, val_df)

        # Log how many samples remain after dropping NaN
        n_train_orig = len(X_train_base)
        n_train_clean = len(X_train)
        if X_val is not None:
            n_val_orig = len(X_val_base)
            n_val_clean = len(X_val)
            tqdm.write(f"{self.model_name} - Horizon {horizon_idx}/{len(self.target_columns)}: {target_col}: "
                       f"Train {n_train_clean}/{n_train_orig} samples, Val {n_val_clean}/{n_val_orig} samples (after dropping NaN)")
        else:
            tqdm.write(f"{self.model_name} - Horizon {horizon_idx}/{len(self.target_columns)}: {target_col}: "
                       f"Train {n_train_clean}/{n_train_orig} samples (after dropping NaN)")

        # Choose hyperparameters
        if shared_mode:
            # Reuse this horizon's shared params. Fall back to a per-column
            # search (or defaults) only if the reference target lacked horizon h.
            best_params = self.params_by_horizon.get(h)
            if best_params is None:
                if X_val is not None and len(X_val) > 0:
                    best_params, best_score = self._search_hparams(
                        X_train, y_train, X_val, y_val, inner_n_jobs=inner_n_jobs)
                    tqdm.write(f"    {target_col}: [shared-mode fallback, h={h} missing from reference] "
                               f"Best params = {best_params}, Val MSE = {best_score:.6f}")
                else:
                    best_params = self.hyperparameters
                    tqdm.write(f"    {target_col}: [shared-mode fallback, h={h} missing from reference] "
                               f"No validation data; using default params")
        elif X_val is not None and self.hyperparameter_grid is not None:
            # Independent search per (target, horizon)
            best_params, best_score = self._search_hparams(
                X_train, y_train, X_val, y_val, inner_n_jobs=inner_n_jobs)
            tqdm.write(f"    {target_col}: Best params = {best_params}, Val MSE = {best_score:.6f}")
        else:
            # No hyperparameter tuning, use default parameters
            best_params = self.hyperparameters

        # Refit on combined train + validation data with best hyperparameters
        if X_val is not None:
            X_combined = pd.concat([X_train, X_val], axis=0)
            y_combined = pd.concat([y_train, y_val], axis=0)
        else:
            X_combined = X_train
            y_combined = y_train

        # Train final model
        final_model = self._create_single_model(**best_params)
        final_model.fit(X_combined, y_combined)

        # Print coefficient info for linear models
        self._print_model_info(final_model, target_col)

        # Incremental mode: flush this model to disk now and hand back None so it
        # is freed immediately instead of accumulating across all horizons.
        if incremental:
            self._persist_one(self.incremental_dir, target_col, final_model, best_params)
            return target_col, None, best_params

        return target_col, final_model, best_params

    def _iterate_grid(self, param_grid: Dict) -> List[Dict]:
        """
        Generate all combinations from parameter grid.

        Args:
            param_grid: Dictionary of parameter names to lists of values

        Returns:
            List of parameter dictionaries
        """
        import itertools

        keys = list(param_grid.keys())
        values = [param_grid[k] if isinstance(param_grid[k], list) else [param_grid[k]]
                  for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def predict(self, test_data, stream_save_paths=None, stream_group_size=4) -> pd.DataFrame:
        """
        Generate predictions for all forecast horizons.

        Args:
            test_data: DataFrame with features or data module
            stream_save_paths: Optional list of parquet paths. When provided, the
                tall forecast is written incrementally to each path (one
                pyarrow ParquetWriter per path, opened on the first group)
                in groups of ~stream_group_size targets, instead of building
                one giant wide frame + a single melt in memory, and the method
                returns None. Peak memory is bounded to ~one group regardless
                of the (target, horizon) column count -- needed for the 78x20
                RF sweep on memory-capped pods. The streamed file holds the same
                row *set* as the in-memory path (grouped melting reorders the tall
                rows, which is fine: evaluate.py inner-joins on keys), with the
                canonical schema/dtypes.
            stream_group_size: Targets per streamed group (each group spans
                stream_group_size*20 columns).

        Returns:
            DataFrame in tall format with columns: 'firm_id', 'target', 'quarter',
            'forecast_horizon', 'prediction', 'model'. Returns None when streaming.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Handle data module vs DataFrame input
        if hasattr(test_data, 'get_test_data'):
            # It's a data module
            test_df = test_data.get_test_data()
        else:
            # It's a DataFrame
            test_df = test_data
        
        # Extract features
        X_test = test_df[self.feature_columns]

        # Drop rows with NaN in any feature
        valid_mask = ~X_test.isna().any(axis=1)
        X_test_clean = X_test[valid_mask]
        test_df_clean = test_df.loc[valid_mask]

        # Log how many samples remain after dropping NaN
        n_test_orig = len(X_test)
        n_test_clean = len(X_test_clean)
        print(f"\n{self.model_name}: Predicting on {n_test_clean}/{n_test_orig} test samples (after dropping NaN)")

        if n_test_clean == 0:
            raise ValueError("No valid test samples remaining after dropping NaN values")

        # Streaming mode: write the tall forecast to disk in groups so peak memory
        # stays bounded by ~one group (avoids the full wide-frame + single melt that
        # OOMs the 78x20 sweep on memory-capped pods). Produces identical rows.
        if stream_save_paths:
            import pyarrow as pa
            # finalize_forecast_frame is imported function-locally (not at module
            # scope) so test_baseline_streaming's crash test can monkeypatch
            # forecast_io.finalize_forecast_frame to inject a mid-stream failure --
            # the patch is only visible if the name is re-read from the module here
            # at call time. Keep it local.
            from forma.models.forecast_io import finalize_forecast_frame, atomic_parquet_writers
            cols = self.target_columns
            # group_cols spans stream_group_size *targets*; the *20 assumes 20
            # horizons. This is only a memory-sizing heuristic, NOT a correctness
            # assumption -- each target column melts independently, so any grouping
            # (or a run with !=20 horizons) yields the same row-set. Don't "fix" the
            # 20 to match horizon count.
            group_cols = max(1, stream_group_size) * 20
            firm_vals = test_df_clean['firm_id'].values
            quarter_vals = test_df_clean['quarter'].values
            # atomic_parquet_writers handles the .tmp + os.replace + crash-cleanup
            # (shared with the Lightning streaming path) and yields a write_table().
            with atomic_parquet_writers(stream_save_paths) as write_table:
                for start in range(0, len(cols), group_cols):
                    group = cols[start:start + group_cols]
                    pdict = {}
                    for target_col in group:
                        model, loaded = self._get_model(target_col)
                        pdict[target_col] = model.predict(X_test_clean)
                        if loaded:
                            del model
                    wide = pd.DataFrame(pdict, index=X_test_clean.index)
                    wide['firm_id'] = firm_vals
                    wide['quarter'] = quarter_vals
                    wide['model'] = self.model_name
                    tall = wide_forecasts_to_tall(wide, group)
                    tall = finalize_forecast_frame(tall)
                    tbl = pa.Table.from_pandas(tall, preserve_index=False)
                    write_table(tbl)
                    del pdict, wide, tall, tbl
                    print(f"  {self.model_name}: streamed cols {start + 1}-{start + len(group)}/{len(cols)}")
            return None

        # Generate predictions for each horizon. Models are fetched one at a time
        # (loaded from disk on demand when they were streamed out during fit / a
        # lazy load) and freed right after predicting, so the 1560-model RF sweep
        # never holds more than one forest resident at once.
        predictions_dict = {}
        for target_col in self.target_columns:
            model, loaded = self._get_model(target_col)
            predictions_dict[target_col] = model.predict(X_test_clean)
            if loaded:
                del model

        # Create DataFrame with all predictions in wide format (using the cleaned index)
        forecast_wide = pd.DataFrame(predictions_dict, index=X_test_clean.index)

        # Add metadata columns
        forecast_wide['firm_id'] = test_df_clean['firm_id'].values
        forecast_wide['quarter'] = test_df_clean['quarter'].values
        forecast_wide['model'] = self.model_name

        # Convert from wide to tall format via the shared melt helper (also used
        # by FeedForwardModel) so the standardized schema lives in one place.
        return wide_forecasts_to_tall(forecast_wide, self.target_columns)

    def save(self, folder_path: str) -> None:
        """Persist all trained models plus metadata to folder_path.

        Memory-lazy: models are handled one at a time. After an incremental fit
        every per-horizon pkl is already on disk and its sidecar already holds the
        metadata/coef-summary record computed at persist time — so this builds the
        full metadata.json straight from the sidecars, without unpickling a single
        forest. Only the non-incremental path (models resident in memory) extracts
        from the live model and dumps the pkl here.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        os.makedirs(folder_path, exist_ok=True)

        # Single pass: persist (if needed), extract metadata, build coef summary.
        model_parameters = {}
        summary_rows = []
        for target_col in self.target_columns:
            pkl_path = self._model_pkl_path(folder_path, target_col)

            # Fast path: a sidecar with cached metadata sits next to its pkl (the
            # incremental/resumed case). Build metadata from it — no model load.
            sc = self._read_sidecar(folder_path, target_col)
            if sc is not None and 'model_param_dict' in sc and os.path.exists(pkl_path):
                model_parameters[target_col] = sc['model_param_dict']
                if sc.get('coef_summary_row'):
                    summary_rows.append(sc['coef_summary_row'])
                continue

            # Fallback: model in memory (non-incremental) or a sidecar without the
            # cached record — load/extract from the live model and dump if needed.
            model, loaded = self._get_model(target_col)
            if not os.path.exists(pkl_path):
                self._atomic_dump(lambda p: joblib.dump(model, p), pkl_path)

            model_parameters[target_col] = self._model_param_dict(model, target_col)
            row = self._coef_summary_row(model, target_col)
            if row is not None:
                summary_rows.append(row)

            if loaded:
                del model

        # Save comprehensive metadata including model parameters
        metadata_path = os.path.join(folder_path, f"{self.model_name}_metadata.json")
        metadata = {
            'hyperparameter_grid': self.hyperparameter_grid,
            'best_params': self.best_params,
            'target_columns': self.target_columns,
            'feature_columns': self.feature_columns,
            'model_parameters': model_parameters
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Also save a summary CSV of coefficients for easy inspection
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(folder_path, f"{self.model_name}_coefficient_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"  Coefficient summary saved to {summary_path}")

    def _model_param_dict(self, model, target_col: str, best_params: Dict = None) -> Dict:
        """Extract a JSON-serialisable parameter/coefficient record for one model.

        best_params is passed explicitly at persist time (when self.best_params is
        not yet populated); save() omits it and falls back to self.best_params.
        """
        param_dict = {}

        # Save the hyperparameters used for this specific model
        hp = best_params if best_params is not None else self.best_params.get(target_col)
        if hp is not None:
            param_dict['hyperparameters'] = hp

        # Extract coefficients/weights if available
        if hasattr(model, 'coef_'):
            coef_array = model.coef_
            param_dict['coefficients'] = {
                'values': coef_array.tolist(),
                'shape': list(coef_array.shape),
                'l1_norm': float(np.abs(coef_array).sum()),
                'l2_norm': float(np.sqrt((coef_array**2).sum())),
                'non_zero_count': int((coef_array != 0).sum()),
                'total_count': int(len(coef_array))
            }

        # Extract intercept if available
        if hasattr(model, 'intercept_'):
            param_dict['intercept'] = float(model.intercept_) if np.isscalar(model.intercept_) else model.intercept_.tolist()

        # Extract tree-based model info if available
        if hasattr(model, 'feature_importances_'):
            param_dict['feature_importances'] = model.feature_importances_.tolist()

        return param_dict


    def _print_model_info(self, model, target_col: str) -> None:
        """Print information about the fitted model's parameters."""
        if hasattr(model, 'coef_'):
            coef_array = model.coef_
            non_zero = (coef_array != 0).sum()
            total = len(coef_array)
            l2_norm = np.sqrt((coef_array**2).sum())
            
            info_parts = [f"Coefficients: {non_zero}/{total} non-zero"]
            info_parts.append(f"L2 norm: {l2_norm:.4f}")
            
            # Add model-specific hyperparameters
            if hasattr(model, 'alpha'):
                info_parts.append(f"alpha: {model.alpha}")
            if hasattr(model, 'l1_ratio'):
                info_parts.append(f"l1_ratio: {model.l1_ratio}")
            
            tqdm.write(f"    {target_col} fitted: {', '.join(info_parts)}")

    def _coef_summary_row(self, model, target_col: str, best_params: Dict = None) -> Optional[Dict]:
        """Build a coefficient-summary CSV row for one model, or None if the model
        has no coefficients (e.g. tree-based models like Random Forest).

        best_params is passed explicitly at persist time; save() omits it and falls
        back to self.best_params.
        """
        if not hasattr(model, 'coef_'):
            return None

        coef_array = model.coef_
        row = {
            'target': target_col,
            'l1_norm': float(np.abs(coef_array).sum()),
            'l2_norm': float(np.sqrt((coef_array**2).sum())),
            'max_abs_coef': float(np.abs(coef_array).max()),
            'mean_abs_coef': float(np.abs(coef_array).mean()),
            'non_zero_count': int((coef_array != 0).sum()),
            'total_count': int(len(coef_array)),
            'sparsity': float((coef_array == 0).sum() / len(coef_array))
        }

        # Add hyperparameters used
        hp = best_params if best_params is not None else self.best_params.get(target_col)
        if hp is not None:
            for param_name, param_value in hp.items():
                row[f'hyperparam_{param_name}'] = param_value

        return row

    def load(self, folder_path: str) -> None:
        """Load a trained model from folder_path for prediction.

        Memory-lazy: this only reads metadata and verifies each per-horizon pkl
        exists. The forests themselves are loaded one at a time by predict() (via
        _get_model) and freed after use, so loading a 1560-model RF sweep for
        inference does not blow up memory the way eagerly unpickling all of them
        would.
        """
        # Load metadata first to get target columns
        metadata_path = os.path.join(folder_path, f"{self.model_name}_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.hyperparameter_grid = metadata.get('hyperparameter_grid', None)
            self.best_params = metadata.get('best_params', {})
            self.target_columns = metadata.get('target_columns', [])
            self.feature_columns = metadata.get('feature_columns', [])

        # Verify every per-horizon model file is present without loading them.
        # predict() loads each on demand from this directory and frees it after.
        self.models_by_horizon = {}
        self._model_dir = folder_path
        for target_col in self.target_columns:
            model_path = self._model_pkl_path(folder_path, target_col)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

        self.is_fitted = True
        print(f"{self.model_name}: Models loaded from {folder_path}")


class SklearnBaselineModel(MultiHorizonBaselineModel):
    """
    Wrapper for sklearn models to conform to BaseModel interface.

    This model supports multi-horizon forecasting by training separate models
    for each forecast horizon (t1, t2, ..., t20).
    """

    def __init__(self, model_type: str = "linear", hyperparameter_grid: Dict = None, **kwargs):
        """
        Initialize sklearn baseline model.

        Args:
            model_type: Type of sklearn model ('linear', 'ridge', 'lasso', 'random_forest')
            hyperparameter_grid: Grid of hyperparameters for grid search (optional)
            **kwargs: Model-specific parameters (used if no grid search)
        """
        super().__init__(f"sklearn_{model_type}", hyperparameter_grid, **kwargs)
        self.model_type = model_type

    def _uses_inner_jobs(self) -> bool:
        """Random forest models use internal parallelization (cuML GPU or sklearn n_jobs)."""
        return self.model_type == "random_forest"

    def _supports_outer_parallel(self) -> bool:
        """The linear family can be parallelized at the outer (target, horizon)
        level. elasticnet/lasso use single-threaded coordinate descent; linear/
        ridge call into multithreaded LAPACK/BLAS, which fit() caps to one thread
        per worker (threadpoolctl) so the outer loop doesn't oversubscribe.
        RandomForest is excluded — it parallelizes internally via n_jobs / cuML."""
        return self.model_type in ("linear", "ridge", "lasso", "elasticnet")

    def _create_single_model(self, **params):
        """Create the appropriate sklearn model based on model_type."""
        # Merge base parameters with grid search parameters
        all_params = {k: v for k, v in self.hyperparameters.items()
                      if k not in ['hyperparameter_grid', 'sklearn_type']}
        all_params.update(params)

        if self.model_type == "linear":
            return LinearRegression(**all_params)
        elif self.model_type == "ridge":
            return Ridge(**all_params)
        elif self.model_type == "lasso":
            return Lasso(**all_params)
        elif self.model_type == "elasticnet":
            return ElasticNet(**all_params)
        elif self.model_type == "random_forest":
            # Use GPU-accelerated cuML if available
            if CUML_AVAILABLE:
                # cuML RandomForest doesn't support n_jobs (it's GPU-native).
                # Different RAPIDS versions accept either `seed` or `random_state`.
                cuml_params = {k: v for k, v in all_params.items() if k not in ['n_jobs', 'random_state', 'seed']}

                if 'max_features' in cuml_params:
                    # cuML accepts 'sqrt', 'log2', 'auto', or float
                    max_feat = cuml_params['max_features']
                    if max_feat == 'sqrt' or max_feat == 'auto':
                        cuml_params['max_features'] = 'sqrt'
                    elif max_feat == 'log2':
                        cuml_params['max_features'] = 'log2'

                _set_cuml_random_forest_seed_param(cuMLRandomForestRegressor, cuml_params, all_params.get('random_state'))

                filtered_params = _callable_supported_kwargs(cuMLRandomForestRegressor, cuml_params)

                # print(f"  Using GPU-accelerated cuML RandomForest")
                try:
                    return cuMLRandomForestRegressor(**filtered_params)
                except TypeError:
                    # Final fallback: drop any seed-like args and retry once.
                    filtered_params.pop('seed', None)
                    filtered_params.pop('random_state', None)
                    return cuMLRandomForestRegressor(**filtered_params)
            else:
                # Fall back to sklearn CPU implementation
                if 'n_jobs' not in all_params:
                    try:
                        cpu_count = os.cpu_count() or 1
                    except Exception:
                        cpu_count = 1
                    # ensure at least 1 core is used
                    default_n_jobs = max(1, cpu_count - 1)
                    all_params['n_jobs'] = default_n_jobs
                return RandomForestRegressor(**all_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

class XGBoostModel(MultiHorizonBaselineModel):
    """
    XGBoost model wrapper for the BaseModel interface.

    This model supports multi-horizon forecasting by training separate models
    for each forecast horizon (t1, t2, ..., t20).

    Automatically uses GPU acceleration if CUDA is available.
    """
    
    def __init__(self, hyperparameter_grid: Dict = None, use_gpu: bool = None, **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            hyperparameter_grid: Grid of hyperparameters for grid search (optional)
            use_gpu: Force GPU usage (True), CPU usage (False), or auto-detect (None)
            **kwargs: XGBoost parameters (used if no grid search)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")
        
        # Auto-detect GPU availability if not specified
        if use_gpu is None:
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False

        self.use_gpu = use_gpu

        # Set default XGBoost parameters
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }

        # Add GPU-specific parameters if GPU is available
        if use_gpu:
            default_params['tree_method'] = 'gpu_hist'
            default_params['device'] = 'cuda'
            print("XGBoost: Using GPU acceleration (tree_method='gpu_hist')")
        else:
            default_params['tree_method'] = 'hist'  # Fast CPU histogram method
            print("XGBoost: Using CPU (tree_method='hist')")

        default_params.update(kwargs)

        super().__init__("xgboost", hyperparameter_grid, **default_params)

    def _uses_inner_jobs(self) -> bool:
        """XGBoost with GPU should not be parallelized at the CV level."""
        return self.use_gpu

    def _create_single_model(self, **params):
        """Create an XGBoost regressor with given parameters."""
        # Ensure GPU settings are preserved if set at init
        if self.use_gpu and 'tree_method' not in params:
            params['tree_method'] = 'gpu_hist'
            params['device'] = 'cuda'
        return xgb.XGBRegressor(**params)

    def _fit_single_model(self, model, X, y):
        """Override to suppress verbose output during XGBoost training."""
        model.fit(X, y, verbose=False)
        return model
