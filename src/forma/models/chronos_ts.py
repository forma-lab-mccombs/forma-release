"""Chronos-2 zero-shot time-series foundation model benchmark (multivariate).

A thin ``BaseModel`` wrapper around Amazon's `Chronos-2
<https://huggingface.co/amazon/chronos-2>`_ foundation model (Apache-2.0) so it
can be benchmarked head-to-head with Forma and the tabular baselines through the
same ``train.py`` / ``evaluate.py`` plumbing.

Design -- multivariate, per firm
--------------------------------
Chronos-2 is a *zero-shot* forecaster that reads a series' own past values and
extrapolates ``H`` steps ahead; Chronos-2 additionally supports *multivariate*
groups (it shares information across the variates of one 2-D input via group
attention). We exploit that to mirror **Forma-at-inference**: for each
``(firm, origin-quarter)`` we feed that firm's whole account panel -- all target
accounts stacked as one ``(n_accounts, history_length)`` group -- and forecast
every account jointly for horizons ``1..H`` in a single call. This matches how
Forma sees a firm (its full account panel at once, no covariates, no cross-firm
features) far better than univariate-per-account would, and costs ~``n_accounts``x
fewer model calls.

Context source & scale: each account's history is its ``{account}_level_0`` value
across the firm's quarters (the tabular ``level_0`` features are imputation-dense,
so the 78-variate panel has no gaps). Empirically ``{acct}_level_0[Q+h] ==
{acct}_t{h}[Q]`` (same per-account asinh z-score regularization), so a forecast
produced from a ``level_0`` context lands in the exact space ``evaluate.py`` scores
the ``{acct}_t{h}`` actuals in -- no rescaling.

Parity with Forma: we emit a forecast for every account, and ``evaluate.py``
inner-joins to the *sparse* ``{acct}_t{h}`` actuals (~12% NaN) -- so both models
are scored on exactly the observed ("present") cells, and the pairwise evaluator
further restricts to cells both models forecast. No explicit present-in-q0 filter
is needed on the forecast side.

Probabilistic output: Chronos returns quantiles. We take the 0.5 quantile as the
point ``prediction`` and derive a Gaussian-equivalent ``sigma`` from a symmetric
quantile pair (default 0.1/0.9): ``sigma = (q_hi - q_lo) / (z_hi - z_lo)``. That
feeds evaluate.py's likelihood track (Gaussian NLL / CRPS) alongside the point
track (R2/MAE from the median).

Zero-shot: ``fit()`` trains nothing -- it captures the train/val history and
metadata. The pretrained weights are imported and loaded lazily (only when a
forecast is actually requested) so importing this module -- and running the rest
of the repo -- never requires ``chronos-forecasting`` to be installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseModel

DEFAULT_MODEL_ID = "amazon/chronos-2"
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]

# Chronos-2's 21 native training quantile levels; used as the integration grid
# for the target-space predictive mean (see ChronosRawTSModel).
DEFAULT_MEAN_QUANTILE_GRID = (
    [0.01, 0.05] + [round(0.05 * i, 2) for i in range(2, 19)] + [0.95, 0.99]
)


def quantile_col(level: float) -> str:
    """Column name for a persisted quantile level: q_0.01 ... q_0.99.

    ``:g`` so 0.10 -> 'q_0.1' deterministically. Scoring code recovers the
    level by parsing the suffix -- keep this the single naming authority."""
    return f"q_{level:g}"


def _norm_ppf(p: float) -> float:
    """Standard-normal quantile (inverse CDF). Imported locally to keep module
    import light; scipy is already a project dependency (evaluate.py uses it)."""
    from scipy.stats import norm
    return float(norm.ppf(p))


class ChronosTSModel(BaseModel):
    """Zero-shot multivariate Chronos-2 forecaster, packaged as a Forma BaseModel.

    Parameters
    ----------
    model_id : str
        HuggingFace id of the Chronos-2 pipeline (default ``amazon/chronos-2``).
    device : str
        ``"auto"`` (cuda if available else cpu), ``"cuda"``, or ``"cpu"``.
    torch_dtype : str
        ``"auto"`` (bf16 on cuda, fp32 on cpu) or an explicit torch dtype name.
    quantile_levels : list[float]
        Quantiles to request (0.5 and the sigma pair are always added). The point
        prediction is the 0.5 quantile.
    sigma_quantile_pair : tuple[float, float]
        Symmetric quantile pair used to derive the Gaussian-equivalent predictive std.
    max_context : int | None
        Cap on trailing history points per series (None = unbounded).
    batch_size : int
        Number of firm-origins (2-D groups) forecast per pipeline call.
    forecast_horizon : int
        Steps ahead. Overridden by the max ``_t{h}`` in the data module's targets.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        torch_dtype: str = "auto",
        quantile_levels: Optional[List[float]] = None,
        sigma_quantile_pair=(0.1, 0.9),
        max_context: Optional[int] = 200,
        batch_size: int = 256,
        forecast_horizon: int = 20,
        model_name: str = "chronos",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.quantile_levels = sorted(quantile_levels or DEFAULT_QUANTILES)
        self.sigma_quantile_pair = (float(sigma_quantile_pair[0]), float(sigma_quantile_pair[1]))
        self.max_context = int(max_context) if max_context else None
        self.batch_size = int(batch_size)
        self.forecast_horizon = int(forecast_horizon)

        qset = set(self.quantile_levels)
        qset.add(0.5)
        qset.update(self.sigma_quantile_pair)
        self._request_quantiles = sorted(qset)

        # Mirror of inference.save_sigma (set by train.py). When False the
        # likelihood-track sigma column is dropped and only the point track is scored.
        self.save_sigma = True

        # Populated in fit().
        self.feature_set = kwargs.get("feature_set", "core")
        self.target_variables: Optional[List[str]] = None
        self.target_columns: Optional[List[str]] = None
        self._hist_wide: Optional[pd.DataFrame] = None  # [firm_id, quarter, <account level_0>...]
        self._pipeline = None  # lazily loaded

    # ------------------------------------------------------------------ #
    # Pipeline loading (lazy)
    # ------------------------------------------------------------------ #
    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from chronos import BaseChronosPipeline
        except Exception as e:  # pragma: no cover - only without the dep
            raise ImportError(
                "ChronosTSModel requires the 'chronos-forecasting' package "
                "(and 'einops'). Install with: pip install chronos-forecasting einops"
            ) from e
        import torch

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = self.torch_dtype
        if dtype == "auto":
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        print(f"{self.model_name}: loading {self.model_id} on {device} ({dtype})...")
        self._pipeline = BaseChronosPipeline.from_pretrained(
            self.model_id, device_map=device, torch_dtype=dtype
        )
        return self._pipeline

    # ------------------------------------------------------------------ #
    # Panel helpers
    # ------------------------------------------------------------------ #
    def _accts(self, df: pd.DataFrame) -> List[str]:
        """Target accounts we can build a panel for: requested targets with a
        ``{acct}_level_0`` column present (falls back to all such columns)."""
        present = {c[: -len("_level_0")] for c in df.columns if c.endswith("_level_0")}
        if self.target_variables:
            return [t for t in self.target_variables if t in present]
        return sorted(present)

    def _levels_wide(self, df: pd.DataFrame, accts: List[str]) -> pd.DataFrame:
        """Return ``[firm_id, quarter, <accts...>]`` where each account column is
        that account's ``{acct}_level_0`` value (the firm's observed panel)."""
        if "firm_id" not in df.columns or "quarter" not in df.columns:
            raise ValueError("Chronos input needs 'firm_id' and 'quarter' columns")
        cols = ["firm_id", "quarter"] + [f"{a}_level_0" for a in accts]
        w = df[cols].copy()
        w.columns = ["firm_id", "quarter"] + list(accts)
        return w

    # ------------------------------------------------------------------ #
    # fit (zero-shot: capture history + metadata only)
    # ------------------------------------------------------------------ #
    def fit(self, train_data, validation_data: Optional[pd.DataFrame] = None) -> None:
        if hasattr(train_data, "get_train_data"):
            dm = train_data
            self.feature_set = getattr(dm, "feature_set", self.feature_set)
            self.target_variables = list(getattr(dm, "target_variables", []) or [])
            self.target_columns = list(getattr(dm, "target_columns", []) or [])

            horizons = []
            for c in self.target_columns:
                if "_t" in c:
                    try:
                        horizons.append(int(c.rsplit("_t", 1)[1]))
                    except ValueError:
                        pass
            if horizons:
                self.forecast_horizon = max(horizons)

            accts = self.target_variables
            hist = []
            for getter in ("get_train_data", "get_val_data"):
                if hasattr(dm, getter):
                    df = getattr(dm, getter)()
                    if df is not None and len(df):
                        hist.append(self._levels_wide(df, [a for a in accts if f"{a}_level_0" in df.columns]))
            self._hist_wide = pd.concat(hist, ignore_index=True) if hist else None
        else:
            self.target_variables = self._accts(train_data)
            self._hist_wide = self._levels_wide(train_data, self.target_variables)

        self.is_fitted = True
        n = 0 if self._hist_wide is None else len(self._hist_wide)
        print(
            f"{self.model_name}: zero-shot fit complete -- captured {n:,} historical firm-quarter "
            f"rows across {len(self.target_variables or [])} target accounts; "
            f"forecast_horizon={self.forecast_horizon}."
        )

    # ------------------------------------------------------------------ #
    # Forecasting
    # ------------------------------------------------------------------ #
    def _forecast_batch(self, contexts_2d: List[np.ndarray]) -> List[np.ndarray]:
        """Forecast a list of 2-D ``(n_variates, L)`` context arrays as multivariate
        groups. Returns a list of ``(n_variates, H, len(self._request_quantiles))``
        arrays (one per input group).

        Passes contexts positionally, which both the Chronos-2 API (``inputs``,
        returns a list of per-group tensors) and the classic API (returns a stacked
        tensor) accept; both return shapes are normalized to the per-group list.
        """
        import torch

        pipe = self._load_pipeline()
        out: List[np.ndarray] = []
        quantiles, _mean = pipe.predict_quantiles(
            [torch.as_tensor(c, dtype=torch.float32) for c in contexts_2d],
            prediction_length=self.forecast_horizon,
            quantile_levels=self._request_quantiles,
        )
        if isinstance(quantiles, (list, tuple)):
            out = [q.detach().to(torch.float32).cpu().numpy() for q in quantiles]  # each (n_var, H, Q)
        else:
            arr = quantiles.detach().to(torch.float32).cpu().numpy()  # classic: (B, H, Q)
            out = [arr[b][None, ...] for b in range(arr.shape[0])]
        return out

    def _emit(self, results: List[np.ndarray], meta_firm, meta_qtr, accts,
              i_med, i_lo, i_hi, z_span) -> pd.DataFrame:
        """Turn a batch of per-origin forecast arrays into the tall forecast schema."""
        A, H = len(accts), self.forecast_horizon
        med = np.stack([r[:, :, i_med] for r in results], axis=0)  # (B, A, H)
        sig = (np.stack([r[:, :, i_hi] for r in results], axis=0)
               - np.stack([r[:, :, i_lo] for r in results], axis=0)) / z_span
        sig = np.clip(sig, 1e-4, None)
        B = med.shape[0]

        # Column order matches FORECAST_SCHEMA_COLS (prediction, sigma, model) so the
        # non-streaming return is canonical without finalize; the streaming path
        # finalizes anyway.
        out = {
            "firm_id": np.repeat(np.asarray(meta_firm), A * H),
            "target": np.tile(np.repeat(np.asarray(accts, dtype=object), H), B),
            "quarter": np.repeat(np.asarray(meta_qtr, dtype="datetime64[ns]"), A * H),
            "forecast_horizon": np.tile(np.tile(np.arange(1, H + 1), A), B).astype("int64"),
            "prediction": med.reshape(-1),
        }
        if self.save_sigma:
            out["sigma"] = sig.reshape(-1)
        out["model"] = self.model_name
        return pd.DataFrame(out)

    def predict(self, test_data, stream_save_paths=None, stream_group_size=None) -> Optional[pd.DataFrame]:
        """Forecast horizons 1..H jointly for every firm-origin's account panel.

        Streams one origin-batch at a time to disk when ``stream_save_paths`` is
        given (returns None), else returns the full tall DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        test_df = test_data.get_test_data() if hasattr(test_data, "get_test_data") else test_data
        if not self.target_variables:
            self.target_variables = self._accts(test_df)
        accts = self._accts(test_df)
        if not accts:
            raise ValueError("No '*_level_0' target columns found in test data for Chronos context")

        # Full observed panel = train/val history + test-period rows.
        test_w = self._levels_wide(test_df, accts)
        frames = [f for f in (self._hist_wide, test_w) if f is not None and len(f)]
        full = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(["firm_id", "quarter"])
            .sort_values(["firm_id", "quarter"])
        )

        # Per-firm panel: quarters array + (n_accts, L) value matrix.
        panels = {}
        for fid, g in full.groupby("firm_id", sort=False):
            panels[fid] = (g["quarter"].to_numpy(), g[accts].to_numpy(dtype="float32").T)

        z_lo, z_hi = _norm_ppf(self.sigma_quantile_pair[0]), _norm_ppf(self.sigma_quantile_pair[1])
        z_span = (z_hi - z_lo) or 1.0
        i_med = self._request_quantiles.index(0.5)
        i_lo = self._request_quantiles.index(self.sigma_quantile_pair[0])
        i_hi = self._request_quantiles.index(self.sigma_quantile_pair[1])

        n_origins = len(test_df[["firm_id", "quarter"]].drop_duplicates())
        print(f"\n{self.model_name}: forecasting {len(accts)} accounts jointly x {n_origins:,} firm-origins "
              f"x {self.forecast_horizon} horizons (zero-shot multivariate)")

        # Accumulate origin contexts, flush a batch at a time.
        ctx_batch: List[np.ndarray] = []
        firm_batch: List = []
        qtr_batch: List = []
        parts: List[pd.DataFrame] = []

        def _process(tall_df, write_table, pa, finalize):
            if write_table is not None:
                write_table(pa.Table.from_pandas(finalize(tall_df), preserve_index=False))
            else:
                parts.append(tall_df)

        def _run_batch(write_table, pa, finalize):
            if not ctx_batch:
                return
            results = self._forecast_batch(ctx_batch)
            tall = self._emit(results, firm_batch, qtr_batch, accts, i_med, i_lo, i_hi, z_span)
            _process(tall, write_table, pa, finalize)
            ctx_batch.clear()
            firm_batch.clear()
            qtr_batch.clear()

        # Streaming vs in-memory setup. ExitStack propagates exceptions into the
        # writer's __exit__, so a crash mid-stream ABORTS the atomic publish (a
        # stray .tmp at worst) instead of os.replace-ing a truncated parquet onto
        # the canonical path that evaluate.py would read as complete.
        from contextlib import ExitStack

        if stream_save_paths:
            import pyarrow as pa
            from forma.models.forecast_io import atomic_parquet_writers, finalize_forecast_frame
            finalize = finalize_forecast_frame
        else:
            pa = finalize = None

        n_done = 0
        with ExitStack() as stack:
            write_table = (
                stack.enter_context(atomic_parquet_writers(stream_save_paths))
                if stream_save_paths else None
            )
            for fid, tg in test_df[["firm_id", "quarter"]].groupby("firm_id", sort=False):
                panel = panels.get(fid)
                if panel is None:
                    continue
                qarr, M = panel
                for oq in tg["quarter"].to_numpy():
                    j = int(np.searchsorted(qarr, oq, side="right"))
                    if j <= 0:
                        continue
                    ctx = M[:, :j]
                    if self.max_context and ctx.shape[1] > self.max_context:
                        ctx = ctx[:, -self.max_context:]
                    ctx_batch.append(ctx)
                    firm_batch.append(fid)
                    qtr_batch.append(oq)
                    if len(ctx_batch) >= self.batch_size:
                        _run_batch(write_table, pa, finalize)
                        n_done += self.batch_size
                        if n_done % (self.batch_size * 20) == 0:
                            print(f"  {self.model_name}: {n_done:,}/{n_origins:,} origins forecast")
            _run_batch(write_table, pa, finalize)

        if stream_save_paths:
            return None
        if not parts:
            from forma.models.forecast_io import FORECAST_SCHEMA_COLS
            return pd.DataFrame(columns=FORECAST_SCHEMA_COLS)
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------ #
    # Persistence (zero-shot: config + captured history)
    # ------------------------------------------------------------------ #
    def save(self, folder_path: str) -> None:
        p = Path(folder_path)
        p.mkdir(parents=True, exist_ok=True)
        cfg = {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "quantile_levels": self.quantile_levels,
            "sigma_quantile_pair": list(self.sigma_quantile_pair),
            "max_context": self.max_context,
            "batch_size": self.batch_size,
            "forecast_horizon": self.forecast_horizon,
            "feature_set": self.feature_set,
            "target_variables": self.target_variables,
            "target_columns": self.target_columns,
        }
        with open(p / "chronos_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        if self._hist_wide is not None:
            self._hist_wide.to_parquet(p / "chronos_history.parquet", index=False)
        print(f"{self.model_name}: saved config (+history) to {p}")

    def load(self, folder_path: str) -> None:
        p = Path(folder_path)
        cfg_path = p / "chronos_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            self.model_id = cfg.get("model_id", self.model_id)
            self.device = cfg.get("device", self.device)
            self.torch_dtype = cfg.get("torch_dtype", self.torch_dtype)
            self.quantile_levels = cfg.get("quantile_levels", self.quantile_levels)
            self.sigma_quantile_pair = tuple(cfg.get("sigma_quantile_pair", self.sigma_quantile_pair))
            self.max_context = cfg.get("max_context", self.max_context)
            self.batch_size = cfg.get("batch_size", self.batch_size)
            self.forecast_horizon = cfg.get("forecast_horizon", self.forecast_horizon)
            self.feature_set = cfg.get("feature_set", self.feature_set)
            self.target_variables = cfg.get("target_variables", self.target_variables)
            self.target_columns = cfg.get("target_columns", self.target_columns)
            qset = set(self.quantile_levels)
            qset.add(0.5)
            qset.update(self.sigma_quantile_pair)
            self._request_quantiles = sorted(qset)
        hist_path = p / "chronos_history.parquet"
        if hist_path.exists():
            self._hist_wide = pd.read_parquet(hist_path)
        self.is_fitted = True


class ChronosRawTSModel(ChronosTSModel):
    """Chronos-2 zero-shot benchmark v2: raw tuple contexts, origin-frozen outputs.

    Three deliberate changes from ``ChronosTSModel`` (v1, ``level_0`` contexts),
    each closing a comparability gap found in the v1 stress test:

    1. **Raw inputs, native regularization.** Context = the firm's *raw
       (unregularized)* account values from the tuple parquets, on a contiguous
       calendar-quarter grid with NaN gaps kept as NaN (Chronos-2 masks missing
       values natively; no cross-sectional imputation). Chronos then applies its
       own context normalization (per-series z-score + asinh) -- the regime it was
       trained in -- instead of double-transforming already-regularized z-scores.

    2. **Exact target space.** Raw-space forecast quantiles are mapped through the
       *origin-frozen* target transform

           ``T_Q(v) = clip((asinh(k * v / scale_Q) - mu_Q) / sigma_Q, +/-max_abs_z)``

       with the same reg-stats parquet, the same ``scale`` account value at the
       origin quarter, and the same clip that ``dataset_creation.py`` uses to build
       the ``{acct}_t{h}`` labels. This *removes* the v1 alignment drift
       (``level_0[Q+h]`` carries quarter-``Q+h`` scale/stats; the label carries
       quarter-``Q``'s). ``T_Q`` is strictly monotone (k, scale, sigma > 0), so
       quantiles map through exactly.

    3. **Forma-parity information set.** ``max_context`` defaults to 12
       observations *including* the origin quarter -- exactly Forma's
       ``max_lookback: 12`` window ``[Q-11, Q]`` (``FormaWindowDataset``:
       ``start_q = center_q - lookback + 1``). Accounts with at least one
       observation in the window are included as context (mirroring Forma, which
       keeps q<=0 tuples of accounts absent at q0); present-in-q0 masking of the
       *labels* happens downstream in the shared eval, identically for both models.

    Point forecasts -- loss geometry. ``T_Q`` is nonlinear, so
    ``E[T_Q(X)] != T_Q(E[X])``: the target-space predictive mean is computed by
    integrating ``T_Q`` over a quantile grid (default: Chronos-2's 21 native
    levels, trapezoid weights, tails treated as flat beyond the 0.01/0.99
    quantiles). The mean is the L2/R2-fair point; the mapped median ``T_Q(q0.5)``
    is the L1/MAE-fair point. In streaming mode both are written in one GPU pass:
    the primary file ``{model_name}`` carries ``point_stat`` (default mean) and a
    sibling file ``{model_name}_med`` carries the median. ``sigma`` (both files) is
    the Gaussian-equivalent predictive std, by default moment-matched over the
    full PRE-clip quantile grid (``sigma_method: "grid"``; ``"pair"`` selects the
    v1-style ``sigma_quantile_pair`` spread). sigma is written RAW -- zeros
    allowed, no floor -- so degenerate-certainty handling is an evaluation-time
    policy (``evaluate.py --sigma-floor``), revisable without a re-run.

    Quantile persistence (issue #246). With ``save_quantiles: true`` (streaming
    mode only) the POST-clip target-space quantiles at the ``mean_quantile_grid``
    levels are written to a sibling file ``...__quantiles.parquet`` next to each
    primary ``...__predictions.parquet`` -- same key columns, one float32
    ``q_{level}`` column per level (see :func:`quantile_col`). Post-clip is
    deliberate: the ``{acct}_t{h}`` labels live in the clipped +/-max_abs_z
    space, so coverage/CRPS of the clipped truth must be scored against the
    clipped predictive; sigma stays pre-clip as before. A sibling file (rather
    than widening the canonical schema) keeps the ``FORECAST_SCHEMA_COLS``
    contract and every existing eval read untouched. Quantiles are written as
    emitted by the pipeline -- any quantile crossing is left to the scoring
    script to sort, so the file is an honest record of the model output.

    Requires the Chronos-2 pipeline (list-of-groups API + NaN masking); the
    classic Bolt fallback in ``_forecast_batch`` would not handle NaN contexts.
    """

    def __init__(
        self,
        processed_dir: Optional[str] = None,
        dataset_suffix: Optional[str] = None,
        firm_id_map_path: str = "data/processed/firm_id_map.csv",
        account_id_map_path: str = "data/processed/account_id_map.csv",
        point_stat: str = "mean",
        write_median_sibling: bool = True,
        save_quantiles: bool = False,
        mean_quantile_grid: Optional[List[float]] = None,
        max_abs_zscore: float = 6.0,
        max_context: Optional[int] = 12,
        sigma_method: str = "grid",
        model_name: str = "chronos_raw",
        **kwargs,
    ):
        super().__init__(max_context=max_context, model_name=model_name, **kwargs)
        if point_stat not in ("mean", "median"):
            raise ValueError(f"point_stat must be 'mean' or 'median', got {point_stat!r}")
        if sigma_method not in ("grid", "pair"):
            raise ValueError(f"sigma_method must be 'grid' or 'pair', got {sigma_method!r}")
        self.processed_dir = processed_dir
        self.dataset_suffix = dataset_suffix
        self.firm_id_map_path = firm_id_map_path
        self.account_id_map_path = account_id_map_path
        self.point_stat = point_stat
        self.write_median_sibling = bool(write_median_sibling)
        self.save_quantiles = bool(save_quantiles)
        self.mean_quantile_grid = sorted(mean_quantile_grid or DEFAULT_MEAN_QUANTILE_GRID)
        self.max_abs_zscore = float(max_abs_zscore)
        # "grid": moment-matched sigma from the full quantile grid,
        #   sigma^2 = E[X^2] - E[X]^2 with the same trapezoid weights as the mean
        #   (pre-clip; tails beyond the extreme levels treated flat -> conservative).
        # "pair": Gaussian-equivalent sigma from the sigma_quantile_pair spread
        #   (the v1 convention; uses only 2 of the 21 levels).
        self.sigma_method = sigma_method

        qset = set(self._request_quantiles)
        qset.update(self.mean_quantile_grid)
        self._request_quantiles = sorted(qset)

    # ------------------------------------------------------------------ #
    # fit (zero-shot: metadata only; raw history is read from tuple files)
    # ------------------------------------------------------------------ #
    def fit(self, train_data, validation_data: Optional[pd.DataFrame] = None) -> None:
        if not hasattr(train_data, "get_train_data"):
            raise ValueError(
                "ChronosRawTSModel requires the TabularDataModule (for target metadata "
                "and the dataset suffix); got a bare DataFrame."
            )
        dm = train_data
        self.feature_set = getattr(dm, "feature_set", self.feature_set)
        self.target_variables = [
            t for t in (getattr(dm, "target_variables", []) or []) if t != "scale"
        ]
        self.target_columns = list(getattr(dm, "target_columns", []) or [])

        horizons = []
        for c in self.target_columns:
            if "_t" in c:
                try:
                    horizons.append(int(c.rsplit("_t", 1)[1]))
                except ValueError:
                    pass
        if horizons:
            self.forecast_horizon = max(horizons)

        if self.processed_dir is None:
            self.processed_dir = str(getattr(dm, "data_dir", "data/processed"))
        if self.dataset_suffix is None:
            tag = getattr(dm, "dataset_tag", None)
            self.dataset_suffix = f"{self.feature_set}__{tag}" if tag else self.feature_set

        self.is_fitted = True
        print(
            f"{self.model_name}: zero-shot fit complete -- raw-tuple contexts from "
            f"'{self.processed_dir}/tuple_*__{self.dataset_suffix}.parquet', "
            f"{len(self.target_variables)} target accounts, max_context={self.max_context}, "
            f"forecast_horizon={self.forecast_horizon}, point_stat={self.point_stat}."
        )

    # ------------------------------------------------------------------ #
    # Raw-data loading helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _q_int(quarters: pd.Series) -> np.ndarray:
        """Datetime quarters -> the tuple pipeline's integer quarter index
        (quarters since base_quarter 1969-12-31: 1970Q1 -> 1)."""
        dt = pd.DatetimeIndex(quarters)
        return ((dt.year - 1970) * 4 + dt.quarter).to_numpy(dtype=np.int64)

    def _load_id_maps(self) -> Tuple[Dict[int, int], Dict[str, int]]:
        """Return (gvkey_int -> firm_id_int, account_name -> account_id)."""
        fm = pd.read_csv(self.firm_id_map_path, dtype={"firm_id": str})
        fm = fm.drop_duplicates(subset=["firm_id"], keep="first")
        firm_map = {
            int(g): int(i) for g, i in zip(fm["firm_id"], fm["firm_id_int"])
        }
        am = pd.read_csv(self.account_id_map_path)
        am = am.drop_duplicates(subset=["account_name"], keep="first")
        acct_map = {str(n): int(i) for n, i in zip(am["account_name"], am["account_id"])}
        return firm_map, acct_map

    def _load_reg_stat_arrays(
        self, accts: List[str], n_quarters: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-account origin-frozen transform parameters.

        Returns (k[A], mu[A, n_quarters], sigma[A, n_quarters]) indexed by the
        integer quarter; NaN where a (account, quarter) has no stats row --
        exactly the cells where dataset_creation.py leaves the labels NaN.
        """
        path = Path(self.processed_dir) / f"regularization_stats__{self.dataset_suffix}.parquet"
        stats = pd.read_parquet(path)
        stats = stats[stats["feature"].isin(accts)]
        qi = self._q_int(stats["quarter"])

        A = len(accts)
        idx = {a: i for i, a in enumerate(accts)}
        k_arr = np.full(A, np.nan)
        mu_arr = np.full((A, n_quarters), np.nan)
        sig_arr = np.full((A, n_quarters), np.nan)
        rows = stats["feature"].map(idx).to_numpy()
        ok = (qi >= 0) & (qi < n_quarters)
        mu_arr[rows[ok], qi[ok]] = stats["mu"].to_numpy()[ok]
        sig_arr[rows[ok], qi[ok]] = stats["sigma"].to_numpy()[ok]
        for a, g in stats.groupby("feature"):
            k_arr[idx[a]] = float(g["k"].iloc[0])
        return k_arr, mu_arr, sig_arr

    def _load_raw_panels(
        self, firm_ints: set, acct_ids: List[int], scale_id: int
    ) -> Dict[int, Tuple[int, np.ndarray, np.ndarray]]:
        """Read the three tuple splits and build per-firm dense raw panels.

        Returns {firm_id_int: (qmin, M, scale_arr)} where ``M`` is
        ``(n_accounts, L)`` float32 raw values on the contiguous integer-quarter
        grid ``qmin..qmin+L-1`` (NaN where unobserved), row order matching
        ``acct_ids``, and ``scale_arr`` is the raw ``scale`` account on the same
        grid. Splits overlap on context quarters -> first occurrence wins.
        """
        keep_ids = set(acct_ids) | {scale_id}
        frames = []
        for split in ("train", "val", "test"):
            path = Path(self.processed_dir) / f"tuple_{split}__{self.dataset_suffix}.parquet"
            if not path.exists():
                # Hard-fail: silently skipping a split would truncate every firm's
                # context to the remaining splits' quarters (e.g. no pre-2010
                # history without tuple_train/val) while still writing a
                # complete-looking forecast file.
                raise FileNotFoundError(
                    f"{self.model_name}: required tuple split missing: {path}. "
                    f"All three tuple_{{train,val,test}}__{self.dataset_suffix} parquets "
                    f"must be present (contexts span split boundaries)."
                )
            df = pd.read_parquet(path, columns=["firm_id", "account_id", "quarter", "value"])
            df = df[df["account_id"].isin(keep_ids) & df["firm_id"].isin(firm_ints)]
            frames.append(df)
        tup = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(["firm_id", "account_id", "quarter"], keep="first")
        )

        row_of = {aid: i for i, aid in enumerate(acct_ids)}
        A = len(acct_ids)
        panels: Dict[int, Tuple[int, np.ndarray, np.ndarray]] = {}
        for fid, g in tup.groupby("firm_id", sort=False):
            q = g["quarter"].to_numpy(dtype=np.int64)
            qmin, qmax = int(q.min()), int(q.max())
            L = qmax - qmin + 1
            M = np.full((A, L), np.nan, dtype=np.float32)
            scale_arr = np.full(L, np.nan, dtype=np.float32)
            aid = g["account_id"].to_numpy(dtype=np.int64)
            val = g["value"].to_numpy(dtype=np.float32)
            col = q - qmin
            is_scale = aid == scale_id
            scale_arr[col[is_scale]] = val[is_scale]
            tgt = ~is_scale
            rows = np.fromiter((row_of[a] for a in aid[tgt]), dtype=np.int64, count=int(tgt.sum()))
            M[rows, col[tgt]] = val[tgt]
            panels[int(fid)] = (qmin, M, scale_arr)
        return panels

    @staticmethod
    def _mean_weights(levels: List[float]) -> np.ndarray:
        """Trapezoid weights over quantile levels for E[T(X)] ~ sum w_i T(q_i).
        Tails beyond the extreme levels are treated as flat (weights normalized)."""
        u = np.asarray(levels, dtype=np.float64)
        lo = np.concatenate(([0.0], u[:-1]))
        hi = np.concatenate((u[1:], [1.0]))
        w = (hi - lo) / 2.0
        w[0] += u[0] / 2.0
        w[-1] += (1.0 - u[-1]) / 2.0
        return w / w.sum()

    # ------------------------------------------------------------------ #
    # Forecasting
    # ------------------------------------------------------------------ #
    def predict(self, test_data, stream_save_paths=None, stream_group_size=None) -> Optional[pd.DataFrame]:
        """Raw-context zero-shot forecast; outputs mapped into the origin-frozen
        target space. Streams primary (+ median sibling) forecast files when
        ``stream_save_paths`` is given, else returns the primary tall DataFrame."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.save_quantiles and not stream_save_paths:
            # The non-streaming return path goes through finalize_forecast_frame,
            # which drops non-canonical columns -- quantiles would be silently
            # lost. Fail before any panel/tuple I/O is spent.
            raise ValueError(
                "save_quantiles=True requires streaming mode (stream_save_paths); "
                "the non-streaming return frame is finalized to FORECAST_SCHEMA_COLS "
                "and cannot carry quantile columns."
            )

        test_df = test_data.get_test_data() if hasattr(test_data, "get_test_data") else test_data
        accts = [t for t in (self.target_variables or []) if t != "scale"]
        if not accts:
            raise ValueError("No target accounts captured at fit() time")

        firm_map, acct_map = self._load_id_maps()
        missing = [a for a in accts if a not in acct_map]
        if missing:
            print(f"  {self.model_name}: {len(missing)} target(s) missing from account map, dropped: {missing}")
            accts = [a for a in accts if a in acct_map]
        acct_ids = [acct_map[a] for a in accts]
        if "scale" not in acct_map:
            raise ValueError("account_id_map has no 'scale' account (needed for the target transform)")
        scale_id = acct_map["scale"]

        origins = test_df[["firm_id", "quarter"]].drop_duplicates()

        def _fid_int(s):
            # Unknown or non-numeric firm ids fall through to -1 (skipped as
            # no_panel) instead of aborting the whole predict on int().
            try:
                return firm_map.get(int(s), -1)
            except (TypeError, ValueError):
                return -1

        origin_fid_int = origins["firm_id"].astype(str).map(_fid_int).to_numpy(dtype=np.int64)
        origin_q_int = self._q_int(origins["quarter"])
        n_quarters = int(origin_q_int.max()) + 1

        print(f"\n{self.model_name}: loading raw tuple panels for "
              f"{len(set(origin_fid_int[origin_fid_int >= 0])):,} firms...")
        panels = self._load_raw_panels(set(origin_fid_int[origin_fid_int >= 0].tolist()), acct_ids, scale_id)
        k_arr, mu_arr, sig_arr = self._load_reg_stat_arrays(accts, n_quarters)

        z_lo, z_hi = _norm_ppf(self.sigma_quantile_pair[0]), _norm_ppf(self.sigma_quantile_pair[1])
        z_span = (z_hi - z_lo) or 1.0
        i_med = self._request_quantiles.index(0.5)
        i_lo = self._request_quantiles.index(self.sigma_quantile_pair[0])
        i_hi = self._request_quantiles.index(self.sigma_quantile_pair[1])
        grid_idx = np.array([self._request_quantiles.index(q) for q in self.mean_quantile_grid])
        grid_w = self._mean_weights(self.mean_quantile_grid)

        n_origins = len(origins)
        print(f"{self.model_name}: forecasting up to {len(accts)} accounts jointly x {n_origins:,} "
              f"firm-origins x {self.forecast_horizon} horizons "
              f"(zero-shot, raw context <= {self.max_context}q, origin-frozen output transform)")

        # Per-batch state: variable-size groups (accounts observed in each window).
        ctx_batch: List[np.ndarray] = []
        meta_batch: List[Tuple[str, object, np.ndarray, float, int]] = []  # (fid, qtr, sel_idx, scale0, q0)
        parts: List[pd.DataFrame] = []
        skipped = {"no_panel": 0, "no_scale": 0, "no_accounts": 0}
        # sigma-floor incidence diagnostic (single-element lists: mutated in the
        # _emit_batch closure), reported at end of run.
        n_cells_emitted, n_floor_cells = [0], [0]

        # Streaming setup: primary file (+ optional median sibling in one pass).
        # Writers are entered on an ExitStack below so an exception mid-stream
        # aborts the atomic publish (no truncated canonical parquet).
        write_primary, write_median, write_quant = None, None, None
        med_paths: List[Path] = []
        quant_paths: List[Path] = []
        if stream_save_paths:
            import pyarrow as pa
            from forma.models.forecast_io import atomic_parquet_writers, finalize_forecast_frame
            if self.write_median_sibling and self.point_stat == "mean":
                for p in stream_save_paths:
                    p = Path(p)
                    prefix = f"{self.model_name}__"
                    if not p.name.startswith(prefix):
                        raise ValueError(
                            f"Cannot derive median sibling path from {p.name!r} "
                            f"(expected prefix {prefix!r})"
                        )
                    med_paths.append(p.with_name(f"{self.model_name}_med__" + p.name[len(prefix):]))
            if self.save_quantiles:
                pred_suffix = "__predictions.parquet"
                for p in stream_save_paths:
                    p = Path(p)
                    if not p.name.endswith(pred_suffix):
                        raise ValueError(
                            f"Cannot derive quantile sibling path from {p.name!r} "
                            f"(expected suffix {pred_suffix!r})"
                        )
                    quant_paths.append(
                        p.with_name(p.name[: -len(pred_suffix)] + "__quantiles.parquet"))
        else:  # save_quantiles + no streaming already rejected at predict() entry
            pa = finalize_forecast_frame = None
        quant_cols = [quantile_col(q) for q in self.mean_quantile_grid]

        H = self.forecast_horizon

        def _emit_batch(results: List[np.ndarray]):
            """Map raw-space quantiles through T_Q, build tall rows for the batch."""
            prim_parts, med_parts, quant_parts = [], [], []
            for r, (fid, qtr, sel, scale0, q0) in zip(results, meta_batch):
                # r: (n_sel, H, Q) raw-space quantiles for this origin's group.
                k_s = (k_arr[sel] / scale0)[:, None, None]
                mu_s = mu_arr[sel, q0][:, None, None]
                sig_s = sig_arr[sel, q0][:, None, None]
                z_pre = (np.arcsinh(r * k_s) - mu_s) / sig_s
                z = np.clip(z_pre, -self.max_abs_zscore, self.max_abs_zscore)

                mean_pt = (z[:, :, grid_idx] * grid_w).sum(axis=2)
                med_pt = z[:, :, i_med]
                # sigma always from PRE-clip quantiles: post-clip, rail-saturated
                # quantiles would fake certainty. Points stay clipped to match the
                # label space (L2-optimal for a clipped label is E[clip(X)]).
                # sigma is written RAW (unfloored, zeros allowed): the floor is an
                # evaluation-time policy (evaluate.py --sigma-floor), so it can be
                # revisited without regenerating the forecast.
                if self.sigma_method == "grid":
                    # Moment-matched: sigma^2 = E[X^2] - E[X]^2 over the quantile
                    # grid. Uses all levels, so a flat 0.1-0.9 core with tail
                    # spread at 0.01/0.99 still yields a nonzero sigma.
                    zg = z_pre[:, :, grid_idx]
                    m1 = (zg * grid_w).sum(axis=2)
                    m2 = (zg * zg * grid_w).sum(axis=2)
                    sig_pt = np.sqrt(np.clip(m2 - m1 * m1, 0.0, None))
                else:
                    # max(., 0): guards a tiny negative spread if the pipeline's
                    # quantile ordering is not exact — numerical, not a floor.
                    sig_pt = np.clip((z_pre[:, :, i_hi] - z_pre[:, :, i_lo]) / z_span, 0.0, None)
                n_cells_emitted[0] += sig_pt.size
                n_floor_cells[0] += int((sig_pt <= 1e-4).sum())

                n_sel = len(sel)
                base = {
                    "firm_id": np.repeat(fid, n_sel * H),
                    "target": np.repeat(np.asarray([accts[i] for i in sel], dtype=object), H),
                    "quarter": np.repeat(np.asarray(qtr, dtype="datetime64[ns]"), n_sel * H),
                    "forecast_horizon": np.tile(np.arange(1, H + 1), n_sel).astype("int64"),
                }
                primary_pt = mean_pt if self.point_stat == "mean" else med_pt
                prim = dict(base, prediction=primary_pt.reshape(-1))
                if self.save_sigma:
                    prim["sigma"] = sig_pt.reshape(-1)
                prim["model"] = self.model_name
                prim_parts.append(pd.DataFrame(prim))
                if write_median is not None:
                    med = dict(base, prediction=med_pt.reshape(-1))
                    if self.save_sigma:
                        med["sigma"] = sig_pt.reshape(-1)
                    med["model"] = f"{self.model_name}_med"
                    med_parts.append(pd.DataFrame(med))
                if write_quant is not None:
                    # POST-clip label-space quantiles (see class docstring), one
                    # float32 column per grid level. z is (n_sel, H, Q_requested);
                    # z[:, :, grid_idx[j]].reshape(-1) flattens (n_sel, H)
                    # row-major -- the same account-major/horizon-minor order as
                    # base's repeat/tile keys and prediction's reshape(-1).
                    quant = dict(base)
                    for j, col in enumerate(quant_cols):
                        quant[col] = z[:, :, grid_idx[j]].astype(np.float32).reshape(-1)
                    quant["model"] = self.model_name
                    quant_parts.append(pd.DataFrame(quant))

            prim_df = pd.concat(prim_parts, ignore_index=True)
            if write_primary is not None:
                write_primary(pa.Table.from_pandas(finalize_forecast_frame(prim_df), preserve_index=False))
                if write_median is not None:
                    med_df = pd.concat(med_parts, ignore_index=True)
                    write_median(pa.Table.from_pandas(finalize_forecast_frame(med_df), preserve_index=False))
                if write_quant is not None:
                    # NOT finalized: finalize_forecast_frame would drop the q_*
                    # columns. Keys are already the canonical dtypes and the q_*
                    # payload is cast float32 above.
                    quant_df = pd.concat(quant_parts, ignore_index=True)
                    write_quant(pa.Table.from_pandas(quant_df, preserve_index=False))
            else:
                parts.append(prim_df)

        def _run_batch():
            if not ctx_batch:
                return
            results = self._forecast_batch(ctx_batch)
            _emit_batch(results)
            ctx_batch.clear()
            meta_batch.clear()

        n_done = 0
        from contextlib import ExitStack
        with ExitStack() as stack:
            if stream_save_paths:
                write_primary = stack.enter_context(atomic_parquet_writers(list(stream_save_paths)))
                if med_paths:
                    write_median = stack.enter_context(atomic_parquet_writers(med_paths))
                    print(f"  {self.model_name}: median sibling -> {med_paths[0].name}")
                if quant_paths:
                    write_quant = stack.enter_context(atomic_parquet_writers(quant_paths))
                    print(f"  {self.model_name}: quantile sibling ({len(quant_cols)} levels) "
                          f"-> {quant_paths[0].name}")
            fids = origins["firm_id"].to_numpy()
            qtrs = origins["quarter"].to_numpy()
            order = np.argsort(origin_fid_int, kind="stable")  # group per-firm work together
            for i in order:
                panel = panels.get(int(origin_fid_int[i]))
                if panel is None:
                    skipped["no_panel"] += 1
                    continue
                qmin, M, scale_arr = panel
                jc = int(origin_q_int[i]) - qmin
                if jc < 0 or jc >= M.shape[1]:
                    skipped["no_panel"] += 1
                    continue
                scale0 = float(scale_arr[jc])
                if not np.isfinite(scale0) or scale0 <= 0:
                    skipped["no_scale"] += 1
                    continue
                lo_c = max(0, jc - (self.max_context or jc + 1) + 1)
                ctx = M[:, lo_c: jc + 1]
                q0 = int(origin_q_int[i])
                sel = np.where(
                    np.isfinite(ctx).any(axis=1)
                    & np.isfinite(k_arr)
                    & np.isfinite(mu_arr[:, q0])
                    & np.isfinite(sig_arr[:, q0])
                )[0]
                if len(sel) == 0:
                    skipped["no_accounts"] += 1
                    continue
                ctx_batch.append(ctx[sel].astype(np.float32))
                meta_batch.append((fids[i], qtrs[i], sel, scale0, q0))
                if len(ctx_batch) >= self.batch_size:
                    _run_batch()
                    n_done += self.batch_size
                    if n_done % (self.batch_size * 20) == 0:
                        print(f"  {self.model_name}: {n_done:,}/{n_origins:,} origins forecast")
            _run_batch()

        if any(skipped.values()):
            print(f"  {self.model_name}: skipped origins -- {skipped} (of {n_origins:,}); "
                  f"skipped cells have NaN labels downstream, so no coverage loss vs the eval.")
        if n_cells_emitted[0]:
            print(f"  {self.model_name}: near-zero sigma (<=1e-4) incidence ({self.sigma_method}): "
                  f"{n_floor_cells[0]:,}/{n_cells_emitted[0]:,} cells "
                  f"({100.0 * n_floor_cells[0] / n_cells_emitted[0]:.3f}%). sigma is written raw "
                  f"(unfloored); the floor is evaluate.py's --sigma-floor policy.")

        if stream_save_paths:
            return None
        if not parts:
            from forma.models.forecast_io import FORECAST_SCHEMA_COLS
            return pd.DataFrame(columns=FORECAST_SCHEMA_COLS)
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------ #
    # Persistence (zero-shot: config only; raw history lives in tuple files)
    # ------------------------------------------------------------------ #
    def save(self, folder_path: str) -> None:
        p = Path(folder_path)
        p.mkdir(parents=True, exist_ok=True)
        cfg = {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "quantile_levels": self.quantile_levels,
            "sigma_quantile_pair": list(self.sigma_quantile_pair),
            "max_context": self.max_context,
            "batch_size": self.batch_size,
            "forecast_horizon": self.forecast_horizon,
            "feature_set": self.feature_set,
            "target_variables": self.target_variables,
            "target_columns": self.target_columns,
            "processed_dir": self.processed_dir,
            "dataset_suffix": self.dataset_suffix,
            "firm_id_map_path": self.firm_id_map_path,
            "account_id_map_path": self.account_id_map_path,
            "point_stat": self.point_stat,
            "write_median_sibling": self.write_median_sibling,
            "save_quantiles": self.save_quantiles,
            "mean_quantile_grid": self.mean_quantile_grid,
            "max_abs_zscore": self.max_abs_zscore,
            "sigma_method": self.sigma_method,
        }
        with open(p / "chronos_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"{self.model_name}: saved config to {p}")

    def load(self, folder_path: str) -> None:
        super().load(folder_path)
        cfg_path = Path(folder_path) / "chronos_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            self.processed_dir = cfg.get("processed_dir", self.processed_dir)
            self.dataset_suffix = cfg.get("dataset_suffix", self.dataset_suffix)
            self.firm_id_map_path = cfg.get("firm_id_map_path", self.firm_id_map_path)
            self.account_id_map_path = cfg.get("account_id_map_path", self.account_id_map_path)
            self.point_stat = cfg.get("point_stat", self.point_stat)
            self.write_median_sibling = cfg.get("write_median_sibling", self.write_median_sibling)
            self.save_quantiles = cfg.get("save_quantiles", self.save_quantiles)
            self.mean_quantile_grid = cfg.get("mean_quantile_grid", self.mean_quantile_grid)
            self.max_abs_zscore = cfg.get("max_abs_zscore", self.max_abs_zscore)
            self.sigma_method = cfg.get("sigma_method", self.sigma_method)
            qset = set(self._request_quantiles)
            qset.update(self.mean_quantile_grid)
            self._request_quantiles = sorted(qset)
