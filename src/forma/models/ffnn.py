"""Feed-forward neural network baseline with a probabilistic vector-output head.

`FeedForwardModel` is the cleanest single-model, parameter-shared foil to Forma.
Where the sklearn/XGBoost baselines train one independent model per
(target, horizon) pair (~1,560 disjoint models, each listwise-deleting rows whose
target is missing), this trains **one** network with a `len(target_columns)`-wide
output head and learns from ragged panels via a **masked loss**: a firm acquired
after 8 quarters contributes loss on its first 8 horizons and is *masked* on the
rest — it is not dropped.

To be a *fair* foil it mirrors Forma's training recipe as closely as a plain MLP
can, so the only remaining difference is architecture (MLP vs Transformer) rather
than the optimization/objective:

* **Optimizer / regularization:** AdamW with Forma's lr (1e-4), weight_decay (0.1),
  and dropout (0.2) defaults — decoupled decay, no LR schedule, no grad clip,
  matching `forma.py`'s `configure_optimizers`.
* **Probabilistic head + loss:** a two-headed output emits a mean `mu` and a
  log-variance `log_sigma_sq` per target, trained under the **same masked
  heteroscedastic NLL** Forma uses (`_masked_nll_reg_space` in `forma.py`):
  Gaussian by default, Student-t optional, with `log_sigma_sq` clamped to
  `[log_sigma_sq_min, log_sigma_sq_max]` for stability. `loss_type='mse'` recovers
  the older point-MSE objective (mean head only).

Design: it is a plain `BaseModel` running its own internal torch training loop (the
way `XGBoostModel` wraps XGBoost), so it slots into the existing sklearn/XGBoost
path in `train.py` with zero changes to the training harness and consumes the exact
same tabular parquet the other baselines do.

Data note: the tabular `_level_`/`_yoy_` feature columns and `_t{h}` target columns
are already regularized at build time (the same `regularize()` transform Forma
applies), so no input-standardization layer and no target transform are needed here
— the loss is computed directly in the regularized target space, and `evaluate.py`
de-regularizes the predicted mean back to raw space uniformly across all models.
The variance head shapes the NLL objective during training but is intentionally NOT
serialized: `predict()` emits the mean only, keeping the standard tall forecast
schema (a second high-entropy float column would roughly double the parquet and add
to evaluate.py's full-frame read).
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base import BaseModel
from .baselines import wide_forecasts_to_tall


_ACTIVATIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
}


def masked_mse_loss(pred: torch.Tensor, y: torch.Tensor,
                    normalization: str = 'global') -> torch.Tensor:
    """Mean squared error over present (non-NaN) target entries only.

    Args:
        pred: (B, T) predictions.
        y:    (B, T) targets, with NaN marking absent observations (delisted/
              acquired firm beyond its last horizon, or a target not present).
        normalization:
            'global'     -> sum of squared errors over present entries divided by
                            the total count of present entries (equal weight per
                            observation).
            'per_target' -> reduce each column by its own present-count, then
                            average across columns that have any observation (equal
                            weight per output). Keeps long-horizon targets — sparse
                            due to delisting attrition — from being down-weighted
                            relative to dense near-horizon ones.

    Because every target is z-scored to ~unit variance upstream, scale
    heterogeneity across the 1,560 outputs is already removed; this lever is purely
    "equal-weight-per-observation vs. equal-weight-per-output", not scale balancing.

    Masked entries contribute exactly zero to the loss, so gradients w.r.t. those
    output positions are zero.
    """
    mask = ~torch.isnan(y)
    mask_f = mask.to(pred.dtype)
    y0 = torch.nan_to_num(y, nan=0.0)
    se = (pred - y0) ** 2 * mask_f
    return _reduce_masked(se, mask_f, normalization)


def masked_nll_loss(mu: torch.Tensor, log_sigma_sq: torch.Tensor, y: torch.Tensor,
                    loss_type: str = 'gaussian', student_t_df: float = 6.0,
                    log_sigma_sq_min: float = -10.0, log_sigma_sq_max: float = 10.0,
                    normalization: str = 'global', beta_nll: float = 0.0) -> torch.Tensor:
    """Masked heteroscedastic negative log-likelihood over present entries only.

    Mirrors Forma's `_masked_nll_reg_space` (forma.py): the per-entry term is

        Gaussian:   0.5 * (y - mu)^2 / sigma^2 + 0.5 * log(sigma^2)
        Student-t:  0.5 * log(sigma^2) + ((nu+1)/2) * log(1 + (y-mu)^2 / (nu*sigma^2))

    (additive constants independent of the parameters are dropped, exactly as in
    Forma). `log_sigma_sq` is clamped to `[log_sigma_sq_min, log_sigma_sq_max]`
    before `exp` for numerical stability. NaN-target entries are masked to a zero
    contribution, so gradients w.r.t. both heads are zero at those positions.

    The `normalization` lever matches `masked_mse_loss`: 'global' (equal weight per
    present observation) or 'per_target' (equal weight per output column).
    """
    mask = ~torch.isnan(y)
    mask_f = mask.to(mu.dtype)
    y0 = torch.nan_to_num(y, nan=0.0)

    lss = log_sigma_sq.clamp(min=log_sigma_sq_min, max=log_sigma_sq_max)
    sigma_sq = torch.exp(lss)
    resid_sq = (y0 - mu) ** 2

    if loss_type == 'gaussian':
        per_entry = 0.5 * resid_sq / sigma_sq + 0.5 * lss
        # Optional beta-NLL (Seitzer et al., ICLR 2022): weight each entry by a
        # stop-grad sigma^(2*beta) (= sigma_sq^beta) so the 1/sigma^2 gradient no
        # longer down-weights high-variance (hard) entries. beta_nll=0.0 (default)
        # leaves per_entry untouched -> bit-identical to the standard Gaussian NLL.
        # Gaussian-only, mirroring Forma (forma.py:818-819).
        if beta_nll > 0.0:
            per_entry = per_entry * sigma_sq.detach().pow(beta_nll)
    elif loss_type == 'student_t':
        nu = float(student_t_df)
        per_entry = 0.5 * lss + 0.5 * (nu + 1.0) * torch.log1p(resid_sq / (nu * sigma_sq))
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Expected 'gaussian', 'student_t', or 'mse'."
        )

    per_entry = per_entry * mask_f
    return _reduce_masked(per_entry, mask_f, normalization)


def _reduce_masked(per_entry: torch.Tensor, mask_f: torch.Tensor,
                   normalization: str) -> torch.Tensor:
    """Shared masked reduction for the point and probabilistic losses."""
    if normalization == 'per_target':
        col_counts = mask_f.sum(dim=0)
        present_cols = col_counts > 0
        if not torch.any(present_cols):
            return per_entry.sum() * 0.0
        col_loss = per_entry.sum(dim=0)[present_cols] / col_counts[present_cols]
        return col_loss.mean()
    elif normalization == 'global':
        denom = mask_f.sum().clamp(min=1.0)
        return per_entry.sum() / denom
    else:
        raise ValueError(
            f"Unknown loss_normalization '{normalization}'. Expected 'global' or "
            f"'per_target'."
        )


class _MLP(nn.Module):
    """MLP trunk feeding parallel mean and log-variance output heads.

    The shared trunk is the hidden stack (Linear -> [norm] -> activation ->
    [dropout]) repeated over `hidden_dims`; two `Linear(prev, out_dim)` heads then
    produce the per-target mean and log-variance. The log-variance head is
    zero-initialized so `log_sigma_sq == 0` (sigma^2 == 1) at step 0 — the right
    prior given targets are z-scored to ~unit variance upstream, and it keeps the
    early NLL well-conditioned.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int],
                 activation: str = 'gelu', dropout: float = 0.0,
                 norm: Optional[str] = None):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Expected one of {sorted(_ACTIVATIONS)}."
            )
        act_layer = _ACTIVATIONS[activation]

        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if norm == 'batchnorm':
                layers.append(nn.BatchNorm1d(h))
            elif norm == 'layernorm':
                layers.append(nn.LayerNorm(h))
            elif norm not in (None, 'none'):
                raise ValueError(
                    f"Unknown norm '{norm}'. Expected None, 'batchnorm', or 'layernorm'."
                )
            layers.append(act_layer())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, out_dim)
        self.logvar_head = nn.Linear(prev, out_dim)

        # Start at log_sigma_sq == 0 (unit variance) so the NLL is well-posed before
        # the variance head has learned anything.
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mean_head(h), self.logvar_head(h)


class FeedForwardModel(BaseModel):
    """Single FFNN with mean+variance output heads over all (target, horizon) pairs."""

    def __init__(self,
                 hidden_dims: List[int] = None,
                 activation: str = 'gelu',
                 dropout: float = 0.2,
                 norm: Optional[str] = None,
                 lr: float = 1e-4,
                 weight_decay: float = 0.1,
                 epochs: int = 100,
                 batch_size: int = 1024,
                 early_stopping_patience: int = 10,
                 loss_normalization: str = 'global',
                 loss_type: str = 'gaussian',
                 beta_nll: float = 0.0,
                 student_t_df: float = 6.0,
                 log_sigma_sq_min: float = -10.0,
                 log_sigma_sq_max: float = 10.0,
                 seed: int = 42,
                 device: str = 'auto',
                 predict_batch_size: int = 16384,
                 assert_feature_range: bool = False,
                 save_sigma: bool = False,
                 **kwargs):
        super().__init__('ffnn', **kwargs)

        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [512, 512, 256]
        self.activation = activation
        self.dropout = float(dropout)
        self.norm = norm
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.early_stopping_patience = int(early_stopping_patience)
        self.loss_normalization = loss_normalization
        self.loss_type = loss_type
        self.beta_nll = float(beta_nll)
        self.student_t_df = float(student_t_df)
        self.log_sigma_sq_min = float(log_sigma_sq_min)
        self.log_sigma_sq_max = float(log_sigma_sq_max)
        self.seed = int(seed)
        self.predict_batch_size = int(predict_batch_size)
        self.assert_feature_range = bool(assert_feature_range)
        # Likelihood track: when True, predict() also serializes the predictive std
        # (`sigma`) so evaluate.py can score mean NLL. Off by default (point-only,
        # standard schema). Meaningless under loss_type='mse' (the variance head is
        # untrained), so it is force-disabled there with a warning.
        self.save_sigma = bool(save_sigma)

        if self.loss_type not in ('gaussian', 'student_t', 'mse'):
            raise ValueError(
                f"Unknown loss_type '{self.loss_type}'. Expected 'gaussian', "
                f"'student_t', or 'mse'."
            )

        if self.beta_nll < 0.0:
            raise ValueError(
                f"beta_nll must be >= 0 (got {self.beta_nll}); 0.0 is the "
                f"standard Gaussian NLL (default)."
            )
        if self.beta_nll > 0.0 and self.loss_type != 'gaussian':
            raise ValueError(
                f"beta_nll={self.beta_nll} is only supported for loss_type="
                f"'gaussian' (got '{self.loss_type}'), matching Forma."
            )

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.network: Optional[_MLP] = None
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []

    # ------------------------------------------------------------------ helpers

    def _build_network(self) -> _MLP:
        return _MLP(
            in_dim=len(self.feature_columns),
            out_dim=len(self.target_columns),
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            norm=self.norm,
        )

    def _compute_loss(self, mu: torch.Tensor, log_sigma_sq: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
        """Dispatch to the configured objective (probabilistic NLL or point MSE)."""
        if self.loss_type == 'mse':
            return masked_mse_loss(mu, y, self.loss_normalization)
        return masked_nll_loss(
            mu, log_sigma_sq, y,
            loss_type=self.loss_type,
            student_t_df=self.student_t_df,
            log_sigma_sq_min=self.log_sigma_sq_min,
            log_sigma_sq_max=self.log_sigma_sq_max,
            normalization=self.loss_normalization,
            beta_nll=self.beta_nll,
        )

    def _resolve_columns(self, train_data, train_df, validation_data):
        """Mirror baselines' metadata handling: prefer data-module column metadata,
        else auto-detect target columns and treat the remainder as features.

        Sets self.feature_columns / self.target_columns and returns val_df.
        """
        if hasattr(train_data, 'get_train_data'):
            val_df = train_data.get_val_data() if validation_data is None else validation_data
            target_variables = getattr(train_data, 'target_variables', None)

            feature_cols = getattr(train_data, 'feature_columns', [])
            target_cols = getattr(train_data, 'target_columns', [])
            if feature_cols and target_cols:
                self.feature_columns = list(feature_cols)
                self.target_columns = list(target_cols)
            else:
                self.target_columns = self._extract_target_columns(train_df, target_variables)
                self.feature_columns = [c for c in train_df.columns if c not in self.target_columns]
                print("Warning: Using auto-detection for feature/target columns. "
                      "Metadata columns may be included as features.")
        else:
            val_df = validation_data
            self.target_columns = self._extract_target_columns(train_df, None)
            self.feature_columns = [c for c in train_df.columns if c not in self.target_columns]
            print("Warning: Using auto-detection for feature/target columns. "
                  "Metadata columns may be included as features.")
        return val_df

    @staticmethod
    def _extract_target_columns(data_df: pd.DataFrame, target_variables) -> List[str]:
        """Auto-detect target columns (parity with MultiHorizonBaselineModel)."""
        if target_variables is None:
            target_cols = [c for c in data_df.columns
                           if '_t' in c and c.split('_t')[-1].replace('-', '').isdigit()]
        else:
            target_cols = []
            for target_var in target_variables:
                target_cols.extend([c for c in data_df.columns if c.startswith(f"{target_var}_t")])
        return sorted(target_cols)

    def _to_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Slice feature columns, coerce to numeric float32 (parity with baselines)."""
        X = df[self.feature_columns].copy()
        non_numeric = X.columns.difference(
            X.select_dtypes(include=[np.number]).columns
        ).tolist()
        for col in non_numeric:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        return X.astype('float32').to_numpy()

    # ----------------------------------------------------------------------- fit

    def fit(self, train_data, validation_data: Optional[pd.DataFrame] = None) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Resolve train/val frames and column metadata.
        if hasattr(train_data, 'get_train_data'):
            train_df = train_data.get_train_data()
        else:
            train_df = train_data
        val_df = self._resolve_columns(train_data, train_df, validation_data)

        # Recognize the no_cv config option (set in data.no_cv or via the --no-cv
        # CLI flag, surfaced on the data module's data_config). The
        # TabularDataModule already folds val into train_data when no_cv is on, so
        # get_train_data() returns the combined train+val frame; here we
        # additionally disable validation, early stopping, and best-epoch selection
        # so the FFNN mirrors Forma's no_cv behavior — fit on the combined
        # train+val for the full epoch budget, then predict on test with the
        # final-epoch weights (no cross-validation / no epoch selection).
        data_cfg = getattr(train_data, 'data_config', None) or {}
        no_cv = bool(data_cfg.get('no_cv', False))

        if len(self.target_columns) == 0:
            raise ValueError(
                "No target columns found in training data. Expected columns like "
                "'niq_t1', 'niq_t2', etc."
            )

        print(f"\n{self.model_name}: Training one network with a "
              f"{len(self.target_columns)}-wide mean+variance head on "
              f"{len(self.feature_columns)} features (loss={self.loss_type})")

        # Build full (N, T) feature/target matrices. KEEP rows with NaN targets
        # (the masked-loss point); drop only rows with NaN in any *feature*, since
        # the forward pass needs a dense input. Features are imputed upstream, so
        # this drop should be near-empty — kept for safety/parity with predict().
        X_train = self._to_feature_matrix(train_df)
        Y_train = train_df[self.target_columns].astype('float32').to_numpy()
        X_train, Y_train = self._drop_nan_feature_rows(X_train, Y_train, split='train')

        if self.assert_feature_range:
            # Features are clipped at build time to ±max_abs_zscore (level/lead
            # columns), and the yoy-difference columns to ±max_abs_zscore·√2 (the
            # difference of two independent z-scores; see dataset_creation.py:253).
            # √2 is therefore the loosest valid bound across all feature columns, so
            # derive it from the same config rather than hard-coding a value that
            # would falsely raise whenever max_abs_zscore != ~3.5.
            fe_cfg = data_cfg.get('feature_engineering') or {}
            max_abs_z = float(fe_cfg.get('max_abs_zscore', 5.0))
            bound = max_abs_z * 1.4142
            finite = X_train[np.isfinite(X_train)]
            if finite.size and np.abs(finite).max() > bound + 1e-3:
                raise AssertionError(
                    f"Train features exceed the expected regularized clip of "
                    f"±{bound:.3f} (max_abs_zscore={max_abs_z} ×√2 for yoy columns) "
                    f"(max abs = {np.abs(finite).max():.3f}). Tabular features should "
                    f"already be regularized; check the dataset build."
                )

        has_val = (not no_cv) and val_df is not None and len(val_df) > 0
        if no_cv:
            print(f"{self.model_name}: NO-CV MODE - fitting on the combined train+val "
                  f"set for the full {self.epochs} epochs; validation, early stopping, "
                  f"and best-epoch selection are disabled (final-epoch weights kept), "
                  f"mirroring Forma's no_cv behavior.")
        if has_val:
            X_val = self._to_feature_matrix(val_df)
            Y_val = val_df[self.target_columns].astype('float32').to_numpy()
            X_val, Y_val = self._drop_nan_feature_rows(X_val, Y_val, split='val')

        self.network = self._build_network().to(self.device)
        n_params = sum(p.numel() for p in self.network.parameters())
        print(f"{self.model_name}: Network has {n_params:,} parameters, "
              f"device={self.device}, optimizer=AdamW(lr={self.lr}, wd={self.weight_decay}), "
              f"dropout={self.dropout}, loss_normalization={self.loss_normalization}")

        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_train_t = torch.from_numpy(X_train)
        Y_train_t = torch.from_numpy(Y_train)
        dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # drop_last only when batchnorm is active: nn.BatchNorm1d throws in train
        # mode on a final batch of size 1. norm defaults to None (no batchnorm), so
        # this drops nothing in the canonical config and only guards a `norm`
        # sweep; dropping at most one partial batch per epoch is harmless there.
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator,
            drop_last=(self.norm == 'batchnorm'))

        if has_val:
            X_val_t = torch.from_numpy(X_val).to(self.device)
            Y_val_t = torch.from_numpy(Y_val).to(self.device)

        best_val = float('inf')
        best_state = None
        best_epoch = None
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.network.train()
            running, n_batches = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                mu, log_sigma_sq = self.network(xb)
                loss = self._compute_loss(mu, log_sigma_sq, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.detach())
                n_batches += 1
            train_loss = running / max(n_batches, 1)

            if has_val:
                self.network.eval()
                with torch.no_grad():
                    val_mu, val_logvar = self._forward_in_chunks(X_val_t)
                    val_loss = float(self._compute_loss(val_mu, val_logvar, Y_val_t))
                improved = val_loss < best_val - 1e-6
                if improved:
                    best_val = val_loss
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.network.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                print(f"  [{self.model_name}] epoch {epoch:3d}/{self.epochs}  "
                      f"train={train_loss:.6f}  val={val_loss:.6f}  "
                      f"best={best_val:.6f}  patience={epochs_no_improve}/{self.early_stopping_patience}")
                if epochs_no_improve >= self.early_stopping_patience:
                    print(f"  [{self.model_name}] early stopping at epoch {epoch} "
                          f"(best val={best_val:.6f})")
                    break
            else:
                print(f"  [{self.model_name}] epoch {epoch:3d}/{self.epochs}  "
                      f"train={train_loss:.6f}  (no validation; using final weights)")

        if best_state is not None:
            self.network.load_state_dict(best_state)
            print(f"{self.model_name}: Restored best-val-loss weights "
                  f"(epoch {best_epoch}, val={best_val:.6f})")

        # Epoch-transfer guidance for the no_cv benchmark (e.g. r9e3 -> r9e4).
        # Recommend the RAW val-optimal epoch count (number of passes), NOT a
        # step-matched rescale. Rationale: best_epoch is where val loss bottomed —
        # an overfitting boundary, not a fixed optimization state. That boundary
        # shifts LATER (in gradient steps) as the training set grows, so matching
        # e3's step count on the larger combined train+val set (which is what
        # best_epoch * n_train/(n_train+n_val) does) stops early and under-trains the
        # benchmark — especially here, where val >= train makes that rescale drop
        # more than half the epochs. The raw epoch count transfers the number of
        # passes, the more robust quantity; the step-matched value is reported only
        # as a lower-bound reference.
        if has_val and best_epoch is not None:
            n_tr, n_val = len(X_train), len(X_val)
            step_matched = max(1, round(best_epoch * n_tr / (n_tr + n_val)))
            print(f"{self.model_name}: best val epoch = {best_epoch} on {n_tr:,} "
                  f"train rows. For the no_cv benchmark on combined train+val "
                  f"({n_tr + n_val:,} rows), set epochs = {best_epoch} (the val-optimal "
                  f"number of passes). NOTE: step-matching would suggest only "
                  f"~{step_matched} (= {best_epoch} x {n_tr}/{n_tr + n_val}), but that "
                  f"under-trains — the val minimum is an overfitting boundary that "
                  f"shifts later with more training data, so use the raw epoch count.")

        self.is_fitted = True
        print(f"{self.model_name}: Training completed")

    @staticmethod
    def _drop_nan_feature_rows(X: np.ndarray, Y: np.ndarray, split: str):
        valid = ~np.isnan(X).any(axis=1)
        n_clean, n_orig = int(valid.sum()), len(X)
        if n_clean < n_orig:
            print(f"  {split}: dropped {n_orig - n_clean}/{n_orig} rows with NaN features "
                  f"(targets with NaN are KEPT and masked)")
        return X[valid], Y[valid]

    # ------------------------------------------------------------------- predict

    def _forward_in_chunks(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over a (possibly large) tensor in row-chunks to bound memory.

        Returns (mu, log_sigma_sq), each (N, T).
        """
        mus, logvars = [], []
        for start in range(0, X.size(0), self.predict_batch_size):
            mu, log_sigma_sq = self.network(X[start:start + self.predict_batch_size])
            mus.append(mu)
            logvars.append(log_sigma_sq)
        if not mus:
            return self.network(X)
        return torch.cat(mus, dim=0), torch.cat(logvars, dim=0)

    def predict(self, test_data, stream_save_paths=None, stream_group_size=4) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if hasattr(test_data, 'get_test_data'):
            test_df = test_data.get_test_data()
        else:
            test_df = test_data

        X_test = self._to_feature_matrix(test_df)

        # Drop rows with NaN in any feature (same valid_mask logic as baselines).
        valid_mask = ~np.isnan(X_test).any(axis=1)
        n_clean, n_orig = int(valid_mask.sum()), len(X_test)
        print(f"\n{self.model_name}: Predicting on {n_clean}/{n_orig} test samples "
              f"(after dropping NaN features)")
        if n_clean == 0:
            raise ValueError("No valid test samples remaining after dropping NaN values")

        test_df_clean = test_df.loc[valid_mask]
        X_clean = torch.from_numpy(X_test[valid_mask]).to(self.device)

        # The variance head shaped the NLL objective. For the point (MSE) track its
        # output is dropped — only the mean is serialized, keeping the standard tall
        # schema. For the likelihood track (save_sigma), we additionally serialize the
        # predictive std so evaluate.py can score mean NLL. Under loss_type='mse' the
        # head is untrained, so sigma would be meaningless: force-disable + warn.
        emit_sigma = self.save_sigma and self.loss_type != 'mse'
        if self.save_sigma and self.loss_type == 'mse':
            print(f"{self.model_name}: save_sigma ignored under loss_type='mse' "
                  f"(variance head is untrained; no meaningful sigma to emit).")

        self.network.eval()
        with torch.no_grad():
            mu, log_sigma_sq = self._forward_in_chunks(X_clean)
            mu = mu.detach().cpu().to(torch.float32).numpy()
            sigma = None
            if emit_sigma:
                # predictive std = exp(0.5 * log_sigma_sq), clamped to the training
                # bounds so it matches the fitted variance.
                lss = log_sigma_sq.clamp(self.log_sigma_sq_min, self.log_sigma_sq_max)
                sigma = torch.exp(0.5 * lss).detach().cpu().to(torch.float32).numpy()

        # Wide frame of means -> standardized tall schema (mean is the point forecast
        # `evaluate.py` consumes); sigma melts alongside it on the likelihood track.
        forecast_wide = pd.DataFrame(mu, columns=self.target_columns, index=test_df_clean.index)
        forecast_wide['firm_id'] = test_df_clean['firm_id'].values
        forecast_wide['quarter'] = test_df_clean['quarter'].values
        forecast_wide['model'] = self.model_name
        sigma_wide = None
        if emit_sigma:
            sigma_wide = pd.DataFrame(sigma, columns=self.target_columns, index=test_df_clean.index)

        # Streaming mode: melt + write the tall forecast to disk in target-column
        # groups so peak memory stays bounded by ~one group, instead of one giant
        # 78x20 wide->tall melt (that in-memory melt peaked ~137 GB host RAM and bit
        # the r9e3/e4 FFNN forecast-save). train.py:1171 always passes
        # stream_save_paths now (per-exp path + global "latest"); this mirrors
        # MultiHorizonBaselineModel.predict so FFNN honors the same save contract.
        # Returns None when streaming (the rows are written, not returned).
        if stream_save_paths:
            import pyarrow as pa
            from .forecast_io import finalize_forecast_frame, atomic_parquet_writers
            cols = self.target_columns
            # group_cols spans stream_group_size *targets* (the *20 is a memory-
            # sizing heuristic assuming ~20 horizons, NOT a correctness assumption --
            # each target column melts independently). Matches baselines.py.
            group_cols = max(1, stream_group_size) * 20
            with atomic_parquet_writers(stream_save_paths) as write_table:
                for start in range(0, len(cols), group_cols):
                    group = cols[start:start + group_cols]
                    # sigma_wide shares forecast_wide's index + target_columns, so
                    # melting it on the same `group` value_vars aligns positionally.
                    tall = wide_forecasts_to_tall(forecast_wide, group, sigma_wide=sigma_wide)
                    tall = finalize_forecast_frame(tall)
                    tbl = pa.Table.from_pandas(tall, preserve_index=False)
                    write_table(tbl)
                    del tall, tbl
                    print(f"  {self.model_name}: streamed cols {start + 1}-{start + len(group)}/{len(cols)}")
            return None

        return wide_forecasts_to_tall(forecast_wide, self.target_columns, sigma_wide=sigma_wide)

    # ---------------------------------------------------------------- save/load

    def save(self, folder_path: str) -> None:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        os.makedirs(folder_path, exist_ok=True)

        state_path = os.path.join(folder_path, f"{self.model_name}_state.pt")
        torch.save(self.network.state_dict(), state_path)

        metadata = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'architecture': {
                'hidden_dims': self.hidden_dims,
                'activation': self.activation,
                'dropout': self.dropout,
                'norm': self.norm,
            },
            'training': {
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'early_stopping_patience': self.early_stopping_patience,
                'loss_normalization': self.loss_normalization,
                'loss_type': self.loss_type,
                'student_t_df': self.student_t_df,
                'log_sigma_sq_min': self.log_sigma_sq_min,
                'log_sigma_sq_max': self.log_sigma_sq_max,
                'seed': self.seed,
            },
        }
        metadata_path = os.path.join(folder_path, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"{self.model_name}: Saved network + metadata to {folder_path}")

    def load(self, folder_path: str) -> None:
        metadata_path = os.path.join(folder_path, f"{self.model_name}_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.feature_columns = metadata['feature_columns']
        self.target_columns = metadata['target_columns']
        arch = metadata.get('architecture', {})
        self.hidden_dims = arch.get('hidden_dims', self.hidden_dims)
        self.activation = arch.get('activation', self.activation)
        self.dropout = arch.get('dropout', self.dropout)
        self.norm = arch.get('norm', self.norm)
        train_meta = metadata.get('training', {})
        self.loss_normalization = train_meta.get('loss_normalization', self.loss_normalization)
        self.loss_type = train_meta.get('loss_type', self.loss_type)
        self.student_t_df = float(train_meta.get('student_t_df', self.student_t_df))
        self.log_sigma_sq_min = float(train_meta.get('log_sigma_sq_min', self.log_sigma_sq_min))
        self.log_sigma_sq_max = float(train_meta.get('log_sigma_sq_max', self.log_sigma_sq_max))

        self.network = self._build_network().to(self.device)
        state_path = os.path.join(folder_path, f"{self.model_name}_state.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"State file not found: {state_path}")
        state = torch.load(state_path, map_location=self.device)
        try:
            self.network.load_state_dict(state)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load FFNN weights from {state_path}: {e}\n"
                "This usually means the checkpoint predates the probabilistic "
                "two-head FFNN (trunk + mean_head + logvar_head). Old single-head "
                "checkpoints are not compatible — retrain under the current recipe "
                "rather than reusing a pre-PR checkpoint."
            ) from e

        self.is_fitted = True
        print(f"{self.model_name}: Loaded network + metadata from {folder_path}")
