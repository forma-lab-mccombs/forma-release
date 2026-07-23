import torch

# --- PATCH START: Fix for NVIDIA NGC 24.02 container ---
# The PyTorch 2.3 alpha in this container is missing 'is_compiling', 
# so we manually define it to always return False.
if hasattr(torch, "compiler") and not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False
# --- PATCH END ---

"""
Master training script for all models.

This script handles training of all model types (baseline and GNN-Transformer)
using configuration files and walk-forward validation.
"""

import argparse
import json
import yaml
from pathlib import Path
import pandas as pd
from typing import Dict, Any
import sys
import os
import torch
from tqdm import tqdm
from contextlib import nullcontext
import numpy as np
import time
import shutil
from datetime import datetime
import subprocess

torch.set_float32_matmul_precision('medium')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from forma.models.base import BaseModel
# Competitor model classes (SklearnBaselineModel / XGBoostModel / FeedForwardModel /
# NaiveModel / NaiveDriftModel) are imported LAZILY inside create_model so that a
# Forma-only inference install never pulls in scikit-learn / xgboost. Only the
# Forma model + the shared data/forecast plumbing are imported eagerly here.
from forma.models.forecast_io import (
    FORECAST_SCHEMA_COLS,
    FORECAST_PARQUET_COMPRESSION,
    finalize_forecast_frame,
)
from forma.models.forma import FormaModel
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from forma.data.tabular_data import TabularDataModule, NaiveDriftDataModule
from forma.data.forma_data import FormaDataModule
from forma.data.transforms import de_regularize, regularize, get_derivative_s


class CurriculumCallback(L.Callback):
    """
    PyTorch Lightning callback to step the curriculum in FormaDataModule.

    Calls data_module.step_curriculum(current_epoch) at the end of each epoch,
    which increases the horizon every `epochs_per_step` epochs.

    The trainer is configured with reload_dataloaders_every_n_epochs=1, so the
    train DataLoader is automatically recreated each epoch with the updated horizon.
    """

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Step curriculum at the end of each training epoch."""
        data_module = trainer.datamodule
        if hasattr(data_module, 'step_curriculum'):
            current_epoch = trainer.current_epoch
            stepped = data_module.step_curriculum(current_epoch)
            if stepped:
                print(f"[CurriculumCallback] Horizon increased to {data_module.current_horizon} at epoch {current_epoch}")
                print(f"[CurriculumCallback] DataLoader will be recreated automatically on next epoch")


class WeightEMACallback(L.Callback):
    """Exponential moving average of model weights (opt-in training recipe).

    Maintains a shadow copy of the model's float parameters, updated after every
    optimizer step as `shadow = decay*shadow + (1-decay)*param`. EMA weights are
    swapped IN at validation START and held through validation END, then restored
    to the raw weights at the START of the next training batch — i.e. EMA is
    "present" for the whole validation window. This is deliberate: Lightning's
    `_CallbackConnector` unconditionally REORDERS all `Checkpoint` callbacks to the
    end of the list (so appending this callback last does NOT make its
    on_validation_end run after the checkpoint save). By restoring on the next
    train batch instead of at validation end, a `ModelCheckpoint` that saves at
    validation end captures the EMA weights, and so does the monitored `val_loss`.
    At train end EMA is baked into the model so the final ckpt + forecasts use it.

    Default-off: only constructed when trainer.weight_ema_decay is set and > 0, so
    the canonical training path never instantiates it.

    Resume caveat: `shadow` is NOT part of the callback `state_dict`, so resuming a
    run from a checkpoint restarts the EMA from the resumed (raw) weights rather
    than continuing the average — this is a fresh-run helper, not resumable EMA.
    """

    def __init__(self, decay: float):
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"weight_ema_decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow = None      # dict[name -> EMA tensor]
        self._backup = None     # dict[name -> raw tensor] while EMA is swapped in

    def _float_params(self, pl_module):
        return [(n, p) for n, p in pl_module.named_parameters() if p.dtype.is_floating_point]

    def on_train_start(self, trainer, pl_module):
        if self.shadow is None:
            self.shadow = {n: p.detach().clone() for n, p in self._float_params(pl_module)}

    @torch.no_grad()
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Restore raw weights before the optimizer sees this batch, in case a
        # validation pass left EMA weights swapped in. No-op outside that window.
        self._restore_raw(pl_module)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.shadow is None:
            return
        d = self.decay
        for n, p in self._float_params(pl_module):
            s = self.shadow.get(n)
            if s is None or s.shape != p.shape:
                self.shadow[n] = p.detach().clone()
            else:
                s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def _swap_in_ema(self, pl_module):
        if self.shadow is None or self._backup is not None:
            return  # nothing to swap, or already swapped in
        self._backup = {}
        for n, p in self._float_params(pl_module):
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def _restore_raw(self, pl_module):
        if self._backup is None:
            return
        for n, p in self._float_params(pl_module):
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = None

    def on_validation_start(self, trainer, pl_module):
        # Swap EMA in for the whole validation window. Held (NOT restored at
        # on_validation_end) so a ModelCheckpoint saving at validation end captures
        # EMA; on_train_batch_start restores raw before the next optimizer step.
        self._swap_in_ema(pl_module)

    @torch.no_grad()
    def on_train_end(self, trainer, pl_module):
        # Make sure raw isn't lingering swapped out, then bake EMA into the model so
        # the saved ckpt + forecasts use it.
        self._restore_raw(pl_module)
        if self.shadow is None:
            return
        for n, p in self._float_params(pl_module):
            if n in self.shadow:
                p.copy_(self.shadow[n])
        print(f"[WeightEMACallback] EMA weights (decay={self.decay}) baked into model at train end.")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r',  encoding='utf-8-sig') as f:
        return yaml.safe_load(f)


def validate_config(config: Dict[str, Any], models_to_train: Dict[str, Any]) -> None:
    """Reject incompatible config combinations up front, before any training/inference.

    Cheaper to fail at startup than to discover the problem mid-predict (after a full
    training run). Currently guards:
    - reconcile + mse: reconciliation reweights the constraint solve by the predictive
      variance, but loss_type='mse' ablates the variance head (forma.py), so the two
      are incompatible — caught here rather than as a NoneType error in the predict loop.
    """
    if config.get('inference', {}).get('reconcile'):
        for name, mcfg in models_to_train.items():
            if (mcfg.get('type') in ('forma', 'gnn_transformer')
                    and mcfg.get('parameters', {}).get('loss_type') == 'mse'):
                raise ValueError(
                    f"Model '{name}': inference.reconcile=true requires a variance head, "
                    f"but loss_type='mse' ablates it. Set inference.reconcile:false for "
                    f"MSE-track Forma runs.")


def create_model(model_config: Dict[str, Any], data_config: Dict[str, Any] = None) -> BaseModel:
    """
    Create model instance based on configuration.
    
    Args:
        model_config: Model configuration dictionary
        data_config: Data configuration dictionary (optional, for data-dependent params)

    Returns:
        Instantiated model following BaseModel interface
    """
    model_type = model_config['type']
    model_params = model_config.get('parameters', {}).copy()  # Make a copy to avoid modifying original

    if model_type == 'forma' or model_type == 'gnn_transformer':
        # Determine num_account_types derived from data
        if data_config:
            processed_dir = Path(data_config.get('processed_dir', 'data/processed'))
            account_map_path = processed_dir / 'account_id_map.csv'
            if account_map_path.exists():
                try:
                    df_acc = pd.read_csv(account_map_path)
                    num_account_types = len(df_acc)
                    print(f"Auto-detected num_account_types={num_account_types} from {account_map_path}")
                    model_params['num_account_types'] = num_account_types
                except Exception as e:
                    print(f"Warning: Failed to read {account_map_path} for num_account_types: {e}")

        if 'num_account_types' not in model_params:
            raise ValueError("num_account_types missing in config and could not be detected from data file.")

        # Inherit industry_mode from data config if not explicitly set in model params
        if 'industry_mode' not in model_params:
            model_params['industry_mode'] = data_config.get('industry_mode', 'none')

        # Auto-detect num_industries from industry_id_map.csv (only if industry is active)
        if model_params.get('industry_mode', 'none') != 'none' and 'num_industries' not in model_params:
            industry_map_path = Path(data_config.get('industry_id_map_path', 'data/processed/industry_id_map.csv'))
            if industry_map_path.exists():
                try:
                    df_ind = pd.read_csv(industry_map_path)
                    num_industries = int(df_ind['industry_id'].max()) + 1
                    print(f"Auto-detected num_industries={num_industries} from {industry_map_path}")
                    model_params['num_industries'] = num_industries
                except Exception as e:
                    print(f"Warning: Failed to read {industry_map_path} for num_industries: {e}")

        # Ensure numeric parameters are correctly typed
        if 'lr' in model_params:
            model_params['lr'] = float(model_params['lr'])
        if 'dropout' in model_params:
            model_params['dropout'] = float(model_params['dropout'])

        model = FormaModel(**model_params)
        model.model_name = 'forma' # Add name for compatibility
        return model
    elif model_type == 'sklearn':
        sklearn_type = model_params.pop('sklearn_type', 'linear')
        # Extract hyperparameter_grid if present
        hyperparameter_grid = model_params.pop('hyperparameter_grid', None)
        from forma.models.baselines import SklearnBaselineModel
        return SklearnBaselineModel(
            model_type=sklearn_type,
            hyperparameter_grid=hyperparameter_grid,
            **model_params
        )
    elif model_type == 'xgboost':
        # Extract hyperparameter_grid if present
        hyperparameter_grid = model_params.pop('hyperparameter_grid', None)
        from forma.models.baselines import XGBoostModel
        return XGBoostModel(
            hyperparameter_grid=hyperparameter_grid,
            **model_params
        )
    elif model_type == 'ffnn':
        # Single FFNN with a vector output head + masked loss. Consumes the same
        # TabularDataModule as the sklearn/XGBoost baselines (create_data_module's
        # else branch), so no data-module change is needed.
        from forma.models.ffnn import FeedForwardModel
        return FeedForwardModel(**model_params)
    elif model_type == 'chained_gbm':
        from forma.models.chained_gbm import ChainedGBMModel
        hyperparameter_grid = model_params.pop('hyperparameter_grid', None)
        return ChainedGBMModel(
            hyperparameter_grid=hyperparameter_grid,
            **model_params
        )
    elif model_type == 'naive_drift':
        from forma.models.naive import NaiveDriftModel
        return NaiveDriftModel(**model_params)
    elif model_type == 'naive':  # New naive (no drift) model
        from forma.models.naive import NaiveModel
        return NaiveModel(**model_params)
    elif model_type == 'chronos':
        # Zero-shot Chronos-2 time-series foundation model. Consumes the same
        # TabularDataModule as the baselines (create_data_module else-branch), but
        # reconstructs per-series context from *_level_0 rather than the feature
        # matrix. Lazy import so the repo runs without chronos-forecasting installed.
        from forma.models.chronos_ts import ChronosTSModel
        return ChronosTSModel(**model_params)
    elif model_type == 'chronos_raw':
        # Chronos-2 benchmark v2: raw tuple contexts (Chronos's own normalization),
        # origin-frozen target-space output transform (exact {acct}_t{h} alignment),
        # Forma-parity 12-quarter context, mean + median point files in one pass.
        from forma.models.chronos_ts import ChronosRawTSModel
        return ChronosRawTSModel(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_data_module(model_type: str, data_config: Dict[str, Any]):
    """
    Create appropriate data module based on model type.
    
    Args:
        model_type: Type of model (determines data format needed)
        data_config: Data configuration dictionary
         
    Returns:
        Data module instance
    """
    data_dir = data_config.get('processed_dir', 'data/processed')
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    # feature_set comes from config -> data -> feature_set
    feature_set = data_config.get('feature_set', 'core')
    
    if model_type == 'forma' or model_type == 'gnn_transformer':
        # Prepare paths
        data_dir_path = Path(data_dir)
        account_map_path = data_dir_path / 'account_id_map.csv'
        firm_id_map_path = data_dir_path / 'firm_id_map.csv'

        # Ensure data_config has what FormaDataModule expects
        if 'feature_set' not in data_config:
            data_config['feature_set'] = feature_set

        return FormaDataModule(
            data_dir_path=data_dir_path,
            account_map_path=account_map_path,
            firm_id_map_path=firm_id_map_path,
            data_config=data_config
        )
    elif model_type in ('naive_drift', 'naive'):  # Share the same data module
        return NaiveDriftDataModule(
            data_dir=data_dir,
            data_config=data_config
        )
    else:
        # For sklearn and XGBoost models
        return TabularDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            data_config=data_config
        )


def walk_forward_validation(model: BaseModel, data_module, config: Dict[str, Any], exp_dir: Path = None) -> None:
    """
    Perform walk-forward validation training.
    
    This implements time-series-aware training to prevent lookahead bias.
    
    Args:
        model: Model instance to train
        data_module: Data module providing train/val/test splits
        config: Full configuration dictionary
    """
    # For now, implement simple train/test split
    # TODO: Implement proper walk-forward validation with multiple time windows
    
    # Check if this is a PyTorch Lightning model (Forma)
    if isinstance(model, L.LightningModule):
        print(f"Training {getattr(model, 'model_name', 'Forma')} model (PyTorch Lightning)...")

        # Setup trainer
        # Try to get max_epochs from curriculum config first, then trainer config, then default to 10
        max_epochs = config.get('data', {}).get('curriculum', {}).get('max_epochs')
        if max_epochs is None:
            max_epochs = config.get('trainer', {}).get('max_epochs', 10)

        callbacks = [CurriculumCallback()]

        # Opt-in: select the best-val_loss checkpoint instead of the final-epoch one.
        # Requires validation data (disabled when --no-cv is set, since val_loss is not logged).
        no_cv = config.get('data', {}).get('no_cv', False)
        select_best_by_val_loss = config.get('trainer', {}).get('select_best_by_val_loss', False)
        best_ckpt_callback = None
        if select_best_by_val_loss and not no_cv:
            best_ckpt_callback = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename='best-{epoch:02d}-{val_loss:.4f}',
            )
            callbacks.append(best_ckpt_callback)
        elif select_best_by_val_loss and no_cv:
            print("select_best_by_val_loss=True ignored: --no-cv is set, no val_loss to monitor.")

        # Optional: per-step timing instrumentation
        if config.get('timing', False):
            class TimingCallback(L.Callback):
                def __init__(self):
                    self.step_times = []
                    self.t0 = None

                def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    self.t0 = time.time()

                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    self.step_times.append(time.time() - self.t0)

                def on_train_end(self, trainer, pl_module):
                    if len(self.step_times) > 2:
                        # Skip first 2 steps (warmup / compilation)
                        times = self.step_times[2:]
                        avg = sum(times) / len(times)
                        print(f"\n{'='*50}")
                        print(f"TIMING RESULTS ({len(times)} steps, warmup excluded)")
                        print(f"  Avg step:  {avg*1000:.1f} ms")
                        print(f"  Min step:  {min(times)*1000:.1f} ms")
                        print(f"  Max step:  {max(times)*1000:.1f} ms")
                        print(f"  Total:     {sum(times):.2f} s")
                        print(f"{'='*50}\n")

            callbacks.append(TimingCallback())

        # Opt-in weight EMA (trainer.weight_ema_decay). The callback holds EMA
        # weights through the whole validation window and restores raw on the next
        # train batch (see WeightEMACallback docstring), so insertion order vs.
        # ModelCheckpoint does not matter — a checkpoint saved at validation end
        # captures EMA regardless. Default-off: absent/<=0 -> no callback ->
        # canonical path unchanged.
        weight_ema_decay = config.get('trainer', {}).get('weight_ema_decay')
        if weight_ema_decay:
            callbacks.append(WeightEMACallback(float(weight_ema_decay)))
            print(f"Weight EMA enabled (decay={weight_ema_decay}).")

        trainer_kwargs = {
            'max_epochs': max_epochs,
            'accelerator': 'auto',
            'devices': 'auto',
            'default_root_dir': str(exp_dir / 'models') if exp_dir else config.get('models_dir', 'results/models'),
            'log_every_n_steps': 10,
            'enable_checkpointing': True,  # Explicitly enable checkpointing for production runs
            'reload_dataloaders_every_n_epochs': 1,  # Reload dataloaders every epoch for curriculum learning
            'callbacks': callbacks,
        }

        # Optional: limit batches per epoch for quick timing runs
        trainer_config = config.get('trainer', {})
        if 'limit_train_batches' in trainer_config:
            trainer_kwargs['limit_train_batches'] = trainer_config['limit_train_batches']
        if 'limit_val_batches' in trainer_config:
            trainer_kwargs['limit_val_batches'] = trainer_config['limit_val_batches']
        # Opt-in gradient clipping (trainer.gradient_clip_val). Insurance against
        # loss/gradient spikes from the unclipped regularized inputs — without it a
        # single extreme batch inflates Adam's second moments and suppresses learning
        # for thousands of steps (the r10e6 Forma-MSE collapse). Config-driven so the
        # canonical Gaussian runs are byte-unchanged unless a config asks for it.
        if 'gradient_clip_val' in trainer_config:
            trainer_kwargs['gradient_clip_val'] = trainer_config['gradient_clip_val']

        if config.get('debug', False):
            print("Applying DEBUG settings to Trainer...")
            trainer_kwargs.update({
                'accelerator': 'cpu',
                'devices': 1,
                'overfit_batches': 1,
                'log_every_n_steps': 1,
                'detect_anomaly': False,
                'max_epochs': 5,
                'enable_checkpointing': False,
                'enable_model_summary': True,
                'enable_progress_bar': True,
                'num_sanity_val_steps': 0,
            })

            # Force small batch size and disable workers for local debugging
            if hasattr(data_module, 'batch_size'):
                data_module.batch_size = 4
                print("Setting data_module.batch_size = 4 for debugging")
            if hasattr(data_module, 'num_workers'):
                data_module.num_workers = 0
                print("Setting data_module.num_workers = 0 for debugging")

        trainer = L.Trainer(**trainer_kwargs)

        # Train
        trainer.fit(model, datamodule=data_module)

        # Save trained model (checkpoint)
        # Always run manual save to ensure we can write to disk, even in debug
        models_dir = exp_dir / 'models' if exp_dir else Path(config.get('models_dir', 'results/models'))
        models_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = models_dir / f"{getattr(model, 'model_name', 'forma')}.ckpt"

        if best_ckpt_callback is not None and best_ckpt_callback.best_model_path:
            best_path = best_ckpt_callback.best_model_path
            best_score = best_ckpt_callback.best_model_score
            print(f"Using best val_loss checkpoint: {best_path} (val_loss={best_score:.4f})")
            # Reload best weights into the in-memory model so downstream
            # generate_predictions() uses the best checkpoint, not the final epoch.
            best_state = torch.load(best_path, map_location='cpu', weights_only=False)
            model.load_state_dict(best_state['state_dict'])
            shutil.copy(best_path, model_save_path)
        else:
            trainer.save_checkpoint(str(model_save_path))
        print(f"Model checkpoint saved to {model_save_path}")

        print(f"Training completed for {getattr(model, 'model_name', 'forma')}")
        return

    print(f"Training {model.model_name} model...")
    
    # Setup data
    data_module.prepare_data()
    data_module.setup(stage="fit")
    
    # Resolve the model output folder up front so baseline models can stream each
    # (target, horizon) sub-model to disk and free it during fit, instead of
    # holding all of them in memory until save(). This bounds peak memory (the
    # 78x20 Random Forest sweep otherwise OOMs) and makes the multi-hour run
    # resumable — a re-launch skips sub-models already on disk. Only
    # MultiHorizonBaselineModel reads incremental_dir; other models ignore it.
    models_dir = exp_dir / 'models' if exp_dir else Path(config.get('models_dir', 'results/models'))
    models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / f"{model.model_name}"
    if hasattr(model, 'incremental_dir'):
        model.incremental_dir = str(model_save_path)

    # Train model - pass data_module so baseline models can access:
    # - train data via get_train_data()
    # - validation data via get_val_data() (for hyperparameter tuning)
    # - column metadata (feature_columns, target_columns)
    model.fit(data_module)

    # Save trained model. In incremental mode the per-horizon pkls are already on
    # disk; save() then only writes metadata/summary (loading models lazily).
    model.save(str(model_save_path))

    print(f"Training completed for {model.model_name}")


def reconcile_batch(batch, mu, sigma_sq, batch_idx):
    """Batched probabilistic reconciliation on GPU via Lagrangian solve.

    Minimizes  sum_i (regularize(V_i) - mu_i)^2 / (2 sigma_i^2)
    subject to A @ V = 0   (accounting identities in raw space)
    for each graph in the batch simultaneously.

    Solves the [nc x nc] system  (A @ W_inv @ A^T) @ lambda = -r  per graph,
    then applies dV = W_inv @ A^T @ lambda.
    Only constraints involving at least one future (predicted) node are enforced;
    constraints on purely observed nodes are skipped.
    """
    from torch_geometric.utils import to_dense_batch

    device = mu.device
    is_future = (batch.x_time > 0)

    k = batch.reg_k
    mu_reg = batch.reg_mu
    sigma_reg = batch.reg_sigma

    num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
    if num_graphs == 0:
        return mu[is_future], {
            'fallback_triggered': False, 'n_graphs': 0,
            'n_clamp_hits': 0, 'n_active_nodes': 0,
        }

    scale_per_node = batch.scale_val[batch_idx]

    v_raw = de_regularize(mu.clamp(-20.0, 20.0), k, mu_reg, sigma_reg) * scale_per_node
    v_obs = batch.x_val_scaled * scale_per_node
    V = torch.where(is_future, v_raw, v_obs)

    constraint_src = batch.constraint_src
    constraint_dst = batch.constraint_dst
    constraint_sign = batch.constraint_sign
    num_constraints = batch.num_constraints
    num_constraint_edges = batch.num_constraint_edges

    nn_per_graph = torch.bincount(batch_idx, minlength=num_graphs)
    node_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    node_offsets[1:] = nn_per_graph.cumsum(0)

    edge_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    edge_offsets[1:] = num_constraint_edges.cumsum(0)

    crow_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    crow_offsets[1:] = num_constraints.cumsum(0)

    max_nc = int(num_constraints.max().item()) if num_constraints.numel() > 0 else 0
    if max_nc == 0:
        return mu[is_future], {
            'fallback_triggered': False, 'n_graphs': int(num_graphs),
            'n_clamp_hits': 0, 'n_active_nodes': 0,
        }

    max_nn = int(nn_per_graph.max().item())

    A_batched = torch.zeros(num_graphs, max_nc, max_nn, device=device, dtype=mu.dtype)
    total_edges = constraint_src.size(0)
    if total_edges > 0:
        edge_indices = torch.arange(total_edges, device=device)
        edge_graph = torch.bucketize(edge_indices, edge_offsets[1:], right=True)
        local_rows = constraint_src - crow_offsets[edge_graph]
        local_cols = constraint_dst - node_offsets[edge_graph]
        A_batched[edge_graph, local_rows, local_cols] = constraint_sign.to(A_batched.dtype)

    V_dense, valid_mask = to_dense_batch(V, batch_idx)
    sigma_sq_dense, _ = to_dense_batch(sigma_sq, batch_idx)
    k_dense, _ = to_dense_batch(k, batch_idx)
    mu_reg_dense, _ = to_dense_batch(mu_reg, batch_idx)
    sigma_reg_dense, _ = to_dense_batch(sigma_reg, batch_idx)
    future_dense, _ = to_dense_batch(is_future.float(), batch_idx)
    future_mask = future_dense.bool()
    scale_dense, _ = to_dense_batch(scale_per_node, batch_idx)

    constraint_valid = torch.arange(max_nc, device=device).unsqueeze(0) < num_constraints.unsqueeze(1)

    has_future_node = (A_batched.abs() * future_mask.unsqueeze(1).float()).sum(dim=2) > 0
    adjustable = constraint_valid & has_future_node
    A_adj = A_batched * adjustable.unsqueeze(2).float()

    s = get_derivative_s(V_dense / (scale_dense + 1e-8), k_dense, mu_reg_dense, sigma_reg_dense) / (scale_dense + 1e-8)
    W_inv_raw = sigma_sq_dense / (s ** 2 + 1e-8)
    # Wide clamp: float64 cond-number headroom is tight at 15 orders of
    # magnitude, but tightening empirically hurt reconciled R² by ~5 ppt
    # across all r7/r8 arms (PR #100 measurement). The rare ill-conditioned
    # batches that DO trigger are caught by the outer try/except below; no
    # need to penalize the median graph to prevent a tail-event crash.
    W_inv = W_inv_raw.clamp(min=1e-6, max=1e9) * future_mask.float() * valid_mask.float()
    W_inv = torch.nan_to_num(W_inv, nan=0.0, posinf=1e9, neginf=0.0)
    # Telemetry on clamp binding — historically benign at this width
    # (~0.0X% per the post-revert reruns); kept so a future regression to
    # significantly higher rates is loud and grep-able from the SUMMARY line.
    active_mask = (future_mask & valid_mask)
    n_active = int(active_mask.sum().item())
    n_clamp_min = int(((W_inv_raw < 1e-6) & active_mask).sum().item()) if n_active else 0
    n_clamp_max = int(((W_inv_raw > 1e9) & active_mask).sum().item()) if n_active else 0

    A_adj_f64 = A_adj.double()
    W_inv_f64 = W_inv.double()

    A_scaled = A_adj_f64 * W_inv_f64.unsqueeze(1)
    Q = torch.bmm(A_scaled, A_adj_f64.transpose(1, 2))
    eye = torch.eye(max_nc, device=device, dtype=torch.float64).unsqueeze(0)
    valid_Q = adjustable.unsqueeze(2) & adjustable.unsqueeze(1)
    Q = Q * valid_Q.double() + (~valid_Q).double() * eye + 1e-8 * eye

    r = torch.bmm(A_adj_f64, V_dense.double().unsqueeze(-1)).squeeze(-1)
    rhs = -r.unsqueeze(-1)

    try:
        lam = torch.linalg.solve(Q, rhs).squeeze(-1)
    except torch.linalg.LinAlgError:
        try:
            lam = torch.linalg.lstsq(Q, rhs).solution.squeeze(-1)
        except torch.linalg.LinAlgError as e:
            # Both batched solve and lstsq SVD fallback failed — Q too ill-conditioned
            # even in float64. Skip reconciliation for this batch (caller gets raw mu);
            # other batches in the test set proceed normally.
            print(f"  [Reconciliation] ERROR: batched solve+lstsq both failed "
                  f"({type(e).__name__}: {str(e)[:120]}); "
                  f"skipping reconciliation for batch of {num_graphs} graphs")
            return mu[is_future], {
                'fallback_triggered': True,
                'n_graphs': int(num_graphs),
                'n_clamp_hits': n_clamp_min + n_clamp_max,
                'n_active_nodes': n_active,
            }
    lam = torch.nan_to_num(lam, nan=0.0)

    At_lam = torch.bmm(A_adj_f64.transpose(1, 2), lam.unsqueeze(-1)).squeeze(-1)
    dV = (W_inv_f64 * At_lam).float()
    
    # Defensive clamping on dV to prevent overflow
    dV = torch.nan_to_num(dV, nan=0.0, posinf=1e15, neginf=-1e15).clamp(-1e15, 1e15)
    
    V_dense = V_dense + dV
    
    # Final safety check on V_dense: prevent values that would explode sinh/asinh or float32
    V_dense = torch.nan_to_num(V_dense, nan=0.0, posinf=1e30, neginf=-1e30).clamp(-1e30, 1e30)

    r_post = torch.bmm(A_adj, V_dense.unsqueeze(-1)).squeeze(-1)
    max_viol = (r_post.abs() * adjustable.float()).max(dim=1).values
    n_bad = int((max_viol > 1.0).sum().item())
    if n_bad > 0:
        print(f"  [Reconciliation] WARNING: {n_bad}/{num_graphs} graphs have constraint violation > $1 "
              f"(max={max_viol.max().item():.2f})")

    rec_mu_dense = regularize(V_dense / scale_dense, k_dense, mu_reg_dense, sigma_reg_dense)
    rec_mu_flat = rec_mu_dense[valid_mask]
    return rec_mu_flat[is_future], {
        'fallback_triggered': False,
        'n_graphs': int(num_graphs),
        'n_clamp_hits': n_clamp_min + n_clamp_max,
        'n_active_nodes': n_active,
    }


# The canonical forecast-write contract (FORECAST_SCHEMA_COLS,
# FORECAST_PARQUET_COMPRESSION, finalize_forecast_frame) lives in
# models/forecast_io.py so train.py and models/baselines.py can both import it
# without a circular import. Imported at the top of this module.


def _density_meta(model) -> dict | None:
    """The predictive-density sidecar payload for a forecast, or None.

    The density is a property of the model (which loss it was fit under). evaluate.py
    reads it from an optional `{stem}.nll.json` and DEFAULTS to gaussian when absent,
    so we only emit a sidecar for the non-gaussian probabilistic families — gaussian
    and mse runs stay byte-identical (no new file), preserving backwards compat.
    """
    lt = getattr(getattr(model, 'hparams', None), 'loss_type', None)
    if lt == 'laplace':
        return {'family': 'laplace', 'df': None}
    if lt == 'student_t':
        return {'family': 'student_t',
                'df': float(getattr(model.hparams, 'student_t_df', 5.0))}
    return None


def _write_nll_sidecar(model, forecast_path: Path) -> None:
    """Write `{stem}.nll.json` next to a forecast so evaluate.py scores the right
    density. No-op for gaussian/mse (the eval default) — keeps their outputs
    unchanged."""
    meta = _density_meta(model)
    if meta is None:
        return
    side = Path(forecast_path).with_name(f"{Path(forecast_path).stem}.nll.json")
    with open(side, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh)
    print(f"  NLL density sidecar written: {side.name} ({meta['family']})")


def _forecast_filename(model_name: str, feature_set: str, split_name: str,
                       seed: int | None = None) -> str:
    """Build a forecast filename, optionally inserting the seed token.

    With ``seed`` set, emits the 5-segment seed-aware form
    ``{model}__{fs}__{seed}__{split}__predictions.parquet`` that evaluate.py's
    discovery groups into a cross-seed mixture family (same ``{model}__{fs}``,
    distinct integer seed token). With ``seed=None`` it emits the legacy
    4-segment ``{model}__{fs}__{split}`` form, so the seed token is strictly
    opt-in (gated by ``inference.seed_in_forecast_name``) and existing pipelines
    are unaffected by default.
    """
    seed_seg = f"{int(seed)}__" if seed is not None else ""
    return f"{model_name}__{feature_set}__{seed_seg}{split_name}__predictions.parquet"


def generate_predictions(model: BaseModel, data_module, config: Dict[str, Any], exp_dir: Path = None) -> None:
    """
    Generate predictions and save to results/forecasts/.

    Saves one file per split:
      {model_name}__{feature_set}__{split}__predictions.parquet
    or, when inference.seed_in_forecast_name is set, the seed-aware form
      {model_name}__{feature_set}__{seed}__{split}__predictions.parquet

    Args:
        model: Trained model instance
        data_module: Data module providing train/val/test splits
        config: Full configuration dictionary
    """

    # Allow a larger batch size for prediction than training.
    data_cfg = config.get('data', {}) if isinstance(config, dict) else {}
    predict_batch_size = int(data_cfg.get('predict_batch_size', data_cfg.get('batch_size', 32)))

    # Generate save dir
    results_dir = exp_dir / 'forecasts' if exp_dir else Path(config.get('results_dir', 'results/forecasts'))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Backward compatibility: Always write to results/forecasts as the "latest" results
    latest_results_dir = Path(config.get('results_dir', 'results/forecasts'))
    latest_results_dir.mkdir(parents=True, exist_ok=True)

    model_name = getattr(model, 'model_name', 'forma')

    # Opt-in seed token in the forecast filename. When `inference.seed_in_forecast_name`
    # is set, every forecast this run writes carries its `seed` as the 5-segment
    # seed-aware name so that same-arm/different-seed runs auto-group into a cross-seed
    # mixture family in evaluate.py (no rename step). Default off -> legacy 4-segment
    # name, so existing rounds/eval pools are unchanged.
    fname_seed = (int(config.get('seed') or 0)
                  if config.get('inference', {}).get('seed_in_forecast_name', False)
                  else None)

    # Single knob: `inference.save_sigma` drives the likelihood-track sigma column for
    # both families. Forma (Lightning) reads it from config in the predict loop below;
    # BaseModel forecasters (FFNN) read it off a `save_sigma` attribute, so mirror the
    # config value onto the model here. hasattr-gated, so it no-ops for models without
    # a variance head.
    if 'save_sigma' in config.get('inference', {}) and hasattr(model, 'save_sigma'):
        model.save_sigma = bool(config['inference']['save_sigma'])

    # Splits to predict. Default is test-only: Phase 3 / cross-experiment eval is
    # test-only, train+val predict passes burn billing/time and have repeatedly
    # hit a MooseFS-corrupt tuple_val that surfaces as a spurious EXIT=1.
    # Pass --forecast-splits test,val,train to restore all three.
    splits_to_predict = config.get('forecast_splits', ['test'])

    # Helper to save a forecasts DF with consistent validation
    def _save_forecasts(forecasts_df: pd.DataFrame, split_name: str):
        required_columns = {'firm_id', 'target', 'quarter', 'forecast_horizon', 'prediction', 'model'}

        # Add missing columns if any (e.g. forecast_horizon might be missing for some Lightning outputs)
        if 'forecast_horizon' not in forecasts_df.columns and isinstance(model, L.LightningModule):
            forecasts_df['forecast_horizon'] = 0

        if not required_columns.issubset(forecasts_df.columns):
            raise ValueError(
                f"Predictions for split={split_name} are missing required columns: "
                f"{required_columns - set(forecasts_df.columns)}"
            )

        # Canonical schema + float32 payload before writing (drops diagnostic
        # columns, downcasts floats). zstd codec for ~10% smaller files.
        forecasts_df = finalize_forecast_frame(forecasts_df)

        save_path = results_dir / _forecast_filename(
            model_name, data_module.feature_set, split_name, fname_seed)
        forecasts_df.to_parquet(save_path, index=False, compression=FORECAST_PARQUET_COMPRESSION)
        print(f"Predictions saved to {save_path}")
        _write_nll_sidecar(model, save_path)

        # Also save to "latest" if we are in an experiment (to avoid double-save in debug/default)
        if exp_dir:
            latest_path = latest_results_dir / _forecast_filename(
                model_name, data_module.feature_set, split_name, fname_seed)
            forecasts_df.to_parquet(latest_path, index=False, compression=FORECAST_PARQUET_COMPRESSION)
            print(f"Latest copy saved to {latest_path}")
            _write_nll_sidecar(model, latest_path)

    # Chunked streaming save for Lightning models. The legacy in-memory path
    # peaks at ~60-80 GB on a 385M-row forecast (Series.map -> object Series of
    # Python strings + multiple astype copies), which OOM-kills on pods with
    # <120 GB cgroup memory (see issue #50). Streams 25M-row chunks into a
    # single parquet via pyarrow.parquet.ParquetWriter; per-chunk peak ~5-8 GB.
    def _stream_save_forecasts(
        fid_arr,
        qtr_arr,
        tgt_arr,
        hor_arr,
        pred_arr,
        int_to_firm,
        id_to_name,
        model_label: str,
        split_name: str,
        chunk_size: int = 25_000_000,
        sigma_arr=None,
    ):
        import pyarrow as pa
        from forma.models.forecast_io import atomic_parquet_writers

        # Downcast the float payload to float32 (asinh target space; lossless for
        # R2/MAE/NLL, halves the column vs float64) before the chunked write.
        pred_arr = np.asarray(pred_arr, dtype=np.float32)
        if sigma_arr is not None:
            sigma_arr = np.asarray(sigma_arr, dtype=np.float32)

        save_path = results_dir / _forecast_filename(
            model_name, data_module.feature_set, split_name, fname_seed)
        latest_path = (
            latest_results_dir / _forecast_filename(
                model_name, data_module.feature_set, split_name, fname_seed)
            if exp_dir else None
        )

        n_rows = len(fid_arr)
        n_chunks = (n_rows + chunk_size - 1) // chunk_size
        print(f"  Streaming {n_rows:,} rows in {n_chunks} chunk(s) of {chunk_size:,} -> {save_path}")

        unmapped_firms = 0
        unmapped_targets = 0
        # atomic_parquet_writers (shared with the baseline streaming path) writes
        # each forecast into a sibling .tmp and os.replace()s it into the canonical
        # path only after a clean close -- a crash mid-stream leaves a stray .tmp,
        # never a truncated parquet a later evaluate.py reads as complete. The
        # latest-copy path is included only when exp_dir set it (else None).
        stream_paths = [save_path] + ([latest_path] if latest_path is not None else [])
        with atomic_parquet_writers(stream_paths) as write_table:
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                df = pd.DataFrame({
                    'firm_id': fid_arr[start:end],
                    'quarter': qtr_arr[start:end],
                    'target':  tgt_arr[start:end],
                    'forecast_horizon': hor_arr[start:end],
                })
                df['prediction'] = pred_arr[start:end]
                # Likelihood track: optional predictive std, same regularized target
                # space as `prediction`. evaluate.py reads it to score mean NLL; absent
                # on point-only (MSE-track) runs, keeping the standard schema/size.
                if sigma_arr is not None:
                    df['sigma'] = sigma_arr[start:end]

                if int_to_firm is not None:
                    df['firm_id'] = df['firm_id'].map(int_to_firm)
                    n_bad = int(df['firm_id'].isna().sum())
                    if n_bad:
                        unmapped_firms += n_bad
                        df = df.dropna(subset=['firm_id'])
                    df['firm_id'] = df['firm_id'].astype(int).astype(str).str.zfill(6)
                else:
                    df['firm_id'] = df['firm_id'].astype(str).str.zfill(6)

                if id_to_name is not None:
                    df['target'] = df['target'].map(id_to_name)
                    n_bad = int(df['target'].isna().sum())
                    if n_bad:
                        unmapped_targets += n_bad
                        df = df.dropna(subset=['target'])

                from forma.data.quarters import qoff_to_quarter_end
                df['quarter'] = qoff_to_quarter_end(df['quarter'].values.astype('int64'))
                df['model'] = model_label

                if start == 0:
                    # Schema-sync guard (once, on the first chunk): this df is built
                    # with exactly the canonical forecast columns. The in-memory path
                    # enforces the schema via finalize_forecast_frame; this path
                    # enforces it by construction. Fail loudly if a non-canonical (e.g.
                    # diagnostic) column ever sneaks into the dict above, instead of
                    # shipping it silently. Explicit raise (not assert) so the guard
                    # survives `python -O`, which strips asserts.
                    extra_cols = [c for c in df.columns if c not in FORECAST_SCHEMA_COLS]
                    if extra_cols:
                        raise AssertionError(
                            f"streaming forecast has non-canonical columns {extra_cols}; keep "
                            f"in sync with FORECAST_SCHEMA_COLS / finalize_forecast_frame")

                tbl = pa.Table.from_pandas(df, preserve_index=False)
                write_table(tbl)
                del df, tbl

        if unmapped_firms:
            print(f"  Warning: dropped {unmapped_firms} rows with unmapped firm_ids")
        if unmapped_targets:
            print(f"  Warning: dropped {unmapped_targets} rows with unmapped targets")
        print(f"  Predictions saved to {save_path}")
        if latest_path is not None:
            print(f"  Latest copy saved to {latest_path}")

    # Lightning models (Forma): generate per split using the appropriate dataloader
    if isinstance(model, L.LightningModule):
        model.eval()
        model.freeze()

        if torch.cuda.is_available():
            model = model.to('cuda')
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # fp16 autocast at predict by default (matches r10 forecasts, which are
        # therefore fp16-quantized). inference.fp32: true gates it OFF for clean
        # fp32 predictions — used by the companion job that de-quantizes r10e8a/b.
        # Default false => use_amp unchanged => bit-identical to prior behavior.
        force_fp32 = bool(config.get('inference', {}).get('fp32', False))
        use_amp = bool(torch.cuda.is_available()) and not force_fp32
        if force_fp32:
            print("  inference.fp32=true -> fp16 autocast DISABLED at predict (fp32 forecasts).")

        # We want deterministic inference for diagnostics. Force deterministic collator for all splits.
        # This mirrors test_dataloader behavior when deterministic_forecast=True.
        # (If you want the stochastic masking behavior for train/val, we can add a config flag.)

        for split_name in splits_to_predict:
            print(f"\nLoading {split_name} data for {model_name}...")

            # Ensure the datamodule has all splits loaded
            if split_name in ('train', 'val'):
                data_module.setup(stage='fit')
            else:
                data_module.setup(stage='test')

            # Set batch_size for prediction
            if hasattr(data_module, 'batch_size'):
                data_module.batch_size = predict_batch_size

            # Select dataloader
            if split_name == 'train':
                if not hasattr(data_module, 'train_dataloader'):
                    print("  No train_dataloader() available, skipping train split...")
                    continue
                loader = data_module.train_dataloader()
            elif split_name == 'val':
                if not hasattr(data_module, 'val_dataloader'):
                    print("  No val_dataloader() available, skipping val split...")
                    continue
                loader = data_module.val_dataloader()
            else:
                if not hasattr(data_module, 'test_dataloader'):
                    print("  No test_dataloader() available, skipping test split...")
                    continue
                loader = data_module.test_dataloader()

            print(
                f"Generating predictions for {model_name} split={split_name} "
                f"on device={device} (amp={use_amp}, batch_size={predict_batch_size})..."
            )

            firm_id_chunks = []
            quarter_chunks = []
            target_chunks = []
            pred_chunks = []
            horizon_chunks = []

            # Likelihood track: when enabled, also collect the predictive std so the
            # forecast file carries a `sigma` column for evaluate.py's NLL. Off by
            # default (point-only forecasts keep their schema/size). The variance head
            # may be ablated (log_sigma_sq is None) — then there is nothing to save.
            save_sigma = bool(config.get('inference', {}).get('save_sigma', False))
            if save_sigma:
                sigma_chunks = []
                _ls_lo = float(getattr(model.hparams, 'log_sigma_sq_min', -10.0))
                _ls_hi = float(getattr(model.hparams, 'log_sigma_sq_max', 10.0))

            do_reconciliation = config.get('inference', {}).get('reconcile', False)
            if do_reconciliation:
                rec_pred_chunks = []
                # Aggregate reconcile_batch telemetry across batches; printed
                # as a single summary line at the end of the predict loop.
                rec_stats_total = {
                    'fallback_batches': 0, 'fallback_graphs': 0,
                    'total_batches': 0, 'total_graphs': 0,
                    'clamp_hits': 0, 'active_nodes': 0,
                }

            quick_frac = config.get('quick_fraction', None)
            max_batches = int(len(loader) * quick_frac) if quick_frac else None

            with torch.inference_mode():
                autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
                with autocast_ctx:
                    for batch_i, batch in enumerate(tqdm(loader, desc=f"Predicting [{split_name}]")):
                        if max_batches and batch_i >= max_batches:
                            print(f"\n  [Quick mode] Stopped after {batch_i}/{len(loader)} batches ({quick_frac*100:.0f}%)")
                            break
                        batch = batch.to(device, non_blocking=True)

                        mu, log_sigma_sq = model(batch)
                        # log_sigma_sq is None when the variance head is ablated
                        # (loss_type='mse'); sigma_sq is only consumed by reconcile. The
                        # mse+reconcile combination is rejected up front in main(), so a
                        # None here always coincides with do_reconciliation=False.
                        sigma_sq = (torch.exp(log_sigma_sq.clamp(-10.0, 10.0))
                                    if log_sigma_sq is not None else None)

                        is_future_time = (batch.x_time > 0)
                        node_mask = is_future_time

                        if not node_mask.any():
                            continue

                        preds = mu[node_mask]

                        batch_idx = getattr(batch, 'x_id_batch', None)
                        if batch_idx is None:
                            batch_idx = getattr(batch, 'batch', None)
                        if batch_idx is None:
                            batch_idx = torch.zeros(batch.x_id.size(0), dtype=torch.long, device=batch.x_id.device)
                        graph_indices = batch_idx[node_mask]
                        firm_ids = batch.firm_id[graph_indices]
                        quarters = batch.center_q[graph_indices]
                        account_ids = batch.x_id[node_mask]
                        horizons = batch.x_time[node_mask]

                        firm_id_chunks.append(firm_ids.detach().cpu().numpy())
                        quarter_chunks.append(quarters.detach().cpu().numpy())
                        target_chunks.append(account_ids.detach().cpu().numpy())
                        pred_chunks.append(preds.detach().cpu().to(torch.float32).numpy())
                        horizon_chunks.append(horizons.detach().cpu().numpy())

                        if save_sigma and log_sigma_sq is not None:
                            # predictive std = exp(0.5 * log_sigma_sq), clamped to the
                            # model's training bounds so it matches the fitted variance.
                            sigma = torch.exp(0.5 * log_sigma_sq.clamp(_ls_lo, _ls_hi))
                            sigma_chunks.append(
                                sigma[node_mask].detach().cpu().to(torch.float32).numpy())

                        if do_reconciliation:
                            with torch.autocast(device_type=device.type, enabled=False):
                                rec_preds, rec_stats = reconcile_batch(batch, mu.float(), sigma_sq.float(), batch_idx)
                            rec_pred_chunks.append(rec_preds.detach().cpu().to(torch.float32).numpy())
                            rec_stats_total['total_batches'] += 1
                            rec_stats_total['total_graphs'] += rec_stats['n_graphs']
                            rec_stats_total['clamp_hits'] += rec_stats['n_clamp_hits']
                            rec_stats_total['active_nodes'] += rec_stats['n_active_nodes']
                            if rec_stats['fallback_triggered']:
                                rec_stats_total['fallback_batches'] += 1
                                rec_stats_total['fallback_graphs'] += rec_stats['n_graphs']

            if do_reconciliation and rec_stats_total['total_batches'] > 0:
                # End-of-predict reconciliation telemetry. fallback_batches > 0
                # means at least one batch's reconciliation was skipped; the
                # affected graphs' rows in the saved parquet are raw mu
                # despite the file label suggesting "reconciled". clamp_hits
                # at >~0.1% of active nodes suggests the W_inv clamp is
                # over-tight; loosen if so.
                fb_b = rec_stats_total['fallback_batches']
                fb_g = rec_stats_total['fallback_graphs']
                tot_b = rec_stats_total['total_batches']
                tot_g = rec_stats_total['total_graphs']
                ch = rec_stats_total['clamp_hits']
                an = rec_stats_total['active_nodes']
                clamp_rate = (ch / an * 100.0) if an else 0.0
                tag = "ERROR" if fb_b > 0 else "OK"
                print(f"  [Reconciliation] SUMMARY [{tag}]: "
                      f"fallback batches={fb_b}/{tot_b} "
                      f"(graphs={fb_g}/{tot_g}); "
                      f"W_inv clamp hits={ch}/{an} ({clamp_rate:.3f}%)")

            if not pred_chunks:
                print(f"Warning: No predictions generated for split={split_name}")
            else:
                print("Converting internal IDs to original format (streaming)...")

                fid_all = np.concatenate(firm_id_chunks)
                qtr_all = np.concatenate(quarter_chunks)
                tgt_all = np.concatenate(target_chunks)
                hor_all = np.concatenate(horizon_chunks)
                prd_all = np.concatenate(pred_chunks)
                # Free per-batch lists so peak memory stays at ~15 GB (one
                # copy of each id array) rather than ~30 GB during conversion.
                firm_id_chunks = quarter_chunks = target_chunks = horizon_chunks = pred_chunks = None
                sig_all = None
                if save_sigma and sigma_chunks:
                    sig_all = np.concatenate(sigma_chunks)
                    sigma_chunks = None
                rec_all = None
                if do_reconciliation:
                    rec_all = np.concatenate(rec_pred_chunks)
                    rec_pred_chunks = None

                firm_map_path = Path(config.get('data', {}).get('firm_id_map_path', 'data/processed/firm_id_map.csv'))
                if firm_map_path.exists():
                    firm_map = pd.read_csv(firm_map_path)
                    int_to_firm = dict(zip(firm_map['firm_id_int'], firm_map['firm_id']))
                else:
                    print(f"Warning: firm_id_map not found at {firm_map_path}, using raw integer as zero-padded string")
                    int_to_firm = None

                account_map_path = Path(config.get('data', {}).get('account_map_path', 'data/processed/account_id_map.csv'))
                if account_map_path.exists():
                    account_map = pd.read_csv(account_map_path)
                    id_to_name = dict(zip(account_map['account_id'], account_map['account_name']))
                else:
                    print(f"Warning: account_id_map not found at {account_map_path}, target will remain as integer")
                    id_to_name = None

                if do_reconciliation:
                    # Preserve legacy single-file-final-overwrite semantics:
                    # raw is written first, then reconciled overwrites it at
                    # the same path. Disambiguating raw vs reconciled outputs
                    # is a separate refactor (see issue #50 caveats).
                    _stream_save_forecasts(
                        fid_all, qtr_all, tgt_all, hor_all, prd_all,
                        int_to_firm, id_to_name,
                        model_label=f"{model_name}_raw",
                        split_name=split_name,
                        sigma_arr=sig_all,
                    )
                    # Reconciled mean has a different predictive center, so the raw
                    # variance head's sigma no longer describes it — omit sigma here.
                    _stream_save_forecasts(
                        fid_all, qtr_all, tgt_all, hor_all, rec_all,
                        int_to_firm, id_to_name,
                        model_label=f"{model_name}_reconciled",
                        split_name=split_name,
                    )
                else:
                    _stream_save_forecasts(
                        fid_all, qtr_all, tgt_all, hor_all, prd_all,
                        int_to_firm, id_to_name,
                        model_label=model_name,
                        split_name=split_name,
                        sigma_arr=sig_all,
                    )

        return

    # Baseline (sklearn/xgboost/naive) models
    for split_name in splits_to_predict:
        print(f"\nLoading {split_name} data for {model.model_name}...")

        if split_name in ('train', 'val'):
            data_module.setup(stage='fit')
        else:
            data_module.setup(stage='test')

        # Retrieve split data
        split_data = None
        if split_name == 'train' and hasattr(data_module, 'get_train_data'):
            split_data = data_module.get_train_data()
        elif split_name == 'val' and hasattr(data_module, 'get_val_data'):
            split_data = data_module.get_val_data()
        elif split_name == 'test' and hasattr(data_module, 'get_test_data'):
            split_data = data_module.get_test_data()

        if split_data is None:
            print(f"  No {split_name} data available (maybe no_cv mode), skipping...")
            continue

        print(f"Generating predictions for {model.model_name} split={split_name}...")
        # Stream the tall forecast to disk in target-groups to bound peak memory
        # (the full wide-frame + single melt OOMs the 78x20 RF sweep on capped pods).
        _fname = _forecast_filename(
            model_name, data_module.feature_set, split_name, fname_seed)
        _stream_paths = [results_dir / _fname]
        _latest = latest_results_dir / _fname
        if _latest != _stream_paths[0]:
            _stream_paths.append(_latest)
        model.predict(split_data, stream_save_paths=_stream_paths)
        print(f"Predictions streamed to: {', '.join(str(p) for p in _stream_paths)}")

    return

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train forecasting models')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model-only', type=str, help='Train only specific model type')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (force CPU, overfit 1 batch, log every step, etc.)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file. If provided, skip training and only generate predictions.')
    parser.add_argument('--no-cv', action='store_true', help='Skip cross-validation and train on full train+validation data')
    parser.add_argument('--quick', type=float, default=None, help='Run only a fraction of prediction batches (e.g. 0.05 for 5%%)')
    parser.add_argument('--forecast-splits', type=str, default=None, help='Comma-separated splits to forecast (e.g. "test" or "test,val,train"). Default: test')
    parser.add_argument('--comment', type=str, help='Comment for the experiment')
    parser.add_argument(
        '--resume-dir', type=str, default=None,
        help='Re-enter an EXISTING experiment directory (e.g. results/r9e2_random_forest_20260604_120000) '
             'instead of creating a fresh timestamped one. Use this to resume a crashed/OOM-killed baseline '
             'run: any (target, horizon) sub-models already in <resume-dir>/models are skipped and training '
             'picks up where it left off. The per-sub-model streaming + resume is what the Random Forest '
             'sweep (78 targets x 20 horizons) relies on after an OOM.',
    )
    parser.add_argument(
        '--reconcile',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Override inference.reconcile from the config. Pass --reconcile or --no-reconcile to force; omit to use the config value. Useful for inference-only re-runs of an existing checkpoint with reconcile flipped without editing the config.',
    )

    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)

    seed = int(config.get('seed') or 0)
    L.seed_everything(seed, workers=True)
    if 'data' not in config:
        config['data'] = {}
    config['data'].setdefault('seed', seed)

    if args.debug:
        print("DEBUG MODE ENABLED: Adjusting configuration for local debugging...")
        config['debug'] = True

    if args.quick is not None:
        config['quick_fraction'] = args.quick
        print(f"QUICK MODE: Will run only {args.quick*100:.0f}% of prediction batches")

    if args.forecast_splits is not None:
        config['forecast_splits'] = [s.strip() for s in args.forecast_splits.split(',')]

    if args.no_cv:
        print("NO-CV MODE: Will train on combined train+validation data...")
        # Store in data config so data modules can access it
        if 'data' not in config:
            config['data'] = {}
        config['data']['no_cv'] = True

    if args.reconcile is not None:
        # CLI override of inference.reconcile. Read by the predict loop in
        # generate_predictions — only affects forecast generation, not training.
        config.setdefault('inference', {})['reconcile'] = args.reconcile
        print(f"RECONCILE OVERRIDE: inference.reconcile = {args.reconcile}")

    # Get list of models to train
    models_config = config.get('models', {})
    
    if args.model_only:
        if args.model_only not in models_config:
            raise ValueError(f"Model {args.model_only} not found in config")
        models_to_train = {args.model_only: models_config[args.model_only]}
    else:
        models_to_train = models_config

    # Fail fast on incompatible config combinations before any training/inference.
    validate_config(config, models_to_train)

    # --resume-dir only takes effect via the experiment-dir machinery below, which
    # is skipped entirely under --debug (exp_dir stays None). Warn rather than
    # silently ignore it.
    if args.resume_dir and args.debug:
        print("Warning: --resume-dir is ignored under --debug (no experiment "
              "directory is created in debug mode).")

    # Create experiment directory
    exp_dir = None
    if not args.debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config.get('experiment', {}).get('name', 'EXP')
        if args.resume_dir:
            # Resume into an existing experiment dir so the streamed-to-disk
            # baseline sub-models there are found and skipped (see fit() resume).
            exp_dir = Path(args.resume_dir)
            if not exp_dir.exists():
                raise FileNotFoundError(
                    f"--resume-dir {exp_dir} does not exist. Pass the existing experiment "
                    f"directory of the run you want to resume.")
            # models/ holds the streamed sub-models; its absence means there is
            # nothing to resume (e.g. the run died before any model was saved).
            # Not fatal — training just starts fresh into this dir — but warn so a
            # mistyped path doesn't look like a successful resume.
            if not (exp_dir / "models").exists():
                print(f"Warning: --resume-dir {exp_dir} has no models/ subdirectory; "
                      f"nothing to resume — training will start from scratch into it.")
            print(f"Resuming into existing experiment directory: {exp_dir}")
        else:
            exp_id = f"{exp_name}_{timestamp}"
            exp_dir = Path("results") / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save snapshots
        # 1. Config
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # 2. Source Code
        snapshot_dir = exp_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        
        # Copy relevant files
        files_to_copy = [
            Path("src/forma/models/forma.py"),
            Path("src/forma/train.py"),
            Path("src/forma/data/forma_data.py"),
            Path(args.config)
        ]
        for p in files_to_copy:
            if p.exists():
                shutil.copy2(p, snapshot_dir / p.name)

        # 3. Comment / Metadata
        with open(exp_dir / "exp_summary.txt", "w") as f:
            f.write(f"Timestamp: {timestamp}\n")
            exp_desc = config.get('experiment', {}).get('description', 'No description')
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Description: {exp_desc}\n")
            
            # Log Git Commit Hash
            # Prefer FORMA_GIT_SHA env var (set by the RunPod launch wrapper before
            # rsync, since the shipped tree has no .git). Fall back to `git -C <repo>`
            # anchored at the repo root so local CWD doesn't matter.
            commit_hash = os.environ.get("FORMA_GIT_SHA")
            if not commit_hash:
                try:
                    repo_root = Path(__file__).resolve().parent.parent
                    commit_hash = subprocess.check_output(
                        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                        stderr=subprocess.DEVNULL,
                    ).decode().strip()
                except Exception as e:
                    commit_hash = f"Unknown (Error: {e})"
            f.write(f"Git commit: {commit_hash}\n")

            if args.comment:
                f.write(f"Comment: {args.comment}\n")
            f.write("-" * 50 + "\n")

        print(f"\nExperiment initialized at: {exp_dir}")
    else:
        print("\nDEBUG MODE: Skipping experiment directory creation.")
    
    # Train each model
    for model_name, model_config in models_to_train.items():
        print(f"\n{'='*50}")
        if args.checkpoint:
            print(f"Loading {model_name} from checkpoint (skipping training)")
        else:
            print(f"Training {model_name}")
        print(f"{'='*50}")
        
        try:
            # Create model
            model = create_model(model_config, config.get('data', {}))

            # The forecast filename derives from model.model_name, which FeedForwardModel
            # hardcodes to 'ffnn' and FormaModel hardcodes to 'forma'. An explicit
            # `forecast_name` in the model config overrides that token for ANY model type
            # — give every seed-config of one arm the SAME forecast_name so their forecasts
            # share a {model}__{fs} id and (with inference.seed_in_forecast_name: true) form
            # a cross-seed mixture family in evaluate.py; give distinct arms distinct names
            # so they stay separate. Absent forecast_name, FFNN still falls back to the
            # config key (so existing ffnn_current/large/small/linear files are unchanged)
            # and other types keep their type-derived default.
            forecast_name = model_config.get('forecast_name')
            if forecast_name:
                model.model_name = str(forecast_name)
            elif model_config['type'] == 'ffnn':
                model.model_name = model_name
            if model_config['type'] == 'ffnn':
                # Fail fast on an un-set benchmark epoch sentinel (-1) rather than
                # silently training zero epochs on untrained weights.
                if getattr(model, 'epochs', 1) < 1:
                    raise ValueError(
                        f"Model '{model_name}': epochs={model.epochs} is not a valid "
                        f"training length. Set it from the matching *_cv run's best-val "
                        f"epoch before running this benchmark.")

            # Create data module
            data_module = create_data_module(model_config['type'], config.get('data', {}))
            
            if args.checkpoint:
                # Load model from checkpoint instead of training
                checkpoint_path = Path(args.checkpoint)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

                if isinstance(model, L.LightningModule):
                    # Load PyTorch Lightning checkpoint
                    print(f"Loading checkpoint from {checkpoint_path}...")
                    model = model.__class__.load_from_checkpoint(str(checkpoint_path))
                    print(f"Checkpoint loaded successfully.")
                else:
                    # For sklearn/XGBoost models, load from saved path
                    model.load(str(checkpoint_path))
                    print(f"Model loaded from {checkpoint_path}")
            else:
                # Train model
                walk_forward_validation(model, data_module, config, exp_dir=exp_dir)

            # Generate predictions
            if config.get('skip_predict', False) or config.get('debug', False):
                print("Skipping prediction generation.")
            else:
                generate_predictions(model, data_module, config, exp_dir=exp_dir)

            print(f"Successfully completed {'predictions' if args.checkpoint else 'training'} for {model_name}")

        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            if config.get('stop_on_error', False):
                raise
            else:
                print("Continuing with next model...")
                continue
    
    print(f"\n{'='*50}")
    print("Training pipeline completed!")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
