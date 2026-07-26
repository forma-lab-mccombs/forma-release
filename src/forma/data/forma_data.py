"""
Data Module for the "Forma" Model
"""

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import pandas as pd
import numpy as np
import json
import re
import collections
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from torch_geometric.data import Data, Batch
from forma.data.transforms import regularize
from forma.data.config_utils import resolve_dataset_path


VALID_INDUSTRY_MODES = ('none', 'node', 'bias')


def _validate_industry_mode(mode: str) -> str:
    """Raise on unknown industry_mode; return the validated string unchanged."""
    if mode not in VALID_INDUSTRY_MODES:
        raise ValueError(
            f"Unknown industry_mode: {mode!r}. Must be one of {VALID_INDUSTRY_MODES}."
        )
    return mode


class FormaData(Data):
    """Custom Data class with proper batch increment for constraint COO tensors."""

    @property
    def num_nodes(self):
        # PyG uses this to identify node-level tensors and create batch vectors.
        # Without an 'x' attribute, PyG can't infer this automatically.
        if hasattr(self, 'x_id') and self.x_id is not None:
            return self.x_id.size(0)
        return super().num_nodes

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == 'constraint_dst':
            return self.x_id.size(0)
        if key == 'constraint_src':
            return self.num_constraints.item()
        if key == 'x_industry_id':
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        if key in ('constraint_src', 'constraint_dst', 'constraint_sign'):
            return 0
        if key in ('num_constraints', 'num_constraint_edges'):
            return 0
        if key in ('x_industry_id', 'is_industry'):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

# -------------------------------------------------------------------
# 1. Identity Schema Helper
# -------------------------------------------------------------------
class IdentitySchema:
    def __init__(self, raw_identities: List[List[str]], account_map_df: pd.DataFrame):
        """
        Parses the identities JSON list into a fast lookup table.
        raw_identities: List of lists, e.g. [["+atq_0", "-ltq_0", "-teqq_0"], ...]
        account_map_df: DF with 'account_name' and 'account_id'
        """
        self.num_identities = len(raw_identities)
        self.name_to_id = dict(zip(account_map_df['account_name'], account_map_df['account_id']))

        # Lookup: { account_id: [ (identity_idx, lag, sign), ... ] }
        self.account_edge_lookup = {}

        # NEW: (identity_idx, lag) -> set(account_id) required by that identity at that lag.
        # This lets us test whether an identity node is "complete" (all terms present) for a given quarter.
        self.identity_terms_by_lag = collections.defaultdict(set)

        # Regex: (+/-)(name)_(lag)
        # e.g. "-cheq_-1" -> Sign: -, Name: cheq, Lag: -1
        pattern = re.compile(r'([+-])(.+)_([0-9-]+)$')

        for idx, equation in enumerate(raw_identities):
            for term in equation:
                match = pattern.match(term)
                if not match:
                    print(f"Warning: Term '{term}' did not match expected pattern.")
                    continue

                sign_str, name, lag_str = match.groups()

                if name not in self.name_to_id:
                    print(f"Warning: Account name '{name}' not found in account map.")
                    continue  # Skip accounts not in our subset

                acc_id = self.name_to_id[name]
                lag = int(lag_str)
                sign = 1.0 if sign_str == '+' else -1.0

                if acc_id not in self.account_edge_lookup:
                    self.account_edge_lookup[acc_id] = []

                self.account_edge_lookup[acc_id].append((idx, lag, sign))
                self.identity_terms_by_lag[(idx, lag)].add(acc_id)

        # Pre-compute terms indexed by identity for faster lookup in get_edges
        # terms_by_identity[identity_idx] = [(lag, set(account_ids)), ...]
        self.terms_by_identity = collections.defaultdict(list)
        for (idx, lag), acc_ids in self.identity_terms_by_lag.items():
            self.terms_by_identity[idx].append((lag, acc_ids))

    def get_edges(self, acc_ids: np.ndarray, qs: np.ndarray, min_q: int, max_q: int) -> Tuple[List[int], List[int], List[float], np.ndarray, np.ndarray]:
        """
        Calculates edges for a single sample (graph).

        Spec: identity nodes should only be created when *all* required account nodes exist.
        Masked accounts still exist as nodes, so they still count as present.

        Creates only edges from the account nodes to the identity nodes -- the reverse edges are handled in the model.

        Returns:
          account_node_idxs: List[int]
          identity_node_idxs: List[int]  (0..num_id_nodes-1 compact indexing)
          signs: List[float]
          id_node_qs: np.ndarray[int] (num_id_nodes,) quarter for each identity node
          id_node_ids: np.ndarray[int] (num_id_nodes,) identity equation id for each identity node
        """
        # Build maps of which nodes exist at each quarter
        # present_by_q[q] = set(account_id)
        present_by_q = collections.defaultdict(set)
        idx_by_acc_q = {}  # (acc_id, q) -> account node index
        for i, (acc_id, q) in enumerate(zip(acc_ids, qs)):
            present_by_q[int(q)].add(int(acc_id))
            idx_by_acc_q[(int(acc_id), int(q))] = i

        # Determine which identity nodes are valid (complete) within [min_q, max_q]
        valid_identity_nodes = []  # list[(identity_q, identity_idx)]
        for identity_q in range(int(min_q), int(max_q) + 1):
            present_accs = present_by_q.get(identity_q, set())
            if not present_accs:
                continue

            for identity_idx in range(self.num_identities):
                # Use pre-computed terms_by_identity for faster lookup
                terms = self.terms_by_identity.get(identity_idx, [])
                if not terms:
                    continue

                # Check if all required (acc_id, q) nodes exist for this identity
                required_all = set()
                complete = True
                for lag, req_acc_ids in terms:
                    required_q = identity_q + int(lag)
                    if required_q < min_q or required_q > max_q:
                        complete = False
                        break
                    for acc_id in req_acc_ids:
                        if (int(acc_id), int(required_q)) not in idx_by_acc_q:
                            complete = False
                            break
                        required_all.add((int(acc_id), int(required_q)))
                    if not complete:
                        break

                if complete:
                    valid_identity_nodes.append((identity_q, identity_idx))

        # Compact index mapping for identity nodes
        # (identity_q, identity_idx) -> compact id_node_idx
        id_node_map = {k: j for j, k in enumerate(valid_identity_nodes)}

        account_node_idxs: List[int] = []
        identity_node_idxs: List[int] = []
        signs: List[float] = []

        # Create edges only to valid identity nodes
        for (identity_q, identity_idx), id_node_idx in id_node_map.items():
            # Iterate required terms and add the corresponding edges
            for (idx, lag), req_acc_ids in self.identity_terms_by_lag.items():
                if idx != identity_idx:
                    continue

                required_q = int(identity_q) + int(lag)
                for acc_id in req_acc_ids:
                    # Find the sign for this (acc_id, identity_idx, lag)
                    # This is small; linear scan is acceptable and keeps schema representation simple.
                    sign = None
                    for i_idx, i_lag, i_sign in self.account_edge_lookup.get(int(acc_id), []):
                        if i_idx == identity_idx and int(i_lag) == int(lag):
                            sign = float(i_sign)
                            break
                    if sign is None:
                        continue

                    src_node = idx_by_acc_q[(int(acc_id), int(required_q))]
                    account_node_idxs.append(src_node)
                    identity_node_idxs.append(id_node_idx)
                    signs.append(sign)

        id_node_qs = np.array([q for (q, _idx) in valid_identity_nodes], dtype=np.int64)
        id_node_ids = np.array([idx for (_q, idx) in valid_identity_nodes], dtype=np.int64)

        return account_node_idxs, identity_node_idxs, signs, id_node_qs, id_node_ids

# -------------------------------------------------------------------
# 2. Forma Window Dataset
# -------------------------------------------------------------------
class FormaWindowDataset(Dataset):
    def __init__(self, index: List[Tuple[int, int]], lookup: Dict, lookback: int, horizon: int,
                 reg_stats_by_q: Dict, slice_cache: Dict, scale_account_id: int,
                 all_account_ids: List[int] = None, use_future_grid: bool = True,
                 present_in_q0: bool = False, oversample_factor: int = 1,
                 firm_industry_lookup: Dict[int, int] = None,
                 industry_unknown_id: int = 0,
                 future_grid_sample_overhang: bool = True,
                 global_max_q: int = None,
                 tuple_max_abs_zscore: float = float('inf')):
        """
        Args:
            horizon: Forecast horizon (number of quarters into the future).
                     For curriculum learning, the DataLoader is recreated with an updated horizon.
            tuple_max_abs_zscore: Clamp every regularized tuple value to this bound,
                     mirroring the tabular pipeline's max_abs_zscore clip
                     (dataset_creation.py clips features AND future targets at build
                     time). Applied after regularization, so it covers both the model
                     inputs (x_val) and the loss targets (y) for every objective, at
                     train and predict alike. Default inf = no clamping (backwards
                     compatible: canonical runs are byte-unchanged). Reg-space
                     outliers from near-zero-sigma reg-stat cells reach |value| in
                     the thousands otherwise — under MSE that collapsed r10e6 (see
                     archive/eval/r10_eval/FINDINGS.md on tj/r10-eval-findings).
            present_in_q0: If True, future nodes are scoped to accounts observed at q=0 for each
                           sample, matching the tabular present_in_q0 experiment. Accounts not
                           present at q=0 are dropped entirely (not just masked) so their mere
                           existence as graph nodes cannot leak look-ahead information.
            future_grid_sample_overhang: Only consulted when use_future_grid is True.
                           True (default, original behaviour): the synthetic future grid
                           is populated for every quarter 1..horizon, including quarters
                           that overhang the end of the dataset/split (no firm has data
                           that far out). False: grid quarters whose absolute quarter
                           exceeds ``global_max_q`` (the split's last quarter) get no
                           synthetic nodes. This is an ex-ante, firm-uniform truncation
                           ("near the data edge you can only ask for short horizons") and
                           introduces NO look-ahead. It still adds grid nodes when a firm
                           disappears *within* the data period (liquidation etc.) — that
                           absence is the genuine survivorship leak we must keep covering.
            global_max_q: Largest absolute quarter present in this split. Required for
                           future_grid_sample_overhang=False to have any effect; ignored
                           otherwise.
            oversample_factor: How many times to duplicate the index per epoch.
            firm_industry_lookup: Dict mapping firm_id (int) -> industry_id (int) for industry node.
            industry_unknown_id: Fallback id used when a firm is missing from the lookup.
                Defaults to 0 (the lookup is unused in 'none' mode and always populated in
                'node'/'bias' modes); set to FF48 ``unknown_id`` (e.g. 48) when wiring the
                DataModule so any escape doesn't silently mislabel firms as Agriculture.
        """
        self.index = index * oversample_factor
        self.lookup = lookup
        self.lookback = lookback
        self.horizon = horizon
        self.reg_stats_by_q = reg_stats_by_q
        self.slice_cache = slice_cache
        self.scale_account_id = scale_account_id
        self.all_account_ids = all_account_ids
        self.use_future_grid = use_future_grid
        self.present_in_q0 = present_in_q0
        self.future_grid_sample_overhang = future_grid_sample_overhang
        self.global_max_q = global_max_q
        self.firm_industry_lookup = firm_industry_lookup or {}
        self.industry_unknown_id = industry_unknown_id
        self.tuple_max_abs_zscore = tuple_max_abs_zscore


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        firm_id, center_q = self.index[idx]
        data = self.lookup[firm_id]

        # Debug: Log horizon access periodically (every 5000 samples)
        if idx % 100000 == 0:
            print(f"[FormaWindowDataset.__getitem__] idx={idx}, horizon={self.horizon}")

        # Use pre-computed slice indices for initial slice (uses max_lookback/max_horizon)
        start_idx, end_idx = self.slice_cache[(firm_id, center_q)]

        # Direct slice - much faster than boolean mask
        account_ids = data['account_id'][start_idx:end_idx]
        quarters = data['quarter'][start_idx:end_idx]
        values = data['value'][start_idx:end_idx]

        # Filter to the current lookback and horizon (for curriculum learning)
        # This ensures that only data within [center_q - lookback + 1, center_q + horizon] is returned
        start_q = center_q - self.lookback + 1
        end_q = center_q + self.horizon

        # Boolean mask for quarters within the current window
        q_mask = (quarters >= start_q) & (quarters <= end_q)

        account_ids = account_ids[q_mask]
        quarters = quarters[q_mask]
        values = values[q_mask]

        # Center the time so q=0 is the forecast date
        quarters = quarters - center_q

        # Convert to numpy arrays (in case they're pandas arrays)
        account_ids_np = np.asarray(account_ids).astype(np.int64)  # Ensure consistent dtype
        quarters_np = np.asarray(quarters).astype(np.int64)  # Ensure consistent dtype
        values_np = np.asarray(values).astype(np.float32)

        # Experiment 3: Data Availability Mask
        # Initially, all current nodes are available
        data_avail_np = np.ones(len(account_ids_np), dtype=bool)

        # --- present_in_q0: drop future tuples for accounts not observed at q=0 ---
        # Accounts absent at q=0 are removed entirely (not just marked unavailable) so
        # their mere presence as graph nodes cannot leak look-ahead information to the model.
        # Must happen BEFORE the future grid is built so q0_accounts reflects raw observations.
        q0_accounts = None
        if self.present_in_q0:
            q0_accounts = set(account_ids_np[quarters_np == 0].tolist())
            q0_accounts.discard(self.scale_account_id)
            keep = (quarters_np <= 0) | np.isin(account_ids_np, list(q0_accounts))
            account_ids_np = account_ids_np[keep]
            quarters_np = quarters_np[keep]
            values_np = values_np[keep]
            data_avail_np = data_avail_np[keep]

        # Experiment 3: Create grid for future quarters.
        # When present_in_q0 is active, only accounts observed at q=0 get synthetic future nodes.
        if self.horizon > 0 and self.use_future_grid and self.all_account_ids is not None:
            # Find which (account, quarter) pairs are missing for q > 0
            existing_pairs = set(zip(account_ids_np, quarters_np))
            new_accs, new_qs, new_vals = [], [], []
            grid_accounts = list(q0_accounts) if q0_accounts is not None else self.all_account_ids

            for q in range(1, self.horizon + 1):
                # future_grid_sample_overhang=False: stop the grid at the split's last
                # quarter. center_q is the absolute forecast quarter; center_q + q is the
                # absolute future quarter for this grid step. q is ascending, so once we
                # pass global_max_q every later q does too -> break. This cut is ex-ante
                # and identical for every firm (no look-ahead); it does NOT key off the
                # firm's own last quarter, so a firm vanishing within the data period
                # still gets grid nodes (the survivorship leak stays covered).
                if (not self.future_grid_sample_overhang
                        and self.global_max_q is not None
                        and center_q + q > self.global_max_q):
                    break
                for acc_id in grid_accounts:
                    if (acc_id, q) not in existing_pairs:
                        # Skip adding the 'scale' account to future grid (it's only needed at q=0)
                        if acc_id == self.scale_account_id:
                            continue
                        new_accs.append(acc_id)
                        new_qs.append(q)
                        new_vals.append(0.0) # Placeholder
            
            if new_accs:
                account_ids_np = np.concatenate([account_ids_np, np.array(new_accs, dtype=np.int64)])
                quarters_np = np.concatenate([quarters_np, np.array(new_qs, dtype=np.int64)])
                values_np = np.concatenate([values_np, np.array(new_vals, dtype=np.float32)])
                # Synthetic nodes are NOT available (False)
                data_avail_np = np.concatenate([data_avail_np, np.zeros(len(new_accs), dtype=bool)])

        # For the scale account only, drop all nodes with q != 0
        if self.scale_account_id is not None:
            keep_mask = ~((account_ids_np == self.scale_account_id) & (quarters_np != 0))
            account_ids_np = account_ids_np[keep_mask]
            quarters_np = quarters_np[keep_mask]
            values_np = values_np[keep_mask]
            data_avail_np = data_avail_np[keep_mask]

        # Get regularization stats for this center_q and these account_ids
        # Per spec: k, mu, sigma are consistent from q=0 for each account type
        reg_k, reg_mu, reg_sigma = self._get_reg_stats(center_q, account_ids_np, data_avail_np)

        # Scale constant across all accounts. Use the value from the 'scale' account at q=0
        # this is the data['value'] where account_id == self.scale_account_id and quarter == center_q

        # find index in the data where account_id == self.scale_account_id and quarter == center_q
        scale_idx = np.where((account_ids_np == self.scale_account_id) & (quarters_np == 0))[0]

        if len(scale_idx) == 0:
            # Debug: Check what accounts exist at q=0
            q0_mask = quarters_np == 0
            q0_accounts = account_ids_np[q0_mask] if q0_mask.any() else np.array([])
            print(f"Warning: Scale account id {self.scale_account_id} missing for firm_id {firm_id} at center_q {center_q}. "
                  f"Accounts at q=0: {q0_accounts.tolist() if len(q0_accounts) > 0 else 'none'}. "
                  f"Quarters range: [{quarters_np.min() if len(quarters_np) > 0 else 'empty'}..{quarters_np.max() if len(quarters_np) > 0 else 'empty'}]. "
                  f"Defaulting scale to 1.0.")
            scale = 1.0  # Default to 1.0 if scale account is missing
        else:
            scale = float(values_np[scale_idx[0]])

        # Convert to numpy for vectorized operations
        k_np = reg_k.numpy()
        mu_np = reg_mu.numpy()
        sigma_np = reg_sigma.numpy()

        # Step 1: Apply scaling (raw / scale)
        # Per spec: scale is the pre-scaling "z" value, applied to make model scale-invariant
        # Apply to everything EXCEPT the 'scale' account itself
        values_scaled_np = values_np / scale
        values_scaled_np[scale_idx] = scale

        # Step 2: Apply regularization using the canonical transform from transforms.py
        values_regularized_np = regularize(values_scaled_np, k_np, mu_np, sigma_np)

        # Step 3 (opt-in): clamp regularized values to +-tuple_max_abs_zscore, the
        # graph-pipeline analog of the tabular build-time clip. One clamp point
        # covers model inputs and loss targets alike ('values' feeds both x_val
        # and y downstream in the collator). values_scaled stays raw: the
        # constraint machinery works in raw space.
        if np.isfinite(self.tuple_max_abs_zscore):
            np.clip(values_regularized_np, -self.tuple_max_abs_zscore,
                    self.tuple_max_abs_zscore, out=values_regularized_np)

        industry_id = self.firm_industry_lookup.get(firm_id, self.industry_unknown_id)

        return {
            'firm_id': firm_id,
            'account_ids': torch.from_numpy(account_ids_np).long(),
            'quarters': torch.from_numpy(quarters_np).long(),  # Already centered (q=0 is forecast date)
            'values_raw': torch.from_numpy(values_np).float(),  # Original raw values
            'values_scaled': torch.from_numpy(values_scaled_np).float(),  # Scaled but not regularized (for GNN de-regularization)
            'values': torch.from_numpy(values_regularized_np).float(),  # Fully regularized (for model input)
            'center_q_real': center_q,
            'scale': scale,
            'reg_k': reg_k,
            'reg_mu': reg_mu,
            'reg_sigma': reg_sigma,
            'data_avail': torch.from_numpy(data_avail_np).bool(),
            'industry_id': industry_id,
        }

    def _get_reg_stats(self, center_q: int, account_ids: np.ndarray, data_avail: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get regularization stats for the given center_q and account_ids.
        center_q is an integer offset from base quarter (1969 Q4).

        Per spec: k, mu, sigma are consistent from q=0 for each account type.

        Args:
            data_avail: Boolean mask indicating which entries have real data.
                        Only warn about missing stats for accounts with real data.

        Returns three tensors of shape [num_accounts]: k, mu, sigma
        """
        # Convert integer quarter offset to YYYYQQ format
        # Base is 1969 Q4 = period index 0
        # quarter offset n -> add n quarters to base
        base_period = pd.Period("1969-12-31", freq="Q")
        data_period = base_period + center_q
        # Convert to YYYYQQ integer: year * 100 + quarter
        yyyyqq = data_period.year * 100 + data_period.quarter

        # Look up stats for this quarter
        if yyyyqq not in self.reg_stats_by_q:
            # Fallback: use nearest available quarter or raise
            # For now, raise to make missing data explicit
            raise KeyError(f"No regularization stats found for center_q={center_q} (YYYYQQ={yyyyqq})")

        stats_dict = self.reg_stats_by_q[yyyyqq]

        # Optimization: Vectorized lookup using a pre-built array if possible,
        # but since account_ids are sparse/arbitrary, we use a fast list comp or map.
        # Given that stats_dict is small (~50 keys) and account_ids is large (~1000) with repeats,
        # we can optimize by fetching unique stats first.

        # Handle empty account_ids
        if len(account_ids) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        # 1. Get unique account IDs to minimize dictionary lookups
        unique_ids, inverse_indices = np.unique(account_ids, return_inverse=True)

        # 2. Fetch stats for unique IDs
        # We use a list comprehension which is fast for small N (~50)
        # Use fallback for missing stats (robustness for Experiment 3)
        # Only warn for accounts that have real (available) data in this sample
        if data_avail is not None:
            avail_account_ids = set(account_ids[data_avail].tolist())
        else:
            avail_account_ids = set(account_ids.tolist())
        unique_stats = []
        for uid in unique_ids:
            if uid in stats_dict:
                unique_stats.append(stats_dict[uid])
            else:
                if uid in avail_account_ids:
                    logging.debug(f"Missing regularization stats for account_id={uid} at center_q={center_q}. Using default (1.0, 0.0, 1.0)")
                unique_stats.append((1.0, 0.0, 1.0))

        # 3. Convert to tensor (shape [num_unique, 3] for k, mu, sigma)
        unique_stats_tensor = torch.tensor(unique_stats, dtype=torch.float)

        # Ensure 2D shape even with single unique ID
        if unique_stats_tensor.dim() == 1:
            unique_stats_tensor = unique_stats_tensor.unsqueeze(0)

        # 4. Broadcast back to full size using inverse indices
        # inverse_indices is a numpy array, convert to torch long for indexing
        indices = torch.from_numpy(inverse_indices).long()

        # Gather: [num_unique, 3] -> [num_total, 3]
        full_stats = unique_stats_tensor[indices]

        # Split into k, mu, sigma
        return full_stats[:, 0], full_stats[:, 1], full_stats[:, 2]

# -------------------------------------------------------------------
# 3. Graph Collator (The Logic Core)
# -------------------------------------------------------------------
class GraphCollator:
    def __init__(self, schema: IdentitySchema, curriculum_config: Dict, industry_mode: str = 'none', seed: Optional[int] = None):
        self.schema = schema
        self.curriculum_config = curriculum_config
        self.industry_mode = _validate_industry_mode(industry_mode)
        # Owns its own Generator (rather than calling np.random.*) because Lightning's
        # seed_everything(workers=True) reseeds the global RNG per worker; a dedicated
        # instance is what makes masking deterministic. Built lazily on first __call__
        # so we can mix in the worker id and avoid lockstep correlation across the
        # num_workers forks that all start from the same parent state.
        self._seed = seed
        self.rng: Optional[np.random.Generator] = None

    def __call__(self, batch_list: List[Dict[str, Union[torch.Tensor, int]]]) -> Batch:
        if self.rng is None:
            wi = torch.utils.data.get_worker_info()
            worker_id = wi.id if wi is not None else 0
            self.rng = np.random.default_rng(None if self._seed is None else [self._seed, worker_id])

        data_objs = []
        num_empty_samples = 0

        for sample in batch_list:
            acc_ids = sample['account_ids']
            qs = sample['quarters']
            vals = sample['values']
            vals_scaled = sample['values_scaled']
            reg_k = sample['reg_k']
            reg_mu = sample['reg_mu']
            reg_sigma = sample['reg_sigma']

            if len(qs) == 0:
                num_empty_samples += 1
                import warnings
                warnings.warn(
                    f"Empty sample encountered (firm_id={sample.get('firm_id', 'unknown')}, "
                    f"center_q={sample.get('center_q_real', 'unknown')}). "
                    f"This will shrink the batch size and may indicate data quality issues.",
                    UserWarning
                )
                continue

            min_q, max_q = qs.min().item(), qs.max().item()
            num_acc_nodes = len(acc_ids)

            account_node_idxs, identity_node_idxs, signs, id_qs_np, id_ids_np = self.schema.get_edges(
                acc_ids.numpy(), qs.numpy(), min_q, max_q
            )

            valid_id_instances = set(zip(id_ids_np, id_qs_np))

            num_id_nodes = int(id_qs_np.shape[0])

            acct_mask = self.get_acct_mask(acc_ids, qs, valid_id_instances, sample.get('data_avail'))

            data_avail = sample.get('data_avail')
            if data_avail is None:
                data_avail = torch.ones(num_acc_nodes, dtype=torch.bool)

            input_vals = vals.clone()
            input_vals[acct_mask] = 0.0

            input_vals_scaled = vals_scaled.clone()
            input_vals_scaled[acct_mask] = 0.0

            num_constraint_edges = len(account_node_idxs)
            if num_constraint_edges > 0:
                constraint_src = torch.tensor(identity_node_idxs, dtype=torch.long)
                constraint_dst = torch.tensor(account_node_idxs, dtype=torch.long)
                constraint_sign = torch.tensor(signs, dtype=torch.float)
            else:
                constraint_src = torch.empty((0,), dtype=torch.long)
                constraint_dst = torch.empty((0,), dtype=torch.long)
                constraint_sign = torch.empty((0,), dtype=torch.float)

            # --- Assemble node tensors based on industry_mode ---
            industry_id = int(sample.get('industry_id', 0))

            if self.industry_mode == 'node':
                # Append one industry node (q=0, val=0, never masked) after account nodes.
                ind_reg_k = torch.tensor([1.0], dtype=torch.float)
                ind_reg_mu = torch.tensor([0.0], dtype=torch.float)
                ind_reg_sigma = torch.tensor([1.0], dtype=torch.float)

                all_x_id = torch.cat([acc_ids, torch.tensor([0], dtype=torch.long)])
                all_x_time = torch.cat([qs, torch.tensor([0], dtype=torch.long)])
                all_x_val = torch.cat([input_vals, torch.tensor([0.0], dtype=torch.float)])
                all_x_val_scaled = torch.cat([input_vals_scaled, torch.tensor([0.0], dtype=torch.float)])
                all_mask = torch.cat([acct_mask, torch.tensor([False], dtype=torch.bool)])
                all_data_avail = torch.cat([data_avail, torch.tensor([True], dtype=torch.bool)])
                all_y = torch.cat([vals, torch.tensor([0.0], dtype=torch.float)])
                all_y_scaled = torch.cat([vals_scaled, torch.tensor([0.0], dtype=torch.float)])
                all_reg_k = torch.cat([reg_k, ind_reg_k])
                all_reg_mu = torch.cat([reg_mu, ind_reg_mu])
                all_reg_sigma = torch.cat([reg_sigma, ind_reg_sigma])

                x_industry_id = torch.cat([
                    torch.zeros(num_acc_nodes, dtype=torch.long),
                    torch.tensor([industry_id], dtype=torch.long),
                ])
                is_industry = torch.cat([
                    torch.zeros(num_acc_nodes, dtype=torch.bool),
                    torch.tensor([True], dtype=torch.bool),
                ])
            elif self.industry_mode == 'bias':
                # Same nodes as 'none'; attach per-node industry_id so the model can
                # add industry_embedding as a bias to every node.
                all_x_id = acc_ids
                all_x_time = qs
                all_x_val = input_vals
                all_x_val_scaled = input_vals_scaled
                all_mask = acct_mask
                all_data_avail = data_avail
                all_y = vals
                all_y_scaled = vals_scaled
                all_reg_k = reg_k
                all_reg_mu = reg_mu
                all_reg_sigma = reg_sigma
                x_industry_id = torch.full((num_acc_nodes,), industry_id, dtype=torch.long)
                is_industry = torch.zeros(num_acc_nodes, dtype=torch.bool)
            else:  # 'none'
                all_x_id = acc_ids
                all_x_time = qs
                all_x_val = input_vals
                all_x_val_scaled = input_vals_scaled
                all_mask = acct_mask
                all_data_avail = data_avail
                all_y = vals
                all_y_scaled = vals_scaled
                all_reg_k = reg_k
                all_reg_mu = reg_mu
                all_reg_sigma = reg_sigma
                x_industry_id = torch.zeros(num_acc_nodes, dtype=torch.long)
                is_industry = torch.zeros(num_acc_nodes, dtype=torch.bool)

            data_kwargs = dict(
                x_id=all_x_id,
                x_time=all_x_time,
                x_val=all_x_val,
                x_val_scaled=all_x_val_scaled,
                mask=all_mask,
                data_avail=all_data_avail,
                y=all_y,
                y_scaled=all_y_scaled,
                reg_k=all_reg_k,
                reg_mu=all_reg_mu,
                reg_sigma=all_reg_sigma,
                constraint_src=constraint_src,
                constraint_dst=constraint_dst,
                constraint_sign=constraint_sign,
                num_constraints=torch.tensor([num_id_nodes], dtype=torch.long),
                num_constraint_edges=torch.tensor([num_constraint_edges], dtype=torch.long),
                firm_id=torch.tensor([sample['firm_id']], dtype=torch.long),
                center_q=torch.tensor([sample['center_q_real']], dtype=torch.long),
                scale_val=torch.tensor([sample['scale']], dtype=torch.float),
            )
            # Only attach industry tensors when in an active mode, to preserve
            # backwards-compatible batches in 'none' mode.
            if self.industry_mode != 'none':
                data_kwargs['x_industry_id'] = x_industry_id
                data_kwargs['is_industry'] = is_industry

            data = FormaData(**data_kwargs)
            data_objs.append(data)

        if num_empty_samples > 0:
            import warnings
            warnings.warn(
                f"GraphCollator dropped {num_empty_samples}/{len(batch_list)} empty samples in this batch. "
                f"Final batch size: {len(data_objs)}",
                UserWarning
            )

        batch = Batch.from_data_list(data_objs)

        # PyG 2.7+ may not create batch vectors for custom Data classes
        # that lack an 'x' attribute.  Build it ourselves if missing.
        if batch.batch is None and len(data_objs) > 0:
            sizes = torch.tensor([d.num_nodes for d in data_objs], dtype=torch.long)
            batch.batch = torch.repeat_interleave(
                torch.arange(len(data_objs), dtype=torch.long), sizes
            )

        return batch

    def get_acct_mask(self, acc_ids: torch.Tensor, qs: torch.Tensor, valid_id_instances: set, data_avail: torch.Tensor = None) -> torch.Tensor:
        mask = torch.zeros(len(acc_ids), dtype=torch.bool)
        
        # Experiment 3: Synthetic nodes are ALWAYS masked
        if data_avail is not None:
            mask = mask | (~data_avail)

        # Optimization: Converting to numpy for faster iteration in Python
        acc_ids_np = acc_ids.numpy()
        qs_np = qs.numpy()

        # --- 1. Future Masking (Temporally Grouped) ---
        future_indices = torch.nonzero(qs > 0, as_tuple=True)[0]

        if len(future_indices) > 0:
            pro_forma_prob = self.curriculum_config.get('pro_forma_mask_probability', 0.5)

            # Scenario 1: Pure Forecasting (Mask all future)
            if self.rng.random() < pro_forma_prob:
                mask[future_indices] = True
            else:
                # Scenario 2: Scenario Analysis (Mask subset)
                future_mask_prob = self.curriculum_config.get('future_mask_probability', 0.95)
                # Generate random probabilities for each future node
                node_probs = self.rng.random(len(future_indices))
                mask_subset = node_probs < future_mask_prob
                mask[future_indices[mask_subset]] = True

        # --- 2. History Masking (Identity Grouped) ---
        # Only considers q <= 0
        group_mask_prob = self.curriculum_config.get('identity_group_mask_probability', 0.1)

        # Map: (identity_idx, identity_q) -> list of node indices
        groups = collections.defaultdict(list)

        acc_ids_ungrouped = set(acc_ids_np)

        for i, (acc_id, q) in enumerate(zip(acc_ids_np, qs_np)):
            if q > 0:
                continue  # Skip future nodes for this step

            if acc_id in self.schema.account_edge_lookup:
                for identity_idx, lag, _ in self.schema.account_edge_lookup[acc_id]:
                    # Identity time: t_id = t_acc - lag
                    id_q = int(q) - int(lag)

                    # Only group mask if the identity instance exists and is complete
                    if (identity_idx, id_q) in valid_id_instances:
                        groups[(int(identity_idx), int(id_q))].append(i)
                        acc_ids_ungrouped.discard(int(acc_id))

        # Iterate over groups and apply masking logic
        # Vectorize: batch random decisions for groups
        if groups:
            group_items = list(groups.items())
            group_decisions = self.rng.random(len(group_items)) < group_mask_prob

            for (_, nodes), should_mask in zip(group_items, group_decisions):
                N = len(nodes)
                if N < 2:
                    continue  # Cannot mask k >= 2 items

                # Decide whether to mask this group
                if should_mask:
                    # Choose k uniformly from [2, N]
                    k = self.rng.integers(2, N + 1)

                    # Select k nodes randomly from this group
                    nodes_array = np.array(nodes)
                    selected_indices = self.rng.choice(nodes_array, size=k, replace=False)

                    mask[selected_indices] = True

        # mask ungrouped account-quarters with a small probability
        ungrouped_mask_prob = self.curriculum_config.get('ungrouped_mask_probability', 0.05)

        # Vectorized: random mask among ungrouped historical nodes
        # Build boolean mask for ungrouped account nodes
        is_ungrouped = np.array([int(acc_id) in acc_ids_ungrouped for acc_id in acc_ids_np])
        is_historical = qs_np <= 0
        ungrouped_historical = is_ungrouped & is_historical

        # Generate random values for all nodes at once and apply mask
        if ungrouped_historical.any():
            ungrouped_indices = np.where(ungrouped_historical)[0]
            random_vals = self.rng.random(len(ungrouped_indices))
            mask_these = ungrouped_indices[random_vals < ungrouped_mask_prob]
            mask[mask_these] = True

        return mask


class EmbeddingCollator(GraphCollator):
    """No-mask collator for BERT-style embedding extraction.

    All account nodes are visible (no masking), so the model sees the full
    context and produces contextual hidden states for every node.
    """

    def get_acct_mask(self, acc_ids, qs, valid_id_instances, data_avail=None):
        return torch.zeros(len(acc_ids), dtype=torch.bool)


class DeterministicForecastCollator(GraphCollator):
    """Inference-time collator for benchmark-style forecasting.

    Visible inputs:
      - all account nodes with q <= 0
    Forecast targets (masked):
      - all account nodes with q > 0

    No random masking is applied.
    """

    def get_acct_mask(self, acc_ids: torch.Tensor, qs: torch.Tensor, valid_id_instances: set, data_avail: torch.Tensor = None) -> torch.Tensor:
        # Mask all future account nodes deterministically.
        # Note: Identity nodes are handled/padded in __call__ (always unmasked).
        mask = (qs > 0).clone().detach()
        if data_avail is not None:
            mask = mask | (~data_avail)
        return mask


class ScenarioCollator(DeterministicForecastCollator):
    """Scenario-conditioning collator (publication plan §4.8, exhibit F2).

    Like DeterministicForecastCollator (mask all future nodes, no random
    masking), except future nodes of the ``pinned_account_ids`` are left
    VISIBLE, so their true realized values feed the model as inputs — the
    inference-time analog of the pinned-future training pattern (50% of
    training samples reveal ~5% of future tuples).

    Only real future tuples can be pinned: synthetic future-grid nodes
    (data_avail=False) have no true value and stay masked regardless. The
    predict loop emits rows for every q>0 node, so pinned accounts still
    appear in the output file — downstream analysis must EXCLUDE the pinned
    accounts (their "predictions" are reconstructions of visible inputs).
    """

    def __init__(self, schema: IdentitySchema, curriculum_config: Dict,
                 industry_mode: str = 'none', pinned_account_ids=()):
        super().__init__(schema, curriculum_config, industry_mode=industry_mode)
        self.pinned_account_ids = {int(a) for a in pinned_account_ids}

    def get_acct_mask(self, acc_ids: torch.Tensor, qs: torch.Tensor, valid_id_instances: set, data_avail: torch.Tensor = None) -> torch.Tensor:
        mask = (qs > 0).clone().detach()
        if self.pinned_account_ids:
            pinned = torch.from_numpy(
                np.isin(acc_ids.numpy(), list(self.pinned_account_ids)))
            mask &= ~pinned
        if data_avail is not None:
            mask = mask | (~data_avail)
        return mask


# -------------------------------------------------------------------
# 4. Worker Initialization for Reproducible RNG
# -------------------------------------------------------------------
def worker_init_fn(worker_id):
    """
    Initialize each DataLoader worker with a unique but reproducible seed.
    This ensures that masking randomness is reproducible across runs when
    a global seed is set, while still allowing each worker to have independent
    random streams.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # Note: Each GraphCollator instance in a worker will use the worker's
    # seeded numpy RNG state


class FormaDataModule(L.LightningDataModule):
    """
    Lightning DataModule for financial statement data.
    
    This DataModule handles loading and preprocessing of financial data
    for the GNN-Transformer model.
    """

    def __init__(
            self,
            data_dir_path: Path,  # Path to folder with processed data
            account_map_path: Path,  # Path to account_id_map.csv
            firm_id_map_path: Path, # Path to firm_id_map.csv
            data_config: Dict # Configuration dictionary, has keys for identities_path, batch_size, window_size
    ):
        super().__init__()
        # 1. Store configuration ONLY
        self.data_dir_path = data_dir_path
        self.account_map_path = account_map_path
        self.firm_id_map_path = firm_id_map_path
        self.data_config = data_config  # Store full config for later access (e.g., no_cv flag)
        self.use_future_grid = data_config.get('use_future_grid', True)
        self.present_in_q0 = data_config.get('targets', {}).get('present_in_q0', False)
        # Only consulted when use_future_grid is True (default True = original behaviour).
        self.future_grid_sample_overhang = data_config.get('future_grid_sample_overhang', True)
        # Per-split largest absolute quarter; set in setup() once the split df is loaded.
        self.train_global_max_q = None
        self.val_global_max_q = None
        self.test_global_max_q = None
        self.identities_json_path = data_config.get("identities_path", str(Path(data_dir_path) / 'identities.json'))
        self.feature_set = data_config.get('feature_set', 'core')
        self.dataset_tag = data_config.get('dataset_tag')
        self.batch_size = data_config.get('batch_size', 32)
        self.num_workers = data_config.get('num_workers', 4)

        # Opt-in clamp on regularized tuple values (graph analog of the tabular
        # build-time max_abs_zscore clip; see FormaWindowDataset docstring).
        # Single config value under feature_engineering; default inf = off.
        self.tuple_max_abs_zscore = float(
            (data_config.get('feature_engineering') or {}).get(
                'tuple_max_abs_zscore', float('inf')))
        if not self.tuple_max_abs_zscore > 0:
            raise ValueError(
                f"feature_engineering.tuple_max_abs_zscore must be positive "
                f"(got {self.tuple_max_abs_zscore}).")

        self.industry_mode = _validate_industry_mode(data_config.get('industry_mode', 'none'))
        self.industry_id_map_path = data_config.get('industry_id_map_path', str(Path(data_dir_path) / 'industry_id_map.csv'))

        # 2. Placeholders for state (to be filled in setup)
        self.account_id_map = None
        self.scale_account_id = None
        self.identity_schema = None
        self.reg_stats_by_q = None
        self.firm_industry_lookup = None
        self.num_industries = None
        self.industry_unknown_id = 0
        self.train_slice_cache = None
        self.val_slice_cache = None
        self.test_slice_cache = None
        self.train_lookup = None
        self.val_lookup = None
        self.test_lookup = None
        self.train_index = []
        self.val_index = []
        self.test_index = []

        # 3. Curriculum learning parameters
        self.curriculum_config = data_config.get('curriculum', {})
        self.min_horizon = self.curriculum_config.get('min_horizon', 1)
        self.initial_horizon = self.curriculum_config.get('initial_horizon', 4)
        self.horizon_step = self.curriculum_config.get('horizon_step', 1)
        self.epochs_per_step = self.curriculum_config.get('epochs_per_step', 5)
        self.max_horizon = self.curriculum_config.get('max_horizon', 20)
        self.min_lookback = self.curriculum_config.get('min_lookback', 1)
        self.max_lookback = self.curriculum_config.get('max_lookback', 20)

        # 4. Curriculum state
        self.current_horizon = self.curriculum_config.get('initial_horizon', 4)
        self._current_epoch = 0

    def step_curriculum(self, epoch: Optional[int] = None) -> bool:
        """
        Step the curriculum based on the current epoch.

        Args:
            epoch: Current epoch number. If None, uses internal counter.

        Returns:
            True if the horizon was increased, False otherwise.
        """
        if epoch is not None:
            self._current_epoch = epoch
        else:
            self._current_epoch += 1

        print(f"[step_curriculum] Called with epoch={epoch}, _current_epoch={self._current_epoch}, "
              f"current_horizon={self.current_horizon}, epochs_per_step={self.epochs_per_step}")

        # Check if it's time to expand the horizon
        if self._current_epoch > 0 and self._current_epoch % self.epochs_per_step == 0:
            old_horizon = self.current_horizon
            self.current_horizon = min(self.current_horizon + self.horizon_step, self.max_horizon)
            if self.current_horizon > old_horizon:
                print(f"[step_curriculum] *** HORIZON EXPANDED *** from {old_horizon} to {self.current_horizon} "
                      f"at epoch {self._current_epoch}")
                # Note: CurriculumCallback will call trainer.reset_train_dataloader() to apply the new horizon
                return True
            else:
                print(f"[step_curriculum] Horizon at max ({self.max_horizon}), no expansion")
        else:
            print(f"[step_curriculum] No expansion: epoch {self._current_epoch} % {self.epochs_per_step} != 0")
        return False

    def get_curriculum_state(self) -> Dict:
        """Return the current curriculum state for logging."""
        return {
            'current_horizon': self.current_horizon,
            'max_horizon': self.max_horizon,
            'current_epoch': self._current_epoch,
            'epochs_per_step': self.epochs_per_step,
        }

    def _split_path(self, kind: str, ext: str = "parquet") -> Path:
        return resolve_dataset_path(
            Path(self.data_dir_path), kind, self.feature_set, self.dataset_tag, ext=ext
        )

    def prepare_data(self):
        """
        Check if processed data exists, otherwise run make_dataset.py
        """

        required_paths = [
            self._split_path('tuple_train'),
            self._split_path('tuple_test'),
            self._split_path('tuple_val'),
            self._split_path('regularization_stats'),
        ]

        missing_files = [str(p) for p in required_paths if not p.exists()]

        # Check account_map_path, firm_id_map_path, and identities_json_path
        if not Path(self.account_map_path).exists():
            missing_files.append(self.account_map_path)
        if not Path(self.firm_id_map_path).exists():
            missing_files.append(self.firm_id_map_path)
        if not Path(self.identities_json_path).exists():
            missing_files.append(self.identities_json_path)
        
        if missing_files:
            print(f"Missing processed data files: {missing_files}")
            print("Please build the ProForma-20Q dataset first with the companion package: proforma20q build")
            raise FileNotFoundError(f"Required processed data files not found: {missing_files}")

        # Validate regularization stats schema
        reg_stats_path = self._split_path('regularization_stats')
        print(f"Loading regularization stats from {reg_stats_path}")
        reg_stats_sample = pd.read_parquet(reg_stats_path)
        required_cols = {'quarter', 'mu', 'sigma', 'feature', 'k'}
        if not required_cols.issubset(reg_stats_sample.columns):
            missing_cols = required_cols - set(reg_stats_sample.columns)
            raise ValueError(f"Regularization stats file missing required columns: {missing_cols}")

    def setup(self, stage: Optional[str] = None):
        # 1. Always load small metadata/helpers
        # account_id_map
        self.account_id_map = pd.read_csv(self.account_map_path)

        # Load regularization stats and build indexed lookup
        if not hasattr(self, 'reg_stats_by_q') or self.reg_stats_by_q is None:
            print("Loading regularization stats...")
            reg_stats_df = pd.read_parquet(self._split_path('regularization_stats'))

            # Create account_name -> account_id mapping
            name_to_id = dict(zip(self.account_id_map['account_name'], self.account_id_map['account_id']))

            # Build nested dict: {quarter: {account_id: (scale, k, mu, sigma)}}
            # Note: scale, k, mu, sigma should be consistent from q=0 for each account across all q
            self.reg_stats_by_q = {}

            # Convert quarter to integer quarter (YYYYQQ format)
            reg_stats_df['quarter_int'] = reg_stats_df['quarter'].dt.year * 100 + reg_stats_df['quarter'].dt.quarter

            for _, row in reg_stats_df.iterrows():
                q_int = row['quarter_int']
                feature_name = row['feature']

                if feature_name not in name_to_id:
                    # Skip features not in our account map
                    continue

                acc_id = name_to_id[feature_name]
                k = row['k']
                mu = row['mu']
                sigma = row['sigma']

                if q_int not in self.reg_stats_by_q:
                    self.reg_stats_by_q[q_int] = {}

                self.reg_stats_by_q[q_int][acc_id] = (k, mu, sigma)

            scale_id = name_to_id.get('scale', None)
            # Ensure scale_account_id is a Python int for consistent comparisons with numpy arrays
            self.scale_account_id = int(scale_id) if scale_id is not None else None

            print(f"Loaded regularization stats for {len(self.reg_stats_by_q)} quarters")

        if self.identity_schema is None:
            print(f"Loading identities from: {self.identities_json_path}")
            identities_path = Path(self.identities_json_path)
            if not identities_path.exists():
                raise FileNotFoundError(f"Identities file not found: {identities_path}")

            with open(identities_path, 'r', encoding='utf-8-sig') as f:
                # load identities json, which is a list of lists. Each inner list contains the variable names for one identity
                # each element has a prefix indicating the sign with which it enters the identity, and a suffix indicating the quarter lag,
                # e.g. a list like ["+atq_0", "-ltq_0", "-teqq_0"] indicates that the identity is:
                # assets (atq) minus liabilities (ltq) minus total equity (teqq) at quarter lag 0 equals zero
                # (identities always phrased as a sum equal to zero)
                raw = json.load(f)
                identities = [entry["terms"] if isinstance(entry, dict) else entry for entry in raw]
                self.identity_schema = IdentitySchema(identities, self.account_id_map)

        # Load industry mapping (firm_id -> industry_id). Skipped when industry_mode='none'.
        if self.firm_industry_lookup is None:
            if self.industry_mode == 'none':
                self.num_industries = 0
                self.industry_unknown_id = 0
                self.firm_industry_lookup = {}
            else:
                industry_map_path = Path(self.industry_id_map_path)
                if not industry_map_path.exists():
                    raise FileNotFoundError(
                        f"industry_mode={self.industry_mode!r} requires an industry map at "
                        f"{industry_map_path}. Re-run make_dataset.py with the current code "
                        f"to regenerate it."
                    )
                ind_map = pd.read_csv(industry_map_path)
                self.num_industries = int(ind_map['industry_id'].max()) + 1
                # By FF48 convention, the largest id is the 'Unknown' bucket; that's where
                # firms missing from the lookup should land (rather than industry 0 = Agric).
                self.industry_unknown_id = self.num_industries - 1
                print(f"Loaded industry map: {self.num_industries} industries (unknown_id={self.industry_unknown_id})")
                self.firm_industry_lookup = {}

        # 2. Conditional Loading
        # 'fit' = Training + Validation
        if stage == 'fit' or stage is None:
            print("Loading Train/Val splits...")
            df_train = pd.read_parquet(self._split_path('tuple_train'))
            df_val = pd.read_parquet(self._split_path('tuple_val'))

            self._update_firm_industry_lookup(df_train)
            self._update_firm_industry_lookup(df_val)

            # If no_cv mode, combine train and validation data for training
            no_cv = self.data_config.get('no_cv', False)
            if no_cv:
                print("NO-CV MODE: Combining train and validation data for training...")
                df_train = pd.concat([df_train, df_val], ignore_index=True)
                # Build only train lookup (val will be empty)
                self.train_lookup, self.train_index, self.train_slice_cache = self._build_lookup_and_index(df_train)
                self.val_lookup, self.val_index, self.val_slice_cache = {}, [], {}
                self.train_global_max_q = int(df_train['quarter'].max())
            else:
                # Build separate lookups (includes pre-computed slice cache)
                self.train_lookup, self.train_index, self.train_slice_cache = self._build_lookup_and_index(df_train)
                self.val_lookup, self.val_index, self.val_slice_cache = self._build_lookup_and_index(df_val)
                self.train_global_max_q = int(df_train['quarter'].max())
                self.val_global_max_q = int(df_val['quarter'].max())

        # 'test' = Test set only
        if stage == 'test':
            print("Loading Test split...")
            df_test = pd.read_parquet(self._split_path('tuple_test'))
            self._update_firm_industry_lookup(df_test)
            self.test_lookup, self.test_index, self.test_slice_cache = self._build_lookup_and_index(df_test)
            self.test_global_max_q = int(df_test['quarter'].max())

        # 'predict' = Inference mode
        if stage == 'predict':
            # Load whatever logic you need for inference
            print("Setting up for prediction...")
            print("Prediction setup not yet implemented.")
            pass

    def _update_firm_industry_lookup(self, df: pd.DataFrame):
        """Extract firm_id -> industry_id mapping from tuple DataFrame."""
        if self.industry_mode == 'none':
            return
        if 'industry_id' not in df.columns:
            return
        pairs = df[['firm_id', 'industry_id']].drop_duplicates('firm_id')
        self.firm_industry_lookup.update(
            dict(zip(pairs['firm_id'].astype(int), pairs['industry_id'].astype(int)))
        )

    def _build_lookup_and_index(self, df):
        lookup = {}
        index = []
        slice_cache = {}  # Pre-computed slice indices for each (firm_id, center_q)

        # Grouping by firm (pandas groupby is slow, so we iterate once)
        grouped = df.groupby('firm_id')

        for firm_id, group in grouped:
            # 1. Store Arrays (Fast O(1) retrieval)
            group = group.sort_values('quarter')

            qs = group['quarter'].values
            accs = group['account_id'].values.astype(np.int64)  # Ensure consistent dtype
            vals = group['value'].values

            lookup[firm_id] = {
                'quarter': qs,
                'account_id': accs,
                'value': vals
            }

            # Identify quarters where scale account is present
            scale_qs_set = None
            if self.scale_account_id is not None:
                scale_mask = (accs == self.scale_account_id)
                # Convert to set of python objects (e.g. ints) to avoid potential numpy scalar type mismatch
                scale_qs_set = set(qs[scale_mask].tolist())

            # 2. Build Valid Centers
            # A valid center 'c' must have data in range [c - half, c + half]
            # We don't need *data* at every point, just that the *time range* exists within the firm's lifespan.

            # Find the min and max quarters this firm actually has
            if len(qs) == 0: continue
            min_q, max_q = qs[0], qs[-1]

            # Define valid range for the center "forecast date"
            start_center = min_q + self.min_lookback - 1
            end_center = max_q - self.min_horizon

            if end_center >= start_center:
                # Create a list of all valid integer quarters between start and end
                # casting to int to be safe
                valid_qs = list(range(int(start_center), int(end_center) + 1))

                # Pre-compute slice indices for each valid center_q
                # This avoids boolean masking in __getitem__
                filtered_valid_qs = []
                for center_q in valid_qs:
                    # Filter: require scale account at center_q
                    if scale_qs_set is not None:
                        if center_q not in scale_qs_set:
                            continue

                    start_q = center_q - self.max_lookback + 1
                    end_q = center_q + self.max_horizon

                    # Binary search for start and end indices since qs is sorted
                    start_idx = np.searchsorted(qs, start_q, side='left')
                    end_idx = np.searchsorted(qs, end_q, side='right')

                    if start_idx < end_idx:
                        # Validate: ensure scale account at center_q is actually in the slice
                        if self.scale_account_id is not None:
                            slice_accs = accs[start_idx:end_idx]
                            slice_qs = qs[start_idx:end_idx]
                            scale_in_slice = np.any((slice_accs == self.scale_account_id) & (slice_qs == center_q))

                            if not scale_in_slice:
                                # This shouldn't happen if our filtering is correct
                                print(f"Warning: Scale account validation failed for firm_id {firm_id} at center_q {center_q}. "
                                      f"Scale quarters in data: {scale_qs_set}. Skipping.")
                                continue

                        slice_cache[(firm_id, center_q)] = (start_idx, end_idx)
                        filtered_valid_qs.append(center_q)

                # Append to index: (firm_id, center_q)
                index.extend([(firm_id, q) for q in filtered_valid_qs])

        return lookup, index, slice_cache

    def train_dataloader(self):
        print(f"[train_dataloader] Creating DataLoader with horizon={self.current_horizon}")
        return DataLoader(
            # Pass current_horizon by value; CurriculumCallback calls trainer.reset_train_dataloader()
            # to recreate the DataLoader with updated horizon when curriculum advances
            FormaWindowDataset(self.train_index, self.train_lookup, self.max_lookback, self.current_horizon,
                               self.reg_stats_by_q, self.train_slice_cache, self.scale_account_id,
                               all_account_ids=self.account_id_map['account_id'].unique().tolist(),
                               use_future_grid=self.use_future_grid,
                               present_in_q0=self.present_in_q0,
                               future_grid_sample_overhang=self.future_grid_sample_overhang,
                               global_max_q=self.train_global_max_q,
                               oversample_factor=self.curriculum_config.get('oversample_factor', 1),
                               firm_industry_lookup=self.firm_industry_lookup,
                               industry_unknown_id=self.industry_unknown_id,
                               tuple_max_abs_zscore=self.tuple_max_abs_zscore),
            batch_size=self.batch_size,
            collate_fn=GraphCollator(
                self.identity_schema,
                self.curriculum_config,
                industry_mode=self.industry_mode,
                seed=self.data_config.get('seed'),
            ),
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=(self.num_workers > 0),
            persistent_workers=False, # persistent_workers=(self.num_workers > 0),
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            # Validation always uses full horizon (max_horizon)
            FormaWindowDataset(self.val_index, self.val_lookup, self.max_lookback, self.max_horizon,
                               self.reg_stats_by_q, self.val_slice_cache, self.scale_account_id,
                               all_account_ids=self.account_id_map['account_id'].unique().tolist(),
                               use_future_grid=self.use_future_grid,
                               present_in_q0=self.present_in_q0,
                               future_grid_sample_overhang=self.future_grid_sample_overhang,
                               global_max_q=self.val_global_max_q,
                               oversample_factor=1,
                               firm_industry_lookup=self.firm_industry_lookup,
                               industry_unknown_id=self.industry_unknown_id,
                               tuple_max_abs_zscore=self.tuple_max_abs_zscore),
            batch_size=self.batch_size,
            collate_fn=GraphCollator(
                self.identity_schema,
                self.curriculum_config,
                industry_mode=self.industry_mode,
                seed=self.data_config.get('seed'),
            ),
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=(self.num_workers > 0),
            persistent_workers=False, # persistent_workers=(self.num_workers > 0),
            worker_init_fn=worker_init_fn
        )

    def test_dataloader(self):
        # Deterministic inference mode: mask all q>0 (future) account nodes and do not apply random masking.
        deterministic_forecast = bool(self.curriculum_config.get('deterministic_forecast', False))
        collator = DeterministicForecastCollator(
            self.identity_schema, self.curriculum_config, industry_mode=self.industry_mode,
        ) if deterministic_forecast else GraphCollator(
            self.identity_schema,
            self.curriculum_config,
            industry_mode=self.industry_mode,
            seed=self.data_config.get('seed'),
        )

        return DataLoader(
            FormaWindowDataset(self.test_index, self.test_lookup, self.max_lookback, self.max_horizon,
                               self.reg_stats_by_q, self.test_slice_cache, self.scale_account_id,
                               all_account_ids=self.account_id_map['account_id'].unique().tolist(),
                               use_future_grid=self.use_future_grid,
                               present_in_q0=self.present_in_q0,
                               future_grid_sample_overhang=self.future_grid_sample_overhang,
                               global_max_q=self.test_global_max_q,
                               oversample_factor=1,
                               firm_industry_lookup=self.firm_industry_lookup,
                               industry_unknown_id=self.industry_unknown_id,
                               tuple_max_abs_zscore=self.tuple_max_abs_zscore),
            batch_size=self.batch_size,
            collate_fn=collator,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=(self.num_workers > 0),
            persistent_workers=False, # persistent_workers=(self.num_workers > 0),
            worker_init_fn=worker_init_fn
        )
