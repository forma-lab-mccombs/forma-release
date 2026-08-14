import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
from torch_geometric.utils import to_dense_batch
from forma.data.transforms import de_regularize, get_derivative_s
from forma.data.forma_data import _validate_industry_mode

def _warmup_cosine_factor(epoch: int, warmup_epochs: int, max_epochs: int,
                          min_factor: float) -> float:
    """LR multiplier for the 'warmup_cosine' schedule (epoch-indexed).

    Linear warmup from 1/warmup_epochs up to 1.0 over the first warmup_epochs,
    then cosine decay from 1.0 down to min_factor over the remaining epochs.
    Pure function (no torch / trainer state) so it is directly unit-testable.
    """
    if max_epochs <= 0:
        return 1.0
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    denom = max(1, max_epochs - warmup_epochs)
    progress = min(1.0, float(epoch - warmup_epochs) / float(denom))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_factor + (1.0 - min_factor) * cosine)


# -------------------------------------------------------------------
# 1. Components
# -------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Pre-compute division term for efficiency
        n_pairs = d_model // 2
        if n_pairs > 0:
            i = torch.arange(n_pairs, dtype=torch.float32)
            div_term = torch.exp((-math.log(10000.0) / d_model) * (2.0 * i))
            self.register_buffer('div_term', div_term)
        else:
            self.register_buffer('div_term', torch.empty(0))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding.

        For even d_model, this matches the standard Transformer sinusoidal PE.
        For odd d_model, we intentionally leave the final column as zeros
        (so we still have complete sin/cos pairs for the first 2*floor(d/2) dims).
        """
        device = q.device
        q = q.float().unsqueeze(1)  # [N, 1]

        pe = torch.zeros((q.size(0), self.d_model), device=device, dtype=torch.float32)

        n_pairs = self.d_model // 2
        if n_pairs > 0:
            pe[:, 0:2 * n_pairs:2] = torch.sin(q * self.div_term)
            pe[:, 1:2 * n_pairs:2] = torch.cos(q * self.div_term)

        # If d_model is odd, pe[:, -1] remains 0 by construction.
        return pe

class ConstraintCorrectionLayer(nn.Module):
    """Basis-invariant constraint correction layer (The Accountant).

    Enforces the accounting identities with a direct matrix solve:
        Delta_v = S^{-2} A^T (A S^{-2} A^T + eps*I)^{-1} (-r)
    This correction is invariant under A -> M A for any invertible M.
    """

    def __init__(self, d_model: int, mean_projector: nn.Module, time_embedding: nn.Module):
        super().__init__()
        self.mean_projector = mean_projector
        self.time_embedding = time_embedding
        self.eps = 1e-4

        self.W_m_vec = nn.Parameter(torch.randn(d_model))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(d_model)

    def _compute_v_and_s_inv_sq(self, x, qs, reg_params):
        """Shared provisional-value computation for both forward paths."""
        k, mu, sigma = reg_params
        time_emb = self.time_embedding(qs)
        h_cat = torch.cat([x, time_emb], dim=-1)
        hat_x = self.mean_projector(h_cat).squeeze(-1)
        v = de_regularize(hat_x, k, mu, sigma)
        s = get_derivative_s(v, k, mu, sigma)
        s_inv_sq = torch.pow(s, -2)
        return v, s_inv_sq

    def _apply_gated_update(self, x, delta_v):
        """Shared gated residual update for both forward paths."""
        g_gate = torch.tanh(self.gamma * delta_v).unsqueeze(-1)
        update = g_gate * self.W_m_vec * delta_v.unsqueeze(-1)
        return self.norm(x + update)

    def _forward_sequential(self, x, constraint_src, constraint_dst, constraint_sign,
                            num_constraints, num_constraint_edges, qs, reg_params, batch_idx):
        """Per-graph loop with SVD rank reduction (original implementation)."""
        v, s_inv_sq = self._compute_v_and_s_inv_sq(x, qs, reg_params)
        delta_v = torch.zeros_like(v)

        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0

        edge_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=x.device)
        edge_offsets[1:] = num_constraint_edges.cumsum(0)

        constraint_row_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=x.device)
        constraint_row_offsets[1:] = num_constraints.cumsum(0)

        node_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=x.device)
        counts = torch.bincount(batch_idx, minlength=num_graphs)
        node_offsets[1:] = counts.cumsum(0)

        node_off = node_offsets.tolist()
        edge_off = edge_offsets.tolist()
        crow_off = constraint_row_offsets.tolist()
        nc_list = num_constraints.tolist()

        for g in range(num_graphs):
            n_start = node_off[g]
            n_end = node_off[g + 1]
            e_start = edge_off[g]
            e_end = edge_off[g + 1]
            cr_start = crow_off[g]
            n_c = nc_list[g]
            n_n = n_end - n_start

            if n_c == 0 or n_n == 0:
                continue

            local_dst = constraint_dst[e_start:e_end] - n_start
            local_src = constraint_src[e_start:e_end] - cr_start
            signs = constraint_sign[e_start:e_end]

            A = torch.zeros(n_c, n_n, device=x.device, dtype=x.dtype)
            A[local_src, local_dst] = signs

            if n_c > 1:
                _, S_vals, Vh = torch.linalg.svd(A, full_matrices=False)
                tol = S_vals[0] * max(n_c, n_n) * torch.finfo(A.dtype).eps
                rank = int((S_vals > tol).sum().item())
                if rank < n_c:
                    A = Vh[:rank]
                    n_c = rank

            v_g = v[n_start:n_end]
            s_inv_sq_g = s_inv_sq[n_start:n_end]

            r = A @ v_g
            Q = (A * s_inv_sq_g.unsqueeze(0)) @ A.t()
            Q = Q + self.eps * torch.eye(Q.size(0), device=x.device, dtype=x.dtype)
            # Layered fallback (see _forward_batched): solve -> lstsq -> skip this
            # graph (zero correction; delta_v stays 0 here) rather than letting a
            # bare-pinv SVD failure crash the forward. The LinAlgError except is
            # mainly the CPU / catastrophic-driver guard: on CUDA the default
            # `gels` lstsq driver assumes full rank and returns non-finite values
            # for a rank-deficient Q WITHOUT raising, so the nan_to_num scrub below
            # is the real GPU protection (maps NaN/+-inf -> 0 == zero correction).
            try:
                lam = torch.linalg.solve(Q, -r)
            except torch.linalg.LinAlgError:
                try:
                    lam = torch.linalg.lstsq(Q, -r.unsqueeze(-1)).solution.squeeze(-1)
                except torch.linalg.LinAlgError as e:
                    print(f"  [ConstraintCorrection] ERROR: per-graph solve+lstsq "
                          f"both failed ({type(e).__name__}: {str(e)[:120]}); "
                          f"applying zero correction for graph {g}")
                    continue
            # Scrub a silently non-finite lstsq solution to 0 (zero correction for
            # this graph) — parity with _forward_batched. +-inf -> 0, not dtype-max.
            lam = torch.nan_to_num(lam, nan=0.0, posinf=0.0, neginf=0.0)
            delta_v[n_start:n_end] = s_inv_sq_g * (A.t() @ lam)

        return self._apply_gated_update(x, delta_v)

    def _forward_batched(self, x, A_batched, batch_idx, qs, reg_params,
                         nc_per_graph, nn_per_graph):
        """Batched forward using pre-built padded constraint matrices.

        Replaces the per-graph loop with batched bmm + solve for GPU parallelism.
        Uses eps regularization instead of SVD rank reduction.
        """
        v, s_inv_sq = self._compute_v_and_s_inv_sq(x, qs, reg_params)

        # Pad to dense batched format [B, max_nn]
        v_dense, valid_mask = to_dense_batch(v, batch_idx)
        s_inv_sq_dense, _ = to_dense_batch(s_inv_sq, batch_idx)

        B, max_nn = v_dense.shape
        max_nc = A_batched.shape[1]

        # r = A @ v  -> [B, max_nc]
        r = torch.bmm(A_batched, v_dense.unsqueeze(-1)).squeeze(-1)

        # Q = A * diag(s_inv_sq) @ A^T + eps*I  -> [B, max_nc, max_nc]
        A_scaled = A_batched * s_inv_sq_dense.unsqueeze(1)
        Q = torch.bmm(A_scaled, A_batched.transpose(1, 2))

        # Build identity and constraint validity mask
        eye = torch.eye(max_nc, device=x.device, dtype=x.dtype).unsqueeze(0)
        constraint_valid = torch.arange(max_nc, device=x.device).unsqueeze(0) < nc_per_graph.unsqueeze(1)
        row_valid = constraint_valid.unsqueeze(2)
        col_valid = constraint_valid.unsqueeze(1)
        valid_Q = row_valid & col_valid

        # Zero out padded rows/cols in Q, set padded diagonal to 1
        Q = Q * valid_Q.float() + (~valid_Q).float() * eye
        Q = Q + self.eps * eye

        # Zero out r for padded constraints
        r = r * constraint_valid.float()

        # Batched solve: Q @ lam = -r  -> [B, max_nc]
        # Layered fallback mirroring reconcile_batch's Fix A (PR #99/#100): the
        # bare `pinv` here was the parallel gap that crashed r8e3a's predict twice
        # — `pinv` runs an SVD internally, so a "SVD did not converge" raises right
        # back out of the except and kills the whole forward. We now degrade
        # gracefully: solve -> lstsq -> zero correction for this batch (+ ERROR
        # log), so one ill-conditioned batch is skipped rather than fatal.
        # The LinAlgError except is mainly the CPU / catastrophic-driver guard: on
        # CUDA the default `gels` lstsq driver assumes full rank and returns
        # non-finite values for a rank-deficient Q WITHOUT raising, so the
        # nan_to_num scrub below is the real GPU protection.
        rhs = -r.unsqueeze(-1)
        try:
            lam = torch.linalg.solve(Q, rhs).squeeze(-1)
        except torch.linalg.LinAlgError:
            try:
                lam = torch.linalg.lstsq(Q, rhs).solution.squeeze(-1)
            except torch.linalg.LinAlgError as e:
                # Both batched solve and the lstsq SVD fallback failed — Q too
                # ill-conditioned even for lstsq. Apply zero correction for this
                # batch (delta_v = 0 -> the layer returns norm(x), the no-violation
                # result) and keep going.
                print(f"  [ConstraintCorrection] ERROR: batched solve+lstsq both "
                      f"failed ({type(e).__name__}: {str(e)[:120]}); applying zero "
                      f"correction for batch of {B} graphs")
                lam = torch.zeros_like(rhs).squeeze(-1)
        # Scrub a silently non-finite lstsq solution (CUDA gels on rank-deficient Q)
        # to 0 == zero correction. Map +-inf to 0 too, NOT to dtype-max as the
        # nan-only default would, so a bad solve degrades to "no correction".
        lam = torch.nan_to_num(lam, nan=0.0, posinf=0.0, neginf=0.0)

        # delta_v = s_inv_sq * (A^T @ lam)  -> [B, max_nn]
        At_lam = torch.bmm(A_batched.transpose(1, 2), lam.unsqueeze(-1)).squeeze(-1)
        delta_v_dense = s_inv_sq_dense * At_lam * valid_mask.float()

        # Scatter back to flat format
        delta_v = delta_v_dense[valid_mask]

        return self._apply_gated_update(x, delta_v)

    def forward(self, x, constraint_src, constraint_dst, constraint_sign,
                num_constraints, num_constraint_edges, qs, reg_params, batch_idx,
                A_batched=None, nc_per_graph=None, nn_per_graph=None):
        """Dispatch to batched or sequential path."""
        if A_batched is not None:
            return self._forward_batched(x, A_batched, batch_idx, qs, reg_params,
                                         nc_per_graph, nn_per_graph)
        return self._forward_sequential(x, constraint_src, constraint_dst, constraint_sign,
                                        num_constraints, num_constraint_edges, qs, reg_params,
                                        batch_idx)


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, 
                 use_temporal_bias: bool = True, use_same_account_bias: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.use_temporal_bias = use_temporal_bias
        self.use_same_account_bias = use_same_account_bias

        if self.use_temporal_bias:
            self.lambda_bias = nn.Parameter(torch.tensor(1.0))
        
        if self.use_same_account_bias:
            self.same_account_bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor, qs: torch.Tensor,
                acc_ids: torch.Tensor = None, is_industry: torch.Tensor = None):
        # All nodes are account nodes (identity nodes removed from graph)

        # Dense batch
        # Now B samples, L max account nodes per sample
        x_dense, valid_pos_mask = to_dense_batch(x, batch_idx) # [B, L, D], [B, L]
        qs_dense, _ = to_dense_batch(qs, batch_idx) # [B, L]

        # Key padding mask (True where padded)
        key_padding_mask = ~valid_pos_mask  # [B, L]

        # Pre-norm
        x_norm = self.norm1(x_dense)

        # Fast path: when no additive biases are requested, pass bool key_padding_mask
        # directly so PyTorch SDPA can dispatch to a fused backend (typically
        # memory-efficient; FlashAttention proper requires no mask at all). Any float
        # attn_mask — even an all-zeros one — forces the slow math fallback.
        if not self.use_temporal_bias and not self.use_same_account_bias:
            attn_output, _ = self.self_attn(x_norm, x_norm, x_norm,
                                            attn_mask=None,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=False)
        else:
            # Temporal bias
            # q_diff: [B, L, L]
            if self.use_temporal_bias:
                q_diff = torch.abs(qs_dense.unsqueeze(2) - qs_dense.unsqueeze(1))
                attn_bias = -self.lambda_bias * torch.log1p(q_diff.float())
            else:
                attn_bias = torch.zeros((x_dense.size(0), x_dense.size(1), x_dense.size(1)), device=x.device)

            # Same account bonus
            if self.use_same_account_bias and acc_ids is not None:
                ids_dense, _ = to_dense_batch(acc_ids, batch_idx)  # [B, L]
                is_same_acc = (ids_dense.unsqueeze(2) == ids_dense.unsqueeze(1))
                # Industry-mode 'node' appends an extra node with x_id=0; without this guard
                # it would share the same-account bias with whichever account has id 0.
                if is_industry is not None:
                    is_ind_dense, _ = to_dense_batch(is_industry, batch_idx)  # [B, L] bool
                    exclude = is_ind_dense.unsqueeze(2) | is_ind_dense.unsqueeze(1)
                    is_same_acc = is_same_acc & ~exclude
                attn_bias = attn_bias + (is_same_acc.float() * self.same_account_bias)

            # Repeat for heads: [B*H, L, L]
            n_heads = self.self_attn.num_heads
            attn_bias = attn_bias.repeat_interleave(n_heads, dim=0)

            # Merge key_padding_mask into attn_bias to avoid PyTorch warning about mismatched types
            kp_mask_flat = key_padding_mask.repeat_interleave(n_heads, dim=0)
            kp_mask_expanded = kp_mask_flat.unsqueeze(1)
            attn_bias = attn_bias.masked_fill(kp_mask_expanded, float('-inf'))

            attn_output, _ = self.self_attn(x_norm, x_norm, x_norm,
                                            attn_mask=attn_bias,
                                            key_padding_mask=None,
                                            need_weights=False)

        x_dense = x_dense + self.dropout1(attn_output)

        # FFN
        x_norm2 = self.norm2(x_dense)
        ff_output = self.linear2(self.dropout(F.gelu(self.linear1(x_norm2))))
        x_dense = x_dense + self.dropout2(ff_output)

        # Scatter back
        x_out = x_dense[valid_pos_mask]

        return x_out

class FormaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, mean_projector, time_embedding,
                 constraint_mode='correction',
                 use_temporal_bias=True, use_same_account_bias=False):
        super().__init__()
        self.constraint_mode = constraint_mode
        if constraint_mode == 'correction':
            self.constraint = ConstraintCorrectionLayer(d_model, mean_projector, time_embedding)
        elif constraint_mode == 'none':
            self.constraint = None
        else:
            raise ValueError(f"Unknown constraint_mode: {constraint_mode}")
        self.transformer = TransformerLayer(d_model, n_heads, d_ff, dropout,
                                            use_temporal_bias=use_temporal_bias,
                                            use_same_account_bias=use_same_account_bias)

    def forward(self, x, constraint_src, constraint_dst, constraint_sign, num_constraints,
                num_constraint_edges, batch_idx, qs, reg_params, acc_ids=None,
                is_industry=None,
                A_batched=None, nc_per_graph=None, nn_per_graph=None):
        if self.constraint_mode == 'correction':
            x = self.constraint(x, constraint_src, constraint_dst, constraint_sign,
                                num_constraints, num_constraint_edges, qs, reg_params, batch_idx,
                                A_batched=A_batched, nc_per_graph=nc_per_graph,
                                nn_per_graph=nn_per_graph)
        # 'none': skip constraint enforcement entirely

        x = self.transformer(x, batch_idx, qs, acc_ids=acc_ids, is_industry=is_industry)
        return x

# -------------------------------------------------------------------
# 3. Main Model
# -------------------------------------------------------------------

class FormaModel(L.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(self,
                 num_account_types: int,
                 d_model: int = 128,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 constraint_mode: str = 'correction',
                 lr: float = 1e-3,
                 weight_decay: float = 1e-2,
                 log_sigma_sq_min: float = -10.0,
                 log_sigma_sq_max: float = 10.0,
                 loss_type: str = 'gaussian',
                 student_t_df: float = 5.0,
                 lr_schedule: str = 'constant',
                 warmup_epochs: int = 0,
                 lr_min_factor: float = 0.0,
                 beta_nll: float = 0.0,
                 use_temporal_bias: bool = True,
                 use_same_account_bias: bool = False,
                 num_industries: int = 49,
                 industry_mode: str = 'none'):
        super().__init__()
        self.save_hyperparameters()

        if loss_type not in ('gaussian', 'student_t', 'laplace', 'mse'):
            raise ValueError(
                f"Unknown loss_type '{loss_type}'. Expected 'gaussian', 'student_t', "
                f"'laplace', or 'mse'.")
        if lr_schedule not in ('constant', 'warmup_cosine'):
            raise ValueError(
                f"Unknown lr_schedule '{lr_schedule}'. Expected 'constant' "
                f"(default) or 'warmup_cosine'.")
        if beta_nll < 0.0:
            raise ValueError(f"beta_nll must be >= 0 (got {beta_nll}); 0.0 is the "
                             f"standard NLL (default; weighting applies to the "
                             f"gaussian and laplace losses).")
        if beta_nll > 0.0 and loss_type not in ('gaussian', 'laplace'):
            print(f"  WARNING: beta_nll={beta_nll} is ignored for "
                  f"loss_type='{loss_type}' (only gaussian/laplace honor it).")

        # Pre-compute Student-t constants (only used if loss_type='student_t')
        # These are registered as buffers so they move to the correct device automatically
        nu = student_t_df
        # log_gamma_term = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
        # log_pi_term = -0.5 * math.log(nu * math.pi)
        # self.register_buffer('_student_t_log_gamma_term', torch.tensor(log_gamma_term))
        # self.register_buffer('_student_t_log_pi_term', torch.tensor(log_pi_term))
        self.register_buffer('_student_t_half_nu_plus_1', torch.tensor((nu + 1) / 2))

        _validate_industry_mode(industry_mode)

        self.account_embedding = nn.Embedding(num_account_types, d_model)
        if industry_mode != 'none':
            self.industry_embedding = nn.Embedding(num_industries, d_model)
        self.time_embedding = TimeEmbedding(d_model)

        # Value embedding parameters
        self.w_x = nn.Parameter(torch.randn(d_model))
        self.x_masked = nn.Parameter(torch.randn(d_model))

        # Projectors
        # MeanProjectorMLP (shared as MeanProjector)
        # Input: Concat(h, E_time(q)) -> d_model * 2
        self.mean_projector = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 1)
        )

        # Variance head (the predictive scale). Ablated for the point (MSE) objective:
        # under loss_type='mse' we train the mean head only, so the variance head is
        # not built at all (fewer params) and forward() returns log_sigma_sq=None.
        if loss_type == 'mse':
            self.var_projector = None
        else:
            self.var_projector = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, 1)
            )

        self.blocks = nn.ModuleList([
            FormaBlock(d_model, n_heads, d_ff, dropout, self.mean_projector, self.time_embedding,
                       constraint_mode=constraint_mode,
                       use_temporal_bias=use_temporal_bias,
                       use_same_account_bias=use_same_account_bias)
            for _ in range(n_layers)
        ])

    def _unpack_batch(self, batch):
        """Extract and normalize fields from a PyG Batch."""
        # PyG names the batch vector after the first node-store key;
        # FormaData uses x_id instead of x, so it becomes x_id_batch.
        batch_idx = getattr(batch, 'x_id_batch', None)
        if batch_idx is None:
            batch_idx = getattr(batch, 'batch', None)
        if batch_idx is None:
            batch_idx = torch.zeros(batch.x_id.size(0), dtype=torch.long, device=batch.x_id.device)
        # Avoid evaluating zeros() on every forward when the attribute is present.
        if hasattr(batch, 'x_industry_id'):
            x_industry_id = batch.x_industry_id
            is_industry = batch.is_industry
        else:
            x_industry_id = torch.zeros_like(batch.x_id)
            is_industry = torch.zeros(batch.x_id.size(0), dtype=torch.bool, device=batch.x_id.device)

        return (batch.x_id, batch.x_time, batch.x_val, batch.mask, batch_idx,
                (batch.reg_k, batch.reg_mu, batch.reg_sigma),
                batch.constraint_src, batch.constraint_dst, batch.constraint_sign,
                batch.num_constraints, batch.num_constraint_edges,
                x_industry_id, is_industry)

    def _build_initial_embeddings(self, x_id, x_time, x_val, mask, x_industry_id, is_industry):
        """Compute initial node embeddings.

        industry_mode:
          - 'none': h = account_emb + time_emb + val_emb
          - 'node': extra industry node per sample; industry nodes use industry_emb
                    instead of account_emb (selected via `is_industry`).
          - 'bias': industry_emb added to every node (all nodes share sample's industry).
        """
        h_acc = self.account_embedding(x_id)

        mode = self.hparams.industry_mode
        if mode == 'node':
            h_ind = self.industry_embedding(x_industry_id)
            h_type = torch.where(is_industry.unsqueeze(-1), h_ind, h_acc)
        elif mode == 'bias':
            h_type = h_acc + self.industry_embedding(x_industry_id)
        else:  # 'none'
            h_type = h_acc

        time_emb = self.time_embedding(x_time)
        val_emb = torch.where(
            mask.unsqueeze(-1),
            self.x_masked.unsqueeze(0),
            self.w_x.unsqueeze(0) * x_val.unsqueeze(-1)
        )
        return h_type + time_emb + val_emb

    def _build_batched_A(self, constraint_src, constraint_dst, constraint_sign,
                         num_constraints, num_constraint_edges, batch_idx):
        """Build padded batched constraint matrices from COO tensors.

        Returns:
            A_batched: [B, max_nc, max_nn] padded constraint matrices
            nc_per_graph: [B] actual constraint count per graph
            nn_per_graph: [B] actual node count per graph
        """
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
        if num_graphs == 0:
            return None, None, None

        nn_per_graph = torch.bincount(batch_idx, minlength=num_graphs)
        node_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=batch_idx.device)
        node_offsets[1:] = nn_per_graph.cumsum(0)

        edge_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=batch_idx.device)
        edge_offsets[1:] = num_constraint_edges.cumsum(0)

        crow_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=batch_idx.device)
        crow_offsets[1:] = num_constraints.cumsum(0)

        max_nc = int(num_constraints.max().item())
        max_nn = int(nn_per_graph.max().item())

        A_batched = torch.zeros(num_graphs, max_nc, max_nn,
                                device=batch_idx.device, dtype=constraint_sign.dtype)

        # Assign each edge to its graph using bucketize on edge offsets
        total_edges = constraint_src.size(0)
        if total_edges > 0:
            edge_indices = torch.arange(total_edges, device=batch_idx.device)
            edge_graph = torch.bucketize(edge_indices, edge_offsets[1:], right=True)

            local_rows = constraint_src - crow_offsets[edge_graph]
            local_cols = constraint_dst - node_offsets[edge_graph]

            A_batched[edge_graph, local_rows, local_cols] = constraint_sign

        return A_batched, num_constraints, nn_per_graph

    def forward(self, batch):
        (x_id, x_time, x_val, mask, batch_idx, reg_params,
         constraint_src, constraint_dst, constraint_sign,
         num_constraints, num_constraint_edges,
         x_industry_id, is_industry) = self._unpack_batch(batch)

        h = self._build_initial_embeddings(x_id, x_time, x_val, mask, x_industry_id, is_industry)

        # Build batched A once for all layers (constraint structure is fixed)
        A_batched, nc_per_graph, nn_per_graph = None, None, None
        if self.hparams.constraint_mode == 'correction':
            A_batched, nc_per_graph, nn_per_graph = self._build_batched_A(
                constraint_src, constraint_dst, constraint_sign,
                num_constraints, num_constraint_edges, batch_idx)

        for block in self.blocks:
            h = block(h, constraint_src, constraint_dst, constraint_sign,
                      num_constraints, num_constraint_edges, batch_idx, x_time,
                      reg_params, acc_ids=x_id, is_industry=is_industry,
                      A_batched=A_batched, nc_per_graph=nc_per_graph,
                      nn_per_graph=nn_per_graph)

        time_emb_final = self.time_embedding(x_time)
        h_final = torch.cat([h, time_emb_final], dim=-1)

        mu = self.mean_projector(h_final).squeeze(-1)
        # Variance head is ablated under loss_type='mse' -> no predictive scale.
        log_sigma_sq = (self.var_projector(h_final).squeeze(-1)
                        if self.var_projector is not None else None)

        return mu, log_sigma_sq

    def extract_embeddings(self, batch, layer_idx=-1):
        """Extract hidden-state embeddings from a specific FormaBlock layer.

        Reuses the same initial embedding construction as ``forward()`` but
        returns hidden states instead of predictions.

        Args:
            batch: PyG Batch produced by EmbeddingCollator (no masking).
            layer_idx: Which FormaBlock to read from.  ``-1`` (default) means
                the last block, ``0`` means the first, etc.

        Returns:
            (embeddings, metadata) where
            - embeddings: Tensor [N_account, d_model] hidden states for account nodes only
            - metadata: dict with keys ``batch_idx``, ``x_id``, ``x_time``,
              ``firm_id``, ``center_q`` -- each a 1-D tensor aligned with *embeddings*
        """
        (x_id, x_time, x_val, mask, batch_idx, reg_params,
         constraint_src, constraint_dst, constraint_sign,
         num_constraints, num_constraint_edges,
         x_industry_id, is_industry) = self._unpack_batch(batch)

        h = self._build_initial_embeddings(x_id, x_time, x_val, mask, x_industry_id, is_industry)

        A_batched, nc_per_graph, nn_per_graph = None, None, None
        if self.hparams.constraint_mode == 'correction':
            A_batched, nc_per_graph, nn_per_graph = self._build_batched_A(
                constraint_src, constraint_dst, constraint_sign,
                num_constraints, num_constraint_edges, batch_idx)

        n_blocks = len(self.blocks)
        target = layer_idx if layer_idx >= 0 else n_blocks + layer_idx
        if target < 0 or target >= n_blocks:
            raise IndexError(
                f"layer_idx={layer_idx} is out of range for a model with {n_blocks} blocks "
                f"(valid range: {-n_blocks} to {n_blocks - 1})"
            )
        for i, block in enumerate(self.blocks):
            h = block(h, constraint_src, constraint_dst, constraint_sign,
                      num_constraints, num_constraint_edges, batch_idx, x_time,
                      reg_params, acc_ids=x_id, is_industry=is_industry,
                      A_batched=A_batched, nc_per_graph=nc_per_graph,
                      nn_per_graph=nn_per_graph)
            if i == target:
                break

        metadata = {
            'batch_idx': batch_idx,
            'x_id': x_id,
            'x_time': x_time,
            'firm_id': batch.firm_id[batch_idx],
            'center_q': batch.center_q[batch_idx],
            'is_industry': is_industry,
        }
        return h, metadata

    def _masked_nll_reg_space(self, *, batch, mu: torch.Tensor, log_sigma_sq: torch.Tensor) -> torch.Tensor:
        """Training loss computed only on masked *account* nodes.

        Dispatches on hparams.loss_type:
        - 'gaussian' / 'student_t' / 'laplace': heteroscedastic NLL using the
          variance head. 'gaussian' and 'laplace' additionally honor
          hparams.beta_nll (0.0 = off = standard NLL; 1.0 = equal-weighted
          location loss — MSE-like for gaussian, plain-MAE for laplace).
          'laplace' reads the head as a predictive std and uses Laplace scale
          b = sigma / sqrt(2). 'student_t' ignores beta_nll.
        - 'mse': point objective on the mean head only (variance head ablated, so
          log_sigma_sq is None and is ignored).

        Expects:
        - mu (and, for the probabilistic objectives, log_sigma_sq) are per-node
          predictions for ALL nodes in the PyG Batch
        - batch.y is account-only targets in the same (scaled+regularized) space as batch.x_val

        Returns a scalar loss. If there are no masked account nodes, returns a 0.0 tensor
        with grad enabled in training (caller can handle early return if desired).
        """
        mask_acc = batch.mask

        if hasattr(batch, 'data_avail'):
            mask_acc = mask_acc & batch.data_avail

        if not mask_acc.any():
            return torch.tensor(0.0, device=mu.device, requires_grad=True)

        if self.hparams.loss_type == 'mse':
            # Point MSE on the mean head; the variance head is ablated (log_sigma_sq
            # is None). Mirrors the Gaussian NLL with sigma fixed, dropping the scale.
            mu_pred = mu[mask_acc]
            y_target = batch.y[mask_acc]
            return torch.pow(y_target - mu_pred, 2).mean()

        mu_pred = mu[mask_acc]
        log_sigma_sq_pred = log_sigma_sq[mask_acc]

        # Clamp log-variance for numerical stability before exp()
        log_sigma_sq_pred = log_sigma_sq_pred.clamp(
            min=self.hparams.log_sigma_sq_min,
            max=self.hparams.log_sigma_sq_max,
        )
        sigma_sq_pred = torch.exp(log_sigma_sq_pred)

        # Targets are account-only and unmodified by masking
        y_target = batch.y[mask_acc]

        if self.hparams.loss_type == 'gaussian':
            # Gaussian NLL: 0.5 * (y - mu)^2 / sigma^2 + 0.5 * log(sigma^2)
            loss = 0.5 * torch.pow(y_target - mu_pred, 2) / sigma_sq_pred + 0.5 * log_sigma_sq_pred
            # Optional beta-NLL (Seitzer et al., ICLR 2022): weight each term by a
            # stop-gradient sigma^(2*beta) to weaken the 1/sigma^2 coupling that lets
            # sigma absorb mu's failures. beta_nll=0.0 (the default) leaves the
            # weight at 1, so the canonical Gaussian path stays bit-identical.
            if getattr(self.hparams, 'beta_nll', 0.0) > 0.0:
                loss = loss * sigma_sq_pred.detach().pow(self.hparams.beta_nll)
        elif self.hparams.loss_type == 'laplace':
            # Laplace NLL: the MAE-natural likelihood (mu -> conditional median);
            # linear-in-|z| penalty, robust to sigma-scale wobble. Parameterized
            # via the SAME head so the saved `sigma` column stays the predictive
            # std: sigma_pred = exp(0.5*log_sigma_sq) is the predictive std, and the
            # Laplace scale is b = sigma_pred / sqrt(2) (=> predictive std =
            # sqrt(2)*b). NLL = log(2b) + |y-mu|/b. Scored with family='laplace'
            # in evaluate.py via the .nll.json sidecar train.py writes.
            sigma_pred = torch.sqrt(sigma_sq_pred)
            b = sigma_pred / math.sqrt(2.0)
            loss = torch.log(2.0 * b) + torch.abs(y_target - mu_pred) / b
            # beta-NLL, Laplace analogue: weight each term by a stop-gradient
            # b^beta. The location term's per-cell weight goes b^(beta-1), so
            # beta interpolates from the fully self-weighted NLL (beta=0,
            # noisy cells down-weighted by 1/b) to the EQUAL-WEIGHTED
            # absolute error |y-mu| at beta=1 (+ b*log(2b) keeping the scale
            # head trained) — the pooled-MAE-natural objective. Mirrors the
            # Gaussian branch, where sigma^(2*beta) makes beta=1 MSE-like.
            # The DETACH is load-bearing: because the weight is constant to
            # autograd, the per-cell scale optimum stays b* = |y-mu| at every
            # beta (naively differentiating the weighted VALUE b*log(2b) would
            # suggest a degenerate error-independent b* = 1/(2e) — that is
            # exactly what happens if the detach is dropped; guarded by
            # test_beta_nll_weight_is_stop_gradient).
            # beta_nll=0.0 leaves the r11e4 pure-Laplace path bit-identical.
            if getattr(self.hparams, 'beta_nll', 0.0) > 0.0:
                loss = loss * b.detach().pow(self.hparams.beta_nll)
        elif self.hparams.loss_type == 'student_t':
            # Student-t NLL using cached constants
            # NLL = 0.5*log(sigma^2) + ((nu+1)/2) * log(1 + (y-mu)^2 / (nu*sigma^2))
            nu = self.hparams.student_t_df
            residual_sq = torch.pow(y_target - mu_pred, 2)
            log_scale_term = 0.5 * log_sigma_sq_pred  # 0.5 * log(sigma^2)
            log_density_term = self._student_t_half_nu_plus_1 * torch.log1p(residual_sq / (nu * sigma_sq_pred))

            # Combine: -log p(y|mu,sigma,nu)
            loss = log_scale_term + log_density_term
        else:
            # 'mse' returns above; __init__ rejects anything else — unreachable.
            raise ValueError(
                f"Unknown loss_type: {self.hparams.loss_type}. Must be 'gaussian', "
                f"'student_t', 'laplace', or 'mse'.")

        return loss.mean()

    def training_step(self, batch, batch_idx):
        mu, log_sigma_sq = self(batch)
        loss = self._masked_nll_reg_space(batch=batch, mu=mu, log_sigma_sq=log_sigma_sq)
        self.log('train_loss', loss, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mu, log_sigma_sq = self(batch)

        if not batch.mask.any():
            return

        loss = self._masked_nll_reg_space(batch=batch, mu=mu, log_sigma_sq=log_sigma_sq)
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        # Default ('constant'): the canonical constant-LR AdamW — return the bare
        # optimizer exactly as before, so the r10 training path is bit-identical.
        if getattr(self.hparams, 'lr_schedule', 'constant') == 'constant':
            return optimizer

        # 'warmup_cosine': linear warmup over warmup_epochs, then cosine decay to
        # lr_min_factor * lr over the remaining epochs. Stepped per EPOCH (the
        # curriculum changes steps-per-epoch, so a step-based schedule would be
        # ill-defined). Total epochs come from the trainer.
        max_epochs = int(getattr(self.trainer, 'max_epochs', 0) or 0)
        warmup = int(self.hparams.warmup_epochs)
        min_factor = float(self.hparams.lr_min_factor)

        def lr_lambda(epoch: int) -> float:
            return _warmup_cosine_factor(epoch, warmup, max_epochs, min_factor)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1},
        }
