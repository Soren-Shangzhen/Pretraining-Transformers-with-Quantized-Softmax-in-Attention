"""Adaptive K-interval attention for training.

Forward definitions are unchanged from the previous research code:

For every causal-valid query row,
    S = QK^T / sqrt(d_h)
    s_min = min(valid S_j)
    s_max = max(valid S_j)
    h = (s_max - s_min) / K

The implementation centers each row at s_max, so z=S-s_max <= 0.

nearest:
    choose the closest interval boundary in forward;
    use weight-space LERP slope as the surrogate backward.

lerp:
    linearly interpolate exp(boundary) values in weight space.

This revision adds diagnostics only; it does NOT change the attention operator:
- score-row min/max/span statistics for exact and quant modes;
- Q/K vector-norm statistics;
- interval occupancy for quant modes;
- actual K+1 nearest reconstruction-level occupancy for nearest mode.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _adaptive_interval_reconstruct(att: torch.Tensor,
                                   causal: torch.Tensor,
                                   K: int,
                                   approx: str,
                                   eps: float = 1e-12,
                                   return_level_idx: bool = False):
    """Reconstruct unnormalized attention weights on a per-row adaptive grid.

    Args:
        att:    (B,H,T,T) fp32 scaled QK scores, without causal fill values.
        causal: (T,T) boolean lower-triangular validity mask.
        K:      number of equal intervals spanning each row's valid min..max.
        approx: 'nearest' or 'lerp'.
        return_level_idx: when True, also return the actual nearest-boundary
            reconstruction level index in [0,K].  For LERP this is None.

    Returns (default):
        w, bin_idx, row_span

    Returns (return_level_idx=True):
        w, bin_idx, row_span, level_idx
    """
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    if approx not in {"nearest", "lerp"}:
        raise ValueError(f"unknown approximation: {approx}")

    B, H, T, _ = att.shape
    valid = causal.view(1, 1, T, T)

    # Calibration statistics are detached exactly as in the previous version.
    row_max = att.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True).detach()
    row_min = att.masked_fill(~valid, torch.inf).amin(dim=-1, keepdim=True).detach()
    row_span = row_max - row_min

    z = att - row_max
    z_min = row_min - row_max
    h = row_span / float(K)
    degenerate = row_span <= eps
    h_safe = torch.where(degenerate, torch.ones_like(h), h)

    rel = (z - z_min) / h_safe
    bin_idx = torch.floor(rel).to(torch.long).clamp_(0, K - 1)
    bin_idx = torch.where(valid, bin_idx, torch.full_like(bin_idx, -1))

    r_idx = bin_idx.clamp_min(0)
    r = r_idx.to(att.dtype)
    b_lo = z_min + r * h_safe

    # K+1 exponentials per row.
    grid_r = torch.arange(K + 1, device=att.device, dtype=att.dtype)
    grid_r = grid_r.view(1, 1, 1, K + 1)
    boundary_z = z_min + grid_r * h_safe
    boundary_w = torch.exp(boundary_z)
    e_lo = torch.gather(boundary_w, -1, r_idx)
    e_hi = torch.gather(boundary_w, -1, (r_idx + 1).clamp_max(K))

    t = ((z - b_lo) / h_safe).clamp(0.0, 1.0)
    w_lerp = (1.0 - t) * e_lo + t * e_hi

    level_idx = None
    if approx == "lerp":
        w = w_lerp
    else:
        choose_upper = t >= 0.5
        w_nearest = torch.where(choose_upper, e_hi, e_lo)
        # Hard nearest forward, LERP-surrogate backward.  Unchanged semantics.
        w = w_lerp + (w_nearest - w_lerp).detach()

        if return_level_idx:
            level_idx = r_idx + choose_upper.to(torch.long)
            level_idx = level_idx.clamp_(0, K)
            # A degenerate row has identical boundaries; assign level 0 only
            # for bookkeeping because all K+1 levels are numerically identical.
            level_idx = torch.where(degenerate.expand_as(level_idx),
                                    torch.zeros_like(level_idx), level_idx)
            level_idx = torch.where(valid, level_idx,
                                    torch.full_like(level_idx, -1))

    w = torch.where(degenerate, torch.ones_like(w), w)
    w = w.masked_fill(~valid, 0.0)

    if return_level_idx:
        return w, bin_idx, row_span, level_idx
    return w, bin_idx, row_span


class QuantCausalSelfAttention(nn.Module):
    """Causal self-attention with matched controls and diagnostics.

    Modes:
        softmax:            PyTorch SDPA baseline.
        exact_fp32_softmax: explicit FP32 QK^T -> exact softmax -> PV control.
        quant:              adaptive K-interval Nearest/LERP attention.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.mode = cfg.attn_mode
        self.K = cfg.attn_K
        self.approx = cfg.attn_approx

        self.collect_stats = False
        self.reset_last_stats()

    def reset_last_stats(self):
        # Common raw accumulators for one forward pass.  Raw sums/counts let
        # train.py pool statistics exactly across all layers.
        self.last_row_span_sum = None
        self.last_row_span_sq_sum = None
        self.last_row_min_sum = None
        self.last_row_max_sum = None
        self.last_row_count = None
        self.last_q_norm_sum = None
        self.last_q_norm_sq_sum = None
        self.last_k_norm_sum = None
        self.last_k_norm_sq_sum = None
        self.last_qk_count = None

        # Quantizer-specific counts.
        self.last_interval_counts = None
        self.last_level_counts = None

        # Backward-compatible convenience fields.
        self.last_occupancy = None
        self.last_level_occupancy = None
        self.last_span_mean = None
        self.last_span_std = None

    @torch.no_grad()
    def _record_common_stats(self, qf, kf, att, causal):
        B, H, T, _ = att.shape
        valid = causal.view(1, 1, T, T)
        row_max = att.masked_fill(~valid, -torch.inf).amax(dim=-1)
        row_min = att.masked_fill(~valid, torch.inf).amin(dim=-1)
        spans = row_max - row_min

        qn = torch.linalg.vector_norm(qf, dim=-1)
        kn = torch.linalg.vector_norm(kf, dim=-1)

        self.last_row_span_sum = spans.sum().double().cpu()
        self.last_row_span_sq_sum = spans.square().sum().double().cpu()
        self.last_row_min_sum = row_min.sum().double().cpu()
        self.last_row_max_sum = row_max.sum().double().cpu()
        self.last_row_count = int(spans.numel())

        self.last_q_norm_sum = qn.sum().double().cpu()
        self.last_q_norm_sq_sum = qn.square().sum().double().cpu()
        self.last_k_norm_sum = kn.sum().double().cpu()
        self.last_k_norm_sq_sum = kn.square().sum().double().cpu()
        self.last_qk_count = int(qn.numel())

        self.last_span_mean = spans.mean().cpu()
        self.last_span_std = spans.std(unbiased=False).cpu()

    def _exact_fp32_softmax(self, q, k, v):
        B, H, T, D = q.shape
        with torch.autocast(device_type=q.device.type, enabled=False):
            qf, kf, vf = q.float(), k.float(), v.float()
            att = (qf @ kf.transpose(-2, -1)) / math.sqrt(D)
            causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            valid = causal.view(1, 1, T, T)

            masked_att = att.masked_fill(~valid, -torch.inf)
            row_max = masked_att.amax(dim=-1, keepdim=True)
            z = masked_att - row_max
            w = torch.exp(z)
            P = w / w.sum(dim=-1, keepdim=True)
            y = P @ vf

            if self.collect_stats:
                self._record_common_stats(qf, kf, att, causal)

        return y.to(q.dtype)

    def _quant_attention(self, q, k, v):
        B, H, T, D = q.shape
        with torch.autocast(device_type=q.device.type, enabled=False):
            qf, kf, vf = q.float(), k.float(), v.float()
            att = (qf @ kf.transpose(-2, -1)) / math.sqrt(D)
            causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()

            w, bins, row_span, level_idx = _adaptive_interval_reconstruct(
                att, causal, self.K, self.approx, return_level_idx=True
            )
            denom = w.sum(dim=-1, keepdim=True)
            P = w / denom
            y = P @ vf

            if self.collect_stats:
                with torch.no_grad():
                    self._record_common_stats(qf, kf, att, causal)
                    valid = causal.view(1, 1, T, T).expand(B, H, T, T)

                    valid_bins = bins[valid]
                    counts = torch.bincount(valid_bins,
                                            minlength=self.K).double().cpu()
                    self.last_interval_counts = counts
                    self.last_occupancy = (counts / counts.sum()).float()

                    if level_idx is not None:
                        valid_levels = level_idx[valid]
                        lcounts = torch.bincount(valid_levels,
                                                 minlength=self.K + 1).double().cpu()
                        self.last_level_counts = lcounts
                        self.last_level_occupancy = (lcounts / lcounts.sum()).float()

        return y.to(q.dtype)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.mode == "softmax":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            # Only diagnostics use an explicit score matrix in this mode.
            if self.collect_stats:
                with torch.no_grad(), torch.autocast(device_type=q.device.type,
                                                     enabled=False):
                    qf, kf = q.float(), k.float()
                    att = (qf @ kf.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    causal = torch.ones(T, T, dtype=torch.bool,
                                        device=q.device).tril()
                    self._record_common_stats(qf, kf, att, causal)
        elif self.mode == "exact_fp32_softmax":
            y = self._exact_fp32_softmax(q, k, v)
        elif self.mode == "quant":
            y = self._quant_attention(q, k, v)
        else:
            raise ValueError(f"unknown attention mode: {self.mode}")

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
