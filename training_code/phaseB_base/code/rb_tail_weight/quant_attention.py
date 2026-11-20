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

Phase B/P1 exposes controlled backward-only ablations while preserving the
forward operator:
- range_grad="detach": historical behavior; row min/max are detached.
- range_grad="detach_proj": same detached calibration backward, followed by a
  causal-valid row-wise zero-sum projection of the score gradient.
- range_grad="full": gradients flow through adaptive row min/max.

The FORWARD values are identical between all three modes.  Phase A1-A4 showed
that full range gradients restore the exact row-shift gradient symmetry while
preserving ordinary piecewise-smooth LERP derivatives.  P1 tests whether
removing only the spurious score-row common-mode component is sufficient to
prevent the historical long-horizon runaway.

Diagnostics include:
- score-row min/max/center/span statistics;
- Q/K activation norms and Q/K projection-weight norms;
- interval occupancy for quant modes;
- actual K+1 nearest reconstruction-level occupancy for nearest mode.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



def _project_causal_row_zero_sum_grad(grad: torch.Tensor,
                                     causal: torch.Tensor) -> torch.Tensor:
    """Project score gradients onto the causal-valid row-zero-sum subspace.

    For every query row i, only keys j <= i are included in the row mean:

        g'_ij = g_ij - mean_{k<=i}(g_ik).

    Invalid causal positions are left unchanged.  In the quantized attention
    path their incoming gradients are already zero because the forward masks
    their reconstructed weights to zero.

    This is a backward-only P1 ablation.  It removes exactly the row-wise
    common-mode component that is inconsistent with the forward identity
    F(S + c 1) = F(S), without restoring any other min/max calibration
    derivatives deleted by range_grad="detach".
    """
    if grad.ndim != 4:
        raise ValueError(f"expected grad with shape (B,H,T,T), got {tuple(grad.shape)}")
    T = grad.shape[-1]
    if grad.shape[-2] != T or causal.shape != (T, T):
        raise ValueError(
            f"shape mismatch: grad={tuple(grad.shape)}, causal={tuple(causal.shape)}"
        )

    valid = causal.view(1, 1, T, T)
    g_valid = grad.masked_fill(~valid, 0.0)
    n_valid = valid.sum(dim=-1, keepdim=True).to(dtype=grad.dtype)
    row_mean = g_valid.sum(dim=-1, keepdim=True) / n_valid
    return torch.where(valid, grad - row_mean, grad)


def _adaptive_interval_reconstruct(att: torch.Tensor,
                                   causal: torch.Tensor,
                                   K: int,
                                   approx: str,
                                   eps: float = 1e-12,
                                   return_level_idx: bool = False,
                                   range_grad: str = "detach"):

    """Reconstruct unnormalized attention weights on a per-row adaptive grid.

    Args:
        att:    (B,H,T,T) fp32 scaled QK scores, without causal fill values.
        causal: (T,T) boolean lower-triangular validity mask.
        K:      number of equal intervals spanning each row's valid min..max.
        approx: 'nearest' or 'lerp'.
        return_level_idx: when True, also return the actual nearest-boundary
            reconstruction level index in [0,K].  For LERP this is None.
        range_grad: 'detach' reproduces the historical calibration backward;
            'detach_proj' uses the same detached calibration values (the score
            gradient projection itself is installed by QuantCausalSelfAttention);
            'full' lets autograd differentiate through adaptive min/max.  All
            three have identical forward numerical values.

    Returns (default):
        w, bin_idx, row_span

    Returns (return_level_idx=True):
        w, bin_idx, row_span, level_idx
    """
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    if approx not in {"nearest", "lerp"}:
        raise ValueError(f"unknown approximation: {approx}")
    if range_grad not in {"detach", "detach_proj", "full"}:
        raise ValueError(
            "range_grad must be 'detach', 'detach_proj', or 'full', "
            f"got {range_grad}"
        )

    B, H, T, _ = att.shape
    valid = causal.view(1, 1, T, T)

    row_max_raw = att.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
    row_min_raw = att.masked_fill(~valid, torch.inf).amin(dim=-1, keepdim=True)
    if range_grad in {"detach", "detach_proj"}:
        # Historical detached calibration.  detach_proj differs only via the
        # score-gradient hook installed in QuantCausalSelfAttention below.
        row_max = row_max_raw.detach()
        row_min = row_min_raw.detach()
    else:
        # Phase-B ablation: same forward values, complete adaptive-calibration
        # derivative almost everywhere (away from min/max ties/bin boundaries).
        row_max = row_max_raw
        row_min = row_min_raw
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


def _relevant_zero_reconstruct(att: torch.Tensor,
                               causal: torch.Tensor,
                               K: int,
                               approx: str,
                               tau: float = 6.0,
                               return_level_idx: bool = False,
                               range_grad: str = "full",
                               tail_policy: str = "zero"):
    """Fixed softmax-relevant-range reconstruction.

    For every causal-valid row:

        z = S - max(S)
        relevant window = [-tau, 0]
        h = tau / K

    Scores z < -tau receive exactly zero unnormalized attention weight.
    Scores inside [-tau,0] use the same Nearest/LERP reconstruction rule
    as the original operator.

    range_grad:
        detach -> detach row_max calibration
        full   -> differentiate through row_max

    The fixed lower boundary -tau is not data-dependent.
    """

    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")

    if approx not in {"nearest", "lerp"}:
        raise ValueError(f"unknown approximation: {approx}")

    if range_grad not in {"detach", "full"}:
        raise ValueError(
            f"range_grad must be 'detach' or 'full', got {range_grad}"
        )

    if not (tau > 0):
        raise ValueError(f"tau must be > 0, got {tau}")

    B, H, T, _ = att.shape
    valid = causal.view(1, 1, T, T)

    # row_min is retained ONLY as a diagnostic raw-score span.
    # It does not calibrate the new quantizer.
    row_max_raw = (
        att.masked_fill(~valid, -torch.inf)
        .amax(dim=-1, keepdim=True)
    )

    row_min_raw = (
        att.masked_fill(~valid, torch.inf)
        .amin(dim=-1, keepdim=True)
    )

    if range_grad == "detach":
        row_max = row_max_raw.detach()
    else:
        row_max = row_max_raw

    # Max-relative score coordinate.
    z = att - row_max

    tau = float(tau)
    h = tau / float(K)

    # Use the fixed grid only for reconstruction.
    # The below-window mask is applied afterward as exact zero.
    zc = z.clamp(min=-tau, max=0.0)

    rel = (zc + tau) / h

    bin_idx = (
        torch.floor(rel)
        .to(torch.long)
        .clamp(0, K - 1)
    )

    bin_idx = torch.where(
        valid,
        bin_idx,
        torch.full_like(bin_idx, -1),
    )

    r_idx = bin_idx.clamp_min(0)
    r = r_idx.to(att.dtype)

    b_lo = -tau + r * h

    # Fixed K+1 exponential lookup values.
    grid = torch.arange(
        K + 1,
        device=att.device,
        dtype=att.dtype,
    )

    boundary_z = -tau + grid * h
    boundary_w = torch.exp(boundary_z)

    # Advanced indexing broadcasts naturally to (B,H,T,T).
    e_lo = boundary_w[r_idx]
    e_hi = boundary_w[(r_idx + 1).clamp_max(K)]

    t = ((zc - b_lo) / h).clamp(0.0, 1.0)

    w_lerp = (
        (1.0 - t) * e_lo
        + t * e_hi
    )

    level_idx = None

    if approx == "lerp":
        w = w_lerp

    else:
        choose_upper = t >= 0.5

        w_nearest = torch.where(
            choose_upper,
            e_hi,
            e_lo,
        )

        # Preserve the original Nearest STE exactly:
        # hard Nearest forward + LERP surrogate backward.
        w = (
            w_lerp
            + (w_nearest - w_lerp).detach()
        )

        if return_level_idx:
            level_idx = (
                r_idx
                + choose_upper.to(torch.long)
            ).clamp(0, K)

    # Tail policy. `zero` is the frozen FWM behaviour and its code path is unchanged.
    #
    #   zero      w = 0                      below -tau   (frozen FWM)
    #   floor     w = e^{-tau}               below -tau   (clamp already produced this; just skip
    #                                                      the zeroing)
    #   resolved  the SAME grid step h = tau/K continues downward without bound, so a tail key is
    #             reconstructed at its own resolution instead of being truncated or flattened.
    #
    # The tail mask is defined on the RAW z, so causally invalid positions are never treated as
    # tail and always fall through to the zero path, which masked_fill sets to 0 below.
    tail_mask = valid & (z < -tau)

    if tail_policy not in ("zero", "floor", "resolved"):
        raise ValueError(f"unknown tail_policy: {tail_policy}")

    if tail_policy == "zero":
        w = torch.where(
            tail_mask,
            torch.zeros_like(w),
            w,
        )
    elif tail_policy == "floor":
        # zc already clamped every tail score to -tau, so w is e^{-tau} there: keep it.
        pass
    else:
        # resolved.
        #
        # Computed from the RAW z, never from the clamped zc -- using zc would send every tail key
        # to -tau and silently turn this into `floor`.  z carries the gradient through the row max
        # under range_grad=full, which is retained here.
        #
        # Causally invalid positions are replaced by a finite value before the branch: torch.where
        # evaluates the gradient of BOTH branches, so a non-finite value in the unselected branch
        # would produce NaN * 0 = NaN and contaminate the whole gradient.
        z_safe = torch.where(valid, z, torch.zeros_like(z))

        rel_t = (z_safe + tau) / h                      # may be negative: no lower clamp
        r_t = torch.floor(rel_t)
        r_t = torch.clamp(r_t, max=float(K - 1))        # upper clamp only, as in the window
        b_lo_t = -tau + r_t * h
        e_lo_t = torch.exp(b_lo_t)
        e_hi_t = torch.exp(b_lo_t + h)
        t_t = ((z_safe - b_lo_t) / h).clamp(0.0, 1.0)   # exact-arithmetic no-op; guards rounding
        w_lerp_t = (1.0 - t_t) * e_lo_t + t_t * e_hi_t

        if approx == "lerp":
            w_tail = w_lerp_t
        else:
            choose_upper_t = t_t >= 0.5
            w_nearest_t = torch.where(choose_upper_t, e_hi_t, e_lo_t)
            # Same straight-through convention as the in-window path.
            w_tail = w_lerp_t + (w_nearest_t - w_lerp_t).detach()

        w = torch.where(
            tail_mask,
            w_tail,
            w,
        )

    w = w.masked_fill(
        ~valid,
        0.0,
    )

    if level_idx is not None:
        if tail_policy == "zero":
            keep = valid & ~tail_mask
        else:
            # floor and resolved both assign the tail a real level, so it is not excluded.
            keep = valid
        level_idx = torch.where(
            keep,
            level_idx,
            torch.full_like(level_idx, -1),
        )

    # Raw observed span retained for geometry diagnostics only.
    raw_row_span = (
        row_max_raw - row_min_raw
    )

    if return_level_idx:
        return (
            w,
            bin_idx,
            raw_row_span,
            level_idx,
            tail_mask,
        )

    return (
        w,
        bin_idx,
        raw_row_span,
        tail_mask,
    )


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
        self.range_grad = getattr(cfg, "attn_range_grad", "detach")
        if self.range_grad not in {"detach", "detach_proj", "full"}:
            raise ValueError(f"unknown attn_range_grad: {self.range_grad}")

        self.range_mode = getattr(
            cfg, "attn_range_mode", "minmax"
        )
        self.range_tau = float(
            getattr(cfg, "attn_range_tau", 6.0)
        )
        self.tail_policy = getattr(cfg, "attn_tail_policy", "zero")
        if self.tail_policy not in {"zero", "floor", "resolved"}:
            raise ValueError(f"unknown attn_tail_policy: {self.tail_policy}")

        if self.range_mode not in {"minmax", "relevant_zero"}:
            raise ValueError(
                f"unknown attn_range_mode: {self.range_mode}"
            )

        if self.range_tau <= 0:
            raise ValueError(
                f"attn_range_tau must be > 0, got {self.range_tau}"
            )

        self.collect_stats = False
        self.reset_last_stats()

    def reset_last_stats(self):
        # Common raw accumulators for one forward pass.  Raw sums/counts let
        # train.py pool statistics exactly across all layers.
        self.last_row_span_sum = None
        self.last_row_span_sq_sum = None
        self.last_row_min_sum = None
        self.last_row_max_sum = None
        self.last_row_center_sum = None
        self.last_row_center_sq_sum = None
        self.last_row_count = None
        self.last_q_norm_sum = None
        self.last_q_norm_sq_sum = None
        self.last_k_norm_sum = None
        self.last_k_norm_sq_sum = None
        self.last_qk_count = None
        self.last_q_proj_weight_norm = None
        self.last_k_proj_weight_norm = None

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
        centers = 0.5 * (row_min + row_max)

        qn = torch.linalg.vector_norm(qf, dim=-1)
        kn = torch.linalg.vector_norm(kf, dim=-1)

        self.last_row_span_sum = spans.sum().double().cpu()
        self.last_row_span_sq_sum = spans.square().sum().double().cpu()
        self.last_row_min_sum = row_min.sum().double().cpu()
        self.last_row_max_sum = row_max.sum().double().cpu()
        self.last_row_center_sum = centers.sum().double().cpu()
        self.last_row_center_sq_sum = centers.square().sum().double().cpu()
        self.last_row_count = int(spans.numel())

        self.last_q_norm_sum = qn.sum().double().cpu()
        self.last_q_norm_sq_sum = qn.square().sum().double().cpu()
        self.last_k_norm_sum = kn.sum().double().cpu()
        self.last_k_norm_sq_sum = kn.square().sum().double().cpu()
        self.last_qk_count = int(qn.numel())

        # c_attn is one fused [Q;K;V] projection.  These Frobenius norms let us
        # distinguish activation-scale drift from projection-weight growth.
        qw, kw, _ = self.c_attn.weight.split(self.n_embd, dim=0)
        self.last_q_proj_weight_norm = torch.linalg.vector_norm(qw).double().cpu()
        self.last_k_proj_weight_norm = torch.linalg.vector_norm(kw).double().cpu()

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

            if self.range_grad == "detach_proj" and att.requires_grad:
                # P1: keep the historical detached-calibration backward, but
                # remove only the causal-valid row common-mode score gradient.
                # register_hook is backward-only, so forward values are unchanged.
                att.register_hook(
                    lambda grad, causal=causal:
                        _project_causal_row_zero_sum_grad(grad, causal)
                )

            tail_mask = None

            if self.range_mode == "minmax":
                w, bins, row_span, level_idx = _adaptive_interval_reconstruct(
                    att,
                    causal,
                    self.K,
                    self.approx,
                    return_level_idx=True,
                    range_grad=self.range_grad,
                )

            elif self.range_mode == "relevant_zero":
                (
                    w,
                    bins,
                    row_span,
                    level_idx,
                    tail_mask,
                ) = _relevant_zero_reconstruct(
                    att,
                    causal,
                    self.K,
                    self.approx,
                    tau=self.range_tau,
                    return_level_idx=True,
                    range_grad=self.range_grad,
                    tail_policy=self.tail_policy,
                )

            else:
                raise RuntimeError(
                    f"unknown range_mode: {self.range_mode}"
                )
            denom = w.sum(dim=-1, keepdim=True)
            P = w / denom
            y = P @ vf

            if self.collect_stats:
                with torch.no_grad():
                    self._record_common_stats(qf, kf, att, causal)
                    valid = causal.view(1, 1, T, T).expand(B, H, T, T)

                    stat_valid = valid

                    if tail_mask is not None:
                        stat_valid = (
                            valid
                            & ~tail_mask
                        )

                    valid_bins = bins[stat_valid]

                    counts = torch.bincount(
                        valid_bins,
                        minlength=self.K,
                    ).double().cpu()

                    self.last_interval_counts = counts
                    self.last_occupancy = (
                        counts / counts.sum()
                    ).float()

                    if level_idx is not None:
                        valid_levels = level_idx[
                            stat_valid
                        ]

                        lcounts = torch.bincount(
                            valid_levels,
                            minlength=self.K + 1,
                        ).double().cpu()

                        self.last_level_counts = lcounts
                        self.last_level_occupancy = (
                            lcounts / lcounts.sum()
                        ).float()

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
