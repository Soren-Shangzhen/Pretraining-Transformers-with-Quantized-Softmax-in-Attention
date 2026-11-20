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

centroid / centroid2 (centroid_v1, 2026-10-13):
    same calibration and same cells as the existing operator (default: the
    K+1 grid-centred Nearest rounding cells); only the representative weight
    of each cell changes.  centroid uses the cell mean of exp(z) (cell mass is
    conserved exactly), centroid2 uses exp(mean(z) + var(z)/2) from three
    per-cell accumulators.  Autograd differentiates the representative w.r.t.
    every member; no straight-through surrogate.  See _centroid_reconstruct.

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
                               range_grad: str = "full"):
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

    # Hard-zero softmax-irrelevant tail.
    tail_mask = valid & (z < -tau)

    w = torch.where(
        tail_mask,
        torch.zeros_like(w),
        w,
    )

    w = w.masked_fill(
        ~valid,
        0.0,
    )

    if level_idx is not None:
        level_idx = torch.where(
            valid & ~tail_mask,
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


CENTROID_RULES = ("centroid", "centroid2")
CELL_MODES = ("nearest", "interval")


def _centroid_cells(att: torch.Tensor,
                    causal: torch.Tensor,
                    K: int,
                    range_mode: str,
                    tau: float):
    """Calibration and cell indices of the existing operator, without gradient.

    The expressions repeat those of _adaptive_interval_reconstruct (MinMax)
    and _relevant_zero_reconstruct (FW) line by line, so bin_idx, level_idx
    and tail_mask are bitwise identical to theirs (checked in
    checks/check_centroid_ops.py and checks/check_gpu_step0.py).

    Returns bin_idx, level_idx, row_span, tail_mask, member, degenerate,
    z_lo, h (z_lo and h: lower window edge and interval width, broadcastable
    to (B,H,T,1)).
    """
    B, H, T, _ = att.shape
    valid = causal.view(1, 1, T, T)
    with torch.no_grad():
        row_max = att.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
        row_min = att.masked_fill(~valid, torch.inf).amin(dim=-1, keepdim=True)
        z = att - row_max
        if range_mode == "minmax":
            row_span = row_max - row_min
            z_min = row_min - row_max
            h = row_span / float(K)
            degenerate = row_span <= 1e-12
            h_safe = torch.where(degenerate, torch.ones_like(h), h)
            rel = (z - z_min) / h_safe
            r_idx = torch.floor(rel).to(torch.long).clamp_(0, K - 1)
            b_lo = z_min + r_idx.to(att.dtype) * h_safe
            t = ((z - b_lo) / h_safe).clamp(0.0, 1.0)
            level_idx = (r_idx + (t >= 0.5).to(torch.long)).clamp_(0, K)
            level_idx = torch.where(degenerate.expand_as(level_idx),
                                    torch.zeros_like(level_idx), level_idx)
            tail_mask = None
            member = valid.expand(B, H, T, T)
            z_lo, h_cell = z_min, h_safe
        elif range_mode == "relevant_zero":
            tau = float(tau)
            h = tau / float(K)
            zc = z.clamp(min=-tau, max=0.0)
            rel = (zc + tau) / h
            r_idx = torch.floor(rel).to(torch.long).clamp(0, K - 1)
            b_lo = -tau + r_idx.to(att.dtype) * h
            t = ((zc - b_lo) / h).clamp(0.0, 1.0)
            level_idx = (r_idx + (t >= 0.5).to(torch.long)).clamp(0, K)
            tail_mask = valid & (z < -tau)
            member = valid & ~tail_mask
            degenerate = None
            row_span = row_max - row_min
            z_lo = torch.full((1, 1, 1, 1), -tau, device=att.device, dtype=att.dtype)
            h_cell = torch.full((1, 1, 1, 1), h, device=att.device, dtype=att.dtype)
        else:
            raise ValueError(f"unknown range_mode: {range_mode}")
        bin_idx = torch.where(valid, r_idx, torch.full_like(r_idx, -1))
        level_idx = torch.where(member, level_idx, torch.full_like(level_idx, -1))
    return bin_idx, level_idx, row_span, tail_mask, member, degenerate, z_lo, h_cell


def _centroid_reconstruct(att: torch.Tensor,
                          causal: torch.Tensor,
                          K: int,
                          rule: str,
                          range_mode: str = "minmax",
                          tau: float = 6.0,
                          cells: str = "nearest",
                          range_grad: str = "full"):
    """Cell-mass-conserving reconstruction on the existing partition.

    Cells (from _centroid_cells, identical to the existing operator):

        cells="nearest":  the K+1 grid-centred rounding cells of the hard
                          Nearest rule, index l_j = r_j + 1[t_j >= 1/2];
        cells="interval": the K calibration intervals, index r_j.

    Under FW, z < -tau keeps exactly zero weight and is not a member of any
    cell (same tail policy as FW Nearest/LERP).  Every member j of cell b gets

        centroid:   w*_b = mean_{i in b} exp(z_i)
        centroid2:  w*_b = exp(mean_b(z) + var_b(z) / 2)

    centroid2 accumulates count, sum(u), sum(u^2) with u = z - c_b, where c_b
    is the cell's grid centre (a constant shift that only conditions the fp32
    sums; mean and variance are unchanged by it).  Cell membership is piecewise
    constant and carries no derivative; given the membership both
    representatives scale with exp(-M), so the normalized output does not
    depend on the calibration statistics except through the membership.

    Per-cell sums use scatter_add over the key axis and the representative is
    read back with gather; on CUDA both use atomic adds, so results are not
    bitwise reproducible run to run (fp32 summation-order noise).

    Degenerate MinMax rows (span <= eps) keep weight 1 on every valid key, as
    in the existing operator.

    Returns w, bin_idx, row_span, level_idx, tail_mask (tail_mask is None for
    MinMax).
    """
    if rule not in CENTROID_RULES:
        raise ValueError(f"unknown centroid rule: {rule}")
    if cells not in CELL_MODES:
        raise ValueError(f"unknown cell mode: {cells}")
    if range_grad not in {"detach", "full"}:
        raise ValueError(
            f"range_grad must be 'detach' or 'full' for {rule}, got {range_grad}"
        )

    B, H, T, _ = att.shape
    valid = causal.view(1, 1, T, T)
    (bin_idx, level_idx, row_span, tail_mask, member, degenerate,
     z_lo, h_cell) = _centroid_cells(att, causal, K, range_mode, tau)

    if cells == "nearest":
        n_cells, cell, centre_offset = K + 1, level_idx, 0.0
    else:
        n_cells, cell, centre_offset = K, bin_idx, 0.5
    member = member & (cell >= 0)
    # Non-members (causally invalid keys, FW tail) go to an extra dump slot.
    idx = torch.where(member, cell, torch.full_like(cell, n_cells))

    row_max = att.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
    if range_grad == "detach":
        row_max = row_max.detach()
    z = torch.where(member, att - row_max, torch.zeros_like(att))

    slots = att.new_zeros(B, H, T, n_cells + 1)
    with torch.no_grad():
        cnt = slots.scatter_add(-1, idx, member.to(att.dtype)).clamp_min(1.0)

    if rule == "centroid":
        e = torch.where(member, torch.exp(z), torch.zeros_like(z))
        rep = slots.scatter_add(-1, idx, e) / cnt
    else:
        with torch.no_grad():
            grid = torch.arange(n_cells + 1, device=att.device, dtype=att.dtype)
            centre_tab = (z_lo + (grid + centre_offset) * h_cell).expand(B, H, T, n_cells + 1)
            centre = centre_tab.gather(-1, idx)
        u = torch.where(member, z - centre, torch.zeros_like(z))
        mean_u = slots.scatter_add(-1, idx, u) / cnt
        var_u = (slots.scatter_add(-1, idx, u * u) / cnt - mean_u * mean_u).clamp_min(0.0)
        rep = torch.exp(centre_tab + mean_u + 0.5 * var_u)

    w = torch.where(member, rep.gather(-1, idx), torch.zeros_like(z))
    if degenerate is not None:
        w = torch.where(degenerate, torch.ones_like(w), w)
    w = w.masked_fill(~valid, 0.0)
    return w, bin_idx, row_span, level_idx, tail_mask


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

        if self.range_mode not in {"minmax", "relevant_zero"}:
            raise ValueError(
                f"unknown attn_range_mode: {self.range_mode}"
            )

        if self.range_tau <= 0:
            raise ValueError(
                f"attn_range_tau must be > 0, got {self.range_tau}"
            )

        self.cells = getattr(cfg, "attn_cells", "nearest")
        if self.cells not in CELL_MODES:
            raise ValueError(f"unknown attn_cells: {self.cells}")
        if self.approx in CENTROID_RULES and self.range_grad not in {"detach", "full"}:
            raise ValueError(f"{self.approx} supports range_grad detach/full only")

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

            if self.approx in CENTROID_RULES:
                w, bins, row_span, level_idx, tail_mask = _centroid_reconstruct(
                    att,
                    causal,
                    self.K,
                    self.approx,
                    range_mode=self.range_mode,
                    tau=self.range_tau,
                    cells=self.cells,
                    range_grad=self.range_grad,
                )

            elif self.range_mode == "minmax":
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
