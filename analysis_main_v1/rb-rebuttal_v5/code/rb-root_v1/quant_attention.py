"""Adaptive K-interval attention for training -- rebuttal root rb-root_v1 (2026-11-10).

Derived from the strict-forward supplement roots of Appendix C.2:
  supplement_four_runs_20260923/code/e2_fwm_probste_strict    (quant_attention 98fe5b93..)
  supplement_four_runs_20260923/code/e2_minmax_probste_strict (quant_attention 92291324..)
The functions and branches those roots already had are kept verbatim; every addition below is
marked "rb-root_v1".  New Nearest runs use the strict hard forward of Appendix C.2:
    Prob-STE    P = sg(P_N) + (P_L - sg(P_L))
    Weight-STE  w = sg(w_N) + (w_L - sg(w_L)),  P = w / sum(w)

Forward definitions of the original operators are unchanged:

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

range_grad (backward only; forward values identical):
- "detach": row min/max (MinMax) or row max (FWM) are detached.
- "detach_proj": detach + causal-valid row zero-sum projection of the score gradient.
- "full": gradients flow through the calibration extrema.

rb-root_v1 additions (all fp32 inside a disabled-autocast region, row max on the autograd path):
- relevant_zero tail_policy "resolved" in the strict code path, with an optional per-layer window
  (E2 freeze-h: tau_l = K * h0_l, so the step is h0_l and the grid continues below -tau_l).
- range_mode "ema" (E4): per-head window width = EMA of the head's mean row span (buffer, not in
  the graph), zero tail, row max in the graph.
- attn modes softmax_mxfp8 / softmax_nvfp4 / softmax_uint8 (E5): e^z quantised in the format,
  forward bit-exact quantised value, backward = fp32 exponential derivative.
- attn mode softmax_bf16chain (E5n): exponential, row sum and division in bfloat16, native autograd.
- attn modes softmax_bf16exp / softmax_fp8e4m3exp (E5, Appendix D.5 modes, same expressions).
- attn mode indexsoftmax (E3a): table lookup forward (variant fp: float normalisation; variant u8:
  UINT8 probabilities round(255 e/sum e)/255), Prob-STE onto an FWM-LERP surrogate on the same
  32 grid points (tau=6.6, K=31).
- attn mode exaq (E3b): EXAQ 3-bit forward with tensor-level window, Prob-STE onto a LERP surrogate
  on the same 8 grid points; variant detach: sigma/min constants; variant graph: differentiable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Supplement E2 (2026-09-23): surrogate composition of the Nearest path.
#   "prob"          -> P = P_N.detach() + (P_L - P_L.detach())   (training)
#   "weight_strict" -> w = w_N.detach() + (w_L - w_L.detach()); P = w / sum(w)
# Selected by train_formal.py --nearest_surrogate (recorded in the checkpoint config).
NEAREST_SURROGATE = "prob"

# rb-root_v1: constants of the two external operators (E3a, E3b).  Same values and operation
# order as the reference forward passes rb-rebuttal_v5/code/e3_ref/ref_ops.py.
IDX_THR = 6.6       # clipping threshold in nats
IDX_BINS = 31       # 5-bit index, 32 table entries
# round(255 * exp(-i * 6.6 / 31)), first entry 255, last entry 0
IDX_TABLE = (255, 206, 167, 135, 109, 88, 71, 57, 46, 38, 30, 25, 20, 16, 13,
             10, 8, 7, 6, 4, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0)
# linear rule C* = a * sigma + b for the clipping point, per bit width
EXAQ_COEF = {2: (-1.66, -1.85), 3: (-1.75, -2.06), 4: (-1.83, -2.27)}

def _c(x, ref: torch.Tensor) -> torch.Tensor:
    """0-dim constant on ref's device.  Division by a Python scalar runs on CUDA as multiplication by
    the reciprocal (1 ulp from true division); a device tensor divisor gives IEEE division on CPU and
    GPU alike, so the quantised values below are the same on both."""
    return torch.tensor(x, dtype=ref.dtype, device=ref.device)


SOFTMAX_FAMILY = ("softmax_bf16exp", "softmax_fp8e4m3exp", "softmax_mxfp8",
                  "softmax_nvfp4", "softmax_uint8", "softmax_bf16chain")



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
                                   range_grad: str = "detach",
                                   nearest_forward: str = "ste"):

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
    if nearest_forward not in {"ste", "direct"}:
        raise ValueError(f"unknown nearest_forward: {nearest_forward}")
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
        if nearest_forward == "direct":
            # Supplement E2 (2026-09-23): the hard-Nearest weights themselves,
            # not re-derived through w_lerp + (w_nearest - w_lerp).
            w = w_nearest
        else:
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
                               nearest_forward: str = "ste",
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

    rb-root_v1: tail_policy "resolved" (Appendix D.4, moved into the strict code path): the same
    step h continues below -tau without bound, so a tail key is reconstructed on the same grid
    instead of being zeroed.  "zero" is the frozen FWM behaviour and its code path is unchanged.
    """

    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")

    if approx not in {"nearest", "lerp"}:
        raise ValueError(f"unknown approximation: {approx}")

    if nearest_forward not in {"ste", "direct"}:
        raise ValueError(f"unknown nearest_forward: {nearest_forward}")

    if range_grad not in {"detach", "full"}:
        raise ValueError(
            f"range_grad must be 'detach' or 'full', got {range_grad}"
        )

    if tail_policy not in {"zero", "resolved"}:
        raise ValueError(f"unknown tail_policy: {tail_policy}")

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

        if nearest_forward == "direct":
            # Supplement E2 (2026-09-23): the hard-Nearest weights themselves,
            # not re-derived through w_lerp + (w_nearest - w_lerp).
            w = w_nearest
        else:
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

    if tail_policy == "zero":
        w = torch.where(
            tail_mask,
            torch.zeros_like(w),
            w,
        )
    else:
        # rb-root_v1, resolved (code of rb_tail_weight/rb_tail_prob, Appendix D.4, with the strict
        # Nearest forward added).  Computed from the RAW z, never from the clamped zc: using zc
        # would send every tail key to -tau.  z carries the gradient through the row max under
        # range_grad=full.  Causally invalid positions get a finite value before the branch, because
        # torch.where evaluates the gradient of both branches.
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
            if nearest_forward == "direct":
                w_tail = w_nearest_t
            else:
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
        keep = (valid & ~tail_mask) if tail_policy == "zero" else valid
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


# --------------------------------------------------------------------------------------------
# rb-root_v1 additions
# --------------------------------------------------------------------------------------------

def _tensor_window_reconstruct(att: torch.Tensor,
                               causal: torch.Tensor,
                               K: int,
                               approx: str,
                               tau: torch.Tensor,
                               nearest_forward: str = "ste"):
    """E4: window [-tau_h, 0] anchored at the row maximum with a per-head width tensor.

    tau: (1, H, 1, 1) fp32, a buffer value (not in the graph).  h = tau / K, grid -tau + r h,
    r = 0..K, zero weight below the window (as FWM), row maximum on the autograd path.
    Returns w, bin_idx, raw_row_span, level_idx, tail_mask.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    if approx not in {"nearest", "lerp"}:
        raise ValueError(f"unknown approximation: {approx}")
    B, H, T, _ = att.shape
    valid = causal.view(1, 1, T, T)
    tau = tau.detach().to(att.dtype)
    if not bool((tau > 0).all()):
        raise ValueError("EMA window width must be > 0")

    row_max = att.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
    row_min = att.masked_fill(~valid, torch.inf).amin(dim=-1, keepdim=True)
    z = att - row_max
    h = tau / float(K)
    zc = torch.minimum(torch.maximum(z, -tau), torch.zeros_like(z))
    rel = (zc + tau) / h
    bin_idx = torch.floor(rel).to(torch.long).clamp(0, K - 1)
    bin_idx = torch.where(valid, bin_idx, torch.full_like(bin_idx, -1))
    r = bin_idx.clamp_min(0).to(att.dtype)
    b_lo = -tau + r * h
    e_lo = torch.exp(b_lo)
    e_hi = torch.exp(-tau + (r + 1.0) * h)
    t = ((zc - b_lo) / h).clamp(0.0, 1.0)
    w_lerp = (1.0 - t) * e_lo + t * e_hi
    level_idx = None
    if approx == "lerp":
        w = w_lerp
    else:
        choose_upper = t >= 0.5
        w_nearest = torch.where(choose_upper, e_hi, e_lo)
        w = w_nearest if nearest_forward == "direct" else w_lerp + (w_nearest - w_lerp).detach()
        level_idx = (bin_idx.clamp_min(0) + choose_upper.to(torch.long)).clamp(0, K)
    tail_mask = valid & (z < -tau)
    w = torch.where(tail_mask, torch.zeros_like(w), w)
    w = w.masked_fill(~valid, 0.0)
    if level_idx is not None:
        level_idx = torch.where(valid & ~tail_mask, level_idx, torch.full_like(level_idx, -1))
    return w, bin_idx, (row_max - row_min), level_idx, tail_mask


def _round_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round nonnegative fp32 values to the nearest FP8 E4M3 value, ties to even, back in fp32.
    Callers only pass values <= 448 (up to fp32 rounding of the scale product); the clamp keeps an
    out-of-range input from becoming NaN in the e4m3fn cast."""
    return x.clamp(max=448.0).to(torch.float8_e4m3fn).to(torch.float32)


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round nonnegative fp32 values to the nearest FP4 E2M1 magnitude {0,.5,1,1.5,2,3,4,6}.

    Ties go to the even code (0, 1, 2, 4 have an even mantissa bit; .5, 1.5, 3, 6 odd);
    values above 6 saturate to 6.
    """
    q = torch.zeros_like(x)
    q = torch.where(x > 0.25, torch.full_like(x, 0.5), q)
    q = torch.where(x >= 0.75, torch.full_like(x, 1.0), q)
    q = torch.where(x > 1.25, torch.full_like(x, 1.5), q)
    q = torch.where(x >= 1.75, torch.full_like(x, 2.0), q)
    q = torch.where(x > 2.5, torch.full_like(x, 3.0), q)
    q = torch.where(x >= 3.5, torch.full_like(x, 4.0), q)
    q = torch.where(x > 5.0, torch.full_like(x, 6.0), q)
    return q


def _mxfp8_quantize(w: torch.Tensor, block: int = 32) -> torch.Tensor:
    """MXFP8-style: blocks of 32 along the key axis, scale X = smallest power of two with
    block_max / X <= 448 (OCP E8M0 range), elements w / X rounded to E4M3 (nearest, ties to even),
    restored value X * e4m3(w / X).  An all-zero block returns zeros.  No gradient."""
    T = w.shape[-1]
    if T % block:
        raise ValueError(f"key length {T} is not a multiple of the block size {block}")
    wb = w.reshape(*w.shape[:-1], T // block, block)
    bmax = wb.amax(dim=-1, keepdim=True)
    pos = bmax > 0
    bmax_d = torch.where(pos, bmax, torch.ones_like(bmax)).double()
    e = torch.ceil(torch.log2(bmax_d / 448.0))
    # exact fix-up: smallest e with bmax <= 448 * 2^e (both sides exact in fp64)
    e = torch.where(bmax_d > 448.0 * torch.exp2(e), e + 1.0, e)
    e = torch.where(bmax_d <= 448.0 * torch.exp2(e - 1.0), e - 1.0, e)
    e = e.clamp(-127.0, 127.0)
    X = torch.exp2(e).to(torch.float32)
    q = _round_to_e4m3(wb / X) * X
    q = torch.where(pos, q, torch.zeros_like(q))
    return q.reshape(w.shape)


def _nvfp4_quantize(w: torch.Tensor, valid: torch.Tensor, block: int = 16) -> torch.Tensor:
    """NVFP4-style: tensor scale s_enc = 6*448 / max(w) over the valid entries of the whole tensor
    (one layer, one forward); blocks of 16 along the key axis with S_b = E4M3(block_max / 6 * s_enc);
    elements q = E2M1(w * s_enc / S_b) (nearest, ties to even, saturating); restored q * S_b / s_enc.
    A block whose S_b is 0 returns zeros.  No gradient."""
    T = w.shape[-1]
    if T % block:
        raise ValueError(f"key length {T} is not a multiple of the block size {block}")
    tmax = w.masked_fill(~valid, 0.0).amax()
    s_enc = _c(6.0 * 448.0, w) / tmax
    wb = w.reshape(*w.shape[:-1], T // block, block)
    bmax = wb.amax(dim=-1, keepdim=True)
    S_b = _round_to_e4m3((bmax / _c(6.0, w)) * s_enc)
    ok = S_b > 0
    S_safe = torch.where(ok, S_b, torch.ones_like(S_b))
    q = _round_to_e2m1((wb * s_enc) / S_safe)
    out = (q * S_safe) / s_enc
    out = torch.where(ok, out, torch.zeros_like(out))
    return out.reshape(w.shape)


def _uint8_quantize(w: torch.Tensor) -> torch.Tensor:
    """UINT8 uniform on [0,1]: round(255 w) / 255, round half to even.  No gradient."""
    return torch.round(w * 255.0) / _c(255.0, w)


def _index_softmax_hard(att: torch.Tensor, valid: torch.Tensor, variant: str,
                        thr: float = IDX_THR) -> torch.Tensor:
    """E3a hard forward (no gradient).  Index trunc((d * 31) / thr) with d = min(rowmax - s, thr);
    table value e in {0..255}; masked keys 0.  variant 'fp': P = e / sum e.  variant 'u8':
    P = round(255 e / sum e) / 255 (rows need not sum to 1)."""
    s = att.masked_fill(~valid, float("-inf"))
    d = s.amax(dim=-1, keepdim=True) - s
    d = torch.clamp(d, max=thr)
    idx = ((d * IDX_BINS) / thr).to(torch.int64)          # truncation
    table = torch.tensor(IDX_TABLE, dtype=torch.float32, device=att.device)
    e = table[idx].masked_fill(~valid, 0.0)
    if variant == "fp":
        return e / e.sum(dim=-1, keepdim=True)
    if variant == "u8":
        # codes round(255 e / sum e) exactly as the reference; the value is the correctly rounded
        # code / 255 on every device (see _c)
        return ((e * 255.0) / e.sum(dim=-1, keepdim=True)).round() / _c(255.0, e)
    raise ValueError(f"unknown indexsoftmax variant: {variant}")


def _exaq_window(att: torch.Tensor, valid_full: torch.Tensor, bits: int):
    """EXAQ tensor-level window: x = s - rowmax; sigma (population std) and min over every valid
    entry of the whole tensor, recomputed on every call; C* = a sigma + b; lo = max(C*, min);
    delta = -lo / (2^bits - 1).  lo and delta are Python floats."""
    s = att.masked_fill(~valid_full, float("-inf"))
    x = s - s.amax(dim=-1, keepdim=True)
    xv = x[valid_full]
    sigma = xv.float().std(unbiased=False).item()
    a, b = EXAQ_COEF[bits]
    c = a * sigma + b
    if not math.isfinite(c) or c >= 0.0:
        c = -1e-6
    lo = max(c, xv.min().item())
    delta = (0.0 - lo) / float(2 ** bits - 1)
    return x, lo, delta


@torch.no_grad()
def _exaq_hard(att: torch.Tensor, valid_full: torch.Tensor, bits: int):
    """E3b hard forward: clamp x to [lo, 0], round to 2**bits levels, exact softmax.
    Returns (P, lo, delta)."""
    x, lo, delta = _exaq_window(att, valid_full, bits)
    x = x.clone()
    xv = x[valid_full].clamp_(min=lo, max=0.0)
    if math.isfinite(delta) and delta > 0.0:
        q = torch.round((xv - lo) / delta).clamp_(0, 2 ** bits - 1)
        xv = lo + q * delta
    x[valid_full] = xv
    return torch.softmax(x, dim=-1), lo, delta


def _exaq_surrogate(att: torch.Tensor, valid: torch.Tensor, valid_full: torch.Tensor,
                    bits: int, variant: str, lo_f: float, delta_f: float) -> torch.Tensor:
    """E3b surrogate P_L: LERP of the exact exponential on the 2**bits grid points lo + k delta,
    argument clamped to [lo, 0], fp32 normalisation, row maximum in the graph.
    variant 'detach': lo, delta are the constants of the hard forward (observer convention).
    variant 'graph':  lo = max(a sigma + b, min) from differentiable sigma and min of the same
                      valid entries, delta = -lo / (2^bits - 1)."""
    n_lev = 2 ** bits - 1
    row_max = att.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
    x = att - row_max
    if variant == "detach":
        lo = torch.tensor(lo_f, dtype=att.dtype, device=att.device)
        delta = torch.tensor(delta_f, dtype=att.dtype, device=att.device)
    elif variant == "graph":
        xv = x[valid_full]
        sigma = xv.std(unbiased=False)
        a, b = EXAQ_COEF[bits]
        c = a * sigma + b
        c = torch.where(torch.isfinite(c) & (c < 0), c, torch.full_like(c, -1e-6))
        lo = torch.maximum(c, xv.min())
        delta = (0.0 - lo) / float(n_lev)
    else:
        raise ValueError(f"unknown exaq variant: {variant}")
    if not (math.isfinite(delta_f) and delta_f > 0.0):
        # degenerate tensor (every valid score equals its row max): uniform rows, no score gradient
        w = valid.to(att.dtype).expand_as(att)
        return w / w.sum(dim=-1, keepdim=True)
    xc = torch.minimum(torch.maximum(x, lo), torch.zeros_like(x))
    k = torch.floor((xc - lo) / delta).clamp(0, n_lev - 1)
    g_lo = lo + k * delta
    t = ((xc - g_lo) / delta).clamp(0.0, 1.0)
    w = (1.0 - t) * torch.exp(g_lo) + t * torch.exp(g_lo + delta)
    w = w.masked_fill(~valid, 0.0)
    return w / w.sum(dim=-1, keepdim=True)


def softmax_family_probs(att: torch.Tensor, causal: torch.Tensor, kind: str) -> torch.Tensor:
    """Probabilities of the SOFTMAX_FAMILY modes from fp32 scores (B,H,T,T).  Exact-softmax path with
    only the production of e^z (and, for bf16chain, the normalisation) changed; the row maximum
    stays on the autograd path in every kind."""
    T = att.shape[-1]
    valid = causal.view(1, 1, T, T)
    masked_att = att.masked_fill(~valid, -torch.inf)
    row_max = masked_att.amax(dim=-1, keepdim=True)
    z = masked_att - row_max

    if kind == "softmax_bf16chain":
        # E5n: exponential, row sum and division on bfloat16 tensors (PyTorch reductions
        # accumulate internally in fp32 and round the result to bf16); native autograd.
        wb = torch.exp(z.to(torch.bfloat16))
        wb = wb.masked_fill(~valid, 0.0)
        return (wb / wb.sum(dim=-1, keepdim=True)).float()
    if kind == "softmax_bf16exp":
        # Appendix D.5 expression, unchanged: native autograd through the casts.
        w = torch.exp(z.to(torch.bfloat16)).float()
    else:
        w_exp = torch.exp(z)
        w_d = w_exp.detach()
        if kind == "softmax_fp8e4m3exp":
            w_q = w_d.to(torch.float8_e4m3fn).float()
        elif kind == "softmax_mxfp8":
            w_q = _mxfp8_quantize(w_d)
        elif kind == "softmax_nvfp4":
            w_q = _nvfp4_quantize(w_d, valid)
        elif kind == "softmax_uint8":
            w_q = _uint8_quantize(w_d)
        else:
            raise ValueError(f"unknown softmax-family mode: {kind}")
        # forward: the quantised value bit for bit; backward: fp32 exp derivative
        w = w_q + (w_exp - w_exp.detach())
    w = w.masked_fill(~valid, 0.0)
    return w / w.sum(dim=-1, keepdim=True)


def external_operator_probs(att: torch.Tensor, causal: torch.Tensor, mode: str, variant: str,
                            exaq_bits: int = 3) -> torch.Tensor:
    """E3a IndexSoftmax / E3b EXAQ probabilities: strict hard forward, Prob-STE onto a LERP surrogate
    on the operator's own grid (row maximum on the autograd path)."""
    B, H, T, _ = att.shape
    valid = causal.view(1, 1, T, T)
    if mode == "indexsoftmax":
        with torch.no_grad():
            P_hard = _index_softmax_hard(att.detach(), valid, variant)
        # surrogate: FWM-LERP on the same 32 grid points, full row-max gradient
        w_lerp, _, _, _ = _relevant_zero_reconstruct(
            att, causal, IDX_BINS, "lerp", tau=IDX_THR, range_grad="full")
        P_lerp = w_lerp / w_lerp.sum(dim=-1, keepdim=True)
    elif mode == "exaq":
        valid_full = valid.expand(B, H, T, T)
        P_hard, lo_f, delta_f = _exaq_hard(att.detach(), valid_full, exaq_bits)
        P_lerp = _exaq_surrogate(att, valid, valid_full, exaq_bits, variant, lo_f, delta_f)
    else:
        raise ValueError(f"unknown external operator: {mode}")
    return P_hard.detach() + (P_lerp - P_lerp.detach())


def ema_window_probs(att: torch.Tensor, causal: torch.Tensor, K: int, tau_t: torch.Tensor,
                     surrogate: str):
    """E4 probabilities for a per-head window tau_t (1,H,1,1): strict hard Nearest forward,
    Weight-STE ('weight_strict') or Prob-STE ('prob') onto the LERP surrogate on the same grid.
    Returns P, w_hard, bins, row_span, level_idx, tail_mask."""
    with torch.no_grad():
        w_hard, bins, row_span, level_idx, tail_mask = _tensor_window_reconstruct(
            att, causal, K, "nearest", tau_t, nearest_forward="direct")
        P_hard = w_hard / w_hard.sum(dim=-1, keepdim=True)
    w_lerp, _, _, _, _ = _tensor_window_reconstruct(att, causal, K, "lerp", tau_t)
    if surrogate == "prob":
        P_lerp = w_lerp / w_lerp.sum(dim=-1, keepdim=True)
        P = P_hard.detach() + (P_lerp - P_lerp.detach())
    elif surrogate == "weight_strict":
        w_ref = w_hard.detach() + (w_lerp - w_lerp.detach())
        P = w_ref / w_ref.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown NEAREST_SURROGATE: {surrogate}")
    return P, w_hard, bins, row_span, level_idx, tail_mask


class QuantCausalSelfAttention(nn.Module):
    """Causal self-attention with matched controls and diagnostics.

    Modes:
        softmax:            PyTorch SDPA baseline.
        exact_fp32_softmax: explicit FP32 QK^T -> exact softmax -> PV control.
        quant:              adaptive K-interval Nearest/LERP attention.
        rb-root_v1:         SOFTMAX_FAMILY, indexsoftmax, exaq (see the module docstring).
    """

    def __init__(self, cfg, layer_idx=None):
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

        if self.range_mode not in {"minmax", "relevant_zero", "ema"}:
            raise ValueError(
                f"unknown attn_range_mode: {self.range_mode}"
            )

        # rb-root_v1 fields
        self.layer_idx = layer_idx
        self.tail_policy = getattr(cfg, "attn_tail_policy", "zero")
        if self.tail_policy not in {"zero", "resolved"}:
            raise ValueError(f"unknown attn_tail_policy: {self.tail_policy}")
        per_layer = getattr(cfg, "attn_range_tau_per_layer", None)
        if per_layer:
            if layer_idx is None or len(per_layer) != cfg.n_layer:
                raise ValueError("attn_range_tau_per_layer needs one value per layer")
            self.range_tau = float(per_layer[layer_idx])
        self.variant = getattr(cfg, "attn_variant", "")
        self.exaq_bits = int(getattr(cfg, "attn_exaq_bits", 3))
        self.ema_decay = float(getattr(cfg, "attn_ema_decay", 0.999))
        if self.mode == "quant" and self.range_mode == "ema":
            # per-head EMA of the mean row span; NaN until the first training forward
            self.register_buffer("ema_span", torch.full((cfg.n_head,), float("nan")))
        if self.mode == "indexsoftmax" and self.variant not in {"fp", "u8"}:
            raise ValueError(f"indexsoftmax needs attn_variant fp|u8, got {self.variant!r}")
        if self.mode == "exaq" and self.variant not in {"detach", "graph"}:
            raise ValueError(f"exaq needs attn_variant detach|graph, got {self.variant!r}")

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

    # ---------------------------------------------------------------- rb-root_v1 modes
    def _softmax_family(self, q, k, v):
        B, H, T, D = q.shape
        with torch.autocast(device_type=q.device.type, enabled=False):
            qf, kf, vf = q.float(), k.float(), v.float()
            att = (qf @ kf.transpose(-2, -1)) / math.sqrt(D)
            causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            P = softmax_family_probs(att, causal, self.mode)
            y = P @ vf

            if self.collect_stats:
                self._record_common_stats(qf, kf, att, causal)

        return y.to(q.dtype)

    def _external_operator(self, q, k, v):
        B, H, T, D = q.shape
        with torch.autocast(device_type=q.device.type, enabled=False):
            qf, kf, vf = q.float(), k.float(), v.float()
            att = (qf @ kf.transpose(-2, -1)) / math.sqrt(D)
            causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            P = external_operator_probs(att, causal, self.mode, self.variant, self.exaq_bits)
            y = P @ vf

            if self.collect_stats:
                self._record_common_stats(qf, kf, att, causal)

        return y.to(q.dtype)

    @torch.no_grad()
    def _ema_update(self, att, causal):
        """E4 observer: per-head mean of M - m over rows i >= 1 of this forward; updated only in
        training forwards with autograd enabled (never during evaluation)."""
        T = att.shape[-1]
        valid = causal.view(1, 1, T, T)
        span = (att.masked_fill(~valid, -torch.inf).amax(dim=-1)
                - att.masked_fill(~valid, torch.inf).amin(dim=-1))      # (B,H,T)
        cur = span[:, :, 1:].mean(dim=(0, 2)).to(self.ema_span.dtype)   # (H,)
        if torch.isnan(self.ema_span).any():
            self.ema_span.copy_(cur)
        else:
            self.ema_span.mul_(self.ema_decay).add_(cur, alpha=1.0 - self.ema_decay)

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
                if self.approx == "nearest":
                    # Strict hard forward (e2_minmax_probste_strict, verbatim):
                    # P's value is hard Nearest; only the score->probability
                    # Jacobian is replaced.
                    with torch.no_grad():
                        w_hard, bins, row_span, level_idx = _adaptive_interval_reconstruct(
                            att, causal, self.K, "nearest", return_level_idx=True,
                            range_grad=self.range_grad, nearest_forward="direct"
                        )
                        P_hard = w_hard / w_hard.sum(dim=-1, keepdim=True)

                    w_lerp, _, _, _ = _adaptive_interval_reconstruct(
                        att, causal, self.K, "lerp", return_level_idx=True,
                        range_grad=self.range_grad
                    )

                    if NEAREST_SURROGATE == "prob":
                        P_lerp = w_lerp / w_lerp.sum(dim=-1, keepdim=True)
                        # Strict hard forward: the value is P_hard itself, the
                        # gradient is the normalized-LERP probability Jacobian.
                        P = P_hard.detach() + (P_lerp - P_lerp.detach())
                    elif NEAREST_SURROGATE == "weight_strict":
                        w = w_hard.detach() + (w_lerp - w_lerp.detach())
                        P = w / w.sum(dim=-1, keepdim=True)
                    else:
                        raise ValueError(
                            f"unknown NEAREST_SURROGATE: {NEAREST_SURROGATE}")
                    w = w_hard

                else:
                    w, bins, row_span, level_idx = _adaptive_interval_reconstruct(
                        att, causal, self.K, self.approx, return_level_idx=True,
                        range_grad=self.range_grad
                    )
                    denom = w.sum(dim=-1, keepdim=True)
                    P = w / denom

            elif self.range_mode == "relevant_zero":
                if self.approx == "nearest":
                    # QRM probability-level STE causal control.
                    #
                    # Forward:
                    #   identical hard QRM-Nearest probabilities.
                    #
                    # Backward w.r.t. scores:
                    #   full normalized QRM-LERP probability Jacobian.
                    #
                    # The QRM active window [-tau, 0] and hard zero-tail
                    # remain unchanged.  P's forward value is hard Nearest,
                    # so the V-gradient through y=P@V also remains the
                    # hard-forward gradient.
                    with torch.no_grad():
                        (
                            w_hard,
                            bins,
                            row_span,
                            level_idx,
                            tail_mask,
                        ) = _relevant_zero_reconstruct(
                            att,
                            causal,
                            self.K,
                            "nearest",
                            tau=self.range_tau,
                            return_level_idx=True,
                            range_grad=self.range_grad,
                            nearest_forward="direct",
                            tail_policy=self.tail_policy,
                        )

                        P_hard = (
                            w_hard
                            / w_hard.sum(dim=-1, keepdim=True)
                        )

                    (
                        w_lerp,
                        _,
                        _,
                        _,
                        _,
                    ) = _relevant_zero_reconstruct(
                        att,
                        causal,
                        self.K,
                        "lerp",
                        tau=self.range_tau,
                        return_level_idx=True,
                        range_grad=self.range_grad,
                        tail_policy=self.tail_policy,
                    )

                    if NEAREST_SURROGATE == "prob":
                        P_lerp = (
                            w_lerp
                            / w_lerp.sum(dim=-1, keepdim=True)
                        )

                        # Strict hard forward: the value is P_hard itself,
                        # the gradient is the normalized-LERP Jacobian.
                        P = (
                            P_hard.detach()
                            + (P_lerp - P_lerp.detach())
                        )
                    elif NEAREST_SURROGATE == "weight_strict":
                        w_ref = (
                            w_hard.detach()
                            + (w_lerp - w_lerp.detach())
                        )
                        P = w_ref / w_ref.sum(dim=-1, keepdim=True)
                    else:
                        raise ValueError(
                            f"unknown NEAREST_SURROGATE: {NEAREST_SURROGATE}")

                    # Retain hard-forward weights for any downstream
                    # bookkeeping in this module.
                    w = w_hard

                else:
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

                    denom = w.sum(dim=-1, keepdim=True)
                    P = w / denom

            elif self.range_mode == "ema":
                # rb-root_v1, E4: semi-static window from the per-head EMA buffer.
                if self.range_grad != "full":
                    raise ValueError("range_mode=ema keeps the row-max gradient: use range_grad=full")
                if self.training and torch.is_grad_enabled():
                    self._ema_update(att.detach(), causal)
                if torch.isnan(self.ema_span).any():
                    raise RuntimeError("EMA window used before its first training forward")
                tau_t = self.ema_span.view(1, H, 1, 1)
                if self.approx != "nearest":
                    raise ValueError("range_mode=ema is defined for Nearest only")
                P, w_hard, bins, row_span, level_idx, tail_mask = ema_window_probs(
                    att, causal, self.K, tau_t, NEAREST_SURROGATE)
                w = w_hard

            else:
                raise RuntimeError(
                    f"unknown range_mode: {self.range_mode}"
                )

            y = P @ vf

            if self.collect_stats:
                with torch.no_grad():
                    self._record_common_stats(qf, kf, att, causal)
                    valid = causal.view(1, 1, T, T).expand(B, H, T, T)

                    stat_valid = valid

                    if tail_mask is not None and self.tail_policy == "zero":
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
        elif self.mode in SOFTMAX_FAMILY:
            y = self._softmax_family(q, k, v)
        elif self.mode in ("indexsoftmax", "exaq"):
            y = self._external_operator(q, k, v)
        elif self.mode == "quant":
            y = self._quant_attention(q, k, v)
        else:
            raise ValueError(f"unknown attention mode: {self.mode}")

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
