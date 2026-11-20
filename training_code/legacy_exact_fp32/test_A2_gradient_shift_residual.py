"""Phase A2: gradient row-shift residual test.

Purpose
-------
A1 verified the exact FORWARD symmetry

    F(S + c_row * 1) = F(S)

for the current attention operators.  If the backward is consistent with that
symmetry, then for every causal query row i the score gradient must satisfy

    sum_{j <= i} dL/dS_ij = 0.

This script measures the normalized residual

    rho_i = |sum_j g_ij| / (sum_j |g_ij| + eps)

on non-degenerate causal rows.

It compares:
  1) Exact causal softmax.
  2) The CURRENT project LERP implementation (detached row min/max).
  3) A local diagnostic-only LERP reference with gradients allowed through
     row min/max ("full-range-gradient").  This does NOT modify training code.
  4) Current Nearest as a supplemental measurement; its backward is the LERP
     surrogate and therefore is not expected to be the exact derivative of its
     hard forward.

Run from the project directory:

    python test_A2_gradient_shift_residual.py

The script expects quant_attention.py to be importable from the current project
folder.  It does not edit any project file and does not start training.
"""

import math
import sys
from dataclasses import dataclass

import torch

from quant_attention import _adaptive_interval_reconstruct


EPS = 1e-30
ACTIVE_GRAD_EPS = 1e-12


@dataclass
class ResidualSummary:
    mean: float
    median: float
    p95: float
    max: float
    active_rows: int
    max_abs_row_sum: float
    mean_abs_row_sum: float
    mean_l1: float


def causal_mask(T: int, device):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def exact_probabilities(att: torch.Tensor, causal: torch.Tensor):
    T = att.shape[-1]
    valid = causal.view(1, 1, T, T)
    masked = att.masked_fill(~valid, -torch.inf)
    row_max = masked.amax(dim=-1, keepdim=True)
    z = masked - row_max
    w = torch.exp(z).masked_fill(~valid, 0.0)
    return w / w.sum(dim=-1, keepdim=True)


def adaptive_lerp_full_range_grad(att: torch.Tensor,
                                  causal: torch.Tensor,
                                  K: int,
                                  eps: float = 1e-12):
    """Diagnostic-only LERP with gradient through row min/max.

    Forward mathematics intentionally matches the project's current LERP.
    The only difference is that row_min and row_max are NOT detached.

    Bin membership is still piecewise constant (floor -> integer index), so this
    represents the almost-everywhere derivative of the adaptive LERP operator
    away from interval boundaries.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")

    B, H, T, _ = att.shape
    valid = causal.view(1, 1, T, T)

    row_max = att.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
    row_min = att.masked_fill(~valid, torch.inf).amin(dim=-1, keepdim=True)
    row_span = row_max - row_min

    z = att - row_max
    z_min = row_min - row_max
    h = row_span / float(K)
    degenerate = row_span <= eps
    h_safe = torch.where(degenerate, torch.ones_like(h), h)

    rel = (z - z_min) / h_safe
    bin_idx = torch.floor(rel).to(torch.long).clamp(0, K - 1)
    bin_idx = torch.where(valid, bin_idx, torch.full_like(bin_idx, -1))

    r_idx = bin_idx.clamp_min(0)
    r = r_idx.to(att.dtype)
    b_lo = z_min + r * h_safe

    grid_r = torch.arange(K + 1, device=att.device, dtype=att.dtype)
    grid_r = grid_r.view(1, 1, 1, K + 1)
    boundary_z = z_min + grid_r * h_safe
    boundary_w = torch.exp(boundary_z)

    e_lo = torch.gather(boundary_w, -1, r_idx)
    e_hi = torch.gather(boundary_w, -1, (r_idx + 1).clamp_max(K))

    t = ((z - b_lo) / h_safe).clamp(0.0, 1.0)
    w = (1.0 - t) * e_lo + t * e_hi
    w = torch.where(degenerate, torch.ones_like(w), w)
    w = w.masked_fill(~valid, 0.0)
    return w


def attention_output_from_weights(w: torch.Tensor, v: torch.Tensor):
    P = w / w.sum(dim=-1, keepdim=True)
    return P @ v


def score_gradient(method: str,
                   base_scores: torch.Tensor,
                   causal: torch.Tensor,
                   v: torch.Tensor,
                   probe: torch.Tensor,
                   K: int | None = None):
    """Return dL/dS for a fixed random scalar probe loss."""
    att = base_scores.detach().clone().requires_grad_(True)

    if method == "exact":
        P = exact_probabilities(att, causal)
        y = P @ v
    elif method == "lerp_detach":
        w, _, _ = _adaptive_interval_reconstruct(att, causal, K=K, approx="lerp")
        y = attention_output_from_weights(w, v)
    elif method == "lerp_fullgrad":
        w = adaptive_lerp_full_range_grad(att, causal, K=K)
        y = attention_output_from_weights(w, v)
    elif method == "nearest":
        w, _, _ = _adaptive_interval_reconstruct(att, causal, K=K, approx="nearest")
        y = attention_output_from_weights(w, v)
    else:
        raise ValueError(method)

    # A generic upstream direction avoids special cancellations from losses such
    # as sum(P), whose derivative is identically zero.
    loss = (y * probe).sum() / y.numel()
    (g,) = torch.autograd.grad(loss, att, create_graph=False)
    return g.detach()


def summarize_residual(g: torch.Tensor, causal: torch.Tensor) -> ResidualSummary:
    B, H, T, _ = g.shape
    valid = causal.view(1, 1, T, T)
    gv = g.masked_fill(~valid, 0.0)

    row_sum = gv.sum(dim=-1)
    row_l1 = gv.abs().sum(dim=-1)

    # Exclude rows with no meaningful gradient (notably the first causal row,
    # whose attention probability is exactly 1 and therefore has zero score
    # gradient).  rho would otherwise be 0/eps and provide no information.
    active = row_l1 > ACTIVE_GRAD_EPS
    if not bool(active.any()):
        raise RuntimeError("No active gradient rows found")

    rho = row_sum.abs()[active] / (row_l1[active] + EPS)
    abs_sum = row_sum.abs()[active]
    l1 = row_l1[active]

    return ResidualSummary(
        mean=float(rho.mean().item()),
        median=float(rho.median().item()),
        p95=float(torch.quantile(rho, 0.95).item()),
        max=float(rho.max().item()),
        active_rows=int(active.sum().item()),
        max_abs_row_sum=float(abs_sum.max().item()),
        mean_abs_row_sum=float(abs_sum.mean().item()),
        mean_l1=float(l1.mean().item()),
    )


def print_summary(label: str, s: ResidualSummary):
    print(
        f"  {label:<19s} | rho mean={s.mean:.3e} median={s.median:.3e} "
        f"p95={s.p95:.3e} max={s.max:.3e} | "
        f"|sum g| mean={s.mean_abs_row_sum:.3e} max={s.max_abs_row_sum:.3e} "
        f"| grad-L1 mean={s.mean_l1:.3e} | rows={s.active_rows}"
    )


def run_case(device: str, score_scale: float, seed: int = 20260824):
    torch.manual_seed(seed)

    # T>32 gives enough causal rows and avoids making min/max behavior depend on
    # a tiny number of entries.  Scores remain fp32 throughout.
    B, H, T, Dv = 2, 3, 41, 13
    causal = causal_mask(T, device)
    scores = torch.randn(B, H, T, T, device=device, dtype=torch.float32) * score_scale
    v = torch.randn(B, H, T, Dv, device=device, dtype=torch.float32)
    probe = torch.randn(B, H, T, Dv, device=device, dtype=torch.float32)

    print(f"--- score scale = {score_scale:g} ---")

    g_exact = score_gradient("exact", scores, causal, v, probe)
    exact = summarize_residual(g_exact, causal)
    print_summary("Exact", exact)

    outputs = {"Exact": exact}

    for K in [2, 4, 8, 16, 32]:
        g_det = score_gradient("lerp_detach", scores, causal, v, probe, K=K)
        det = summarize_residual(g_det, causal)
        print_summary(f"K={K} LERP-detach", det)

        g_full = score_gradient("lerp_fullgrad", scores, causal, v, probe, K=K)
        full = summarize_residual(g_full, causal)
        print_summary(f"K={K} LERP-full", full)

        # Supplemental: current hard Nearest / LERP-surrogate backward.
        g_near = score_gradient("nearest", scores, causal, v, probe, K=K)
        near = summarize_residual(g_near, causal)
        print_summary(f"K={K} Nearest", near)

        outputs[f"K{K}_detach"] = det
        outputs[f"K{K}_full"] = full
        outputs[f"K{K}_nearest"] = near

    return outputs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 118)
    print("A2 — Gradient row-shift residual")
    print("=" * 118)
    print(f"torch={torch.__version__} | device={device} | dtype=float32")
    print("Forward symmetry from A1 implies the exact backward identity:")
    print("    sum_{j<=i} dL/dS_ij = 0")
    print("Metric:")
    print("    rho = |sum_j g_j| / (sum_j |g_j| + eps)")
    print("Smaller is better; exact/full-gradient should be near fp32 numerical noise.\n")

    all_results = []
    # Probe several score scales.  scale=1 approximates early/moderate logits;
    # larger scales test whether the inconsistency worsens as attention sharpens.
    for scale in [1.0, 3.0, 8.0]:
        all_results.append((scale, run_case(device, scale)))
        print()

    # Automated diagnostic, deliberately based on orders of magnitude rather
    # than one brittle absolute number.
    exact_max = max(r["Exact"].max for _, r in all_results)
    full_max = max(
        r[f"K{K}_full"].max
        for _, r in all_results
        for K in [2, 4, 8, 16, 32]
    )
    detach_medians = [
        r[f"K{K}_detach"].median
        for _, r in all_results
        for K in [2, 4, 8, 16, 32]
    ]
    detach_peak = max(
        r[f"K{K}_detach"].max
        for _, r in all_results
        for K in [2, 4, 8, 16, 32]
    )

    print("=" * 118)
    print("A2 summary")
    print("=" * 118)
    print(f"worst Exact rho_max      : {exact_max:.3e}")
    print(f"worst LERP-full rho_max  : {full_max:.3e}")
    print(f"median detached rho range: [{min(detach_medians):.3e}, {max(detach_medians):.3e}]")
    print(f"worst LERP-detach rho_max: {detach_peak:.3e}")

    # In fp32, exact/full residuals can be around ~1e-6 to ~1e-5 depending on
    # the random case.  What matters here is the qualitative separation from
    # detached calibration, typically many orders larger.
    if exact_max < 1e-4 and full_max < 1e-4 and detach_peak > 1e-3:
        print("\nPASS / POSITIVE DIAGNOSIS:")
        print("  - Exact softmax backward respects row-shift symmetry.")
        print("  - Diagnostic LERP with gradient through min/max also respects it.")
        print("  - Current detached-range LERP violates the required gradient identity.")
        print("\nInterpretation:")
        print("  A1 showed the forward is shift-invariant, while A2 shows the current")
        print("  detached calibration can produce a spurious common-mode score gradient.")
        print("  Proceed to A3 (finite-difference directional derivative) before changing")
        print("  the training implementation.")
    else:
        print("\nREVIEW REQUIRED:")
        print("  The expected clean separation was not observed under the automatic")
        print("  thresholds. Inspect the table before drawing a conclusion.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        raise
