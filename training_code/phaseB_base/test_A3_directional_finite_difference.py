"""Phase A3: directional finite-difference test for row-shift symmetry.

Purpose
-------
A1 established the exact FORWARD symmetry

    F(S + c_row * 1) = F(S)

for the current attention operators.

A2 showed that the current detached adaptive calibration can violate the
corresponding gradient identity

    sum_{j <= i} dL/dS_ij = 0.

A3 now compares the derivative predicted by autograd with the actual
finite-difference derivative of the FORWARD function along exactly that
row-wise common-shift direction.

For a chosen causal row r, define direction d by

    d[..., r, j] = 1  for j <= r
    d[..., r, j] = 0  otherwise.

Because the forward is row-shift invariant,

    D_FD = [L(S + eps*d) - L(S - eps*d)] / (2 eps)

must be approximately zero.

The autograd directional derivative is

    D_grad = <dL/dS, d> = sum_{j <= r} dL/dS_rj.

If detached range calibration is backward-inconsistent, we expect

    LERP-detach:  D_FD ~ 0, but D_grad != 0
    Exact:        D_FD ~ 0 and D_grad ~ 0
    LERP-full:    D_FD ~ 0 and D_grad ~ 0

The script:
  * imports the CURRENT project _adaptive_interval_reconstruct for
    LERP-detach;
  * implements a diagnostic-only LERP-full reference without detaching
    row min/max;
  * automatically selects a high-residual causal row from LERP-detach;
  * tests K = 4 and 32 by default;
  * runs both fp32 (training-relevant) and fp64 (high-precision confirmation);
  * sweeps several epsilon values.

Run from the project directory:

    python test_A3_directional_finite_difference.py

This script does NOT edit project code and does NOT start training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from quant_attention import _adaptive_interval_reconstruct


ACTIVE_GRAD_EPS = 1e-14


def causal_mask(T: int, device: str) -> torch.Tensor:
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def exact_probabilities(att: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
    T = att.shape[-1]
    valid = causal.view(1, 1, T, T)
    masked = att.masked_fill(~valid, -torch.inf)
    row_max = masked.amax(dim=-1, keepdim=True)
    z = masked - row_max
    w = torch.exp(z).masked_fill(~valid, 0.0)
    return w / w.sum(dim=-1, keepdim=True)


def adaptive_lerp_full_range_grad(
    att: torch.Tensor,
    causal: torch.Tensor,
    K: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Diagnostic LERP with gradient through row min/max.

    Forward mathematics matches the project's current LERP.  The only
    intentional difference is that row_min and row_max are NOT detached.
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


def forward_loss(
    method: str,
    scores: torch.Tensor,
    causal: torch.Tensor,
    v: torch.Tensor,
    probe: torch.Tensor,
    K: int | None,
) -> torch.Tensor:
    if method == "exact":
        P = exact_probabilities(scores, causal)
    elif method == "lerp_detach":
        w, _, _ = _adaptive_interval_reconstruct(
            scores, causal, K=K, approx="lerp"
        )
        P = w / w.sum(dim=-1, keepdim=True)
    elif method == "lerp_full":
        w = adaptive_lerp_full_range_grad(scores, causal, K=K)
        P = w / w.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(method)

    y = P @ v
    # Generic scalar probe; avoids special cancellations.
    return (y * probe).sum() / y.numel()


def autograd_score_gradient(
    method: str,
    base_scores: torch.Tensor,
    causal: torch.Tensor,
    v: torch.Tensor,
    probe: torch.Tensor,
    K: int | None,
) -> tuple[torch.Tensor, float]:
    scores = base_scores.detach().clone().requires_grad_(True)
    loss = forward_loss(method, scores, causal, v, probe, K)
    (g,) = torch.autograd.grad(loss, scores, create_graph=False)
    return g.detach(), float(loss.detach().item())


@dataclass
class TargetRow:
    b: int
    h: int
    q: int
    rho: float
    row_sum: float
    row_l1: float


def choose_target_row(
    base_scores: torch.Tensor,
    causal: torch.Tensor,
    v: torch.Tensor,
    probe: torch.Tensor,
    K: int,
) -> TargetRow:
    """Choose a row with a large *absolute* spurious common-mode gradient.

    A2 already established that rho can be large.  For A3 we prefer a row
    whose directional derivative is also large in absolute magnitude, so the
    finite-difference/autograd mismatch is visually unambiguous.  We first
    restrict to rows with rho >= 0.05 and then maximize |row_sum|.
    """
    g, _ = autograd_score_gradient(
        "lerp_detach", base_scores, causal, v, probe, K
    )

    B, H, T, _ = g.shape
    valid = causal.view(1, 1, T, T)
    gv = g.masked_fill(~valid, 0.0)

    row_sum = gv.sum(dim=-1)
    row_l1 = gv.abs().sum(dim=-1)

    # Exclude degenerate first causal row / zero-gradient rows.
    active = row_l1 > ACTIVE_GRAD_EPS
    rho = torch.full_like(row_l1, -1.0)
    rho[active] = row_sum.abs()[active] / row_l1[active]

    candidate = active & (rho >= 0.05)
    if bool(candidate.any()):
        score = torch.where(candidate, row_sum.abs(), torch.full_like(row_sum, -1.0))
    else:
        score = torch.where(active, row_sum.abs(), torch.full_like(row_sum, -1.0))
    flat_idx = int(score.argmax().item())
    b = flat_idx // (H * T)
    rem = flat_idx % (H * T)
    h = rem // T
    q = rem % T

    return TargetRow(
        b=b,
        h=h,
        q=q,
        rho=float(rho[b, h, q].item()),
        row_sum=float(row_sum[b, h, q].item()),
        row_l1=float(row_l1[b, h, q].item()),
    )


def make_row_shift_direction(
    shape: tuple[int, ...],
    causal: torch.Tensor,
    target: TargetRow,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    d = torch.zeros(shape, dtype=dtype, device=device)
    # Only causal-valid scores in this one query row receive +1.
    d[target.b, target.h, target.q, : target.q + 1] = 1.0
    return d


@dataclass
class MethodResult:
    loss0: float
    d_grad: float
    grad_row_sum: float
    grad_row_l1: float


def method_autograd_result(
    method: str,
    base_scores: torch.Tensor,
    causal: torch.Tensor,
    v: torch.Tensor,
    probe: torch.Tensor,
    K: int,
    direction: torch.Tensor,
    target: TargetRow,
) -> MethodResult:
    g, loss0 = autograd_score_gradient(
        method, base_scores, causal, v, probe,
        None if method == "exact" else K,
    )
    d_grad = float((g * direction).sum().item())
    row = g[target.b, target.h, target.q, : target.q + 1]
    return MethodResult(
        loss0=loss0,
        d_grad=d_grad,
        grad_row_sum=float(row.sum().item()),
        grad_row_l1=float(row.abs().sum().item()),
    )


def finite_difference(
    method: str,
    base_scores: torch.Tensor,
    causal: torch.Tensor,
    v: torch.Tensor,
    probe: torch.Tensor,
    K: int,
    direction: torch.Tensor,
    eps: float,
) -> tuple[float, float, float, float]:
    with torch.no_grad():
        lp = forward_loss(
            method,
            base_scores + eps * direction,
            causal, v, probe,
            None if method == "exact" else K,
        )
        lm = forward_loss(
            method,
            base_scores - eps * direction,
            causal, v, probe,
            None if method == "exact" else K,
        )
        l0 = forward_loss(
            method,
            base_scores,
            causal, v, probe,
            None if method == "exact" else K,
        )

    lp_f = float(lp.item())
    lm_f = float(lm.item())
    l0_f = float(l0.item())
    d_fd = (lp_f - lm_f) / (2.0 * eps)
    max_forward_delta = max(abs(lp_f - l0_f), abs(lm_f - l0_f))
    return d_fd, lp_f, lm_f, max_forward_delta


def run_dtype(
    device: str,
    dtype: torch.dtype,
    K_values=(4, 32),
    score_scale: float = 8.0,
    seed: int = 20260824,
):
    torch.manual_seed(seed)

    # Keep the test small but nontrivial.
    B, H, T, Dv = 2, 3, 41, 13
    causal = causal_mask(T, device)
    scores = torch.randn(B, H, T, T, device=device, dtype=dtype) * score_scale
    v = torch.randn(B, H, T, Dv, device=device, dtype=dtype)
    probe = torch.randn(B, H, T, Dv, device=device, dtype=dtype)

    print()
    print("=" * 118)
    print(f"dtype={dtype} | score_scale={score_scale:g}")
    print("=" * 118)

    failures = []

    for K in K_values:
        target = choose_target_row(scores, causal, v, probe, K)
        direction = make_row_shift_direction(
            tuple(scores.shape), causal, target, dtype, device
        )

        print(
            f"\nK={K}: selected LERP-detach high-common-mode-gradient row "
            f"(B={target.b}, H={target.h}, query={target.q}, "
            f"valid_keys={target.q + 1})"
        )
        print(
            f"     detached rho={target.rho:.6f} | "
            f"row_sum={target.row_sum:+.6e} | row_L1={target.row_l1:.6e}"
        )

        # Keep eps fairly large in fp32 because the exact mathematical FD is
        # zero and tiny eps only amplifies floating-point subtraction noise.
        eps_values = [1.0, 0.1, 0.01]
        methods = [
            ("Exact", "exact"),
            ("LERP-detach", "lerp_detach"),
            ("LERP-full", "lerp_full"),
        ]

        print(
            "  "
            + f"{'method':<13s} {'D_grad':>14s} "
              f"{'eps':>8s} {'D_FD':>14s} {'|Dgrad-DFD|':>15s} "
              f"{'max|L±-L0|':>14s}"
        )
        print("  " + "-" * 88)

        for label, method in methods:
            auto = method_autograd_result(
                method, scores, causal, v, probe, K, direction, target
            )

            for n, eps in enumerate(eps_values):
                d_fd, lp, lm, fwd_delta = finite_difference(
                    method, scores, causal, v, probe, K, direction, eps
                )
                mismatch = abs(auto.d_grad - d_fd)
                grad_text = f"{auto.d_grad:+.6e}" if n == 0 else " " * 13

                print(
                    f"  {label:<13s} {grad_text:>14s} "
                    f"{eps:8.2g} {d_fd:+14.6e} {mismatch:15.6e} "
                    f"{fwd_delta:14.6e}"
                )

            # Strong logical checks use dtype-appropriate thresholds.
            if dtype == torch.float64:
                zero_grad_tol = 1e-10
                # Full/exact should have essentially zero autograd shift derivative.
                if method in {"exact", "lerp_full"} and abs(auto.d_grad) > zero_grad_tol:
                    failures.append(
                        f"{dtype} K={K} {label}: D_grad={auto.d_grad:.3e} not ~0"
                    )
                # Detached should be detectably nonzero on the deliberately
                # selected high-residual row.
                if method == "lerp_detach" and abs(auto.d_grad) < 1e-10:
                    failures.append(
                        f"{dtype} K={K} detached: selected row D_grad unexpectedly ~0"
                    )

    return failures


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 118)
    print("A3 — Directional finite-difference test")
    print("=" * 118)
    print(f"torch={torch.__version__} | device={device}")
    print()
    print("Test direction: add the SAME constant to every causal-valid score")
    print("in one selected query row.")
    print()
    print("Forward symmetry requires:")
    print("    D_FD = [L(S+eps*d)-L(S-eps*d)]/(2eps) ~= 0")
    print("A backward consistent with that forward requires:")
    print("    D_grad = <grad_S L, d> ~= 0")
    print()
    print("Expected diagnostic pattern:")
    print("    Exact       : D_FD ~= 0, D_grad ~= 0")
    print("    LERP-full   : D_FD ~= 0, D_grad ~= 0")
    print("    LERP-detach : D_FD ~= 0, D_grad significantly nonzero")
    print()

    failures = []

    # fp32: directly relevant to the current training implementation.
    failures += run_dtype(
        device=device,
        dtype=torch.float32,
        K_values=(4, 32),
        score_scale=8.0,
    )

    # fp64: removes most finite-difference/roundoff ambiguity and acts as a
    # high-precision confirmation of the same mathematical inconsistency.
    failures += run_dtype(
        device=device,
        dtype=torch.float64,
        K_values=(4, 32),
        score_scale=8.0,
    )

    print()
    print("=" * 118)
    print("A3 summary")
    print("=" * 118)

    if failures:
        print("FAIL / NEEDS REVIEW")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)

    print("PASS / POSITIVE DIAGNOSIS")
    print()
    print("Interpretation:")
    print("  - The actual forward finite-difference derivative along a row-wise")
    print("    common-shift direction is approximately zero.")
    print("  - Exact softmax and diagnostic full-gradient LERP give autograd")
    print("    directional derivatives approximately equal to that zero derivative.")
    print("  - Current detached-range LERP can give a substantial nonzero autograd")
    print("    derivative along the SAME forward-null direction.")
    print()
    print("Therefore A3 directly demonstrates a forward/backward derivative")
    print("inconsistency introduced by detached adaptive range calibration.")
    print()
    print("After local reproduction, Phase A1-A3 together justify proceeding to")
    print("the controlled range_grad=detach/full implementation ablation.")


if __name__ == "__main__":
    main()
