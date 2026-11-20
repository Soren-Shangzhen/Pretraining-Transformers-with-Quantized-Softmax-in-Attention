"""Phase A4: ordinary local finite-difference gradient check.

A4 validates gradients on ordinary, locally smooth score coordinates:
  * causal-valid;
  * NOT the row argmin or argmax;
  * comfortably away from K-interval boundaries.

For such a coordinate, an infinitesimal perturbation does not change row
min/max or bin membership. Therefore the forward is locally smooth and an
ordinary central finite-difference derivative is a clean reference.

We compare:
  1) Exact softmax autograd vs exact finite difference;
  2) current LERP-detach autograd vs LERP finite difference;
  3) diagnostic LERP-full autograd vs the SAME LERP finite difference.

Important interpretation:
  On NON-extremum, boundary-safe coordinates, LERP-detach is expected to pass
  too, because row_min/row_max have zero local derivative with respect to an
  ordinary score. A4 is therefore not intended to rediscover the detach bug.
  A2/A3 already localized that bug to adaptive calibration/common-mode
  directions. A4 checks that the ordinary local LERP derivative itself is
  correct and that enabling full range gradients does not corrupt it.

Run from the project directory:
    python test_A4_local_finite_difference.py

This script does not modify project code or start training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from quant_attention import _adaptive_interval_reconstruct


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
    """Same LERP forward as project code, but min/max remain in autograd."""
    if K < 2:
        raise ValueError(K)

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
    return w.masked_fill(~valid, 0.0)


def forward_loss(
    method: str,
    scores: torch.Tensor,
    causal: torch.Tensor,
    v: torch.Tensor,
    probe: torch.Tensor,
    K: int,
) -> torch.Tensor:
    if method == "exact":
        P = exact_probabilities(scores, causal)
    elif method == "lerp_detach":
        w, _, _ = _adaptive_interval_reconstruct(scores, causal, K=K, approx="lerp")
        P = w / w.sum(dim=-1, keepdim=True)
    elif method == "lerp_full":
        w = adaptive_lerp_full_range_grad(scores, causal, K=K)
        P = w / w.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(method)

    y = P @ v
    return (y * probe).sum() / y.numel()


def gradient(
    method: str,
    scores0: torch.Tensor,
    causal: torch.Tensor,
    v: torch.Tensor,
    probe: torch.Tensor,
    K: int,
) -> torch.Tensor:
    s = scores0.detach().clone().requires_grad_(True)
    loss = forward_loss(method, s, causal, v, probe, K)
    (g,) = torch.autograd.grad(loss, s)
    return g.detach()


@dataclass
class Point:
    b: int
    h: int
    q: int
    k: int
    t: float
    boundary_margin: float
    value: float


def candidate_points(
    scores: torch.Tensor,
    causal: torch.Tensor,
    K: int,
    max_points: int = 8,
    t_margin: float = 0.20,
) -> list[Point]:
    """Select non-extremum points safely inside their current interval."""
    B, H, T, _ = scores.shape
    valid = causal.view(1, 1, T, T).expand(B, H, T, T)

    masked_hi = scores.masked_fill(~valid, -torch.inf)
    masked_lo = scores.masked_fill(~valid, torch.inf)
    row_max, argmax = masked_hi.max(dim=-1, keepdim=True)
    row_min, argmin = masked_lo.min(dim=-1, keepdim=True)
    span = row_max - row_min
    hwidth = span / float(K)

    z = scores - row_max
    z_min = row_min - row_max
    rel = (z - z_min) / hwidth
    bins = torch.floor(rel).to(torch.long).clamp(0, K - 1)
    b_lo = z_min + bins.to(scores.dtype) * hwidth
    t = (z - b_lo) / hwidth

    points = []
    for b in range(B):
        for hd in range(H):
            for q in range(2, T):  # avoid tiny/degenerate causal rows
                amin = int(argmin[b, hd, q, 0].item())
                amax = int(argmax[b, hd, q, 0].item())
                hw = float(hwidth[b, hd, q, 0].item())
                if not math.isfinite(hw) or hw <= 0:
                    continue
                for kk in range(q + 1):
                    if kk == amin or kk == amax:
                        continue
                    tt = float(t[b, hd, q, kk].item())
                    if not (t_margin < tt < 1.0 - t_margin):
                        continue
                    margin = min(tt, 1.0 - tt) * hw
                    points.append(
                        Point(
                            b=b, h=hd, q=q, k=kk,
                            t=tt,
                            boundary_margin=margin,
                            value=float(scores[b, hd, q, kk].item()),
                        )
                    )

    # Prefer points with the largest absolute score-gradient under full LERP,
    # but selection itself is deterministic and boundary-safe.  Here we first
    # spread across rows by taking every Nth candidate after sorting by margin.
    points.sort(key=lambda p: p.boundary_margin, reverse=True)
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    picked = points[::step][:max_points]
    return picked


def fd_one_coordinate(
    method: str,
    scores0: torch.Tensor,
    causal: torch.Tensor,
    v: torch.Tensor,
    probe: torch.Tensor,
    K: int,
    p: Point,
    eps: float,
) -> float:
    plus = scores0.detach().clone()
    minus = scores0.detach().clone()
    plus[p.b, p.h, p.q, p.k] += eps
    minus[p.b, p.h, p.q, p.k] -= eps
    with torch.no_grad():
        lp = forward_loss(method, plus, causal, v, probe, K)
        lm = forward_loss(method, minus, causal, v, probe, K)
    return float(((lp - lm) / (2.0 * eps)).item())


def choose_eps(dtype: torch.dtype, p: Point) -> float:
    # Epsilon must be far smaller than the nearest interval boundary margin.
    if dtype == torch.float64:
        base = 1e-5 * max(1.0, abs(p.value))
        return max(1e-7, min(base, p.boundary_margin * 0.02))
    else:
        base = 2e-3 * max(1.0, abs(p.value))
        return max(2e-4, min(base, p.boundary_margin * 0.02))


def pass_check(dtype: torch.dtype, g: float, fd: float) -> tuple[bool, float, float]:
    abs_err = abs(g - fd)
    scale = max(abs(g), abs(fd), 1e-12)
    rel_err = abs_err / scale
    if dtype == torch.float64:
        ok = abs_err <= 2e-9 + 5e-4 * max(abs(g), abs(fd))
    else:
        ok = abs_err <= 2e-6 + 5e-2 * max(abs(g), abs(fd))
    return ok, abs_err, rel_err


def run_case(device: str, dtype: torch.dtype, K: int, score_scale: float, seed: int):
    torch.manual_seed(seed)
    B, H, T, Dv = 1, 3, 37, 11
    causal = causal_mask(T, device)
    scores = torch.randn(B, H, T, T, device=device, dtype=dtype) * score_scale
    v = torch.randn(B, H, T, Dv, device=device, dtype=dtype)
    probe = torch.randn(B, H, T, Dv, device=device, dtype=dtype)

    points = candidate_points(scores, causal, K, max_points=8)
    if len(points) < 4:
        raise RuntimeError(f"Too few safe points for K={K}, dtype={dtype}: {len(points)}")

    grads = {
        m: gradient(m, scores, causal, v, probe, K)
        for m in ("exact", "lerp_detach", "lerp_full")
    }

    print("\n" + "=" * 132)
    print(f"dtype={dtype} | K={K} | score_scale={score_scale:g} | safe_points={len(points)}")
    print("=" * 132)
    print(
        f"{'pt':>2s} {'coord (b,h,q,k)':>15s} {'t':>7s} {'margin':>10s} {'eps':>10s} "
        f"{'method':<12s} {'autograd':>14s} {'finite-diff':>14s} {'abs err':>12s} {'rel err':>10s} {'result':>7s}"
    )
    print("-" * 132)

    failures = []
    max_abs = {m: 0.0 for m in grads}
    max_rel = {m: 0.0 for m in grads}

    for i, p in enumerate(points):
        eps = choose_eps(dtype, p)
        # By construction, eps is < 2% of the closest boundary margin.
        assert eps < p.boundary_margin * 0.05 + 1e-15

        for method in ("exact", "lerp_detach", "lerp_full"):
            g = float(grads[method][p.b, p.h, p.q, p.k].item())
            fd = fd_one_coordinate(method, scores, causal, v, probe, K, p, eps)
            ok, ae, re = pass_check(dtype, g, fd)
            max_abs[method] = max(max_abs[method], ae)
            max_rel[method] = max(max_rel[method], re if math.isfinite(re) else 0.0)
            if not ok:
                failures.append((i, p, method, g, fd, ae, re))

            print(
                f"{i:2d} {(p.b,p.h,p.q,p.k)!s:>15s} {p.t:7.3f} {p.boundary_margin:10.3e} {eps:10.3e} "
                f"{method:<12s} {g:+14.6e} {fd:+14.6e} {ae:12.3e} {re:10.3e} {'PASS' if ok else 'FAIL':>7s}"
            )

    print("\nmax errors:")
    for m in ("exact", "lerp_detach", "lerp_full"):
        print(f"  {m:<12s} max_abs={max_abs[m]:.3e} max_rel={max_rel[m]:.3e}")

    return failures


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 132)
    print("A4 — Ordinary local finite-difference gradient check")
    print("=" * 132)
    print(f"torch={torch.__version__} | device={device}")
    print()
    print("Coordinates are selected to be:")
    print("  - causal-valid")
    print("  - not row argmin/argmax")
    print("  - at least 20% of an interval away from either boundary")
    print()
    print("Expected result:")
    print("  Exact, LERP-detach, and LERP-full should ALL match ordinary finite differences")
    print("  at these local coordinates. The detach pathology diagnosed in A2/A3 is calibration/")
    print("  common-mode specific, not a failure of the ordinary interior LERP slope.")

    failures = []
    # FP64 is the primary correctness check. FP32 is a training-relevant check
    # with appropriately looser finite-difference tolerance.
    for dtype in (torch.float64, torch.float32):
        for K in (4, 32):
            failures.extend(run_case(device, dtype, K, score_scale=3.0, seed=20260824 + K))

    print("\n" + "=" * 132)
    print("A4 summary")
    print("=" * 132)
    if failures:
        print(f"FAIL / NEEDS REVIEW: {len(failures)} coordinate-method checks failed.")
        for item in failures[:12]:
            i, p, method, g, fd, ae, re = item
            print(
                f"  point={i} coord={(p.b,p.h,p.q,p.k)} method={method} "
                f"autograd={g:+.6e} fd={fd:+.6e} abs={ae:.3e} rel={re:.3e}"
            )
        raise SystemExit(1)

    print("PASS")
    print()
    print("Interpretation:")
    print("  - Exact softmax local autograd agrees with finite differences.")
    print("  - Current detached LERP local interior gradients agree with finite differences")
    print("    when the perturbed score is not an extremum and does not cross a bin boundary.")
    print("  - Diagnostic full-gradient LERP also agrees with the same finite differences.")
    print("  - Therefore enabling gradient through min/max preserves the correct ordinary")
    print("    piecewise-smooth LERP derivative; the issue identified by A2/A3 is specifically")
    print("    the missing adaptive-calibration derivative, not the basic LERP interpolation slope.")
    print()
    print("Phase A1-A4 is now complete. The next controlled step is Phase B:")
    print("implement --range_grad detach|full without changing the forward operator.")


if __name__ == "__main__":
    main()
