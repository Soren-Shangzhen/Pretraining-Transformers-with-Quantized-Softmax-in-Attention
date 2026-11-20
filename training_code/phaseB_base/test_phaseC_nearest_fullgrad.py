"""Phase C sanity test: Nearest hard-forward with full adaptive-range gradient.

Run from the Phase-B project directory:
    python test_phaseC_nearest_fullgrad.py

This script does NOT edit project code and does NOT train the model.

It verifies the exact semantics needed for the next controlled experiment:

1) Forward parity:
      Nearest(range_grad=detach) == Nearest(range_grad=full)
   exactly in forward.  Therefore changing range_grad is backward-only.

2) Calibration-symmetry restoration:
      With range_grad=full, the Nearest surrogate backward again respects the
      exact row-wise additive-shift symmetry:
          sum_j dL/dS_ij ~= 0.
   The historical detached variant is expected to violate this identity.

3) Hard-forward / surrogate-backward mismatch remains:
      For a non-extremal score that is safely inside one nearest Voronoi cell,
      the actual hard Nearest forward is locally constant, so its finite-
      difference derivative is zero.  The surrogate autograd derivative is
      generally nonzero.  This is intentional and is the variable Phase C
      will probe in long-horizon training.

Passing this test means:
    adaptive-calibration gradient pathology is removed,
    while the hard-forward / surrogate-gradient mismatch is preserved.
"""

from __future__ import annotations

import math
import torch

from quant_attention import _adaptive_interval_reconstruct


def causal_mask(T, device):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def probabilities(scores, causal, K, range_grad):
    w, bins, span, levels = _adaptive_interval_reconstruct(
        scores,
        causal,
        K=K,
        approx="nearest",
        return_level_idx=True,
        range_grad=range_grad,
    )
    P = w / w.sum(dim=-1, keepdim=True)
    return P, w, bins, span, levels


def scalar_loss(scores, causal, K, range_grad, v, probe):
    P, *_ = probabilities(scores, causal, K, range_grad)
    y = P @ v
    return (y * probe).sum() / y.numel()


def grad_and_loss(scores0, causal, K, range_grad, v, probe):
    scores = scores0.detach().clone().requires_grad_(True)
    loss = scalar_loss(scores, causal, K, range_grad, v, probe)
    (g,) = torch.autograd.grad(loss, scores)
    return g.detach(), float(loss.detach())


def row_shift_residual(g, causal):
    T = g.shape[-1]
    valid = causal.view(1, 1, T, T)
    gv = g.masked_fill(~valid, 0.0)
    row_sum = gv.sum(dim=-1)
    row_l1 = gv.abs().sum(dim=-1)

    active = row_l1 > 1e-14
    rho = row_sum.abs()[active] / (row_l1[active] + 1e-30)
    return {
        "mean": float(rho.mean()),
        "median": float(rho.median()),
        "max": float(rho.max()),
    }


def choose_safe_hard_coordinate(scores, causal, K):
    """Pick a valid non-extremal coordinate far from bin and nearest thresholds."""
    B, H, T, _ = scores.shape
    valid = causal.view(1, 1, T, T)

    row_max = scores.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
    row_min = scores.masked_fill(~valid, torch.inf).amin(dim=-1, keepdim=True)
    span = row_max - row_min
    h = span / float(K)
    h_safe = torch.where(span <= 1e-12, torch.ones_like(h), h)

    z = scores - row_max
    z_min = row_min - row_max
    rel = (z - z_min) / h_safe
    bin_idx = torch.floor(rel).long().clamp(0, K - 1)
    r = bin_idx.to(scores.dtype)
    b_lo = z_min + r * h_safe
    t = ((z - b_lo) / h_safe).clamp(0, 1)

    # Identify exact row extrema.
    is_min = torch.isclose(scores, row_min, rtol=0, atol=1e-12)
    is_max = torch.isclose(scores, row_max, rtol=0, atol=1e-12)

    # Safe means far from:
    #   t=0 or 1 (bin boundary) and t=0.5 (Nearest threshold).
    dist = torch.minimum(
        torch.minimum(t, 1.0 - t),
        (t - 0.5).abs(),
    )
    safe = valid & (~is_min) & (~is_max) & (dist > 0.15)

    coords = safe.nonzero(as_tuple=False)
    if coords.numel() == 0:
        raise RuntimeError("Could not find a safe hard-Nearest interior coordinate.")

    # Prefer a coordinate with large safety margin.
    vals = dist[safe]
    idx = int(vals.argmax())
    coord = coords[idx]
    b, hh, q, k = map(int, coord.tolist())

    width = float(h_safe[b, hh, q, 0])
    tt = float(t[b, hh, q, k])
    margin_t = min(tt, 1.0 - tt, abs(tt - 0.5))
    # Perturb no more than 10% of distance to the nearest relevant threshold.
    eps = max(1e-7, 0.1 * margin_t * width)

    return (b, hh, q, k), tt, width, eps


def finite_difference_coordinate(scores, causal, K, range_grad, v, probe, coord, eps):
    b, h, q, k = coord
    d = torch.zeros_like(scores)
    d[b, h, q, k] = 1.0

    with torch.no_grad():
        lp = scalar_loss(scores + eps * d, causal, K, range_grad, v, probe)
        lm = scalar_loss(scores - eps * d, causal, K, range_grad, v, probe)
        l0 = scalar_loss(scores, causal, K, range_grad, v, probe)

    dfd = float((lp - lm) / (2.0 * eps))
    fwd_delta = max(float((lp - l0).abs()), float((lm - l0).abs()))
    return dfd, fwd_delta


def main():
    torch.manual_seed(20260826)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 104)
    print("Phase C sanity — Nearest hard-forward + full adaptive-range gradient")
    print("=" * 104)
    print(f"torch={torch.__version__} | device={device}")

    # ------------------------------------------------------------------
    # 1) Forward parity in training-relevant fp32.
    # ------------------------------------------------------------------
    B, H, T = 2, 3, 47
    K_values = [4, 8, 16, 32]
    causal = causal_mask(T, device)
    scores32 = torch.randn(B, H, T, T, device=device, dtype=torch.float32) * 5.0

    print("\n[1] Forward parity: nearest detach vs full")
    for K in K_values:
        out_d = probabilities(scores32, causal, K, "detach")
        out_f = probabilities(scores32, causal, K, "full")

        P_d, w_d, bins_d, span_d, levels_d = out_d
        P_f, w_f, bins_f, span_f, levels_f = out_f

        assert torch.equal(w_d, w_f), f"K={K}: nearest weights differ"
        assert torch.equal(P_d, P_f), f"K={K}: normalized P differs"
        assert torch.equal(bins_d, bins_f), f"K={K}: bins differ"
        assert torch.equal(span_d, span_f), f"K={K}: spans differ"
        assert torch.equal(levels_d, levels_f), f"K={K}: levels differ"
        print(f"  K={K:<2d} PASS | forward bitwise identical")

    # ------------------------------------------------------------------
    # 2) Row-shift gradient residual.
    # ------------------------------------------------------------------
    print("\n[2] Row-shift gradient residual")
    Dv = 11
    v32 = torch.randn(B, H, T, Dv, device=device, dtype=torch.float32)
    probe32 = torch.randn(B, H, T, Dv, device=device, dtype=torch.float32)

    for K in [4, 32]:
        gd, _ = grad_and_loss(scores32, causal, K, "detach", v32, probe32)
        gf, _ = grad_and_loss(scores32, causal, K, "full", v32, probe32)
        rd = row_shift_residual(gd, causal)
        rf = row_shift_residual(gf, causal)

        print(
            f"  K={K:<2d} detach rho mean={rd['mean']:.3e} "
            f"median={rd['median']:.3e} max={rd['max']:.3e}"
        )
        print(
            f"       full   rho mean={rf['mean']:.3e} "
            f"median={rf['median']:.3e} max={rf['max']:.3e}"
        )

        assert rf["max"] < 2e-5, (
            f"K={K}: full-gradient Nearest did not restore shift symmetry; "
            f"rho_max={rf['max']:.3e}"
        )
        assert rd["mean"] > 1e-4, (
            f"K={K}: detached control unexpectedly has tiny residual"
        )

    # ------------------------------------------------------------------
    # 3) Hard-forward vs surrogate-gradient mismatch, in fp64.
    # ------------------------------------------------------------------
    print("\n[3] Hard-forward / surrogate-backward local mismatch")
    B2, H2, T2, Dv2 = 1, 2, 41, 9
    causal64 = causal_mask(T2, device)
    scores64 = torch.randn(B2, H2, T2, T2, device=device, dtype=torch.float64) * 3.0
    v64 = torch.randn(B2, H2, T2, Dv2, device=device, dtype=torch.float64)
    probe64 = torch.randn(B2, H2, T2, Dv2, device=device, dtype=torch.float64)

    mismatch_found = False

    for K in [4, 32]:
        coord, t, width, eps = choose_safe_hard_coordinate(scores64, causal64, K)

        g, _ = grad_and_loss(scores64, causal64, K, "full", v64, probe64)
        auto = float(g[coord])
        dfd, fwd_delta = finite_difference_coordinate(
            scores64, causal64, K, "full", v64, probe64, coord, eps
        )

        print(
            f"  K={K:<2d} coord={coord} t={t:.3f} width={width:.3f} eps={eps:.3e}\n"
            f"       surrogate autograd={auto:+.6e} | "
            f"hard-forward finite-diff={dfd:+.6e} | "
            f"max|L±-L0|={fwd_delta:.3e}"
        )

        # Hard forward should be locally constant for this deliberately safe
        # non-extremal coordinate.  The surrogate derivative should usually be
        # detectably nonzero.
        assert abs(dfd) < 1e-10, (
            f"K={K}: hard forward was not locally constant; D_FD={dfd:.3e}"
        )
        if abs(auto) > 1e-9:
            mismatch_found = True

    assert mismatch_found, (
        "Did not observe a nonzero surrogate derivative on the selected safe "
        "hard-forward coordinates. Try another random seed."
    )

    print("\n" + "=" * 104)
    print("PASS")
    print("=" * 104)
    print("Interpretation:")
    print("  - Nearest detach/full have exactly the same hard forward.")
    print("  - range_grad=full removes the adaptive-calibration row-shift pathology.")
    print("  - The intended hard-forward / LERP-surrogate backward mismatch remains.")
    print("")
    print("Phase C is therefore a clean ablation of the remaining surrogate-gradient")
    print("mismatch. The next experiment is Q32-Nearest + range_grad=full, 100M schedule.")


if __name__ == "__main__":
    main()
