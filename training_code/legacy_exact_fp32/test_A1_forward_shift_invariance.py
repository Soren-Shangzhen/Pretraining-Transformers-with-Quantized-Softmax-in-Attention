"""Phase A1: forward row-shift invariance test.

Purpose
-------
Verify the exact forward symmetry

    F(S + c 1) = F(S)

for each causal query row, where c can differ across (batch, head, query).

This test is intentionally forward-only.  It does NOT inspect gradients and it
DOES NOT modify the training operator.

It checks:
  1) Exact causal softmax probabilities and outputs.
  2) Adaptive Nearest reconstruction: raw weights, bins, span, normalized P, PV.
  3) Adaptive LERP reconstruction: raw weights, bins, span, normalized P, PV.
  4) K in {2, 4, 8, 16, 32}.
  5) Several row-wise shift scales/signs.

Run from the project directory:

    python test_A1_forward_shift_invariance.py

The script expects quant_attention.py to be importable from the current working
project directory.
"""

import math
import sys
from dataclasses import dataclass

import torch

from quant_attention import _adaptive_interval_reconstruct


@dataclass
class ErrorStats:
    abs_max: float
    rel_max: float


def causal_mask(T: int, device):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def max_errors(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> ErrorStats:
    d = (a - b).abs()
    scale = torch.maximum(a.abs(), b.abs()).clamp_min(eps)
    return ErrorStats(float(d.max().item()), float((d / scale).max().item()))


def exact_softmax_from_scores(att: torch.Tensor, causal: torch.Tensor):
    """Exact stable causal softmax from an already-formed score tensor."""
    T = att.shape[-1]
    valid = causal.view(1, 1, T, T)
    masked = att.masked_fill(~valid, -torch.inf)
    row_max = masked.amax(dim=-1, keepdim=True)
    z = masked - row_max
    w = torch.exp(z)
    w = w.masked_fill(~valid, 0.0)
    P = w / w.sum(dim=-1, keepdim=True)
    return w, P


def assert_close(name, a, b, atol=2e-5, rtol=2e-5):
    err = max_errors(a, b)
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} FAILED: max_abs={err.abs_max:.3e}, max_rel={err.rel_max:.3e}, "
            f"atol={atol}, rtol={rtol}"
        )
    return err


def run_one(device: str, shift_scale: float, dtype=torch.float32):
    torch.manual_seed(20260824)

    B, H, T, Dv = 2, 3, 17, 11
    causal = causal_mask(T, device)

    # Scores are deliberately moderate so that the test checks the mathematical
    # symmetry rather than catastrophic loss of fp32 precision after adding a
    # huge common offset.
    att = (torch.randn(B, H, T, T, device=device, dtype=dtype) * 2.5)
    v = torch.randn(B, H, T, Dv, device=device, dtype=dtype)

    # Independent constant for every (batch, head, query) row.
    shifts = torch.randn(B, H, T, 1, device=device, dtype=dtype) * shift_scale
    shifted = att + shifts

    # ------------------------------------------------------------------
    # Exact softmax
    # ------------------------------------------------------------------
    ew1, ep1 = exact_softmax_from_scores(att, causal)
    ew2, ep2 = exact_softmax_from_scores(shifted, causal)
    ey1 = ep1 @ v
    ey2 = ep2 @ v

    exact_w_err = assert_close("Exact raw exp weights", ew1, ew2)
    exact_p_err = assert_close("Exact probabilities", ep1, ep2)
    exact_y_err = assert_close("Exact attention output", ey1, ey2)

    print(
        f"  Exact      | P abs={exact_p_err.abs_max:.3e} "
        f"| Y abs={exact_y_err.abs_max:.3e}"
    )

    # ------------------------------------------------------------------
    # Quantized operators
    # ------------------------------------------------------------------
    for K in [2, 4, 8, 16, 32]:
        for approx in ["nearest", "lerp"]:
            w1, bins1, span1 = _adaptive_interval_reconstruct(
                att, causal, K=K, approx=approx
            )
            w2, bins2, span2 = _adaptive_interval_reconstruct(
                shifted, causal, K=K, approx=approx
            )

            if not torch.equal(bins1, bins2):
                nbad = int((bins1 != bins2).sum().item())
                raise AssertionError(f"K={K} {approx}: {nbad} bin assignments changed")

            span_err = assert_close(f"K={K} {approx} span", span1, span2)
            w_err = assert_close(f"K={K} {approx} raw weights", w1, w2)

            p1 = w1 / w1.sum(dim=-1, keepdim=True)
            p2 = w2 / w2.sum(dim=-1, keepdim=True)
            y1 = p1 @ v
            y2 = p2 @ v

            p_err = assert_close(f"K={K} {approx} P", p1, p2)
            y_err = assert_close(f"K={K} {approx} Y", y1, y2)

            print(
                f"  K={K:<2d} {approx:<7s} | span abs={span_err.abs_max:.3e} "
                f"| w abs={w_err.abs_max:.3e} | P abs={p_err.abs_max:.3e} "
                f"| Y abs={y_err.abs_max:.3e}"
            )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 88)
    print("A1 — Forward row-shift invariance")
    print("=" * 88)
    print(f"torch={torch.__version__} | device={device} | dtype=float32")
    print("Expected identity: F(S + c_row * 1) = F(S)")
    print("This is a FORWARD-ONLY test; gradients are intentionally not inspected.\n")

    # Moderate and fairly large shifts.  Avoid absurd offsets such as 1e8 in
    # fp32 because those probe floating-point cancellation rather than the
    # operator's mathematical invariance.
    for scale in [0.1, 1.0, 10.0, 50.0]:
        print(f"--- row-shift scale = {scale:g} ---")
        run_one(device, scale)
        print()

    print("PASS: A1 forward row-shift invariance holds for Exact, Nearest, and LERP")
    print("      across K={2,4,8,16,32} for the tested fp32 shifts.")
    print("\nInterpretation:")
    print("  - The forward operator itself respects row-wise additive-shift symmetry.")
    print("  - This does NOT validate the backward gradient.")
    print("  - Proceed next to A2: gradient row-shift residual.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        raise
