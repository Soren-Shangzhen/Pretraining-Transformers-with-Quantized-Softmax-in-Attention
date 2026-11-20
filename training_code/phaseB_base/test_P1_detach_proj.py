#!/usr/bin/env python3
"""P1-0 sanity tests for range_grad=detach_proj.

Pass criteria:
1) detach and detach_proj have bitwise-identical forward reconstruction.
2) detach_proj score gradient equals the explicit causal-valid row-zero-sum
   projection of the detach score gradient.
3) projected score gradients have near-zero causal-valid row sums.
4) all row-zero-sum tangent directional derivatives are preserved.
5) module forward is bitwise identical between detach and detach_proj.
6) detach_proj is compatible with non-reentrant gradient checkpointing.

Run from the repository root after installing the patched files:
    python test_P1_detach_proj.py
"""

import math
import copy

import torch

from quant_attention import (
    _adaptive_interval_reconstruct,
    _project_causal_row_zero_sum_grad,
    QuantCausalSelfAttention,
)
from model import GPT, GPTConfig


def causal_mask(T, device):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def score_loss(scores, causal, K, range_grad, v, probe, install_projection=False):
    if install_projection:
        scores.register_hook(
            lambda grad, causal=causal:
                _project_causal_row_zero_sum_grad(grad, causal)
        )
    w, bins, span = _adaptive_interval_reconstruct(
        scores, causal, K=K, approx="lerp", range_grad=range_grad
    )
    P = w / w.sum(dim=-1, keepdim=True)
    y = P @ v
    loss = (y * probe).sum() / y.numel()
    return loss, w, bins, span


def score_grad(base, causal, K, mode, v, probe):
    s = base.detach().clone().requires_grad_(True)
    install = mode == "detach_proj"
    loss, w, bins, span = score_loss(
        s, causal, K, mode, v, probe, install_projection=install
    )
    (g,) = torch.autograd.grad(loss, s)
    return {
        "loss": loss.detach(),
        "w": w.detach(),
        "bins": bins.detach(),
        "span": span.detach(),
        "g": g.detach(),
    }


def row_residual(g, causal):
    T = g.shape[-1]
    valid = causal.view(1, 1, T, T)
    gv = g.masked_fill(~valid, 0.0)
    row_sum = gv.sum(dim=-1)
    row_l1 = gv.abs().sum(dim=-1)
    active = row_l1 > 1e-30
    rho = row_sum.abs()[active] / (row_l1[active] + 1e-30)
    return float(rho.max()), float(rho.mean())


def make_zero_sum_tangent(shape, causal, dtype, device):
    d = torch.randn(shape, dtype=dtype, device=device)
    T = shape[-1]
    valid = causal.view(1, 1, T, T)
    dv = d.masked_fill(~valid, 0.0)
    n = valid.sum(dim=-1, keepdim=True).to(dtype=dtype)
    mean = dv.sum(dim=-1, keepdim=True) / n
    d = torch.where(valid, d - mean, torch.zeros_like(d))
    return d


def test_score_level(device):
    print("\n[P1-0.1] score-level projection")
    torch.manual_seed(20260828)

    dtype = torch.float64
    B, H, T, Dv, K = 2, 3, 37, 11, 32
    causal = causal_mask(T, device)

    base = torch.randn(B, H, T, T, dtype=dtype, device=device) * 3.0
    v = torch.randn(B, H, T, Dv, dtype=dtype, device=device)
    probe = torch.randn(B, H, T, Dv, dtype=dtype, device=device)

    detach = score_grad(base, causal, K, "detach", v, probe)
    proj = score_grad(base, causal, K, "detach_proj", v, probe)
    full = score_grad(base, causal, K, "full", v, probe)

    assert torch.equal(detach["w"], proj["w"]), "detach/proj forward weights differ"
    assert torch.equal(detach["bins"], proj["bins"]), "detach/proj bins differ"
    assert torch.equal(detach["span"], proj["span"]), "detach/proj spans differ"
    assert torch.equal(detach["loss"], proj["loss"]), "detach/proj losses differ"

    expected = _project_causal_row_zero_sum_grad(detach["g"], causal)
    max_proj_err = float((proj["g"] - expected).abs().max())
    assert max_proj_err < 1e-15, f"projection mismatch: {max_proj_err:.3e}"

    rho_max, rho_mean = row_residual(proj["g"], causal)
    assert rho_max < 1e-12, f"projected row residual too large: {rho_max:.3e}"

    # Projection must preserve every tangent direction whose valid entries sum
    # to zero independently in each query row.
    tangent_errs = []
    for _ in range(8):
        d = make_zero_sum_tangent(base.shape, causal, dtype, device)
        dd = torch.sum(detach["g"] * d)
        dp = torch.sum(proj["g"] * d)
        tangent_errs.append(float((dd - dp).abs()))
    tangent_max = max(tangent_errs)
    assert tangent_max < 1e-12, f"zero-sum tangent derivative changed: {tangent_max:.3e}"

    # Diagnostic only: detach_proj is not intended to reproduce the full
    # calibration derivative; it removes only the row common mode.
    valid = causal.view(1, 1, T, T)
    full_diff = float(
        ((proj["g"] - full["g"]).masked_fill(~valid, 0.0)).abs().max()
    )

    print("  forward detach == detach_proj: BITWISE PASS")
    print(f"  max |g_proj - Pi(g_detach)| = {max_proj_err:.3e}")
    print(f"  projected rho_max           = {rho_max:.3e}")
    print(f"  projected rho_mean          = {rho_mean:.3e}")
    print(f"  tangent derivative max err  = {tangent_max:.3e}")
    print(f"  max |g_proj - g_full|       = {full_diff:.3e}  (diagnostic)")


def test_module_forward(device):
    print("\n[P1-0.2] actual attention-module forward parity")
    torch.manual_seed(20260828)

    base_cfg = GPTConfig(
        block_size=32,
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_embd=32,
        attn_mode="quant",
        attn_K=32,
        attn_approx="lerp",
        attn_range_grad="detach",
    )
    cfg_p = copy.deepcopy(base_cfg)
    cfg_p.attn_range_grad = "detach_proj"

    a = QuantCausalSelfAttention(base_cfg).to(device)
    b = QuantCausalSelfAttention(cfg_p).to(device)
    b.load_state_dict(a.state_dict())

    x1 = torch.randn(2, 32, 32, device=device, dtype=torch.float32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = a(x1)
    y2 = b(x2)

    if not torch.equal(y1, y2):
        max_forward = float((y1 - y2).abs().max())
        raise AssertionError(f"module forward not bitwise identical: {max_forward:.3e}")

    probe = torch.randn_like(y1)
    (y1 * probe).sum().backward()
    (y2 * probe).sum().backward()

    # c_proj sees exactly the same forward attention output, so its gradients
    # should be identical. c_attn is expected to differ in Q/K gradient paths.
    assert torch.equal(a.c_proj.weight.grad, b.c_proj.weight.grad), \
        "c_proj gradients differ despite identical forward attention output"

    print("  module output: BITWISE PASS")
    print("  c_proj grad  : BITWISE PASS")


def collect_grads(model):
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def test_gradient_checkpointing(device):
    print("\n[P1-0.3] non-reentrant gradient-checkpoint compatibility")
    torch.manual_seed(20260828)

    cfg = GPTConfig(
        block_size=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=32,
        attn_mode="quant",
        attn_K=32,
        attn_approx="lerp",
        attn_range_grad="detach_proj",
    )

    m0 = GPT(cfg).to(device)
    m1 = GPT(copy.deepcopy(cfg)).to(device)
    m1.load_state_dict(m0.state_dict())

    m0.train()
    m1.train()
    m0.grad_checkpointing = False
    m1.grad_checkpointing = True

    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), device=device)

    _, loss0 = m0(idx, tgt)
    _, loss1 = m1(idx, tgt)
    loss0.backward()
    loss1.backward()

    loss_err = abs(float(loss0.detach()) - float(loss1.detach()))
    g0 = collect_grads(m0)
    g1 = collect_grads(m1)
    assert g0.keys() == g1.keys()

    max_abs = 0.0
    num = 0.0
    den = 0.0
    for k in g0:
        diff = (g0[k] - g1[k]).double()
        max_abs = max(max_abs, float(diff.abs().max()))
        num += float((diff * diff).sum())
        den += float((g0[k].double() ** 2).sum())
    rel_l2 = math.sqrt(num / max(den, 1e-300))

    # CPU is normally bitwise/near-bitwise here; CUDA checkpoint recomputation
    # can differ at tiny floating-point scale.
    assert loss_err < 1e-6, f"GC loss mismatch: {loss_err:.3e}"
    assert rel_l2 < 2e-5, f"GC grad relative L2 too large: {rel_l2:.3e}"

    print(f"  loss abs err = {loss_err:.3e}")
    print(f"  grad max abs = {max_abs:.3e}")
    print(f"  grad rel L2  = {rel_l2:.3e}")
    print("  checkpointing compatibility: PASS")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("P1-0 sanity: detach + causal-valid row-zero-sum score-gradient projection")
    print("=" * 80)
    print("torch :", torch.__version__)
    print("device:", device)

    test_score_level(device)
    test_module_forward(device)
    test_gradient_checkpointing(device)

    print("\n" + "=" * 80)
    print("P1-0 ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
