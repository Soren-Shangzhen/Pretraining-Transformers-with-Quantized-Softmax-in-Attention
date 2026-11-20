"""Quick correctness checks. Run: python test_sanity.py"""

import numpy as np
import torch

from quant_attention import (_adaptive_interval_reconstruct,
                             QuantCausalSelfAttention)
from model import GPT, GPTConfig


def _causal(T):
    return torch.ones(T, T, dtype=torch.bool).tril()


def test_exact_nearest_and_lerp_values():
    # One full valid row with scores spanning [0, 4], K=2 -> boundaries 0,2,4.
    # We embed it as the final causal row of a 4x4 attention matrix.
    att = torch.zeros(1, 1, 4, 4)
    att[0, 0, 3] = torch.tensor([0.0, 0.8, 2.6, 4.0])
    causal = _causal(4)

    wn, bn, _ = _adaptive_interval_reconstruct(att, causal, K=2, approx="nearest")
    wl, bl, _ = _adaptive_interval_reconstruct(att, causal, K=2, approx="lerp")

    # Stable centered boundaries are [-4,-2,0].
    z = torch.tensor([-4.0, -3.2, -1.4, 0.0])
    # nearest boundary: [-4,-4,-2,0]
    expect_n = torch.exp(torch.tensor([-4.0, -4.0, -2.0, 0.0]))
    assert torch.allclose(wn[0, 0, 3, :4], expect_n, atol=1e-6)

    # weight-space interpolation in the appropriate interval.
    # z=-3.2: t=.4 between exp(-4),exp(-2)
    # z=-1.4: t=.3 between exp(-2),exp(0)
    expect_l = torch.tensor([
        torch.exp(torch.tensor(-4.0)),
        0.6 * torch.exp(torch.tensor(-4.0)) + 0.4 * torch.exp(torch.tensor(-2.0)),
        0.7 * torch.exp(torch.tensor(-2.0)) + 0.3 * torch.tensor(1.0),
        1.0,
    ])
    assert torch.allclose(wl[0, 0, 3, :4], expect_l, atol=1e-6)
    assert torch.equal(bn[0, 0, 3, :4], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(bl[0, 0, 3, :4], torch.tensor([0, 0, 1, 1]))
    print("exact nearest/LERP reconstruction ok")


def test_row_adaptive_range_and_shift_invariance():
    torch.manual_seed(0)
    T = 8
    causal = _causal(T)
    att = torch.randn(2, 3, T, T)
    # Row-common shifts must not change forward weights because min/max move too.
    shifts = torch.randn(2, 3, T, 1) * 7.0
    for approx in ["nearest", "lerp"]:
        w1, b1, s1 = _adaptive_interval_reconstruct(att, causal, 8, approx)
        w2, b2, s2 = _adaptive_interval_reconstruct(att + shifts, causal, 8, approx)
        assert torch.allclose(w1, w2, atol=2e-6, rtol=2e-6)
        assert torch.equal(b1, b2)
        assert torch.allclose(s1, s2, atol=2e-6, rtol=2e-6)
    print("row-adaptive range + forward shift invariance ok")


def test_degenerate_rows_and_mask():
    T = 6
    causal = _causal(T)
    att = torch.zeros(1, 1, T, T, requires_grad=True)
    for approx in ["nearest", "lerp"]:
        w, bins, _ = _adaptive_interval_reconstruct(att, causal, 8, approx)
        valid = causal.view(1, 1, T, T)
        assert torch.all(w[valid] == 1.0)
        assert torch.all(w[~valid] == 0.0)
        assert torch.all(bins[~valid] == -1)
    print("degenerate rows + causal mask ok")


def test_gradients_flow():
    torch.manual_seed(1)
    T = 12
    causal = _causal(T)
    for approx in ["nearest", "lerp"]:
        att = torch.randn(2, 2, T, T, requires_grad=True)
        w, _, _ = _adaptive_interval_reconstruct(att, causal, 8, approx)
        P = w / w.sum(-1, keepdim=True)
        loss = (P * torch.randn_like(P)).sum()
        loss.backward()
        g = att.grad
        assert g is not None and torch.isfinite(g).all()
        assert g.abs().sum() > 0, f"{approx}: zero gradient"
        print(f"gradient '{approx}' ok | norm {g.norm():.4f}")


def test_nearest_backward_uses_lerp_slope():
    # With extrema detached, an interior score's nearest-forward gradient should
    # equal the local weight-space LERP slope.
    causal = _causal(4)
    att = torch.zeros(1, 1, 4, 4, requires_grad=True)
    with torch.no_grad():
        att[0, 0, 3] = torch.tensor([0.0, 0.8, 2.6, 4.0])
    w, _, _ = _adaptive_interval_reconstruct(att, causal, 2, "nearest")
    w[0, 0, 3, 1].backward()
    # row centered range [-4,0], first interval [-4,-2], width=2
    expect = (torch.exp(torch.tensor(-2.0)) - torch.exp(torch.tensor(-4.0))) / 2.0
    assert abs(att.grad[0, 0, 3, 1].item() - expect.item()) < 1e-6
    print("nearest forward / LERP-surrogate backward ok")


def test_exact_fp32_softmax_matches_reference():
    """Explicit FP32 control must match a manual causal softmax reference."""
    torch.manual_seed(123)
    cfg = GPTConfig(block_size=16, vocab_size=128, n_layer=1, n_head=2,
                    n_embd=32, attn_mode="exact_fp32_softmax")
    mod = QuantCausalSelfAttention(cfg)
    q = torch.randn(2, 2, 9, 16, requires_grad=True)
    k = torch.randn(2, 2, 9, 16, requires_grad=True)
    v = torch.randn(2, 2, 9, 16, requires_grad=True)

    y = mod._exact_fp32_softmax(q, k, v)
    scores = (q.float() @ k.float().transpose(-2, -1)) / (16 ** 0.5)
    causal = torch.ones(9, 9, dtype=torch.bool).tril().view(1, 1, 9, 9)
    scores = scores.masked_fill(~causal, -torch.inf)
    p_ref = torch.softmax(scores, dim=-1)
    y_ref = p_ref @ v.float()
    assert torch.allclose(y.float(), y_ref, atol=2e-6, rtol=2e-6)

    probe = torch.randn_like(y)
    (y * probe).sum().backward()
    for name, t in [("q", q), ("k", k), ("v", v)]:
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
    # Diagnostics must be available without altering the forward result.
    mod.collect_stats = True
    y_stats = mod._exact_fp32_softmax(q.detach(), k.detach(), v.detach())
    assert torch.allclose(y_stats.float(), y_ref.detach(), atol=2e-6, rtol=2e-6)
    assert mod.last_span_mean is not None
    assert mod.last_q_norm_sum is not None and mod.last_k_norm_sum is not None
    print("exact FP32 softmax matches causal reference + gradients/stats ok")



def test_range_grad_forward_identical_and_full_restores_shift_gradient():
    """Phase-B core invariant: range_grad changes backward, never forward."""
    torch.manual_seed(2026)
    T = 17
    causal = _causal(T)
    att0 = torch.randn(2, 2, T, T) * 3.0

    for K in [4, 32]:
        for approx in ["nearest", "lerp"]:
            wd, bd, sd = _adaptive_interval_reconstruct(
                att0, causal, K, approx, range_grad="detach"
            )
            wf, bf, sf = _adaptive_interval_reconstruct(
                att0, causal, K, approx, range_grad="full"
            )
            assert torch.equal(bd, bf)
            assert torch.equal(wd, wf), f"forward changed for {approx} K={K}"
            assert torch.equal(sd, sf)

    # LERP-full must restore the exact row-shift gradient identity while the
    # historical detached backward should exhibit a detectable residual.
    probe = torch.randn(1, 1, T, T)
    valid = causal.view(1, 1, T, T)

    def row_rho(range_grad):
        att = (torch.randn(1, 1, T, T) * 8.0).requires_grad_(True)
        w, _, _ = _adaptive_interval_reconstruct(
            att, causal, 32, "lerp", range_grad=range_grad
        )
        P = w / w.sum(-1, keepdim=True)
        loss = (P * probe).sum()
        (g,) = torch.autograd.grad(loss, att)
        gv = g.masked_fill(~valid, 0.0)
        row_sum = gv.sum(-1).abs()
        row_l1 = gv.abs().sum(-1)
        active = row_l1 > 1e-12
        rho = row_sum[active] / row_l1[active]
        return float(rho.mean()), float(rho.max())

    # Use identical random scores for the two branches.
    torch.manual_seed(77)
    mean_detach, max_detach = row_rho("detach")
    torch.manual_seed(77)
    mean_full, max_full = row_rho("full")

    assert max_full < 2e-5, (mean_full, max_full)
    assert mean_detach > 1e-3, (mean_detach, max_detach)
    print(
        "range_grad forward parity + shift-gradient restoration ok | "
        f"rho detach mean={mean_detach:.3e}, full max={max_full:.3e}"
    )


def test_phase_b_stats_fields():
    cfg = GPTConfig(block_size=16, vocab_size=128, n_layer=1, n_head=2,
                    n_embd=32, attn_mode="quant", attn_K=8,
                    attn_approx="lerp", attn_range_grad="full")
    mod = QuantCausalSelfAttention(cfg)
    mod.collect_stats = True
    _ = mod(torch.randn(1, 8, 32))
    assert mod.last_row_center_sum is not None
    assert mod.last_row_center_sq_sum is not None
    assert mod.last_q_proj_weight_norm is not None
    assert mod.last_k_proj_weight_norm is not None
    print("Phase-B row-center/projection-norm diagnostics ok")

def test_attention_module():
    torch.manual_seed(0)
    for approx in ["nearest", "lerp"]:
        cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=2, n_head=2,
                        n_embd=32, attn_mode="quant", attn_K=8,
                        attn_approx=approx)
        attn = QuantCausalSelfAttention(cfg)
        x = torch.randn(2, 16, 32, requires_grad=True)
        y = attn(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.sum().backward()
        assert torch.isfinite(x.grad).all()
        attn.collect_stats = True
        _ = attn(torch.randn(1, 8, 32))
        occ = attn.last_occupancy
        assert occ is not None and abs(occ.sum().item() - 1.0) < 1e-5
        assert attn.last_span_mean is not None
        assert attn.last_q_norm_sum is not None
        assert attn.last_k_norm_sum is not None
        if approx == "nearest":
            level_occ = attn.last_level_occupancy
            assert level_occ is not None
            assert len(level_occ) == 9   # K+1 reconstruction boundaries
            assert abs(level_occ.sum().item() - 1.0) < 1e-5
        else:
            assert attn.last_level_occupancy is None
        print(f"attention {approx} ok | occupancy sum={occ.sum().item():.3f}")


def test_tiny_training_all_modes():
    torch.manual_seed(0)
    vocab = 128
    data = torch.tensor(np.tile(np.arange(64) % vocab, 200), dtype=torch.long)

    def batch(bs=8, T=32):
        ix = torch.randint(0, len(data) - T - 1, (bs,))
        x = torch.stack([data[i:i + T] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + T] for i in ix])
        return x, y

    for mode, approx, range_grad in [("softmax", "lerp", "detach"),
                                     ("exact_fp32_softmax", "lerp", "detach"),
                                     ("quant", "nearest", "detach"),
                                     ("quant", "lerp", "detach"),
                                     ("quant", "lerp", "full")]:
        torch.manual_seed(1)
        cfg = GPTConfig(block_size=32, vocab_size=vocab, n_layer=2, n_head=2,
                        n_embd=64, attn_mode=mode, attn_K=8,
                        attn_approx=approx, attn_range_grad=range_grad)
        model = GPT(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        first = last = None
        for i in range(60):
            x, y = batch()
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(gn), f"{mode}/{approx}/{range_grad}: non-finite grad"
            opt.step()
            if i == 0:
                first = loss.item()
            last = loss.item()
        assert last < first * 0.7, \
            f"{mode}/{approx}/{range_grad}: loss not decreasing ({first:.3f}->{last:.3f})"
        print(f"train {mode:8s}/{approx:7s}/{range_grad:6s} ok | "
              f"loss {first:.3f} -> {last:.3f}")


if __name__ == "__main__":
    test_exact_nearest_and_lerp_values()
    test_row_adaptive_range_and_shift_invariance()
    test_degenerate_rows_and_mask()
    test_gradients_flow()
    test_nearest_backward_uses_lerp_slope()
    test_exact_fp32_softmax_matches_reference()
    test_range_grad_forward_identical_and_full_restores_shift_gradient()
    test_phase_b_stats_fields()
    test_attention_module()
    test_tiny_training_all_modes()
    print("\nall sanity checks passed")
