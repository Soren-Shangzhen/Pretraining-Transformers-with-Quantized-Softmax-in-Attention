#!/usr/bin/env python3
"""13.3 checks for the FWM-family operators of rb-root_v1.

E1/E1b/E11  FWM-LERP training entry: closed-form gradient vs autograd (float64 random rows, error
            < 1e-9, row gradients sum to zero); the training forward equals the direct computation
            bit for bit (CPU, and GPU with TF32 off).
E11         FWM detach switch: detach and full forwards bitwise identical (LERP and strict
            Weight-STE Nearest); under detach the gradient equals the closed form WITHOUT the
            row-maximum term (derivative through the row maximum is zero); under full the row
            gradient sums to zero.
E2          (operator part) the two strict surrogates give bitwise identical forwards at equal
            parameters (MinMax and FWM); the resolved operator with per-layer step (freeze-h)
            matches its closed-form gradient and its direct forward.

The training forward is read from QuantCausalSelfAttention._quant_attention with V = identity
(head_dim = T), so y = P exactly.
"""
import sys
import torch
from rb_common import RB_ROOT, import_root, write_json, env_info

mm, qa = import_root(RB_ROOT)
RES = {"env": env_info(), "checks": []}
OK = True


def record(name, passed, **info):
    global OK
    OK &= bool(passed)
    RES["checks"].append(dict(name=name, passed=bool(passed), **info))
    print(f"{'PASS' if passed else 'FAIL'}  {name}  " + " ".join(f"{k}={v}" for k, v in info.items()), flush=True)


def module_P(q, k, sur="prob", device="cpu", layer_idx=0, **kw):
    """P from the training module: V = identity so that y = P @ I = P exactly."""
    B, H, T, D = q.shape
    assert D == T
    qa.NEAREST_SURROGATE = sur
    cfg = mm.GPTConfig(block_size=T, n_layer=max(1, len(kw.get("attn_range_tau_per_layer") or [0])),
                       n_head=H, n_embd=H * T, **kw)
    attn = qa.QuantCausalSelfAttention(cfg, layer_idx).to(device)
    v = torch.eye(T, device=device, dtype=q.dtype).expand(B, H, T, T)
    if cfg.attn_mode == "quant":
        return attn._quant_attention(q, k, v)
    raise ValueError(cfg.attn_mode)


def scores(q, k):
    return (q.float() @ k.float().transpose(-2, -1)) / (q.shape[-1] ** 0.5)


# ------------------------------------------------------------------ closed forms (float64, one row)
def cf_window_lerp(s, gw, K, tau, tail, with_max):
    """dL/ds for w = LERP on the max-anchored grid -tau + r h, h = tau/K.
    tail='zero': w = 0 below -tau (FWM);  tail='resolved': the grid continues without a lower bound.
    with_max: include the row-maximum calibration term (range_grad=full)."""
    m = s.max(); im = s.argmax(); h = tau / K
    z = s - m
    if tail == "zero":
        inwin = z >= -tau
        zc = z.clamp(-tau, 0)
        r = torch.floor((zc + tau) / h).clamp(0, K - 1)
    else:
        inwin = torch.ones_like(z, dtype=torch.bool)
        zc = z
        r = torch.floor((z + tau) / h).clamp(max=K - 1)
    eA, eB = torch.exp(-tau + r * h), torch.exp(-tau + (r + 1) * h)
    t = ((zc - (-tau + r * h)) / h).clamp(0, 1)
    w = torch.where(inwin, (1 - t) * eA + t * eB, torch.zeros_like(z))
    Delta = torch.where(inwin, (eB - eA) / h, torch.zeros_like(z))
    g = gw * Delta
    if with_max:
        g[im] += -(gw * Delta).sum()
    return w, g


def cf_minmax_lerp(s, gw, K):
    m, mn = s.max(), s.min(); im, imn = s.argmax(), s.argmin()
    z, zmin, h = s - m, mn - m, (m - mn) / K
    r = torch.floor((z - zmin) / h).clamp(0, K - 1)
    t = ((z - (zmin + r * h)) / h).clamp(0, 1)
    eA, eB = torch.exp(zmin + r * h), torch.exp(zmin + (r + 1) * h)
    w = (1 - t) * eA + t * eB
    Delta = (eB - eA) / h
    X = ((1 - t) * r * eA + t * (r + 1) * eB) / K
    rho = (r + t) / K
    dw_dm = -w - Delta * rho + X
    dw_dmn = -Delta * (1 - rho) + w - X
    g = gw * Delta
    g[im] += (gw * dw_dm).sum()
    g[imn] += (gw * dw_dmn).sum()
    return w, g


def hard_window_nearest(s, K, tau, tail):
    m = s.max(); h = tau / K; z = s - m
    if tail == "zero":
        zc = z.clamp(-tau, 0); r = torch.floor((zc + tau) / h).clamp(0, K - 1)
    else:
        zc = z; r = torch.floor((z + tau) / h).clamp(max=K - 1)
    t = ((zc - (-tau + r * h)) / h).clamp(0, 1)
    w = torch.where(t >= 0.5, torch.exp(-tau + (r + 1) * h), torch.exp(-tau + r * h))
    if tail == "zero":
        w = torch.where(z >= -tau, w, torch.zeros_like(w))
    return w


def autograd_row_grad(fn, att, row, up):
    a = att.detach().clone().requires_grad_(True)
    P = fn(a)
    (P[0, 0, row] * up).sum().backward()
    return a.grad[0, 0, row].clone(), P.detach()


def closed_form_checks():
    torch.manual_seed(0)
    T = 9
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    row = T - 1
    worst = {}
    zs_worst = {}
    n = 0
    for trial in range(200):
        att = (torch.randn(1, 1, T, T, dtype=torch.float64) * 4)
        up = torch.randn(T, dtype=torch.float64)
        s = att[0, 0, row]
        for K in (4, 16):
            for rg in ("full", "detach"):
                tau = 6.0
                # FWM-LERP (E1/E1b/E11)
                def f_lerp(a):
                    w = qa._relevant_zero_reconstruct(a, causal, K, "lerp", tau=tau, range_grad=rg)[0]
                    return w / w.sum(-1, keepdim=True)
                auto, _ = autograd_row_grad(f_lerp, att, row, up)
                wl, _ = cf_window_lerp(s, torch.zeros(T, dtype=torch.float64), K, tau, "zero", False)
                gw = (up - (wl / wl.sum() * up).sum()) / wl.sum()
                _, g = cf_window_lerp(s, gw, K, tau, "zero", rg == "full")
                key = f"fwm_lerp_K{K}_{rg}"
                worst[key] = max(worst.get(key, 0), (g - auto).abs().max().item())
                if rg == "full":
                    zs_worst[key] = max(zs_worst.get(key, 0), auto.sum().abs().item())
                # FWM-Nearest strict Weight-STE (E11 FWM-Weight)
                for sur in ("weight_strict", "prob"):
                    def f_near(a, sur=sur):
                        with torch.no_grad():
                            wh = qa._relevant_zero_reconstruct(a, causal, K, "nearest", tau=tau, range_grad=rg,
                                                               nearest_forward="direct")[0]
                        wl_ = qa._relevant_zero_reconstruct(a, causal, K, "lerp", tau=tau, range_grad=rg)[0]
                        if sur == "weight_strict":
                            w = wh.detach() + (wl_ - wl_.detach())
                            return w / w.sum(-1, keepdim=True)
                        Pl = wl_ / wl_.sum(-1, keepdim=True)
                        return (wh / wh.sum(-1, keepdim=True)).detach() + (Pl - Pl.detach())
                    auto, _ = autograd_row_grad(f_near, att, row, up)
                    if sur == "weight_strict":
                        wn = hard_window_nearest(s, K, tau, "zero")
                        gw = (up - (wn / wn.sum() * up).sum()) / wn.sum()
                    else:
                        gw = (up - (wl / wl.sum() * up).sum()) / wl.sum()
                    _, g = cf_window_lerp(s, gw, K, tau, "zero", rg == "full")
                    key = f"fwm_nearest_{sur}_K{K}_{rg}"
                    worst[key] = max(worst.get(key, 0), (g - auto).abs().max().item())
                    if rg == "full":
                        zs_worst[key] = max(zs_worst.get(key, 0), auto.sum().abs().item())
            # resolved operator with a per-layer step h0 (E2 freeze-h), K=4 only in the runs
            if K == 4:
                for h0 in (0.8, 1.7, 3.1, 6.5):
                    tau_l = K * h0
                    for sur in ("weight_strict", "prob"):
                        def f_res(a, sur=sur, tau_l=tau_l):
                            with torch.no_grad():
                                wh = qa._relevant_zero_reconstruct(a, causal, K, "nearest", tau=tau_l,
                                                                   range_grad="full", nearest_forward="direct",
                                                                   tail_policy="resolved")[0]
                            wl_ = qa._relevant_zero_reconstruct(a, causal, K, "lerp", tau=tau_l, range_grad="full",
                                                                tail_policy="resolved")[0]
                            if sur == "weight_strict":
                                w = wh.detach() + (wl_ - wl_.detach())
                                return w / w.sum(-1, keepdim=True)
                            Pl = wl_ / wl_.sum(-1, keepdim=True)
                            return (wh / wh.sum(-1, keepdim=True)).detach() + (Pl - Pl.detach())
                        auto, P = autograd_row_grad(f_res, att, row, up)
                        wlr, _ = cf_window_lerp(s, torch.zeros(T, dtype=torch.float64), K, tau_l, "resolved", False)
                        wnr = hard_window_nearest(s, K, tau_l, "resolved")
                        Wn = wnr if sur == "weight_strict" else wlr
                        gw = (up - (Wn / Wn.sum() * up).sum()) / Wn.sum()
                        _, g = cf_window_lerp(s, gw, K, tau_l, "resolved", True)
                        key = f"resolved_{sur}_K4"
                        worst[key] = max(worst.get(key, 0), (g - auto).abs().max().item())
                        zs_worst[key] = max(zs_worst.get(key, 0), auto.sum().abs().item())
                        # hard forward equals the direct resolved Nearest weights
                        fwd = (P[0, 0, row] - wnr / wnr.sum()).abs().max().item()
                        worst[key + "_forward_vs_direct_f64"] = max(worst.get(key + "_forward_vs_direct_f64", 0), fwd)
        n += 1
    for key, v in worst.items():
        record(f"closed-form vs autograd (float64, {n} random rows): {key}", v < 1e-9, max_abs_err=f"{v:.2e}")
    for key, v in zs_worst.items():
        record(f"row gradient sums to zero (full): {key}", v < 1e-9, max_abs_rowsum=f"{v:.2e}")


def direct_fwm_lerp_P(att, K, tau):
    """Independent direct computation of FWM-LERP probabilities from the definition."""
    T = att.shape[-1]
    valid = torch.ones(T, T, dtype=torch.bool, device=att.device).tril().view(1, 1, T, T)
    M = att.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
    z = att - M
    h = tau / K
    grid_w = torch.exp(-tau + torch.arange(K + 1, device=att.device, dtype=att.dtype) * h)  # e^{b_r}
    zc = z.clamp(min=-tau, max=0.0)
    r = torch.floor((zc + tau) / h).to(torch.long).clamp(0, K - 1)
    t = ((zc - (-tau + r.to(att.dtype) * h)) / h).clamp(0.0, 1.0)
    w = (1.0 - t) * grid_w[r] + t * grid_w[r + 1]
    w = torch.where(z < -tau, torch.zeros_like(w), w).masked_fill(~valid, 0.0)
    return w / w.sum(-1, keepdim=True)


def forward_checks(device):
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1)
    B, H, T = 1, 3, 64
    for scale in (0.5, 2.0, 6.0):
        q = torch.randn(B, H, T, T, device=device) * scale
        k = torch.randn(B, H, T, T, device=device)
        att = scores(q, k)
        for K in (4, 16):
            Pf = module_P(q, k, device=device, attn_mode="quant", attn_approx="lerp", attn_K=K,
                          attn_range_mode="relevant_zero", attn_range_grad="full")
            Pd = module_P(q, k, device=device, attn_mode="quant", attn_approx="lerp", attn_K=K,
                          attn_range_mode="relevant_zero", attn_range_grad="detach")
            Pdir = direct_fwm_lerp_P(att, K, 6.0)
            Pref64 = direct_fwm_lerp_P(att.double(), K, 6.0)
            record(f"[{device}] FWM-LERP K{K} scale{scale}: training forward == direct computation (bitwise)",
                   torch.equal(Pf, Pdir), max_abs=f"{(Pf - Pdir).abs().max().item():.1e}",
                   vs_f64=f"{(Pf.double() - Pref64).abs().max().item():.1e}")
            record(f"[{device}] FWM-LERP K{K} scale{scale}: detach forward == full forward (bitwise)", torch.equal(Pf, Pd))
            for sur in ("weight_strict", "prob"):
                Nf = module_P(q, k, sur=sur, device=device, attn_mode="quant", attn_approx="nearest", attn_K=K,
                              attn_range_mode="relevant_zero", attn_range_grad="full")
                Nd = module_P(q, k, sur=sur, device=device, attn_mode="quant", attn_approx="nearest", attn_K=K,
                              attn_range_mode="relevant_zero", attn_range_grad="detach")
                record(f"[{device}] FWM-Nearest {sur} K{K} scale{scale}: detach forward == full forward (bitwise)",
                       torch.equal(Nf, Nd))
            # E2: the two strict surrogates have the same forward at equal parameters
            for rm, extra in (("minmax", {}), ("relevant_zero", {})):
                Pw = module_P(q, k, sur="weight_strict", device=device, attn_mode="quant", attn_approx="nearest",
                              attn_K=K, attn_range_mode=rm, attn_range_grad="full", **extra)
                Pp = module_P(q, k, sur="prob", device=device, attn_mode="quant", attn_approx="nearest",
                              attn_K=K, attn_range_mode=rm, attn_range_grad="full", **extra)
                record(f"[{device}] {rm}-Nearest K{K} scale{scale}: strict Weight-STE forward == strict Prob-STE forward (bitwise)",
                       torch.equal(Pw, Pp))
        # resolved with per-layer step: both surrogates, and layer index selects the tau
        taus = (4 * 1.3, 4 * 2.9)
        for li in (0, 1):
            Pw = module_P(q, k, sur="weight_strict", device=device, layer_idx=li, attn_mode="quant", attn_approx="nearest",
                          attn_K=4, attn_range_mode="relevant_zero", attn_range_grad="full", attn_tail_policy="resolved",
                          attn_range_tau_per_layer=taus)
            Pp = module_P(q, k, sur="prob", device=device, layer_idx=li, attn_mode="quant", attn_approx="nearest",
                          attn_K=4, attn_range_mode="relevant_zero", attn_range_grad="full", attn_tail_policy="resolved",
                          attn_range_tau_per_layer=taus)
            att64 = att.double()
            Pdirect = torch.stack([torch.stack([
                torch.cat([(lambda w: w / w.sum())(hard_window_nearest(att64[0, hh, i, :i + 1], 4, taus[li], "resolved")),
                           torch.zeros(T - i - 1, dtype=torch.float64, device=device)])
                for i in range(T)]) for hh in range(H)]).unsqueeze(0)
            record(f"[{device}] resolved freeze-h layer{li} (h0={taus[li]/4}) scale{scale}: Weight-STE fwd == Prob-STE fwd (bitwise)",
                   torch.equal(Pw, Pp), vs_direct_f64=f"{(Pw.double() - Pdirect).abs().max().item():.1e}")


def main():
    closed_form_checks()
    forward_checks("cpu")
    if torch.cuda.is_available():
        forward_checks("cuda")
    RES["all_pass"] = OK
    print("written", write_json(f"13_3_fwm_ops_torch{torch.__version__.split('+')[0]}.json", RES))
    print("ALL PASS" if OK else "SOME FAILED")
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
