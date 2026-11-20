"""Numerically verify the report's closed-form gradients against autograd of the frozen training code."""
import sys, torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
import os
KI_REPO = os.environ.get("KI_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
SNAP = os.path.join(KI_REPO, "analysis_main_v1/D_robustness_124m_100m_5seed/code_snapshots")
sys.path.insert(0, f"{SNAP}/local_phaseB_rr_qrm")          # a402 = base MinMax path + relevant_zero (FWM) ; hash-identical to training_code/formal_code_factorial250_v1/formal_qrm_v1
import quant_attention as qa

T, K, tau = 7, 4, 6.0
causal = torch.ones(T, T, dtype=torch.bool).tril()
row = T - 1                                                  # last row has all T keys valid

def closed_form_lerp_minmax(s, gw):
    """dL/ds for w = LERP_minmax(s) given dL/dw (gw), full range gradient."""
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

def closed_form_lerp_qrm(s, gw):
    m = s.max(); im = s.argmax(); h = tau / K
    z = s - m; inwin = z >= -tau
    zc = z.clamp(-tau, 0)
    r = torch.floor((zc + tau) / h).clamp(0, K - 1)
    eA, eB = torch.exp(-tau + r * h), torch.exp(-tau + (r + 1) * h)
    t = ((zc - (-tau + r * h)) / h).clamp(0, 1)
    w = torch.where(inwin, (1 - t) * eA + t * eB, torch.zeros_like(z))
    Delta = torch.where(inwin, (eB - eA) / h, torch.zeros_like(z))
    g = gw * Delta
    g[im] += -(gw * Delta).sum()
    return w, g

ok = True
for trial in range(200):
    att = (torch.randn(1, 1, T, T) * 4).requires_grad_(True)
    up = torch.randn(T)                                      # dL/dP for the last row
    for mode in ("minmax", "qrm"):
        for est in ("lerp", "weight_ste", "prob_ste"):
            a = att.detach().clone().requires_grad_(True)
            if mode == "minmax":
                recon = lambda ap, approx: qa._adaptive_interval_reconstruct(ap, causal, K, approx, return_level_idx=True, range_grad="full")[0]
            else:
                recon = lambda ap, approx: qa._relevant_zero_reconstruct(ap, causal, K, approx, tau=tau, return_level_idx=True, range_grad="full")[0]
            if est == "prob_ste":
                with torch.no_grad():
                    wh = recon(a, "nearest"); Ph = wh / wh.sum(-1, keepdim=True)
                wl = recon(a, "lerp"); Pl = wl / wl.sum(-1, keepdim=True)
                P = Pl + (Ph - Pl).detach()
            else:
                w = recon(a, "lerp" if est == "lerp" else "nearest")
                P = w / w.sum(-1, keepdim=True)
            (P[0, 0, row] * up).sum().backward()
            auto = a.grad[0, 0, row]
            s = att.detach()[0, 0, row]
            cf = closed_form_lerp_minmax if mode == "minmax" else closed_form_lerp_qrm
            wl_cf, _ = cf(s, torch.zeros(T))
            if est == "weight_ste":                           # normalization Jacobian at HARD weights
                wh_row = recon(att.detach(), "nearest")[0, 0, row]
                Wn, Pn = wh_row.sum(), wh_row / wh_row.sum()
                gw = (up - (Pn * up).sum()) / Wn
            else:                                             # LERP and Prob-STE: Jacobian at LERP weights
                Wl, Pl_ = wl_cf.sum(), wl_cf / wl_cf.sum()
                gw = (up - (Pl_ * up).sum()) / Wl
            _, g = cf(s, gw)
            err = (g - auto).abs().max().item()
            if err > 1e-9:
                ok = False; print(mode, est, trial, err)
            zero_sum = auto.sum().abs().item()
            if zero_sum > 1e-9:
                ok = False; print("zero-sum violated", mode, est, trial, zero_sum)
print("ALL CLOSED-FORM GRADIENTS MATCH AUTOGRAD (and row gradients sum to 0)" if ok else "MISMATCH")
