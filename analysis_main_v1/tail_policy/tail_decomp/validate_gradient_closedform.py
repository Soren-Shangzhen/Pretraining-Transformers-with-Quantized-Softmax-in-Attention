#!/usr/bin/env python3
"""Check 3 proper: closed-form backward vs autograd, float64, 200 rows, resolved x {Weight, Prob}.

Closed form (spec §3.4).  With gamma_i = dL/dw_i and Delta_i = dw_i/dz_i (the LERP secant slope of
the segment key i falls in):

    dL/ds_i = gamma_i * Delta_i  -  1[i = argmax] * sum_j gamma_j * Delta_j

because z_i = s_i - m and dz_i/ds_j = delta_ij - 1[j = argmax].

  Weight-STE : w has the hard VALUE and the LERP GRADIENT, P = w/W, so
               gamma_k = (G_k - sum_i G_i P_i) / W          with forward P, W
  Prob-STE   : P = P_lerp + sg(P_hard - P_lerp), so the Jacobian is P_lerp's:
               gamma_k = (G_k - sum_i G_i P_lerp_i) / W_lerp

  Delta_i    = (e^{b_{r+1}} - e^{b_r}) / h   for every key that is reconstructed on a segment
               (in-window under every policy; ALSO the tail under `resolved`)
             = 0                              for a `zero` or `floor` tail key (w is constant there)

Criterion: max |closed - autograd| < 1e-9 over 200 rows, both surrogates.
"""
import importlib, sys
import torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


PB = (KI_REPO + "/"
      "training_code/phaseB_base")
TAU, K = 6.0, 4
H = TAU / K
res = []


def ck(n, ok, d=""):
    res.append(bool(ok)); print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f" — {d}" if d else ""), flush=True)
    return bool(ok)


def load(root):
    for m in ("model", "quant_attention"):
        sys.modules.pop(m, None)
    sys.path.insert(0, f"{PB}/code/{root}")
    try:
        return importlib.import_module("quant_attention")
    finally:
        sys.path.remove(f"{PB}/code/{root}")


def rec(qa, att, causal, approx, policy):
    return qa._relevant_zero_reconstruct(att, causal, K, approx, tau=TAU, return_level_idx=True,
                                         range_grad="full", tail_policy=policy)[0]


def segment_slope(z, valid, policy):
    """Delta_i = dw_i/dz_i under the LERP surrogate, for the segment key i occupies."""
    z_safe = torch.where(valid, z, torch.zeros_like(z))
    in_win = z_safe >= -TAU
    # in-window uses the clamped coordinate, tail (resolved only) uses raw z
    zc = z_safe.clamp(min=-TAU, max=0.0)
    r_in = torch.floor((zc + TAU) / H).clamp(0, K - 1)
    r_t = torch.clamp(torch.floor((z_safe + TAU) / H), max=float(K - 1))
    r = torch.where(in_win, r_in, r_t)
    b_lo = -TAU + r * H
    delta = (torch.exp(b_lo + H) - torch.exp(b_lo)) / H
    if policy == "resolved":
        keep = valid
    else:                      # zero / floor: w is constant on the tail -> slope 0
        keep = valid & in_win
    return torch.where(keep, delta, torch.zeros_like(delta))


def closed_form(qa, att, causal, G, surrogate, policy):
    T = att.shape[-1]
    valid = causal.view(1, 1, T, T)
    with torch.no_grad():
        rowmax = att.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
        z = att - rowmax
        is_max = valid & (att == rowmax)
        # keep exactly one argmax per row (data is continuous, ties have measure zero)
        first_max = is_max & (is_max.cumsum(-1) == 1)
        delta = segment_slope(z, valid.expand_as(z), policy)
        if surrogate == "weight":
            w = rec(qa, att, causal, "nearest", policy)
            W = w.sum(-1, keepdim=True)
            P = w / W
            gamma = (G - (G * P).sum(-1, keepdim=True)) / W
        else:
            w_l = rec(qa, att, causal, "lerp", policy)
            W_l = w_l.sum(-1, keepdim=True)
            P_l = w_l / W_l
            gamma = (G - (G * P_l).sum(-1, keepdim=True)) / W_l
        gd = gamma * delta
        gd = torch.where(valid, gd, torch.zeros_like(gd))
        out = gd - first_max.to(gd.dtype) * gd.sum(-1, keepdim=True)
        return torch.where(valid, out, torch.zeros_like(out))


def autograd_grad(qa, att, causal, G, surrogate, policy):
    a = att.clone().requires_grad_(True)
    if surrogate == "weight":
        w = rec(qa, a, causal, "nearest", policy)
        P = w / w.sum(-1, keepdim=True)
    else:
        with torch.no_grad():
            w_h = rec(qa, a.detach(), causal, "nearest", policy)
            P_h = w_h / w_h.sum(-1, keepdim=True)
        w_l = rec(qa, a, causal, "lerp", policy)
        P_l = w_l / w_l.sum(-1, keepdim=True)
        P = P_l + (P_h - P_l).detach()
    (G * P).sum().backward()
    return a.grad


def main():
    torch.manual_seed(7)
    T, Hh = 50, 4                      # 4 heads x 50 rows = 200 rows
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    valid = causal.view(1, 1, T, T)
    att = torch.randn(1, Hh, T, T, dtype=torch.float64) * 8.0
    G = torch.randn(1, Hh, T, T, dtype=torch.float64)
    n_rows = Hh * T
    z = att - att.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
    print(f"  rows={n_rows}  tail entries={(valid & (z < -TAU)).sum().item()}  "
          f"rows with a tail={((valid & (z < -TAU)).any(-1)).sum().item()}")

    for root, sur in (("rb_tail_weight", "weight"), ("rb_tail_prob", "prob")):
        qa = load(root)
        for policy in ("resolved", "zero"):
            ag = autograd_grad(qa, att, causal, G, sur, policy)
            cf = closed_form(qa, att, causal, G, sur, policy)
            err = (ag - cf)[valid.expand_as(ag)].abs().max().item()
            tag = f"check3 {sur:6s} {policy:8s}"
            ck(f"{tag}: closed form vs autograd < 1e-9", err < 1e-9, f"max |diff| = {err:.3e}")
            rs = ag.masked_fill(~valid, 0.0).sum(-1).abs().max().item()
            ck(f"{tag}: per-row gradient sum ~ 0", rs < 1e-9, f"{rs:.3e}")

    n = sum(res)
    print(f"\nCLOSED-FORM GRADIENT {'PASS' if n == len(res) else 'FAIL'}  ({n}/{len(res)})")
    return n == len(res)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
