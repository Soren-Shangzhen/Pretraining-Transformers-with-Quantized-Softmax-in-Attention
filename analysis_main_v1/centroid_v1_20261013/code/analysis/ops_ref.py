"""Analysis-side operator library for centroid_v1 (one place for every rule used by O1/O2).

Cell indices and the Centroid / Centroid-2 representatives come from the frozen training root
(code/centroid_root/quant_attention.py: _centroid_cells, _centroid_reconstruct); LERP and the Nearest STE backwards
come from the unchanged existing functions of the same root.  Rules:

    exact            softmax
    nearest          hard Nearest forward computed strictly from the level index: w_j = exp(b_{l_j}) (no STE
                     arithmetic in the forward); zero tail under FW; degenerate MinMax rows -> 1
    nearest_wste     Weight-STE  w = w_L + sg(w_N - w_L)               (existing root expression)
    nearest_pste     Prob-STE    P = P_L + sg(P_N - P_L), P_N strict   (as the strict supplement roots)
    lerp             LERP, autograd through the calibration (range_grad full)
    centroid         cell mean of exp(z) on the K+1 Nearest cells
    centroid2        exp(mean z + var z / 2) on the K+1 Nearest cells
    centroid0        exp(mean z) on the K+1 Nearest cells (the zero-variance term of Centroid-2)
    centroid_int     Centroid on the K calibration intervals

cal: "minmax" | "fw" (fixed window tau, zero tail).  All functions take scores att (B,H,T,T) without causal fill.
"""
import importlib
import math
import os
import sys

import torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))  # repository root

C = (KI_REPO + "/analysis_main_v1/centroid_v1_20261013")
ROOT = os.path.join(C, "code", "centroid_root")


def import_root(root=ROOT):
    for m in ("model", "quant_attention"):
        sys.modules.pop(m, None)
    sys.path.insert(0, root)
    try:
        mm = importlib.import_module("model")
        qa = importlib.import_module("quant_attention")
    finally:
        sys.path.remove(root)
    return mm, qa


MM, QA = import_root()

RULES_FWD = ("exact", "nearest", "lerp", "centroid", "centroid2", "centroid0", "centroid_int")
RULES_GRAD = ("exact", "nearest_wste", "nearest_pste", "lerp", "centroid", "centroid2", "centroid0", "centroid_int")


def _rm(cal):
    return {"minmax": "minmax", "fw": "relevant_zero"}[cal]


def causal_mask(T, device):
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def exact_P(att, causal):
    T = att.shape[-1]
    valid = causal.view(1, 1, T, T)
    m = att.masked_fill(~valid, -torch.inf)
    z = m - m.amax(dim=-1, keepdim=True)
    w = torch.exp(z)
    return w / w.sum(dim=-1, keepdim=True)


def nearest_hard_w(att, causal, K, cal, tau=6.0):
    with torch.no_grad():
        bins, level, span, tail, member, degenerate, z_lo, h = QA._centroid_cells(att, causal, K, _rm(cal), tau)
        lv = level.clamp_min(0).to(att.dtype)
        w = torch.exp(z_lo + lv * h)
        w = torch.where(level >= 0, w, torch.zeros_like(w))
        if degenerate is not None:
            T = att.shape[-1]
            w = torch.where(degenerate & causal.view(1, 1, T, T), torch.ones_like(w), w)
    return w


def centroid0_w(att, causal, K, cal, tau=6.0):
    """exp(mean z) per Nearest cell (autograd through the members' scores)."""
    B, H, T, _ = att.shape
    valid = causal.view(1, 1, T, T)
    bins, level, span, tail, member, degenerate, z_lo, h = QA._centroid_cells(att, causal, K, _rm(cal), tau)
    member = member & (level >= 0)
    nc = K + 1
    idx = torch.where(member, level, torch.full_like(level, nc))
    row_max = att.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
    z = torch.where(member, att - row_max, torch.zeros_like(att))
    slots = att.new_zeros(B, H, T, nc + 1)
    with torch.no_grad():
        cnt = slots.scatter_add(-1, idx, member.to(att.dtype)).clamp_min(1.0)
    mean_z = slots.scatter_add(-1, idx, z) / cnt
    w = torch.where(member, torch.exp(mean_z).gather(-1, idx), torch.zeros_like(z))
    if degenerate is not None:
        w = torch.where(degenerate, torch.ones_like(w), w)
    return w.masked_fill(~valid, 0.0)


def operator_P(att, causal, rule, cal=None, K=None, tau=6.0):
    """Normalized attention of `rule` with the autograd semantics used in training."""
    if rule == "exact":
        return exact_P(att, causal)
    rm = _rm(cal)
    if rule == "nearest":
        w = nearest_hard_w(att, causal, K, cal, tau)
    elif rule == "nearest_wste":
        w = _existing(att, causal, K, rm, "nearest", tau)
    elif rule == "nearest_pste":
        wl = _existing(att, causal, K, rm, "lerp", tau)
        PL = wl / wl.sum(-1, keepdim=True)
        wn = nearest_hard_w(att, causal, K, cal, tau)
        PN = wn / wn.sum(-1, keepdim=True)
        return PL + (PN - PL).detach()
    elif rule == "lerp":
        w = _existing(att, causal, K, rm, "lerp", tau)
    elif rule in ("centroid", "centroid2"):
        w = QA._centroid_reconstruct(att, causal, K, rule, range_mode=rm, tau=tau, cells="nearest",
                                     range_grad="full")[0]
    elif rule == "centroid_int":
        w = QA._centroid_reconstruct(att, causal, K, "centroid", range_mode=rm, tau=tau, cells="interval",
                                     range_grad="full")[0]
    elif rule == "centroid0":
        w = centroid0_w(att, causal, K, cal, tau)
    else:
        raise ValueError(rule)
    return w / w.sum(dim=-1, keepdim=True)


def _existing(att, causal, K, rm, approx, tau):
    if rm == "minmax":
        return QA._adaptive_interval_reconstruct(att, causal, K, approx, range_grad="full")[0]
    return QA._relevant_zero_reconstruct(att, causal, K, approx, tau=tau, range_grad="full")[0]


# ---------------------------------------------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------------------------------------------

def load_gpt(ckpt_path, device="cuda"):
    """Build the 124M GPT of the centroid root in exact_fp32_softmax mode and load any suite-D state dict."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    c = ck["config"]
    cfg = MM.GPTConfig(block_size=int(c["block_size"]), n_layer=int(c["n_layer"]), n_head=int(c["n_head"]),
                       n_embd=int(c["n_embd"]), attn_mode="exact_fp32_softmax")
    model = MM.GPT(cfg)
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, c


def set_operator(model, fn):
    """fn(att, causal) -> P, run inside the fp32 no-autocast region; None restores the exact operator."""
    import types

    def make(attn):
        def _op(self, q, k, v):
            B, H, T, D = q.shape
            with torch.autocast(device_type=q.device.type, enabled=False):
                qf, kf, vf = q.float(), k.float(), v.float()
                att = (qf @ kf.transpose(-2, -1)) / math.sqrt(D)
                causal = causal_mask(T, q.device)
                P = exact_P(att, causal) if fn is None else fn(att, causal)
                y = P @ vf
            return y.to(q.dtype)
        return types.MethodType(_op, attn)

    for blk in model.transformer.h:
        blk.attn._exact_fp32_softmax = make(blk.attn)
