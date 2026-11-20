#!/usr/bin/env python3
"""Checks for the addendum root code/centroid_root_v2 (mode nearest_centroid_bwd: hard Nearest forward, Centroid
backward) before its single training run.

V1  regression: every mode of the frozen centroid_root (softmax, exact_fp32_softmax, LERP, Nearest, Centroid,
    Centroid-2; MinMax/FW; detach/full; K 4/8) gives bitwise-identical loss and parameter gradients in v2 (CPU fp32)
V2  strict forward: whole-GPT loss of nearest_centroid_bwd (FW tau 6, K 4) equals the E2 strict FWM-Prob root
    (supplement code/e2_fwm_probste_strict, NEAREST_SURROGATE=prob) bitwise, CPU fp32 and GPU; module-level P equal
V3  backward (fp64, module level): dL/dS equals the closed form e_j (gbar_b - <g>_{P_C}) / W_C with g = dL/dP taken
    at the hard forward, and equals the VJP of the pure Centroid probabilities with the same g
V4  backward differs from the strict Prob-STE backward (sanity: the surrogate really changed) and the forward
    does not depend on the surrogate
V5  GPU step 0 with the training recipe (seed 7, first micro-batch): loss, finite gradients, layer-0 Q/K grad norm
Output: checks/check_v2_nearest_centroid_bwd.json
"""
import importlib
import json
import math
import os
import sys

import numpy as np
import torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))  # repository root

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
V1 = os.path.join(C, "code", "centroid_root")
V2 = os.path.join(C, "code", "centroid_root_v2")
E2 = KI_REPO + "/analysis_main_v1/supplement_four_runs_20260923/code/e2_fwm_probste_strict"
DATA = KI_REPO + "/training_code/phaseB_base/data/wikitext103"


def import_root(root):
    for m in ("model", "quant_attention"):
        sys.modules.pop(m, None)
    sys.path.insert(0, root)
    try:
        mm = importlib.import_module("model")
        qa = importlib.import_module("quant_attention")
    finally:
        sys.path.remove(root)
    return mm, qa


def gpt_loss_grads(root, kw, x, y, device="cpu", surrogate=None, scale_qk=12.0, cfg_kw=None):
    mm, qa = import_root(root)
    if surrogate is not None:
        qa.NEAREST_SURROGATE = surrogate
    torch.manual_seed(1234)
    cfg = mm.GPTConfig(**(cfg_kw or dict(block_size=64, n_layer=2, n_head=4, n_embd=64)), **kw)
    model = mm.GPT(cfg).to(device)
    if scale_qk:
        with torch.no_grad():
            for blk in model.transformer.h:
                blk.attn.c_attn.weight[:2 * cfg.n_embd].mul_(scale_qk)
    _, loss = model(x.to(device), y.to(device))
    loss.backward()
    return loss.detach().cpu(), {n: p.grad.detach().cpu() for n, p in model.named_parameters()}


def main():
    res = {}
    x = torch.randint(0, 50304, (2, 64), generator=torch.Generator().manual_seed(5))
    y = torch.randint(0, 50304, (2, 64), generator=torch.Generator().manual_seed(6))
    # ---------------- V1 ----------------
    modes = [dict(attn_mode="softmax"), dict(attn_mode="exact_fp32_softmax")]
    for rm in ("minmax", "relevant_zero"):
        for approx in ("lerp", "nearest", "centroid", "centroid2"):
            for rg in ("detach", "full"):
                for K in (4, 8):
                    modes.append(dict(attn_mode="quant", attn_K=K, attn_approx=approx, attn_range_grad=rg,
                                      attn_range_mode=rm, attn_range_tau=6.0))
    ok = True
    for md in modes:
        l1, g1 = gpt_loss_grads(V1, md, x, y)
        l2, g2 = gpt_loss_grads(V2, md, x, y)
        ok &= torch.equal(l1, l2) and all(torch.equal(g1[n], g2[n]) for n in g1)
    res["V1_regression_all_modes_bitwise"] = dict(n_modes=len(modes), equal=bool(ok))
    assert ok
    # ---------------- V2 ----------------
    kw_new = dict(attn_mode="quant", attn_K=4, attn_approx="nearest_centroid_bwd", attn_range_grad="full",
                  attn_range_mode="relevant_zero", attn_range_tau=6.0)
    kw_e2 = dict(attn_mode="quant", attn_K=4, attn_approx="nearest", attn_range_grad="full",
                 attn_range_mode="relevant_zero", attn_range_tau=6.0)
    v2 = {}
    for dev in ("cpu", "cuda"):
        if dev == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        ln, gn = gpt_loss_grads(V2, kw_new, x, y, dev)
        le, ge = gpt_loss_grads(E2, kw_e2, x, y, dev, surrogate="prob")
        lc, gc = gpt_loss_grads(V2, dict(kw_new, attn_approx="centroid"), x, y, dev)
        same_V_grad = torch.equal(gn["transformer.h.1.attn.c_attn.weight"][-64:],
                                  ge["transformer.h.1.attn.c_attn.weight"][-64:])
        v2[dev] = dict(loss_new=float(ln), loss_e2_strict_prob=float(le), loss_centroid_forward=float(lc),
                       loss_bitwise_equal_e2=bool(torch.equal(ln, le)),
                       qk_grad_differs_from_e2=bool(not torch.equal(gn["transformer.h.0.attn.c_attn.weight"][:128],
                                                                    ge["transformer.h.0.attn.c_attn.weight"][:128])),
                       last_layer_v_grad_equal_e2=bool(same_V_grad))
        assert v2[dev]["loss_bitwise_equal_e2"], v2[dev]
    res["V2_strict_forward_vs_E2_strict_prob"] = v2
    # ---------------- V3 / V4 (fp64 module level) ----------------
    torch.set_default_dtype(torch.float64)
    _, qa2 = import_root(V2)
    rng = np.random.default_rng(11)
    T, K = 24, 4
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    worst_cf, worst_vjp, diff_pste = 0.0, 0.0, 0.0
    for scale in (0.7, 2.0, 4.0):
        att = torch.from_numpy(rng.normal(0, scale, size=(1, 3, T, T))).requires_grad_(True)
        G = torch.from_numpy(rng.normal(size=(1, 3, T, T)))
        w_c, *_ = qa2._centroid_reconstruct(att, causal, K, "centroid", range_mode="relevant_zero", tau=6.0)
        P_c = w_c / w_c.sum(-1, keepdim=True)
        w_n, _, _, level, tail = qa2._nearest_hard_from_cells(att, causal, K, "relevant_zero", 6.0)
        P_n = w_n / w_n.sum(-1, keepdim=True)
        P = P_n.detach() + (P_c - P_c.detach())
        assert torch.equal(P.detach(), P_n)
        gS, = torch.autograd.grad((P * G).sum(), att, retain_graph=True)
        gV, = torch.autograd.grad(P_c, att, grad_outputs=G)
        worst_vjp = max(worst_vjp, float((gS - gV).abs().max() / gV.abs().max()))
        # closed form
        Wc = w_c.detach().sum(-1, keepdim=True)
        Pc = P_c.detach()
        gmean = (Pc * G).sum(-1, keepdim=True)
        zrow = att.detach() - att.detach().masked_fill(~causal, -torch.inf).amax(-1, keepdim=True)
        e = torch.exp(zrow)
        cf = torch.zeros_like(gS)
        mem = (level >= 0)
        for c in range(K + 1):
            mc = mem & (level == c)
            cnt = mc.sum(-1, keepdim=True).clamp_min(1)
            gbar = torch.where(mc, G, torch.zeros_like(G)).sum(-1, keepdim=True) / cnt
            cf = torch.where(mc, e * (gbar - gmean) / Wc, cf)
        worst_cf = max(worst_cf, float((gS - cf).abs().max() / gS.abs().max()))
        # Prob-STE (strict) backward for comparison
        w_l, *_ = qa2._relevant_zero_reconstruct(att, causal, K, "lerp", tau=6.0, range_grad="full")
        P_l = w_l / w_l.sum(-1, keepdim=True)
        gP, = torch.autograd.grad(P_l, att, grad_outputs=G)
        diff_pste = max(diff_pste, float((gS - gP).abs().max() / gP.abs().max()))
    res["V3_backward_equals_centroid_vjp_max_rel"] = worst_vjp
    res["V3_backward_equals_closed_form_max_rel"] = worst_cf
    res["V4_backward_differs_from_prob_ste_max_rel"] = diff_pste
    assert worst_vjp < 1e-12 and worst_cf < 1e-12 and diff_pste > 1e-2
    torch.set_default_dtype(torch.float32)
    # ---------------- V5 step 0 on GPU ----------------
    torch.backends.cuda.matmul.allow_tf32 = True
    data = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    ix = np.random.default_rng(7).integers(0, len(data) - 1025, 1)
    xb = torch.from_numpy(np.stack([data[i:i + 1024] for i in ix]).astype(np.int64)).cuda()
    yb = torch.from_numpy(np.stack([data[i + 1:i + 1025] for i in ix]).astype(np.int64)).cuda()
    res["V5_step0"] = {}
    for name, root, kw, sur in (("nearest_centroid_bwd (v2)", V2, kw_new, None),
                                ("FW-Prob strict (E2 root)", E2, kw_e2, "prob"),
                                ("FW-Centroid (v2)", V2, dict(kw_new, attn_approx="centroid"), None),
                                ("exact_fp32_softmax (v2)", V2, dict(attn_mode="exact_fp32_softmax"), None)):
        mm, qa = import_root(root)
        if sur is not None:
            qa.NEAREST_SURROGATE = sur
        torch.manual_seed(7)
        np.random.seed(7)
        model = mm.GPT(mm.GPTConfig(block_size=1024, n_layer=12, n_head=12, n_embd=768, **kw)).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        loss.backward()
        res["V5_step0"][name] = dict(
            loss=float(loss), grads_finite=bool(all(torch.isfinite(p.grad).all() for p in model.parameters())),
            qk_grad_l2_by_layer=[round(float(b.attn.c_attn.weight.grad[:2 * 768].norm()), 5)
                                 for b in model.transformer.h])
        del model
        torch.cuda.empty_cache()
    out = os.path.join(C, "checks", "check_v2_nearest_centroid_bwd.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
