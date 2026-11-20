#!/usr/bin/env python3
"""O2: row-level and gradient fidelity of every rule on the score rows of frozen 124M checkpoints.

For each model the first N validation blocks are run forward and backward (LM loss) through the model's OWN training
operator; every attention layer supplies S (fp32 scores), Q, K, V and the upstream gradient dY of its output.  Then,
for each rule x calibration x K, the local operator is applied to the same S, V and contracted with the same dY:

  forward:  TV(P_hat, P_exact) per row, cell-level mass error on the Nearest cells (+ FW tail), relative output error
  backward: cosine and relative L2 of dS, dQ = dS K / sqrt(d), dK = dS^T Q / sqrt(d), dV = P_hat^T dY against the
            exact softmax VJP at the same S; norm ratio ||dS|| / ||dS_exact||
  exact-gradient structure: ||p_j (g_j - gbar_b)||^2 / ||dS_exact||^2, the within-cell contrast that a
            cell-averaged backward cannot represent (not an orthogonal split, so the ratio can exceed 1), and the FW
            tail share; g = dY V^T
  representatives (Nearest cells with >= 2 members): lme - grid point (interior cells, against h^2/24),
            Centroid-2 log gap (zbar + var/2 - lme), share of cells whose Centroid-2 value exceeds the largest member
            weight; unweighted and exact-mass-weighted
Rows i >= 1 (row 0 has one key).  Output: results/o2_rows_grads/<model>.json (per layer and layer means).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
import ops_ref  # noqa: E402
from common import ValBlocks, sha256  # noqa: E402
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))  # repository root

TR = KI_REPO
PB = f"{TR}/training_code/phaseB_base"
VAL = f"{PB}/data/wikitext103/val.bin"
OUT = os.environ.get("O2_OUT", os.path.join(ops_ref.C, "results", "o2_rows_grads"))
KS = tuple(int(k) for k in os.environ.get("O2_KS", "2,4,8,16,32,64").split(","))
CALS = ("minmax", "fw")
GRAD_RULES = ("nearest_wste", "nearest_pste", "lerp", "centroid", "centroid2", "centroid0", "centroid_int")


def models():
    import csv
    man = {(r["cond"], int(r["seed"])): r for r in csv.DictReader(
        open(f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv"), delimiter="\t")}
    m = {}
    for t in (20, 40, 60, 80, 100):
        m[f"Exact_s7_{t:03d}m"] = (man[("Exact", 7)][f"checkpoint_{t}m"], ("exact", None, None))
    for s in (42, 253, 1337, 2026):
        m[f"Exact_s{s}_100m"] = (man[("Exact", s)]["checkpoint_100m"], ("exact", None, None))
    m["LERP_K4_s7_100m"] = (man[("LERP K4", 7)]["checkpoint_100m"], ("lerp", "minmax", 4))
    m["MinMax-Prob_K4_s7_100m"] = (man[("MinMax-Prob K4", 7)]["checkpoint_100m"], ("nearest_pste", "minmax", 4))
    m["FWM-Prob_K4_s7_100m"] = (man[("QRM-Prob K4", 7)]["checkpoint_100m"], ("nearest_pste", "fw", 4))
    m["FWM-Weight_K4_s7_100m"] = (man[("QRM-Weight K4", 7)]["checkpoint_100m"], ("nearest_wste", "fw", 4))
    for run, op in (("cv1_s7_fw_centroid_k4_tau6_100m", ("centroid", "fw", 4)),
                    ("cv1_s7_minmax_centroid_k4_100m", ("centroid", "minmax", 4)),
                    ("cv1_s7_fw_centroid2_k4_tau6_100m", ("centroid2", "fw", 4)),
                    ("cv1_s7_minmax_centroid2_k4_100m", ("centroid2", "minmax", 4))):
        for t, stem in ((20, "ckpt_step000153_20.1m"), (100, "ckpt_step000763_100.0m")):
            m[f"{run}_{t:03d}m"] = (f"{PB}/centroid_runs_20261013/{run}/{stem}.pt", op)
    return m


def capture(model, x, y, own):
    """Forward/backward through the model's own operator; returns per-layer dict of S, Q, K, V, dY."""
    import types
    store = {}
    rule, cal, K = own

    def make(li, attn):
        def _op(self, q, k, v):
            B, H, T, D = q.shape
            with torch.autocast(device_type=q.device.type, enabled=False):
                qf, kf, vf = q.float(), k.float(), v.float()
                att = (qf @ kf.transpose(-2, -1)) / math.sqrt(D)
                causal = ops_ref.causal_mask(T, q.device)
                P = ops_ref.operator_P(att, causal, rule, cal, K)
                out = P @ vf
                rec = dict(S=att.detach(), Q=qf.detach(), K=kf.detach(), V=vf.detach())
                store[li] = rec
                out.register_hook(lambda g, rec=rec: rec.__setitem__("dY", g.detach()))
            return out.to(q.dtype)
        return types.MethodType(_op, attn)

    for li, blk in enumerate(model.transformer.h):
        blk.attn._exact_fp32_softmax = make(li, blk.attn)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return store, float(loss)


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@torch.no_grad()
def cell_stats(S, causal, K, cal, P, G, dS_e, rowsel):
    """Exact-gradient structure and representative statistics on the Nearest cells of (cal, K)."""
    B, H, T, _ = S.shape
    bins, level, span, tail, member, degenerate, z_lo, h = ops_ref.QA._centroid_cells(S, causal, K, ops_ref._rm(cal), 6.0)
    member = member & (level >= 0)
    if degenerate is not None:
        member = member & ~degenerate
    nc = K + 1
    idx = torch.where(member, level, torch.full_like(level, nc))
    valid = causal.view(1, 1, T, T)
    zf = S - S.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
    z = torch.where(member, zf, torch.zeros_like(zf))
    slots = S.new_zeros(B, H, T, nc + 1)
    mf = member.to(S.dtype)
    cnt = slots.scatter_add(-1, idx, mf)
    gsum = slots.scatter_add(-1, idx, torch.where(member, G, torch.zeros_like(G)))
    gbar = (gsum / cnt.clamp_min(1)).gather(-1, idx)
    contrast = torch.where(member, P * (G - gbar), torch.zeros_like(P))
    rs = rowsel.view(1, 1, T, 1)
    den = (dS_e * rs).square().sum()
    # p_j (gbar_b - <g>) + p_j (g_j - gbar_b) = dS_exact is not an orthogonal split, so this is a norm ratio
    out = {"contrast_norm2_ratio": float((contrast * rs).square().sum() / den)}
    if tail is not None:
        out["tail_share"] = float((torch.where(tail, dS_e, torch.zeros_like(dS_e)) * rs).square().sum() / den)
        out["exact_tail_mass_mean"] = float((torch.where(tail, P, torch.zeros_like(P)).sum(-1))[..., 1:].mean())
    # representative statistics (float64 per cell)
    zd = z.double()
    e = torch.where(member, torch.exp(zd), torch.zeros_like(zd))
    s64 = slots.double()
    cnt64 = cnt.double()
    se = s64.scatter_add(-1, idx, e)
    s1 = s64.scatter_add(-1, idx, zd)
    s2 = s64.scatter_add(-1, idx, zd * zd)
    zmax = s64.fill_(-1e30).scatter_reduce(-1, idx, torch.where(member, zd, torch.full_like(zd, -1e30)), "amax")
    A = s64.new_zeros(s64.shape).scatter_add(-1, idx, torch.where(member, P.double(), torch.zeros_like(zd)))
    keep = (cnt64 >= 2)
    keep[..., nc] = False
    keep[..., :1, :] = False  # row 0
    n = cnt64.clamp_min(1)
    lme = torch.log((se / n).clamp_min(1e-300))
    zbar = s1 / n
    var = (s2 / n - zbar * zbar).clamp_min(0)
    gap2 = zbar + var / 2 - lme
    gap0 = zbar - lme
    over = (zbar + var / 2) > zmax
    grid = torch.arange(nc + 1, device=S.device, dtype=torch.float64)
    centre = (z_lo.double() + grid * h.double())
    if centre.shape[-2] == 1:
        centre = centre.expand_as(lme)
    d_centre = lme - centre
    interior = keep.clone()
    interior[..., 0] = False
    interior[..., K] = False
    hh = h.double().expand_as(lme)
    w = A * keep
    W = w.sum()

    def m(x, mask):
        return float(x[mask].mean()) if mask.any() else float("nan")

    def wm(x, mask):
        ww = A * mask
        return float((x * ww).sum() / ww.sum()) if mask.any() else float("nan")

    out.update(
        cells_ge2=int(keep.sum()), mass_in_cells_ge2=float(W / A[..., :nc].sum()),
        lme_minus_gridpoint_interior_mean=m(d_centre, interior),
        h2_over_24_interior_mean=m(hh * hh / 24, interior),
        lme_minus_gridpoint_over_h2_24_interior_median=float((d_centre / (hh * hh / 24))[interior].median())
        if interior.any() else float("nan"),
        c2_log_gap_mean=m(gap2, keep), c2_log_gap_abs_mean=m(gap2.abs(), keep),
        c2_log_gap_abs_massw=wm(gap2.abs(), keep),
        c0_log_gap_mean=m(gap0, keep), c0_log_gap_abs_massw=wm(gap0.abs(), keep),
        c2_exceeds_max_member_share=m(over.double(), keep), c2_exceeds_max_member_massw=wm(over.double(), keep),
    )
    return out


def analyse_layer(rec, KS_, cals):
    S, Q, Kt, V, dY = rec["S"], rec["Q"], rec["K"], rec["V"], rec["dY"]
    B, H, T, D = Q.shape
    causal = ops_ref.causal_mask(T, S.device)
    rowsel = torch.arange(T, device=S.device) >= 1
    rs4 = rowsel.view(1, 1, T, 1)
    a = S.clone().requires_grad_(True)
    P = ops_ref.exact_P(a, causal)
    y = P @ V
    dS_e, = torch.autograd.grad((y * dY).sum(), a)
    P = P.detach()
    y = y.detach()
    sc = 1.0 / math.sqrt(D)
    dQ_e, dK_e, dV_e = dS_e @ Kt * sc, dS_e.transpose(-1, -2) @ Q * sc, P.transpose(-1, -2) @ dY
    G = dY @ V.transpose(-1, -2)
    res = {}
    for cal in cals:
        for K in KS_:
            key = f"{cal}|K{K}"
            res[key] = {"cells": cell_stats(S, causal, K, cal, P, G, dS_e, rowsel)}
            with torch.no_grad():
                _, level, _, _, _, _, _, _ = ops_ref.QA._centroid_cells(S, causal, K, ops_ref._rm(cal), 6.0)
                nc = K + 1
                idx = torch.where(level >= 0, level, torch.full_like(level, nc))
                valid = causal.view(1, 1, T, T).expand_as(level)
                idx = torch.where(valid, idx, torch.full_like(idx, nc + 1))
            for rule in GRAD_RULES:
                a = S.clone().requires_grad_(True)
                Ph = ops_ref.operator_P(a, causal, rule, cal, K)
                yh = Ph @ V
                dS, = torch.autograd.grad((yh * dY).sum(), a)
                with torch.no_grad():
                    Ph = Ph.detach()
                    # Centroid-2 can overflow fp32 in very wide cells (exp(var/2)); such rows are counted and
                    # every other metric of the configuration is then reported over all rows as NaN
                    nonfinite = (~torch.isfinite(Ph).all(-1))[..., 1:].float().mean()
                    dS = torch.nan_to_num(dS)
                    dQ, dK, dV = dS @ Kt * sc, dS.transpose(-1, -2) @ Q * sc, Ph.transpose(-1, -2) @ dY
                    tv = (0.5 * (Ph - P).abs().sum(-1))[..., 1:].mean()
                    # cell-level mass error on the Nearest cells of (cal, K), FW tail as its own slot
                    dm = S.new_zeros(B, H, T, nc + 2).scatter_add(-1, idx, Ph - P)[..., :nc + 1]
                    mass_err = dm.abs().sum(-1)[..., 1:].mean()
                    # in-window cells only, exact P renormalized to the window (isolates the representative)
                    Pw = S.new_zeros(B, H, T, nc + 2).scatter_add(-1, idx, P)
                    Phw = S.new_zeros(B, H, T, nc + 2).scatter_add(-1, idx, Ph)
                    Pw_n = Pw[..., :nc] / Pw[..., :nc].sum(-1, keepdim=True).clamp_min(1e-30)
                    Phw_n = Phw[..., :nc] / Phw[..., :nc].sum(-1, keepdim=True).clamp_min(1e-30)
                    mass_err_win = (Phw_n - Pw_n).abs().sum(-1)[..., 1:].mean()
                    res[key][rule] = dict(
                        nonfinite_row_frac=float(nonfinite),
                        tv=float(tv), cell_mass_err=float(mass_err), cell_mass_err_window=float(mass_err_win),
                        out_rel=rel(yh.detach() * rs4, y * rs4),
                        dS_cos=cos(dS * rs4, dS_e * rs4), dS_rel=rel(dS * rs4, dS_e * rs4),
                        dS_norm_ratio=float((dS * rs4).norm() / (dS_e * rs4).norm()),
                        dQ_cos=cos(dQ, dQ_e), dQ_rel=rel(dQ, dQ_e), dK_cos=cos(dK, dK_e), dK_rel=rel(dK, dK_e),
                        dV_cos=cos(dV, dV_e), dV_rel=rel(dV, dV_e))
    return res


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    n_blocks = int(os.environ.get("O2_BLOCKS", "4"))
    blocks = ValBlocks(VAL, 1024)
    os.makedirs(OUT, exist_ok=True)
    want = sys.argv[1:]
    for name, (ck, own) in models().items():
        if want and name not in want:
            continue
        out = os.path.join(OUT, f"{name}.json")
        if os.path.exists(out) or not os.path.exists(ck):
            continue
        t0 = time.time()
        model, cfg = ops_ref.load_gpt(ck)
        per_layer = None
        losses = []
        for b in range(n_blocks):
            x, yt = blocks.batch(b, b + 1, "cuda")
            store, loss = capture(model, x, yt, own)
            losses.append(loss)
            for li in sorted(store):
                r = analyse_layer(store[li], KS, CALS)
                if per_layer is None:
                    per_layer = {l: [] for l in sorted(store)}
                per_layer[li].append(r)
            del store
            torch.cuda.empty_cache()
        # mean over blocks, then over layers
        def avg(items):
            # plain mean: a NaN in any layer/block (non-finite Centroid-2 rows) propagates to the summary
            if isinstance(items[0], dict):
                return {k: avg([it[k] for it in items]) for k in items[0]}
            return float(np.mean(items))
        layers = {str(li): avg(v) for li, v in per_layer.items()}
        summary = avg(list(layers.values()))
        json.dump(dict(model=name, checkpoint=ck, own_operator=own, blocks=n_blocks, loss_blocks=losses,
                       ops_ref_sha256=sha256(ops_ref.__file__), script_sha256=sha256(__file__),
                       layer_mean=summary, per_layer=layers, seconds=round(time.time() - t0, 1)),
                  open(out, "w"), indent=1)
        print(f"{name} done {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
