#!/usr/bin/env python3
"""FP64 / CPU operator checks for the Centroid and Centroid-2 reconstruction rules (centroid_v1).

C1  forward equals an independent per-row NumPy reference (both rules, both calibrations, both cell modes, K 2..16)
C2  three-level mass conservation: per-cell mass (Centroid exact; Centroid-2 reported), global normalization
C3  autograd score gradient equals the closed form
        Centroid:   dL/ds_j = e_j * mean_{i in b(j)} gamma_i,           gamma_i = (g_i - <g>_P) / W
        Centroid-2: dL/ds_j = w*_b (1 + z_j - zbar_b) * mean_{i in b} gamma_i
C4  central finite differences agree with autograd away from cell boundaries
C5  shift invariance F(s + c1) = F(s) and the row zero-sum gradient identity
C6  calibration decoupling: range_grad full and detach give the same gradient (to rounding)
C7  edge cases: degenerate MinMax row, single-key row, FW row with every other key in the tail, ties at t = 1/2,
    singleton / constant cells reproduce exp(z) exactly
C8  representative properties: Lloyd (mean minimizes within-cell squared weight error), h^2/24 for a uniform cell,
    Centroid-2 exact for a two-point symmetric cell? no -- reported as the measured gap
C9  regression: every pre-existing path (softmax, exact_fp32_softmax, LERP, Nearest; MinMax and FW; detach/full)
    of the new root equals the phaseB_rr source root bitwise (whole GPT forward + all parameter gradients)
Output: checks/check_centroid_ops.json
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
NEW_ROOT = os.path.join(C, "code", "centroid_root")
SRC_ROOT = KI_REPO + "/training_code/phaseB_base/code/phaseB_rr"


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


# ---------------------------------------------------------------------------------------------------------------
# independent NumPy reference (one row at a time)
# ---------------------------------------------------------------------------------------------------------------

def ref_row(s, K, rule, range_mode, tau, cells):
    """s: 1-D float64 valid scores of one causal row. Returns w_hat (unnormalized), cell index (-1 tail), z."""
    s = np.asarray(s, dtype=np.float64)
    M, m = s.max(), s.min()
    z = s - M
    n = len(s)
    if range_mode == "minmax":
        span = M - m
        if span <= 1e-12:
            return np.ones(n), np.zeros(n, dtype=int), z
        h = span / K
        zlo = m - M
        zc = z
        tail = np.zeros(n, dtype=bool)
    else:
        h = tau / K
        zlo = -tau
        zc = np.clip(z, -tau, 0.0)
        tail = z < -tau
    rel = (zc - zlo) / h
    r = np.clip(np.floor(rel), 0, K - 1).astype(int)
    t = np.clip((zc - (zlo + r * h)) / h, 0.0, 1.0)
    lev = r + (t >= 0.5).astype(int)
    cell = lev if cells == "nearest" else r
    cell = np.where(tail, -1, cell)
    w = np.zeros(n)
    for c in np.unique(cell[cell >= 0]):
        idx = cell == c
        if rule == "centroid":
            w[idx] = np.exp(z[idx]).mean()
        else:
            w[idx] = math.exp(z[idx].mean() + z[idx].var() / 2.0)
    return w, cell, z


def random_scores(rng, B, H, T, scale):
    return torch.from_numpy(rng.normal(0.0, scale, size=(B, H, T, T)))


def run_op(qa, att, K, rule, range_mode, tau, cells, range_grad):
    T = att.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    return qa._centroid_reconstruct(att, causal, K, rule, range_mode=range_mode, tau=tau, cells=cells,
                                    range_grad=range_grad)


def normalized_output(qa, att, v, K, rule, range_mode, tau, cells, range_grad):
    w, _, _, _, _ = run_op(qa, att, K, rule, range_mode, tau, cells, range_grad)
    P = w / w.sum(-1, keepdim=True)
    return P, P @ v


def main():
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(20261013)
    _, qa = import_root(NEW_ROOT)
    res = {"new_root": NEW_ROOT}
    tau = 6.0
    combos = [(rule, rm, cells) for rule in ("centroid", "centroid2") for rm in ("minmax", "relevant_zero")
              for cells in ("nearest", "interval")]

    # ---------------- C1 forward vs reference, C2 mass conservation ----------------
    c1_max, c2 = 0.0, {}
    for (rule, rm, cells) in combos:
        mass_err_max, glob_err_max, n_cells_checked = 0.0, 0.0, 0
        for K in (2, 4, 8, 16):
            for scale in (0.5, 2.0, 5.0):
                att = random_scores(rng, 2, 2, 24, scale)
                w, _, _, _, _ = run_op(qa, att, K, rule, rm, tau, cells, "full")
                for b in range(2):
                    for hh in range(2):
                        for i in range(24):
                            s = att[b, hh, i, :i + 1].numpy()
                            wr, cell, z = ref_row(s, K, rule, rm, tau, cells)
                            wt = w[b, hh, i, :i + 1].numpy()
                            c1_max = max(c1_max, float(np.max(np.abs(wt - wr) / np.maximum(wr, 1e-300))))
                            assert np.all(w[b, hh, i, i + 1:].numpy() == 0)
                            e = np.exp(z)
                            span = s.max() - s.min()
                            if rm == "minmax" and span <= 1e-12:
                                continue
                            for c in np.unique(cell[cell >= 0]):
                                idx = cell == c
                                mass_err_max = max(mass_err_max,
                                                   abs(wt[idx].sum() - e[idx].sum()) / e[idx].sum())
                                n_cells_checked += 1
                            Zw = e[cell >= 0].sum()
                            glob_err_max = max(glob_err_max, abs(wt.sum() - Zw) / Zw)
        c2[f"{rule}/{rm}/{cells}"] = dict(cells_checked=n_cells_checked, max_rel_cell_mass_error=mass_err_max,
                                         max_rel_denominator_minus_Z=glob_err_max)
    res["C1_forward_vs_numpy_reference_max_rel"] = c1_max
    res["C2_mass_conservation"] = c2
    assert c1_max < 1e-12, c1_max
    for k, v in c2.items():
        if k.startswith("centroid/"):
            assert v["max_rel_cell_mass_error"] < 1e-12 and v["max_rel_denominator_minus_Z"] < 1e-12, (k, v)

    # ---------------- C3 closed-form gradient, C6 full vs detach ----------------
    c3, c6 = {}, {}
    for (rule, rm, cells) in combos:
        err, dd = 0.0, 0.0
        for K in (2, 4, 8):
            att = random_scores(rng, 1, 2, 20, 2.0).requires_grad_(True)
            G = torch.from_numpy(rng.normal(size=(1, 2, 20, 20)))
            w, _, _, _, _ = run_op(qa, att, K, rule, rm, tau, cells, "full")
            P = w / w.sum(-1, keepdim=True)
            (P * G).sum().backward()
            ga = att.grad.clone()
            att2 = att.detach().clone().requires_grad_(True)
            w2, _, _, _, _ = run_op(qa, att2, K, rule, rm, tau, cells, "detach")
            P2 = w2 / w2.sum(-1, keepdim=True)
            (P2 * G).sum().backward()
            dd = max(dd, float((att2.grad - ga).abs().max() / ga.abs().max()))
            for hh in range(2):
                for i in range(20):
                    s = att[0, hh, i, :i + 1].detach().numpy()
                    wr, cell, z = ref_row(s, K, rule, rm, tau, cells)
                    W = wr.sum()
                    Pr = wr / W
                    g = G[0, hh, i, :i + 1].numpy()
                    gam = (g - (Pr * g).sum()) / W
                    gs = np.zeros(i + 1)
                    span = s.max() - s.min()
                    if not (rm == "minmax" and span <= 1e-12):
                        for c in np.unique(cell[cell >= 0]):
                            idx = cell == c
                            mg = gam[idx].mean()
                            if rule == "centroid":
                                gs[idx] = np.exp(z[idx]) * mg
                            else:
                                gs[idx] = wr[idx][0] * (1.0 + z[idx] - z[idx].mean()) * mg
                    gt = ga[0, hh, i, :i + 1].numpy()
                    # normalized by the largest gradient entry of the whole tensor (rows whose closed form is
                    # exactly zero, e.g. an FW row with only the maximum in the window, carry rounding noise)
                    err = max(err, float(np.abs(gt - gs).max() / float(ga.abs().max())))
                    assert np.all(ga[0, hh, i, i + 1:].numpy() == 0)
        c3[f"{rule}/{rm}/{cells}"] = err
        c6[f"{rule}/{rm}/{cells}"] = dd
        assert err < 1e-10, (rule, rm, cells, err)
        assert dd < 1e-12, (rule, rm, cells, dd)
    res["C3_grad_vs_closed_form_max_rel"] = c3
    res["C6_full_minus_detach_grad_max_rel"] = c6

    # ---------------- C4 finite differences ----------------
    c4 = {}
    for (rule, rm, cells) in combos:
        K = 4
        att = random_scores(rng, 1, 1, 12, 1.5)
        v = torch.from_numpy(rng.normal(size=(1, 1, 12, 5)))
        G = torch.from_numpy(rng.normal(size=(1, 1, 12, 5)))
        a = att.clone().requires_grad_(True)
        _, y = normalized_output(qa, a, v, K, rule, rm, tau, cells, "full")
        (y * G).sum().backward()
        worst, tested = 0.0, 0
        eps = 1e-6
        for i in range(1, 12):
            for j in range(i + 1):
                ap, am = att.clone(), att.clone()
                ap[0, 0, i, j] += eps
                am[0, 0, i, j] -= eps
                # skip perturbations that change any cell membership (or the extrema identity)
                _, cb, _, lb, _ = run_op(qa, att, K, rule, rm, tau, cells, "full")
                _, cbp, _, lbp, tp = run_op(qa, ap, K, rule, rm, tau, cells, "full")
                _, cbm, _, lbm, tm = run_op(qa, am, K, rule, rm, tau, cells, "full")
                if not (torch.equal(cb, cbp) and torch.equal(cb, cbm) and torch.equal(lb, lbp)
                        and torch.equal(lb, lbm)):
                    continue
                row = att[0, 0, i, :i + 1]
                if rm == "minmax" and (abs(float(row.max() - row[j])) < 1e-4 or abs(float(row.min() - row[j])) < 1e-4) \
                        and j != int(row.argmax()) and j != int(row.argmin()):
                    continue
                _, yp = normalized_output(qa, ap, v, K, rule, rm, tau, cells, "full")
                _, ym = normalized_output(qa, am, v, K, rule, rm, tau, cells, "full")
                fd = float(((yp - ym) * G).sum() / (2 * eps))
                an = float(a.grad[0, 0, i, j])
                worst = max(worst, abs(fd - an) / float(a.grad.abs().max()))
                tested += 1
        c4[f"{rule}/{rm}/{cells}"] = dict(tested=tested, max_abs_err_over_max_grad=worst)
        assert tested > 30 and worst < 1e-6, (rule, rm, cells, tested, worst)
    res["C4_finite_difference"] = c4

    # ---------------- C5 shift invariance + zero-sum ----------------
    c5 = {}
    for (rule, rm, cells) in combos:
        att = random_scores(rng, 1, 2, 16, 2.0)
        v = torch.from_numpy(rng.normal(size=(1, 2, 16, 4)))
        G = torch.from_numpy(rng.normal(size=(1, 2, 16, 4)))
        _, y0 = normalized_output(qa, att, v, 4, rule, rm, tau, cells, "full")
        _, y1 = normalized_output(qa, att + 3.25, v, 4, rule, rm, tau, cells, "full")
        a = att.clone().requires_grad_(True)
        _, y = normalized_output(qa, a, v, 4, rule, rm, tau, cells, "full")
        (y * G).sum().backward()
        causal = torch.ones(16, 16, dtype=torch.bool).tril()
        rs = a.grad.masked_fill(~causal, 0).sum(-1).abs().max()
        c5[f"{rule}/{rm}/{cells}"] = dict(max_abs_output_shift_change=float((y1 - y0).abs().max()),
                                          max_abs_row_grad_sum=float(rs),
                                          grad_scale=float(a.grad.abs().max()))
        assert c5[f"{rule}/{rm}/{cells}"]["max_abs_output_shift_change"] < 1e-12
        assert float(rs) < 1e-12 * max(1.0, float(a.grad.abs().max())) * 16
    res["C5_shift_invariance"] = c5

    # ---------------- C7 edge cases ----------------
    c7 = {}
    T = 5
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    for rule in ("centroid", "centroid2"):
        # degenerate MinMax row (all scores equal) and single-key row 0
        att = torch.full((1, 1, T, T), 0.7)
        w, *_ = qa._centroid_reconstruct(att, causal, 4, rule, range_mode="minmax")
        c7[f"{rule}/minmax_degenerate_all_ones"] = bool(torch.equal(w[0, 0][causal], torch.ones(int(causal.sum()))))
        # FW row where everything except the max lies in the tail
        att = torch.tensor([[[[0.0, -50, -50, -50, -50]] * T]]).clone()
        att[0, 0, :, 0] = 10.0
        w, _, _, lev, tail = qa._centroid_reconstruct(att, causal, 4, rule, range_mode="relevant_zero", tau=6.0)
        row = w[0, 0, 4]
        c7[f"{rule}/fw_only_max_in_window"] = dict(weights=row.tolist(), tail=tail[0, 0, 4].tolist())
        assert float(row[0]) == 1.0 and float(row[1:].abs().sum()) == 0.0
        # ties at t = 1/2 go to the upper cell, same as the Nearest level index
        att = torch.tensor([[[[0.0, -1.5, -0.75, -3.0, -6.0]] * T]])
        _, _, _, lev_c, _ = qa._centroid_reconstruct(att, causal, 4, rule, range_mode="relevant_zero", tau=6.0)
        _, _, _, lev_n, _ = qa._relevant_zero_reconstruct(att, causal, 4, "nearest", tau=6.0, return_level_idx=True)
        c7[f"{rule}/levels_equal_nearest"] = bool(torch.equal(lev_c, lev_n))
        assert c7[f"{rule}/levels_equal_nearest"]
        # singleton / constant cells reproduce exp(z)
        att = torch.tensor([[[[0.0, -1.5, -3.0, -4.5, -6.0]] * T]])
        w, *_ = qa._centroid_reconstruct(att, causal, 4, rule, range_mode="relevant_zero", tau=6.0)
        ex = torch.exp(att[0, 0, 4])
        c7[f"{rule}/singleton_cells_exact_max_rel"] = float(((w[0, 0, 4] - ex).abs() / ex).max())
        assert c7[f"{rule}/singleton_cells_exact_max_rel"] < 1e-14
    res["C7_edge_cases"] = c7

    # ---------------- C8 representative properties ----------------
    c8 = {"uniform_cell_lme_minus_centre": {}, "lloyd_check": None, "centroid2_log_gap_examples": {}}
    for h in (0.1875, 0.375, 0.75, 1.5, 3.0, 6.0):
        x = np.linspace(-h / 2, h / 2, 200001)
        lme = math.log(np.exp(x).mean())
        c8["uniform_cell_lme_minus_centre"][str(h)] = dict(lme_minus_centre=lme, h2_over_24=h * h / 24,
                                                           closed_form=math.log(math.sinh(h / 2) / (h / 2)),
                                                           centroid2_value=(h * h / 12) / 2)
    z = rng.uniform(-1.5, 0, size=7)
    e = np.exp(z)
    base = ((e - e.mean()) ** 2).sum()
    worse = all(((e - (e.mean() + d)) ** 2).sum() > base for d in (-1e-3, 1e-3, -0.1, 0.1))
    c8["lloyd_check"] = bool(worse)
    for name, zz in {"two_point_width_1.5": [0.0, -1.5], "two_point_width_6": [0.0, -6.0],
                     "one_high_nine_low_width_6": [0.0] + [-6.0] * 9,
                     "uniform_width_6": list(np.linspace(-6, 0, 50))}.items():
        zz = np.array(zz)
        lme = math.log(np.exp(zz).mean())
        c2v = zz.mean() + zz.var() / 2
        c8["centroid2_log_gap_examples"][name] = dict(log_centroid=lme, log_centroid2=c2v, gap=c2v - lme)
    res["C8_representative_properties"] = c8

    # ---------------- C10 cell indices bitwise equal to the existing functions ----------------
    c10 = {}
    for dtype in (torch.float64, torch.float32):
        for rm in ("minmax", "relevant_zero"):
            ok = True
            for K in (2, 4, 8, 16, 32):
                for scale in (0.3, 2.0, 8.0):
                    att = random_scores(rng, 2, 3, 40, scale).to(dtype)
                    att[0, 0, 7, :8] = 1.25          # one degenerate row
                    causal = torch.ones(40, 40, dtype=torch.bool).tril()
                    b, l, sp, tm, *_ = qa._centroid_cells(att, causal, K, rm, 6.0)
                    if rm == "minmax":
                        _, b0, sp0, l0 = qa._adaptive_interval_reconstruct(att, causal, K, "nearest",
                                                                           return_level_idx=True)
                        ok &= torch.equal(b, b0) and torch.equal(l, l0) and torch.equal(sp, sp0)
                    else:
                        _, b0, sp0, l0, tm0 = qa._relevant_zero_reconstruct(att, causal, K, "nearest", tau=6.0,
                                                                            return_level_idx=True)
                        ok &= torch.equal(b, b0) and torch.equal(l, l0) and torch.equal(tm, tm0) \
                            and torch.equal(sp, sp0)
            c10[f"{rm}/{str(dtype).split('.')[-1]}"] = bool(ok)
            assert ok, (rm, dtype)
    res["C10_cell_indices_equal_existing_functions"] = c10

    # ---------------- C9 regression of all pre-existing paths ----------------
    res["C9_regression_vs_phaseB_rr"] = regression()

    out = os.path.join(C, "checks", "check_centroid_ops.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("C1_forward_vs_numpy_reference_max_rel", "C3_grad_vs_closed_form_max_rel",
                                          "C6_full_minus_detach_grad_max_rel")}, indent=1))
    print("regression:", res["C9_regression_vs_phaseB_rr"]["all_bitwise_equal"])
    print("wrote", out)


def regression():
    """Whole-GPT forward/backward on CPU (fp32 model, as in training) for every pre-existing mode."""
    torch.set_default_dtype(torch.float32)
    modes = [dict(attn_mode="softmax"), dict(attn_mode="exact_fp32_softmax")]
    for rm in ("minmax", "relevant_zero"):
        for approx in ("lerp", "nearest"):
            for rg in ("detach", "full"):
                for K in (4, 8):
                    modes.append(dict(attn_mode="quant", attn_K=K, attn_approx=approx, attn_range_grad=rg,
                                      attn_range_mode=rm, attn_range_tau=6.0))
    out = {"cases": [], "all_bitwise_equal": True}
    x = torch.randint(0, 50304, (2, 64), generator=torch.Generator().manual_seed(5))
    y = torch.randint(0, 50304, (2, 64), generator=torch.Generator().manual_seed(6))
    results = {}
    for root in (SRC_ROOT, NEW_ROOT):
        mm, _ = import_root(root)
        for i, md in enumerate(modes):
            torch.manual_seed(1234)
            cfg = mm.GPTConfig(block_size=64, n_layer=2, n_head=4, n_embd=64, **md)
            model = mm.GPT(cfg)
            # scale up Q/K so that scores span several nats and every bin/tail path is exercised
            with torch.no_grad():
                for blk in model.transformer.h:
                    blk.attn.c_attn.weight[:128].mul_(12.0)
            _, loss = model(x, y)
            loss.backward()
            results.setdefault(i, []).append((loss.detach().clone(),
                                              {n: p.grad.detach().clone() for n, p in model.named_parameters()}))
    for i, md in enumerate(modes):
        (l0, g0), (l1, g1) = results[i]
        eq = bool(torch.equal(l0, l1)) and all(torch.equal(g0[n], g1[n]) for n in g0)
        out["cases"].append(dict(mode=md, loss=float(l0), bitwise_equal=eq))
        out["all_bitwise_equal"] &= eq
    assert out["all_bitwise_equal"]
    torch.set_default_dtype(torch.float64)
    return out


if __name__ == "__main__":
    main()
