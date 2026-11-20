"""E1 operator-level checks for the range_grad single-channel variants (LERP K32).

A. CPU gate (primary): new full/detach vs the frozen phaseB base root, same
   environment, eager, one thread, fp32 through the real `_quant_attention`
   path (exact score feed, see ops_check_e2.py) and fp64 through the
   dtype-generic `_adaptive_interval_reconstruct` + the source's normalization
   `P = w / w.sum(-1, keepdim=True)` (no source patch is needed for fp64: the
   reconstruct function takes its dtype from the scores).  Old roots are
   repeated to confirm self-consistency first.
B. Four-mode forward identity (full/detach/min_only/max_only), CPU and GPU.
C. Analytic single-channel gradients (Section 4.3 formulas) in fp64 on rows
   with unique, exactly representable extrema (M = 0, m = -8, h = 0.25 at K = 32)
   and interior keys kept >= 0.01 from every grid point.
D. Zero-sum residual rho = |sum g| / sum |g| per row for each mode.
E. Tie / degenerate / mask behaviour of the new modes.

Usage: python ops_check_e1.py --out <json> [--devices cpu,cuda] [--real-scores <pt>]
"""

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sup_common import (ROOTS, compare, dump_json, env_record, load_root,  # noqa: E402
                        root_hashes, sha256_file, tensor_sha)
from ops_check_e2 import run_module  # noqa: E402  (exact score feed through the real path)

K = 32
MODES_NEW = ("full", "detach", "min_only", "max_only")


def make_module(mm, mode):
    cfg = mm.GPTConfig(n_layer=1, n_head=1, n_embd=64, attn_mode="quant", attn_K=K,
                       attn_approx="lerp", attn_range_grad=mode)
    m = mm.QuantCausalSelfAttention(cfg)
    m.collect_stats = False
    return m


def recon_P(qa, att, mode):
    T = att.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=att.device).tril()
    w = qa._adaptive_interval_reconstruct(att, causal, K, "lerp", range_grad=mode)[0]
    return w, w / w.sum(dim=-1, keepdim=True)


def fp64_path(qa, att64, mode, G64):
    a = att64.clone().requires_grad_(True)
    w, P = recon_P(qa, a, mode)
    assert w.dtype == torch.float64 and P.dtype == torch.float64
    (P * G64).sum().backward()
    return P.detach(), a.grad.detach()


# ----------------------------------------------------------------------------- inputs

def synthetic(device, T=256, H=4, seed=31):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for span in (1.0, 4.0, 12.0, 30.0):
        out[f"random_span{int(span)}"] = torch.randn(1, H, T, T, generator=g) * (span / 5.0)
    out["ties_integer"] = torch.randint(-3, 4, (1, H, T, T), generator=g).float()
    out["degenerate_constant"] = torch.full((1, H, T, T), -0.75)
    out["tiny_span_1e-7"] = torch.randn(1, H, T, T, generator=g) * 1e-7
    # rows on exact grid points / midpoints of a K=32 grid over [-8, 0]
    grid = torch.randint(0, 2 * K + 1, (1, H, T, T), generator=g).float() * (8.0 / (2 * K)) - 8.0
    grid[..., 0] = 0.0
    grid[..., 1] = -8.0
    out["grid_points_and_midpoints"] = grid
    return {k: v.to(device) for k, v in out.items()}


def clean_rows(T=64, seed=5):
    """fp64 rows: unique extrema M=0, m=-8 (exact), interior keys >= 0.01 from
    every K=32 grid point of [-8,0]; row i uses keys 0..i."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = 8.0 / K
    att = torch.empty(1, 1, T, T, dtype=torch.float64)
    for i in range(T):
        r = torch.randint(0, K, (T,), generator=g).double()
        frac = 0.04 + 0.92 * torch.rand(T, generator=g, dtype=torch.float64)
        row = -8.0 + (r + frac) * h
        if i >= 1:
            perm = torch.randperm(i + 1, generator=g)
            row[perm[0]] = 0.0
            row[perm[1]] = -8.0
        att[0, 0, i] = row
    return att


# ----------------------------------------------------------------------------- analytic (fp64)

def analytic_grads(att, u):
    """Section 4.3 formulas, evaluated per row in fp64 (row 0 is degenerate and skipped)."""
    T = att.shape[-1]
    out = {m: torch.zeros(T, T, dtype=torch.float64) for m in MODES_NEW}
    ident = []
    for i in range(1, T):
        s = att[0, 0, i, : i + 1]
        uu = u[0, 0, i, : i + 1]
        M, m = s.max(), s.min()
        jM, jm = int(s.argmax()), int(s.argmin())
        h = (M - m) / K
        z = s - M
        zmin = m - M
        r = torch.clamp(torch.floor((z - zmin) / h), 0, K - 1)
        b_r = zmin + r * h
        b_r1 = zmin + (r + 1) * h
        t = (z - b_r) / h
        e_r, e_r1 = torch.exp(b_r), torch.exp(b_r1)
        w = (1 - t) * e_r + t * e_r1
        W = w.sum()
        P = w / W
        gamma = (uu - (P * uu).sum()) / W
        Delta = (e_r1 - e_r) / h
        q = (r + t) / K
        X = ((1 - t) * r * e_r + t * (r + 1) * e_r1) / K
        dM = -w - Delta * q + X
        dm = -Delta * (1 - q) + w - X
        ident.append(float((Delta + dM + dm).abs().max()))
        gd = gamma * Delta
        Am = (gamma * dm).sum()
        AM = (gamma * dM).sum()
        gmin = gd.clone(); gmin[jm] += Am
        gmax = gd.clone(); gmax[jM] += AM
        gfull = gd.clone(); gfull[jm] += Am; gfull[jM] += AM
        out["detach"][i, : i + 1] = gd
        out["min_only"][i, : i + 1] = gmin
        out["max_only"][i, : i + 1] = gmax
        out["full"][i, : i + 1] = gfull
    return out, max(ident)


def rho_rows(g, valid):
    gs = (g * valid).sum(-1)
    ga = (g.abs() * valid).sum(-1)
    zero_den = ga == 0
    rho = torch.where(zero_den, torch.zeros_like(ga), gs.abs() / torch.where(zero_den, torch.ones_like(ga), ga))
    r = rho[~zero_den].detach().cpu()
    qs = torch.quantile(r.double(), torch.tensor([0.0, 0.5, 0.9, 1.0], dtype=torch.float64)) if r.numel() else None
    return {"n_rows": int(rho.numel()), "n_zero_denominator": int(zero_den.sum()),
            "min_median_p90_max": [float(x) for x in qs] if qs is not None else None,
            "frac_rho_gt_1e-6": float((r > 1e-6).double().mean()) if r.numel() else None}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--devices", default="cpu,cuda")
    ap.add_argument("--real-scores", default="", help="optional .pt with {name: att(1,H,T,T) fp32}")
    ap.add_argument("--deterministic", action="store_true",
                    help="diagnostic only: torch.use_deterministic_algorithms(True)")
    args = ap.parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    qa_old, mm_old, _ = load_root("old_phaseB_base")
    qa_new, mm_new, _ = load_root("new_e1")
    rep = {"script": os.path.abspath(__file__), "script_sha256": sha256_file(__file__),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "env": env_record(), "K": K,
           "roots": {k: {"path": ROOTS[k], "sha256": root_hashes(ROOTS[k])} for k in ("old_phaseB_base", "new_e1")},
           "fp64_note": ("fp64 uses the dtype-generic _adaptive_interval_reconstruct of each root plus the "
                         "source normalization P = w / w.sum(-1, keepdim=True); no source patch"),
           "cpu_gate": {}, "forward_identity": {}, "gpu_aux": {}, "analytic": {}, "zero_sum": {}, "edge": {}}

    for device in args.devices.split(","):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        inputs = synthetic(device)
        if args.real_scores and os.path.exists(args.real_scores):
            for k, v in torch.load(args.real_scores, map_location="cpu").items():
                inputs[f"real_{k}"] = v.float().to(device)
        for name, att in inputs.items():
            gen = torch.Generator(device="cpu").manual_seed(99)
            G = torch.randn(att.shape, generator=gen).to(device)
            res = {}
            for mode in ("full", "detach"):
                m_old = make_module(mm_old, mode)
                a1 = run_module(m_old, att, G)
                a2 = run_module(m_old, att, G)
                res[f"old_{mode}"] = a1
                res[f"old_{mode}_repeat"] = compare(a1[0], a2[0]), compare(a1[1], a2[1])
            for mode in MODES_NEW:
                res[f"new_{mode}"] = run_module(make_module(mm_new, mode), att, G)
            rec = {"att_sha256": tensor_sha(att), "upstream_sha256": tensor_sha(G), "dtype": "float32",
                   "old_self_repeat": {m: {"fwd": res[f"old_{m}_repeat"][0], "grad": res[f"old_{m}_repeat"][1]}
                                       for m in ("full", "detach")},
                   "new_vs_old": {m: {"fwd": compare(res[f"new_{m}"][0], res[f"old_{m}"][0]),
                                      "score_grad": compare(res[f"new_{m}"][1], res[f"old_{m}"][1])}
                                  for m in ("full", "detach")},
                   "finite": {m: bool(torch.isfinite(res[f"new_{m}"][0]).all() and torch.isfinite(res[f"new_{m}"][1]).all())
                              for m in MODES_NEW}}
            fi = {f"{m}_vs_full": compare(res[f"new_{m}"][0], res["new_full"][0]) for m in MODES_NEW[1:]}
            fi["new_full_vs_old_full"] = compare(res["new_full"][0], res["old_full"][0])
            rep["forward_identity"][f"{device}/{name}"] = fi
            T = att.shape[-1]
            valid = torch.ones(T, T, dtype=torch.bool, device=device).tril().view(1, 1, T, T).to(att.dtype)
            rep["zero_sum"][f"{device}/{name}/random_upstream"] = {m: rho_rows(res[f"new_{m}"][1], valid) for m in MODES_NEW}
            if device == "cpu":
                att64, G64 = att.double(), G.double()
                o = {m: (fp64_path(qa_old, att64, m, G64), fp64_path(qa_old, att64, m, G64)) for m in ("full", "detach")}
                n = {m: fp64_path(qa_new, att64, m, G64) for m in MODES_NEW}
                rec["fp64"] = {
                    "old_self_repeat": {m: {"fwd": compare(o[m][0][0], o[m][1][0]), "grad": compare(o[m][0][1], o[m][1][1])}
                                        for m in o},
                    "new_vs_old": {m: {"fwd": compare(n[m][0], o[m][0][0]), "score_grad": compare(n[m][1], o[m][0][1])}
                                   for m in ("full", "detach")},
                    "forward_identity": {f"{m}_vs_full": compare(n[m][0], n["full"][0]) for m in MODES_NEW[1:]},
                }
                rep["cpu_gate"][name] = rec
            else:
                rep["gpu_aux"][name] = rec
            print(device, name, "new==old full fwd/grad:",
                  rec["new_vs_old"]["full"]["fwd"]["eq_rawbits"], rec["new_vs_old"]["full"]["score_grad"]["eq_rawbits"],
                  "detach:", rec["new_vs_old"]["detach"]["fwd"]["eq_rawbits"], rec["new_vs_old"]["detach"]["score_grad"]["eq_rawbits"],
                  "4-mode fwd:", all(v["eq_rawbits"] for v in fi.values()), flush=True)

    # --- C. analytic single-channel formulas (fp64, CPU)
    att = clean_rows()
    gen = torch.Generator(device="cpu").manual_seed(3)
    u = torch.randn(att.shape, generator=gen, dtype=torch.float64)
    ana, ident = analytic_grads(att, u)
    T = att.shape[-1]
    valid = torch.ones(T, T, dtype=torch.bool).tril()
    arec = {"rows": T - 1, "identity_Delta_plus_dM_plus_dm_max_abs": ident, "atol": 1e-9, "rtol": 1e-7}
    auto = {}
    for m in MODES_NEW:
        _, g = fp64_path(qa_new, att, m, u)
        g = g[0, 0]
        auto[m] = g
        a = ana[m]
        err = (g - a).abs()[valid]
        tol = 1e-9 + 1e-7 * a.abs()[valid]
        arec[m] = {"max_abs_err": float(err.max()), "max_rel_err": float((err / a.abs()[valid].clamp_min(1e-300)).max()),
                   "pass": bool((err <= tol).all())}
    lin = (auto["min_only"] + auto["max_only"] - auto["detach"] - auto["full"]).abs()[valid]
    arec["linearity_full_eq_min_plus_max_minus_detach_max_abs"] = float(lin.max())
    # full must also match finite differences of the true LERP forward
    d = torch.randn(att.shape, generator=gen, dtype=torch.float64) * valid
    eps = 1e-7
    with torch.no_grad():
        fp = (recon_P(qa_new, att + eps * d, "full")[1] * u).sum()
        fm = (recon_P(qa_new, att - eps * d, "full")[1] * u).sum()
    fd = float((fp - fm) / (2 * eps))
    ad = float((auto["full"] * d[0, 0]).sum())
    arec["full_vs_finite_difference"] = {"autograd": ad, "fd": fd, "abs_err": abs(ad - fd),
                                         "rel_err": abs(ad - fd) / max(1e-300, abs(fd))}
    # zero-sum residuals on this fixed diagnostic upstream (structure check, not loss gradients)
    vv = valid.double().view(1, 1, T, T)
    arec["zero_sum_rho_clean_rows"] = {m: rho_rows(auto[m].view(1, 1, T, T), vv) for m in MODES_NEW}
    rep["analytic"] = arec
    print("analytic:", {m: arec[m]["pass"] for m in MODES_NEW}, "identity", ident, flush=True)

    # --- E. ties: min/max gradient split over tied extrema (PyTorch amax/amin convention)
    tie = torch.tensor([[0.0, -8.0, -3.1, 0.0, -8.0, -5.3, -1.7]], dtype=torch.float64)
    T = tie.shape[-1]
    att_t = torch.full((1, 1, T, T), 0.0, dtype=torch.float64)
    att_t[0, 0, T - 1] = tie[0]
    ut = torch.randn(1, 1, T, T, generator=gen, dtype=torch.float64)
    gt = {m: fp64_path(qa_new, att_t, m, ut)[1][0, 0, T - 1] for m in MODES_NEW}
    gt_old = {m: fp64_path(qa_old, att_t, m, ut)[1][0, 0, T - 1] for m in ("full", "detach")}
    rep["edge"]["ties_last_row"] = {
        "scores": tie[0].tolist(),
        "grad": {m: gt[m].tolist() for m in MODES_NEW},
        "min_only_minus_detach": (gt["min_only"] - gt["detach"]).tolist(),
        "max_only_minus_detach": (gt["max_only"] - gt["detach"]).tolist(),
        "new_full_eq_old_full": bool(torch.equal(gt["full"], gt_old["full"])),
        "new_detach_eq_old_detach": bool(torch.equal(gt["detach"], gt_old["detach"])),
        "all_finite": all(bool(torch.isfinite(v).all()) for v in gt.values()),
    }
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump_json(args.out, rep)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
