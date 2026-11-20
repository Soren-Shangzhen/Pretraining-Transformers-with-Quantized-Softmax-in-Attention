"""E2 operator-level checks: strict Prob-STE vs direct hard, Weight references,
old frozen Weight/Prob, and the normalized-LERP VJP.

All attention values come from each root's real `_quant_attention` path.  The
scores are fed exactly by choosing D = T (sqrt(D) a power of two) and k = I so
that att = q / sqrt(D) exactly, and V = I so that y = P exactly (TF32 off, which
is required for the identity products to be exact; the training-precision path
with TF32 on is covered by the model-level step-0 checks).  The score gradient
is recovered exactly as sqrt(D) * q.grad.

Usage:  python ops_check_e2.py --out <json> [--devices cpu,cuda] [--real]
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sup_common import (ROOTS, LocalsCapture, compare, dump_json, env_record,  # noqa: E402
                        load_root, root_hashes, sha256_file, tensor_sha, CLOUD)

TAU = 6.0
K = 4

CALIB = {
    "minmax": dict(old_w="old_phaseB_base", old_p="old_mm_prob", new="new_mm_prob", extra={}),
    "fwm": dict(old_w="old_phaseB_rr", old_p="old_fw_prob", new="new_fw_prob",
                extra=dict(attn_range_mode="relevant_zero", attn_range_tau=TAU)),
}


def make_module(mm, approx, extra, range_grad="full"):
    cfg = mm.GPTConfig(n_layer=1, n_head=1, n_embd=64, attn_mode="quant", attn_K=K,
                       attn_approx=approx, attn_range_grad=range_grad, **extra)
    m = mm.QuantCausalSelfAttention(cfg)
    m.collect_stats = False
    return m


def run_module(mod, att, G):
    """Return (P, dL/dscores, captured locals) through the real operator path."""
    B, H, T, _ = att.shape
    D = T
    sd = math.sqrt(D)
    assert sd == int(sd) and (int(sd) & (int(sd) - 1)) == 0, "sqrt(D) must be a power of two"
    q = (att * sd).detach().clone().requires_grad_(True)
    eye = torch.eye(T, dtype=att.dtype, device=att.device).view(1, 1, T, T).expand(B, H, T, T)
    k = eye.contiguous()
    v = eye.contiguous()
    cap = {}
    with LocalsCapture(["_quant_attention"], ["P", "P_hard", "w_hard", "P_lerp", "att"],
                       lambda fn, d: cap.update(d)):
        y = mod._quant_attention(q, k, v)
    # the exact-feed construction must reproduce the intended scores bit for bit
    assert torch.equal(cap["att"].detach(), att), "exact score feed failed"
    y.backward(G)
    g = q.grad * sd
    return y.detach(), g.detach(), cap


def direct_hard(qa, att, calib, dtype=None):
    """w_N and w_L from the NEW root's reconstruct (nearest_forward='direct')."""
    T = att.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=att.device).tril()
    with torch.no_grad():
        if calib == "minmax":
            wN = qa._adaptive_interval_reconstruct(att, causal, K, "nearest", range_grad="full",
                                                   nearest_forward="direct")[0]
            wL = qa._adaptive_interval_reconstruct(att, causal, K, "lerp", range_grad="full")[0]
        else:
            wN = qa._relevant_zero_reconstruct(att, causal, K, "nearest", tau=TAU, range_grad="full",
                                               nearest_forward="direct")[0]
            wL = qa._relevant_zero_reconstruct(att, causal, K, "lerp", tau=TAU, range_grad="full")[0]
    return wN, wL, causal


def old_weight_w(qa_old, att, calib):
    T = att.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=att.device).tril()
    with torch.no_grad():
        if calib == "minmax":
            return qa_old._adaptive_interval_reconstruct(att, causal, K, "nearest", range_grad="full")[0]
        return qa_old._relevant_zero_reconstruct(att, causal, K, "nearest", tau=TAU, range_grad="full")[0]


def ratio_stats(num, den, valid_nonzero):
    r = (num.double() / den.double())[valid_nonzero]
    if r.numel() == 0:
        return {"n": 0}
    outside = ((r < 0.5) | (r > 2.0))
    return {"n": int(r.numel()), "min": float(r.min()), "max": float(r.max()),
            "frac_outside_half_two": float(outside.double().mean()),
            "n_outside_half_two": int(outside.sum())}


# ----------------------------------------------------------------------------- inputs

def synthetic_inputs(device, seed=20260923):
    g = torch.Generator(device="cpu").manual_seed(seed)
    T, H = 256, 4
    out = {}
    for span in (1.0, 4.0, 8.0, 16.0, 32.0):
        out[f"random_span{int(span)}"] = torch.randn(1, H, T, T, generator=g) * (span / 5.0)
    out["ties_integer"] = torch.randint(-4, 5, (1, H, T, T), generator=g).float()
    out["ties_coarse_half"] = torch.randint(-12, 1, (1, H, T, T), generator=g).float() * 0.5
    out["degenerate_constant"] = torch.full((1, H, T, T), 1.25)
    out["tiny_span_1e-7"] = torch.randn(1, H, T, T, generator=g) * 1e-7
    out["tiny_span_1e-3"] = torch.randn(1, H, T, T, generator=g) * 1e-3
    # FWM boundaries: z exactly on grid points, midpoints, on/around -tau, and far tail.
    h = TAU / K
    pts = [0.0, -TAU, -TAU - 1e-6, -TAU + 1e-6, -TAU - 0.5, -30.0]
    pts += [-TAU + r * h for r in range(K + 1)]
    pts += [-TAU + (r + 0.5) * h for r in range(K)]
    pts += [np.nextafter(np.float32(-TAU + (r + 0.5) * h), np.float32(0)).item() for r in range(K)]
    pts += [np.nextafter(np.float32(-TAU + (r + 0.5) * h), np.float32(-100)).item() for r in range(K)]
    pts = torch.tensor(pts, dtype=torch.float32)
    idx = torch.randint(0, len(pts), (1, H, T, T), generator=g)
    fb = pts[idx]
    fb[..., 0] = 0.0          # key 0 is valid for every row and sits at the row max
    out["fwm_boundaries"] = fb          # row max is exactly 0, so z = att exactly
    out["fwm_boundaries_shifted"] = fb + 2.5   # same rows shifted (z = att - max then rounds)
    # MinMax boundaries: integer-grid rows whose span is exactly K, so every key is
    # on a grid point or a midpoint.
    mm = torch.randint(0, 2 * K + 1, (1, H, T, T), generator=g).float() * 0.5
    mm[..., 0] = 0.0
    mm[..., 1] = float(K)
    out["minmax_grid_and_midpoints"] = mm
    return {k: v.to(device) for k, v in out.items()}


def real_score_inputs(device, layers=(0, 5, 11)):
    """Scores captured from real seed-7 checkpoints through their own frozen roots,
    at the training precision (bf16 autocast outside the fp32 attention kernel,
    TF32 on).  Input: first 1024 tokens of the WikiText-103 validation split."""
    val = np.memmap(f"{CLOUD}/data/wikitext103/val.bin", dtype=np.uint16, mode="r")
    x = torch.from_numpy(val[:1024].astype(np.int64)).view(1, 1024).to(device)
    B5 = f"{CLOUD}/formal_runs_probste_5seeds_100m_v1"
    UP = f"{CLOUD}/supplement_four_runs_20260923/uploads"
    ckpts = [
        ("mm_prob_s7_20m", "old_mm_prob", f"{B5}/formal_124m_s7_minmax_nearest_k4_probste_100m/ckpt_step000153_20.1m.pt", "minmax"),
        ("mm_prob_s7_60m", "old_mm_prob", f"{B5}/formal_124m_s7_minmax_nearest_k4_probste_100m/ckpt_step000459_60.2m.pt", "minmax"),
        ("mm_prob_s7_100m", "old_mm_prob", f"{B5}/formal_124m_s7_minmax_nearest_k4_probste_100m/ckpt_step000763_100.0m.pt", "minmax"),
        ("fw_prob_s7_20m", "old_fw_prob", f"{B5}/formal_124m_s7_qrm_nearest_k4_tau6_probste_100m/ckpt_step000153_20.1m.pt", "fwm"),
        ("fw_prob_s7_60m", "old_fw_prob", f"{B5}/formal_124m_s7_qrm_nearest_k4_tau6_probste_100m/ckpt_step000459_60.2m.pt", "fwm"),
        ("fw_prob_s7_100m", "old_fw_prob", f"{B5}/formal_124m_s7_qrm_nearest_k4_tau6_probste_100m/ckpt_step000763_100.0m.pt", "fwm"),
        ("mm_weight_s7_100m", "old_phaseB_base", f"{UP}/MM_W4_s7_100m.model_only.pt", "minmax"),
        ("fw_weight_s7_100m", "old_phaseB_rr", f"{UP}/FW_W4_s7_100m.model_only.pt", "fwm"),
    ]
    out, prov = {}, {}
    tf32 = torch.backends.cuda.matmul.allow_tf32
    for name, rootkey, path, calib in ckpts:
        if not os.path.exists(path):
            prov[name] = {"path": path, "status": "missing (not available on the cloud disk at run time)"}
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        c = ck["config"]
        qa, mm, _ = load_root(rootkey)
        extra = {}
        if c.get("range_mode") == "relevant_zero":
            extra = dict(attn_range_mode="relevant_zero", attn_range_tau=float(c["range_tau"]))
        cfg = mm.GPTConfig(block_size=c["block_size"], n_layer=c["n_layer"], n_head=c["n_head"],
                           n_embd=c["n_embd"], attn_mode=c["attn"], attn_K=c["K"],
                           attn_approx=c["approx"], attn_range_grad=c["range_grad"], **extra)
        model = mm.GPT(cfg)
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        caps = []
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            ctx = torch.autocast(device_type="cpu", enabled=False)
        with torch.no_grad(), ctx, LocalsCapture(["_quant_attention"], ["att"],
                                                 lambda fn, d: caps.append(d["att"].detach())):
            model(x)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for li in layers:
            out[f"real_{name}_L{li}"] = (caps[li].float().contiguous(), calib)
        prov[name] = {"path": path, "sha256": sha256_file(path) if path.endswith("model_only.pt") else "see manifest",
                      "step": ck["step"], "tokens": ck["tokens_done"], "root": rootkey, "layers": list(layers)}
        del model, ck
        if device == "cuda":
            torch.cuda.empty_cache()
    return out, prov


# ----------------------------------------------------------------------------- core

def check_one(att, calib, device):
    spec = CALIB[calib]
    qa_new, mm_new, _ = load_root(spec["new"], tag=f"{spec['new']}__prob", surrogate="prob")
    qa_ws, mm_ws, _ = load_root(spec["new"], tag=f"{spec['new']}__wstrict", surrogate="weight_strict")
    qa_ow, mm_ow, _ = load_root(spec["old_w"])
    qa_op, mm_op, _ = load_root(spec["old_p"])
    extra = spec["extra"]
    T = att.shape[-1]
    gen = torch.Generator(device="cpu").manual_seed(7)
    G = torch.randn(att.shape, generator=gen).to(device)

    mods = {
        "new_prob": make_module(mm_new, "nearest", extra),
        "strict_weight_ref": make_module(mm_ws, "nearest", extra),
        "old_weight": make_module(mm_ow, "nearest", extra),
        "old_prob": make_module(mm_op, "nearest", extra),
        "lerp_newroot": make_module(mm_new, "lerp", extra),
        "lerp_oldweightroot": make_module(mm_ow, "lerp", extra),
    }
    res = {k: run_module(m, att, G) for k, m in mods.items()}
    P = {k: v[0] for k, v in res.items()}
    g = {k: v[1] for k, v in res.items()}

    wN, wL, causal = direct_hard(qa_new, att, calib)
    PN = wN / wN.sum(dim=-1, keepdim=True)
    PL = wL / wL.sum(dim=-1, keepdim=True)
    w_old = old_weight_w(qa_ow, att, calib)
    valid = causal.view(1, 1, T, T)
    nz = valid & (wN > 0) & (wL > 0)

    cap_new = res["new_prob"][2]
    rec = {
        "shape": list(att.shape), "device": device, "dtype": str(att.dtype),
        "att_sha256": tensor_sha(att), "upstream_sha256": tensor_sha(G),
        # --- forward identity of the new implementation
        "fwd_newprob_vs_directPN": compare(P["new_prob"], PN),
        "fwd_newprob_vs_strict_weight_ref": compare(P["new_prob"], P["strict_weight_ref"]),
        "fwd_repeat_newprob": compare(P["new_prob"], run_module(mods["new_prob"], att, G)[0]),
        # --- VJP of the new implementation: normalized LERP incl. calibration derivative
        "vjp_newprob_vs_lerp": compare(g["new_prob"], g["lerp_newroot"]),
        "vjp_oldprob_vs_lerp": compare(g["old_prob"], g["lerp_newroot"]),
        "vjp_lerp_newroot_vs_oldroot": compare(g["lerp_newroot"], g["lerp_oldweightroot"]),
        "fwd_lerp_newroot_vs_oldroot": compare(P["lerp_newroot"], P["lerp_oldweightroot"]),
        "hard_branch_requires_grad": {
            "P_hard": bool(cap_new["P_hard"].requires_grad) if cap_new.get("P_hard") is not None else None,
            "w_hard": bool(cap_new["w_hard"].requires_grad) if cap_new.get("w_hard") is not None else None,
        },
        # --- old implementations against the strict hard forward
        "w_oldweight_vs_direct_wN": compare(w_old, wN),
        "fwd_oldweight_vs_directPN": compare(P["old_weight"], PN),
        "fwd_oldprob_vs_directPN": compare(P["old_prob"], PN),
        "fwd_oldprob_vs_oldweight": compare(P["old_prob"], P["old_weight"]),
        "vjp_oldweight_vs_strict_weight_ref": compare(g["old_weight"], g["strict_weight_ref"]),
        # --- finiteness incl. masked positions and degenerate rows
        "finite": {k: bool(torch.isfinite(P[k]).all() and torch.isfinite(g[k]).all()) for k in P},
        "masked_P_zero": {k: bool((P[k].masked_select(~valid.expand_as(P[k])) == 0).all()) for k in P},
        # --- ratios on nonzero valid keys (Sterbenz sufficient condition bookkeeping)
        "ratio_wN_over_wL": ratio_stats(wN, wL, nz),
        "ratio_PN_over_PL": ratio_stats(PN, PL, nz),
    }
    if calib == "fwm":
        h = TAU / K
        r = (PN.double() / PL.double())[nz]
        rec["ratio_PN_over_PL"]["within_exp_pm_h"] = bool(((r >= math.exp(-h) * (1 - 1e-6)) &
                                                          (r <= math.exp(h) * (1 + 1e-6))).all()) if r.numel() else None
        rec["ratio_PN_over_PL"]["exp_pm_h"] = [math.exp(-h), math.exp(h)]
    return rec


def fd_lerp_fp64(calib, device, seed=11):
    """float64 formula diagnostic: autograd VJP of normalized LERP (full calibration)
    vs central finite differences, on smooth rows (unique extrema, away from bins)."""
    spec = CALIB[calib]
    qa_new, _, _ = load_root(spec["new"], tag=f"{spec['new']}__prob", surrogate="prob")
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = 16
    att = (torch.randn(1, 2, T, T, generator=g, dtype=torch.float64) * 2.0).to(device)
    causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    G = torch.randn(1, 2, T, T, generator=g, dtype=torch.float64).to(device)
    d = torch.randn(1, 2, T, T, generator=g, dtype=torch.float64).to(device)
    d = d * causal

    def f(a):
        if calib == "minmax":
            w = qa_new._adaptive_interval_reconstruct(a, causal, K, "lerp", range_grad="full")[0]
        else:
            w = qa_new._relevant_zero_reconstruct(a, causal, K, "lerp", tau=TAU, range_grad="full")[0]
        return w / w.sum(dim=-1, keepdim=True)

    a = att.clone().requires_grad_(True)
    (f(a) * G).sum().backward()
    ad = float((a.grad * d).sum())
    eps = 1e-6
    with torch.no_grad():
        fd = float(((f(att + eps * d) - f(att - eps * d)) * G).sum() / (2 * eps))
    # detach-calibration VJP for contrast: shows the calibration term is present
    a2 = att.clone().requires_grad_(True)
    if calib == "minmax":
        w2 = qa_new._adaptive_interval_reconstruct(a2, causal, K, "lerp", range_grad="detach")[0]
    else:
        w2 = qa_new._relevant_zero_reconstruct(a2, causal, K, "lerp", tau=TAU, range_grad="detach")[0]
    ((w2 / w2.sum(dim=-1, keepdim=True)) * G).sum().backward()
    ad_det = float((a2.grad * d).sum())
    return {"autograd_dir_deriv": ad, "central_fd": fd, "abs_err": abs(ad - fd),
            "rel_err": abs(ad - fd) / max(1e-300, abs(fd)),
            "detach_calibration_dir_deriv": ad_det,
            "calibration_term_nonzero": abs(ad - ad_det) > 1e-9, "dtype": "float64", "eps": eps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--devices", default="cpu,cuda")
    ap.add_argument("--real", action="store_true", help="also check real checkpoint scores")
    ap.add_argument("--deterministic", action="store_true",
                    help="diagnostic only: torch.use_deterministic_algorithms(True)")
    args = ap.parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"script": os.path.abspath(__file__), "script_sha256": sha256_file(__file__),
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "env": env_record(), "K": K, "tau": TAU,
              "roots": {k: {"path": v, "sha256": root_hashes(v)} for k, v in ROOTS.items()
                        if k in ("old_phaseB_base", "old_phaseB_rr", "old_mm_prob", "old_fw_prob",
                                 "new_mm_prob", "new_fw_prob")},
              "precision_note": ("operator checks run eager, fp32, TF32 off (required for the exact "
                                 "identity feed); training precision is covered by the step-0 checks"),
              "cases": {}}
    for device in args.devices.split(","):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        inputs = {k: (v, None) for k, v in synthetic_inputs(device).items()}
        if args.real and device == "cuda":
            real, prov = real_score_inputs(device)
            report.setdefault("real_score_provenance", {}).update(prov)
            inputs.update(real)
        for name, (att, only) in inputs.items():
            for calib in (["minmax", "fwm"] if only is None else [only]):
                key = f"{device}/{calib}/{name}"
                t0 = time.time()
                report["cases"][key] = check_one(att, calib, device)
                report["cases"][key]["seconds"] = round(time.time() - t0, 2)
                print(key, "newprob==PN:", report["cases"][key]["fwd_newprob_vs_directPN"]["eq_rawbits"],
                      "vjp==lerp:", report["cases"][key]["vjp_newprob_vs_lerp"]["eq_rawbits"],
                      "oldW==PN:", report["cases"][key]["fwd_oldweight_vs_directPN"]["eq_rawbits"],
                      "oldW mism:", report["cases"][key]["w_oldweight_vs_direct_wN"]["mismatch_count"],
                      flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
        for calib in ("minmax", "fwm"):
            report["cases"][f"{device}/{calib}/fd_lerp_fp64"] = fd_lerp_fp64(calib, device)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump_json(args.out, report)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
