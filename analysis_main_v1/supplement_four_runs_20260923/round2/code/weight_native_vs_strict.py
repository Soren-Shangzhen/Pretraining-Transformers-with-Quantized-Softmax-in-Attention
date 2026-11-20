#!/usr/bin/env python3
"""Round-2 Task 1: historical Weight-STE K4 endpoints evaluated with their native forward and with a strict
hard forward, on the local RTX 3090 (torch 2.5.1, bta env).  Both forwards of a run use the same GPU.

native : the run's own frozen code root, unmodified (byte-identical local copy; hashes recorded).
strict : a copy of that root in which only the surrogate composition
         `w = w_lerp + (w_nearest - w_lerp).detach()`  ->  `w = w_nearest.detach() + (w_lerp - w_lerp.detach())`
         was changed (roots/strict_*; diff recorded).
Checks : on GPU, the strict reconstruct equals the direct hard weights w_N bit for bit, on random rows and on
         the checkpoint's own scores (all 12 layers, first validation block).  w_N comes from an independent
         code path (the supplement E2 roots' nearest_forward="direct").
Protocol (as the paper): full validation split (FineWebEdu 976 blocks for suites A/B, WikiText-103 243 blocks for
         suite D), batch 1, bf16 autocast outside the fp32 attention kernel, TF32 off, eval mode.
"""
import csv, importlib.util, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.dirname(HERE)
SUP = os.path.dirname(R2)
TR = KI_REPO + ""
AM = f"{TR}/analysis_main_v1"
PB = f"{TR}/training_code/phaseB_base"
sys.path.insert(0, f"{SUP}/code/checks")
sys.path.insert(0, f"{SUP}/code/eval")
from sup_common import LocalsCapture, compare, env_record, sha256_file  # noqa: E402
from common import ValBlocks, build_model, per_block_nll  # noqa: E402
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


ROOTS = f"{R2}/task1/roots"
MO = f"{R2}/model_only"
VAL_FW = f"{R2}/data/finewebedu_3b/val.bin"
VAL_WT = f"{PB}/data/wikitext103/val.bin"
OUT = f"{R2}/task1"
N_BOOT, BOOT_BLOCK, RNG_SEED = 4000, 16, 20260920

RUNS = [  # id, suite, ckpt, native root, strict root, calib, val, reported source
    ("A_MM_W4", "A", f"{MO}/A_MM_W4.model_only.pt", "native_formal_base_v2", "strict_formal_base_v2", "minmax", VAL_FW,
     ("A_longhorizon_124m_2p5b", "formal_124m_s1337_nearest_k4_fullgrad_2p5b")),
    ("A_FW_W4", "A", f"{MO}/A_FW_W4.model_only.pt", "native_formal_qrm_v1", "strict_formal_qrm_v1", "fwm", VAL_FW,
     ("A_longhorizon_124m_2p5b", "formal_124m_s1337_qrm_nearest_k4_tau6_fullgrad_2p5b")),
    ("B_MM_W4", "B", f"{MO}/B_MM_W4.model_only.pt", "native_formal_base_v2", "strict_formal_base_v2", "minmax", VAL_FW,
     ("B_factorial_124m_250m", "formal_124m_s1337_nearest_k4_weightste_250m")),
    ("B_FW_W4", "B", f"{MO}/B_FW_W4.model_only.pt", "native_formal_qrm_v1", "strict_formal_qrm_v1", "fwm", VAL_FW,
     ("B_factorial_124m_250m", "formal_124m_s1337_qrm_k4_tau6_matched250m")),
    ("D_MM_W4_s7", "D", f"{PB}/runs/p3_s7_nearest_k4_fullgrad_100m/ckpt_step000763_100.0m.pt", "native_local_phaseB_base",
     "strict_local_phaseB_base", "minmax", VAL_WT, ("D", "MinMax-Weight_K4_s7")),
    ("D_FW_W4_s7", "D", f"{PB}/qrm_runs_v1/qrm_s7_nearest_k4_tau6_fullgrad_100m/ckpt_step000763_100.0m.pt",
     "native_local_phaseB_rr_qrm", "strict_local_phaseB_rr_qrm", "fwm", VAL_WT, ("D", "QRM-Weight_K4_s7")),
]
SOFTMAX = {"A": (f"{MO}/A_EXACT.model_only.pt", "native_formal_base_v2", VAL_FW, ("A_longhorizon_124m_2p5b", "formal_124m_s1337_exact_2p5b")),
           "B": (f"{MO}/B_EXACT.model_only.pt", "native_formal_base_v2", VAL_FW, ("B_factorial_124m_250m", "formal_124m_s1337_exact_matched250m")),
           "D": (f"{PB}/runs/p3_s7_exact_100m/ckpt_step000763_100.0m.pt", "native_local_phaseB_base", VAL_WT, ("D", "Exact_s7"))}
DIRECT = {"minmax": f"{SUP}/code/e2_minmax_probste_strict", "fwm": f"{SUP}/code/e2_fwm_probste_strict"}
_MODS = {}


def load(root, tag):
    """model.py + quant_attention.py of one root under unique module names."""
    if tag in _MODS:
        return _MODS[tag]
    spec = importlib.util.spec_from_file_location(f"qa_{tag}", f"{root}/quant_attention.py")
    qa = importlib.util.module_from_spec(spec); spec.loader.exec_module(qa)
    saved = sys.modules.get("quant_attention")
    sys.modules["quant_attention"] = qa
    try:
        spec = importlib.util.spec_from_file_location(f"mm_{tag}", f"{root}/model.py")
        mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
    finally:
        if saved is None:
            sys.modules.pop("quant_attention", None)
        else:
            sys.modules["quant_attention"] = saved
    _MODS[tag] = (qa, mm)
    return qa, mm


def recon(qa, att, calib, **kw):
    T = att.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=att.device).tril()
    with torch.no_grad():
        if calib == "minmax":
            return qa._adaptive_interval_reconstruct(att, causal, 4, "nearest", range_grad="full", **kw)[0]
        return qa._relevant_zero_reconstruct(att, causal, 4, "nearest", tau=6.0, range_grad="full", **kw)[0]


def boot(d):
    rng = np.random.default_rng(RNG_SEED)
    n = len(d); nb = int(np.ceil(n / BOOT_BLOCK))
    st = rng.integers(0, n, size=(N_BOOT, nb))
    idx = (st[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]).reshape(N_BOOT, -1)[:, :n] % n
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def reported(src):
    suite, run = src
    if suite == "D":
        d = f"{AM}/D_robustness_124m_100m_5seed/eval_native_v1/{run}/100m_ckpt_step000763_100.0m"
        return float(np.load(f"{d}/per_block_nll.npy").mean()), f"{d}/per_block_nll.npy (local RTX 3090 eval)"
    for r in csv.DictReader(open(f"{AM}/{suite}/endpoint_fullval/endpoint_table.csv")):
        if r["run_name"] == run:
            return float(r["mean_nll"]), f"{AM}/{suite}/endpoint_fullval/endpoint_table.csv (gpu-host-A RTX 5090 eval)"
    raise KeyError(src)


def capture_P_and_att(model, x):
    Ps, atts = [], []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
            LocalsCapture(["_quant_attention"], ["P", "att"], lambda fn, d: (Ps.append(d["P"].clone()), atts.append(d["att"].clone()))):
        model(x)
    return Ps, atts


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(f"{OUT}/per_block", exist_ok=True)
    rep = {"script": os.path.abspath(__file__), "script_sha256": sha256_file(__file__), "env": env_record(),
           "precision": "bf16 autocast outside fp32 attention, TF32 off, batch 1, eval mode",
           "roots": {d: {f: sha256_file(f"{ROOTS}/{d}/{f}") for f in ("model.py", "quant_attention.py")} for d in sorted(os.listdir(ROOTS))},
           "direct_wN_roots": {k: sha256_file(f"{v}/quant_attention.py") for k, v in DIRECT.items()},
           "val": {"finewebedu_val_sha256": sha256_file(VAL_FW), "wikitext_val_sha256": sha256_file(VAL_WT)},
           "runs": {}, "softmax": {}}
    blocks = {VAL_FW: ValBlocks(VAL_FW, 1024), VAL_WT: ValBlocks(VAL_WT, 1024)}
    assert blocks[VAL_FW].n_blocks == 976 and blocks[VAL_WT].n_blocks == 243
    # ---------------- softmax baselines (native only)
    soft = {}
    for suite, (ck, root, val, src) in SOFTMAX.items():
        qa, mm = load(f"{ROOTS}/{root}", root)
        c = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
        model, cfg, op = build_model(mm, c, "cuda")
        t0 = time.time()
        nll = per_block_nll(model, blocks[val], 0, blocks[val].n_blocks, "cuda", 1)
        np.save(f"{OUT}/per_block/{suite}_EXACT__native.npy", nll)
        rv, rs = reported(src)
        soft[suite] = nll
        rep["softmax"][suite] = {"checkpoint": ck, "checkpoint_sha256": sha256_file(ck), "mean_nll": float(nll.mean()),
                                 "reported": rv, "reported_source": rs, "native_minus_reported": float(nll.mean()) - rv,
                                 "n_blocks": len(nll), "seconds": round(time.time() - t0, 1)}
        print("softmax", suite, rep["softmax"][suite]["mean_nll"], "vs reported", rv, flush=True)
        del model, c
        torch.cuda.empty_cache()
    # ---------------- Weight-STE runs
    for rid, suite, ck, nroot, sroot, calib, val, src in RUNS:
        qa_n, mm_n = load(f"{ROOTS}/{nroot}", nroot)
        qa_s, mm_s = load(f"{ROOTS}/{sroot}", sroot)
        qa_d, _ = load(DIRECT[calib], f"direct_{calib}")
        c = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
        cfgd = c["config"]
        assert cfgd["K"] == 4 and cfgd["approx"] == "nearest" and cfgd["range_grad"] == "full"
        assert (cfgd.get("range_mode") == "relevant_zero") == (calib == "fwm")
        mn, _, _ = build_model(mm_n, c, "cuda")
        ms, _, _ = build_model(mm_s, c, "cuda")
        R = {"checkpoint": ck, "checkpoint_sha256": sha256_file(ck), "step": c["step"], "tokens": c["tokens_done"],
             "native_root": nroot, "strict_root": sroot, "calibration": calib}
        # ---- GPU verification: strict reconstruct == direct w_N (random rows, real scores)
        g = torch.Generator(device="cpu").manual_seed(4)
        ver = {"random_rows": [], "real_scores_block0": []}
        for span in (1.0, 4.0, 8.0, 16.0, 32.0):
            att = (torch.randn(1, 12, 256, 256, generator=g) * span / 5.0).cuda()
            wn, ws, wd = recon(qa_n, att, calib), recon(qa_s, att, calib), recon(qa_d, att, calib, nearest_forward="direct")
            ver["random_rows"].append({"span_scale": span, "strict_vs_direct": compare(ws, wd), "native_vs_direct_mismatch": compare(wn, wd)["mismatch_count"]})
        x0, _ = blocks[val].batch(0, 1, "cuda")
        _, atts = capture_P_and_att(mn, x0)
        for li, att in enumerate(atts):
            att = att.float()
            ws, wd, wn = recon(qa_s, att, calib), recon(qa_d, att, calib, nearest_forward="direct"), recon(qa_n, att, calib)
            T = att.shape[-1]
            nvalid = 12 * T * (T + 1) // 2
            ver["real_scores_block0"].append({"layer": li, "strict_vs_direct": compare(ws, wd)["eq_rawbits"],
                                              "strict_vs_direct_mismatch": compare(ws, wd)["mismatch_count"],
                                              "native_vs_direct_mismatch_frac_valid": compare(wn, wd)["mismatch_count"] / nvalid})
        R["verification"] = ver
        R["strict_equals_direct_all"] = all(v["strict_vs_direct"]["eq_rawbits"] for v in ver["random_rows"]) and \
            all(v["strict_vs_direct"] for v in ver["real_scores_block0"])
        # ---- per-layer attention-weight mismatch on the first 8 blocks (in-model, propagated) and same-input
        mism_model = np.zeros(12); mism_local = np.zeros(12); tot = 0
        for b in range(8):
            xb, _ = blocks[val].batch(b, b + 1, "cuda")
            Pn, an = capture_P_and_att(mn, xb)
            Ps, _ = capture_P_and_att(ms, xb)
            T = xb.shape[1]
            valid = torch.ones(T, T, dtype=torch.bool, device="cuda").tril().view(1, 1, T, T).expand(1, 12, T, T)
            for li in range(12):
                mism_model[li] += int((Pn[li].view(torch.int32) != Ps[li].view(torch.int32))[valid].sum())
                att = an[li].float()
                wl_n, wl_s = recon(qa_n, att, calib), recon(qa_s, att, calib)
                Pl_n = wl_n / wl_n.sum(-1, keepdim=True); Pl_s = wl_s / wl_s.sum(-1, keepdim=True)
                mism_local[li] += int((Pl_n.view(torch.int32) != Pl_s.view(torch.int32))[valid].sum())
            tot += int(valid.sum())
            del Pn, Ps, an
        R["attn_weight_mismatch_first8"] = {"in_model_frac_per_layer": (mism_model / tot).tolist(),
                                             "same_input_frac_per_layer": (mism_local / tot).tolist(),
                                             "valid_elements_per_layer": tot,
                                             "note": "in_model: each forward on its own activations (differences propagate); same_input: both operators on the native model's scores of that layer"}
        # ---- full-split evaluation
        t0 = time.time()
        nn_ = per_block_nll(mn, blocks[val], 0, blocks[val].n_blocks, "cuda", 1)
        ns_ = per_block_nll(ms, blocks[val], 0, blocks[val].n_blocks, "cuda", 1)
        np.save(f"{OUT}/per_block/{rid}__native.npy", nn_); np.save(f"{OUT}/per_block/{rid}__strict.npy", ns_)
        d = nn_ - ns_
        m, lo, hi = boot(d)
        rv, rs = reported(src)
        sm = soft[suite]
        R.update({"n_blocks": len(d), "native_mean_nll": float(nn_.mean()), "strict_mean_nll": float(ns_.mean()),
                  "delta_native_minus_strict": m, "delta_ci95": [lo, hi], "mean_abs_delta": float(np.abs(d).mean()),
                  "max_abs_delta": float(np.abs(d).max()), "n_blocks_exactly_equal": int((d == 0).sum()),
                  "native_minus_softmax": boot(nn_ - sm), "strict_minus_softmax": boot(ns_ - sm),
                  "reported": rv, "reported_source": rs, "native_minus_reported": float(nn_.mean()) - rv,
                  "eval_seconds": round(time.time() - t0, 1)})
        rep["runs"][rid] = R
        print(rid, "strict==direct", R["strict_equals_direct_all"], "native", R["native_mean_nll"], "strict", R["strict_mean_nll"],
              "delta %.2e [%.2e,%.2e]" % (m, lo, hi), "vs reported %.2e" % R["native_minus_reported"], flush=True)
        json.dump(rep, open(f"{OUT}/weight_native_vs_strict.json", "w"), indent=1, default=str)
        del mn, ms, c
        torch.cuda.empty_cache()
    json.dump(rep, open(f"{OUT}/weight_native_vs_strict.json", "w"), indent=1, default=str)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
