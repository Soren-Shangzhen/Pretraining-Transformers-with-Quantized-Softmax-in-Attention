#!/usr/bin/env python3
"""Round-2 Task 1b: historical Prob-STE K4 endpoints (suites A, B and suite D seed 7) evaluated with their native
forward and with the strict hard forward P^N, on the local RTX 3090 while the seed-42 E1 run trains.

native : the run's own frozen code root (formal_nearest_probste_v1 / formal_qrm_probste_v1), unmodified.
strict : a copy of that root with only the surrogate compositions changed:
         reconstruct  w = w_lerp + (w_nearest - w_lerp).detach()  ->  w = w_nearest.detach() + (w_lerp - w_lerp.detach())
                      (the historical P_hard is built from this expression, so it must return w_N itself)
         Prob         P = P_lerp + (P_hard - P_lerp).detach()     ->  P = P_hard.detach() + (P_lerp - P_lerp.detach())
Checks : on GPU, (i) the strict reconstruct equals the direct w_N on random rows, and (ii) the probabilities the strict
         MODEL actually uses equal w_N / sum(w_N) computed directly from the same scores, all 12 layers, first block.
Protocol and statistics exactly as weight_native_vs_strict.py.  The process is capped at 33% of GPU memory
(torch.cuda.set_per_process_memory_fraction) so that it cannot take memory from the concurrent training run.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from weight_native_vs_strict import (OUT, PB, ROOTS, VAL_FW, VAL_WT, DIRECT, boot, capture_P_and_att, load, recon,  # noqa: E402
                                     reported, compare, env_record, sha256_file, ValBlocks, build_model, per_block_nll)

R2 = os.path.dirname(HERE)
MO = f"{R2}/model_only_prob"
C5 = f"{PB}/cloud_5seed_complement_20260916/runs/formal_runs_probste_5seeds_100m_v1"
RUNS = [
    ("A_MM_P4", "A", f"{MO}/A_MM_P4.model_only.pt", "native_formal_nearest_probste_v1", "strict_formal_nearest_probste_v1", "minmax", VAL_FW,
     ("A_longhorizon_124m_2p5b", "formal_124m_s1337_minmax_nearest_k4_probste_2p5b")),
    ("A_FW_P4", "A", f"{MO}/A_FW_P4.model_only.pt", "native_formal_qrm_probste_v1", "strict_formal_qrm_probste_v1", "fwm", VAL_FW,
     ("A_longhorizon_124m_2p5b", "formal_124m_s1337_qrm_nearest_k4_tau6_probste_2p5b")),
    ("B_MM_P4", "B", f"{MO}/B_MM_P4.model_only.pt", "native_formal_nearest_probste_v1", "strict_formal_nearest_probste_v1", "minmax", VAL_FW,
     ("B_factorial_124m_250m", "formal_124m_s1337_nearest_k4_probste_250m")),
    ("B_FW_P4", "B", f"{MO}/B_FW_P4.model_only.pt", "native_formal_qrm_probste_v1", "strict_formal_qrm_probste_v1", "fwm", VAL_FW,
     ("B_factorial_124m_250m", "formal_124m_s1337_qrm_k4_tau6_probste_250m")),
    ("D_MM_P4_s7", "D", f"{C5}/formal_124m_s7_minmax_nearest_k4_probste_100m/ckpt_step000763_100.0m.pt", "native_formal_nearest_probste_v1",
     "strict_formal_nearest_probste_v1", "minmax", VAL_WT, ("D", "MinMax-Prob_K4_s7")),
    ("D_FW_P4_s7", "D", f"{C5}/formal_124m_s7_qrm_nearest_k4_tau6_probste_100m/ckpt_step000763_100.0m.pt", "native_formal_qrm_probste_v1",
     "strict_formal_qrm_probste_v1", "fwm", VAL_WT, ("D", "QRM-Prob_K4_s7")),
]


def main():
    torch.cuda.set_per_process_memory_fraction(0.33, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rep = {"script": os.path.abspath(__file__), "script_sha256": sha256_file(__file__), "env": env_record(),
           "precision": "bf16 autocast outside fp32 attention, TF32 off, batch 1, eval mode",
           "concurrency": "evaluated while the seed-42 E1 training run used the same GPU; process capped at 33% of GPU memory",
           "roots": {d: {f: sha256_file(f"{ROOTS}/{d}/{f}") for f in ("model.py", "quant_attention.py")}
                     for d in sorted(os.listdir(ROOTS)) if "probste" in d},
           "softmax_per_block": "task1/per_block/{A,B,D}_EXACT__native.npy (same GPU, weight_native_vs_strict.py)", "runs": {}}
    blocks = {VAL_FW: ValBlocks(VAL_FW, 1024), VAL_WT: ValBlocks(VAL_WT, 1024)}
    soft = {s: np.load(f"{OUT}/per_block/{s}_EXACT__native.npy") for s in ("A", "B", "D")}
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
        g = torch.Generator(device="cpu").manual_seed(4)
        ver = {"random_rows_w": [], "real_scores_block0_P": []}
        for span in (1.0, 4.0, 8.0, 16.0, 32.0):
            att = (torch.randn(1, 12, 256, 256, generator=g) * span / 5.0).cuda()
            ver["random_rows_w"].append({"span_scale": span, "strict_vs_direct": compare(recon(qa_s, att, calib),
                                                                                         recon(qa_d, att, calib, nearest_forward="direct"))})
        x0, _ = blocks[val].batch(0, 1, "cuda")
        Ps, atts_s = capture_P_and_att(ms, x0)
        Pn, atts_n = capture_P_and_att(mn, x0)
        T = x0.shape[1]
        nvalid = 12 * T * (T + 1) // 2
        for li in range(12):
            wd = recon(qa_d, atts_s[li].float(), calib, nearest_forward="direct")
            Pd = wd / wd.sum(dim=-1, keepdim=True)
            wdn = recon(qa_d, atts_n[li].float(), calib, nearest_forward="direct")
            Pdn = wdn / wdn.sum(dim=-1, keepdim=True)
            cs = compare(Ps[li], Pd)
            ver["real_scores_block0_P"].append({"layer": li, "strict_model_P_vs_direct_PN": cs["eq_rawbits"],
                                                "mismatch": cs["mismatch_count"],
                                                "native_model_P_vs_direct_PN_mismatch_frac_valid": compare(Pn[li], Pdn)["mismatch_count"] / nvalid})
        del Ps, Pn, atts_s, atts_n
        R["verification"] = ver
        R["strict_equals_direct_all"] = all(v["strict_vs_direct"]["eq_rawbits"] for v in ver["random_rows_w"]) and \
            all(v["strict_model_P_vs_direct_PN"] for v in ver["real_scores_block0_P"])
        mism_model = np.zeros(12); mism_local = np.zeros(12); tot = 0
        for b in range(8):
            xb, _ = blocks[val].batch(b, b + 1, "cuda")
            Pn, an = capture_P_and_att(mn, xb)
            Ps, _ = capture_P_and_att(ms, xb)
            valid = torch.ones(T, T, dtype=torch.bool, device="cuda").tril().view(1, 1, T, T).expand(1, 12, T, T)
            for li in range(12):
                mism_model[li] += int((Pn[li].view(torch.int32) != Ps[li].view(torch.int32))[valid].sum())
                wd = recon(qa_d, an[li].float(), calib, nearest_forward="direct")
                Pd = wd / wd.sum(dim=-1, keepdim=True)
                mism_local[li] += int((Pn[li].view(torch.int32) != Pd.view(torch.int32))[valid].sum())
            tot += int(valid.sum())
            del Pn, Ps, an
        R["attn_weight_mismatch_first8"] = {"in_model_frac_per_layer": (mism_model / tot).tolist(),
                                             "same_input_frac_per_layer": (mism_local / tot).tolist(),
                                             "valid_elements_per_layer": tot,
                                             "note": "in_model: each forward on its own activations; same_input: native P vs P^N computed from the native model's scores of that layer"}
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
        json.dump(rep, open(f"{OUT}/prob_native_vs_strict.json", "w"), indent=1, default=str)
        print(rid, "strict==P^N", R["strict_equals_direct_all"], "native", R["native_mean_nll"], "strict", R["strict_mean_nll"],
              "delta %.2e [%.2e,%.2e]" % (m, lo, hi), "vs reported %.2e" % R["native_minus_reported"], flush=True)
        del mn, ms, c
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
