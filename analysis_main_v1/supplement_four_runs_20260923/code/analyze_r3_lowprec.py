#!/usr/bin/env python3
"""Round 3 analysis: naive reduced-precision softmax baselines (bf16-exp, FP8 e4m3-exp) at 124M@100M WikiText-103
seed 7, against the same-seed exact softmax and the K-interval operators.  All endpoints evaluated on the local
RTX 3090 with the historical protocol (243 blocks, batch 1, bf16 autocast outside fp32 attention, TF32 off).
Bootstrap: circular moving-block 16, 4000 resamples, 95%, RNG seed 20260920 re-seeded per contrast.
Five-seed context (seed-level, not evaluation CIs) from D_robustness_124m_100m_5seed/analysis_v1/endpoint_nll_by_seed.csv."""
import csv, glob, json, os
import numpy as np
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

TR = KI_REPO + ""
SUP = f"{TR}/analysis_main_v1/supplement_four_runs_20260923"
HIST = f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1"
PB = f"{TR}/training_code/phaseB_base"
T = [20, 40, 60, 80, 100]
SER = {"softmax": f"{HIST}/Exact_s7", "bf16exp": f"{SUP}/eval/rtx3090/sp_s7_softmax_bf16exp_100m",
       "fp8e4m3exp": f"{SUP}/eval/rtx3090/sp_s7_softmax_fp8e4m3exp_100m",
       "LERP K4": f"{HIST}/LERP_K4_s7", "LERP K32": f"{HIST}/LERP_K32_s7",
       "MinMax-Weight K4 (hist)": f"{HIST}/MinMax-Weight_K4_s7", "MinMax-Prob K4 (hist)": f"{HIST}/MinMax-Prob_K4_s7",
       "FWM-Weight K4 (hist)": f"{HIST}/QRM-Weight_K4_s7", "FWM-Prob K4 (hist)": f"{HIST}/QRM-Prob_K4_s7",
       "FWM-Prob K4 strict (new)": f"{SUP}/eval/rtx3090/sp_s7_fwm_nearest_k4_tau6_probste_strict_100m",
       "FWM-Weight K4 strict (new)": f"{SUP}/eval/rtx3090/sp_s7_fwm_nearest_k4_tau6_weightste_strict_100m",
       "MinMax-Prob K4 strict (new)": f"{SUP}/eval/rtx3090/sp_s7_minmax_nearest_k4_probste_strict_100m",
       "MinMax-Weight K4 strict (new)": f"{SUP}/eval/rtx3090/sp_s7_minmax_nearest_k4_weightste_strict_100m"}
LOGS = {"softmax": f"{PB}/runs/p3_s7_exact_100m/log.csv",
        "bf16exp": f"{PB}/supplement_runs_20260923/sp_s7_softmax_bf16exp_100m/log.csv",
        "fp8e4m3exp": f"{PB}/supplement_runs_20260923/sp_s7_softmax_fp8e4m3exp_100m/log.csv"}


def boot(d):
    rng = np.random.default_rng(20260920)
    n = len(d); nb = int(np.ceil(n / 16))
    st = rng.integers(0, n, size=(4000, nb))
    idx = (st[:, :, None] + np.arange(16)[None, None, :]).reshape(4000, -1)[:, :n] % n
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def pb(k, t=100):
    m = sorted(glob.glob(f"{SER[k]}/{t:03d}m_*"))
    return np.load(f"{m[0]}/per_block_nll.npy") if m else None


def geom(k, t=100):
    m = sorted(glob.glob(f"{SER[k]}/{t:03d}m_*"))
    if not m or not os.path.exists(f"{m[0]}/geometry_layers.csv"):
        return {}
    rows = list(csv.DictReader(open(f"{m[0]}/geometry_layers.csv")))
    return {c: float(np.mean([float(r[c]) for r in rows])) for c in ("span_mean", "exact_entropy_mean", "q_norm_mean", "k_norm_mean")}


def logstats(k):
    rows = list(csv.DictReader(open(LOGS[k])))
    tl = np.array([float(r["train_loss"]) for r in rows]); gn = np.array([float(r["grad_norm"]) for r in rows])
    return {"n_logged": len(rows), "grad_norm_max": float(gn.max()), "grad_norm_last": float(gn[-1]),
            "max_train_loss_increase_between_logs": float(np.max(np.diff(tl))) if len(tl) > 1 else None,
            "any_nonfinite": bool(not np.all(np.isfinite(tl)))}


def main():
    sm = pb("softmax")
    out = {"vs_softmax_100m": {}, "trajectory_vs_softmax": {}, "abs_nll": {}, "geometry_100m": {}, "logs": {}}
    for k in SER:
        a = pb(k)
        if a is None:
            out["vs_softmax_100m"][k] = "missing"; continue
        if k != "softmax":
            out["vs_softmax_100m"][k] = boot(a - sm)
        out["abs_nll"][k] = {t: (float(pb(k, t).mean()) if pb(k, t) is not None else None) for t in T}
        out["geometry_100m"][k] = geom(k)
    for k in ("bf16exp", "fp8e4m3exp", "LERP K4", "LERP K32"):
        out["trajectory_vs_softmax"][k] = {t: float(pb(k, t).mean() - pb("softmax", t).mean()) for t in T if pb(k, t) is not None}
    for k in LOGS:
        out["logs"][k] = logstats(k)
    out["pairwise_100m"] = {
        "bf16exp - LERP K4": boot(pb("bf16exp") - pb("LERP K4")),
        "fp8e4m3exp - LERP K4": boot(pb("fp8e4m3exp") - pb("LERP K4")),
        "fp8e4m3exp - FWM-Prob K4 strict (new)": boot(pb("fp8e4m3exp") - pb("FWM-Prob K4 strict (new)")),
        "fp8e4m3exp - bf16exp": boot(pb("fp8e4m3exp") - pb("bf16exp")),
    }
    R = {r["cond"]: [float(r[s]) for s in ("7", "42", "253", "1337", "2026")]
         for r in csv.DictReader(open(f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/endpoint_nll_by_seed.csv"))}
    ctx = {}
    for c in ("LERP K4", "LERP K32", "MinMax-Weight K4", "MinMax-Prob K4", "QRM-Weight K4", "QRM-Prob K4", "QRM-Prob K16"):
        if c in R:
            d = np.array(R[c]) - np.array(R["Exact"])
            ctx[f"{c} - Exact"] = {"per_seed": d.round(5).tolist(), "mean": float(d.mean()), "sd": float(d.std(ddof=1))}
    ctx["Exact seed SD"] = float(np.std(R["Exact"], ddof=1))
    out["five_seed_context"] = ctx
    json.dump(out, open(f"{SUP}/round3/reports/r3_lowprec_results.json", "w"), indent=1)
    with open(f"{SUP}/round3/reports/r3_lowprec_contrasts.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["contrast", "delta", "ci_lo", "ci_hi", "n_blocks", "evaluator"])
        for k, v in out["vs_softmax_100m"].items():
            if v != "missing":
                w.writerow([f"{k} - softmax (seed 7, 100M)", *v, 243, "local RTX 3090"])
        for k, v in out["pairwise_100m"].items():
            w.writerow([f"{k} (seed 7, 100M)", *v, 243, "local RTX 3090"])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
