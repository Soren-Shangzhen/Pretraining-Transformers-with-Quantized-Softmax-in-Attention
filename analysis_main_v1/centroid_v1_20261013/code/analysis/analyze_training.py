#!/usr/bin/env python3
"""Part T analysis: seed-7 Centroid / Centroid-2 K4 runs (124M@100M WikiText-103) against the seed-7 controls.

All endpoints were evaluated on the local RTX 3090 with the historical protocol (243 blocks, batch 1, bf16 autocast
outside the fp32 operator, TF32 off).  Bootstrap: circular moving-block 16, 4000 resamples, 95%, RNG 20260920
re-seeded per contrast (identical to round 3).  Labels follow manifests/protocol_centroid_v1.json.
Five-seed context (seed-level, not evaluation CIs): D_robustness_124m_100m_5seed/analysis_v1/delta_by_seed.csv.
Output: results/training/{training_results.json, contrasts.csv, trajectory.csv}
"""
import csv
import glob
import json
import os

import numpy as np
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))  # repository root

TR = KI_REPO
C = f"{TR}/analysis_main_v1/centroid_v1_20261013"
SUP = f"{TR}/analysis_main_v1/supplement_four_runs_20260923"
HIST = f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1"
PB = f"{TR}/training_code/phaseB_base"
OUT = os.path.join(C, "results", "training")
T = [20, 40, 60, 80, 100]
NEW = {"FW-Centroid K4": "cv1_s7_fw_centroid_k4_tau6_100m", "MinMax-Centroid K4": "cv1_s7_minmax_centroid_k4_100m",
       "FW-Centroid-2 K4": "cv1_s7_fw_centroid2_k4_tau6_100m",
       "MinMax-Centroid-2 K4": "cv1_s7_minmax_centroid2_k4_100m"}
SER = {"softmax": f"{HIST}/Exact_s7", "LERP K4": f"{HIST}/LERP_K4_s7",
       "MinMax-Weight K4 (hist)": f"{HIST}/MinMax-Weight_K4_s7", "MinMax-Prob K4 (hist)": f"{HIST}/MinMax-Prob_K4_s7",
       "FWM-Weight K4 (hist)": f"{HIST}/QRM-Weight_K4_s7", "FWM-Prob K4 (hist)": f"{HIST}/QRM-Prob_K4_s7",
       "FWM-Prob K4 strict": f"{SUP}/eval/rtx3090/sp_s7_fwm_nearest_k4_tau6_probste_strict_100m",
       "FWM-Weight K4 strict": f"{SUP}/eval/rtx3090/sp_s7_fwm_nearest_k4_tau6_weightste_strict_100m",
       "MinMax-Prob K4 strict": f"{SUP}/eval/rtx3090/sp_s7_minmax_nearest_k4_probste_strict_100m",
       "MinMax-Weight K4 strict": f"{SUP}/eval/rtx3090/sp_s7_minmax_nearest_k4_weightste_strict_100m",
       "bf16-exp": f"{SUP}/eval/rtx3090/sp_s7_softmax_bf16exp_100m",
       "FP8-e4m3-exp": f"{SUP}/eval/rtx3090/sp_s7_softmax_fp8e4m3exp_100m"}
SER.update({k: f"{C}/eval/rtx3090/{v}" for k, v in NEW.items()})
LOGS = {"softmax": f"{PB}/runs/p3_s7_exact_100m/log.csv",
        "FWM-Weight K4 (hist)": f"{PB}/qrm_runs_v1/qrm_s7_nearest_k4_tau6_fullgrad_100m/log.csv",
        "MinMax-Weight K4 (hist)": f"{PB}/runs/p3_s7_nearest_k4_fullgrad_100m/log.csv",
        "LERP K4": f"{PB}/runs/p3_s7_lerp_k4_fullgrad_100m/log.csv"}
LOGS.update({k: f"{PB}/centroid_runs_20261013/{v}/log.csv" for k, v in NEW.items()})
PRIMARY = [("FW-Centroid K4", "softmax"), ("FW-Centroid K4", "FWM-Prob K4 strict"), ("FW-Centroid K4", "FWM-Prob K4 (hist)"),
           ("FW-Centroid K4", "FWM-Weight K4 (hist)"), ("MinMax-Centroid K4", "softmax"),
           ("MinMax-Centroid K4", "MinMax-Prob K4 strict"), ("MinMax-Centroid K4", "MinMax-Weight K4 (hist)"),
           ("MinMax-Centroid K4", "LERP K4")]
SECONDARY = [("FW-Centroid-2 K4", "FW-Centroid K4"), ("MinMax-Centroid-2 K4", "MinMax-Centroid K4"),
             ("FW-Centroid-2 K4", "softmax"), ("FW-Centroid-2 K4", "FWM-Prob K4 strict"),
             ("MinMax-Centroid-2 K4", "softmax"), ("MinMax-Centroid-2 K4", "MinMax-Prob K4 strict"),
             ("FW-Centroid K4", "FWM-Weight K4 strict"), ("MinMax-Centroid K4", "MinMax-Weight K4 strict"),
             ("MinMax-Centroid K4", "MinMax-Prob K4 (hist)"), ("FW-Centroid K4", "LERP K4"),
             ("FW-Centroid K4", "FP8-e4m3-exp"), ("MinMax-Centroid K4", "FW-Centroid K4")]
EXACT_SD = 0.0062


def boot(d):
    rng = np.random.default_rng(20260920)
    n = len(d)
    nb = int(np.ceil(n / 16))
    st = rng.integers(0, n, size=(4000, nb))
    idx = (st[:, :, None] + np.arange(16)[None, None, :]).reshape(4000, -1)[:, :n] % n
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def label(m, lo, hi):
    if not np.isfinite(m):
        return "non-finite"
    if abs(m) < EXACT_SD or lo <= 0 <= hi:
        return "not_separable"
    if abs(m) < 3 * EXACT_SD:
        return "suggestive"
    return "clear_single_seed"


def pb(k, t=100):
    m = sorted(glob.glob(f"{SER[k]}/{t:03d}m_*"))
    return np.load(f"{m[0]}/per_block_nll.npy") if m else None


def geom(k, t=100):
    m = sorted(glob.glob(f"{SER[k]}/{t:03d}m_*"))
    if not m or not os.path.exists(f"{m[0]}/geometry_layers.csv"):
        return {}
    rows = list(csv.DictReader(open(f"{m[0]}/geometry_layers.csv")))
    cols = ("span_mean", "exact_entropy_mean", "exact_pmax_mean", "q_norm_mean", "k_norm_mean", "tv_native_exact_mean",
            "keys_top_bucket_mean", "exact_mass_top_bucket_mean", "relevant_keys_tau6_mean", "exact_mass_below_tau6_mean")
    return {c: float(np.mean([float(r[c]) for r in rows if r.get(c) not in (None, "", "nan")]))
            for c in cols if any(r.get(c) not in (None, "", "nan") for r in rows)}


def logstats(k):
    p = LOGS.get(k)
    if not p or not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    tl = np.array([float(r["train_loss"]) for r in rows])
    vl = np.array([float(r["val_loss"]) for r in rows])
    gn = np.array([float(r["grad_norm"]) for r in rows])
    tok = np.array([float(r["tokens"]) for r in rows])
    return {"n_logged": len(rows), "last_tokens": float(tok[-1]), "grad_norm_max": float(np.nanmax(gn)),
            "grad_norm_last": float(gn[-1]), "max_train_loss_increase_between_logs":
                float(np.max(np.diff(tl))) if len(tl) > 1 else None,
            "any_nonfinite": bool(not (np.all(np.isfinite(tl)) and np.all(np.isfinite(vl)) and np.all(np.isfinite(gn)))),
            "tok_per_s_median": float(np.median([float(r["tok_per_s"]) for r in rows[1:]])) if len(rows) > 1 else None,
            "val_loss_curve": {int(t): float(v) for t, v in zip(tok, vl)}}


def main():
    os.makedirs(OUT, exist_ok=True)
    sm = pb("softmax")
    res = {"abs_nll": {}, "vs_softmax_trajectory": {}, "contrasts_100m": {}, "geometry_100m": {}, "logs": {},
           "missing": []}
    for k in SER:
        if pb(k) is None:
            res["missing"].append(k)
        res["abs_nll"][k] = {t: (float(pb(k, t).mean()) if pb(k, t) is not None else None) for t in T}
        res["geometry_100m"][k] = geom(k)
        res["logs"][k] = logstats(k)
    for k in list(NEW) + ["LERP K4", "FWM-Prob K4 (hist)", "FWM-Weight K4 (hist)", "MinMax-Prob K4 (hist)",
                          "MinMax-Weight K4 (hist)", "FWM-Prob K4 strict"]:
        res["vs_softmax_trajectory"][k] = {t: float(pb(k, t).mean() - pb("softmax", t).mean())
                                           for t in T if pb(k, t) is not None and pb("softmax", t) is not None}
    rows = []
    for tier, lst in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for a, b in lst:
            A, B = pb(a), pb(b)
            if A is None or B is None:
                continue
            m, lo, hi = boot(A - B)
            lab = label(m, lo, hi)
            res["contrasts_100m"][f"{a} - {b}"] = dict(tier=tier, delta=m, ci95=[lo, hi], label=lab)
            rows.append([tier, f"{a} - {b}", m, lo, hi, lab])
    R = {r["cond"]: [float(r[s]) for s in ("7", "42", "253", "1337", "2026")]
         for r in csv.DictReader(open(f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/delta_by_seed.csv"))}
    E = {r["cond"]: [float(r[s]) for s in ("7", "42", "253", "1337", "2026")]
         for r in csv.DictReader(open(f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/analysis_v1/endpoint_nll_by_seed.csv"))}
    ctx = {c: {"per_seed": np.round(R[c], 5).tolist(), "mean": float(np.mean(R[c])), "sd": float(np.std(R[c], ddof=1))}
           for c in ("LERP K4", "MinMax-Weight K4", "MinMax-Prob K4", "QRM-Weight K4", "QRM-Prob K4", "QRM-Prob K8",
                     "QRM-Prob K16", "LERP K32") if c in R}
    ctx["Exact seed SD"] = float(np.std(E["Exact"], ddof=1))
    res["five_seed_context_delta_vs_exact"] = ctx
    json.dump(res, open(os.path.join(OUT, "training_results.json"), "w"), indent=1)
    with open(os.path.join(OUT, "contrasts.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "contrast (seed 7, 100M)", "delta", "ci_lo", "ci_hi", "label"])
        w.writerows(rows)
    with open(os.path.join(OUT, "trajectory.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series"] + [f"{t}M" for t in T])
        for k, v in res["vs_softmax_trajectory"].items():
            w.writerow([k] + [v.get(t, "") for t in T])
    for r in rows:
        print(f"{r[0]:9s} {r[1]:48s} {r[2]:+.4f} [{r[3]:+.4f},{r[4]:+.4f}] {r[5]}")
    print("missing:", res["missing"])


if __name__ == "__main__":
    main()
