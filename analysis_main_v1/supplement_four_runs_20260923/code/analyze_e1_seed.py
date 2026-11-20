#!/usr/bin/env python3
"""E1 single-channel analysis for one seed (default 42): paired endpoint contrasts vs the same-seed LERP K32 full /
detach / softmax controls, trajectories, gap recovery, geometry and real-loss zero-sum residuals.  All evaluations on
the local RTX 3090 with the historical protocol (243 blocks, batch 1, bf16 autocast outside fp32 attention, TF32 off).
Bootstrap: circular moving-block, block 16, 4000 resamples, 95%, RNG seed 20260920 re-seeded per contrast."""
import csv, glob, json, os, sys
import numpy as np
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

SEED = int(os.environ.get("SEED", "42"))
TR = KI_REPO + ""
SUP = f"{TR}/analysis_main_v1/supplement_four_runs_20260923"
HIST = f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1"
PB = f"{TR}/training_code/phaseB_base"
T = [20, 40, 60, 80, 100]
SER = {"full": f"{HIST}/LERP_K32_s{SEED}", "detach": f"{HIST}/LERP-Detach_K32_s{SEED}", "softmax": f"{HIST}/Exact_s{SEED}",
       "min_only": f"{SUP}/eval/rtx3090/sp_s{SEED}_lerp_k32_minonly_100m", "max_only": f"{SUP}/eval/rtx3090/sp_s{SEED}_lerp_k32_maxonly_100m"}
LOGS = {"full": f"{PB}/runs/p3_s{SEED}_lerp_k32_fullgrad_100m/log.csv", "detach": f"{PB}/runs_20260921/rb_s{SEED}_lerp_k32_detach_100m/log.csv",
        "softmax": f"{PB}/runs/p3_s{SEED}_exact_100m/log.csv", "min_only": f"{PB}/supplement_runs_20260923/sp_s{SEED}_lerp_k32_minonly_100m/log.csv",
        "max_only": f"{PB}/supplement_runs_20260923/sp_s{SEED}_lerp_k32_maxonly_100m/log.csv"}


def boot(d):
    rng = np.random.default_rng(20260920)
    n = len(d); nb = int(np.ceil(n / 16))
    st = rng.integers(0, n, size=(4000, nb))
    idx = (st[:, :, None] + np.arange(16)[None, None, :]).reshape(4000, -1)[:, :n] % n
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def pb(k, t):
    m = sorted(glob.glob(f"{SER[k]}/{t:03d}m_*"))
    return np.load(f"{m[0]}/per_block_nll.npy") if m else None


def geom(k, t):
    m = sorted(glob.glob(f"{SER[k]}/{t:03d}m_*"))
    rows = list(csv.DictReader(open(f"{m[0]}/geometry_layers.csv")))
    return {c: float(np.mean([float(r[c]) for r in rows])) for c in ("span_mean", "row_min_mean", "exact_entropy_mean", "q_norm_mean", "k_norm_mean")} | \
           {"tv": float(np.mean([float(r["tv_native_exact_mean"]) for r in rows])) if rows[0].get("tv_native_exact_mean") not in ("", "nan", None) else float("nan")}


def gn(k, t):
    rows = list(csv.DictReader(open(LOGS[k])))
    return float(min(rows, key=lambda r: abs(int(r["tokens"]) / 1e6 - t))["grad_norm"])


def main():
    out = {"seed": SEED, "contrasts_100m": {}, "trajectory_vs_full": {}, "abs_nll": {}, "geometry_100m": {}, "grad_norm_log": {}}
    for a, b in (("min_only", "full"), ("max_only", "full"), ("min_only", "detach"), ("max_only", "detach"),
                 ("min_only", "softmax"), ("max_only", "softmax"), ("min_only", "max_only"), ("detach", "full")):
        out["contrasts_100m"][f"{a} - {b}"] = boot(pb(a, 100) - pb(b, 100))
    nf, nd = pb("full", 100).mean(), pb("detach", 100).mean()
    out["gap_recovery"] = {k: float((nd - pb(k, 100).mean()) / (nd - nf)) for k in ("min_only", "max_only")}
    for k in SER:
        out["abs_nll"][k] = {t: float(pb(k, t).mean()) for t in T}
        out["geometry_100m"][k] = geom(k, 100)
        out["grad_norm_log"][k] = {t: gn(k, t) for t in T}
    for k in ("detach", "min_only", "max_only"):
        out["trajectory_vs_full"][k] = {t: float(pb(k, t).mean() - pb("full", t).mean()) for t in T}
    zs = f"{SUP}/reports/supplement_e1_zero_sum_loss_gradient_s{SEED}.csv"
    rows = list(csv.DictReader(open(zs)))
    out["zero_sum_100m"] = {k: {"rho_median_layer_mean": float(np.mean([float(r["rho_median"]) for r in rows if r["series"] == k and r["tokens_m"] == "100"])),
                                 "frac_gt_1e-3_layer_mean": float(np.mean([float(r["frac_rho_gt_1e_3"]) for r in rows if r["series"] == k and r["tokens_m"] == "100"]))}
                            for k in ("lerp_full", "lerp_detach", "e1_min", "e1_max")}
    json.dump(out, open(f"{SUP}/reports/supplement_e1_seed{SEED}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
