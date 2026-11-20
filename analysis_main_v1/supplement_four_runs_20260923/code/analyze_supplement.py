#!/usr/bin/env python3
"""Supplement analysis (local): endpoint contrasts, trajectories, evaluator-hardware shift.

Evaluations (all: 243 WikiText-103 val blocks, batch 1, bf16 autocast outside fp32 attention, TF32 off,
each checkpoint through an explicitly named frozen code root):
  L3090 controls : D_robustness_124m_100m_5seed/eval_native_v1/<control>/  (historical, local RTX 3090)
  L3090 new      : eval/rtx3090/<run>/100m_*                              (new endpoints, local RTX 3090)
  H5090          : eval/rtx5090/<label>/<tok>m_*                              (new runs all checkpoints; old Prob
                                                                              checkpoints native and strict-forward)
Contrasts are paired over blocks and only formed between evaluations on the SAME GPU; every row
records its evaluator.  Bootstrap: circular moving-block, block 16, 4000 resamples, 95% percentile,
numpy RNG seed 20260920 re-seeded per contrast (so a CI does not depend on which other rows exist).  CIs are evaluation-sample uncertainty only.
"""
import csv, glob, json, os
import numpy as np
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


TR = KI_REPO + ""
SUP = f"{TR}/analysis_main_v1/supplement_four_runs_20260923"
HIST = f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1"
N_BOOT, BOOT_BLOCK, RNG_SEED = 4000, 16, 20260920
TOK = [20, 40, 60, 80, 100]

CTRL = {"softmax": "Exact_s7", "lerp_full": "LERP_K32_s7", "lerp_detach": "LERP-Detach_K32_s7",
        "mm_weight_old": "MinMax-Weight_K4_s7", "mm_prob_old": "MinMax-Prob_K4_s7",
        "fw_weight_old": "QRM-Weight_K4_s7", "fw_prob_old": "QRM-Prob_K4_s7"}
NEW = {"mm_prob_new": "sp_s7_minmax_nearest_k4_probste_strict_100m",
       "mm_weight_new": "sp_s7_minmax_nearest_k4_weightste_strict_100m",
       "fw_prob_new": "sp_s7_fwm_nearest_k4_tau6_probste_strict_100m",
       "fw_weight_new": "sp_s7_fwm_nearest_k4_tau6_weightste_strict_100m",
       "e1_min": "sp_s7_lerp_k32_minonly_100m", "e1_max": "sp_s7_lerp_k32_maxonly_100m"}
OLDP_H = {"mm_prob_old__native": "C-MM-P4__native", "mm_prob_old__strict": "C-MM-P4__strictfwd",
          "fw_prob_old__native": "C-FW-P4__native", "fw_prob_old__strict": "C-FW-P4__strictfwd"}


def boot(d, rng):
    n = len(d)
    nb = int(np.ceil(n / BOOT_BLOCK))
    st = rng.integers(0, n, size=(N_BOOT, nb))
    idx = (st[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]).reshape(N_BOOT, -1)[:, :n] % n
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def pb(path_glob):
    m = sorted(glob.glob(path_glob))
    if not m:
        return None
    return np.load(os.path.join(m[0], "per_block_nll.npy"))


def load_all():
    ev = {}   # (evaluator, key, tokens) -> per-block array
    for k, d in CTRL.items():
        for t in TOK:
            a = pb(f"{HIST}/{d}/{t:03d}m_*")
            if a is not None:
                ev[("L3090", k, t)] = a
    for k, d in NEW.items():
        for t in TOK:
            a = pb(f"{SUP}/eval/rtx5090/{d}/{t:03d}m_*")
            if a is not None:
                ev[("H5090", k, t)] = a
        for t in (TOK if k.startswith("e1") else [100]):   # E1 trained + evaluated on the local 3090
            a = pb(f"{SUP}/eval/rtx3090/{d}/{t:03d}m_*")
            if a is not None:
                ev[("L3090", k, t)] = a
    for k, d in OLDP_H.items():
        for t in TOK:
            a = pb(f"{SUP}/eval/rtx5090/{d}/{t:03d}m_*")
            if a is not None:
                ev[("H5090", k, t)] = a
    return ev


CONTRASTS = [
    # (id, a, b, purpose, evaluator)
    ("E2-MM new strict Prob - new strict Weight", "mm_prob_new", "mm_weight_new", "K4 surrogate contrast, same code/stack, strict hard forward in both (backward-only)", "H5090"),
    ("E2-FWM new strict Prob - new strict Weight", "fw_prob_new", "fw_weight_new", "K4 surrogate contrast, same code/stack, strict hard forward in both (backward-only)", "H5090"),
    ("E2-MM new strict Prob - new strict Weight [3090 eval]", "mm_prob_new", "mm_weight_new", "same contrast, evaluated on the local RTX 3090", "L3090"),
    ("E2-FWM new strict Prob - new strict Weight [3090 eval]", "fw_prob_new", "fw_weight_new", "same contrast, evaluated on the local RTX 3090", "L3090"),
    ("E2-MM new Prob - old Prob (old native forward)", "mm_prob_new", "mm_prob_old__native", "historical vs supplement protocol, total difference", "H5090"),
    ("E2-FWM new Prob - old Prob (old native forward)", "fw_prob_new", "fw_prob_old__native", "historical vs supplement protocol, total difference", "H5090"),
    ("E2-MM new Prob - old Prob (strict forward)", "mm_prob_new", "mm_prob_old__strict", "learned weights under a common evaluation operator", "H5090"),
    ("E2-FWM new Prob - old Prob (strict forward)", "fw_prob_new", "fw_prob_old__strict", "learned weights under a common evaluation operator", "H5090"),
    ("E2-MM old Prob native - old Prob strict", "mm_prob_old__native", "mm_prob_old__strict", "frozen-weight implementation difference", "H5090"),
    ("E2-FWM old Prob native - old Prob strict", "fw_prob_old__native", "fw_prob_old__strict", "frozen-weight implementation difference", "H5090"),
    ("E2-MM new Prob - old Weight (historical, non-strict fwd, 3090-trained)", "mm_prob_new", "mm_weight_old", "NOT backward-only: old Weight failed the reuse verification", "L3090"),
    ("E2-FWM new Prob - old Weight (historical, non-strict fwd, 3090-trained)", "fw_prob_new", "fw_weight_old", "NOT backward-only: old Weight failed the reuse verification", "L3090"),
    ("E2-MM new strict Weight - old Weight", "mm_weight_new", "mm_weight_old", "historical Weight control vs strict Weight replacement", "L3090"),
    ("E2-FWM new strict Weight - old Weight", "fw_weight_new", "fw_weight_old", "historical Weight control vs strict Weight replacement", "L3090"),
    ("E2-MM new Prob - softmax", "mm_prob_new", "softmax", "absolute gap to the baseline", "L3090"),
    ("E2-FWM new Prob - softmax", "fw_prob_new", "softmax", "absolute gap to the baseline", "L3090"),
    ("E2-MM new strict Weight - softmax", "mm_weight_new", "softmax", "absolute gap to the baseline", "L3090"),
    ("E2-FWM new strict Weight - softmax", "fw_weight_new", "softmax", "absolute gap to the baseline", "L3090"),
    ("HIST E2-MM old Prob - old Weight", "mm_prob_old", "mm_weight_old", "historical seed-7 contrast (reference)", "L3090"),
    ("HIST E2-FWM old Prob - old Weight", "fw_prob_old", "fw_weight_old", "historical seed-7 contrast (reference)", "L3090"),
    ("E1 min_only - full", "e1_min", "lerp_full", "primary E1 metric", "L3090"),
    ("E1 max_only - full", "e1_max", "lerp_full", "primary E1 metric", "L3090"),
    ("E1 min_only - detach", "e1_min", "lerp_detach", "E1 secondary", "L3090"),
    ("E1 max_only - detach", "e1_max", "lerp_detach", "E1 secondary", "L3090"),
    ("E1 min_only - softmax", "e1_min", "softmax", "E1 secondary", "L3090"),
    ("E1 max_only - softmax", "e1_max", "softmax", "E1 secondary", "L3090"),
    ("E1 min_only - max_only", "e1_min", "e1_max", "same code/stack", "L3090"),
    ("HIST E1 detach - full", "lerp_detach", "lerp_full", "historical seed-7 contrast (reference)", "L3090"),
]


def main():
    ev = load_all()
    rng = np.random.default_rng(RNG_SEED)
    rows, arrays = [], {}
    for cid, a, b, why, e in CONTRASTS:
        A, B = ev.get((e, a, 100)), ev.get((e, b, 100))
        if A is None or B is None:
            rows.append(dict(contrast=cid, a=a, b=b, evaluator=e, purpose=why, status="missing",
                             missing=",".join(x for x, y in ((a, A), (b, B)) if y is None)))
            continue
        d = A - B
        m, lo, hi = boot(d, np.random.default_rng(RNG_SEED))
        arrays[cid] = d
        rows.append(dict(contrast=cid, a=a, b=b, evaluator=e, purpose=why, status="ok", n_blocks=len(d),
                         mean_a=float(A.mean()), mean_b=float(B.mean()), delta=m, ci_lo=lo, ci_hi=hi,
                         forward_mode_a=("strict hard" if a.endswith("_new") and not a.startswith("e1") else "native"),
                         forward_mode_b=("strict hard" if (b.endswith("_new") and not b.startswith("e1")) or b.endswith("__strict") else "native")))
    # interaction of the strict pairs
    for e in ("H5090", "L3090"):
        k = [(e, x, 100) for x in ("mm_prob_new", "mm_weight_new", "fw_prob_new", "fw_weight_new")]
        if all(x in ev for x in k):
            d = (ev[k[0]] - ev[k[1]]) - (ev[k[2]] - ev[k[3]])
            m, lo, hi = boot(d, np.random.default_rng(RNG_SEED))
            cid = f"E2 interaction (MM P-W) - (FWM P-W), strict pairs [{e}]"
            arrays[cid] = d
            rows.append(dict(contrast=cid, a="MM(P-W)", b="FWM(P-W)", evaluator=e, purpose="supplement interaction",
                             status="ok", n_blocks=len(d), delta=m, ci_lo=lo, ci_hi=hi))
    # E1 gap recovery (point estimates on the 3090 evaluator)
    rec = {}
    if all(("L3090", x, 100) in ev for x in ("lerp_full", "lerp_detach")):
        nf, nd = ev[("L3090", "lerp_full", 100)].mean(), ev[("L3090", "lerp_detach", 100)].mean()
        for x in ("e1_min", "e1_max"):
            if ("L3090", x, 100) in ev:
                nv = ev[("L3090", x, 100)].mean()
                rec[x] = float((nd - nv) / (nd - nf))
    # evaluator shift 3090 vs 5090 on identical checkpoints
    shift = []
    for k in NEW:
        if ("L3090", k, 100) in ev and ("H5090", k, 100) in ev:
            d = ev[("H5090", k, 100)] - ev[("L3090", k, 100)]
            shift.append(dict(checkpoint=f"{NEW[k]} 100M", mean=float(d.mean()), mean_abs=float(np.abs(d).mean()),
                              max_abs=float(np.abs(d).max())))
    for k, hk in (("mm_prob_old", "mm_prob_old__native"), ("fw_prob_old", "fw_prob_old__native")):
        for t in TOK:
            if ("L3090", k, t) in ev and ("H5090", hk, t) in ev:
                d = ev[("H5090", hk, t)] - ev[("L3090", k, t)]
                shift.append(dict(checkpoint=f"{CTRL[k]} {t}M", mean=float(d.mean()), mean_abs=float(np.abs(d).mean()),
                                  max_abs=float(np.abs(d).max())))
    cols = ["contrast", "a", "b", "evaluator", "status", "n_blocks", "mean_a", "mean_b", "delta", "ci_lo", "ci_hi",
            "forward_mode_a", "forward_mode_b", "purpose", "missing"]
    with open(f"{SUP}/reports/supplement_endpoint_contrasts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    np.savez(f"{SUP}/reports/supplement_endpoint_per_block_deltas.npz", **{k.replace(" ", "_"): v for k, v in arrays.items()})
    np.savez(f"{SUP}/reports/supplement_per_block_nll_all.npz",
             **{f"{e}__{k}__{t}m": v for (e, k, t), v in ev.items()})
    json.dump({"gap_recovery_L3090": rec, "evaluator_shift_5090_minus_3090": shift},
              open(f"{SUP}/reports/supplement_e1_recovery_and_eval_shift.json", "w"), indent=1)
    trajectories(ev)
    for r in rows:
        if r["status"] == "ok":
            print(f"{r['contrast']:75s} [{r['evaluator']}] {r['delta']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")
        else:
            print(f"{r['contrast']:75s} [{r['evaluator']}] MISSING {r.get('missing')}")
    print("gap recovery", rec)
    print("eval shift", json.dumps(shift, indent=0)[:1500])


def geom_means(path_glob):
    m = sorted(glob.glob(path_glob))
    if not m or not os.path.exists(os.path.join(m[0], "geometry_layers.csv")):
        return {}
    rows = list(csv.DictReader(open(os.path.join(m[0], "geometry_layers.csv"))))
    out = {}
    for c in ("span_mean", "span_p90", "width_native_mean", "q_norm_mean", "k_norm_mean", "exact_entropy_mean",
              "tv_native_exact_mean", "row_min_mean", "row_max_mean"):
        out[f"geom_{c}_layermean"] = float(np.mean([float(r[c]) for r in rows]))
    out["geom_span_mean_maxlayer"] = float(max(float(r["span_mean"]) for r in rows))
    return out


def log_at(logcsv, tokens_m):
    if not os.path.exists(logcsv):
        return {}
    rows = list(csv.DictReader(open(logcsv)))
    best = min(rows, key=lambda r: abs(int(r["tokens"]) / 1e6 - tokens_m))
    return {"log_step": best["step"], "log_train_loss": best["train_loss"], "log_val_loss_100batches": best["val_loss"],
            "log_grad_norm_preclip": best["grad_norm"]}


def trajectories(ev):
    PB = f"{TR}/training_code/phaseB_base"
    logs = {"softmax": f"{PB}/runs/p3_s7_exact_100m/log.csv", "lerp_full": f"{PB}/runs/p3_s7_lerp_k32_fullgrad_100m/log.csv",
            "lerp_detach": f"{PB}/runs_20260921/rb_s7_lerp_k32_detach_100m/log.csv",
            "mm_weight_old": f"{PB}/runs/p3_s7_nearest_k4_fullgrad_100m/log.csv",
            "fw_weight_old": f"{PB}/qrm_runs_v1/qrm_s7_nearest_k4_tau6_fullgrad_100m/log.csv",
            "mm_prob_old": f"{PB}/formal_runs_probste_5seeds_100m_v1/formal_124m_s7_minmax_nearest_k4_probste_100m/log.csv",
            "fw_prob_old": f"{PB}/formal_runs_probste_5seeds_100m_v1/formal_124m_s7_qrm_nearest_k4_tau6_probste_100m/log.csv"}
    for k, d in NEW.items():
        logs[k] = (f"{PB}/supplement_runs_20260923/{d}/log.csv" if k.startswith("e1")
                   else f"{SUP}/runs_small/{d}/log.csv")
    out = []
    for (e, k, t), a in sorted(ev.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        if e == "L3090" and k in CTRL:
            g = geom_means(f"{HIST}/{CTRL[k]}/{t:03d}m_*")
        elif e == "H5090" and k in NEW:
            g = geom_means(f"{SUP}/eval/rtx5090/{NEW[k]}/{t:03d}m_*")
        elif e == "H5090":
            g = geom_means(f"{SUP}/eval/rtx5090/{OLDP_H[k]}/{t:03d}m_*")
        else:
            g = geom_means(f"{SUP}/eval/rtx3090/{NEW[k]}/{t:03d}m_*")
        row = dict(series=k, evaluator=e, tokens_m=t, mean_nll_243=float(a.mean()), **g,
                   **log_at(logs.get(k.split("__")[0], ""), t),
                   clip_fraction="not logged (historical protocol: no per-step clip log)")
        out.append(row)
    cols = sorted({c for r in out for c in r}, key=lambda c: (c not in ("series", "evaluator", "tokens_m", "mean_nll_243"), c))
    with open(f"{SUP}/reports/supplement_trajectories.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
