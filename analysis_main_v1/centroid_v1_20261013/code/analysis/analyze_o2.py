#!/usr/bin/env python3
"""O2 summary tables: forward and gradient fidelity per rule x calibration x K.

  table_exact100: mean over the five Exact 100M endpoints (layer means of 4 blocks each); SD over seeds for dS_cos
  table_traj:     Exact seed 7 at 20/40/60/80/100M, K = 4 and 16
  table_trained:  K4-trained endpoints at their own operating point (own calibration, K = 4)
Output: results/o2_rows_grads/summary.json and CSVs."""
import csv
import glob
import json
import os

import numpy as np
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))  # repository root

C = (KI_REPO + "/analysis_main_v1/centroid_v1_20261013")
O = os.path.join(C, "results", "o2_rows_grads")
RULES = ("nearest_wste", "nearest_pste", "lerp", "centroid", "centroid2", "centroid0", "centroid_int")
FWD = ("tv", "out_rel", "cell_mass_err", "cell_mass_err_window", "nonfinite_row_frac")
GRD = ("dS_cos", "dS_norm_ratio", "dQ_cos", "dK_cos", "dV_cos", "dS_rel")
CELL = ("contrast_norm2_ratio", "tail_share", "exact_tail_mass_mean", "mass_in_cells_ge2",
        "lme_minus_gridpoint_interior_mean", "h2_over_24_interior_mean",
        "lme_minus_gridpoint_over_h2_24_interior_median", "c2_log_gap_abs_mean", "c2_log_gap_abs_massw",
        "c0_log_gap_abs_massw", "c2_exceeds_max_member_share", "c2_exceeds_max_member_massw")


def load(name):
    p = os.path.join(O, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def flat(lm):
    rows = []
    for ck, v in lm.items():
        cal, K = ck.split("|")
        base = dict(cal=cal, K=int(K[1:]))
        cells = v["cells"]
        for rule in RULES:
            r = dict(base, rule=rule)
            r.update({m: v[rule].get(m) for m in FWD + GRD})
            r.update({m: cells.get(m) for m in CELL})
            rows.append(r)
    return rows


def main():
    out = {}
    ex = [load(f"Exact_s{s}_100m") if s != 7 else load("Exact_s7_100m") for s in (7, 42, 253, 1337, 2026)]
    ex = [e for e in ex if e]
    if ex:
        per = [flat(e["layer_mean"]) for e in ex]
        tab = []
        for i, r in enumerate(per[0]):
            rec = {k: r[k] for k in ("cal", "K", "rule")}
            for m in FWD + GRD + CELL:
                vals = np.array([p[i][m] if p[i][m] is not None else np.nan for p in per], dtype=float)
                rec[m] = float(np.mean(vals))
                if m in ("dS_cos", "out_rel", "dS_norm_ratio"):
                    rec[m + "_sd_seeds"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            tab.append(rec)
        out["table_exact100"] = dict(n_seeds=len(ex), rows=tab)
        with open(os.path.join(O, "table_exact100.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tab[0].keys()))
            w.writeheader()
            w.writerows(tab)
    traj = {}
    for t in (20, 40, 60, 80, 100):
        e = load(f"Exact_s7_{t:03d}m")
        if e:
            traj[t] = [r for r in flat(e["layer_mean"]) if r["K"] in (4, 16)]
    out["table_traj"] = traj
    trained = {}
    for f in sorted(glob.glob(os.path.join(O, "*.json"))):
        name = os.path.basename(f)[:-5]
        if name.startswith("Exact") or name in ("summary",):
            continue
        e = json.load(open(f))
        rule, cal, K = e["own_operator"]
        trained[name] = dict(own=e["own_operator"],
                             rows=[r for r in flat(e["layer_mean"]) if r["cal"] == cal and r["K"] == K],
                             loss_blocks=e["loss_blocks"])
    out["table_trained"] = trained
    json.dump(out, open(os.path.join(O, "summary.json"), "w"), indent=1)
    if "table_exact100" in out:
        print(f"Exact 100M, {out['table_exact100']['n_seeds']} seeds")
        for r in out["table_exact100"]["rows"]:
            if r["K"] in (4, 16):
                print(f"{r['cal']:6s} K{r['K']:<3d} {r['rule']:13s} tv {r['tv']:.3f} out {r['out_rel']:.3f} "
                      f"mass {r['cell_mass_err_window']:.3f} | dS cos {r['dS_cos']:.3f} ratio {r['dS_norm_ratio']:.3f} "
                      f"dQ {r['dQ_cos']:.3f} dK {r['dK_cos']:.3f} | contrast {r['contrast_norm2_ratio']:.3f}")


if __name__ == "__main__":
    main()
