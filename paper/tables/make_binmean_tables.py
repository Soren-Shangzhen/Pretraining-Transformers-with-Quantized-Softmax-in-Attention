#!/usr/bin/env python3
"""Tables of the bin-mean ("Centroid") reconstruction control (Appendix J).

Reads analysis_main_v1/centroid_v1_20261013/results/** and writes, next to this script,
  tab_binmean_train.tex   training from scratch (Table S54)
  tab_binmean_2x2.tex     forward--backward decomposition (Table S55)
  tab_binmean_grad.tex    score-gradient alignment (Table S56)
  tab_binmean_fwd.tex     frozen-model substitution (Table S57)
and analysis_main_v1/centroid_v1_20261013/manifests/gpu_hours_from_training_logs.csv (GPU hours of the five runs, from
training_code/phaseB_base/centroid_runs_20261013/; the bin-mean row of Table S4 is written by make_paper_tables.py
from the same logs). Each file is the tabular of the corresponding table of Appendix J, which the paper's source holds
inline. Run from anywhere; KI_REPO overrides the repository root.
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = os.environ.get("KI_REPO", str(Path(__file__).resolve().parents[2]))   # the root of this repository
C = os.path.join(REPO, "analysis_main_v1", "centroid_v1_20261013")
OUT = str(Path(__file__).resolve().parent)

def sgn(x, nd=4): return f"{x:+.{nd}f}"
def ci(lo, hi, nd=4): return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"

# --- Table S54: training from scratch ------------------------------------------------------------------------------
con = pd.read_csv(os.path.join(C, "results/training/contrasts.csv")).set_index("contrast (seed 7, 100M)")
d = lambda k: float(con.loc[k, "delta"])
fw, mm = d("FW-Centroid K4 - softmax"), d("MinMax-Centroid K4 - softmax")
rows = [
 ("\\FW bin-mean", f"${sgn(fw)}$", ci(con.loc["FW-Centroid K4 - softmax", "ci_lo"], con.loc["FW-Centroid K4 - softmax", "ci_hi"])),
 ("MinMax bin-mean", f"${sgn(mm)}$", ci(con.loc["MinMax-Centroid K4 - softmax", "ci_lo"], con.loc["MinMax-Centroid K4 - softmax", "ci_hi"])),
 None,
 ("\\FW--Prob (strict forward, Appendix~\\ref{app:strict})", f"${sgn(fw - d('FW-Centroid K4 - FWM-Prob K4 strict'))}$", "control"),
 ("\\FW--Weight (historical)", f"${sgn(fw - d('FW-Centroid K4 - FWM-Weight K4 (hist)'))}$", "control"),
 ("MinMax--Prob (historical)", f"${sgn(mm - d('MinMax-Centroid K4 - MinMax-Prob K4 (hist)'))}$", "control"),
 ("MinMax--Weight (historical)", f"${sgn(mm - d('MinMax-Centroid K4 - MinMax-Weight K4 (hist)'))}$", "control"),
 ("LERP $K{=}4$", f"${sgn(fw - d('FW-Centroid K4 - LERP K4'))}$", "control"),
]
body = ["\\begin{tabular}{lrl}", "\\toprule", "Condition (seed 7, $K{=}4$, 100M) & $\\dNLL$ & 95\\% CI \\\\", "\\midrule"]
for r in rows: body.append("\\midrule" if r is None else " & ".join(r) + " \\\\")
body += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_binmean_train.tex"), "w").write("\n".join(body) + "\n")

# --- Table S55: forward--backward decomposition --------------------------------------------------------------------
add = json.load(open(os.path.join(C, "results/addendum/addendum.json")))
nf = add["contrasts_100m"]["NfCb - softmax"]
body = ["\\begin{tabular}{lrr}", "\\toprule", "forward $\\backslash$ backward & LERP surrogate (Prob-STE) & bin-mean \\\\", "\\midrule",
        f"hard \\CH & ${sgn(fw - d('FW-Centroid K4 - FWM-Prob K4 strict'))}$ & ${sgn(nf['delta'])}$ {ci(*nf['ci95'])} \\\\",
        f"bin-mean & not trained & ${sgn(fw)}$ \\\\", "\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_binmean_2x2.tex"), "w").write("\n".join(body) + "\n")

# --- Table S56: score-gradient alignment (O2) ------------------------------------------------------------------------
o2 = pd.read_csv(os.path.join(C, "results/o2_rows_grads/table_exact100.csv"))
def o2v(cal, K, rule, col): return float(o2[(o2.cal == cal) & (o2.K == K) & (o2.rule == rule)][col].iloc[0])
body = ["\\begin{tabular}{lrlll}", "\\toprule",
        "Calibration & $K$ & bin-mean cos / norm ratio & Prob-STE cos / norm ratio & Weight-STE cos \\\\", "\\midrule"]
for cal, K in [("fw", 2), ("fw", 4), ("fw", 8), ("fw", 16), ("fw", 64), ("minmax", 4), ("minmax", 16)]:
    body.append(f"{'FWM' if cal == 'fw' else 'MinMax'} & {K} & {o2v(cal,K,'centroid','dS_cos'):.3f} / {o2v(cal,K,'centroid','dS_norm_ratio'):.3f} & "
                f"{o2v(cal,K,'nearest_pste','dS_cos'):.3f} / {o2v(cal,K,'nearest_pste','dS_norm_ratio'):.3f} & {o2v(cal,K,'nearest_wste','dS_cos'):.3f} \\\\")
body += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_binmean_grad.tex"), "w").write("\n".join(body) + "\n")

# --- Table S57: frozen-model substitution (O1) -----------------------------------------------------------------------
o1 = pd.read_csv(os.path.join(C, "results/o1_substitution/summary.csv"))
def o1v(cal, K, rule): return float(o1[(o1.cal == cal) & (o1.K == K) & (o1.rule == rule)]["mean"].iloc[0])
body = ["\\begin{tabular}{lrrrr}", "\\toprule", "Calibration & $K$ & \\CH & bin-mean & LERP \\\\", "\\midrule"]
for cal, K in [("fw", 4), ("fw", 8), ("fw", 16), ("fw", 32), ("minmax", 4), ("minmax", 16)]:
    nd = 4 if cal == "fw" else 3
    body.append(f"{'FWM' if cal == 'fw' else 'MinMax'} & {K} & " + " & ".join(sgn(o1v(cal, K, r), nd) for r in ("nearest", "centroid", "lerp")) + " \\\\")
body += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_binmean_fwd.tex"), "w").write("\n".join(body) + "\n")

# --- GPU hours of the five runs, from the training logs ------------------------------------------------------
LOGS = os.path.join(REPO, "training_code", "phaseB_base", "centroid_runs_20261013")
recs = []
for run in sorted(os.listdir(LOGS)):
    l = pd.read_csv(os.path.join(LOGS, run, "log.csv")); l = l[l.tok_per_s > 0]
    dt = np.diff(np.r_[0, l.tokens.values]); recs.append((run, float((dt / l.tok_per_s.values).sum() / 3600)))
gh = pd.DataFrame(recs, columns=["run", "gpu_hours"]); gh.loc[len(gh)] = ["TOTAL", gh.gpu_hours.sum()]
gh.to_csv(os.path.join(C, "manifests", "gpu_hours_from_training_logs.csv"), index=False)
print("wrote 4 tables to", OUT, "; GPU hours total %.2f over %d runs" % (gh.gpu_hours.iloc[-1], len(gh) - 1))
