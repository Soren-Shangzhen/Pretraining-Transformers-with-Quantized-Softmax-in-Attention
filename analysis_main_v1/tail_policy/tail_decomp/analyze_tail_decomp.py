#!/usr/bin/env python3
"""§7 analysis: tail-policy decomposition, seed 1337, K=4, tau=6.

Every contrast is paired per validation block against the suite D seed-1337 runs. Uncertainty is a
circular moving-block bootstrap over the 243 blocks (L=16, B=4000, 95% percentile), matching the
paper; it reflects evaluation sampling only, not training-seed variation. |Delta| < 0.01 is flagged
"below single-seed interpretation margin". Numbers only.
"""
import csv, json
import numpy as np
from pathlib import Path
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


AM = Path(KI_REPO + "/analysis_main_v1")
E = AM / "D_robustness_124m_100m_5seed/eval_native_v1"
OUT = Path(__file__).resolve().parent
MARGIN = 0.01
RUNS = {"Exact": "Exact_s1337",
        ("MinMax", "Weight"): "MinMax-Weight_K4_s1337", ("MinMax", "Prob"): "MinMax-Prob_K4_s1337",
        ("zero", "Weight"): "QRM-Weight_K4_s1337",     ("zero", "Prob"): "QRM-Prob_K4_s1337",
        ("resolved", "Weight"): "rb_s1337_tail_resolved_weight_100m",
        ("resolved", "Prob"): "rb_s1337_tail_resolved_prob_100m"}


def endpoint(run):
    d = [x for x in (E / run).iterdir() if x.name.startswith("100m") and ".tmp_" not in x.name]
    assert len(d) == 1, (run, d)
    return np.load(d[0] / "per_block_nll.npy")


X = {k: endpoint(v) for k, v in RUNS.items()}
n = len(X["Exact"]); assert all(len(v) == n for v in X.values())
rng = np.random.default_rng(20260917)
B, L = 4000, 16
st = rng.integers(0, n, (B, int(np.ceil(n / L))))
IDX = (st[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % n


def ci(d):
    b = d[IDX].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


rows = []
def add(section, surrogate, contrast, d):
    m, lo, hi = ci(d)
    rows.append(dict(section=section, surrogate=surrogate, contrast=contrast, delta=m,
                     ci_lo=lo, ci_hi=hi, ci_excludes_zero=bool(lo > 0 or hi < 0),
                     below_margin=bool(abs(m) < MARGIN)))

for s in ("Weight", "Prob"):
    for pol in ("MinMax", "zero", "resolved"):
        add("1_vs_exact", s, f"{pol} - Exact", X[(pol, s)] - X["Exact"])
    add("2_decomposition", s, "MinMax - resolved (resolution)", X[("MinMax", s)] - X[("resolved", s)])
    add("2_decomposition", s, "resolved - zero (truncation)", X[("resolved", s)] - X[("zero", s)])

ratios = []
for s in ("Weight", "Prob"):
    num = X[("MinMax", s)] - X[("resolved", s)]
    den = X[("MinMax", s)] - X[("zero", s)]
    rb = num[IDX].mean(1) / den[IDX].mean(1)
    lo, hi = np.percentile(rb, [2.5, 97.5])
    ratios.append(dict(surrogate=s, ratio=float(num.mean() / den.mean()), ci_lo=float(lo), ci_hi=float(hi)))

for pol in ("MinMax", "zero", "resolved"):
    add("3_surrogate_effect", "Prob - Weight", f"under {pol}", X[(pol, "Prob")] - X[(pol, "Weight")])

with open(OUT / "tail_decomp_contrasts.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
with open(OUT / "tail_decomp_ratio.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ratios[0]), lineterminator="\n"); w.writeheader(); w.writerows(ratios)
with open(OUT / "tail_decomp_mean_nll.csv", "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["condition", "run", "mean_nll", "n_blocks"])
    for k, v in RUNS.items():
        w.writerow([k if isinstance(k, str) else f"{k[0]}-{k[1]}", v, f"{X[k].mean():.6f}", n])

print(f"n_blocks={n}; bootstrap L={L}, B={B}; margin |Delta|<{MARGIN}\n")
print("mean NLL:")
for k, v in RUNS.items():
    print(f"  {(k if isinstance(k,str) else k[0]+'-'+k[1]):16s} {X[k].mean():.6f}")
for sec in ("1_vs_exact", "2_decomposition", "3_surrogate_effect"):
    print(f"\n{sec}:")
    for r in rows:
        if r["section"] == sec:
            f = "  [below single-seed interpretation margin]" if r["below_margin"] else ""
            print(f"  {r['surrogate']:13s} {r['contrast']:32s} {r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]{f}")
print("\nratio (MinMax - resolved)/(MinMax - zero):")
for r in ratios:
    print(f"  {r['surrogate']:13s} {r['ratio']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
