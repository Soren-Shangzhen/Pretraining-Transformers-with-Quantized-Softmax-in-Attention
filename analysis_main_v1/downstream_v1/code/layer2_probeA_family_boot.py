#!/usr/bin/env python3
"""Probe A, config 'all': recompute the paired CIs with a FAMILY-cluster bootstrap (2026-09-25 addendum).
The 2,000 induction samples are 400 families x 5 distances; the five samples of a family share the same base
sequence and (A,B) tokens, so a sample-level resample treats dependent items as independent. This script keeps the
original probe_scores.csv untouched and writes probe_scores_A_all_familyboot_20260925.csv with cluster CIs
(resample the 400 families with replacement, keep every distance / arm / method pairing inside a family)."""
import numpy as np, pandas as pd, glob, os
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

ROOT = KI_REPO + "/analysis_main_v1/downstream_v1"
PS = f"{ROOT}/results/layer2/per_sample"; OUT = f"{ROOT}/results/layer2"
md = pd.read_csv(f"{ROOT}/data/probes/probe_A_meta.csv")
ind = (md.arm == "induction").values; ctl = (md.arm == "control").values
fam_ind = md.family.values[ind]; fams = np.unique(fam_ind); fam_idx = {f: np.where(fam_ind == f)[0] for f in fams}
assert len(fams) == 400 and ind.sum() == 2000
old = pd.read_csv(f"{OUT}/probe_scores.csv")
old = old[(old.probe == "A") & (old.config == "all") & (old.forward_mode == "native")]
runs = {os.path.basename(f).split("__")[0]: f for f in glob.glob(f"{PS}/*__A__native.npz")}
def load(run):
    z = np.load(runs[run]); return z["nll"], z["hit"]
# map condition -> run name via probe_scores (condition, run) is not stored; rebuild from manifest
man = pd.read_csv(f"{ROOT}/../manifests/primary_37_runs.tsv", sep="\t"); man = man[man.suite == "A_longhorizon_124m_2p5b"]
man["cond"] = [l if l == "Exact" else f"{l} K{int(float(k))}" for l, k in zip(man.label, man.K)]
cond2run = dict(zip(man.cond, man.run_name))
rng = np.random.default_rng(20260920); NB = 4000
fam_samples = rng.integers(0, len(fams), size=(NB, len(fams)))
b_nll, b_hit = load(cond2run["Exact"])
rows = []
for cond in old.condition.unique():
    if cond == "Exact": continue
    nll, hit = load(cond2run[cond])
    mets = {"target_nll": nll[ind] - b_nll[ind], "induction_gain": (nll[ctl] - nll[ind]) - (b_nll[ctl] - b_nll[ind]), "top1": hit[ind] - b_hit[ind]}
    for metric, d in mets.items():
        fam_mean = np.array([d[fam_idx[f]].mean() for f in fams])      # each family contributes its 5-distance mean
        boots = fam_mean[fam_samples].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        o = old[(old.condition == cond) & (old.metric == metric)].iloc[0]
        rows.append(dict(condition=cond, metric=metric, n_samples=2000, n_families=400, delta=float(d.mean()),
                         ci_lo_family=lo, ci_hi_family=hi, ci_lo_sample=o.ci_lo, ci_hi_sample=o.ci_hi,
                         detectable_family=bool(lo > 0 or hi < 0), detectable_sample=bool(o.detectable)))
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/probe_scores_A_all_familyboot_20260925.csv", index=False)
chg = df[df.detectable_family != df.detectable_sample]
print(f"rows {len(df)}; detectability changes: {len(chg)}"); print(chg[["condition", "metric", "delta", "ci_lo_family", "ci_hi_family"]].to_string() if len(chg) else "none")
print("max CI half-width sample vs family:", round(float(((df.ci_hi_sample - df.ci_lo_sample) / 2).max()), 4), round(float(((df.ci_hi_family - df.ci_lo_family) / 2).max()), 4))
print(df[df.metric == "target_nll"][["condition", "delta", "ci_lo_sample", "ci_hi_sample", "ci_lo_family", "ci_hi_family"]].round(3).to_string())
