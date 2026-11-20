#!/usr/bin/env python3
"""Stage B support: realised paired-SD shrinkage, measured on data already collected.

The pre-registered detection floor 2*SD/sqrt(n) uses the MARGINAL per-example SD, which is the correct floor
only if the condition and Exact correlate at rho = 0.5. What actually enters a paired test is SD(delta), where
delta is the per-example difference. This script measures

    shrink = SD(x_cond - x_exact) / SD(x_exact)      and     rho = corr(x_cond, x_exact)

on every per-example set we already have: layer-1 per-block NLL on WikiText/PTB/C4 (natural text, the closest
analogue to a downstream task's log-likelihoods) and the layer-2 probes. Read-only.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
MAN = f"{os.path.dirname(BASE)}/manifests/primary_37_runs.tsv"
OUT = f"{BASE}/results/layer3/stageB"
TIER = {**{c: "pathological" for c in ["MinMax-Weight K4", "MinMax-Weight K16", "LERP K1", "MinMax-Weight K32",
                                        "QRM-Weight K2"]},
        **{c: "intermediate" for c in ["MinMax-Prob K4", "QRM-Weight K4", "MinMax-Weight K64", "QRM-Prob K4"]},
        **{c: "good" for c in ["LERP K2", "LERP K4", "QRM-Weight K16", "LERP K32", "LERP K16"]}}


def conds():
    r = pd.read_csv(MAN, sep="\t")
    r = r[r.suite == "A_longhorizon_124m_2p5b"].copy()
    r["cond"] = [l if l == "Exact" else f"{l} K{int(float(k))}" for l, k in zip(r.label, r.K)]
    return {x.cond: x.run_name for x in r.itertuples()}


def layer1_rows(cmap):
    f = f"{BASE}/results/layer1/per_block_rows.csv.gz"
    if not os.path.exists(f):
        return []
    d = pd.read_csv(f)
    key = "run_name" if "run_name" in d.columns else "run_id"
    d = d[d.forward_mode == "native"]
    dsc = "corpus"
    val = [c for c in ["nll", "per_block_nll", "loss"] if c in d.columns][0]
    idx = "block_id"
    ex = cmap["Exact"]
    out = []
    for ds, g in (d.groupby(dsc) if dsc else [("all", d)]):
        e = g[g[key] == ex].sort_values(idx)[val].to_numpy()
        for cond, run in cmap.items():
            if cond == "Exact":
                continue
            x = g[g[key] == run].sort_values(idx)[val].to_numpy()
            if len(x) != len(e) or len(e) < 10:
                continue
            out.append(dict(source=f"layer1:{ds}", condition=cond, tier=TIER.get(cond, "?"), n=len(e),
                            sd_exact=float(e.std(ddof=1)), sd_delta=float((x - e).std(ddof=1)),
                            rho=float(np.corrcoef(x, e)[0, 1]), mean_delta=float((x - e).mean())))
    return out


def layer2_rows(cmap):
    PS = f"{BASE}/results/layer2/per_sample"
    out = []
    for probe in ["A", "B", "C"]:
        ef = f"{PS}/{cmap['Exact']}__{probe}__native.npz"
        if not os.path.exists(ef):
            continue
        e = np.load(ef)["nll"]
        for cond, run in cmap.items():
            if cond == "Exact":
                continue
            f = f"{PS}/{run}__{probe}__native.npz"
            if not os.path.exists(f):
                continue
            x = np.load(f)["nll"]
            out.append(dict(source=f"layer2:probe{probe}", condition=cond, tier=TIER.get(cond, "?"), n=len(e),
                            sd_exact=float(e.std(ddof=1)), sd_delta=float((x - e).std(ddof=1)),
                            rho=float(np.corrcoef(x, e)[0, 1]), mean_delta=float((x - e).mean())))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    cmap = conds()
    d = pd.DataFrame(layer1_rows(cmap) + layer2_rows(cmap))
    d["shrink"] = d.sd_delta / d.sd_exact
    d.to_csv(f"{OUT}/paired_shrinkage.csv", index=False)
    piv = d.pivot_table(index=["source", "tier"], values=["shrink", "rho"], aggfunc="median").reset_index()
    print(piv.round(4).to_string(index=False))
    summ = d.groupby("tier").shrink.describe()[["count", "min", "50%", "max"]]
    print("\nshrink = SD(cond - exact) / SD(exact), all sources pooled")
    print(summ.round(4).to_string())
    json.dump({t: dict(median=float(g.shrink.median()), lo=float(g.shrink.min()), hi=float(g.shrink.max()),
                       rho_median=float(g.rho.median()))
               for t, g in d.groupby("tier")}, open(f"{OUT}/paired_shrinkage.json", "w"), indent=2)


if __name__ == "__main__":
    main()
