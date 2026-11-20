#!/usr/bin/env python3
"""Layer 2 analysis: paired statistics, P3 (same-scale difference), P4 (distance slope), P5 (exact-forward
reversal on probes vs LM). Runs locally (pandas/scipy/matplotlib) on the mirrored results.

Outputs (results/layer2/):
  probe_scores.csv      condition x probe x config x mode, paired delta vs Exact, CI, BH q, detectable, A
  distance_curves.csv   probe A/B degradation by distance/position, with the P4 log2(distance) slope and CI
  nll_vs_probe.csv      the P3 scatter data
  p5_exact_forward.csv  R(c, task) = delta_exact - delta_native for probes vs LM
  figures/*.png
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / "results/layer2/per_sample"
DATA = ROOT / "data/probes"
OUT = ROOT / "results/layer2"
FIG = OUT / "figures"
MAIN = ROOT.parents[0] / "A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_table.csv"
L1 = ROOT / "results/layer1/ppl_transfer.csv"
N_BOOT, SEED = 4000, 20260920
TIER = {**{c: "pathological" for c in ["MinMax-Weight K4", "MinMax-Weight K16", "LERP K1", "MinMax-Weight K32",
                                        "QRM-Weight K2"]},
        **{c: "intermediate" for c in ["MinMax-Prob K4", "QRM-Weight K4", "MinMax-Weight K64", "QRM-Prob K4"]},
        **{c: "good" for c in ["LERP K2", "LERP K4", "QRM-Weight K16", "LERP K32", "LERP K16"]},
        "Exact": "reference"}
INK, GRID = "#0b0b0b", "#e6e5e0"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.grid": True,
                     "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
COLOR = {"pathological": "#eb6834", "intermediate": "#eda100", "good": "#2a78d6", "reference": "#52514e"}


def bh(p):
    p = np.asarray(p, float)
    n, order, q, prev = len(p), np.argsort(p), np.empty(len(p)), 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def boot_mean(d, idx):
    return d[idx].mean(axis=1)


def paired(a, b, rng):
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    m = boot_mean(d, idx)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return dict(n=len(d), delta=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                p_value=float(min(max(p, 1 / (N_BOOT + 1)), 1.0)), detectable=bool(lo > 0 or hi < 0))


def load_meta(probe):
    return pd.read_csv(DATA / f"probe_{probe}_meta.csv")


def load_scores(runs_df, probe, mode):
    out = {}
    for _, r in runs_df.iterrows():
        f = PS / f"{r.run_name}__{probe}__{mode}.npz"
        if f.exists():
            z = np.load(f)
            out[r.cond] = (z["nll"], z["hit"])
    return out


def main():
    runs_df = pd.read_csv(ROOT.parents[0] / "manifests/primary_37_runs.tsv", sep="\t")
    runs_df = runs_df[runs_df.suite == "A_longhorizon_124m_2p5b"].copy()
    runs_df["cond"] = [l if l == "Exact" else f"{l} K{int(float(k))}" for l, k in zip(runs_df.label, runs_df.K)]
    dnll_lm = pd.read_csv(MAIN).set_index("cond").delta_vs_exact      # FineWebEdu 2.5B endpoint (Table 23)
    rng = np.random.default_rng(SEED)
    FIG.mkdir(parents=True, exist_ok=True)

    rows, dist_rows, scatter = [], [], []
    for mode in ("native", "exact"):
        for probe in ("A", "B", "C"):
            md = load_meta(probe)
            sc = load_scores(runs_df, probe, mode)
            if "Exact" not in sc:
                continue
            groups = {"all": np.ones(len(md), bool)}
            for cfg in md.config.unique():
                groups[cfg] = (md.config == cfg).values
            for cond, (nll, hit) in sc.items():
                base_nll, base_hit = sc["Exact"]
                for gname, m in groups.items():
                    if probe == "A":
                        ind = (md.arm == "induction").values & m
                        ctl = (md.arm == "control").values & m
                        gain = nll[ctl] - nll[ind]
                        base_gain = base_nll[ctl] - base_nll[ind]
                        metrics = {"target_nll": (nll[ind], base_nll[ind]),
                                   "induction_gain": (gain, base_gain),
                                   "top1": (hit[ind], base_hit[ind])}
                    else:
                        metrics = {"target_nll": (nll[m], base_nll[m]), "top1": (hit[m], base_hit[m])}
                    for metric, (x, b) in metrics.items():
                        rec = dict(condition=cond, tier=TIER[cond], K=runs_df.set_index("cond").loc[cond, "K"],
                                   forward_mode=mode, probe=probe, config=gname, metric=metric, n=len(x),
                                   score=float(np.mean(x)))
                        if cond == "Exact":
                            rec.update(delta=0.0, ci_lo=0.0, ci_hi=0.0, p_value=1.0, detectable=False)
                        else:
                            rec.update({k: v for k, v in paired(x, b, rng).items() if k != "n"})
                        rows.append(rec)
    df = pd.DataFrame(rows)
    # BH within each confirmatory family (14 conditions vs Exact), per mode/probe/config/metric
    df["q_value"] = 1.0
    for key, grp in df[df.condition != "Exact"].groupby(["forward_mode", "probe", "config", "metric"]):
        df.loc[grp.index, "q_value"] = bh(grp.p_value.values)
    # P3: same-scale difference A = delta_probe - delta_LM (probe target-NLL only)
    df["delta_nll_lm"] = df.condition.map(dnll_lm)
    m = (df.metric == "target_nll") & (df.config == "all")
    df.loc[m, "A"] = df.loc[m, "delta"] - df.loc[m, "delta_nll_lm"]
    df.loc[m, "A_ci_lo"] = df.loc[m, "ci_lo"] - df.loc[m, "delta_nll_lm"]
    df.loc[m, "A_ci_hi"] = df.loc[m, "ci_hi"] - df.loc[m, "delta_nll_lm"]
    df.to_csv(OUT / "probe_scores.csv", index=False)

    # ---- P4: slope of paired target-NLL degradation vs log2(distance), probe A, native
    mdA = load_meta("A")
    scA = load_scores(runs_df, "A", "native")
    ind_mask = (mdA.arm == "induction").values
    fam = mdA.family.values[ind_mask]
    dist = mdA.distance.values[ind_mask]
    uniq_fam = np.unique(fam)
    fam_index = {f: i for i, f in enumerate(uniq_fam)}
    fidx = np.array([fam_index[f] for f in fam])
    dists = np.unique(dist)
    for cond, (nll, _) in scA.items():
        if cond == "Exact":
            continue
        d = nll[ind_mask] - scA["Exact"][0][ind_mask]
        per_d = {int(dd): d[dist == dd] for dd in dists}
        x = np.log2(dists.astype(float))
        y = np.array([per_d[int(dd)].mean() for dd in dists])
        slope = np.polyfit(x, y, 1)[0]
        boots = []
        for _ in range(1000):
            pick = rng.integers(0, len(uniq_fam), size=len(uniq_fam))
            sel = np.isin(fidx, pick)
            yb = np.array([d[(dist == dd) & sel].mean() for dd in dists])
            boots.append(np.polyfit(x, yb, 1)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        for dd in dists:
            dist_rows.append(dict(condition=cond, tier=TIER[cond], probe="A", distance=int(dd),
                                  delta_target_nll=float(per_d[int(dd)].mean())))
        dist_rows.append(dict(condition=cond, tier=TIER[cond], probe="A", distance=-1,
                              delta_target_nll=np.nan, p4_slope=float(slope), p4_ci_lo=float(lo),
                              p4_ci_hi=float(hi), p4_positive=bool(lo > 0)))
    # probe B near/far
    mdB = load_meta("B")
    scB = load_scores(runs_df, "B", "native")
    for cond, (nll, _) in scB.items():
        if cond == "Exact":
            continue
        for cfg in mdB.config.unique():
            mm = (mdB.config == cfg).values
            dist_rows.append(dict(condition=cond, tier=TIER[cond], probe="B", config=cfg,
                                  delta_target_nll=float((nll[mm] - scB["Exact"][0][mm]).mean())))
    pd.DataFrame(dist_rows).to_csv(OUT / "distance_curves.csv", index=False)

    # ---- P3 scatter data + figure
    sc_df = df[(df.metric == "target_nll") & (df.config == "all") & (df.forward_mode == "native")
               & (df.condition != "Exact")][["condition", "tier", "probe", "delta", "ci_lo", "ci_hi",
                                              "delta_nll_lm", "A", "A_ci_lo", "A_ci_hi"]]
    sc_df.to_csv(OUT / "nll_vs_probe.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharex=True)
    for ax, probe in zip(axes, ("A", "B", "C")):
        s = sc_df[sc_df.probe == probe]
        lim = [min(-0.05, s.delta.min() * 1.1), max(s.delta.max() * 1.1, s.delta_nll_lm.max() * 1.1, 0.1)]
        ax.plot(lim, lim, color="#52514e", lw=1, ls=(0, (4, 3)), label="y = x")
        for tier in ("pathological", "intermediate", "good"):
            t = s[s.tier == tier]
            ax.errorbar(t.delta_nll_lm, t.delta, yerr=[t.delta - t.ci_lo, t.ci_hi - t.delta], fmt="o", ms=5,
                        color=COLOR[tier], ecolor=COLOR[tier], elinewidth=1, capsize=2, label=tier)
        for _, r in s.iterrows():
            ax.annotate(r.condition.replace("MinMax-", "MM-").replace("QRM-", "Q-"), (r.delta_nll_lm, r.delta),
                        fontsize=6, xytext=(4, 2), textcoords="offset points", color=INK)
        ax.set_title({"A": "Probe A: induction target NLL", "B": "Probe B: key-value retrieval",
                      "C": "Probe C: prefix copy"}[probe], fontsize=10)
        ax.set_xlabel("Δ NLL on FineWebEdu LM (nats)")
    axes[0].set_ylabel("Δ probe target NLL (nats)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("P3: probe degradation vs LM degradation (points above y=x mean retrieval-specific amplification)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "p3_nll_vs_probe.png", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)

    # distance curve figure
    dc = pd.DataFrame(dist_rows)
    dcd = dc[(dc.probe == "A") & (dc.distance > 0)]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for cond, g in dcd.groupby("condition"):
        g = g.sort_values("distance")
        ax.plot(g.distance, g.delta_target_nll, marker="o", ms=3.5, lw=1.6,
                color=COLOR[TIER[cond]], alpha=0.85, label=cond)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("induction distance (tokens)")
    ax.set_ylabel("Δ target NLL vs Exact (nats)")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_title("Probe A: retrieval degradation vs distance (P4)", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "p4_distance_curves.png", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)

    # ---- P5: exact-forward reversal on probes vs LM
    l1 = pd.read_csv(L1)
    lm_r = (l1[l1.forward_mode == "exact"].groupby("condition").delta_vs_exact.mean()
            - l1[l1.forward_mode == "native"].groupby("condition").delta_vs_exact.mean())
    p5 = []
    for cond in df.condition.unique():
        if cond == "Exact" or cond not in lm_r.index:
            continue
        for probe in ("A", "B", "C"):
            n = df[(df.condition == cond) & (df.probe == probe) & (df.config == "all")
                   & (df.metric == "target_nll") & (df.forward_mode == "native")]
            e = df[(df.condition == cond) & (df.probe == probe) & (df.config == "all")
                   & (df.metric == "target_nll") & (df.forward_mode == "exact")]
            if not len(n) or not len(e):
                continue
            p5.append(dict(condition=cond, tier=TIER[cond], probe=probe,
                           delta_native=float(n.delta.iloc[0]), delta_exact=float(e.delta.iloc[0]),
                           R_probe=float(e.delta.iloc[0] - n.delta.iloc[0]), R_lm=float(lm_r[cond]),
                           P5_holds=bool(e.delta.iloc[0] - n.delta.iloc[0] > lm_r[cond])))
    pd.DataFrame(p5).to_csv(OUT / "p5_exact_forward.csv", index=False)
    json.dump(dict(n_rows=len(df), probes=["A", "B", "C"], boot=N_BOOT, seed=SEED),
              open(OUT / "layer2_meta.json", "w"), indent=2)
    print(df[(df.forward_mode == "native") & (df.config == "all") & (df.metric == "target_nll")]
          [["condition", "tier", "probe", "score", "delta", "ci_lo", "ci_hi", "q_value", "detectable", "A"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
