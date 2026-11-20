#!/usr/bin/env python3
"""Layer 2 diagnostics (no model re-run; reads results/layer2/per_sample + data/probes).

D1  Probe C scale: confirm per-token vs summed target NLL, and break the result down by segment length.
D2  P4 ceiling check: Exact's own induction gain vs distance, probe B position curve, matched-family audit,
    and, if the baseline decays, the two pre-specified re-tests (restricted distance range; relative
    degradation).
D3  Probe B: is the good-tier effect a retrieval loss or an output-sharpness effect? NLL vs top-1, with the
    uniform reference and the fraction of Exact's headroom.

Pre-specified threshold for D2(a): keep distances where Exact's induction gain is at least 50% of its d=64
value. Declared here before looking at the decay curve.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / "results/layer2/per_sample"
DATA = ROOT / "data/probes"
OUT = ROOT / "results/layer2/diagnostics"
N_BOOT, SEED = 4000, 20260920
GAIN_KEEP_FRAC = 0.50          # D2(a), pre-specified
VOCAB = 50257
TIER = {**{c: "pathological" for c in ["MinMax-Weight K4", "MinMax-Weight K16", "LERP K1", "MinMax-Weight K32",
                                        "QRM-Weight K2"]},
        **{c: "intermediate" for c in ["MinMax-Prob K4", "QRM-Weight K4", "MinMax-Weight K64", "QRM-Prob K4"]},
        **{c: "good" for c in ["LERP K2", "LERP K4", "QRM-Weight K16", "LERP K32", "LERP K16"]},
        "Exact": "reference"}
COLOR = {"pathological": "#eb6834", "intermediate": "#eda100", "good": "#2a78d6", "reference": "#52514e"}
INK = "#0b0b0b"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
rng = np.random.default_rng(SEED)


def ci(x, n_boot=N_BOOT):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return ci(d)


def conds():
    runs = pd.read_csv(ROOT.parents[0] / "manifests/primary_37_runs.tsv", sep="\t")
    runs = runs[runs.suite == "A_longhorizon_124m_2p5b"].copy()
    runs["cond"] = [l if l == "Exact" else f"{l} K{int(float(k))}" for l, k in zip(runs.label, runs.K)]
    return {r.cond: r.run_name for r in runs.itertuples()}


def load(run, probe, mode="native"):
    z = np.load(PS / f"{run}__{probe}__{mode}.npz")
    return z["nll"], z["hit"]


# ----------------------------------------------------------------- D1: probe C scale
def d1(cmap):
    md = pd.read_csv(DATA / "probe_C_meta.csv")
    z = np.load(DATA / "probe_C.npz")
    toff = z["target_offsets"]
    n_targets = np.diff(toff)
    seg = md.seg_len.values
    # the evaluator averaged cross-entropy over target positions -> per-token mean; verify the claim directly
    per_token = bool(np.all(n_targets == seg))          # targets == every token of the second copy
    ex = load(cmap["Exact"], "C")[0]
    by_len = {}
    for L in sorted(md.seg_len.unique()):
        m = (md.seg_len == L).values
        by_len[int(L)] = dict(n_samples=int(m.sum()), n_target_tokens=int(n_targets[m][0]),
                              exact_nll=float(ex[m].mean()))
    rows = []
    for cond, run in cmap.items():
        nll, hit = load(run, "C")
        for L in sorted(md.seg_len.unique()):
            m = (md.seg_len == L).values
            d, lo, hi = paired(nll[m], ex[m]) if cond != "Exact" else (0.0, 0.0, 0.0)
            rows.append(dict(condition=cond, tier=TIER[cond], seg_len=int(L), n=int(m.sum()),
                             target_tokens_per_sample=int(n_targets[m][0]),
                             nll_per_token=float(nll[m].mean()), delta_per_token=d, ci_lo=lo, ci_hi=hi,
                             top1=float(hit[m].mean()),
                             implied_sum_if_multiplied=float(nll[m].mean() * n_targets[m][0])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d1_probe_c_scale.csv", index=False)
    return dict(per_token_confirmed=per_token, targets_equal_segment_length=per_token,
                exact_by_length=by_len,
                note=("score() averages cross-entropy over target positions, so probe C values are per-token "
                      "means, not sums"))


# ----------------------------------------------------------------- D2: P4 ceiling check
def d2(cmap):
    md = pd.read_csv(DATA / "probe_A_meta.csv")
    z = np.load(DATA / "probe_A.npz")
    tok, off = z["tokens"], z["offsets"]
    ind = (md.arm == "induction").values
    ctl = (md.arm == "control").values
    dist = md.distance.values
    fam = md.family.values
    dists = sorted(md.distance.unique())

    # matched-family audit: for one family, induction samples at two distances must agree everywhere except
    # the four inserted positions; A and B must not occur elsewhere.
    audit = []
    fams = pd.unique(fam)[:20]
    for f in fams:
        idxs = [i for i in range(len(md)) if fam[i] == f and ind[i]]
        seqs = {int(dist[i]): tok[off[i]:off[i + 1]] for i in idxs}
        base_d = dists[0]
        b = seqs[base_d]
        q = len(b) - 2
        a_tok, b_tok = int(b[q]), int(b[q + 1])
        extra_a = int((b[:q] == a_tok).sum()) - 1      # one legitimate earlier A
        extra_b = int((b[:q] == b_tok).sum()) - 1
        for d in dists[1:]:
            s = seqs[d]
            mask = np.ones(len(b), bool)
            for p in (q - base_d, q - base_d + 1, q - d, q - d + 1, q, q + 1):
                mask[p] = False
            audit.append(dict(family=f, d_ref=base_d, d_other=d,
                              identical_elsewhere=bool(np.array_equal(b[mask], s[mask])),
                              len_equal=bool(len(b) == len(s)),
                              extra_A_occurrences=extra_a, extra_B_occurrences=extra_b))
    au = pd.DataFrame(audit)
    au.to_csv(OUT / "d2_matched_family_audit.csv", index=False)

    ex_nll = load(cmap["Exact"], "A")[0]
    rows = []
    gain_exact = {}
    for d in dists:
        mi = ind & (dist == d)
        mc = ctl & (dist == d)
        g, lo, hi = ci(ex_nll[mc] - ex_nll[mi])
        gain_exact[int(d)] = g
        rows.append(dict(condition="Exact", tier="reference", distance=int(d), gain=g, gain_ci_lo=lo,
                         gain_ci_hi=hi, delta_gain=0.0, rel_degradation=0.0,
                         target_nll=float(ex_nll[mi].mean())))
    for cond, run in cmap.items():
        if cond == "Exact":
            continue
        nll = load(run, "A")[0]
        for d in dists:
            mi = ind & (dist == d)
            mc = ctl & (dist == d)
            g, lo, hi = ci(nll[mc] - nll[mi])
            dg, dlo, dhi = paired(nll[mc] - nll[mi], ex_nll[mc] - ex_nll[mi])
            rows.append(dict(condition=cond, tier=TIER[cond], distance=int(d), gain=g, gain_ci_lo=lo,
                             gain_ci_hi=hi, delta_gain=dg, delta_gain_ci_lo=dlo, delta_gain_ci_hi=dhi,
                             rel_degradation=dg / gain_exact[int(d)] if gain_exact[int(d)] else np.nan,
                             target_nll=float(nll[mi].mean())))
    df = pd.DataFrame(rows)

    # (a) restricted range: distances where Exact keeps >= GAIN_KEEP_FRAC of its d=64 gain
    keep = [d for d in dists if gain_exact[int(d)] >= GAIN_KEEP_FRAC * gain_exact[int(dists[0])]]
    slopes = []
    for cond in cmap:
        if cond == "Exact":
            continue
        sub = df[df.condition == cond].set_index("distance")
        for label, ds in (("all_distances", dists), ("restricted", keep)):
            if len(ds) < 3:
                continue
            x = np.log2(np.asarray(ds, float))
            y_nll = np.array([sub.loc[int(d), "target_nll"] for d in ds]) - \
                np.array([df[(df.condition == "Exact") & (df.distance == int(d))].target_nll.iloc[0] for d in ds])
            y_gain = np.array([sub.loc[int(d), "delta_gain"] for d in ds])
            y_rel = np.array([sub.loc[int(d), "rel_degradation"] for d in ds])
            slopes.append(dict(condition=cond, tier=TIER[cond], range=label, n_distances=len(ds),
                               slope_target_nll=float(np.polyfit(x, y_nll, 1)[0]),
                               slope_delta_gain=float(np.polyfit(x, y_gain, 1)[0]),
                               slope_relative_degradation=float(np.polyfit(x, y_rel, 1)[0])))
    sl = pd.DataFrame(slopes)
    df.to_csv(OUT / "d2_distance_curves.csv", index=False)
    sl.to_csv(OUT / "d2_p4_slopes.csv", index=False)

    # probe B position curve
    mdB = pd.read_csv(DATA / "probe_B_meta.csv")
    exB = load(cmap["Exact"], "B")[0]
    brows = []
    for cfg in sorted(mdB.config.unique()):
        m = (mdB.config == cfg).values
        v, lo, hi = ci(exB[m])
        brows.append(dict(condition="Exact", config=cfg, nll=v, ci_lo=lo, ci_hi=hi,
                          headroom_vs_uniform=float(np.log(VOCAB) - v)))
    pd.DataFrame(brows).to_csv(OUT / "d2_probeB_position_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    gx = [int(d) for d in dists]
    gy = [gain_exact[d] for d in gx]
    gl = [df[(df.condition == "Exact") & (df.distance == d)].gain_ci_lo.iloc[0] for d in gx]
    gh = [df[(df.condition == "Exact") & (df.distance == d)].gain_ci_hi.iloc[0] for d in gx]
    ax.fill_between(gx, gl, gh, color=COLOR["reference"], alpha=0.2, lw=0)
    ax.plot(gx, gy, color=COLOR["reference"], marker="o", lw=2, label="Exact induction gain")
    ax.axhline(GAIN_KEEP_FRAC * gy[0], color="#eb6834", ls=(0, (4, 3)), lw=1.2,
               label=f"{int(GAIN_KEEP_FRAC*100)}% of d=64 gain")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("induction distance (tokens)")
    ax.set_ylabel("induction gain (nats)")
    ax.set_title("D2: does the Exact baseline itself decay with distance?", fontsize=10, color=INK)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    bb = pd.DataFrame(brows)
    ax.bar(range(len(bb)), bb.nll, color=COLOR["reference"])
    ax.axhline(np.log(VOCAB), color="#eb6834", ls=(0, (4, 3)), lw=1.2, label="uniform over vocab")
    ax.set_xticks(range(len(bb)), bb.config, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylim(9.5, 11.2)
    ax.set_ylabel("Exact KV target NLL (nats)")
    ax.set_title("Probe B: Exact by configuration", fontsize=10, color=INK)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "d2_exact_baseline_curves.png", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return dict(exact_gain_by_distance=gain_exact,
                gain_retained_at_896=float(gain_exact[max(gx)] / gain_exact[min(gx)]),
                kept_distances=keep, matched_family_ok=bool(au.identical_elsewhere.all() and au.len_equal.all()),
                extra_A_occurrences_max=int(au.extra_A_occurrences.max()),
                extra_B_occurrences_max=int(au.extra_B_occurrences.max()))


# ----------------------------------------------------------------- D3: probe B sharpness vs retrieval
def d3(cmap):
    mdB = pd.read_csv(DATA / "probe_B_meta.csv")
    exN, exH = load(cmap["Exact"], "B")
    uniform = float(np.log(VOCAB))
    headroom = uniform - float(exN.mean())
    rows = []
    for cond, run in cmap.items():
        nll, hit = load(run, "B")
        dn, dnlo, dnhi = paired(nll, exN) if cond != "Exact" else (0.0, 0.0, 0.0)
        dh, dhlo, dhhi = paired(hit, exH) if cond != "Exact" else (0.0, 0.0, 0.0)
        nll_worse = dnlo > 0
        top1_worse = dhhi < 0
        if cond == "Exact":
            verdict = "reference"
        elif nll_worse and top1_worse:
            verdict = "retrieval loss (NLL and top-1 both degrade)"
        elif nll_worse and not top1_worse:
            verdict = "sharpness effect (NLL degrades, top-1 does not)"
        elif not nll_worse and top1_worse:
            verdict = "top-1 degrades without NLL"
        else:
            verdict = "no detectable degradation"
        rows.append(dict(condition=cond, tier=TIER[cond], uniform_nll=uniform, exact_nll=float(exN.mean()),
                         exact_headroom_vs_uniform=headroom, nll=float(nll.mean()), delta_nll=dn,
                         delta_nll_ci_lo=dnlo, delta_nll_ci_hi=dnhi,
                         frac_of_exact_headroom=dn / headroom,
                         top1=float(hit.mean()), delta_top1=dh, delta_top1_ci_lo=dhlo, delta_top1_ci_hi=dhhi,
                         verdict=verdict))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d3_probeB_sharpness_vs_retrieval.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for tier in ("reference", "pathological", "intermediate", "good"):
        t = df[df.tier == tier]
        ax.scatter(t.delta_nll, t.delta_top1 * 100, s=38, color=COLOR[tier], label=tier, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r.condition.replace("MinMax-", "MM-").replace("QRM-", "Q-"),
                    (r.delta_nll, r.delta_top1 * 100), fontsize=6.5, xytext=(4, 3),
                    textcoords="offset points", color=INK)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_xlabel("Δ KV target NLL vs Exact (nats; positive = worse)")
    ax.set_ylabel("Δ top-1 accuracy vs Exact (percentage points)")
    ax.set_title("D3: probe B — does NLL degradation come with top-1 degradation?", fontsize=10, color=INK)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "d3_probeB_nll_vs_top1.png", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return dict(uniform_nll=uniform, exact_nll=float(exN.mean()), exact_headroom=headroom,
                exact_top1=float(exH.mean()),
                verdict_counts=df.verdict.value_counts().to_dict())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmap = conds()
    summary = dict(D1=d1(cmap), D2=d2(cmap), D3=d3(cmap), gain_keep_frac=GAIN_KEEP_FRAC)
    json.dump(summary, open(OUT / "diagnostics_summary.json", "w"), indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
