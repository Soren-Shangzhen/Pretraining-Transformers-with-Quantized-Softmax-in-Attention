#!/usr/bin/env python3
"""Probe C length trend: per-token ΔNLL at L ∈ {32,64,128} with paired CIs, the slope against log2(L),
and the paired slope contrast between MinMax-Weight K16 and MinMax-Prob K4 (same K scale, same calibration,
only the surrogate position differs).

Common random numbers: one bootstrap index set per (length) is shared by every condition, so slope
differences between conditions are properly paired.
"""
import json
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
LENGTHS = [32, 64, 128]
PAIR = ("MinMax-Weight K16", "MinMax-Prob K4")      # as requested, but these differ in K as well as surrogate
MATCHED = [("MinMax-Weight K4", "MinMax-Prob K4"),  # same calibration and same K: isolates surrogate position
           ("QRM-Weight K4", "QRM-Prob K4")]
TIER = {**{c: "pathological" for c in ["MinMax-Weight K4", "MinMax-Weight K16", "LERP K1", "MinMax-Weight K32",
                                        "QRM-Weight K2"]},
        **{c: "intermediate" for c in ["MinMax-Prob K4", "QRM-Weight K4", "MinMax-Weight K64", "QRM-Prob K4"]},
        **{c: "good" for c in ["LERP K2", "LERP K4", "QRM-Weight K16", "LERP K32", "LERP K16"]},
        "Exact": "reference"}
COLOR = {"pathological": "#eb6834", "intermediate": "#eda100", "good": "#2a78d6", "reference": "#52514e"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def main():
    runs = pd.read_csv(ROOT.parents[0] / "manifests/primary_37_runs.tsv", sep="\t")
    runs = runs[runs.suite == "A_longhorizon_124m_2p5b"].copy()
    runs["cond"] = [l if l == "Exact" else f"{l} K{int(float(k))}" for l, k in zip(runs.label, runs.K)]
    cmap = {r.cond: r.run_name for r in runs.itertuples()}
    md = pd.read_csv(DATA / "probe_C_meta.csv")
    masks = {L: (md.seg_len == L).values for L in LENGTHS}
    ex = np.load(PS / f"{cmap['Exact']}__C__native.npz")["nll"]

    rng = np.random.default_rng(SEED)
    idx = {L: rng.integers(0, int(masks[L].sum()), size=(N_BOOT, int(masks[L].sum()))) for L in LENGTHS}
    x = np.log2(np.array(LENGTHS, float))
    xc = x - x.mean()

    rows, boot_slopes = [], {}
    for cond, run in cmap.items():
        if cond == "Exact":
            continue
        nll = np.load(PS / f"{run}__C__native.npz")["nll"]
        per_L, boots = {}, {}
        for L in LENGTHS:
            d = nll[masks[L]] - ex[masks[L]]
            b = d[idx[L]].mean(axis=1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            per_L[L] = (float(d.mean()), float(lo), float(hi))
            boots[L] = b
        Y = np.vstack([boots[L] for L in LENGTHS])                  # (3, N_BOOT)
        slope_boot = (xc[:, None] * (Y - Y.mean(axis=0))).sum(axis=0) / (xc ** 2).sum()
        y = np.array([per_L[L][0] for L in LENGTHS])
        slope = float(np.polyfit(x, y, 1)[0])
        slo, shi = np.percentile(slope_boot, [2.5, 97.5])
        boot_slopes[cond] = slope_boot
        rows.append(dict(condition=cond, tier=TIER[cond],
                         **{f"delta_L{L}": per_L[L][0] for L in LENGTHS},
                         **{f"ci_lo_L{L}": per_L[L][1] for L in LENGTHS},
                         **{f"ci_hi_L{L}": per_L[L][2] for L in LENGTHS},
                         slope_vs_log2L=slope, slope_ci_lo=float(slo), slope_ci_hi=float(shi),
                         slope_excludes_zero=bool(slo > 0 or shi < 0)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probeC_length_trend.csv", index=False)

    a, b = PAIR
    diff = boot_slopes[a] - boot_slopes[b]
    dlo, dhi = np.percentile(diff, [2.5, 97.5])
    ra = df[df.condition == a].iloc[0]
    rb = df[df.condition == b].iloc[0]
    contrast = dict(
        condition_a=a, slope_a=ra.slope_vs_log2L, ci_a=[ra.slope_ci_lo, ra.slope_ci_hi],
        condition_b=b, slope_b=rb.slope_vs_log2L, ci_b=[rb.slope_ci_lo, rb.slope_ci_hi],
        slope_difference=float(diff.mean()), diff_ci_lo=float(dlo), diff_ci_hi=float(dhi),
        cis_overlap=bool(not (ra.slope_ci_hi < rb.slope_ci_lo or rb.slope_ci_hi < ra.slope_ci_lo)),
        each_ci_excludes_zero=bool(ra.slope_excludes_zero and rb.slope_excludes_zero),
        difference_excludes_zero=bool(dlo > 0 or dhi < 0))
    contrast["verdict"] = ("reportable in the main text (descriptive): slopes differ, neither CI covers 0"
                           if contrast["difference_excludes_zero"] and contrast["each_ci_excludes_zero"]
                           and not contrast["cis_overlap"]
                           else "not separable: layer 2 stays an appendix")
    def contrast_of(a, b, note):
        diff = boot_slopes[a] - boot_slopes[b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        ra, rb = df[df.condition == a].iloc[0], df[df.condition == b].iloc[0]
        return dict(condition_a=a, slope_a=ra.slope_vs_log2L, ci_a=[ra.slope_ci_lo, ra.slope_ci_hi],
                    condition_b=b, slope_b=rb.slope_vs_log2L, ci_b=[rb.slope_ci_lo, rb.slope_ci_hi],
                    slope_difference=float(diff.mean()), diff_ci_lo=float(lo), diff_ci_hi=float(hi),
                    cis_overlap=bool(not (ra.slope_ci_hi < rb.slope_ci_lo or rb.slope_ci_hi < ra.slope_ci_lo)),
                    each_ci_excludes_zero=bool(ra.slope_excludes_zero and rb.slope_excludes_zero),
                    difference_excludes_zero=bool(lo > 0 or hi < 0), note=note)

    contrast["caveat"] = ("these two conditions differ in K (16 vs 4) as well as in surrogate position, so the "
                          "contrast is not a clean surrogate-only comparison")
    out = dict(requested_pair=contrast,
               matched_same_K=[contrast_of(a, b, "same calibration and K; only the surrogate position differs")
                               for a, b in MATCHED])
    json.dump(out, open(OUT / "probeC_slope_contrast.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _, r in df.iterrows():
        y = [r[f"delta_L{L}"] for L in LENGTHS]
        lo = [r[f"ci_lo_L{L}"] for L in LENGTHS]
        hi = [r[f"ci_hi_L{L}"] for L in LENGTHS]
        emph = r.condition in PAIR
        ax.fill_between(LENGTHS, lo, hi, color=COLOR[r.tier], alpha=0.25 if emph else 0.10, lw=0)
        ax.plot(LENGTHS, y, marker="o", ms=4.5 if emph else 3, lw=2.4 if emph else 1.3,
                color=COLOR[r.tier], alpha=1.0 if emph else 0.65,
                label=r.condition if emph else None)
        ax.annotate(r.condition.replace("MinMax-", "MM-").replace("QRM-", "Q-"), (LENGTHS[-1], y[-1]),
                    fontsize=6.5, xytext=(4, 0), textcoords="offset points", color="#0b0b0b")
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENGTHS, [str(L) for L in LENGTHS])
    ax.set_xlabel("copied segment length L (tokens)")
    ax.set_ylabel("per-token Δ NLL vs Exact (nats)")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.legend(fontsize=8)
    ax.set_title("Probe C: per-token degradation vs copied length (bands: 95% paired CI)", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(OUT / "probeC_length_trend.png", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(df[["condition", "tier", "delta_L32", "delta_L64", "delta_L128", "slope_vs_log2L", "slope_ci_lo",
              "slope_ci_hi", "slope_excludes_zero"]].round(3).to_string(index=False))
    print("\ncontrast:", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
