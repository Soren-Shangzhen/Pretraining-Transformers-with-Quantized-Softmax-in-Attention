#!/usr/bin/env python3
"""Appendix I figures, generated from the result tables of Appendix I, csv/
(rb-dossier_20261114) and, for the reference lines of the original runs, from the trajectory files of the formal suites.

Outputs: figI_*.{pdf,png}, written next to this script.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import paperstyle as ps  # noqa: E402

TRAIN = Path(os.environ.get("KI_REPO", Path(__file__).resolve().parents[2]))   # the root of this repository
AM = TRAIN / "analysis_main_v1"
DOSSIER = Path(os.environ.get("KI_DOSSIER", AM / "rb-rebuttal_v5/reports/rb-dossier_20261114")) / "csv"
A, B = AM / "A_longhorizon_124m_2p5b", AM / "B_factorial_124m_250m"
OUT = Path(__file__).resolve().parent
TRJ = pd.read_csv(DOSSIER / "trajectories.csv")
PRB = pd.read_csv(DOSSIER / "probes_layer_mean.csv")
ps.apply()

C_W, C_P, C_F, C_SM = "#e34948", "#1baf7a", "#4a3aa7", "#52514e"   # Weight-STE, Prob-STE, frozen step, softmax (paper palette)


def save(fig, name):
    ps.save(fig, OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)


def tr(run, paired=True):
    d = TRJ[TRJ.run_id == run].sort_values("tokens_m")
    return d[d.dnll.notna()] if paired else d


def band(ax, d, color, ls="-", label=None, lw=1.0, marker=None):
    ax.plot(d.tokens_m, d.dnll, color=color, ls=ls, lw=lw, marker=marker, ms=2.5, label=label)
    ax.fill_between(d.tokens_m, d.ci_lo, d.ci_hi, color=color, alpha=0.2, lw=0)


def fig_e11():
    fig, axs = ps.figure(ncols=2, height=1.7, gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    for runs, col, lab in ((("rb-E11-01", "rb-E11-02"), "#2a78d6", "FWM–LERP K=4"), (("rb-E11-03", "rb-E11-04"), "#7fb3ea", "FWM–LERP K=16"),
                           (("rb-E11-05", "rb-E11-07"), C_F, "FWM–Weight K=4")):
        for r, ls in zip(runs, ("-", "--")):
            band(ax, tr(r), col, ls=ls, label=lab if ls == "-" else None, marker="o")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("training tokens (M)"); ax.set_ylabel("detach $-$ full (nats)")
    ax.legend(loc="upper left", fontsize=ps.MIN_PT); ps.panel_label(ax, "a", "100M, seeds 7 (solid) and 42 (dashed)")
    ax = axs[1]
    band(ax, tr("rb-E11-09"), "#2a78d6", marker="o")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("training tokens (M)"); ps.panel_label(ax, "b", "250M, FWM–LERP K=4")
    save(fig, "figI_e11_detach_minus_full")


def fig_e2ab():
    b = pd.read_csv(B / "trajectory/contrast_trajectories.csv")
    fig, axs = ps.figure(ncols=2, height=1.8, sharey=True)
    runs = {50: ["rb-E2ab-01", "rb-E2ab-02", "rb-E2ab-03", "rb-E2ab-04", "rb-E2ab-05"],
            125: ["rb-E2ab-06", "rb-E2ab-07", "rb-E2ab-08", "rb-E2ab-09", "rb-E2ab-10"]}
    sty = [(C_W, "-", "W→W"), (C_P, "--", "W→P"), (C_F, "-", "W→W, frozen step"), (C_P, "-", "P→P"), (C_W, "--", "P→W")]
    for ax, bp, lt in ((axs[0], 50, "a"), (axs[1], 125, "b")):
        for cname, lab in (("MinMax-Weight K4 - Exact", "original MinMax–Weight"), ("MinMax-Prob K4 - Exact", "original MinMax–Prob")):
            d = b[b.contrast == cname].sort_values("tokens_m")
            ax.plot(d.tokens_m, d.delta, color=ps.LIGHTGRAY, lw=2.2, zorder=0, label="original MinMax\u2013Weight / \u2013Prob" if bp == 50 and "Weight" in cname else None)
        for r, (c, ls, lab) in zip(runs[bp], sty):
            d = tr(r)
            ax.plot(d.tokens_m, d.dnll, color=c, ls=ls, marker="o", ms=2.5, label=lab if bp == 50 else None)
        ax.axvline(bp, color=ps.GRAY, lw=0.6, ls=":")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("training tokens (M)")
        ps.panel_label(ax, lt, f"branches from {bp}M")
    axs[0].set_ylabel("NLL $-$ softmax (nats)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncols=6, fontsize=ps.MIN_PT)
    save(fig, "figI_e2ab_branches")


def fig_e2c():
    a = pd.read_csv(A / "trajectory/checkpoint_nll.csv")
    fig, ax = ps.figure(height=1.8)
    for cond, c, lab in (("MinMax-Weight K4", ps.LIGHTGRAY, "suite A MinMax–Weight (every 125M)"), ("Exact", C_SM, "suite A softmax")):
        d = a[(a.cond == cond) & (a.tokens_m <= 626)].sort_values("tokens_m")
        ax.plot(d.tokens_m, d.mean_nll, color=c, lw=2.2 if cond != "Exact" else 1.0, marker="o", ms=2.5, label=lab, zorder=0)
    for r, c, ls, lab in (("rb-E2c-01", C_W, "-", "W→W"), ("rb-E2c-02", C_P, "--", "W→P"), ("rb-E2c-03", C_F, "-", "W→W, frozen step")):
        d = tr(r, paired=False)
        ax.plot(d.tokens_m, d.nll, color=c, ls=ls, marker="o", ms=2.0, label=lab)
    ax.set_yscale("log"); ax.set_yticks([3.5, 4, 5, 7, 10, 20]); ax.set_yticklabels(["3.5", "4", "5", "7", "10", "20"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("training tokens (M)"); ax.set_ylabel("validation NLL (log scale)")
    ax.legend(loc="upper right", fontsize=ps.MIN_PT, ncols=2)
    save(fig, "figI_e2c_branches")


def fig_e3a25():
    fig, axs = ps.figure(ncols=2, height=1.7)
    ax = axs[0]
    band(ax, tr("rb-E3a-2.5B-01"), "#2a78d6", marker="o")
    ax.axhline(0.01, color=ps.GRAY, lw=0.6, ls="--"); ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("training tokens (M)"); ax.set_ylabel("$\\Delta$NLL vs softmax (nats)")
    ps.panel_label(ax, "a", "IndexSoftmax, UINT8, 2.5B")
    ax = axs[1]
    p = PRB[PRB.run_id == "rb-E3a-2.5B-01"]
    for L, c in (("1-64", "#c6dbef"), ("65-256", "#6baed6"), ("257-512", "#2171b5"), ("513-1024", "#08306b")):
        d = p[p.metric == f"len{L}_zeroed_softmax_mass"].sort_values("tokens_m")
        ax.plot(d.tokens_m, d.value, color=c, marker="o", ms=2.0, label=f"{L} keys")
    ax.set_xlabel("training tokens (M)"); ax.set_ylabel("softmax mass set to 0")
    ax.set_ylim(0, 0.32); ax.legend(loc="upper center", ncols=2, fontsize=ps.MIN_PT); ps.panel_label(ax, "b", "by row length")
    save(fig, "figI_e3a_2p5b")


def fig_e8():
    a = pd.read_csv(A / "trajectory/contrast_trajectories.csv")
    fig, ax = ps.figure(height=1.7)
    band(ax, tr("rb-E8c-01"), "#2a78d6", label="LERP K=16 $-$ softmax, seed 7", marker="o")
    d = a[a.contrast == "LERP K16 - Exact"].sort_values("tokens_m")
    ax.plot(d.tokens_m, d.delta, color="#2a78d6", ls="--", marker="s", ms=2.0, label="LERP K=16 $-$ softmax, seed 1337 (suite A)")
    ax.fill_between(d.tokens_m, d.ci_lo, d.ci_hi, color="#2a78d6", alpha=0.12, lw=0)
    band(ax, tr("rb-E8a-01"), C_SM, label="softmax seed 7 $-$ seed 1337", marker="^")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("training tokens (M)"); ax.set_ylabel("$\\Delta$NLL (nats)")
    ax.legend(loc="upper right", fontsize=ps.MIN_PT)
    save(fig, "figI_e8_seeds")


if __name__ == "__main__":
    fig_e11(); fig_e2ab(); fig_e2c(); fig_e3a25(); fig_e8()
