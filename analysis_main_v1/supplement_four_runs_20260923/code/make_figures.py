#!/usr/bin/env python3
"""Report figures from reports/supplement_trajectories.csv (243-block NLL; geometry on 8 blocks)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


SUP = KI_REPO + "/analysis_main_v1/supplement_four_runs_20260923"
rows = list(csv.DictReader(open(f"{SUP}/reports/supplement_trajectories.csv")))
T = [20, 40, 60, 80, 100]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S1, S2, S3, S4, NEU = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"


def val(k, e, t, c="mean_nll_243"):
    for x in rows:
        if x["series"] == k and x["evaluator"] == e and int(x["tokens_m"]) == t and x.get(c) not in (None, ""):
            return float(x[c])
    return float("nan")


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel("training tokens (M)", color=INK2, fontsize=9)
    ax.set_xticks(T)


def endlabel(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=8.5, color=INK)


# ---- E1: four calibration-gradient modes, seed 7, LERP K32
series = [("lerp_full", "full", S1, "-"), ("lerp_detach", "detach", S2, "-"),
          ("e1_min", "min_only", S3, "-"), ("e1_max", "max_only", S4, "-"), ("softmax", "softmax", NEU, "--")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
offs = {"full": 0, "max_only": 12, "softmax": -12, "detach": 7, "min_only": -7}
for k, lab, col, ls in series:
    y = [val(k, "L3090", t) for t in T]
    axes[0].plot(T, y, ls, color=col, lw=2, marker="o", ms=5, label=lab, markeredgecolor=SURF, markeredgewidth=1.5)
    endlabel(axes[0], T[-1], y[-1], lab, offs[lab])
style(axes[0], "a  Validation NLL (243 blocks)", "mean NLL (nats)")
for k, lab, col, ls in series:
    y = [val(k, "L3090", t, "geom_span_mean_layermean") for t in T]
    axes[1].plot(T, y, ls, color=col, lw=2, marker="o", ms=5, label=lab, markeredgecolor=SURF, markeredgewidth=1.5)
    endlabel(axes[1], T[-1], y[-1], lab, {"full": -7, "softmax": 5, "max_only": 13, "detach": -6, "min_only": 6}[lab])
axes[1].set_yscale("log")
style(axes[1], "b  Mean row score span (8 blocks, layer mean)", "span (log scale)")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
for ax in axes:
    ax.set_xlim(15, 118)
fig.suptitle("E1 - LERP K32, seed 7, 124M @ 100M WikiText-103: which calibration channel keeps training healthy",
             x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout()
fig.savefig(f"{SUP}/reports/figures/e1_four_modes.png", dpi=160)

# ---- E2: Prob - Weight trajectories
fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURF)
e2 = [("mm", "MinMax", S1), ("fw", "FWM", S2)]
for c, name, col in e2:
    new = [val(f"{c}_prob_new", "H5090", t) - val(f"{c}_weight_new", "H5090", t) for t in T]
    old = [val(f"{c}_prob_old", "L3090", t) - val(f"{c}_weight_old", "L3090", t) for t in T]
    ax.plot(T, new, "-", color=col, lw=2, marker="o", ms=5, markeredgecolor=SURF, markeredgewidth=1.5,
            label=f"{name}: new strict pair")
    ax.plot(T, old, "--", color=col, lw=1.6, marker="s", ms=4.5, markeredgecolor=SURF, markeredgewidth=1.2,
            label=f"{name}: historical pair")
    endlabel(ax, T[-1], new[-1], f"{name} strict {new[-1]:+.3f}", 7 if c == "mm" else -7)
    endlabel(ax, T[-1], old[-1], f"{name} hist. {old[-1]:+.3f}", -7 if c == "mm" else 7)
ax.axhline(0, color=INK2, lw=0.8)
style(ax, "E2 - Prob-STE minus Weight-STE, K4, seed 7", "delta mean NLL (nats)")
ax.set_xlim(15, 130)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
fig.text(0.01, 0.01, "new pairs: strict hard forward, gpu-host-A RTX 5090 evaluator; historical pairs: RTX 3090 evaluator",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{SUP}/reports/figures/e2_prob_minus_weight.png", dpi=160)
print("ok")

# ---- E1 two seeds: delta vs full and span (seed 7 from trajectories.csv, seed 42 from supplement_e1_seed42.json)
import json
s42 = json.load(open(f"{SUP}/reports/supplement_e1_seed42.json"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
cols = {"detach": S2, "min_only": S3, "max_only": S4}
k7 = {"detach": "lerp_detach", "min_only": "e1_min", "max_only": "e1_max"}
for m, col in cols.items():
    y7 = [val(k7[m], "L3090", t) - val("lerp_full", "L3090", t) for t in T]
    y42 = [s42["trajectory_vs_full"][m][str(t)] for t in T]
    axes[0].plot(T, y7, "-", color=col, lw=2, marker="o", ms=5, markeredgecolor=SURF, markeredgewidth=1.5, label=f"{m} seed 7")
    axes[0].plot(T, y42, "--", color=col, lw=1.8, marker="s", ms=4.5, markeredgecolor=SURF, markeredgewidth=1.2, label=f"{m} seed 42")
axes[0].axhline(0, color=INK2, lw=0.8)
style(axes[0], "a  NLL minus same-seed full (243 blocks)", "delta mean NLL (nats)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
for m, col in list(cols.items()) + [("full", S1), ("softmax", NEU)]:
    y7 = [val({"full": "lerp_full", "softmax": "softmax"}.get(m, k7.get(m)), "L3090", t, "geom_span_mean_layermean") for t in T]
    axes[1].plot(T, y7, "-" if m != "softmax" else ":", color=col, lw=2, marker="o", ms=4.5, markeredgecolor=SURF, markeredgewidth=1.2,
                 label=f"{m} seed 7")
    if m in ("full", "softmax", "detach", "min_only", "max_only"):
        v42 = s42["geometry_100m"][m]["span_mean"]
        axes[1].plot([100], [v42], "D", color=col, ms=6, markeredgecolor=INK, markeredgewidth=0.6)
axes[1].set_yscale("log")
style(axes[1], "b  Mean row span; lines seed 7, diamonds seed 42 at 100M", "span (log scale)")
axes[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("E1 - LERP K32, 124M @ 100M WikiText-103: single-channel calibration gradients on two seeds",
             x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout()
fig.savefig(f"{SUP}/reports/figures/e1_two_seeds.png", dpi=160)
print("ok two seeds")
