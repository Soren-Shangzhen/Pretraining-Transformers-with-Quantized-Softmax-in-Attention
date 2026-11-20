#!/usr/bin/env python3
"""Round-3 figure: NLL minus same-seed exact softmax over training (seed 7, 243 blocks, local RTX 3090)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

SUP = KI_REPO + "/analysis_main_v1/supplement_four_runs_20260923"
r = json.load(open(f"{SUP}/round3/reports/r3_lowprec_results.json"))
T = [20, 40, 60, 80, 100]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ser = [("bf16exp", "bf16-exp (645 levels)", "#2a78d6", "-"), ("fp8e4m3exp", "FP8 e4m3-exp (56 levels, zero tail)", "#eb6834", "-"),
       ("LERP K32", "LERP K32", "#1baf7a", "--")]
fig, ax = plt.subplots(figsize=(7.6, 4.3), facecolor=SURF)
for k, lab, col, ls in ser:
    y = [r["trajectory_vs_softmax"][k][str(t)] for t in T]
    ax.plot(T, y, ls, color=col, lw=2, marker="o", ms=5, markeredgecolor=SURF, markeredgewidth=1.5, label=lab)
fw = [r["abs_nll"]["FWM-Prob K4 (hist)"][str(t)] - r["abs_nll"]["softmax"][str(t)] for t in T]
ax.plot(T, fw, ":", color="#eda100", lw=2, marker="s", ms=5, markeredgecolor=SURF, markeredgewidth=1.2, label="FWM-Prob K4 (5 levels + zero tail), historical")
ax.plot([100], [r["vs_softmax_100m"]["FWM-Prob K4 strict (new)"][0]], "D", color="#eda100", ms=7, markeredgecolor=INK, markeredgewidth=0.6,
        label="FWM-Prob K4 strict hard forward (round 1), 100M")
ax.axhline(0, color=INK2, lw=0.8)
ax.axhspan(-0.0062, 0.0062, color=GRID, alpha=0.6, lw=0)
ax.text(21, 0.0066, "±1 five-seed SD of exact softmax (0.0062)", fontsize=7.5, color=INK2)
ax.set_facecolor(SURF)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9); ax.set_xticks(T)
ax.set_xlabel("training tokens (M)", color=INK2, fontsize=9); ax.set_ylabel("NLL − exact softmax (nats)", color=INK2, fontsize=9)
ax.set_title("Round 3 - naive reduced-precision softmax vs K-interval, 124M @ 100M, seed 7", loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=7.8, loc="lower right")
fig.text(0.01, 0.01, "LERP K4 is off scale (+0.094 at 100M). Each line is one run (single seed); no confidence intervals are drawn.", fontsize=7.2, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{SUP}/round3/reports/r3_lowprec_vs_softmax.png", dpi=160)
print("ok")
