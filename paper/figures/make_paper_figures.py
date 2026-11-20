#!/usr/bin/env python3
"""Paper figures for the training-side K-interval attention paper.

Every number is read from analysis_main_v1 CSVs (or, for the dense detach logs, from the same raw
training logs that E_detach_history/ was built from). Nothing is typed in by hand.

Outputs: fig*.{pdf,png}, written next to this script.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import paperstyle as ps  # noqa: E402
import t_encoding as te  # noqa: E402

import os
TRAIN = Path(os.environ.get("KI_REPO", Path(__file__).resolve().parents[2]))   # the root of this repository
AM = TRAIN / "analysis_main_v1"
A, B, C = AM / "A_longhorizon_124m_2p5b", AM / "B_factorial_124m_250m", AM / "C_scale_1b_100m"
D = AM / "D_robustness_124m_100m_5seed/analysis_v1"
E = AM / "E_detach_history"
DS = AM / "downstream_v1/results"
LEG = TRAIN / "training_code/legacy_exact_fp32/runs"
PB = TRAIN / "training_code/phaseB_base/runs"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# --- palette: fixed hue per operator family (validated categorical palette, fixed order, never cycled)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0", "#ffffff"
FAM = {"Exact": INK2, "LERP": "#2a78d6", "MinMax-Weight": "#e34948", "MinMax-Prob": "#eb6834",
       "QRM-Weight": "#4a3aa7", "QRM-Prob": "#1baf7a"}
FAM_ORDER = ["LERP", "MinMax-Weight", "MinMax-Prob", "QRM-Weight", "QRM-Prob"]
# Display names (paper naming): calibration MinMax / FWM, reconstruction LERP / Nearest, surrogate Weight-STE /
# Prob-STE. Data keys (CSV condition names, run names) keep their historical labels QRM-*/MinMax-Weight/-Prob.
NAME = {"QRM-Weight": "FWM\u2013Weight", "QRM-Prob": "FWM\u2013Prob", "MinMax-Weight": "MinMax\u2013Weight",
        "MinMax-Prob": "MinMax\u2013Prob", "LERP": "LERP", "Exact": "softmax"}
COMPACT = {"QRM-Weight": "FWM\u2013W", "QRM-Prob": "FWM\u2013P", "MinMax-Weight": "MM\u2013W",
           "MinMax-Prob": "MM\u2013P", "LERP": "LERP", "Exact": "softmax"}
PRE, POST = "Weight-STE", "Prob-STE"
FWMLERP = dict(color="#2a78d6", ls=(0, (1, 1)), lw=1.4)   # FWM-LERP K4 trajectory in Fig. 4b (LERP hue, dotted)
C_DET, C_FULL, C_PROJ = "#eb6834", "#2a78d6", "#e87ba4"

ps.apply()  # figure style (paperstyle.py); colours/encoding from t_encoding.py


def fam_of(cond):
    return cond.split(" K")[0] if cond != "Exact" else "Exact"


def k_of(cond):
    return int(cond.split(" K")[1]) if " K" in cond else None


def pretty(cond):
    f = fam_of(cond)
    return NAME[f] + (f" K{k_of(cond)}" if k_of(cond) else "")


def save(fig, name):
    """Lint + save at exact physical size (PDF for LaTeX, PNG preview)."""
    ps.save(fig, OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)


def endpoints(suite):
    return pd.read_csv(suite / "endpoint_fullval/endpoint_table.csv")


def _fam_style(fam, K=None, k_shade=None, marker=True):
    """Encoding of a condition family: colour = calibration, line/fill = surrogate."""
    if fam == "Exact":
        st = dict(te.EXACT)
    elif fam == "LERP":
        st = dict(te.LERP)
    else:
        cal = "MinMax" if fam.startswith("MinMax") else "FW"
        sur = "Weight" if fam.endswith("Weight") else "Prob"
        st = te.cond_style(cal, sur, k_shade=k_shade)
    if not marker:
        st.pop("marker", None); st.pop("markerfacecolor", None)
    return st


def _scatter_style(fam):
    if fam == "LERP":
        return te.scatter_style("LERP")
    cal = "MinMax" if fam.startswith("MinMax") else "FW"
    return te.scatter_style(cal, "Weight" if fam.endswith("Weight") else "Prob")


SYMLOG_TICKS = [0, 0.01, 0.1, 1]


def _symlog_axis(ax, ticks=SYMLOG_TICKS, linthresh=0.02):
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_yticks(ticks)
    ax.set_yticklabels([("%g" % t) for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def _log_axis(ax, ticks):
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.set_yticklabels([("%g" % t) for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def _family_legend(fig, axes_with_handles, order, ncols):
    hl = {}
    for ax in axes_with_handles:
        h, l = ax.get_legend_handles_labels()
        hl.update(dict(zip(l, h)))
    keys = [o for o in order if o in hl]
    fig.legend([hl[k] for k in keys], keys, loc="outside upper center", ncols=ncols, handlelength=1.8)



def _graph_panel(ax, kind):
    """Computational graph of one surrogate placement (kind: 'W' or 'P'). Boxes = objects, arrows = forward;
    the dashed box marks where the stop-gradient difference branch enters."""
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off"); ax.set_xlim(0, 10.6); ax.set_ylim(0, 10)
    bw, bh = 1.9, 1.35
    def box(x, y, lab, fc="#f2f2f2", ec=ps.OI["black"], ls="-"):
        ax.add_patch(plt.Rectangle((x, y), bw, bh, fc=fc, ec=ec, lw=0.6, ls=ls, zorder=2, alpha=0.45))
        ax.text(x + bw / 2, y + bh / 2, lab, ha="center", va="center", fontsize=ps.SMALL_PT, zorder=3).set_in_layout(False)
    def arrow(x0, y0, x1, y1, color=ps.OI["black"], ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=color, lw=0.7, ls=ls, shrinkA=0, shrinkB=0))
    yT, yB = 7.0, 3.4  # two rows: relaxed (LERP) path on top, hard path below; sg node between them
    sw = 1.9  # width of the stop-gradient node; its full label is printed below the node
    def sgnode(x, y, lab):
        ax.add_patch(plt.Rectangle((x, y), sw, 0.8, fc="white", ec=ps.GRAY, lw=0.6, ls="--", zorder=2, alpha=0.45))
        ax.text(x + sw / 2, y + 0.4, "sg", ha="center", va="center", fontsize=ps.MIN_PT, color=ps.GRAY, zorder=3).set_in_layout(False)
        ax.text(x + sw / 2, y - 0.25, lab, ha="center", va="top", fontsize=ps.MIN_PT, color=ps.GRAY, zorder=3).set_in_layout(False)
    box(0.2, yT, "$s$")
    if kind == "W":
        box(2.7, yT, "$w^L$"); box(2.7, yB, "$w^N$", fc="white", ls="--")
        box(5.3, yT, "$w$"); box(7.9, yT, "$P$")
        arrow(0.2 + bw, yT + bh / 2, 2.7, yT + bh / 2); arrow(0.2 + bw, yT + 0.3, 2.7, yB + bh / 2)
        arrow(2.7 + bw, yT + bh / 2, 5.3, yT + bh / 2)
        # difference branch: both w^L and w^N feed sg(w^N - w^L); its output is added to w^L
        sgnode(5.3, yB + 0.25, "sg($w^N{-}w^L$)")
        arrow(2.7 + bw, yB + bh / 2, 5.3, yB + 0.65, color=ps.GRAY)
        arrow(2.7 + bw / 2, yT, 5.3 + 0.35, yB + 1.05, color=ps.GRAY)
        arrow(5.3 + bw / 2 + 0.3, yB + 1.05, 5.3 + bw / 2 + 0.3, yT, color=ps.GRAY, ls="--")
        arrow(5.3 + bw, yT + bh / 2, 7.9, yT + bh / 2)
        ax.text(5.0, 0.6, "Jacobian: hard weights", ha="center", va="center", fontsize=ps.SMALL_PT).set_in_layout(False)
    else:
        box(2.7, yT, "$w^L$"); box(2.7, yB, "$w^N$", fc="white", ls="--")
        box(5.3, yT, "$P^L$"); box(5.3, yB, "$P^N$", fc="white", ls="--")
        box(7.9, yT, "$P$")
        arrow(0.2 + bw, yT + bh / 2, 2.7, yT + bh / 2); arrow(0.2 + bw, yT + 0.3, 2.7, yB + bh / 2)
        arrow(2.7 + bw, yT + bh / 2, 5.3, yT + bh / 2); arrow(2.7 + bw, yB + bh / 2, 5.3, yB + bh / 2)
        arrow(5.3 + bw, yT + bh / 2, 7.9, yT + bh / 2)
        # difference branch: both P^L and P^N feed sg(P^N - P^L); its output is added to P^L
        sgnode(7.9, yB + 0.25, "sg($P^N{-}P^L$)")
        arrow(5.3 + bw, yB + bh / 2, 7.9, yB + 0.65, color=ps.GRAY)
        arrow(5.3 + bw / 2, yT, 7.9 + 0.35, yB + 1.05, color=ps.GRAY)
        arrow(7.9 + bw / 2 + 0.3, yB + 1.05, 7.9 + bw / 2 + 0.3, yT, color=ps.GRAY, ls="--")
        ax.text(5.0, 0.6, "Jacobian: relaxed weights", ha="center", va="center", fontsize=ps.SMALL_PT).set_in_layout(False)



def _recon_curves(ax, K, zlo, b, h, ex_scores, near_color, tail=None):
    """Exponential, LERP and Nearest reconstruction curves on a K-interval grid b (K+1 values)."""
    z = np.linspace(-8, 0, 800)
    ax.vlines(b, 0, 1.05, color=ps.LIGHTGRAY, lw=0.5, zorder=0)
    if tail is not None:
        ax.axvspan(-8, tail, color=ps.LIGHTGRAY, alpha=0.45, lw=0, zorder=0)
    ax.plot(z, np.exp(z), color=ps.OI["black"], lw=1.0, label=r"$e^{z}$")
    zc = np.clip(z, zlo, 0); r = np.clip(np.floor((zc - zlo) / h), 0, K - 1).astype(int); t = (zc - b[r]) / h
    lerp = (1 - t) * np.exp(b[r]) + t * np.exp(b[r + 1]); near = np.exp(b[r + (t >= 0.5)])
    if tail is not None:
        lerp[z < tail] = 0; near[z < tail] = 0
    ax.plot(z, lerp, color=ps.OI["green"], lw=1.2, label="LERP")
    ax.step(z, near, where="post", color=near_color, lw=1.1, label="Nearest")
    ax.scatter(b, np.exp(b), s=10, color=ps.OI["black"], zorder=4, label="grid ($K{+}1$ values)")
    ax.scatter(ex_scores, np.zeros_like(ex_scores) - 0.03, marker="|", s=18, color=ps.GRAY, zorder=4, clip_on=False)
    ax.set_xticks([-8, -4, 0]); ax.set_ylim(-0.04, 1.05); ax.tick_params(labelbottom=False)


def _bin_strip(ax, K, zlo, b, h, ex_scores, tail=None):
    """Nearest rounding bins [b_r - h/2, b_r + h/2) truncated to the valid range; midpoint ties round up.
    Shading marks score regions only (no counts). Example scores (ticks) point to the grid value they take."""
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticks([-8, -4, 0])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    for r in range(K + 1):
        lo, hi = max(zlo, b[r] - h / 2), min(0.0, b[r] + h / 2)
        ax.axvspan(lo, hi, ymin=0.3, ymax=1.0, color=(ps.LIGHTGRAY if r % 2 == 0 else "white"), alpha=0.9, lw=0, zorder=0)
    mids = b[:-1] + h / 2
    ax.vlines(mids, 0.3, 1.0, color=ps.GRAY, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    if tail is not None:
        ax.axvspan(-8, tail, ymin=0.3, ymax=1.0, color=ps.LIGHTGRAY, alpha=0.45, lw=0, zorder=0)
        ax.vlines([tail], 0.3, 1.0, color=ps.GRAY, lw=0.6, zorder=1)
    ax.scatter(b, np.full_like(b, 0.65), s=10, color=ps.OI["black"], zorder=4)
    for zs in ex_scores:
        if tail is not None and zs < tail:
            continue
        rr = int(np.clip(np.floor((zs - zlo) / h), 0, K - 1)); tt = (zs - b[rr]) / h; l_ = rr + (tt >= 0.5)
        ax.annotate("", xy=(b[l_], 0.6), xytext=(zs, 0.12), arrowprops=dict(arrowstyle="->", color=ps.GRAY, lw=0.5, shrinkA=0, shrinkB=1.5))
        ax.plot([zs], [0.1], marker="|", color=ps.GRAY, ms=4, mew=0.7, zorder=3)
    ax.set_xlabel("$z=s-M$")


def fig1_schematic():
    fig = plt.figure(figsize=(ps.TEXTWIDTH_IN, 1.75), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 1.15], width_ratios=[1, 1, 1.2, 1.2])
    axa, axa2 = fig.add_subplot(gs[0, 0]), None
    axa2 = fig.add_subplot(gs[1, 0], sharex=axa)
    axb = fig.add_subplot(gs[0, 1], sharey=axa); axb2 = fig.add_subplot(gs[1, 1], sharex=axb)
    axc, axd = fig.add_subplot(gs[:, 2]), fig.add_subplot(gs[:, 3])
    K = 4
    ex_scores = np.array([-7.3, -5.9, -3.1, -1.6, -0.7, -0.2, 0.0])  # example scores (z = s - m), drawn for illustration
    # (a) MinMax: grid over [mu, m] = [-8, 0], h = 2
    zmin = -8.0; h = -zmin / K; b = zmin + np.arange(K + 1) * h
    _recon_curves(axa, K, zmin, b, h, ex_scores, ps.OI["vermillion"])
    axa.annotate("", xy=(b[1], 0.30), xytext=(b[2], 0.30), arrowprops=dict(arrowstyle="<->", color=ps.OI["black"], lw=0.6))
    axa.text((b[1] + b[2]) / 2 + 0.3, 0.34, "$h=(M-m)/K$", ha="center", va="bottom", fontsize=ps.SMALL_PT)
    axa.text(-7.9, 0.98, "lower $m$ $\\rightarrow$\nwider bins", ha="left", va="top", fontsize=ps.SMALL_PT, color=ps.GRAY)
    axa.annotate("", xy=(-8.0, 0.56), xytext=(-7.0, 0.56), arrowprops=dict(arrowstyle="->", color=ps.GRAY, lw=0.6))
    axa.set_ylabel("weight $w$")
    _bin_strip(axa2, K, zmin, b, h, ex_scores)
    axa2.set_ylabel("bins", fontsize=ps.MIN_PT, color=ps.GRAY, rotation=0, ha="right", va="center", labelpad=2)
    RED = "#d62728"
    axa.plot([-h / 2, 0.0], [1.0, 1.0], color=RED, lw=0.6, solid_capstyle="butt", zorder=6)  # top bin [-h/2, 0]: w^N = 1
    axa.text(-3.5, 0.70, "top bin $w^N{=}1$", ha="left", va="top", fontsize=ps.MIN_PT, color=RED)
    ps.panel_label(axa, "a", "MinMax")
    # (b) FWM: window [-tau, 0], h = tau/K, zero tail
    tau = 6.0; hb = tau / K; bb = -tau + np.arange(K + 1) * hb
    _recon_curves(axb, K, -tau, bb, hb, ex_scores, ps.OI["blue"], tail=-tau)
    axb.annotate("", xy=(bb[1], 0.30), xytext=(bb[2], 0.30), arrowprops=dict(arrowstyle="<->", color=ps.OI["black"], lw=0.6))
    axb.text((bb[1] + bb[2]) / 2, 0.34, "$h=\\tau/K$", ha="center", va="bottom", fontsize=ps.SMALL_PT)
    axb.text(-7.9, 0.98, "width indep.\nof lower tail", ha="left", va="top", fontsize=ps.SMALL_PT, color=ps.GRAY)
    axb.text(-7.9, 0.30, "tail\n$w=0$", fontsize=ps.SMALL_PT, color=ps.GRAY)
    axb.tick_params(labelleft=False)
    _bin_strip(axb2, K, -tau, bb, hb, ex_scores, tail=-tau)
    axb.plot([-hb / 2, 0.0], [1.0, 1.0], color=RED, lw=0.6, solid_capstyle="butt", zorder=6)
    axb.text(-3.1, 0.66, "top bin $w^N{=}1$", ha="left", va="top", fontsize=ps.MIN_PT, color=RED)
    ps.panel_label(axb, "b", "FWM, $\\tau=6$")
    # (c, d) surrogate placements
    _graph_panel(axc, "W"); ps.panel_label(axc, "c", PRE)
    _graph_panel(axd, "P"); ps.panel_label(axd, "d", POST)
    h_, l_ = axa.get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside upper center", ncols=4)
    save(fig, "fig1_operator_schematic")


# ----------------------------------------------------------------------------------------------------
# Fig 2: same forward, different calibration backward -- LERP K32 as the central example
# ----------------------------------------------------------------------------------------------------
def _detach_series():
    dense = pd.read_csv(E / "dense_logged_gap_detach_vs_full.csv")
    end = pd.read_csv(E / "endpoint_detach_vs_full_s1337.csv")
    conds = [("Nearest K4", "q4_nearest_100m", "q4_nearest_fullgrad_100m"),
             ("Nearest K16", "q16_nearest_100m", "q16_nearest_fullgrad_100m"),
             ("LERP K32", "q32_lerp_100m", "q32_lerp_fullgrad_100m")]
    return dense, end, conds


def _plot_detach_pair(ax, ax2, dense, name, det, full, with_proj):
    GM = te.GRAD_MODE
    d = dense[dense.condition == name]
    for var, key in (("full", "full"), ("detach", "detach")):
        dd = d[d.variant == var]
        ax.plot(dd.tokens_m, dd.logged_gap, label=var, **GM[key])
    dsp = pd.read_csv(LEG / det / ("attention_stats.csv" if "lerp" in det else "occupancy.csv"))
    fsp = pd.read_csv(PB / full / "attention_stats.csv")
    ax2.plot(fsp.tokens / 1e6, fsp.row_span_mean, **GM["full"])
    ax2.plot(dsp.tokens / 1e6, dsp.row_span_mean, **GM["detach"])
    if with_proj:
        pl = pd.read_csv(PB / "q32_lerp_detachproj_100m/log.csv")
        ex_log = pd.read_csv(LEG / "exact_fp32_softmax_100m/log.csv")
        mp = pl.merge(ex_log[["tokens", "val_loss"]], on="tokens", suffixes=("", "_exact"))
        ax.plot(mp.tokens / 1e6, mp.val_loss - mp.val_loss_exact, label="detach + zero-sum projection", **GM["detach+proj"])
        psp = pd.read_csv(PB / "q32_lerp_detachproj_100m/attention_stats.csv")
        ax2.plot(psp.tokens / 1e6, psp.row_span_mean, **GM["detach+proj"])
    ax.axhline(0, color=ps.GRAY, lw=0.5)
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
    ax2.set_xlim(0, 100); _log_axis(ax2, [1, 10, 100, 1000]); ax2.set_ylim(1, 3000); ax2.set_xticks([0, 50, 100])


def _forest(ax, e, col, lo, hi, color, xlabel, marker="o", ci=True):
    e = e.dropna(subset=[col]).copy()
    e["label"] = e.condition.str.replace("Nearest", "MM\u2013W").str.replace("LERP K32 detach+projection (P1)", "LERP K32 +proj")
    y = np.arange(len(e))[::-1]
    if ci:
        ax.errorbar(e[col], y, xerr=[e[col] - e[lo], e[hi] - e[col]], fmt=marker, color=color, ms=3.5, mew=1.0, ecolor="black", elinewidth=0.6, capsize=1.5)
    else:
        ax.plot(e[col], y, marker=marker, ms=3.0, ls="none", color=color, markerfacecolor=color, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(e.label, fontsize=ps.MIN_PT); ax.yaxis.tick_right()
    ax.set_xlabel(xlabel)
    return e


def _e1_endpoints():
    """Single-channel endpoints per seed against that seed's own full-gradient run (Table tab:e1 sources):
    seed 7 from supplement_endpoint_contrasts.csv, seed 42 from supplement_e1_seed42.json; 95% block-bootstrap CIs
    over validation blocks (evaluation sampling only)."""
    import json
    c = pd.read_csv(AM / "supplement_four_runs_20260923/reports/supplement_endpoint_contrasts.csv").set_index("contrast")
    j = json.load(open(AM / "supplement_four_runs_20260923/reports/supplement_e1_seed42.json"))["contrasts_100m"]
    rows = []
    for mode, k7, k42 in (("detach", "HIST E1 detach - full", "detach - full"), ("min_only", "E1 min_only - full", "min_only - full"), ("max_only", "E1 max_only - full", "max_only - full")):
        r = c.loc[k7]; rows.append(dict(seed=7, mode=mode, d=r.delta, lo=r.ci_lo, hi=r.ci_hi))
        d, lo, hi = j[k42]; rows.append(dict(seed=42, mode=mode, d=d, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def fig2_detach():
    dense, end, conds = _detach_series()
    GM = te.GRAD_MODE
    fig, axs = ps.figure(nrows=2, ncols=2, height=2.36, gridspec_kw=dict(width_ratios=[1, 1.08], height_ratios=[0.82, 1.18]))
    ax, ax2 = axs[0][0], axs[0][1]
    _plot_detach_pair(ax, ax2, dense, *conds[2], with_proj=True)
    ax.set_ylim(-0.1, 1.6); ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.axvspan(25, 30, color=ps.LIGHTGRAY, alpha=0.45, lw=0, zorder=0)
    ax.text(27.5, 1.52, "separation\n(descriptive)", ha="center", va="top", fontsize=ps.MIN_PT, color=ps.GRAY)
    ax.set_ylabel("val loss $-$ softmax (nats)"); ax.set_xlabel("training tokens (M)")
    ps.panel_label(ax, "a", "LERP K32, 100M tokens")
    ax2.set_ylabel("mean row span"); ax2.set_xlabel("training tokens (M)")
    ps.panel_label(ax2, "b", "row span (log)")
    ax = axs[1][0]
    rep = pd.read_csv(AM / "detach_replication/detach_replication_paired.csv")
    rep = rep[rep.ckpt == "100m"].sort_values("seed")
    # one shared row for the two replication seeds (triangle markers as in panel d)
    end2 = pd.concat([end, pd.DataFrame({"condition": ["LERP K32, seeds 7/42"], "detach_minus_full": [float(rep.dmf.iloc[0])],
                                          "dmf_lo": [float(rep.lo.iloc[0])], "dmf_hi": [float(rep.hi.iloc[0])]})], ignore_index=True)
    _forest(ax, end2, "detach_minus_full", "dmf_lo", "dmf_hi", GM["detach"]["color"], "detach $-$ full (nats)", marker="o", ci=False)
    ax.plot([float(rep.dmf.iloc[0])], [0.12], marker="^", ms=3.5, ls="none", color=GM["detach"]["color"])
    ax.plot([float(rep.dmf.iloc[1])], [-0.12], marker="v", ms=3.5, ls="none", color=GM["detach"]["color"])
    ax.axvline(0, color=ps.GRAY, lw=0.5)
    ax.set_xlim(0, 3.6); ax.set_xticks([0, 1, 2, 3]); ax.set_ylim(-0.4, len(end2) - 0.6)
    ps.panel_label(ax, "c", "100M endpoints, every matched pair")
    # (d) single-channel ablation: each seed against its own full-gradient run, evaluation-sampling CIs
    ax = axs[1][1]
    e1 = _e1_endpoints()
    modes = ["detach", "min_only", "max_only"]; y0 = {m: i for i, m in enumerate(modes[::-1])}
    col = {"detach": GM["detach"]["color"], "min_only": ps.OI["orange"], "max_only": ps.OI["sky"]}
    for seed, mk, off in ((7, "^", 0.16), (42, "v", -0.16)):
        for m in modes:
            r = e1[(e1.seed == seed) & (e1["mode"] == m)].iloc[0]
            ax.errorbar(r.d, y0[m] + off, xerr=[[r.d - r.lo], [r.hi - r.d]], fmt=mk, ms=3.5, color=col[m], ecolor="black", elinewidth=0.6, capsize=1.5)
    ax.axvline(0, color=ps.GRAY, lw=0.5)
    ax.set_yticks([y0[m] for m in modes]); ax.set_yticklabels(modes, fontsize=ps.MIN_PT); ax.yaxis.tick_right()
    ax.set_ylim(-0.5, 2.5); ax.set_xlim(-0.05, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("mode $-$ own full-gradient run (nats)")
    # seed markers in neutral grey: the marker shape, not the colour, identifies the seed
    seed_h = [Line2D([], [], marker=mk, ls="none", ms=3.5, color=ps.GRAY, label=f"seed {s}") for s, mk in ((7, "^"), (42, "v"))]
    ax.legend(handles=seed_h, loc="center left", fontsize=ps.MIN_PT, handlelength=0.8, borderpad=0.3, labelspacing=0.2, bbox_to_anchor=(0.08, 0.5))
    ps.panel_label(ax, "d", "single-channel endpoints, seeds 7/42")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncols=3)
    save(fig, "fig2_detach_vs_full")


def figS_detach_rep():
    """Appendix: detach - Exact on seeds 7 and 42 (five checkpoints each) with the matched full-gradient runs."""
    rep = pd.read_csv(AM / "detach_replication/detach_replication_paired.csv")
    GM = te.GRAD_MODE
    fig, ax = ps.figure(width=3.4, height=1.6)
    xs = [20, 40, 60, 80, 100]
    for seed, mk in ((7, "^"), (42, "v")):
        dd = rep[(rep.seed == seed) & (rep.ckpt != "100m_vs_exact")]
        ex = {c.parent.name.split("_")[0]: float(np.load(c).mean()) for c in sorted((D.parent / "eval_native_v1" / f"Exact_s{seed}").glob("*/per_block_nll.npy"))}
        ax.plot(xs, [float(x.detach) - ex[x.ckpt] for x in dd.itertuples()], marker=mk, ms=3.5, color=GM["detach"]["color"], ls=GM["detach"]["linestyle"], label=f"detach, seed {seed}")
        ax.plot(xs, [float(x.full) - ex[x.ckpt] for x in dd.itertuples()], marker=mk, ms=3.5, color=GM["full"]["color"], ls="-", label=f"full, seed {seed}")
    ax.axhline(0, color=ps.GRAY, lw=0.5); ax.set_xlim(0, 105); ax.set_xticks([0, 50, 100]); ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("training tokens (M)"); ax.set_ylabel("val loss $-$ softmax (nats)")
    ax.legend(loc="upper left", ncols=2, handlelength=1.6)
    save(fig, "figS_detach_rep")


def _e1_seed_series(seed):
    """Per-checkpoint mean NLL (243 blocks) and layer-mean row span for full/detach/min_only/max_only at one seed,
    read from the evaluation directories (same files analyze_e1_seed.py uses): historical controls in
    D_robustness/eval_native_v1, single-channel runs in supplement eval/rtx3090."""
    import glob
    HIST = AM / "D_robustness_124m_100m_5seed/eval_native_v1"; SUPE = AM / "supplement_four_runs_20260923/eval/rtx3090"
    dirs = {"lerp_full": HIST / f"LERP_K32_s{seed}", "lerp_detach": HIST / f"LERP-Detach_K32_s{seed}",
            "e1_min": SUPE / f"sp_s{seed}_lerp_k32_minonly_100m", "e1_max": SUPE / f"sp_s{seed}_lerp_k32_maxonly_100m"}
    out = {}
    for k, d in dirs.items():
        rows = []
        for t in (20, 40, 60, 80, 100):
            m = sorted(glob.glob(str(d / f"{t:03d}m_*")))
            assert len(m) == 1, (k, t, m)
            nll = float(np.load(f"{m[0]}/per_block_nll.npy").mean())
            g = pd.read_csv(f"{m[0]}/geometry_layers.csv")
            rows.append((t, nll, float(g.span_mean.mean())))
        out[k] = pd.DataFrame(rows, columns=["tokens_m", "mean_nll_243", "span"])
    return out


def figS_e1_channels():
    """Appendix: single-channel ablation (LERP K32), seeds 7 and 42: loss minus the same-seed full run and layer-mean
    row span, four modes. Rows = seeds, columns = (loss - full, span). All series from per-checkpoint eval files."""
    GM = te.GRAD_MODE
    sty = {"lerp_full": dict(GM["full"], label="full"), "lerp_detach": dict(GM["detach"], label="detach"),
           "e1_min": dict(color=ps.OI["orange"], linestyle=(0, (3, 1.5, 1, 1.5)), label="min_only"),
           "e1_max": dict(color=ps.OI["sky"], linestyle="-", label="max_only")}
    fig, axes = ps.figure(nrows=2, ncols=2, height=1.75)
    for i, seed in enumerate((7, 42)):
        ser = _e1_seed_series(seed)
        if seed == 7:  # cross-check against the published trajectory CSV (same evaluator)
            t = pd.read_csv(AM / "supplement_four_runs_20260923/reports/supplement_trajectories.csv"); t = t[t.evaluator == "L3090"]
            for k in ser:
                ref = t[t.series == k].sort_values("tokens_m")
                assert np.allclose(ref.mean_nll_243.values, ser[k].mean_nll_243.values, atol=1e-9), k
        full = ser["lerp_full"].set_index("tokens_m").mean_nll_243
        for k, st in sty.items():
            d = ser[k]
            axes[i][0].plot(d.tokens_m, d.mean_nll_243.values - full.loc[d.tokens_m].values, marker="o", ms=3, **st)
            axes[i][1].plot(d.tokens_m, d.span, marker="o", ms=3, **st)
        axes[i][0].axhline(0, color=ps.GRAY, lw=0.5)
        axes[i][0].set_xlim(0, 105); axes[i][0].set_xticks([0, 50, 100]); axes[i][0].set_ylim(-0.1, 1.0)
        _log_axis(axes[i][1], [1, 10, 100, 1000]); axes[i][1].set_ylim(1, 3000)
        axes[i][1].set_xlim(0, 105); axes[i][1].set_xticks([0, 50, 100])
        ps.panel_label(axes[i][0], "ac"[i], f"seed {seed}: loss minus full"); ps.panel_label(axes[i][1], "bd"[i], f"seed {seed}: row span (log)")
    for ax in axes[1]: ax.set_xlabel("training tokens (M)")
    for ax in axes[0]: ax.tick_params(labelbottom=False)
    axes[1][0].set_ylabel("loss $-$ full (nats)"); axes[1][1].set_ylabel("row span")
    h, l = axes[0][0].get_legend_handles_labels(); fig.legend(h, l, loc="outside upper center", ncols=4)
    save(fig, "figS_e1_channels")


def figS_lowprec():
    """Appendix: reduced-precision exponential baselines (round 3), 124M @ 100M, seed 7. Two panels from
    round3/reports/r3_lowprec_results.json: (a) full range incl. LERP K4; (b) near-reference zoom incl. historical FWM-Prob K4.
    Every series: one run, five evaluated checkpoints, same RTX 3090 evaluator; no CIs; no cross-stack endpoint markers."""
    import json
    r = json.load(open(AM / "supplement_four_runs_20260923/round3/reports/r3_lowprec_results.json"))
    T = [20, 40, 60, 80, 100]
    def traj(key):
        if key in r["trajectory_vs_softmax"]:
            return [r["trajectory_vs_softmax"][key][str(t)] for t in T]
        return [r["abs_nll"][key][str(t)] - r["abs_nll"]["softmax"][str(t)] for t in T]
    sty = {"bf16exp": dict(color=ps.OI["blue"], linestyle="-", marker="o", label="BF16-exp"),
           "fp8e4m3exp": dict(color=ps.OI["vermillion"], linestyle="-", marker="s", label="FP8-rounded exp"),
           "LERP K32": dict(color=ps.OI["green"], linestyle="--", marker="^", label="LERP K32"),
           "LERP K4": dict(color=ps.OI["black"], linestyle=(0, (3, 1.5, 1, 1.5)), marker="v", label="LERP K4"),
           "FWM-Prob K4 (hist)": dict(color=ps.OI["orange"], linestyle=":", marker="D", label="FWM–Prob K4 (historical)")}
    fig, axes = ps.figure(ncols=2, height=1.3)
    for key in ("bf16exp", "fp8e4m3exp", "LERP K32", "LERP K4"):
        axes[0].plot(T, traj(key), ms=3, **sty[key])
    for key in ("bf16exp", "fp8e4m3exp", "LERP K32", "FWM-Prob K4 (hist)"):
        axes[1].plot(T, traj(key), ms=3, **sty[key])
    for ax in axes:
        ax.axhline(0, color=ps.GRAY, lw=0.5); ax.set_xlim(15, 105); ax.set_xticks([20, 40, 60, 80, 100]); ax.set_xlabel("training tokens (M)")
    axes[0].set_ylabel("$\\Delta$NLL vs softmax (nats)"); axes[0].set_ylim(-0.02, 0.10)
    axes[1].set_ylim(-0.015, 0.015); axes[1].set_yticks([-0.01, 0, 0.01])
    ps.panel_label(axes[0], "a", "full range (LERP K4 included)"); ps.panel_label(axes[1], "b", "near-reference zoom (LERP K4 excluded)")
    h, l = axes[0].get_legend_handles_labels(); h2, l2 = axes[1].get_legend_handles_labels()
    for hh, ll in zip(h2, l2):
        if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, loc="outside upper center", ncols=5)
    save(fig, "figS_lowprec")


def figS_detach_extra():
    """Appendix: the MinMax-Weight K4/K16 detach pairs, detach vs full: loss gap and row span (one row, two panels).
    The full-gradient endpoints - softmax are in the table tab:detach."""
    dense, end, conds = _detach_series()
    GM = te.GRAD_MODE
    fig, axes = ps.figure(width=0.72 * ps.TEXTWIDTH_IN, ncols=2, height=1.55)
    for j in range(2):
        name, det, full = conds[j]
        K = {0: "K4", 1: "K16"}[j]; kst = {} if j == 0 else dict(alpha=0.45)
        d = dense[dense.condition == name]
        for var in ("full", "detach"):
            dd = d[d.variant == var]
            axes[0].plot(dd.tokens_m, dd.logged_gap, label=f"{var}, {K}", **GM[var], **kst)
        dsp = pd.read_csv(LEG / det / "occupancy.csv"); fsp = pd.read_csv(PB / full / "attention_stats.csv")
        axes[1].plot(fsp.tokens / 1e6, fsp.row_span_mean, **GM["full"], **kst)
        axes[1].plot(dsp.tokens / 1e6, dsp.row_span_mean, **GM["detach"], **kst)
    axes[0].axhline(0, color=ps.GRAY, lw=0.5); axes[0].set_ylim(-0.1, 1.6); axes[0].set_yticks([0, 0.5, 1.0, 1.5])
    _log_axis(axes[1], [1, 10, 100, 1000]); axes[1].set_ylim(1, 3000)
    for ax in axes:
        ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xlabel("training tokens (M)")
    axes[0].set_ylabel("val loss $-$ softmax (nats)"); axes[1].set_ylabel("row span")
    ps.panel_label(axes[0], "a", "MinMax\u2013Weight K4 / K16: loss gap"); ps.panel_label(axes[1], "b", "row span (log)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncols=4)
    save(fig, "figS_detach_extra")


# ----------------------------------------------------------------------------------------------------
# Fig 3: calibration x surrogate interaction plots + five-seed replication over K
# ----------------------------------------------------------------------------------------------------
def _interaction_panel(ax, df, K, kind):
    names = {"MinMax": "MinMax", "FW": "QRM"}
    for cal in ("MinMax", "FW"):
        vals, lo, hi = [], [], []
        for sur in ("Weight", "Prob"):
            row = df[df.cond == f"{names[cal]}-{sur} K{K}"].iloc[0]
            v, l, h = (row["delta"], row["ci_lo"], row["ci_hi"]) if kind == "seed" else (row.delta_vs_exact, row.ci_lo, row.ci_hi)
            vals.append(v); lo.append(l); hi.append(h)
        st = te.cond_style(cal, "Weight"); st.pop("markerfacecolor", None); st.pop("marker", None)
        ax.plot([0, 1], vals, label={"MinMax": "MinMax", "FW": "FWM"}[cal], zorder=3, **st)
        for xi, sur, v, l, h in zip([0, 1], ("Weight", "Prob"), vals, lo, hi):
            ss = te.cond_style(cal, sur)
            ax.errorbar([xi], [v], yerr=[[v - l], [h - v]], fmt=ss["marker"], color=ss["color"], markerfacecolor=ss["markerfacecolor"],
                        ecolor="black", elinewidth=0.6, capsize=1.5, ms=4, zorder=4)
    lerp = df[df.cond == f"LERP K{K}"]
    if len(lerp):
        lv = lerp.iloc[0]["delta"] if kind == "seed" else lerp.iloc[0].delta_vs_exact
        ax.axhline(lv, color=ps.OI["green"], ls=":", lw=1.0, zorder=2, label="LERP, same K")
    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Weight-STE", "Prob-STE"], fontsize=ps.MIN_PT); ax.set_xlim(-0.35, 1.35)


def fig3_factorial():
    d5 = pd.read_csv(D / "endpoint_by_condition.csv")
    fig, axes = ps.figure(ncols=4, height=1.3)
    _interaction_panel(axes[0], endpoints(B), 4, None); ps.panel_label(axes[0], "a", "250M, K=4")
    _interaction_panel(axes[1], endpoints(B), 16, None); ps.panel_label(axes[1], "b", "250M, K=16")
    axes[0].set_ylabel("\u0394NLL vs softmax")
    axes[0].set_ylim(-0.02, 0.45); axes[1].set_ylim(-0.02, 0.45); axes[1].set_yticklabels([])
    # (c) five-seed mean delta vs K, all five families
    ax = axes[2]
    for f in FAM_ORDER:
        d = d5[d5.label == f].sort_values("K")
        ax.errorbar(d.K, d.delta, yerr=[d.delta - d.ci_lo, d.ci_hi - d.delta], ecolor="black", elinewidth=0.6,
                    capsize=1.5, label=NAME[f], zorder=3, **_fam_style(f))
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axhline(0, color="black", lw=0.5); ax.set_yticks([0, 0.1, 0.2]); ax.set_ylim(-0.02, 0.23)
    ax.set_xlabel("K"); ax.set_ylabel("\u0394NLL vs softmax")
    ps.panel_label(ax, "c", "5 seeds, 100M")
    # (d) Prob - Weight under each calibration vs K, paired by seed
    ax = axes[3]
    dbs = pd.read_csv(D / "delta_by_seed.csv").set_index("cond")
    from scipy import stats as _st
    for cal, pre in (("MinMax", "MinMax"), ("FW", "QRM")):
        Ks, m, lo, hi = [], [], [], []
        for K in (4, 8, 16, 32, 64):
            x = dbs.loc[f"{pre}-Prob K{K}"].values.astype(float) - dbs.loc[f"{pre}-Weight K{K}"].values.astype(float)
            se = x.std(ddof=1) / np.sqrt(len(x)); tcrit = _st.t.ppf(0.975, len(x) - 1)
            Ks.append(K); m.append(x.mean()); lo.append(x.mean() - tcrit * se); hi.append(x.mean() + tcrit * se)
        m, lo, hi = map(np.array, (m, lo, hi))
        st = te.cond_style(cal, "Weight"); st.pop("markerfacecolor", None); st.pop("marker", None)
        ax.errorbar(Ks, m, yerr=[m - lo, hi - m], ecolor="black", elinewidth=0.6, capsize=1.5, marker="o", ms=3.5, zorder=3,
                    label={"MinMax": "under MinMax", "FW": "under FWM"}[cal], **st)
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axhline(0, color="black", lw=0.5); ax.set_yticks([-0.1, -0.05, 0]); ax.set_ylim(-0.115, 0.02)
    ax.set_xlabel("K"); ax.set_ylabel("Prob \u2212 Weight")
    ps.panel_label(ax, "d", "Prob \u2212 Weight")  # colours as in the figure legend (MinMax / fixed window)
    # figure legend: calibration lines + surrogate marker fill + families of (c)
    h1, l1 = axes[0].get_legend_handles_labels()
    fill_h = [Line2D([], [], marker="o", color="black", markerfacecolor="black", ls="none", ms=4),
              Line2D([], [], marker="o", color="black", markerfacecolor="white", ls="none", ms=4)]
    fig.legend(h1 + fill_h, l1 + ["filled = " + PRE, "hollow = " + POST], loc="outside upper center", ncols=5, handlelength=1.8)
    save(fig, "fig3_factorial_2x2")


def figS_seed_noise():
    """Appendix: five-seed 2x2xK effects and the seed-SD vs evaluation-SE scatter."""
    fa = pd.read_csv(D / "factorial_calibration_backward.csv")
    sn = pd.read_csv(D / "seed_noise.csv")
    fig, axes = ps.figure(ncols=2, height=1.7, gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    eff = [("calibration QRM minus MinMax", "calibration (FWM $-$ MinMax)", dict(color="black", linestyle="-", marker="o", markerfacecolor="black")),
           ("backward Prob-STE minus Weight-STE", "surrogate (Prob $-$ Weight)", dict(color=ps.GRAY, linestyle="-", marker="s", markerfacecolor=ps.GRAY)),
           ("interaction", "interaction", dict(color="black", linestyle="--", marker="^", markerfacecolor="black"))]
    for key, lab, st in eff:
        d = fa[(fa.effect == key) & (fa.K != "pooled")].copy(); d["K"] = d.K.astype(int); d = d.sort_values("K")
        ax.errorbar(d.K, d.delta, yerr=[d.delta - d.ci_lo, d.ci_hi - d.delta], ecolor="black", elinewidth=0.6, capsize=1.5, zorder=3, label=lab, **st)
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axhline(0, color="black", lw=0.5); ax.set_yticks([-0.15, -0.1, -0.05, 0, 0.05]); ax.set_ylim(-0.175, 0.085)
    ax.set_xlabel("$K$"); ax.set_ylabel("effect (nats), 95% $t$-CI")
    ax.legend(loc="lower right", handlelength=1.6, fontsize=ps.MIN_PT)
    ps.panel_label(ax, "a", "2×2×K effects, paired by seed")
    ax = axes[1]
    s = sn.dropna(subset=["sd_seed_of_delta"]).copy(); s["fam"] = s.cond.map(fam_of)
    for f in FAM_ORDER:
        d = s[s.fam == f]
        ax.scatter(d.eval_se_paired * 1000, d.sd_seed_of_delta * 1000, s=14, zorder=3, label=NAME[f], **_scatter_style(f))
    ax.plot([0, 15], [0, 15], color=ps.GRAY, lw=0.6, ls=":")
    ax.text(2.55, 0.9, "1:1", fontsize=ps.SMALL_PT, color=ps.GRAY, ha="left", va="bottom")
    ax.set_xlim(0, 3.2); ax.set_ylim(0, 15); ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("paired eval SE (10\u207b\u00b3 nats)"); ax.set_ylabel("seed SD (10\u207b\u00b3 nats)")
    ps.panel_label(ax, "b", "seed SD vs eval SE")
    _family_legend(fig, [axes[1]], ["LERP", "MinMax\u2013Weight", "MinMax\u2013Prob", "FWM\u2013Weight", "FWM\u2013Prob"], 5)
    save(fig, "figS_seed_noise")


# ----------------------------------------------------------------------------------------------------
# Fig 4: K scaling and horizon (2.5B suite)
# ----------------------------------------------------------------------------------------------------
def fig4_k_horizon():
    ea = endpoints(A)
    tr = pd.read_csv(A / "trajectory/contrast_trajectories.csv")
    fig, axes = ps.figure(ncols=3, height=1.3, gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    ax = axes[0]
    for f in FAM_ORDER:
        d = ea[ea.label == f].sort_values("K")
        if d.empty:
            continue
        ax.errorbar(d.K, d.delta_vs_exact, yerr=[d.delta_vs_exact - d.ci_lo, d.ci_hi - d.delta_vs_exact],
                    ecolor="black", elinewidth=0.6, capsize=1.5, label=NAME[f], zorder=3, **_fam_style(f))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 16, 32, 64]); ax.set_xticklabels(["1", "2", "4", "16", "32", "64"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    _symlog_axis(ax); ax.set_ylim(-0.004, 1.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("$K$ (measured values only)"); ax.set_ylabel("$\\Delta$NLL vs softmax (nats)")
    ps.panel_label(ax, "a", "endpoint at 2.5B")
    ax = axes[1]
    series = [("LERP K4", _fam_style("LERP", marker=False)),
              ("MinMax-Weight K4", _fam_style("MinMax-Weight", marker=False)),
              ("MinMax-Prob K4", _fam_style("MinMax-Prob", marker=False)),
              ("QRM-Weight K4", _fam_style("QRM-Weight", marker=False)),
              ("QRM-Prob K4", _fam_style("QRM-Prob", marker=False))]
    for cname, st in series:
        d = tr[tr.contrast == f"{cname} - Exact"].sort_values("tokens_m")
        ax.plot(d.tokens_m / 1000, d.delta, label=pretty(cname), **st)
    # FWM-LERP K4 at 2.5B (Appendix I.2), from the Appendix I result tables csv/ (run rb-E1b-01 minus suite A softmax)
    e1b = pd.read_csv(AM / "rb-rebuttal_v5/reports/rb-dossier_20261114/csv/trajectories.csv")
    e1b = e1b[(e1b.run_id == "rb-E1b-01") & e1b.dnll.notna()].sort_values("tokens_m")
    ax.plot(e1b.tokens_m / 1000, e1b.dnll, label="FWM\u2013LERP", **FWMLERP)
    _symlog_axis(ax, ticks=[0, 0.01, 0.1, 1, 5]); ax.set_ylim(-0.004, 8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("training tokens (B)")
    ps.panel_label(ax, "b", "along training, $K{=}4$")
    ax = axes[2]
    for cname, lab, ylab in (("QRM-Weight K4 - LERP K4", "FWM\u2013W", 0.046), ("QRM-Prob K4 - LERP K4", "FWM\u2013P", 0.018),
                             ("MinMax-Prob K4 - LERP K4", "MM\u2013P", 0.074)):
        d = tr[tr.contrast == cname].sort_values("tokens_m")
        st = _fam_style(fam_of(cname.split(" - ")[0]), marker=False)
        ax.plot(d.tokens_m / 1000, d.delta, **st)
        ax.fill_between(d.tokens_m / 1000, d.ci_lo, d.ci_hi, color=st["color"], alpha=0.25, lw=0)
        ax.text(d.tokens_m.iloc[-1] / 1000 + 0.08, ylab, lab + " $-$ LERP", va="center", ha="left", fontsize=ps.SMALL_PT, color=st["color"])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlim(0, 3.9); ax.set_xticks([0, 1, 2]); ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel("training tokens (B)"); ax.set_ylabel("paired $\\Delta$NLL (nats)")
    ps.panel_label(ax, "c", "Nearest $-$ LERP, $K{=}4$")
    fams = ["LERP", "MinMax-Weight", "MinMax-Prob", "QRM-Weight", "QRM-Prob"]
    fig.legend([Line2D([], [], **_fam_style(f)) for f in fams] + [Line2D([], [], **FWMLERP)], [NAME[f] for f in fams] + ["FWM\u2013LERP (b)"],
               loc="outside upper center", ncols=6, handlelength=1.8)
    save(fig, "fig4_k_scaling_horizon")


def figS_minmaxw_ksweep():
    tr = pd.read_csv(A / "trajectory/contrast_trajectories.csv")
    fig, ax = ps.figure(width=3.4, height=1.5)
    for K in (4, 16, 32, 64):
        st = _fam_style("MinMax-Weight", k_shade=(None if K == 4 else K), marker=False)
        if K == 64:
            st["linestyle"] = ":"
        d = tr[tr.contrast == f"MinMax-Weight K{K} - Exact"].sort_values("tokens_m")
        ax.plot(d.tokens_m / 1000, d.delta, label=f"MinMax\u2013Weight K{K}", **st)
    _symlog_axis(ax, ticks=[0, 0.01, 0.1, 1, 5]); ax.set_ylim(-0.004, 8)
    ax.axhline(0, color="black", lw=0.6); ax.set_xlabel("training tokens (B)"); ax.set_ylabel("$\\Delta$NLL vs softmax (nats)")
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="outside upper center", ncols=2, handlelength=1.8)
    save(fig, "figS_minmaxw_ksweep")


# ----------------------------------------------------------------------------------------------------
# Fig 5 (main): geometry and resolution -- h and TV along training, endpoint gap vs h
# ----------------------------------------------------------------------------------------------------
def fig5_geometry_main():
    g = pd.read_csv(A / "geometry/geometry_with_nll_gap.csv")
    conds = ["LERP K4", "MinMax-Weight K4", "MinMax-Prob K4", "QRM-Weight K4", "QRM-Prob K4", "MinMax-Weight K64"]
    fig, axes = ps.figure(ncols=3, height=1.42, gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for ax, (m, lab, letter, logt) in zip(axes[:2], (("width_native_mean", "$h$ (nats)", "a", [1, 10]),
                                                      ("tv_native_exact_mean", "TV", "b", None))):
        for cname in conds:
            d = g[g.cond == cname].sort_values("tokens_m")
            if d[m].isna().all():
                continue
            st = _fam_style(fam_of(cname), k_shade=(64 if cname == "MinMax-Weight K64" else None), marker=False)
            if cname == "MinMax-Weight K64":
                st["linestyle"] = ":"
            ax.plot(d.tokens_m / 1000, d[m], label=pretty(cname), **st)
        if logt:
            _log_axis(ax, logt)
        ax.set_xlabel("training tokens (B)"); ax.set_xticks([1, 2]); ax.set_ylabel(lab)
        ps.panel_label(ax, letter, {"a": "native width $h$", "b": "TV(native, softmax)"}[letter])
    axes[1].set_ylim(0, 1)
    # (c) endpoint gap vs endpoint h, all conditions with a native width
    ax = axes[2]
    last = g.sort_values("tokens_m").groupby("cond").tail(1)
    last = last[last.cond != "Exact"].dropna(subset=["width_native_mean"])
    for f in FAM_ORDER:
        d = last[last.label == f]
        if d.empty:
            continue
        ax.scatter(d.width_native_mean, d.gap_vs_exact, s=16, zorder=3, label=NAME[f], marker=("s" if f == "LERP" else "o"), **_scatter_style(f))
    r64 = last[last.cond == "MinMax-Weight K64"].iloc[0]
    ax.scatter([r64.width_native_mean], [r64.gap_vs_exact], marker="+", s=55, color="black", linewidths=0.8, zorder=5)

    ax.set_xscale("log"); ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    _symlog_axis(ax); ax.set_ylim(-0.004, 1.5); ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("endpoint native width $h$ (nats)"); ax.set_ylabel("$\\Delta$NLL vs softmax")
    ps.panel_label(ax, "c", "gap vs $h$ (symlog)")
    hl = {}
    for a_ in axes[:2]:
        h_, l_ = a_.get_legend_handles_labels(); hl.update(dict(zip(l_, h_)))
    ab_order = ["MinMax\u2013Weight K4", "MinMax\u2013Prob K4", "MinMax\u2013Weight K64", "FWM\u2013Weight K4", "FWM\u2013Prob K4", "LERP K4"]
    def pt(**kw):
        kw.setdefault("ms", 4.5)
        return Line2D([], [], ls="none", **kw)
    c_h = [pt(marker="o", color=te.CAL_COLOR["MinMax"], markerfacecolor=te.CAL_COLOR["MinMax"]),
           pt(marker="o", color=te.CAL_COLOR["MinMax"], markerfacecolor="white"),
           pt(marker="o", color=te.CAL_COLOR["FW"], markerfacecolor=te.CAL_COLOR["FW"]),
           pt(marker="o", color=te.CAL_COLOR["FW"], markerfacecolor="white"),
           pt(marker="s", color=ps.OI["green"], markerfacecolor=ps.OI["green"]),
           pt(marker="+", color="black", ms=7, markeredgewidth=0.8)]
    c_l = ["MinMax\u2013Weight", "MinMax\u2013Prob", "FWM\u2013Weight", "FWM\u2013Prob", "LERP", "MinMax\u2013Weight K64"]
    fig.legend([hl[o] for o in ab_order] + c_h, ab_order + c_l, loc="outside upper center", ncols=4, handlelength=1.6, columnspacing=1.0)
    save(fig, "fig5_geometry_main")


def fig5_geometry():
    """Appendix: span and top-set mass along training at 2.5B (h and TV are plotted by fig5_geometry_main from the same CSV,
    same six conditions and checkpoints; softmax has no native h and TV=0)."""
    g = pd.read_csv(A / "geometry/geometry_with_nll_gap.csv")
    conds = ["Exact", "LERP K4", "MinMax-Weight K4", "MinMax-Prob K4", "QRM-Weight K4", "QRM-Prob K4", "MinMax-Weight K64"]
    fig, axes = ps.figure(width=0.72 * ps.TEXTWIDTH_IN, ncols=2, height=1.5)
    metrics = [("span_mean", "row score span", [10, 100]), ("exact_mass_top_bucket_mean", "top level/interval mass", None)]
    for ax, (m, lab, logt) in zip(axes, metrics):
        for cname in conds:
            d = g[g.cond == cname].sort_values("tokens_m")
            if d[m].isna().all():
                continue
            st = _fam_style(fam_of(cname), k_shade=(64 if cname == "MinMax-Weight K64" else None), marker=False)
            if cname == "MinMax-Weight K64":
                st["linestyle"] = ":"
            ax.plot(d.tokens_m / 1000, d[m], label=pretty(cname), **st)
        if logt:
            _log_axis(ax, logt)
        ax.set_xlabel("training tokens (B)"); ax.set_xticks([1, 2])
        ps.panel_label(ax, "ab"[metrics.index((m, lab, logt))], lab)
    _family_legend(fig, [axes[0]], ["softmax", "LERP K4", "MinMax\u2013Weight K4", "MinMax\u2013Weight K64", "MinMax\u2013Prob K4", "FWM\u2013Weight K4", "FWM\u2013Prob K4"], 4)
    save(fig, "fig5_geometry_2p5b")


# ----------------------------------------------------------------------------------------------------
# Fig S: downstream (2.5B suite): transfer NLL and benchmark accuracies
# ----------------------------------------------------------------------------------------------------
def fig7_downstream():
    l1 = pd.read_csv(DS / "layer1/ppl_transfer.csv")
    pc = pd.read_csv(DS / "layer3/stageC/tables/paired_comparisons.csv")
    ea = endpoints(A)
    order = ea[ea.cond != "Exact"].sort_values("delta_vs_exact", ascending=False).cond.tolist()
    # two panels: BLiMP and LAMBADA accuracy differences; the held-out NLL is not plotted
    fig, axes = ps.figure(width=0.62 * ps.TEXTWIDTH_IN, ncols=2, height=1.95, sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (cfg, metric, title) in zip(axes, (("blimp", "acc_macro", "BLiMP acc (pp)"), ("lambada_openai", "acc", "LAMBADA acc (pp)"))):
        d = pc[(pc.cfg == cfg) & (pc.metric == metric)].set_index("method").loc[order]
        yy = np.arange(len(d))[::-1]
        ax.errorbar(d["diff"], yy, xerr=[d["diff"] - d.ci_lo, d.ci_hi - d["diff"]], fmt="none", ecolor="black", capsize=1.5, elinewidth=0.6)
        for yi, c, v in zip(yy, d.index, d["diff"]):
            ax.scatter([v], [yi], s=16, zorder=3, **_scatter_style(fam_of(c)))
        ax.axvline(0, color=ps.GRAY, lw=0.5)
        ax.set_xlabel("method $-$ softmax (pp)")
    axes[0].set_yticks(y); axes[0].set_yticklabels([pretty(c) for c in order]); axes[1].tick_params(labelleft=False)
    ps.panel_label(axes[0], "a", "BLiMP acc (pp)"); ps.panel_label(axes[1], "b", "LAMBADA acc (pp)")
    save(fig, "fig7_downstream")



def fig8_250m_onset():
    tr = pd.read_csv(B / "trajectory/contrast_trajectories.csv")
    g = pd.read_csv(B / "geometry/geometry_with_nll_gap.csv")
    fig, axes = ps.figure(ncols=3, height=1.35)
    ax = axes[0]
    for cname in ("LERP K4", "MinMax-Weight K4", "MinMax-Prob K4", "QRM-Weight K4", "QRM-Prob K4"):
        d = tr[tr.contrast == f"{cname} - Exact"].sort_values("tokens_m")
        st = _fam_style(fam_of(cname), marker=False)
        ax.plot(d.tokens_m, d.delta, label=pretty(cname), **st)
        ax.fill_between(d.tokens_m, d.ci_lo, d.ci_hi, color=st["color"], alpha=0.25, lw=0)
    ax.axhline(0, color="black", lw=0.5); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("training tokens (M)"); ax.set_ylabel("$\\Delta$ NLL vs softmax (nats)")
    ps.panel_label(ax, "a", "124M @ 250M, K4")
    for ax, (m, lab, letter) in zip(axes[1:], (("span_mean", "row score span", "b"), ("tv_native_exact_mean", "TV(native, exact)", "c"))):
        for cname in ("Exact", "LERP K4", "MinMax-Weight K4", "MinMax-Prob K4", "QRM-Weight K4", "QRM-Prob K4"):
            d = g[g.cond == cname].sort_values("tokens_m")
            if d[m].isna().all():
                continue
            ax.plot(d.tokens_m, d[m], label=pretty(cname), **_fam_style(fam_of(cname), marker=False))
        ax.set_xlabel("training tokens (M)")
        ps.panel_label(ax, letter, lab)
    _family_legend(fig, [axes[1]], ["softmax", "LERP K4", "MinMax\u2013Weight K4", "MinMax\u2013Prob K4", "FWM\u2013Weight K4", "FWM\u2013Prob K4"], 3)
    save(fig, "fig8_250m_onset_geometry")


# ----------------------------------------------------------------------------------------------------
# Fig 6 / S5: downstream heatmaps (method - Exact, pp)
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# Fig 6: downstream heatmaps (matched K=4 | near-baseline group), with training-domain dNLL and Exact accuracy
# ----------------------------------------------------------------------------------------------------
CFGS = [("blimp", "acc_macro", "BLiMP"), ("hellaswag", "acc_norm", "HellaSwag"), ("piqa", "acc_norm", "PIQA"),
        ("arc_easy", "acc_norm", "ARC-Easy"), ("lambada_openai", "acc", "LAMBADA"), ("sciq_default", "acc", "SciQ"),
        ("sciq_nosupport", "acc", "SciQ no-supp.")]


def _heat_matrix(conds):
    pc = pd.read_csv(DS / "layer3/stageC/tables/paired_comparisons.csv")
    pe = pd.read_csv(DS / "layer3/stageC/tables/point_estimates.csv")
    M = np.zeros((len(conds), len(CFGS))); S = np.zeros_like(M, dtype=bool); ex = []
    for j, (cfg, met, _) in enumerate(CFGS):
        ex.append(100 * pe[(pe.cfg == cfg) & (pe.metric == met) & (pe.condition == "Exact")].value.iloc[0])
        for i, c in enumerate(conds):
            r = pc[(pc.cfg == cfg) & (pc.metric == met) & (pc.method == c)].iloc[0]
            M[i, j] = r["diff"]; S[i, j] = bool(r.significant_q05)
    return M, S, ex


def _heat_panel(ax, conds, vmax, title, cbar=True, seps=()):
    ea = endpoints(A).set_index("cond")
    M, S, ex = _heat_matrix(conds)
    im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(CFGS))); ax.set_xticklabels([f"{t}\n({e:.1f})" for (_, _, t), e in zip(CFGS, ex)], fontsize=ps.SMALL_PT)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([f"{pretty(c)}  {ea.loc[c].delta_vs_exact:+.3f}" for c in conds], fontsize=ps.SMALL_PT)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(conds)):
        for j in range(len(CFGS)):
            ax.text(j, i, f"{M[i, j]:+.1f}" + ("*" if S[i, j] else ""), ha="center", va="center", fontsize=ps.SMALL_PT,
                    color="white" if abs(M[i, j]) > 0.55 * vmax else "black")
    ax.axvline(4.5, color="white", lw=2.0)  # SciQ pair grouped right of LAMBADA
    for y in seps:
        ax.axhline(y, color="white", lw=2.0)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("pp", fontsize=ps.MIN_PT); cb.ax.tick_params(labelsize=ps.MIN_PT)
    ax.set_title(title, loc="left")


def fig6_downstream_heatmap():
    fig, axes = ps.figure(nrows=2, height=1.65)
    k4 = ["LERP K4", "MinMax-Weight K4", "MinMax-Prob K4", "QRM-Weight K4", "QRM-Prob K4"]
    near = ["LERP K2", "LERP K4", "LERP K16", "LERP K32", "QRM-Weight K16"]
    _heat_panel(axes[0], k4, 20, "$\\bf{a}$  matched $K{=}4$ (colour scale $\\pm$20 pp)")
    _heat_panel(axes[1], near, 5, "$\\bf{b}$  near-baseline group (colour scale $\\pm$5 pp)")
    axes[0].set_xticklabels([]); axes[0].tick_params(labelbottom=False)
    save(fig, "fig9_downstream_heatmap")


def figS_downstream_heatmap_all():
    ea = endpoints(A)
    order = ea[ea.cond != "Exact"].sort_values("delta_vs_exact", ascending=False).cond.tolist()
    fig, ax = ps.figure(height=3.6)
    _heat_panel(ax, order, 20, "all 14 conditions by 2.5B $\\Delta$NLL (large-gap | intermediate | near-baseline)", seps=(4.5, 8.5))
    save(fig, "fig9b_downstream_heatmap_all")


def fig10_width_gap():
    g = pd.read_csv(D / "geometry_by_condition.csv").set_index("cond")
    ep = pd.read_csv(D / "endpoint_by_condition.csv").set_index("cond")
    ga = pd.read_csv(D / "geometry_association.csv").set_index("metric")
    KS = {4: 6, 8: 10, 16: 15, 32: 22, 64: 30}
    fig, axes = ps.figure(ncols=2, height=1.65)
    for ax, (m, lab, letter) in zip(axes, (("width_native_mean_mean", "native interval width $h$ (mean over layers, seeds)", "a"),
                                           ("span_mean_mean", "row score span (mean over layers, seeds)", "b"))):
        for f in FAM_ORDER:
            cs = [c for c in ep.index if c != "Exact" and fam_of(c) == f]
            ax.scatter(g.loc[cs, m], ep.loc[cs, "delta"], s=[KS[k_of(c)] for c in cs], label=NAME[f], zorder=3, **_scatter_style(f))
        rho = ga.loc[m.replace("_mean_mean", "_mean"), "spearman_vs_delta"]
        ax.set_yscale("symlog", linthresh=0.01); ax.set_yticks([0, 0.01, 0.1, 0.2]); ax.set_yticklabels(["0", "0.01", "0.1", "0.2"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.axhline(0, color="black", lw=0.5); ax.set_xlabel(lab)
        ps.panel_label(ax, letter, f"Spearman ρ = {rho:+.3f} (n = 25 conditions)")
    axes[0].set_ylabel("$\\Delta$NLL vs softmax (nats)")
    h, l = axes[0].get_legend_handles_labels()
    size_h = [plt.scatter([], [], s=KS[k], facecolors="white", edgecolors="black", linewidths=0.7) for k in (4, 16, 64)]
    fig.legend(h + size_h, l + ["K=4", "K=16", "K=64 (marker area grows with K)"], loc="outside upper center", ncols=4)
    save(fig, "fig10_width_vs_gap_5seed")



if __name__ == "__main__":
    fig1_schematic()
    fig2_detach()
    figS_detach_extra()
    # figS_detach_rep() is not called: its figure is not in the paper
    figS_e1_channels()
    figS_lowprec()
    fig3_factorial()
    figS_seed_noise()
    fig4_k_horizon()
    figS_minmaxw_ksweep()
    fig5_geometry_main()
    fig5_geometry()
    fig7_downstream()
    fig8_250m_onset()
    fig6_downstream_heatmap()
    # figS_downstream_heatmap_all() is not called: tab_downstream_stageC gives these differences
    fig10_width_gap()
