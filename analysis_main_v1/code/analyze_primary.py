#!/usr/bin/env python3
"""analysis_main_v1 — statistics, tables and figures for the three primary suites.

Runs locally (pandas/matplotlib) on a mirror of the gpu-host-A outputs:
  manifests/primary_37_runs.tsv, manifests/primary_37_checkpoints.tsv
  eval_native_v1/<suite>/<run>/<ckpt>/{summary.json, per_block_nll.npy, geometry_layers.csv}
  training_logs/<suite>/<run>/{log.csv, attention_stats.csv, training_health.csv, occupancy.csv}

Writes per suite:
  <suite>/endpoint_fullval/   endpoint table, matched contrasts, factorial effects, anchor reproduction
  <suite>/trajectory/         checkpoint NLL table, contrast trajectories, separation onset
  <suite>/geometry/           layer-mean geometry per checkpoint, final layer profiles, association
  <suite>/training_logs/      dense logged curves (recovery-deduplicated), health summary
  <suite>/figures/            PNG figures

Statistics: every contrast is a paired difference of per-block validation NLL on identical blocks.
Uncertainty is a circular moving-block bootstrap over validation blocks (block length 16) and only
reflects evaluation-set sampling. All primary suites are single training seed (1337); no CI here
speaks to training-seed variance.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 4000
BOOT_BLOCK = 16
RNG_SEED = 20260917

SUITES = {
    "B_factorial_124m_250m": dict(title="124M @ 250M controlled factorial (FineWebEdu-3B)", dataset="FineWebEdu-3B"),
    "A_longhorizon_124m_2p5b": dict(title="124M @ 2.5B long-horizon (FineWebEdu-3B)", dataset="FineWebEdu-3B"),
    "C_scale_1b_100m": dict(title="1B @ 100M scale-up (WikiText-103)", dataset="WikiText-103"),
}

# Categorical slots in fixed order (reference palette, light mode). Exact is the neutral reference ink.
COLOR = {"Exact": "#52514e", "LERP": "#2a78d6", "MinMax-Weight": "#eb6834", "QRM-Weight": "#1baf7a",
         "MinMax-Prob": "#eda100", "QRM-Prob": "#e87ba4"}
DASH = {"Exact": (0, (1, 1.5)), "LERP": "-", "MinMax-Weight": "-", "QRM-Weight": "-",
        "MinMax-Prob": (0, (5, 2)), "QRM-Prob": (0, (5, 2))}
LABEL_ORDER = ["Exact", "LERP", "MinMax-Weight", "MinMax-Prob", "QRM-Weight", "QRM-Prob"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
})


def cond(label, K):
    return label if label == "Exact" else f"{label} K{K}"


# ----------------------------------------------------------------------------- loading
def load_runs():
    runs = pd.read_csv(ROOT / "manifests/primary_37_runs.tsv", sep="\t", dtype=str)
    runs["K"] = runs["K"].fillna("")
    runs["cond"] = [cond(l, k) for l, k in zip(runs.label, runs.K)]
    return runs


def load_eval(runs):
    rows, nll, geo = [], {}, []
    for _, r in runs.iterrows():
        base = ROOT / "eval_native_v1" / r.suite / r.run_name
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            s = d / "summary.json"
            if not s.exists() or ".tmp_" in d.name:
                continue
            j = json.loads(s.read_text())
            key = (r.run_name, j["ckpt_index"])
            nll[key] = np.load(d / "per_block_nll.npy")
            rows.append(dict(suite=r.suite, run_name=r.run_name, label=r.label, K=r.K, cond=r.cond,
                             backward=r.backward, calibration=r.calibration, ckpt_index=j["ckpt_index"],
                             is_final=j["is_final"], step=j["step"], tokens_m=j["tokens_done"] / 1e6,
                             mean_nll=j["mean_nll"], n_blocks=j["n_blocks"], checkpoint=j["checkpoint"],
                             eval_code=j["eval_code"]))
            g = d / "geometry_layers.csv"
            if g.exists():
                gd = pd.read_csv(g).rename(columns={"z_p5": "z_p50"})
                gd.insert(0, "run_name", r.run_name)
                gd.insert(1, "ckpt_index", j["ckpt_index"])
                gd.insert(2, "tokens_m", j["tokens_done"] / 1e6)
                geo.append(gd)
    ev = pd.DataFrame(rows)
    geo = pd.concat(geo, ignore_index=True) if geo else pd.DataFrame()
    return ev, nll, geo


# ----------------------------------------------------------------------------- statistics
def boot_indices(n, rng):
    """Circular moving-block bootstrap indices, shape (N_BOOT, n)."""
    n_blk = int(np.ceil(n / BOOT_BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, n_blk))
    idx = (starts[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n
    return idx.reshape(N_BOOT, -1)[:, :n]


def linear_contrast(vecs, weights, idx):
    d = sum(w * v for w, v in zip(weights, vecs))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(delta=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                frac_blocks_negative=float((d < 0).mean()), n_blocks=int(len(d)))


def ci_class(row, margin=0.0):
    if row["ci_lo"] > margin:
        return "positive"
    if row["ci_hi"] < -margin:
        return "negative"
    return "CI includes 0"


# ----------------------------------------------------------------------------- contrasts per suite
def contrast_specs(suite, conds):
    """Return list of (name, family, [(cond, weight), ...]). Only emitted when all conds exist."""
    S = []

    def add(name, fam, terms):
        if any(name == x[0] for x in S):
            return
        if all(c in conds for c, _ in terms):
            S.append((name, fam, terms))

    Ks = sorted({int(c.split(" K")[1]) for c in conds if " K" in c})
    for c in sorted(conds):
        if c != "Exact":
            add(f"{c} - Exact", "vs_exact", [(c, 1), ("Exact", -1)])
    for K in Ks:
        L = f"LERP K{K}"
        for lab in ("MinMax-Weight", "MinMax-Prob", "QRM-Weight", "QRM-Prob"):
            add(f"{lab} K{K} - {L}", "vs_lerp_matched_K", [(f"{lab} K{K}", 1), (L, -1)])
        add(f"MinMax-Prob K{K} - MinMax-Weight K{K}", "backward_prob_minus_weight",
            [(f"MinMax-Prob K{K}", 1), (f"MinMax-Weight K{K}", -1)])
        add(f"QRM-Prob K{K} - QRM-Weight K{K}", "backward_prob_minus_weight",
            [(f"QRM-Prob K{K}", 1), (f"QRM-Weight K{K}", -1)])
        add(f"QRM-Weight K{K} - MinMax-Weight K{K}", "calibration_qrm_minus_minmax",
            [(f"QRM-Weight K{K}", 1), (f"MinMax-Weight K{K}", -1)])
        add(f"QRM-Prob K{K} - MinMax-Prob K{K}", "calibration_qrm_minus_minmax",
            [(f"QRM-Prob K{K}", 1), (f"MinMax-Prob K{K}", -1)])
        cells = [f"MinMax-Weight K{K}", f"MinMax-Prob K{K}", f"QRM-Weight K{K}", f"QRM-Prob K{K}"]
        mw, mp, qw, qp = cells
        add(f"K{K} 2x2 main effect: calibration (QRM - MinMax, avg over backward)", "factorial_2x2",
            [(qw, .5), (qp, .5), (mw, -.5), (mp, -.5)])
        add(f"K{K} 2x2 main effect: backward (Prob - Weight, avg over calibration)", "factorial_2x2",
            [(mp, .5), (qp, .5), (mw, -.5), (qw, -.5)])
        add(f"K{K} 2x2 interaction: (QRM Prob-Weight) - (MinMax Prob-Weight)", "factorial_2x2",
            [(qp, 1), (qw, -1), (mp, -1), (mw, 1)])
    # 2x2x2 with K (250M factorial): K16 - K4 per cell and averaged
    if {4, 16} <= set(Ks):
        for lab in ("MinMax-Weight", "MinMax-Prob", "QRM-Weight", "QRM-Prob"):
            add(f"{lab} K16 - {lab} K4", "k_effect_within_cell", [(f"{lab} K16", 1), (f"{lab} K4", -1)])
        add("K main effect (K16 - K4, avg over 4 hard-Nearest cells)", "factorial_2x2x2",
            [(f"{l} K16", .25) for l in ("MinMax-Weight", "MinMax-Prob", "QRM-Weight", "QRM-Prob")]
            + [(f"{l} K4", -.25) for l in ("MinMax-Weight", "MinMax-Prob", "QRM-Weight", "QRM-Prob")])
        add("K x calibration interaction: (QRM - MinMax at K16) - (QRM - MinMax at K4), avg backward",
            "factorial_2x2x2",
            [("QRM-Weight K16", .5), ("QRM-Prob K16", .5), ("MinMax-Weight K16", -.5), ("MinMax-Prob K16", -.5),
             ("QRM-Weight K4", -.5), ("QRM-Prob K4", -.5), ("MinMax-Weight K4", .5), ("MinMax-Prob K4", .5)])
        add("K x backward interaction: (Prob - Weight at K16) - (Prob - Weight at K4), avg calibration",
            "factorial_2x2x2",
            [("MinMax-Prob K16", .5), ("QRM-Prob K16", .5), ("MinMax-Weight K16", -.5), ("QRM-Weight K16", -.5),
             ("MinMax-Prob K4", -.5), ("QRM-Prob K4", -.5), ("MinMax-Weight K4", .5), ("QRM-Weight K4", .5)])
    # within-family K scaling (adjacent available K)
    for lab in ("LERP", "MinMax-Weight", "QRM-Weight"):
        fam = sorted(int(c.split(" K")[1]) for c in conds if c.startswith(lab + " K"))
        for a, b in zip(fam[:-1], fam[1:]):
            add(f"{lab} K{b} - {lab} K{a}", "k_scaling_adjacent", [(f"{lab} K{b}", 1), (f"{lab} K{a}", -1)])
    return S


def run_contrasts(sub_ev, nll, specs, ckpt_index, rng):
    by_cond = {r.cond: nll[(r.run_name, ckpt_index)] for r in sub_ev.itertuples()
               if (r.run_name, ckpt_index) in nll}
    if not by_cond:
        return pd.DataFrame()
    n = len(next(iter(by_cond.values())))
    idx = boot_indices(n, rng)
    out = []
    for name, fam, terms in specs:
        if not all(c in by_cond for c, _ in terms):
            continue
        res = linear_contrast([by_cond[c] for c, _ in terms], [w for _, w in terms], idx)
        out.append(dict(contrast=name, family=fam, terms=" ".join(f"{w:+g}*[{c}]" for c, w in terms), **res))
    df = pd.DataFrame(out)
    if len(df):
        df["ci_class"] = df.apply(ci_class, axis=1)
    return df


# ----------------------------------------------------------------------------- training logs
def dedupe_recovery(df):
    """Keep post-resume rows as canonical: when tokens decrease, drop earlier rows with tokens >= resumed token."""
    keep = []
    for _, row in df.iterrows():
        while keep and keep[-1]["tokens"] >= row["tokens"]:
            keep.pop()
        keep.append(row)
    return pd.DataFrame(keep).reset_index(drop=True)


def load_training_logs(suite, runs):
    logs, att, health = [], [], []
    for r in runs.itertuples():
        d = ROOT / "training_logs" / suite / r.run_name
        for fname, sink in (("log.csv", logs), ("attention_stats.csv", att), ("training_health.csv", health)):
            f = d / fname
            if f.exists():
                x = dedupe_recovery(pd.read_csv(f))
                x.insert(0, "run_name", r.run_name)
                x.insert(1, "cond", r.cond)
                x.insert(2, "label", r.label)
                x.insert(3, "K", r.K)
                sink.append(x)
    cat = lambda z: pd.concat(z, ignore_index=True) if z else pd.DataFrame()  # noqa: E731
    return cat(logs), cat(att), cat(health)


def health_summary(health, warmup_tokens, stages):
    out = []
    for (run, c), g in health.groupby(["run_name", "cond"], sort=False):
        g = g[g.tokens > warmup_tokens]
        for sname, (a, b) in stages.items():
            s = g[(g.tokens > a) & (g.tokens <= b)]
            if not len(s):
                continue
            out.append(dict(cond=c, run_name=run, stage=sname, n_windows=len(s),
                            clip_fraction_mean=s.clip_fraction.mean(),
                            windows_clip_ge_50pct=int((s.clip_fraction >= .5).sum()),
                            grad_norm_mean_median=s.grad_norm_mean.median(),
                            grad_norm_max_max=s.grad_norm_max.max(),
                            windows_gradmax_gt_10=int((s.grad_norm_max > 10).sum()),
                            nonfinite_grad_updates=int(s.nonfinite_grad_updates.sum()),
                            nonfinite_loss_steps=int(s.nonfinite_loss_steps.sum())))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- figures
def style_for(label):
    return dict(color=COLOR[label], linestyle=DASH[label])


def fig_save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def fig_nll_traj(tr, path, title, logx):
    labels = [l for l in LABEL_ORDER if l in set(tr.label)]
    fig, axes = plt.subplots(1, len(labels), figsize=(3.0 * len(labels), 3.0), sharey=True, squeeze=False)
    ex = tr[tr.label == "Exact"].sort_values("tokens_m")
    lo = tr.mean_nll.quantile(0.02) - 0.05
    hi = min(tr.mean_nll.max(), ex.mean_nll.iloc[len(ex) // 3] + 1.2) if len(ex) else tr.mean_nll.max()
    for ax, lab in zip(axes[0], labels):
        sub = tr[tr.label == lab]
        Ks = sorted(sub.K.unique(), key=lambda k: int(k) if k else 0)
        alphas = np.linspace(0.35, 1.0, len(Ks)) if len(Ks) > 1 else [1.0]
        if lab != "Exact" and len(ex):
            ax.plot(ex.tokens_m, ex.mean_nll, color=COLOR["Exact"], linestyle=DASH["Exact"], lw=1.5, label="Exact")
        for a, K in zip(alphas, Ks):
            s = sub[sub.K == K].sort_values("tokens_m")
            ax.plot(s.tokens_m, s.mean_nll, color=COLOR[lab], alpha=a, lw=2, marker="o", ms=3,
                    label=cond(lab, K))
        ax.set_title(lab, color=INK)
        ax.set_xlabel("training tokens (M)")
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(lo, hi)
        ax.legend(fontsize=7, loc="upper right")
    axes[0][0].set_ylabel("full-validation NLL (nats)")
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig_save(fig, path)


def fig_contrast_traj(ct, names, path, title, logx):
    names = [n for n in names if n in set(ct.contrast)]
    if not names:
        return
    ncol = min(3, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.7 * nrow), squeeze=False)
    for ax, n in zip(axes.flat, names):
        s = ct[ct.contrast == n].sort_values("tokens_m")
        first = n.split(" - ")[0].split(" K")[0]
        col = COLOR.get(first, INK2)
        ax.fill_between(s.tokens_m, s.ci_lo, s.ci_hi, color=col, alpha=0.18, lw=0)
        ax.plot(s.tokens_m, s.delta, color=col, marker="o", ms=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(n, fontsize=8, color=INK)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel("training tokens (M)")
        ax.set_ylabel("Δ NLL (nats)")
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle(title + "\n(band: 95% block-bootstrap CI over validation blocks; single training seed)",
                 color=INK, x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    fig_save(fig, path)


def fig_endpoint_bars(end, path, title):
    d = end.copy()
    d["order"] = d.label.map({l: i for i, l in enumerate(LABEL_ORDER)})
    d["Kn"] = d.K.replace("", "0").astype(int)
    d = d.sort_values(["Kn", "order"])
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(d) + 0.9))
    y = np.arange(len(d))
    ax.barh(y, d.delta_vs_exact, color=[COLOR[l] for l in d.label], height=0.7,
            hatch=["///" if "Prob" in l else "" for l in d.label], edgecolor="#fcfcfb", linewidth=0)
    ax.errorbar(d.delta_vs_exact, y, xerr=[d.delta_vs_exact - d.ci_lo, d.ci_hi - d.delta_vs_exact],
                fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    ax.set_yticks(y, d["cond"])
    ax.axvline(0, color=INK2, lw=0.8)
    for yi, v, nl in zip(y, d.delta_vs_exact, d.mean_nll):
        ax.text(max(v, 0) + 0.004, yi, f"{v:+.4f}  (NLL {nl:.4f})", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("final standardized NLL − Exact (nats)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", color=INK)
    xmax = d.ci_hi.max()
    ax.set_xlim(min(0, d.ci_lo.min()) - 0.01, xmax * 1.45 + 0.02)
    fig_save(fig, path)


GEO_METRICS = [
    ("span_mean", "row score span (max−min)"),
    ("width_native_mean", "native interval width h"),
    ("z_p001", "0.1% quantile of z = s − row max"),
    ("q_norm_mean", "query norm"),
    ("exact_mass_top_bucket_mean", "exact-softmax mass in top bucket"),
    ("eff_buckets_mass", "effective buckets (mass-weighted)"),
    ("tv_native_exact_mean", "TV(native P, exact P) on same scores"),
    ("frac_entries_below_tau6", "fraction of scores with z < −6"),
    ("exact_entropy_mean", "exact attention entropy (nats)"),
]


def fig_geometry_traj(gm, runs_sub, path, title, logx, conds=None):
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.4), squeeze=False)
    rr = runs_sub if conds is None else runs_sub[runs_sub.cond.isin(conds)]
    for ax, (m, lab) in zip(axes.flat, GEO_METRICS):
        for r in rr.itertuples():
            s = gm[gm.run_name == r.run_name].sort_values("tokens_m")
            if not len(s) or s[m].isna().all():
                continue
            ax.plot(s.tokens_m, s[m], label=r.cond, marker="o", ms=2.5, lw=1.6, **style_for(r.label))
        ax.set_title(lab, fontsize=8.5, color=INK)
        if logx:
            ax.set_xscale("log")
        if m in ("span_mean", "width_native_mean"):
            ax.set_yscale("log")
        ax.set_xlabel("training tokens (M)", fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(6, len(l)), fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title + " — layer-mean geometry on 8 fixed validation blocks", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig_save(fig, path)


RAMP = {"LERP": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"],
        "MinMax-Weight": ["#f5b08f", "#f08a5e", "#eb6834", "#c4501f", "#9c3c12", "#742c0c"],
        "QRM-Weight": ["#8fdcbd", "#52c79b", "#1baf7a", "#138a5f", "#0d6746", "#084a32"]}


def fig_k_scaling_geometry(gm, R, path, title, logx):
    fams = [f for f in RAMP if (R.label == f).sum() >= 2]
    if not fams:
        return
    ms = [("span_mean", "row span"), ("width_native_mean", "native width h"),
          ("exact_mass_top_bucket_mean", "exact mass in top bucket"), ("tv_native_exact_mean", "TV(native, exact)")]
    fig, axes = plt.subplots(len(fams), 4, figsize=(13, 2.9 * len(fams)), squeeze=False)
    ex = R[R.label == "Exact"].run_name
    for fi, fam in enumerate(fams):
        rr = R[R.label == fam].copy()
        rr["Kn"] = rr.K.astype(int)
        rr = rr.sort_values("Kn")
        cols = RAMP[fam][-len(rr):] if len(rr) <= len(RAMP[fam]) else RAMP[fam]
        for mi, (m, lab) in enumerate(ms):
            ax = axes[fi][mi]
            if m == "span_mean" and len(ex):
                s = gm[gm.run_name == ex.iloc[0]].sort_values("tokens_m")
                ax.plot(s.tokens_m, s[m], color=COLOR["Exact"], linestyle=DASH["Exact"], lw=1.5, label="Exact")
            for col, r in zip(cols, rr.itertuples()):
                s = gm[gm.run_name == r.run_name].sort_values("tokens_m")
                if len(s):
                    ax.plot(s.tokens_m, s[m], color=col, marker="o", ms=2.5, lw=1.6, label=r.cond)
            ax.set_title(f"{fam}: {lab}", fontsize=8.5, color=INK)
            if logx:
                ax.set_xscale("log")
            if m in ("span_mean", "width_native_mean"):
                ax.set_yscale("log")
            ax.set_xlabel("training tokens (M)", fontsize=8)
        axes[fi][0].legend(fontsize=7)
    fig.suptitle(title + " — within-family K scaling of geometry (darker = larger K)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig_save(fig, path)


def fig_layer_profile(geo, runs_sub, ckpt_index_by_run, path, title, conds):
    ms = [("span_mean", "row span"), ("width_native_mean", "native width h"),
          ("exact_mass_top_bucket_mean", "exact mass in top bucket"), ("tv_native_exact_mean", "TV(native, exact)")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    for ax, (m, lab) in zip(axes, ms):
        for r in runs_sub[runs_sub.cond.isin(conds)].itertuples():
            s = geo[(geo.run_name == r.run_name) & (geo.ckpt_index == ckpt_index_by_run[r.run_name])]
            if len(s) and not s[m].isna().all():
                ax.plot(s.layer, s[m], marker="o", ms=3, label=r.cond, **style_for(r.label))
        ax.set_title(lab, fontsize=9, color=INK)
        ax.set_xlabel("layer")
        if m in ("span_mean", "width_native_mean"):
            ax.set_yscale("log")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(title, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig_save(fig, path)


def fig_dense_logs(logs, att, runs_sub, path, title, logx, conds):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for r in runs_sub[runs_sub.cond.isin(conds)].itertuples():
        a = logs[logs.run_name == r.run_name]
        b = att[att.run_name == r.run_name]
        st = style_for(r.label)
        axes[0].plot(a.tokens / 1e6, a.val_loss, lw=1.4, label=r.cond, **st)
        axes[1].plot(b.tokens / 1e6, b.row_span_mean, lw=1.4, **st)
        axes[2].plot(a.tokens / 1e6, a.grad_norm, lw=1.0, **st)
    axes[0].set_title("logged val loss (100 random batches)", fontsize=9)
    axes[1].set_title("logged row span mean", fontsize=9)
    axes[1].set_yscale("log")
    axes[2].set_title("logged grad norm (single step)", fontsize=9)
    axes[2].set_yscale("log")
    ex = logs[logs.cond == "Exact"]
    if len(ex):
        axes[0].set_ylim(ex.val_loss.iloc[-1] - 0.2, ex.val_loss.iloc[len(ex) // 4] + 1.0)
    for ax in axes:
        ax.set_xlabel("training tokens (M)")
        if logx:
            ax.set_xscale("log")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(title + " — dense training logs (descriptive only)", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig_save(fig, path)


# ----------------------------------------------------------------------------- per-suite pipeline
KEY_TRAJ = {
    "B_factorial_124m_250m": [
        "MinMax-Weight K4 - LERP K4", "MinMax-Prob K4 - MinMax-Weight K4", "QRM-Weight K4 - MinMax-Weight K4",
        "QRM-Prob K4 - QRM-Weight K4", "QRM-Prob K4 - MinMax-Prob K4", "LERP K4 - Exact",
        "MinMax-Prob K16 - MinMax-Weight K16", "QRM-Weight K16 - MinMax-Weight K16", "QRM-Prob K16 - QRM-Weight K16"],
    "A_longhorizon_124m_2p5b": [
        "MinMax-Weight K4 - LERP K4", "MinMax-Prob K4 - MinMax-Weight K4", "QRM-Weight K4 - MinMax-Weight K4",
        "QRM-Prob K4 - QRM-Weight K4", "QRM-Prob K4 - MinMax-Prob K4", "LERP K4 - Exact",
        "MinMax-Weight K16 - LERP K16", "QRM-Weight K16 - MinMax-Weight K16", "MinMax-Weight K32 - LERP K32"],
    "C_scale_1b_100m": [
        "MinMax-Weight K4 - LERP K4", "MinMax-Prob K4 - MinMax-Weight K4", "QRM-Weight K4 - MinMax-Weight K4",
        "QRM-Prob K4 - QRM-Weight K4", "QRM-Prob K4 - MinMax-Prob K4", "LERP K4 - Exact",
        "MinMax-Weight K16 - LERP K16", "QRM-Weight K16 - MinMax-Weight K16", "MinMax-Weight K32 - LERP K32"],
}
ANCHORS = {  # previously obtained endpoint values (columns anchor_nll and minus_anchor of the endpoint table)
    "formal_124m_s1337_minmax_nearest_k16_probste_250m": 3.90539, "formal_124m_s1337_minmax_nearest_k16_weightste_250m": 4.01271,
    "formal_124m_s1337_qrm_k16_tau6_probste_250m": 3.88625, "formal_124m_s1337_qrm_k16_tau6_weightste_250m": 3.88506,
    "formal_124m_s1337_qrm_k4_tau6_probste_250m": 3.89510, "formal_124m_s1337_exact_matched250m": 3.88828,
    "formal_124m_s1337_qrm_k4_tau6_matched250m": 3.92710, "formal_124m_s1337_lerp_k4_matched250m": 3.94006,
    "formal_124m_s1337_nearest_k4_probste_250m": 3.96180, "formal_124m_s1337_nearest_k4_weightste_250m": 4.27744,
    "formal_124m_s1337_exact_2p5b": 3.192805103, "formal_124m_s1337_lerp_k16_fullgrad_2p5b": 3.193076352,
    "formal_124m_s1337_lerp_k32_fullgrad_2p5b": 3.194670522, "formal_124m_s1337_lerp_k4_fullgrad_2p5b": 3.197485245,
    "formal_124m_s1337_nearest_k16_fullgrad_2p5b": 3.639744250, "formal_124m_s1337_nearest_k32_fullgrad_2p5b": 3.352759427,
    "formal_124m_s1337_nearest_k4_fullgrad_2p5b": 4.083653730, "formal_1b_s1337_exact_100m": 4.090479332,
    "formal_1b_s1337_lerp_k16_fullgrad_100m": 4.132135012, "formal_1b_s1337_lerp_k32_fullgrad_100m": 4.110082612,
    "formal_1b_s1337_lerp_k4_fullgrad_100m": 4.186703054, "formal_1b_s1337_nearest_k16_fullgrad_100m": 4.165172336,
    "formal_1b_s1337_nearest_k32_fullgrad_100m": 4.135654841, "formal_1b_s1337_nearest_k4_fullgrad_100m": 4.310271285,
}


def onset_table(ct, fam_names, margin=0.01):
    """First checkpoint where the contrast CI lower bound exceeds margin, and first after which it stays so."""
    out = []
    for n in fam_names:
        s = ct[ct.contrast == n].sort_values("tokens_m")
        if not len(s):
            continue
        sep = (s.ci_lo > margin).values
        first = s.tokens_m.values[sep.argmax()] if sep.any() else np.nan
        persist = np.nan
        for i in range(len(sep)):
            if sep[i:].all():
                persist = s.tokens_m.values[i]
                break
        out.append(dict(contrast=n, margin_nats=margin, first_separated_tokens_m=first,
                        persistent_from_tokens_m=persist, n_checkpoints=len(s),
                        final_delta=s.delta.values[-1], max_delta=s.delta.max(),
                        tokens_at_max_delta_m=s.tokens_m.values[s.delta.values.argmax()]))
    return pd.DataFrame(out)


def fmt_table(df, cols, floatfmt="{:+.4f}"):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df[cols].iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else (f"{v:.4f}" if c in ("mean_nll", "anchor_nll", "tokens_m")
                                                      else floatfmt.format(v)))
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def analyze_suite(suite, runs, ev, nll, geo, rng):
    meta = SUITES[suite]
    out = ROOT / suite
    for sub in ("endpoint_fullval", "trajectory", "geometry", "training_logs", "figures", "contrasts"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    R = runs[runs.suite == suite].copy()
    E = ev[ev.suite == suite].copy()
    logx = False  # checkpoints are evenly spaced; linear axes avoid colliding log minor labels
    status = dict(runs=len(R), checkpoints_expected=int(R.n_checkpoints.astype(int).sum()), checkpoints_evaluated=len(E),
                  endpoints_evaluated=int(E.is_final.sum()) if len(E) else 0)
    R.to_csv(out / "MANIFEST_USED.tsv", sep="\t", index=False)
    if not len(E):
        return status

    conds = set(R.cond)
    specs = contrast_specs(suite, conds)
    fin_idx = {r.run_name: int(r.n_checkpoints) - 1 for r in R.itertuples()}

    # ---------------- endpoints
    endE = E[E.is_final == 1].copy()
    endpoint_idx = int(R.n_checkpoints.astype(int).iloc[0]) - 1
    econ = run_contrasts(endE, nll, specs, endpoint_idx, rng)
    econ.to_csv(out / "endpoint_fullval/endpoint_contrasts.csv", index=False)
    ex_row = endE[endE.label == "Exact"]
    rows = []
    for r in endE.itertuples():
        d = dict(cond=r.cond, label=r.label, calibration=r.calibration, backward=r.backward, K=r.K,
                 run_name=r.run_name, tokens_m=r.tokens_m, mean_nll=r.mean_nll, ppl=float(np.exp(r.mean_nll)))
        m = econ[econ.contrast == f"{r.cond} - Exact"]
        if len(m):
            d.update(delta_vs_exact=m.delta.iloc[0], ci_lo=m.ci_lo.iloc[0], ci_hi=m.ci_hi.iloc[0])
        elif r.label == "Exact":
            d.update(delta_vs_exact=0.0, ci_lo=0.0, ci_hi=0.0)
        L = econ[econ.contrast == f"{r.cond} - LERP K{r.K}"]
        d["delta_vs_lerp_same_K"] = L.delta.iloc[0] if len(L) else np.nan
        d["anchor_nll"] = ANCHORS.get(r.run_name, np.nan)
        d["minus_anchor"] = r.mean_nll - d["anchor_nll"]
        d["checkpoint"] = r.checkpoint
        d["eval_code"] = r.eval_code
        rows.append(d)
    endpoint = pd.DataFrame(rows)
    if len(endpoint):
        endpoint["order"] = endpoint.label.map({l: i for i, l in enumerate(LABEL_ORDER)})
        endpoint["Kn"] = endpoint.K.replace("", "0").astype(int)
        endpoint = endpoint.sort_values(["Kn", "order"]).drop(columns=["order", "Kn"])
        endpoint.to_csv(out / "endpoint_fullval/endpoint_table.csv", index=False)
        if "delta_vs_exact" in endpoint and len(ex_row):
            fig_endpoint_bars(endpoint.dropna(subset=["delta_vs_exact"]), out / "figures/endpoint_delta_vs_exact.png",
                              f"{meta['title']}: final checkpoint, Δ NLL vs Exact (95% block-bootstrap CI)")

    # ---------------- trajectories
    traj = E.sort_values(["run_name", "ckpt_index"])[["cond", "label", "K", "run_name", "ckpt_index", "step",
                                                       "tokens_m", "mean_nll", "is_final"]]
    traj.to_csv(out / "trajectory/checkpoint_nll.csv", index=False)
    ct_all = []
    for ci in sorted(E.ckpt_index.unique()):
        sub = E[E.ckpt_index == ci]
        c = run_contrasts(sub, nll, specs, ci, rng)
        if len(c):
            c.insert(0, "ckpt_index", ci)
            c.insert(1, "tokens_m", sub.tokens_m.median())
            ct_all.append(c)
    ct = pd.concat(ct_all, ignore_index=True) if ct_all else pd.DataFrame()
    if len(ct):
        # keep only contrasts whose terms were all evaluated at that checkpoint
        ct.to_csv(out / "trajectory/contrast_trajectories.csv", index=False)
        fam = [n for n in ct.contrast.unique()
               if n.endswith(" - Exact") or " - LERP K" in n or "QRM-Weight K" in n and "MinMax-Weight" in n
               or "Prob K" in n]
        onset_table(ct, fam).to_csv(out / "trajectory/separation_onset.csv", index=False)
        fig_contrast_traj(ct, KEY_TRAJ[suite], out / "figures/contrast_trajectories.png",
                          f"{meta['title']}: matched contrasts across saved checkpoints", logx)
    fig_nll_traj(E, out / "figures/nll_trajectories.png", f"{meta['title']}: full-validation NLL per saved checkpoint", logx)

    # ---------------- geometry
    G = geo[geo.run_name.isin(R.run_name)] if len(geo) else pd.DataFrame()
    if len(G):
        G.to_csv(out / "geometry/geometry_layers_all_checkpoints.csv", index=False)
        num = [c for c in G.columns if c not in ("run_name", "ckpt_index", "tokens_m", "layer", "n_buckets", "rows")]
        gm = G.groupby(["run_name", "ckpt_index", "tokens_m"], as_index=False)[num].mean()
        gmax = G.groupby(["run_name", "ckpt_index"], as_index=False)[["span_mean", "span_p99", "q_norm_mean",
                                                                      "tv_native_exact_mean"]].max()
        gm = gm.merge(gmax, on=["run_name", "ckpt_index"], suffixes=("", "_layermax"))
        gm = gm.merge(R[["run_name", "cond", "label", "K"]], on="run_name")
        gm.to_csv(out / "geometry/geometry_layer_mean_by_checkpoint.csv", index=False)
        for K in sorted({k for k in R.K if k}, key=int):
            cK = [c for c in R.cond if c == "Exact" or c.endswith(f" K{K}")]
            fig_geometry_traj(gm, R, out / f"figures/geometry_trajectories_K{K}.png",
                              meta["title"] + f" (K{K} + Exact)", logx, cK)
        fig_k_scaling_geometry(gm, R, out / "figures/geometry_k_scaling_by_family.png", meta["title"], logx)
        k4 = [c for c in R.cond if c == "Exact" or c.endswith(" K4")]
        fig_layer_profile(G, R, fin_idx, out / "figures/geometry_layer_profile_final_K4.png",
                          f"{meta['title']}: final-checkpoint layer profile (K4 + Exact)", k4)
        # association: geometry vs NLL gap to Exact at the same checkpoint
        gap = traj.merge(traj[traj.label == "Exact"][["ckpt_index", "mean_nll"]].rename(columns={"mean_nll": "exact_nll"}),
                         on="ckpt_index")
        gap["gap_vs_exact"] = gap.mean_nll - gap.exact_nll
        J = gm.merge(gap[["run_name", "ckpt_index", "gap_vs_exact"]], on=["run_name", "ckpt_index"])
        J.to_csv(out / "geometry/geometry_with_nll_gap.csv", index=False)
        assoc = []
        mets = ["span_mean", "width_native_mean", "z_p001", "q_norm_mean", "exact_mass_top_bucket_mean",
                "eff_buckets_mass", "tv_native_exact_mean"]
        for m in mets:
            for scope, D in [("pooled_hard_nearest_runs", J[J.label.str.contains("MinMax|QRM")]),
                             ("pooled_all_quant_runs", J[J.label != "Exact"])]:
                D = D.dropna(subset=[m, "gap_vs_exact"])
                if len(D) >= 6:
                    rho, p = spearmanr(D[m], D.gap_vs_exact)
                    assoc.append(dict(metric=m, scope=scope, n_points=len(D), spearman_rho=rho, p_value=p))
            for r in R[R.label != "Exact"].itertuples():
                D = J[J.run_name == r.run_name].dropna(subset=[m, "gap_vs_exact"])
                if len(D) >= 5:
                    rho, p = spearmanr(D[m], D.gap_vs_exact)
                    assoc.append(dict(metric=m, scope=f"within_run:{r.cond}", n_points=len(D), spearman_rho=rho,
                                      p_value=p))
        pd.DataFrame(assoc).to_csv(out / "geometry/association_geometry_vs_gap.csv", index=False)

    # ---------------- training logs
    logs, att, health = load_training_logs(suite, R)
    if len(logs):
        logs.to_csv(out / "training_logs/log_dedup.csv", index=False)
        att.to_csv(out / "training_logs/attention_stats_dedup.csv", index=False)
        health.to_csv(out / "training_logs/training_health_dedup.csv", index=False)
        warm = {"A_longhorizon_124m_2p5b": 25e6, "B_factorial_124m_250m": 25e6, "C_scale_1b_100m": 10e6}[suite]
        horizon = {"A_longhorizon_124m_2p5b": 2.6e9, "B_factorial_124m_250m": 2.6e8, "C_scale_1b_100m": 1.1e8}[suite]
        stages = ({"post-warmup to 250M": (warm, 250e6), "250M-1B": (250e6, 1e9), "1B-2.5B": (1e9, horizon)}
                  if suite == "A_longhorizon_124m_2p5b" else {"post-warmup": (warm, horizon)})
        health_summary(health, warm, stages).to_csv(out / "training_logs/health_summary.csv", index=False)
        for K in sorted({k for k in R.K if k}, key=int):
            cK = [c for c in R.cond if c == "Exact" or c.endswith(f" K{K}")]
            fig_dense_logs(logs, att, R, out / f"figures/dense_training_logs_K{K}.png",
                           meta["title"] + f" (K{K} + Exact)", logx, cK)
    status.update(contrast_specs=len(specs))
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default=",".join(SUITES))
    args = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)
    runs = load_runs()
    ev, nll, geo = load_eval(runs)
    st = {}
    for s in args.suites.split(","):
        st[s] = analyze_suite(s, runs, ev, nll, geo, rng)
        print(s, st[s], flush=True)
    (ROOT / "analysis_status.json").write_text(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
