#!/usr/bin/env python3
"""D_robustness_124m_100m_5seed — analysis of the 124M @ 100M WikiText-103 five-seed matrix.

26 conditions (Exact + 5 operator families x K in {4,8,16,32,64}) x 5 training seeds {7,42,253,1337,2026}
= 130 runs, each with 5 trajectory checkpoints (20/40/60/80/100M tokens), evaluated natively on the full
WikiText-103 validation split (243 non-overlapping 1024-token blocks, batch 1, each run's own frozen
training code).

What this suite can do that the single-seed primary suites cannot: separate **training-seed variability**
from the operator effect. Every contrast here is therefore paired *by seed* — condition seed s minus Exact
seed s — and its uncertainty is the spread of those five paired differences, not block resampling. Block
resampling is reported alongside, but only to show how small evaluation noise is next to seed noise.

Statistical note that governs everything below. With five paired seeds, an exact two-sided sign-flip test
has only 2^5 = 32 sign assignments and therefore a minimum attainable p-value of 2/32 = 0.0625. Thresholded
permutation significance is therefore not used as the primary evidence for this analysis. Instead we report
paired effect sizes, 95% t intervals across the five paired seed differences (df = 4), and the magnitude of
each effect relative to the across-seed variability of Exact. The floor is per comparison and BH inherits it:
when several comparisons sit at 0.0625 the step-up returns q = 0.0625 for all of them, so the q column cannot
cross 0.05 either. The t interval assumes approximate normality of the seed effect at n = 5.

No downstream-task evaluation is run for this suite: at 100M tokens the models are deliberately undertrained.

Outputs under D_robustness_124m_100m_5seed/analysis_v1/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, spearmanr, t as tdist  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

SUITE = Path(__file__).resolve().parents[1] / "D_robustness_124m_100m_5seed"
OUT = SUITE / "analysis_v1"
SEEDS = [7, 42, 253, 1337, 2026]
KS = [4, 8, 16, 32, 64]
FAMILIES = ["LERP", "MinMax-Weight", "MinMax-Prob", "QRM-Weight", "QRM-Prob"]
TOKENS_M = [20, 40, 60, 80, 100]
N_BOOT, BOOT_BLOCK, RNG_SEED = 4000, 16, 20260920

COLOR = {"Exact": "#52514e", "LERP": "#2a78d6", "MinMax-Weight": "#eb6834", "QRM-Weight": "#1baf7a",
         "MinMax-Prob": "#eda100", "QRM-Prob": "#e87ba4"}
DASH = {"Exact": (0, (1, 1.5)), "LERP": "-", "MinMax-Weight": "-", "QRM-Weight": "-",
        "MinMax-Prob": (0, (5, 2)), "QRM-Prob": (0, (5, 2))}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0})


# ----------------------------------------------------------------------------- loading
def load():
    runs = pd.read_csv(SUITE / "manifests/five_seed_wikitext_130.tsv", sep="\t", dtype=str)
    runs["K"] = runs["K"].fillna("")
    runs["seed"] = runs["seed"].astype(int)
    runs["run_name"] = [f"{c.replace(' ', '_')}_s{s}" for c, s in zip(runs.cond, runs.seed)]
    rows, blocks, geo = [], {}, []
    for r in runs.itertuples():
        base = SUITE / "eval_native_v1" / r.run_name
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            s = d / "summary.json"
            if not s.exists() or ".tmp_" in d.name:
                continue
            j = json.loads(s.read_text())
            key = (r.run_name, int(j["ckpt_index"]))
            blocks[key] = np.load(d / "per_block_nll.npy")
            rows.append(dict(run_name=r.run_name, cond=r.cond, label=r.label, K=r.K, seed=r.seed,
                             calibration=r.calibration, reconstruction=r.reconstruction, backward=r.backward,
                             source_origin=r.source_origin, ckpt_index=int(j["ckpt_index"]),
                             is_final=int(j["is_final"]), step=int(j["step"]),
                             tokens_m=round(j["tokens_done"] / 1e6), mean_nll=float(j["mean_nll"]),
                             n_blocks=int(j["n_blocks"]), eval_code=j["eval_code"]))
            g = d / "geometry_layers.csv"
            if g.exists():
                gd = pd.read_csv(g)
                gd.insert(0, "run_name", r.run_name)
                gd.insert(1, "cond", r.cond)
                gd.insert(2, "label", r.label)
                gd.insert(3, "K", r.K)
                gd.insert(4, "seed", r.seed)
                gd.insert(5, "ckpt_index", int(j["ckpt_index"]))
                geo.append(gd)
    ev = pd.DataFrame(rows)
    return runs, ev, blocks, (pd.concat(geo, ignore_index=True) if geo else pd.DataFrame())


# ----------------------------------------------------------------------------- statistics
def seed_paired(d):
    """Five paired seed-level differences -> point estimate, t interval (df=4), exact sign-flip p.

    The test enumerates all 2^n sign assignments exhaustively and returns the raw proportion
    #{|T*| >= |T_obs|} / 2^n, with no add-one correction (none is needed: the null is enumerated, not
    sampled). The statistic is linear in the paired differences and therefore antisymmetric under sign
    flips, so |T| is maximised at the observed assignment when the differences share a sign; the minimum
    attainable two-sided p is then exactly 2/2^n = 0.0625 at n = 5.
    """
    d = np.asarray([x for x in d if np.isfinite(x)], dtype=float)
    n = len(d)
    if n == 0:
        return dict(n_seeds=0)
    m, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    tcrit = tdist.ppf(0.975, n - 1) if n > 1 else float("nan")
    signs = np.array(list(itertools.product([1, -1], repeat=n)), dtype=float)
    null = (signs * d[None, :]).mean(axis=1)
    p = float((np.abs(null) >= abs(m) - 1e-15).mean())   # two-sided; minimum 2/2^n (0.0625 at n=5)
    return dict(n_seeds=n, delta=m, sd_seed=sd, se_seed=se,
                ci_lo=m - tcrit * se if n > 1 else float("nan"),
                ci_hi=m + tcrit * se if n > 1 else float("nan"),
                p_signflip=p, ci_excludes_zero=bool(n > 1 and (m - tcrit * se) * (m + tcrit * se) > 0),
                per_seed=list(np.round(d, 6)))


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q, prev = np.empty(m), 1.0
    for rank in range(m - 1, -1, -1):
        i = o[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        q[i] = prev
    return q


def block_boot_ci(d, rng):
    """Circular moving-block bootstrap over validation blocks (evaluation noise only)."""
    n = len(d)
    nb = int(np.ceil(n / BOOT_BLOCK))
    st = rng.integers(0, n, size=(N_BOOT, nb))
    idx = (st[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]).reshape(N_BOOT, -1)[:, :n] % n
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# ----------------------------------------------------------------------------- analyses
def endpoint_table(ev, blocks):
    e = ev[ev.is_final == 1]
    piv = e.pivot_table(index="cond", columns="seed", values="mean_nll")
    ex = piv.loc["Exact"] if "Exact" in piv.index else None
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    meta = e.drop_duplicates("cond").set_index("cond")
    for c in piv.index:
        v = piv.loc[c]
        r = dict(cond=c, label=meta.loc[c, "label"], K=meta.loc[c, "K"],
                 calibration=meta.loc[c, "calibration"], reconstruction=meta.loc[c, "reconstruction"],
                 backward=meta.loc[c, "backward"], source_origin=meta.loc[c, "source_origin"],
                 n_seeds=int(v.notna().sum()), mean_nll=float(v.mean()),
                 sd_nll_seed=float(v.std(ddof=1)) if v.notna().sum() > 1 else np.nan,
                 min_nll=float(v.min()), max_nll=float(v.max()))
        # evaluation noise for this condition, averaged over its seeds
        ses = []
        for s in SEEDS:
            key = (f"{c.replace(' ', '_')}_s{s}", 4)
            if key in blocks:
                _, lo, hi = block_boot_ci(blocks[key], rng)
                ses.append((hi - lo) / (2 * 1.96))
        r["eval_se_block_boot"] = float(np.mean(ses)) if ses else np.nan
        if ex is not None and c != "Exact":
            r.update(seed_paired([v[s] - ex[s] for s in piv.columns if pd.notna(v.get(s)) and pd.notna(ex.get(s))]))
        rows.append(r)
    d = pd.DataFrame(rows)
    sub = d[d.cond != "Exact"].dropna(subset=["p_signflip"])
    if len(sub):
        d.loc[sub.index, "q_bh"] = bh(sub.p_signflip.values)
    return d, piv


def seed_noise(ev, blocks, endp):
    """Training-seed variability against evaluation noise, for the quantity actually being compared.

    Two evaluation-noise numbers are reported and they answer different questions:
      eval_se_marginal   block-bootstrap SE of one run's own mean NLL. Large, because validation blocks differ
                         wildly in difficulty. It is NOT the noise of a contrast.
      eval_se_paired     block-bootstrap SE of the per-block difference (condition minus Exact, same seed,
                         identical blocks). Block difficulty cancels, so this is the evaluation noise that
                         actually limits a paired contrast, and it is the number to compare with the seed SD.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    for c, g in endp.set_index("cond").iterrows():
        paired = []
        if c != "Exact":
            for s in SEEDS:
                kc, ke = (f"{c.replace(' ', '_')}_s{s}", 4), (f"Exact_s{s}", 4)
                if kc in blocks and ke in blocks and len(blocks[kc]) == len(blocks[ke]):
                    _, lo, hi = block_boot_ci(blocks[kc] - blocks[ke], rng)
                    paired.append((hi - lo) / (2 * 1.96))
        ep = float(np.mean(paired)) if paired else np.nan
        rows.append(dict(cond=c, sd_across_seeds=g.sd_nll_seed, sd_seed_of_delta=g.get("sd_seed", np.nan),
                         eval_se_marginal=g.eval_se_block_boot, eval_se_paired=ep,
                         ratio_seedDelta_over_evalPaired=(g.get("sd_seed", np.nan) / ep)
                         if ep and np.isfinite(ep) else np.nan))
    d = pd.DataFrame(rows)
    ex = d[d.cond == "Exact"].iloc[0] if (d.cond == "Exact").any() else None
    if ex is not None:
        e2 = endp.set_index("cond")
        d["abs_delta_vs_exact"] = [abs(e2.loc[c, "delta"]) if c in e2.index and pd.notna(e2.loc[c].get("delta"))
                                   else np.nan for c in d.cond]
        d["delta_over_exact_seed_sd"] = d.abs_delta_vs_exact / ex.sd_across_seeds
        d["delta_over_eval_se_paired"] = d.abs_delta_vs_exact / d.eval_se_paired
    return d


def factorial(ev):
    """Complete 2x2xK factorial: calibration (MinMax vs QRM) x backward (Weight-STE vs Prob-STE).

    LERP is not part of this design (different reconstruction) and is reported separately.
    Each effect is formed per (K, seed) and then treated as five paired seed-level observations.
    """
    e = ev[(ev.is_final == 1) & ev.label.isin(["MinMax-Weight", "MinMax-Prob", "QRM-Weight", "QRM-Prob"])]
    piv = e.pivot_table(index=["K", "seed"], columns="label", values="mean_nll")
    rows = []
    for K in KS:
        sub = piv.loc[str(K)] if str(K) in piv.index.get_level_values(0) else None
        if sub is None or sub.isna().any().any():
            continue
        cal = ((sub["QRM-Weight"] + sub["QRM-Prob"]) - (sub["MinMax-Weight"] + sub["MinMax-Prob"])) / 2
        bwd = ((sub["MinMax-Prob"] + sub["QRM-Prob"]) - (sub["MinMax-Weight"] + sub["QRM-Weight"])) / 2
        inter = (sub["QRM-Prob"] - sub["QRM-Weight"]) - (sub["MinMax-Prob"] - sub["MinMax-Weight"])
        for name, series in [("calibration QRM minus MinMax", cal),
                             ("backward Prob-STE minus Weight-STE", bwd),
                             ("interaction", inter)]:
            rows.append(dict(effect=name, K=K, **seed_paired(series.values)))
    # pooled over K
    for name, fn in [("calibration QRM minus MinMax", lambda s: ((s["QRM-Weight"] + s["QRM-Prob"])
                                                                 - (s["MinMax-Weight"] + s["MinMax-Prob"])) / 2),
                     ("backward Prob-STE minus Weight-STE", lambda s: ((s["MinMax-Prob"] + s["QRM-Prob"])
                                                                       - (s["MinMax-Weight"] + s["QRM-Weight"])) / 2),
                     ("interaction", lambda s: (s["QRM-Prob"] - s["QRM-Weight"])
                                                - (s["MinMax-Prob"] - s["MinMax-Weight"]))]:
        per_seed = []
        for s in SEEDS:
            vals = [fn(piv.loc[(str(K), s)]) for K in KS
                    if (str(K), s) in piv.index and not piv.loc[(str(K), s)].isna().any()]
            if vals:
                per_seed.append(np.mean(vals))
        if per_seed:
            rows.append(dict(effect=name, K="pooled", **seed_paired(per_seed)))
    d = pd.DataFrame(rows)
    if len(d):
        d["q_bh"] = bh(d.p_signflip.values)
    return d


def k_trend(ev):
    """Slope of delta-vs-Exact against log2(K), computed per seed then pooled across seeds."""
    e = ev[ev.is_final == 1]
    piv = e.pivot_table(index="seed", columns="cond", values="mean_nll")
    rows = []
    x = np.log2(np.array(KS, float))
    for fam in FAMILIES:
        per_seed = []
        for s in piv.index:
            y = [piv.loc[s].get(f"{fam} K{k}", np.nan) - piv.loc[s].get("Exact", np.nan) for k in KS]
            y = np.array(y, float)
            if np.isfinite(y).sum() == len(KS):
                per_seed.append(np.polyfit(x, y, 1)[0])
        if per_seed:
            rows.append(dict(family=fam, **seed_paired(per_seed)))
    d = pd.DataFrame(rows)
    if len(d):
        d["q_bh"] = bh(d.p_signflip.values)
    return d


def trajectory(ev):
    e = ev.copy()
    piv = e.pivot_table(index=["cond", "ckpt_index"], columns="seed", values="mean_nll")
    rows = []
    for (c, ci), v in piv.iterrows():
        exk = piv.loc[("Exact", ci)] if ("Exact", ci) in piv.index else None
        r = dict(cond=c, ckpt_index=ci, tokens_m=TOKENS_M[ci], n_seeds=int(v.notna().sum()),
                 mean_nll=float(v.mean()), sd_nll_seed=float(v.std(ddof=1)) if v.notna().sum() > 1 else np.nan)
        if exk is not None and c != "Exact":
            dd = [v[s] - exk[s] for s in v.index if pd.notna(v.get(s)) and pd.notna(exk.get(s))]
            r.update({k: val for k, val in seed_paired(dd).items() if k != "per_seed"})
        rows.append(r)
    d = pd.DataFrame(rows)
    onset = []
    for c, g in d[d.cond != "Exact"].groupby("cond"):
        g = g.sort_values("ckpt_index")
        worse = g[(g.ci_excludes_zero == True) & (g.delta > 0)]      # noqa: E712  degradation only
        better = g[(g.ci_excludes_zero == True) & (g.delta < 0)]
        onset.append(dict(cond=c,
                          first_degrading_tokens_m=int(worse.tokens_m.iloc[0]) if len(worse) else None,
                          n_checkpoints_better_than_exact=int(len(better)),
                          checkpoints_better=";".join(f"{int(x)}M" for x in better.tokens_m),
                          delta_100m=float(g[g.ckpt_index == 4].delta.iloc[0]) if (g.ckpt_index == 4).any()
                          else np.nan))
    return d, pd.DataFrame(onset)


def rank_stability(ev):
    """Do the 25 non-Exact conditions rank the same way in every seed?"""
    e = ev[ev.is_final == 1]
    piv = e.pivot_table(index="cond", columns="seed", values="mean_nll")
    ex = piv.loc["Exact"] if "Exact" in piv.index else None
    piv = piv.drop(index="Exact", errors="ignore")
    if ex is not None:
        piv = piv.sub(ex, axis=1)
    piv = piv.dropna()
    if piv.shape[1] < 2 or len(piv) < 3:
        return pd.DataFrame(), pd.DataFrame(), np.nan
    ranks = piv.rank(axis=0)
    n, k = len(piv), piv.shape[1]
    S = ((ranks.sum(axis=1) - ranks.sum(axis=1).mean()) ** 2).sum()
    W = 12 * S / (k ** 2 * (n ** 3 - n))
    pair = [dict(seed_a=a, seed_b=b, spearman=float(spearmanr(piv[a], piv[b]).statistic),
                 kendall=float(kendalltau(piv[a], piv[b]).statistic))
            for a, b in itertools.combinations(piv.columns, 2)]
    return piv.reset_index(), pd.DataFrame(pair), float(W)


def geometry_summary(geo, endp):
    if geo.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = geo[geo.ckpt_index == 4]
    cols = [c for c in ["span_mean", "width_native_mean", "tv_native_exact_mean", "exact_entropy_mean",
                        "keys_top_bucket_mean", "exact_mass_top_bucket_mean", "eff_buckets_mass",
                        "relevant_keys_tau6_mean", "frac_entries_below_tau6"] if c in g.columns]
    per_run = g.groupby(["cond", "label", "K", "seed"], as_index=False)[cols].mean()
    by_cond = per_run.groupby(["cond", "label", "K"], as_index=False)[cols].agg(["mean", "std"])
    by_cond.columns = ["_".join(c).rstrip("_") for c in by_cond.columns.to_flat_index()]
    by_cond = by_cond.reset_index() if by_cond.index.name else by_cond
    m = endp.set_index("cond")
    assoc = []
    seedmean = per_run.groupby("cond", as_index=False)[cols].mean()
    seedmean["delta"] = [m.loc[c, "delta"] if c in m.index else np.nan for c in seedmean.cond]
    s = seedmean.dropna(subset=["delta"])
    for c in cols:
        if s[c].notna().sum() > 3:
            r = spearmanr(s[c], s.delta)
            assoc.append(dict(metric=c, spearman_vs_delta=float(r.statistic), p=float(r.pvalue),
                              n_conditions=int(s[c].notna().sum())))
    return by_cond, pd.DataFrame(assoc)


# ----------------------------------------------------------------------------- figures
def figures(endp, traj, piv, seedn):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for fam in FAMILIES:
        sub = endp[endp.label == fam].copy()
        if sub.empty:
            continue
        sub["Ki"] = sub.K.astype(int)
        sub = sub.sort_values("Ki")
        ax.errorbar(sub.Ki, sub.delta, yerr=[sub.delta - sub.ci_lo, sub.ci_hi - sub.delta],
                    marker="o", ms=4, capsize=3, color=COLOR[fam], linestyle=DASH[fam], label=fam)
    ex_sd = seedn[seedn.cond == "Exact"].sd_across_seeds.iloc[0]
    ax.axhspan(-2 * ex_sd, 2 * ex_sd, color=GRID, zorder=0)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.set_xlabel("K (interval budget)")
    ax.set_ylabel("Δ validation NLL vs Exact (nats), paired by seed")
    ax.set_title("124M @ 100M WikiText-103, five seeds: endpoint gap vs Exact\n"
                 "bars = 95% t interval over 5 seeds; band = ±2 × Exact's own seed SD", color=INK)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures/endpoint_delta_vs_K.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for c, g in traj[traj.cond != "Exact"].groupby("cond"):
        lab = g.cond.iloc[0].rsplit(" K", 1)[0]
        g = g.sort_values("tokens_m")
        ax.plot(g.tokens_m, g.delta, color=COLOR.get(lab, INK2), lw=1.2, alpha=0.75,
                linestyle=DASH.get(lab, "-"))
    for fam in FAMILIES:
        ax.plot([], [], color=COLOR[fam], linestyle=DASH[fam], label=fam)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("Δ NLL vs Exact (nats)")
    ax.set_title("Gap to Exact along training, seed-averaged (25 conditions)", color=INK)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures/trajectory_delta.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    d = seedn.dropna(subset=["sd_across_seeds"]).copy()
    d["lab"] = [c.rsplit(" K", 1)[0] for c in d.cond]
    d = d.sort_values("sd_across_seeds")
    ax.barh(range(len(d)), d.sd_across_seeds, color=[COLOR.get(l, INK2) for l in d.lab])
    ax.set_yticks(range(len(d)), d.cond, fontsize=6)
    ax.set_xlabel("SD of endpoint validation NLL across the 5 training seeds (nats)")
    ax.set_title("Training-seed variability per condition", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "figures/seed_sd.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    runs, ev, blocks, geo = load()
    n_final = int((ev.is_final == 1).sum()) if len(ev) else 0
    print(f"loaded {len(ev)} evaluated checkpoints of 646; endpoints {n_final}/130", flush=True)
    if n_final < 130 and not a.allow_partial:
        raise SystemExit("incomplete endpoint coverage; rerun when evaluation finishes or pass --allow-partial")

    endp, piv = endpoint_table(ev, blocks)
    seedn = seed_noise(ev, blocks, endp)
    fact = factorial(ev)
    ktr = k_trend(ev)
    traj, onset = trajectory(ev)
    spread, pairs, W = rank_stability(ev)
    gcond, gassoc = geometry_summary(geo, endp)

    endp.to_csv(OUT / "endpoint_by_condition.csv", index=False)
    piv.to_csv(OUT / "endpoint_nll_by_seed.csv")
    seedn.to_csv(OUT / "seed_noise.csv", index=False)
    fact.to_csv(OUT / "factorial_calibration_backward.csv", index=False)
    ktr.to_csv(OUT / "k_trend.csv", index=False)
    traj.to_csv(OUT / "trajectory.csv", index=False)
    onset.to_csv(OUT / "separation_onset.csv", index=False)
    if len(spread):
        spread.to_csv(OUT / "delta_by_seed.csv", index=False)
        pairs.to_csv(OUT / "rank_stability_pairs.csv", index=False)
    if len(gcond):
        gcond.to_csv(OUT / "geometry_by_condition.csv", index=False)
        gassoc.to_csv(OUT / "geometry_association.csv", index=False)
    json.dump(dict(n_checkpoints_evaluated=len(ev), n_endpoints=n_final, kendall_W_rank_stability=W,
                   seeds=SEEDS, n_boot=N_BOOT, boot_block=BOOT_BLOCK,
                   min_attainable_p_two_sided=2 / 2 ** len(SEEDS),
                   p_note=("an exact two-sided sign-flip test on 5 paired seeds has 2^5 = 32 sign "
                           "assignments and therefore a minimum attainable p of 2/32 = 0.0625; thresholded "
                           "permutation significance is not used as primary evidence. Primary evidence: "
                           "paired effect sizes, 95% t intervals across the 5 paired seed differences "
                           "(df = 4), and effect magnitude relative to the across-seed variability of "
                           "Exact. The floor is per comparison and BH inherits it."),
                   t_interval_df=len(SEEDS) - 1),
              open(OUT / "analysis_meta.json", "w"), indent=2)
    figures(endp, traj, piv, seedn)

    pd.set_option("display.width", 250)
    print("\n=== endpoint, paired by seed ===")
    print(endp[endp.cond != "Exact"][["cond", "n_seeds", "mean_nll", "sd_nll_seed", "delta", "sd_seed",
                                      "ci_lo", "ci_hi", "p_signflip", "q_bh"]].round(4).to_string(index=False))
    print("\n=== factorial ===")
    print(fact[["effect", "K", "delta", "ci_lo", "ci_hi", "p_signflip", "q_bh"]].round(4).to_string(index=False))
    print("\n=== K trend (slope of Δ vs log2 K) ===")
    print(ktr[["family", "delta", "ci_lo", "ci_hi", "p_signflip", "q_bh"]].round(4).to_string(index=False))
    print(f"\nKendall W (rank stability of 25 conditions across 5 seeds) = {W:.4f}")
    print(f"minimum attainable two-sided sign-flip p with {len(SEEDS)} seeds = {2/2**len(SEEDS):.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
