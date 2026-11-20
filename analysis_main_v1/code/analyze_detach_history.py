#!/usr/bin/env python3
"""Development record: detached vs full calibration gradient (WikiText-103, 124M, 100M tokens, seed 1337)
plus the early 5M-token confirmatory seeds 1338/1339. Writes E_detach_history/{tables,figures}."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


TRAIN = Path(KI_REPO)
ROOT = TRAIN / "analysis_main_v1"
OUT = ROOT / "E_detach_history"
DET = OUT / "eval_native_v1"
FIVE = ROOT / "D_robustness_124m_100m_5seed/eval_native_v1"
LEG = TRAIN / "training_code/legacy_exact_fp32/runs"
PB = TRAIN / "training_code/phaseB_base/runs"
rng = np.random.default_rng(20260917)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
C_DET, C_FULL, C_PROJ, C_EX = "#eb6834", "#2a78d6", "#e87ba4", "#52514e"

CONDS = [("Nearest K2", "q2_nearest_100m", None, None),
         ("Nearest K4", "q4_nearest_100m", "MinMax-Weight_K4_s1337", "q4_nearest_fullgrad_100m"),
         ("Nearest K8", "q8_nearest_100m", "MinMax-Weight_K8_s1337", "q8_nearest_fullgrad_100m"),
         ("Nearest K16", "q16_nearest_100m", "MinMax-Weight_K16_s1337", "q16_nearest_fullgrad_100m"),
         ("LERP K4", "q4_lerp_100m", "LERP_K4_s1337", "q4_lerp_fullgrad_100m"),
         ("LERP K32", "q32_lerp_100m", "LERP_K32_s1337", "q32_lerp_fullgrad_100m")]


def endpoint(dirpath, step=763):
    ds = sorted(p for p in Path(dirpath).iterdir() if (p / "summary.json").exists())
    fin = [p for p in ds if json.loads((p / "summary.json").read_text())["step"] == step]
    if len(fin) != 1:
        raise RuntimeError(f"{dirpath}: expected one checkpoint at step {step}, found {len(fin)}")
    p = fin[0]
    return np.load(p / "per_block_nll.npy"), pd.read_csv(p / "geometry_layers.csv").mean(numeric_only=True)


def boot(d, n=4000, L=16):
    m = len(d)
    starts = rng.integers(0, m, size=(n, int(np.ceil(m / L))))
    idx = ((starts[:, :, None] + np.arange(L)) % m).reshape(n, -1)[:, :m]
    b = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def logs(run_dir, span_file):
    lg = pd.read_csv(run_dir / "log.csv")
    sp = pd.read_csv(run_dir / span_file) if (run_dir / span_file).exists() else None
    return lg, sp


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    ex_nll, ex_geo = endpoint(FIVE / "Exact_s1337")
    rows, geo_rows = [], []
    for name, det, full, _ in CONDS:
        d_nll, d_geo = endpoint(DET / det)
        r = dict(condition=name, detach_nll=d_nll.mean())
        r["detach_minus_exact"], r["detach_lo"], r["detach_hi"] = boot(d_nll - ex_nll)
        if full:
            f_nll, f_geo = endpoint(FIVE / full)
            r["full_nll"] = f_nll.mean()
            r["full_minus_exact"], r["full_lo"], r["full_hi"] = boot(f_nll - ex_nll)
            r["detach_minus_full"], r["dmf_lo"], r["dmf_hi"] = boot(d_nll - f_nll)
            geo_rows.append(dict(condition=name, variant="full", **f_geo[["span_mean", "width_native_mean", "q_norm_mean",
                                                                           "exact_mass_top_bucket_mean", "eff_buckets_mass",
                                                                           "tv_native_exact_mean", "exact_entropy_mean"]].to_dict()))
        geo_rows.append(dict(condition=name, variant="detach", **d_geo[["span_mean", "width_native_mean", "q_norm_mean",
                                                                         "exact_mass_top_bucket_mean", "eff_buckets_mass",
                                                                         "tv_native_exact_mean", "exact_entropy_mean"]].to_dict()))
        rows.append(r)
    p_nll, p_geo = endpoint(DET / "q32_lerp_detachproj_100m")
    f32, _ = endpoint(FIVE / "LERP_K32_s1337")
    r = dict(condition="LERP K32 detach+projection (P1)", detach_nll=p_nll.mean())
    r["detach_minus_exact"], r["detach_lo"], r["detach_hi"] = boot(p_nll - ex_nll)
    r["full_nll"] = f32.mean()
    r["full_minus_exact"], r["full_lo"], r["full_hi"] = boot(f32 - ex_nll)
    r["detach_minus_full"], r["dmf_lo"], r["dmf_hi"] = boot(p_nll - f32)
    rows.append(r)
    geo_rows.append(dict(condition="LERP K32 detach+projection (P1)", variant="detach+proj",
                         **p_geo[["span_mean", "width_native_mean", "q_norm_mean", "exact_mass_top_bucket_mean",
                                  "eff_buckets_mass", "tv_native_exact_mean", "exact_entropy_mean"]].to_dict()))
    geo_rows.append(dict(condition="Exact", variant="exact", **ex_geo[["span_mean", "q_norm_mean", "exact_entropy_mean"]].to_dict()))
    end = pd.DataFrame(rows)
    end["exact_nll"] = ex_nll.mean()
    end.to_csv(OUT / "endpoint_detach_vs_full_s1337.csv", index=False)
    pd.DataFrame(geo_rows).to_csv(OUT / "geometry_detach_vs_full_s1337.csv", index=False)

    # standardized trajectories where saved checkpoints exist (LERP K4/K32 detach, detach+proj, full)
    traj = []
    for name, det, full in [("LERP K4", "q4_lerp_100m", "LERP_K4_s1337"), ("LERP K32", "q32_lerp_100m", "LERP_K32_s1337"),
                            ("LERP K32 detach+proj", "q32_lerp_detachproj_100m", None)]:
        for variant, base in (("detach" if "proj" not in name else "detach+proj", DET / det), ("full", FIVE / full if full else None)):
            if base is None or not base.exists():
                continue
            for p in sorted(base.iterdir()):
                if (p / "summary.json").exists():
                    s = json.loads((p / "summary.json").read_text())
                    g = pd.read_csv(p / "geometry_layers.csv").mean(numeric_only=True)
                    traj.append(dict(condition=name.replace(" detach+proj", ""), variant=variant, tokens_m=s["tokens_done"] / 1e6,
                                     nll=s["mean_nll"], span_mean=g["span_mean"], tv=g["tv_native_exact_mean"],
                                     q_norm=g["q_norm_mean"]))
    pd.DataFrame(traj).sort_values(["condition", "variant", "tokens_m"]).to_csv(OUT / "standardized_trajectories_lerp.csv", index=False)

    # dense logged trajectories (matched validation sampling, 100 batches every 5M tokens)
    ex_log = pd.read_csv(LEG / "exact_fp32_softmax_100m/log.csv")
    dense = []
    fig, axes = plt.subplots(2, 6, figsize=(17, 5.6), sharex=True)
    for j, (name, det, _, full_dir) in enumerate(CONDS):
        dl = pd.read_csv(LEG / det / "log.csv")
        dsp = pd.read_csv(LEG / det / ("attention_stats.csv" if "lerp" in det else "occupancy.csv"))
        ax, ax2 = axes[0][j], axes[1][j]
        mex = dl.merge(ex_log[["tokens", "val_loss"]], on="tokens", suffixes=("", "_exact"))
        ax.plot(mex.tokens / 1e6, mex.val_loss - mex.val_loss_exact, color=C_DET, ls=(0, (5, 2)), lw=2, label="detach")
        ax2.plot(dsp.tokens / 1e6, dsp.row_span_mean, color=C_DET, ls=(0, (5, 2)), lw=2, label="detach")
        for _, rr in mex.iterrows():
            dense.append(dict(condition=name, variant="detach", tokens_m=rr.tokens / 1e6, logged_val=rr.val_loss,
                              logged_gap=rr.val_loss - rr.val_loss_exact))
        if full_dir:
            fl = pd.read_csv(PB / full_dir / "log.csv")
            fsp = pd.read_csv(PB / full_dir / "attention_stats.csv")
            mf = fl.merge(ex_log[["tokens", "val_loss"]], on="tokens", suffixes=("", "_exact"))
            ax.plot(mf.tokens / 1e6, mf.val_loss - mf.val_loss_exact, color=C_FULL, lw=2, label="full")
            ax2.plot(fsp.tokens / 1e6, fsp.row_span_mean, color=C_FULL, lw=2, label="full")
            for _, rr in mf.iterrows():
                dense.append(dict(condition=name, variant="full", tokens_m=rr.tokens / 1e6, logged_val=rr.val_loss,
                                  logged_gap=rr.val_loss - rr.val_loss_exact))
        if name == "LERP K32":
            pl = pd.read_csv(PB / "q32_lerp_detachproj_100m/log.csv")
            psp = pd.read_csv(PB / "q32_lerp_detachproj_100m/attention_stats.csv")
            mp = pl.merge(ex_log[["tokens", "val_loss"]], on="tokens", suffixes=("", "_exact"))
            ax.plot(mp.tokens / 1e6, mp.val_loss - mp.val_loss_exact, color=C_PROJ, lw=1.6, ls=(0, (1, 1.2)), label="detach+proj")
            ax2.plot(psp.tokens / 1e6, psp.row_span_mean, color=C_PROJ, lw=1.6, ls=(0, (1, 1.2)), label="detach+proj")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(name, fontsize=10, color=INK)
        ax2.set_yscale("log")
        ax2.set_xlabel("training tokens (M)")
        if j == 0:
            ax.set_ylabel("logged val loss − Exact (nats)")
            ax2.set_ylabel("logged row span mean")
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("124M @ 100M, WikiText-103, seed 1337: detached vs full calibration gradient (dense training logs)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "figures/detach_vs_full_dense_logs.png", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    pd.DataFrame(dense).to_csv(OUT / "dense_logged_gap_detach_vs_full.csv", index=False)

    # early confirmatory 5M seeds
    ms = []
    for s in (1338, 1339):
        ex5, _ = endpoint(DET / f"exact_fp32_softmax_5m_s{s}", step=39)
        exl = pd.read_csv(LEG / f"exact_fp32_softmax_5m_s{s}/log.csv")
        for K in (2, 4, 8):
            q, _ = endpoint(DET / f"q{K}_nearest_5m_s{s}", step=39)
            ql = pd.read_csv(LEG / f"q{K}_nearest_5m_s{s}/log.csv")
            d, lo, hi = boot(q - ex5)
            ms.append(dict(seed=s, condition=f"Nearest K{K} (detach)", tokens_m=5.0, exact_nll=ex5.mean(), nll=q.mean(),
                           delta=d, ci_lo=lo, ci_hi=hi, logged_gap=ql.val_loss.iloc[-1] - exl.val_loss.iloc[-1]))
    pd.DataFrame(ms).to_csv(OUT / "early_5m_seeds_detach.csv", index=False)
    print(end.round(4).to_string(index=False))
    print(pd.DataFrame(ms).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
