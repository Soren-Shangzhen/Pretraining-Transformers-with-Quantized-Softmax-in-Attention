#!/usr/bin/env python3
"""LaTeX tables of the paper (main text and Appendices A-H), generated from the result files in analysis_main_v1/.

Outputs: *.tex (booktabs), written next to this script. Each file is \\input by the manuscript.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import os
import re
from scipy import stats

TRAIN = Path(os.environ.get("KI_REPO", Path(__file__).resolve().parents[2]))   # the root of this repository
AM = TRAIN / "analysis_main_v1"
A, B, C = AM / "A_longhorizon_124m_2p5b", AM / "B_factorial_124m_250m", AM / "C_scale_1b_100m"
D = AM / "D_robustness_124m_100m_5seed/analysis_v1"
E = AM / "E_detach_history"
DS = AM / "downstream_v1/results"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# Display names (paper naming; LaTeX en-dashes). Data keys keep their historical labels QRM-*/MinMax-Weight/-Prob.
NAME = {"QRM-Weight": "FWM--Weight", "QRM-Prob": "FWM--Prob", "MinMax-Weight": "MinMax--Weight",
        "MinMax-Prob": "MinMax--Prob", "FW-Weight": "FWM--Weight", "FW-Prob": "FWM--Prob"}


def tex_name(cond):
    if cond == "Exact":
        return "softmax"
    for k, v in NAME.items():
        cond = cond.replace(k, v)
    return cond.replace("QRM", "FWM")


def f3(x, plus=True):
    if pd.isna(x):
        return "---"
    return f"{x:+.3f}" if plus else f"{x:.3f}"


def ci(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]"


def ci_c(lo, hi):
    """compact interval for the wide merged cross-suite table (signs kept only when negative)"""
    return f"[{lo:.3f}, {hi:.3f}]"


def write(name, body):
    (OUT / name).write_text(body)
    print("wrote", name)


def endpoints(s):
    return pd.read_csv(s / "endpoint_fullval/endpoint_table.csv")


def contrasts(s):
    return pd.read_csv(s / "endpoint_fullval/endpoint_contrasts.csv").set_index("contrast")


# ---------------------------------------------------------------- five-seed paired contrasts by seed
def five_seed_contrast(a, b):
    """paired per seed: cond a minus cond b (both as delta vs softmax of the same seed)."""
    d = pd.read_csv(D / "delta_by_seed.csv").set_index("cond")
    x = d.loc[a].values.astype(float) - d.loc[b].values.astype(float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return m, m - t * se, m + t * se


def five_seed_vs_exact(a):
    ep = pd.read_csv(D / "endpoint_by_condition.csv").set_index("cond")
    r = ep.loc[a]
    return r.delta, r.ci_lo, r.ci_hi


# ---------------------------------------------------------------- Table: cross-suite matched contrasts
def tab_cross_suite():
    rows = [
        ("MinMax-Weight K4 $-$ softmax", "MinMax-Weight K4 - Exact", ("MinMax-Weight K4", None)),
        ("MinMax-Prob K4 $-$ softmax", "MinMax-Prob K4 - Exact", ("MinMax-Prob K4", None)),
        ("FW-Weight K4 $-$ softmax", "QRM-Weight K4 - Exact", ("QRM-Weight K4", None)),
        ("FW-Prob K4 $-$ softmax", "QRM-Prob K4 - Exact", ("QRM-Prob K4", None)),
        ("LERP K4 $-$ softmax", "LERP K4 - Exact", ("LERP K4", None)),
        ("\\midrule Prob $-$ Weight, MinMax, K4", "MinMax-Prob K4 - MinMax-Weight K4", ("MinMax-Prob K4", "MinMax-Weight K4")),
        ("Prob $-$ Weight, FWM, K4", "QRM-Prob K4 - QRM-Weight K4", ("QRM-Prob K4", "QRM-Weight K4")),
        ("FWM $-$ MinMax, Weight-STE, K4", "QRM-Weight K4 - MinMax-Weight K4", ("QRM-Weight K4", "MinMax-Weight K4")),
        ("FWM $-$ MinMax, Prob-STE, K4", "QRM-Prob K4 - MinMax-Prob K4", ("QRM-Prob K4", "MinMax-Prob K4")),
        ("Interaction, K4", "K4 2x2 interaction: (QRM Prob-Weight) - (MinMax Prob-Weight)", ("interaction", "4")),
        ("Main effect, calibration (FWM $-$ MinMax), K4", "K4 2x2 main effect: calibration (QRM - MinMax", ("calibration QRM minus MinMax", "4")),
        ("Main effect, surrogate (Prob $-$ Weight), K4", "K4 2x2 main effect: backward (Prob - Weight", ("backward Prob-STE minus Weight-STE", "4")),
        ("\\midrule FWM $-$ MinMax, Weight-STE, K16", "QRM-Weight K16 - MinMax-Weight K16", ("QRM-Weight K16", "MinMax-Weight K16")),
        ("Main effect, calibration, K16", "K16 2x2 main effect: calibration (QRM - MinMax", ("calibration QRM minus MinMax", "16")),
        ("Main effect, surrogate, K16", "K16 2x2 main effect: backward (Prob - Weight", ("backward Prob-STE minus Weight-STE", "16")),
        ("Interaction, K16", "K16 2x2 interaction: (QRM Prob-Weight) - (MinMax Prob-Weight)", ("interaction", "16")),
        ("$K\\times$calibration (K16 $-$ K4)", "K x calibration interaction", None),
        ("$K\\times$surrogate (K16 $-$ K4)", "K x backward interaction", None),
        ("MinMax-Weight K4 $-$ LERP K4", "MinMax-Weight K4 - LERP K4", ("MinMax-Weight K4", "LERP K4")),
        ("FW-Weight K4 $-$ LERP K4", "QRM-Weight K4 - LERP K4", ("QRM-Weight K4", "LERP K4")),
        ("FW-Prob K4 $-$ LERP K4", "QRM-Prob K4 - LERP K4", ("QRM-Prob K4", "LERP K4")),
    ]
    cb, ca, cc = contrasts(B), contrasts(A), contrasts(C)
    fa = pd.read_csv(D / "factorial_calibration_backward.csv")
    lines = []
    for label, key, five in rows:
        cells = []
        for cdf in (cb, ca, cc):
            hit = [i for i in cdf.index if i == key or i.startswith(key + ",") or i.startswith(key + ":")]
            if hit:
                r = cdf.loc[hit[0]]
                cells.append(f"{r.delta:+.3f} {ci_c(r.ci_lo, r.ci_hi)}")
            else:
                cells.append("---")
        if five is None:
            cells.append("---")
        else:
            if five[0] in ("interaction", "calibration QRM minus MinMax", "backward Prob-STE minus Weight-STE"):
                r = fa[(fa.effect == five[0]) & (fa.K == five[1])].iloc[0]
                m, lo, hi = r.delta, r.ci_lo, r.ci_hi
            elif five[1] is None:
                m, lo, hi = five_seed_vs_exact(five[0])
            else:
                m, lo, hi = five_seed_contrast(*five)
            cells.append(f"{m:+.3f} {ci_c(lo, hi)}")
        lines.append(f"{tex_name(label)} & " + " & ".join(cells) + " \\\\")
    hw = max((r.ci_hi - r.ci_lo) / 2 for cdf in (cb, ca, cc) for k, r in cdf.iterrows()
             if k.startswith("K4 2x2 interaction") or k in ("MinMax-Prob K4 - MinMax-Weight K4", "QRM-Prob K4 - QRM-Weight K4", "QRM-Weight K4 - MinMax-Weight K4", "QRM-Prob K4 - MinMax-Prob K4"))
    print(f"[tab_cross_suite] max single-seed CI half-width over the five K4 contrasts of Table 2: {hw:.5f}")
    body = r"""\setlength{\tabcolsep}{3pt}\begin{tabular}{lllll}
\toprule
Contrast (nats) & B: 124M@250M & A: 124M@2.5B & C: 1B@100M & D: 124M@100M, 5 seeds \\
& FineWebEdu [block CI] & FineWebEdu [block CI] & WikiText [block CI] & WikiText, mean [$t$-CI] \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_cross_suite.tex", body)
    short = [re.sub(r" \[[^\]]*\]", "", ln, count=3) for ln in lines[5:10]]   # main-text Table 2 keeps point estimates + five-seed CI
    short = [ln.replace("Prob $-$ Weight, MinMax, K4", "Prob $-$ Weight under MinMax, K4").replace("Prob $-$ Weight, FWM, K4", "Prob $-$ Weight under FWM, K4")
               .replace("FWM $-$ MinMax, Weight-STE, K4", "FWM $-$ MinMax under Weight-STE, K4").replace("FWM $-$ MinMax, Prob-STE, K4", "FWM $-$ MinMax under Prob-STE, K4")
               for ln in short]
    short[0] = short[0].replace("\\midrule ", "")
    body = r"""\begin{tabular}{lrrrl}
\toprule
Contrast, $K{=}4$ (nats) & B: 124M@250M & A: 124M@2.5B & C: 1B@100M & D: 5 seeds, mean [95\% $t$-CI] \\
\midrule
""" + "\n".join(short) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_cross_suite_short.tex", body)


# ---------------------------------------------------------------- Table: 2x2 main effects and interactions (single seed suites)
def tab_factorial_single():
    cb, ca, cc = contrasts(B), contrasts(A), contrasts(C)
    keys = [("calibration (FWM $-$ MinMax)", "K4 2x2 main effect: calibration (QRM - MinMax"),
            ("surrogate (Prob $-$ Weight)", "K4 2x2 main effect: backward (Prob - Weight"),
            ("interaction", "K4 2x2 interaction: (QRM Prob-Weight) - (MinMax Prob-Weight)")]
    lines = []
    for lab, k in keys:
        cells = []
        for cdf in (cb, ca, cc):
            hit = [i for i in cdf.index if i.startswith(k)]
            r = cdf.loc[hit[0]]
            cells.append(f"{r.delta:+.3f} {ci(r.ci_lo, r.ci_hi)}")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    # K16 at 250M
    lines.append("\\midrule")
    for lab, k in [("calibration, K16 (250M only)", "K16 2x2 main effect: calibration (QRM - MinMax"),
                   ("surrogate, K16 (250M only)", "K16 2x2 main effect: backward (Prob - Weight"),
                   ("interaction, K16 (250M only)", "K16 2x2 interaction: (QRM Prob-Weight) - (MinMax Prob-Weight)"),
                   ("K $\\times$ backward", "K x backward interaction"),
                   ("K $\\times$ calibration", "K x calibration interaction")]:
        hit = [i for i in cb.index if i.startswith(k)]
        r = cb.loc[hit[0]]
        lines.append(f"{lab} & {r.delta:+.3f} {ci(r.ci_lo, r.ci_hi)} & --- & --- \\\\")
    body = r"""\begin{tabular}{llll}
\toprule
K4 effect (nats) & 124M@250M & 124M@2.5B & 1B@100M \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_factorial_single.tex", body)


# ---------------------------------------------------------------- Table: five-seed factorial
def tab_five_seed_factorial():
    fa = pd.read_csv(D / "factorial_calibration_backward.csv")
    kt = pd.read_csv(D / "k_trend.csv")
    effs = [("calibration QRM minus MinMax", "calibration: FWM $-$ MinMax"), ("backward Prob-STE minus Weight-STE", "surrogate: Prob $-$ Weight"), ("interaction", "interaction")]
    lines = []
    for K in ["4", "8", "16", "32", "64", "pooled"]:
        cells = []
        for eff, _ in effs:
            r = fa[(fa.effect == eff) & (fa.K == K)].iloc[0]
            cells.append(f"{r.delta:+.4f} ({r.sd_seed:.4f})")
        lines.append((f"$K{{=}}{K}$" if K != "pooled" else "pooled") + " & " + " & ".join(cells) + " \\\\")
    body = r"""\begin{tabular}{lrrr}
\toprule
$K$ & calibration: FWM $-$ MinMax & surrogate: Prob $-$ Weight & interaction \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_five_seed_factorial.tex", body)
    lines = [f"{tex_name(r.family)} & {r.delta:+.4f} & {ci(r.ci_lo, r.ci_hi)} & {r.sd_seed:.4f} \\\\" for r in kt.itertuples()]
    body = r"""\begin{tabular}{lrlr}
\toprule
Family & slope & 95\% $t$-CI & seed SD \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_five_seed_ktrend.tex", body)


# ---------------------------------------------------------------- Table: five-seed endpoints (appendix)
def tab_five_seed_endpoints():
    ep = pd.read_csv(D / "endpoint_by_condition.csv")
    sn = pd.read_csv(D / "seed_noise.csv").set_index("cond")
    ex = ep[ep.cond == "Exact"].iloc[0]
    order = ["LERP", "MinMax-Weight", "MinMax-Prob", "QRM-Weight", "QRM-Prob"]
    lines = []
    for fam in order:
        d = ep[ep.label == fam].sort_values("K")
        for r in d.itertuples():
            s = sn.loc[r.cond]
            lines.append(f"{tex_name(r.cond)} & {r.mean_nll:.4f} & {r.sd_nll_seed:.4f} & {r.delta:+.4f} & {ci(r.ci_lo, r.ci_hi)} & {r.sd_seed:.4f} & {s.eval_se_paired:.4f} \\\\")
    body = r"""\begin{tabular}{lrrrlrr}
\toprule
Condition & mean NLL & seed SD & $\Delta$ vs softmax & 95\% t-CI & seed SD of $\Delta$ & paired eval SE \\
\midrule
softmax & """ + f"{ex.mean_nll:.4f} & {ex.sd_nll_seed:.4f}" + r""" & --- & --- & --- & --- \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_five_seed_endpoints.tex", body)


# ---------------------------------------------------------------- Table: all single-seed endpoints (appendix)
def tab_endpoints_all():
    def block(s, title):
        e = endpoints(s)
        ex = e[e.cond == "Exact"].iloc[0]
        e = e[e.cond != "Exact"].copy()
        e["fam"] = e.label
        e["Kn"] = e.K.fillna(0).astype(int)
        order = {"LERP": 0, "MinMax-Weight": 1, "MinMax-Prob": 2, "QRM-Weight": 3, "QRM-Prob": 4}
        e = e.sort_values(["fam", "Kn"], key=lambda c: c.map(order) if c.name == "fam" else c)
        ls = [f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\",
              f"softmax & {ex.mean_nll:.4f} & --- & --- \\\\"]
        for r in e.itertuples():
            ls.append(f"{tex_name(r.cond)} & {r.mean_nll:.4f} & {r.delta_vs_exact:+.4f} & {ci(r.ci_lo, r.ci_hi)} \\\\")
        return ls
    # compact layout: two tabulars side by side (B and C left, A right); PPL = exp(NLL) stated in the caption
    left = block(B, "124M @ 250M, FineWebEdu-3B, 976 blocks") + ["\\midrule"] + block(C, "1B @ 100M, WikiText-103, 243 blocks")
    right = block(A, "124M @ 2.5B, FineWebEdu-3B, 976 blocks")
    hdr = r"""\begin{tabular}{lrrl}
\toprule
Condition & NLL & $\Delta$ vs softmax & 95\% block CI \\
\midrule
"""
    body = hdr + "\n".join(left) + "\n\\bottomrule\n\\end{tabular}\\hfill\n" + hdr + "\n".join(right) + "\n\\bottomrule\n\\end{tabular}\n"
    write("tab_endpoints_all.tex", body)


# ---------------------------------------------------------------- Table: detach vs full
def tab_detach():
    e = pd.read_csv(E / "endpoint_detach_vs_full_s1337.csv")
    lines = []
    for r in e.itertuples():
        name = tex_name(r.condition.replace("Nearest", "MinMax-Weight")).replace(" detach+projection (P1)", "")   # no run code in the row name
        det = f"{r.detach_minus_exact:+.3f} {ci(r.detach_lo, r.detach_hi)}"
        full = "---" if pd.isna(r.full_minus_exact) else f"{r.full_minus_exact:+.4f} {ci(r.full_lo, r.full_hi)}"
        dmf = "---" if pd.isna(r.detach_minus_full) else f"{r.detach_minus_full:+.3f} {ci(r.dmf_lo, r.dmf_hi)}"
        bw = "detach + zero-sum proj." if "P1" in r.condition else "detach"
        lines.append(f"{name} & {bw} & {det} & {full} & {dmf} \\\\")
    body = r"""\begin{tabular}{lllll}
\toprule
Condition (seed 1337, 100M) & historical backward & historical $-$ softmax & full-gradient $-$ softmax & historical $-$ full \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_detach.tex", body)


# ---------------------------------------------------------------- Table: downstream stage C (appendix)
def tab_downstream():
    pe = pd.read_csv(DS / "layer3/stageC/tables/point_estimates.csv")
    pc = pd.read_csv(DS / "layer3/stageC/tables/paired_comparisons.csv")
    cfgs = [("blimp", "acc_macro", "BLiMP"), ("hellaswag", "acc_norm", "HellaSwag"), ("piqa", "acc_norm", "PIQA"),
            ("arc_easy", "acc_norm", "ARC-E"), ("lambada_openai", "acc", "LAMBADA"), ("sciq_default", "acc", "SciQ"),
            ("sciq_nosupport", "acc", "SciQ-ns")]
    ea = endpoints(A)
    order = ["Exact"] + ea[ea.cond != "Exact"].sort_values("delta_vs_exact", ascending=False).cond.tolist()
    hdr = " & ".join(t for _, _, t in cfgs)
    # merged display: softmax absolute in the header row, then per-item paired differences (pp) from paired_comparisons.csv
    lines = []
    ex = [f"{pe[(pe.cfg == cfg) & (pe.metric == met) & (pe.condition == 'Exact')].value.iloc[0] * 100:.1f}" for cfg, met, _ in cfgs]
    lines.append("softmax (absolute, \\%) & " + " & ".join(ex) + " \\\\ \\midrule")
    for i, cnd in enumerate(order[1:]):
        cells = []
        for cfg, met, _ in cfgs:
            q = pc[(pc.cfg == cfg) & (pc.metric == met) & (pc.method == cnd)].iloc[0]
            assert q.unit == "percentage points"
            mark = "$^{*}$" if q.significant_q05 else ("$^{\\circ}$" if q.ci_excludes_zero else "")
            cells.append(f"{q['diff']:+.1f}{mark}")
        lines.append(f"{tex_name(cnd)} & " + " & ".join(cells) + " \\\\" + (" \\midrule" if i in (4, 8) else ""))
    body = r"""\begin{tabular}{l""" + "r" * len(cfgs) + r"""}
\toprule
Condition & """ + hdr + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_downstream_stageC.tex", body)
    # probe C slopes
    pcf = pd.read_csv(DS / "layer2/diagnostics/probeC_length_trend.csv")
    pcf = pcf.set_index("condition").loc[order[1:]]
    ps_ = pd.read_csv(DS / "layer2/probe_scores.csv")
    ex = ps_[(ps_.condition == "Exact") & (ps_.probe == "C") & (ps_.forward_mode == "native") & (ps_.metric == "target_nll")].set_index("config").score
    lines = [f"softmax, absolute target NLL & {ex['L32']:.3f} & {ex['L64']:.3f} & {ex['L128']:.3f} & --- \\\\ \\midrule"]
    lines += [f"{tex_name(r.Index)} & {r.delta_L32:+.2f} & {r.delta_L64:+.2f} & {r.delta_L128:+.2f} & {r.slope_vs_log2L:+.3f} {ci(r.slope_ci_lo, r.slope_ci_hi)} \\\\" for r in pcf.itertuples()]
    body = r"""\begin{tabular}{lrrrl}
\toprule
Condition & $\Delta$ at L=32 & L=64 & L=128 & slope vs $\log_2 L$ [95\% CI] \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_downstream_probeC.tex", body)


def tab_layer1_nearbaseline_ci():
    """Held-out paired 95% CIs for the near-baseline group (V4 audit): from layer1/ppl_transfer.csv, native forward.
    CI kind as declared per corpus: circular moving-block bootstrap (block 16) for WikiText-103 and PTB, document-cluster
    bootstrap for C4; 4000 resamples; BH q within the 14-comparison family of each corpus x forward mode."""
    l1 = pd.read_csv(DS / "layer1/ppl_transfer.csv")
    near = ["LERP K2", "LERP K4", "LERP K16", "LERP K32", "QRM-Weight K16"]
    corp = [("wikitext103_test", "WikiText-103 test"), ("ptb_test", "PTB test"), ("c4_validation", "C4 validation")]
    lines = []
    n_excl = n_in = 0
    for cnd in near:
        for ck, cl in corp:
            r = l1[(l1.condition == cnd) & (l1.corpus == ck) & (l1.forward_mode == "native")].iloc[0]
            excl = bool(r.ci_lo > 0 or r.ci_hi < 0); inside = bool(r.ci_lo >= -0.01 and r.ci_hi <= 0.01)
            n_excl += excl; n_in += inside
            lines.append(f"{tex_name(cnd)} & {cl} & {r.delta_vs_exact:+.4f} & {ci4(r.ci_lo, r.ci_hi)} & {('<0.001' if r.q_value < 0.0005 else f'{r.q_value:.3f}')} & {'yes' if excl else 'no'} & {'yes' if inside else 'no'} \\\\")
    body = r"""\begin{tabular}{llrlrcc}
\toprule
Condition & Corpus & $\dNLL$ & 95\% CI & BH $q$ & CI excl.\ 0 & CI inside $\pm0.01$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_layer1_nearbaseline_ci.tex", body)
    print(f"near-baseline held-out: {n_excl}/15 CIs exclude zero; {n_in}/15 wholly inside +-0.01")


def ci4(lo, hi):
    return f"[{lo:+.4f}, {hi:+.4f}]"


def tab_substitution():
    l1 = pd.read_csv(DS / "layer1/ppl_transfer.csv")
    ea = endpoints(A)
    order = ea[ea.cond != "Exact"].sort_values("delta_vs_exact", ascending=False).cond.tolist()
    lines = []
    for cnd in order:
        cells = []
        for corpus in ("wikitext103_test", "ptb_test", "c4_validation"):
            n = l1[(l1.condition == cnd) & (l1.corpus == corpus) & (l1.forward_mode == "native")].delta_vs_exact.iloc[0]
            e = l1[(l1.condition == cnd) & (l1.corpus == corpus) & (l1.forward_mode == "exact")].delta_vs_exact.iloc[0]
            cells += [f"{n:+.3f}", f"{e:+.3f}", f"{e - n:+.3f}"]
        lines.append(f"{tex_name(cnd)} & " + " & ".join(cells) + " \\\\")
    body = r"""\begin{tabular}{lrrrrrrrrr}
\toprule
& \multicolumn{3}{c}{WikiText-103 test} & \multicolumn{3}{c}{PTB test} & \multicolumn{3}{c}{C4 validation} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Condition & native & softmax-fwd & diff & native & softmax-fwd & diff & native & softmax-fwd & diff \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_downstream_substitution.tex", body)


def tab_detach_replication():
    """Detach replication on seeds 7 and 42 (LERP K32, phaseB base code root): detach minus matched full-gradient run,
    paired per block, circular block bootstrap (block 16, 4000 resamples); from detach_replication/detach_replication_paired.csv."""
    r = pd.read_csv(AM / "detach_replication/detach_replication_paired.csv")
    rows = [("020m", "20M"), ("040m", "40M"), ("060m", "60M"), ("080m", "80M"), ("100m", "100M"), ("100m_vs_exact", "100M, detach $-$ softmax")]
    lines = []
    for ck, lab in rows:
        cells = []
        for seed in (7, 42):
            x = r[(r.seed == seed) & (r.ckpt == ck)].iloc[0]
            cells.append(f"{x.dmf:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}]")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    body = r"""\setlength{\tabcolsep}{3pt}\begin{tabular}{lll}
\toprule
Ckpt & seed 7 & seed 42 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_detach_replication.tex", body)


def tab_tail_decomp():
    """Resolution / truncation decomposition of MinMax -> FWM (seed 1337, K=4, tau=6): from
    tail_policy/tail_decomp/tail_decomp_contrasts.csv and tail_decomp_ratio.csv."""
    c = pd.read_csv(AM / "tail_policy/tail_decomp/tail_decomp_contrasts.csv")
    r = pd.read_csv(AM / "tail_policy/tail_decomp/tail_decomp_ratio.csv").set_index("surrogate")
    names = {"MinMax - Exact": "MinMax $-$ softmax", "zero - Exact": "FWM $-$ softmax", "resolved - Exact": "resolved $-$ softmax",
             "MinMax - resolved (resolution)": "MinMax $-$ resolved (untruncated fixed-step calibration contrast)", "resolved - zero (truncation)": "resolved $-$ FWM (tail-policy contrast, fixed grid rule)"}
    lines = []
    for sur in ("Weight", "Prob"):
        d = c[(c.surrogate == sur) & (c.section != "3_surrogate_effect")]
        for x in d.itertuples():
            flag = " $^{\\circ}$" if x.below_margin else ""
            lines.append(f"{sur}-STE & {names[x.contrast]} & {x.delta:+.4f} & [{x.ci_lo:+.4f}, {x.ci_hi:+.4f}]{flag} \\\\")
        rr = r.loc[sur]
        lines.append(f"{sur}-STE & path ratio (descriptive) & {rr.ratio:.3f} & [{rr.ci_lo:.3f}, {rr.ci_hi:.3f}] \\\\ \\midrule")
    lines[-1] = lines[-1].replace(" \\midrule", "")
    body = r"""\begin{tabular}{llrl}
\toprule
Surrogate & Contrast (nats) & $\Delta$ & 95\% CI \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_tail_decomp.tex", body)


SUP = AM / "supplement_four_runs_20260923/reports"


def tab_strict_forward():
    c = pd.read_csv(SUP / "supplement_endpoint_contrasts.csv").set_index("contrast")
    def row(label, key, flip=False, note=""):
        r = c.loc[key]; d, lo, hi = (-r.delta, -r.ci_hi, -r.ci_lo) if flip else (r.delta, r.ci_lo, r.ci_hi)
        return f"{label} & {r.evaluator.replace('H5090','RTX 5090').replace('L3090','RTX 3090')} & {d:+.4f} & [{lo:+.4f}, {hi:+.4f}]{note} \\\\"
    grp = lambda t: "\\multicolumn{4}{@{}l}{\\emph{" + t + "}} \\\\"
    lines = [grp("Strict pairs: the four strict-forward runs, strict-hard forward, evaluated on the training GPU"),
             row("MinMax: Prob $-$ Weight (strict pair)", "E2-MM new strict Prob - new strict Weight"),
             row("FWM: Prob $-$ Weight (strict pair)", "E2-FWM new strict Prob - new strict Weight"),
             row("Interaction $(\\mathrm{FWM}_P-\\mathrm{FWM}_W)-(\\mathrm{MM}_P-\\mathrm{MM}_W)$", "E2 interaction (MM P-W) - (FWM P-W), strict pairs [H5090]", flip=True),
             grp("Secondary evaluator: the same strict pairs re-evaluated on the RTX 3090"),
             row("MinMax: Prob $-$ Weight (strict pair)", "E2-MM new strict Prob - new strict Weight [3090 eval]"),
             row("FWM: Prob $-$ Weight (strict pair)", "E2-FWM new strict Prob - new strict Weight [3090 eval]"),
             row("Interaction, as above", "E2 interaction (MM P-W) - (FWM P-W), strict pairs [L3090]", flip=True),
             "\\midrule " + grp("Historical comparisons: new strict runs vs the historical seed-7 runs (frozen weights, different training stack)"),
             row("MinMax: new Prob $-$ historical Prob, strict forward on both", "E2-MM new Prob - old Prob (strict forward)"),
             row("FWM: new Prob $-$ historical Prob, strict forward on both", "E2-FWM new Prob - old Prob (strict forward)"),
             row("MinMax: new strict Weight $-$ historical Weight", "E2-MM new strict Weight - old Weight"),
             row("FWM: new strict Weight $-$ historical Weight", "E2-FWM new strict Weight - old Weight"),
             row("MinMax: historical Prob, native $-$ strict forward (frozen weights)", "E2-MM old Prob native - old Prob strict"),
             row("FWM: historical Prob, native $-$ strict forward (frozen weights)", "E2-FWM old Prob native - old Prob strict"),
             "\\midrule " + grp("Background: historical seed-7 runs only (not a same-stack control)"),
             row("MinMax: historical Prob $-$ historical Weight, seed 7", "HIST E2-MM old Prob - old Weight"),
             row("FWM: historical Prob $-$ historical Weight, seed 7", "HIST E2-FWM old Prob - old Weight")]
    body = r"""\begin{tabular}{llrl}
\toprule
Contrast (nats) & Evaluator & $\Delta$ & 95\% CI \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_strict_forward.tex", body)


def _e1_seed42():
    """Seed-42 single-channel replication: analysis JSON written by supplement code/analyze_e1_seed.py."""
    import json
    return json.load(open(SUP / "supplement_e1_seed42.json"))


def tab_e1_channels():
    """Two-seed single-channel ablation. Seed 7 from supplement_endpoint_contrasts.csv / supplement_trajectories.csv,
    seed 42 from supplement_e1_seed42.json (same 3090 evaluator, 243 blocks, paired per block)."""
    c = pd.read_csv(SUP / "supplement_endpoint_contrasts.csv").set_index("contrast")
    t = pd.read_csv(SUP / "supplement_trajectories.csv"); t = t[(t.tokens_m == 100) & (t.evaluator == "L3090")].set_index("series")
    j = _e1_seed42()
    def fmt(d, lo, hi): return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    def ci7(k): r = c.loc[k]; return fmt(r.delta, r.ci_lo, r.ci_hi)
    def ci42(k): d, lo, hi = j["contrasts_100m"][k]; return fmt(d, lo, hi)
    full7, det7 = t.loc["lerp_full", "mean_nll_243"], t.loc["lerp_detach", "mean_nll_243"]
    rows7 = [("full (control)", t.loc["lerp_full", "mean_nll_243"], "---", "---", t.loc["lerp_full", "geom_span_mean_layermean"], 1.0),
             ("detach (control)", det7, ci7("HIST E1 detach - full"), "---", t.loc["lerp_detach", "geom_span_mean_layermean"], 0.0),
             ("min\\_only", t.loc["e1_min", "mean_nll_243"], ci7("E1 min_only - full"), ci7("E1 min_only - detach"), t.loc["e1_min", "geom_span_mean_layermean"], None),
             ("max\\_only", t.loc["e1_max", "mean_nll_243"], ci7("E1 max_only - full"), ci7("E1 max_only - detach"), t.loc["e1_max", "geom_span_mean_layermean"], None)]
    a, g = j["abs_nll"], j["geometry_100m"]
    full42, det42 = a["full"]["100"], a["detach"]["100"]
    rows42 = [("full (control)", full42, "---", "---", g["full"]["span_mean"], 1.0),
              ("detach (control)", det42, ci42("detach - full"), "---", g["detach"]["span_mean"], 0.0),
              ("min\\_only", a["min_only"]["100"], ci42("min_only - full"), ci42("min_only - detach"), g["min_only"]["span_mean"], None),
              ("max\\_only", a["max_only"]["100"], ci42("max_only - full"), ci42("max_only - detach"), g["max_only"]["span_mean"], None)]
    lines = []
    for seed, rows, full, det in ((7, rows7, full7, det7), (42, rows42, full42, det42)):
        for i, (lab, nll, vf, vd, span, rec) in enumerate(rows):
            rec = (det - nll) / (det - full) if rec is None else rec
            lines.append(f"{seed if i == 0 else ''} & {lab} & {nll:.4f} & {vf} & {vd} & {span:.1f} & {rec:.3f} \\\\")
        if seed == 7: lines.append("\\midrule")
    body = r"""\begin{tabular}{llrllrr}
\toprule
Seed & Mode & NLL @100M & $-$ full & $-$ detach & span & gap recovery \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_e1_channels.tex", body)


R2 = AM / "supplement_four_runs_20260923/round2"


def tab_grad_decomp():
    """Fixed-upstream three-term decomposition at the 20M/40M checkpoints of the full-gradient LERP K32 runs, seeds 7
    and 42; per-layer ratios averaged over the 12 layers (round2/task2*/grad_decomp_local_layers.csv)."""
    lines = []
    for seed, d in ((7, "task2"), (42, "task2_s42")):
        df = pd.read_csv(R2 / d / "grad_decomp_local_layers.csv")
        g = df.groupby("tokens_m").mean(numeric_only=True)
        for i, (tok, r) in enumerate(g.iterrows()):
            assert abs(r["decomp__resid_rel"]) < 1e-12
            lines.append(f"{seed if i == 0 else ''} & {int(tok)}M & {r['decomp__ratio_self']:.3f} & {r['decomp__ratio_M']:.3f} & {r['decomp__ratio_m']:.4f} & "
                         f"{r['detach__aggregate_rel']:.3f} & {r['min_only__aggregate_rel']:.3f} & {r['max_only__aggregate_rel']:.4f} \\\\")
    body = r"""\begin{tabular}{llrrrrrr}
\toprule
 & & \multicolumn{3}{c}{norm ratio to $g_{\mathrm{full}}$} & \multicolumn{3}{c}{relative error of the implemented gradient} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Seed & ckpt & self & $M$ term & $m$ term & detach & min\_only & max\_only \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_grad_decomp.tex", body)


def tab_strict_audit():
    """12 frozen K=4 endpoints (suites A, B, D seed 7; both surrogates, both calibrations): historical native forward
    minus direct hard forward on one RTX 3090 (round2/task1/{weight,prob}_native_vs_strict.json)."""
    import json
    W = json.load(open(R2 / "task1/weight_native_vs_strict.json"))["runs"]
    Pj = json.load(open(R2 / "task1/prob_native_vs_strict.json"))["runs"]
    suite = {"A": "A: 124M @ 2.5B", "B": "B: 124M @ 250M", "D": "D: 124M @ 100M, seed 7"}
    cal = {"minmax": "MinMax", "fwm": "FWM"}
    lines = []
    for sur, runs in (("Weight-STE", W), ("Prob-STE", Pj)):
        for k in sorted(runs, key=lambda x: (x[0], x[2:4])):
            r = runs[k]; lo, hi = r["delta_ci95"]
            assert r["strict_equals_direct_all"]
            lines.append(f"{sur} & {suite[k[0]]} & {cal[r['calibration']]} & {r['n_blocks']} & {sci(r['delta_native_minus_strict'])} & "
                         f"[{sci(lo)}, {sci(hi)}] & {sci(r['max_abs_delta'])} \\\\")
        if sur == "Weight-STE": lines.append("\\midrule")
    mx = max(abs(r["delta_native_minus_strict"]) for runs in (W, Pj) for r in runs.values())
    mxb = max(r["max_abs_delta"] for runs in (W, Pj) for r in runs.values())
    print(f"[tab_strict_audit] max |mean delta| = {mx:.3e}, max per-block |delta| = {mxb:.4f}")
    body = r"""\begin{tabular}{lllrlll}
\toprule
Surrogate & Suite & Calib. & blocks & $\Delta$ mean NLL & 95\% CI & max $|\Delta|$ per block \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_strict_audit.tex", body)


def _gpu_hours(run_dirs):
    """Approximate wall-clock GPU hours from the logged throughput: sum of window_time_s (training_health.csv) when present,
    otherwise sum over log.csv rows of (tokens in window / tok_per_s). One GPU per run."""
    import os
    tot, n = 0.0, 0
    for r in run_dirs:
        r = str(r).replace("<REPO>", str(TRAIN))   # manifest paths are relative to the repository root
        th, lg = os.path.join(r, "training_health.csv"), os.path.join(r, "log.csv")
        if os.path.exists(th):
            h = pd.read_csv(th)
            if "window_time_s" in h and h.window_time_s.sum() > 0:
                tot += h.window_time_s.sum() / 3600; n += 1; continue
        if os.path.exists(lg):
            l = pd.read_csv(lg); l = l[l.tok_per_s > 0]
            if len(l):
                dt = np.diff(np.r_[0, l.tokens.values]); tot += float((dt / l.tok_per_s.values).sum() / 3600); n += 1
    return tot, n


def tab_hardware():
    """Table S4: training and evaluation hardware per suite and group, with approximate GPU hours. The stack labels are
    constants of this function; the suite-D run counts come from five_seed_wikitext_130.tsv; the GPU hours are computed from
    the training logs (the entries 11 and 4.5 are constants; the bin-mean control of Appendix J from the logs of its five
    runs) and, for Appendix I, read from compute_per_experiment.csv."""
    import glob
    PB = TRAIN / "training_code/phaseB_base"
    hA, nA = _gpu_hours(sorted(glob.glob(str(AM / "training_logs/A_longhorizon_124m_2p5b/*"))))
    hB, nB = _gpu_hours(sorted(glob.glob(str(AM / "training_logs/B_factorial_124m_250m/*"))))
    hC, nC = _gpu_hours(sorted(glob.glob(str(AM / "training_logs/C_scale_1b_100m/*"))))
    dm = pd.read_csv(AM / "D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv", sep="\t")
    hD, nD = _gpu_hours(list(dm.run_path))
    n_local = int((dm.source_origin == "original_local").sum()); n_cloud = len(dm) - n_local
    hE1, _ = _gpu_hours([str(PB / f"supplement_runs_20260923/sp_s{s}_lerp_k32_{m}only_100m") for s in (7, 42) for m in ("min", "max")])
    hRep, _ = _gpu_hours([str(PB / f"runs_20260921/rb_s{s}_lerp_k32_detach_100m") for s in (7, 42)])
    hTail, _ = _gpu_hours([str(PB / f"runs_20260921/rb_s1337_tail_resolved_{m}_100m") for m in ("prob", "weight")])
    hBin, nBin = _gpu_hours(sorted(glob.glob(str(PB / "centroid_runs_20261013/*"))))
    assert (nA, nB, nC, nD, nBin) == (15, 10, 12, 130, 5), (nA, nB, nC, nD, nBin)
    print(f"[tab_hardware] GPU hours A {hA:.0f} B {hB:.0f} C {hC:.0f} D {hD:.0f} (local {n_local}, cloud {n_cloud}); E1 {hE1:.0f} rep {hRep:.0f} tail {hTail:.0f} bin-mean {hBin:.0f}")
    # stack IDs are defined once in the caption (S1: RTX 5090, torch 2.13.0+cu132, py3.12; S2: RTX 5090, PyTorch version not recorded; S3: RTX 3090, torch 2.5.1+cu121, py3.11)
    rows = [
        ("A (124M @ 2.5B)", "15", "S1 (13 runs); S2 (2 Prob-STE runs)", "RTX 5090", f"{hA:.0f}"),
        ("B (124M @ 250M)", "10", "S1 (5 runs); S2 (5 runs)", "RTX 5090", f"{hB:.0f}"),
        ("C (1B @ 100M)", "12", "S1 (10 runs); S2 (2 Prob-STE runs)", "RTX 5090", f"{hC:.0f}"),
        ("D (124M @ 100M, 5 seeds)", "130", f"S3 ({n_local} runs); S2 ({n_cloud} runs); family split: Table~\\ref{{tab:dstack}}", "RTX 3090", f"{hD:.0f}"),
        ("\\midrule Historical detach record (\\S\\ref{sec:res-detach})", "16", "S3", "RTX 3090", "---"),
        ("Detach replication, seeds 7/42 (App.~\\ref{app:detach})", "2", "S3", "RTX 3090", f"{hRep:.0f}"),
        ("Single-channel ablation, seeds 7/42 (App.~\\ref{app:detach})", "4", "S3", "RTX 3090", f"{hE1:.0f}"),
        ("Strict-forward retraining (App.~\\ref{app:strict})", "4", "S1", "RTX 5090 and RTX~3090", "11"),   # the tie keeps "RTX 3090" on one line
        ("Tail-policy (App.~\\ref{app:taildecomp})", "2", "S3", "RTX 3090", f"{hTail:.0f}"),
        ("Reduced-precision exponential baselines (App.~\\ref{app:lowprec})", "2", "S3", "RTX 3090", "4.5"),
        ("Bin-mean control (App.~\\ref{app:binmean})", f"{nBin}", "S3", "RTX 3090", f"{hBin:.0f}"),
    ]
    # Appendix I runs, from compute_per_experiment.csv (machines A: 2 GPUs, B: 8 GPUs; stack S1)
    cp = pd.read_csv(AM / "rb-rebuttal_v5/reports/rb-dossier_20261114/csv/compute_per_experiment.csv").set_index("experiment")
    cells = ["Appendix~\\ref{app:rb} (a 2-GPU and an 8-GPU RTX 5090 machine)", f"{int(cp.loc['total'].runs)}",
             "S1 (Python 3.12.11, CUDA 13.2)", "RTX 5090", f"{cp.loc['total'].gpu_hours:.1f}"]
    rows.append(tuple(["\\midrule " + cells[0]] + cells[1:]))
    body = r"""\begin{tabular}{>{\raggedright\arraybackslash}p{5.2cm}rp{5.0cm}>{\raggedright\arraybackslash}p{2.4cm}r}
\toprule
Suite / group & Runs & Training stack (one GPU per run) & Evaluation GPU & GPU-h \\
\midrule
""" + "\n".join(f"{a} & {b} & {c} & {d} & {e} \\\\" for a, b, c, d, e in rows) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_hardware.tex", body)


def tab_dstack():
    """Table S2: suite D by operator family and training GPU, counted in complete training runs (five_seed_wikitext_130.tsv:
    family from the columns backward and calibration, GPU from source_origin, K and backward): original_local -> RTX 3090;
    cloud runs -> RTX 5090 (the GPU model was logged for the K=64 Prob-STE runs and not for the other cloud runs). The runs
    of the appendices are not counted here; Table S4 lists them."""
    d = pd.read_csv(AM / "D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv", sep="\t")
    fam = {("native_exact", "none"): "softmax", ("native_differentiable", "minmax_row_adaptive"): "LERP",
           ("weight_ste", "minmax_row_adaptive"): "MinMax--Weight", ("prob_ste", "minmax_row_adaptive"): "MinMax--Prob",
           ("weight_ste", "relevant_zero"): "FWM--Weight", ("prob_ste", "relevant_zero"): "FWM--Prob"}
    d["family"] = [fam[(b, c)] for b, c in zip(d.backward, d.calibration)]
    def stack(r):
        if r.source_origin == "original_local": return "local"
        if int(r.K) == 64 and r.backward == "prob_ste": return "cloud_logged"
        return "cloud_reported"
    d["stack"] = d.apply(stack, axis=1)
    order = ["softmax", "LERP", "MinMax--Weight", "MinMax--Prob", "FWM--Weight", "FWM--Prob"]
    ct = d.groupby(["family", "stack"]).size().unstack(fill_value=0).reindex(order)
    for c in ("local", "cloud_logged", "cloud_reported"): ct[c] = ct.get(c, 0)
    assert int(ct.values.sum()) == 130
    # one RTX 5090 column; the software stacks S3 and S2 are given in the caption
    lines = [f"{f} & {int(r.local)} & {int(r.cloud_logged + r.cloud_reported)} & {int(r.local + r.cloud_logged + r.cloud_reported)} \\\\" for f, r in ct.iterrows()]
    lines.append("\\midrule all & %d & %d & 130 \\\\" % (ct.local.sum(), ct.cloud_logged.sum() + ct.cloud_reported.sum()))
    print("[tab_dstack]\n" + ct.to_string())
    body = r"""\begin{tabular}{lrrr}
\toprule
Family & RTX 3090 & RTX 5090 & runs \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_dstack.tex", body)


def tab_lowprec():
    """Round-3 reduced-precision exponential baselines: endpoint dNLL vs the same-seed softmax at 100M (seed 7, 243 blocks,
    one RTX 3090 evaluator), with the training stack of every row (round3/reports/r3_lowprec_contrasts.csv)."""
    c = pd.read_csv(AM / "supplement_four_runs_20260923/round3/reports/r3_lowprec_contrasts.csv")
    c["key"] = c.contrast.str.replace(r" - softmax \(seed 7, 100M\)", "", regex=True)
    c = c[c.contrast.str.contains(r"- softmax")].set_index("key")
    L3090 = "RTX 3090; torch 2.5.1+cu121"
    rows = [("BF16-exp (new)", "bf16exp", L3090, "manifest, log"),
            ("FP8-rounded exp, FP32 exp backward (new)", "fp8e4m3exp", L3090, "manifest, log"),
            ("LERP $K{=}32$", "LERP K32", L3090, "manifest"),
            ("LERP $K{=}4$", "LERP K4", L3090, "manifest"),
            ("MinMax--Prob $K{=}4$ (historical)", "MinMax-Prob K4 (hist)", "RTX 5090; PyTorch version not recorded", "pod log (no GPU model)"),
            ("MinMax--Weight $K{=}4$ (historical)", "MinMax-Weight K4 (hist)", L3090, "manifest"),
            ("FWM--Prob $K{=}4$ (historical)", "FWM-Prob K4 (hist)", "RTX 5090; PyTorch version not recorded", "pod log (no GPU model)"),
            ("FWM--Prob $K{=}4$ (strict forward, App.~\\ref{app:strict})", "FWM-Prob K4 strict (new)", "RTX 5090; torch 2.13.0+cu132", "manifest")]
    lines = []
    for lab, key, st, src in rows:
        r = c.loc[key]; assert r.n_blocks == 243 and "3090" in r.evaluator
        lines.append(f"{lab} & {st} & {r.delta:+.4f} & [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] \\\\")
    body = r"""\begin{tabular}{p{5.2cm}p{5.3cm}rl}
\toprule
Operator (seed 7, 124M @ 100M) & Training GPU; software stack & $\Delta$NLL & 95\% eval.\ CI \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_lowprec.tex", body)


def tab_early5m():
    """Historical 5M-token confirmatory Nearest (detach) runs on seeds 1338/1339 against the same-seed 5M softmax
    reference: endpoint NLL, paired per validation block, block bootstrap CI (E_detach_history/early_5m_seeds_detach.csv,
    produced by code/analyze_detach_history.py from per_block_nll.npy of both runs)."""
    e = pd.read_csv(AM / "E_detach_history/early_5m_seeds_detach.csv")
    assert len(e) == 6 and set(e.seed) == {1338, 1339}
    assert set(round(x, 1) for x in e.tokens_m) == {5.0}   # 5M tokens for every row (stated in the caption)
    lines = [f"{int(r.seed)} & {r.condition.replace(' (detach)', '').replace('Nearest ', '')} & {r.nll:.4f} & {r.exact_nll:.4f} & {r.delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] \\\\"
             for r in e.sort_values(["seed", "condition"]).itertuples()]
    body = r"""\setlength{\tabcolsep}{3pt}\begin{tabular}{llrrl}
\toprule
Seed & $K$ & NLL & softmax & $\Delta$ [95\% CI] \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_early5m.tex", body)


def tab_probeAB():
    """Layer-2 probes A (induction at distance) and B (key-value retrieval): every evaluated condition, native forward,
    against the softmax reference (downstream_v1/results/layer2/probe_scores.csv, config 'all'). Paired bootstrap CIs
    over samples (4000 resamples). No condition is omitted."""
    d = pd.read_csv(DS / "layer2/probe_scores.csv")
    d = d[(d.forward_mode == "native") & (d.config == "all")].copy()
    fb = pd.read_csv(DS / "layer2/probe_scores_A_all_familyboot_20260925.csv")   # 2026-09-25 addendum: probe A CIs resampled by family
    for r in fb.itertuples():
        m = (d.probe == "A") & (d.condition == r.condition) & (d.metric == r.metric)
        assert m.sum() == 1 and abs(d.loc[m, "delta"].iloc[0] - r.delta) < 1e-9
        d.loc[m, "ci_lo"] = r.ci_lo_family; d.loc[m, "ci_hi"] = r.ci_hi_family
    ea = endpoints(A); order = ["Exact"] + ea[ea.cond != "Exact"].sort_values("delta_vs_exact", ascending=False).cond.tolist()
    def cell(cond, probe, metric, fmt="{:+.3f}"):
        r = d[(d.condition == cond) & (d.probe == probe) & (d.metric == metric)]
        if r.empty: return "---"
        r = r.iloc[0]
        if cond == "Exact": return fmt.format(r.score).replace("+", "")
        if metric == "top1": return fmt.format(r.delta)
        return f"{fmt.format(r.delta)} [{fmt.format(r.ci_lo)}, {fmt.format(r.ci_hi)}]"
    lines = []
    for c in order:
        lab = "softmax (absolute)" if c == "Exact" else tex_name(c)
        lines.append(f"{lab} & {cell(c,'A','target_nll')} & {cell(c,'A','induction_gain')} & {cell(c,'A','top1','{:+.4f}')} & {cell(c,'B','target_nll')} & {cell(c,'B','top1','{:+.4f}')} \\\\")
        if c == "Exact": lines.append("\\midrule")
    nA = int(d[(d.probe == 'A') & (d.condition == 'Exact') & (d.metric == 'target_nll')].n.iloc[0]); nB = int(d[(d.probe == 'B') & (d.condition == 'Exact') & (d.metric == 'target_nll')].n.iloc[0])
    print(f"[tab_probeAB] nA={nA} nB={nB}")
    body = r"""\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{3}{c}{Probe A: induction ($n{=}2{,}000$ induction samples)} & \multicolumn{2}{c}{Probe B: retrieval ($n{=}2{,}400$)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Condition & target NLL & induction gain & top-1 & target NLL & top-1 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_downstream_probeAB.tex", body)


def sci(x):
    if x == 0:
        return "$0$"
    m, e = f"{x:+.2e}".split("e") if x < 0 else f"{x:.2e}".split("e")
    return f"${m}\\mathrm{{e}}{{{int(e)}}}$"


def tab_rounding():
    r = pd.read_csv(AM / "rounding_audit/rounding_full_split.csv")
    r = r[r.backward == "prob_ste"].copy()
    r["order"] = r.suite.map({"B_factorial_124m_250m": 0, "A_longhorizon_124m_2p5b": 1, "C_scale_1b_100m": 2})
    r = r.sort_values(["order", "label", "K"])
    short = {"A_longhorizon_124m_2p5b": "A", "B_factorial_124m_250m": "B", "C_scale_1b_100m": "C"}
    lines = [f"{short[x.suite]}: {tex_name(x.label)} K{x.K} & {sci(x.delta_mean_nll)} & {sci(x.block_max_abs_delta)} \\\\" for x in r.itertuples()]
    body = r"""\begin{tabular}{lrr}
\toprule
Run (suite: condition) & mean $\Delta$ & max $|\Delta|$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_rounding_full_split.tex", body)


if __name__ == "__main__":
    tab_cross_suite()
    # tab_factorial_single() is not called: its rows are part of tab_cross_suite
    tab_five_seed_factorial()
    tab_five_seed_endpoints()
    tab_endpoints_all()
    tab_detach()
    tab_downstream()
    tab_substitution()
    tab_layer1_nearbaseline_ci()
    tab_rounding()
    tab_detach_replication()
    tab_grad_decomp()
    tab_strict_audit()
    tab_hardware()
    tab_dstack()
    tab_lowprec()
    tab_early5m()
    tab_probeAB()
    tab_tail_decomp()
    tab_strict_forward()
    tab_e1_channels()
