#!/usr/bin/env python3
"""Numbers and tables of the experiments of Appendix I (and of the main-text sentences that cite them), generated
from the result tables of Appendix I (csv/ of rb-dossier_20261114).

Outputs (written next to this script):
  rb_numbers.tex   -- one \\rbnset{key}{value} per number printed in the text (\\rbn{key} in the manuscript)
  tab_calgrad.tex  -- main-text table: the calibration gradient across calibration types
  tab_rb_*.tex     -- Appendix I tables

Rows 1-2 of tab_calgrad come from the result files of the earlier suites (the same sources as Tables S19, S21, S23, S25 in
make_paper_tables.py); every other number comes from the dossier csv/, except the step-0 check of Appendix I.3, which
comes from reports/e2c_step0_check/e2c_step0_check.csv (a read-only evaluation of three parent checkpoints on a local
RTX 3090, added after the experiments of Appendix I; not a training run). Nothing is typed in by hand: a number missing
from a CSV raises an error.
"""
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN = Path(os.environ.get("KI_REPO", Path(__file__).resolve().parents[2]))   # the root of this repository
AM = TRAIN / "analysis_main_v1"
DOSSIER = Path(os.environ.get("KI_DOSSIER", AM / "rb-rebuttal_v5/reports/rb-dossier_20261114")) / "csv"
E = AM / "E_detach_history"
SUP = AM / "supplement_four_runs_20260923/reports"
OUT = Path(__file__).resolve().parent
NUM = {}          # key -> LaTeX string
SOURCE = {}       # key -> provenance (csv file / row), written as comments


def rd(name):
    return pd.read_csv(DOSSIER / f"{name}.csv")


CMP, SM, RUNS, TRJ = rd("comparisons"), rd("seed_means"), rd("runs").set_index("run_id"), rd("trajectories")
STEP0 = Path(os.environ.get("KI_E2C_STEP0", AM / "rb-rebuttal_v5/reports/e2c_step0_check/e2c_step0_check.csv"))
PRB = rd("probes_layer_mean")
DSB, DSH = rd("downstream_benchmarks"), rd("downstream_heldout")


# ------------------------------------------------------------------ formatting
def sgn(x, d):
    """signed number in math mode, LaTeX minus"""
    s = f"{x:+.{d}f}"
    if float(s) == 0 and s.startswith("-"):
        s = "+" + s[1:]
    return f"${s}$"


def uns(x, d):
    return f"{x:.{d}f}"


def civ(lo, hi, d):
    return f"[{sgn(lo, d)}, {sgn(hi, d)}]"


def put(key, text, src):
    assert key not in NUM, key
    NUM[key] = text
    SOURCE[key] = src


# ------------------------------------------------------------------ lookups (each returns exactly one row)
def one(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    r = df[m]
    assert len(r) == 1, (kw, len(r))
    return r.iloc[0]


def cmp(section, comparison="primary pair", run=None, seed=None):
    kw = dict(section=section, comparison=comparison)
    if run is not None:
        kw["run"] = run
    if seed is not None:
        kw["seed"] = str(seed) if CMP.seed.dtype == object else seed
    return one(CMP, **kw)


def smean(section, condition_prefix):
    r = SM[(SM.section == section) & SM.condition.str.startswith(condition_prefix)]
    assert len(r) == 1, (section, condition_prefix, len(r))
    return r.iloc[0]


def traj(run, tokens_m):
    r = TRJ[(TRJ.run_id == run) & (abs(TRJ.tokens_m - tokens_m) < 0.6)]
    assert len(r) == 1, (run, tokens_m, len(r))
    return r.iloc[0]


def probe(run, metric, step=None):
    r = PRB[(PRB.run_id == run) & (PRB.metric == metric)]
    if step is None:
        step = r.step.max()
    r = r[r.step == step]
    assert len(r) == 1, (run, metric, step, len(r))
    return float(r.value.iloc[0])


def ds(table, comparison, key, metric=None):
    if table is DSB:
        r = one(DSB, comparison=comparison, config=key, metric=metric)
    else:
        r = one(DSH, comparison=comparison, corpus=key)
    return r


# ------------------------------------------------------------------ main-text numbers
def numbers_main():
    # E1b (FWM-LERP K4 at 2.5B) and E1 (100M, 3 seeds)
    r = cmp("E1b", run="rb-E1b-01")
    put("E1b.final3", sgn(r.dnll, 3), "comparisons E1b primary")
    put("E1b.final", sgn(r.dnll, 4), "comparisons E1b primary")
    put("E1b.final.ci", civ(r.ci_lo, r.ci_hi, 4), "comparisons E1b primary")
    put("E1b.at125", sgn(traj("rb-E1b-01", 125).dnll, 3), "trajectories rb-E1b-01 125M")
    put("E1b.at250", sgn(traj("rb-E1b-01", 250.1).dnll, 3), "trajectories rb-E1b-01 250.1M")
    for K in (4, 16):
        s = smean("E1", f"FWM-LERP | K={K} |")
        put(f"E1.K{K}.mean", sgn(s.mean_dnll, 4), "seed_means E1")
        put(f"E1.K{K}.ci", civ(s.t_lo, s.t_hi, 4), "seed_means E1")
    # E11: detach row max under FWM
    k4 = [cmp("E11", run=r).dnll for r in ("rb-E11-01", "rb-E11-02", "rb-E11-05", "rb-E11-07")]
    k16 = [cmp("E11", run=r).dnll for r in ("rb-E11-03", "rb-E11-04")]
    put("E11.K4.lo", sgn(min(k4), 2), "comparisons E11 K4 detach-full"); put("E11.K4.hi", sgn(max(k4), 2), "")
    put("E11.K16.lo", sgn(min(k16), 2), "comparisons E11 K16 detach-full"); put("E11.K16.hi", sgn(max(k16), 2), "")
    # E4: EMA window, Weight - Prob and both vs softmax
    s = smean("E4", "E4 EMA Weight − EMA Prob")
    put("E4.WmP.mean", sgn(s.mean_dnll, 3), "seed_means E4 W-P"); put("E4.WmP.ci", civ(s.t_lo, s.t_hi, 3), "seed_means E4 W-P")
    ema = [smean("E4", f"EMA-window Nearest (per layer and head, decay 0.999, zero tail) | K=4 | {sur}").mean_dnll
           for sur in ("Weight-STE", "Prob-STE")] + [cmp("E4", run=r).dnll for r in ("rb-E4-04", "rb-E4-08")]
    put("E4.vsSoftmax.lo", uns(min(ema), 1), "E4 vs softmax: 100M three-seed means and 250M runs, min")
    put("E4.vsSoftmax.hi", uns(max(ema), 1), "E4 vs softmax: max")
    # E5 formats: 250M endpoints max |dNLL|
    e5 = [abs(cmp("E5", run=f"rb-E5-{i:02d}").dnll) for i in (3, 6, 10, 14, 18)] + [abs(cmp("E5n", run="rb-E5n-04").dnll)]
    put("E5.250.maxabs", uns(math.ceil(max(e5) * 1e4) / 1e4, 4), "comparisons E5 (5 formats) + E5n 250M runs, max |dNLL| rounded up to 4 decimals")
    # E3a-2.5B and E3b
    r = cmp("E3a-2.5B", run="rb-E3a-2.5B-01")
    put("E3a25.final3", sgn(r.dnll, 3), "comparisons E3a-2.5B primary")
    put("E3a25.final", sgn(r.dnll, 4), ""); put("E3a25.final.ci", civ(r.ci_lo, r.ci_hi, 4), "")
    g = smean("E3b", "EXAQ 3-bit, E3b-g"); d = smean("E3b", "EXAQ 3-bit, E3b-d")
    put("E3b.g.mean3", sgn(g.mean_dnll, 3), "seed_means E3b-g"); put("E3b.d.mean3", sgn(d.mean_dnll, 3), "seed_means E3b-d")
    # E2a/b chain analysis and E2d
    e2 = rd("e2_analysis")
    for bp in (50, 125):
        x = e2[(e2.group == "E2a/b") & (e2.branch_point_m == bp)].iloc[0]
        put(f"E2.{bp}.closes", uns(x.wp_closes_pct, 1), "e2_analysis wp_closes_pct")
        put(f"E2.{bp}.opens", uns(x.pw_opens_pct, 1), "e2_analysis pw_opens_pct")
        put(f"E2.{bp}.freeze", uns(x.freeze_removes_pct, 1), "e2_analysis freeze_removes_pct")
    for key, cond in (("WP", "W→P − W→W (same parent)"), ("PW", "P→W − P→P (same parent)")):
        s = smean("E2d", cond)
        put(f"E2d.{key}.mean", sgn(s.mean_dnll, 3), "seed_means E2d"); put(f"E2d.{key}.ci", civ(s.t_lo, s.t_hi, 3), "seed_means E2d")
    e2d = rd("e2d_endpoints")
    mx = max(abs(e2d.wp_minus_pp).max(), abs(e2d.pw_minus_ww).max())
    put("E2d.target.max", uns(math.ceil(mx * 1000) / 1000, 3), "e2d_endpoints max |switched - target| (rounded up)")
    # E8: second seed at 2.5B
    put("E8.val.s7", sgn(cmp("E8c", run="rb-E8c-01").dnll, 4), "comparisons E8c primary")
    put("E8.val.s7.ci", civ(cmp("E8c", run="rb-E8c-01").ci_lo, cmp("E8c", run="rb-E8c-01").ci_hi, 4), "")
    r = one(CMP, section="E8", comparison="seed-1337 counterpart: LERP K16 − softmax (paper)")
    put("E8.val.s1337", sgn(r.dnll, 4), "comparisons E8 seed-1337 counterpart"); put("E8.val.s1337.ci", civ(r.ci_lo, r.ci_hi, 4), "")
    ss = "rb-E8a-01 − softmax s1337 (seed 7 vs seed 1337)"
    put("E8.ss.wt", sgn(ds(DSH, ss, "wikitext103_test")["diff"], 3), "downstream_heldout softmax s7-s1337 wikitext")
    put("E8.ss.ptb", sgn(ds(DSH, ss, "ptb_test")["diff"], 3), "downstream_heldout softmax s7-s1337 ptb")
    put("E8.ss.c4", sgn(ds(DSH, ss, "c4_validation")["diff"], 3), "downstream_heldout softmax s7-s1337 c4")
    r = ds(DSB, ss, "lambada_openai", "acc")
    put("E8.ss.lambada", sgn(r["diff"], 2), "downstream_benchmarks softmax s7-s1337 lambada acc")
    put("E8.ss.lambada.ci", civ(r.ci_lo, r.ci_hi, 2), "")
    r = ds(DSB, "rb-E8c-01 − rb-E8a-01 (LERP K16 − softmax, seed 7)", "blimp", "acc_macro")
    put("E8.blimp.s7", sgn(r["diff"], 2), "downstream_benchmarks E8c-E8a blimp"); put("E8.blimp.s7.ci", civ(r.ci_lo, r.ci_hi, 2), "")
    r = ds(DSB, "LERP K16 − softmax, seed 1337 (paper)", "blimp", "acc_macro")
    put("E8.blimp.s1337", sgn(r["diff"], 2), "downstream_benchmarks paper blimp"); put("E8.blimp.s1337.ci", civ(r.ci_lo, r.ci_hi, 2), "")
    # E3a-2.5B forward swaps
    fw = rd("e3a_2p5b_forwards")
    sw = fw[fw.forward.str.startswith("IndexSoftmax with float") | fw.forward.str.startswith("exact")].d_native
    put("E3a25.swap.lo", uns(sw.min(), 3), "e3a_2p5b_forwards d_native min"); put("E3a25.swap.hi", uns(sw.max(), 3), "max")
    # E6: largest effect of the single-pass variant
    e6 = rd("e6")
    on = e6[e6.setting == "online128"]
    put("E6.online.max", sgn(on.dnll_vs_ref.max(), 4), "e6 online128 max dNLL vs ref")
    x = on.loc[on.dnll_vs_ref.idxmax()]
    put("E6.online.max.ci", civ(x.ci_lo, x.ci_hi, 4), "e6 online128 at the max: 95% CI")


# ------------------------------------------------------------------ Table X (main text): calibration gradient across types
def tab_calgrad():
    # rows 1-2: the result files of the earlier suites (sources of Tables S19, S21, S23, S25)
    leg = pd.read_csv(E / "endpoint_detach_vs_full_s1337.csv").set_index("condition")
    rep = pd.read_csv(AM / "detach_replication/detach_replication_paired.csv")
    rep = rep[rep.ckpt == "100m"].set_index("seed")
    geo = pd.read_csv(E / "geometry_detach_vs_full_s1337.csv")
    g_det = one(geo, condition="LERP K32", variant="detach").span_mean
    g_full = one(geo, condition="LERP K32", variant="full").span_mean
    c7 = pd.read_csv(SUP / "supplement_endpoint_contrasts.csv").set_index("contrast")
    t7 = pd.read_csv(SUP / "supplement_trajectories.csv")
    t7 = t7[(t7.tokens_m == 100) & (t7.evaluator == "L3090")].set_index("series")
    j42 = json.load(open(SUP / "supplement_e1_seed42.json"))
    d1 = [leg.loc["LERP K32"].detach_minus_full, rep.loc[7].dmf, rep.loc[42].dmf]
    sp_off1 = [g_det, t7.loc["lerp_detach", "geom_span_mean_layermean"], j42["geometry_100m"]["detach"]["span_mean"]]
    sp_on = [g_full, t7.loc["lerp_full", "geom_span_mean_layermean"], j42["geometry_100m"]["full"]["span_mean"]]
    d2 = [c7.loc["E1 min_only - full"].delta, j42["contrasts_100m"]["min_only - full"][0]]
    sp_off2 = [t7.loc["e1_min", "geom_span_mean_layermean"], j42["geometry_100m"]["min_only"]["span_mean"]]
    sp_on2 = sp_on[1:]

    def span_rng(v):
        v = sorted(v)
        return f"{v[0]:.0f}" if round(v[0]) == round(v[-1]) else f"{v[0]:.0f}--{v[-1]:.0f}"

    def seeds(v, d=2):
        return ", ".join(sgn(x, d) for x in v)

    rows = [("per-row MinMax", "$M$, $m$", "LERP $K{=}32$", seeds(d1) + r"$^{a}$", "---",
             f"{span_rng(sp_off1)} vs {span_rng(sp_on)}"),
            ("per-row MinMax", r"$M$ ($m$ kept)$^{b}$", "LERP $K{=}32$", seeds(d2), "---",
             f"{span_rng(sp_off2)} vs {span_rng(sp_on2)}")]
    # rows 3-5: E11 under FWM (dossier)
    def e11(runs, full_runs, r250=None):
        d = [cmp("E11", run=r).dnll for r in runs]
        off = [probe(r, "span") for r in runs]
        on = [probe(r, "span") for r in full_runs]
        d250 = sgn(cmp("E11", run=r250).dnll, 2) if r250 else "---"
        return seeds(d), d250, f"{span_rng(off)} vs {span_rng(on)}"
    for op, runs, full, r250 in (("\\FW--LERP $K{=}4$", ("rb-E11-01", "rb-E11-02"), ("rb-E1-01", "rb-E1-02"), "rb-E11-09"),
                                 ("\\FW--LERP $K{=}16$", ("rb-E11-03", "rb-E11-04"), ("rb-E1-05", "rb-E1-06"), None),
                                 ("\\FW--Weight $K{=}4$", ("rb-E11-05", "rb-E11-07"), ("rb-E11-06", "rb-E11-08"), None)):
        a, b, c = e11(runs, full, r250)
        rows.append(("static (\\FW)", "$M$", op, a, b, c))
    # row 6: EXAQ tensor-level statistics (E3b-d - E3b-g)
    s = smean("E3b", "E3b-d − E3b-g")
    r250 = one(CMP, section="E3b", comparison="E3b-d − E3b-g (250M)")
    off = [probe(f"rb-E3b-0{i}", "span") for i in (1, 2, 3)]
    on = [probe(f"rb-E3b-0{i}", "span") for i in (5, 6, 7)]
    rows.append(("tensor-level (EXAQ)", r"$\sigma$, minimum", "3-bit, Prob-STE",
                 f"{sgn(s.mean_dnll, 4)} {civ(s.t_lo, s.t_hi, 4)}$^{{c}}$", sgn(r250.dnll, 3), f"{span_rng(off)} vs {span_rng(on)}"))
    put("calgrad.exaq.mean", sgn(s.mean_dnll, 4), "seed_means E3b d-g"); put("calgrad.exaq.ci", civ(s.t_lo, s.t_hi, 4), "")
    put("calgrad.exaq.250", sgn(r250.dnll, 3), "comparisons E3b d-g 250M")
    # row 7: per-head EMA window (no on-graph control): vs softmax, Weight / Prob
    w = smean("E4", "EMA-window Nearest (per layer and head, decay 0.999, zero tail) | K=4 | Weight-STE")
    p = smean("E4", "EMA-window Nearest (per layer and head, decay 0.999, zero tail) | K=4 | Prob-STE")
    w250, p250 = cmp("E4", run="rb-E4-04").dnll, cmp("E4", run="rb-E4-08").dnll
    spw = [probe(f"rb-E4-0{i}", "span") for i in (1, 2, 3)]
    spp = [probe(f"rb-E4-0{i}", "span") for i in (5, 6, 7)]
    sps = [probe(f"rb-C0-0{i}", "span") for i in (1, 2, 3)]
    rows.append(("per-head EMA", "width (buffer)", "\\CH $K{=}4$",
                 f"{sgn(w.mean_dnll, 3)} (W), {sgn(p.mean_dnll, 3)} (P)$^{{d}}$", f"{sgn(w250, 3)}, {sgn(p250, 3)}",
                 f"{span_rng(spw)} (W), {span_rng(spp)} (P) vs {span_rng(sps)}$^{{e}}$"))
    body = r"""\setlength{\tabcolsep}{2pt}\begin{tabular}{@{}lllllp{2.35cm}@{}}
\toprule
Calibration & Off the graph & Operator & $\dNLL$, 100M & $\dNLL$, 250M & Span, off vs on (100M) \\
\midrule
""" + "\n".join(" & ".join(r) + r" \\" for r in rows) + r"""
\bottomrule
\end{tabular}
"""
    write("tab_calgrad.tex", body)


# ------------------------------------------------------------------ Appendix I: helpers
UNI = [("−", "$-$"), ("→", "$\\to$"), ("≤", "$\\le$"), ("≥", "$\\ge$"), ("×", "$\\times$"), ("·", "$\\cdot$"),
       ("Δ", "$\\Delta$"), ("σ", "$\\sigma$"), ("τ", "$\\tau$"), ("≈", "$\\approx$"), ("…", "\\ldots{}"), ("–", "--"),
       ("—", "---"), ("’", "'"), ("‘", "`"), ("“", "``"), ("”", "''"), ("₀", ""), ("·", "$\\cdot$")]


def tx(s):
    """plain CSV text -> LaTeX text (verbatim wording, escaped)"""
    s = str(s)
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("^", "\\^{}"), ("~", "$\\sim$")):
        s = s.replace(a, b)
    s = s.replace("->", "$\\to$").replace("<=", "$\\le$").replace(">=", "$\\ge$")
    s = s.replace("|", "$|$").replace("<", "$<$").replace(">", "$>$")
    for a, b in UNI:
        s = s.replace(a, b)
    return s


def table(spec, header, rows, mid=None):
    """booktabs tabular; rows: list of lists, or the string 'MID' for a \\midrule"""
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MID":
            out.append("\\midrule")
            continue
        line = " & ".join(str(c) for c in r) + " \\\\"
        out.append(line + ("\\addlinespace[2pt]" if "\\begin{tabular}" in line else ""))
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def d4(x):
    return sgn(x, 4)


def dci(r, lo="ci_lo", hi="ci_hi", val="dnll", d=4):
    return f"{sgn(r[val], d)} {civ(r[lo], r[hi], d)}"


def two(top, bottom):
    """a value and its interval stacked in one cell"""
    return "\\begin{tabular}[c]{@{}l@{}}" + top + "\\\\{" + bottom + "}\\end{tabular}"


def dci2(r, lo="ci_lo", hi="ci_hi", val="dnll", d=4):
    return two(sgn(r[val], d), civ(r[lo], r[hi], d))


def smrow2(s, d=4):
    return two(sgn(s.mean_dnll, d), civ(s.t_lo, s.t_hi, d))


def seedpts(section, runs):
    """per-seed point estimates (the per-seed intervals are in the result files)"""
    return [sgn(cmp(section, run=r).dnll, 4) for r in runs]


def seedrow(section, runs):
    """per-seed primary-pair dNLL for a list of run ids"""
    return [dci(cmp(section, run=r)) for r in runs]


def smrow(s, d=4):
    return f"{sgn(s.mean_dnll, d)} {civ(s.t_lo, s.t_hi, d)}"


# ------------------------------------------------------------------ I.1 overview, predictions
# The overview table groups the experiments by the question they answer (subsection order). The run codes are used in this script only, to read the csv files.
GROUPS = [("Controls on the software stack of these experiments (Appendix~\\ref{app:rb-overview})",
           [("C0", "same-stack softmax controls (100M, three seeds)"), ("C0b", "same-stack softmax control (250M)")]),
          ("Can LERP be paired with the fixed window, and does the calibration-gradient result carry over to a static grid? (\\ref{app:rb-fwmlerp})",
           [("E1", "\\FW--LERP, 100M and 250M"), ("E1b", "\\FW--LERP $K{=}4$ at 2.5B"), ("E11", "row maximum detached under \\FW")]),
          ("Is the surrogate rule a cause of the failure under MinMax? (\\ref{app:rb-interv})",
           [("E2a/b", "surrogate switch, frozen step"), ("E2c", "the same on the 2.5B schedule"), ("E2d", "surrogate switch on three seeds")]),
          ("Does a window that tracks the scores without being differentiated train? (\\ref{app:rb-ema})",
           [("E4", "per-head exponential-moving-average (EMA) window")]),
          ("Do the rounding rules of standard number formats train? (\\ref{app:rb-formats})",
           [("E5", "rounding rules of five formats, incl. MXFP8, NVFP4"), ("E5n", "bf16 row sum and division")]),
          ("Does the calibration-gradient result hold for a published operator with tensor-level calibration? (\\ref{app:rb-exaq})",
           [("E3b", "EXAQ, statistics detached or in the graph")]),
          ("Does a published lookup operator train, with and without reduced-precision normalization? (\\ref{app:rb-index})",
           [("E3a", "IndexSoftmax, (i) float, (ii) UINT8 normalization"), ("E3a-2.5B", "IndexSoftmax at 2.5B")]),
          ("Which downstream differences persist on a second seed? (\\ref{app:rb-seed})",
           [("E8a", "second softmax seed at 2.5B"), ("E8c", "second LERP $K{=}16$ seed at 2.5B")]),
          ("Does TF32 in training change the result? (\\ref{app:rb-evalset})",
           [("E7", "TF32 off in training")])]


def tab_rb_overview():
    rr = RUNS.reset_index()
    cp = rd("compute_per_experiment").set_index("experiment")
    listed = [e for _, rows in GROUPS for e, _ in rows]
    assert sorted(listed) == sorted(set(rr.experiment)) and len(listed) == len(set(listed)), "every experiment exactly once"
    W = "12.9cm"
    out = ["\\begin{tabular}{@{}p{6.6cm}p{2.6cm}lrr@{}}", "\\toprule",
           "Question and experiment & Suite protocol & Seeds & Runs & GPU-h \\\\", "\\midrule"]
    for gi, (head, rows) in enumerate(GROUPS):
        if gi:
            out.append("\\addlinespace[3pt]")
        out.append(f"\\multicolumn{{5}}{{@{{}}p{{{W}}}@{{}}}}{{\\emph{{{head}}}}} \\\\")
        for e, text in rows:
            g = rr[rr.experiment == e]
            prot = sorted(set(g.protocol.str.split(" ").str[0]))
            prot = ", ".join({"suiteD": "D (100M)", "suiteB": "B (250M)", "suiteA": "A (2.5B)"}.get(x, x) for x in prot)
            seeds = ", ".join(str(s) for s in sorted(set(g.seed.astype(int))))
            assert len(g) == int(cp.loc[e].runs)
            out.append(f"\\quad {text} & {prot} & {seeds} & {len(g)} & {uns(cp.loc[e].gpu_hours, 1)} \\\\")
    out += ["\\midrule", f"Total &  &  & {int(cp.loc['total'].runs)} & {uns(cp.loc['total'].gpu_hours, 1)} \\\\", "\\bottomrule", "\\end{tabular}"]
    put("I.runs", str(int(cp.loc["total"].runs)), "compute_per_experiment total runs")
    put("I.gpuh", uns(cp.loc["total"].gpu_hours, 1), "compute_per_experiment total GPU-h")
    ev = rd("compute_evaluation")
    put("I.evalh", uns(ev.gpu_hours.sum(), 1), "compute_evaluation sum of GPU-h")
    write("tab_rb_overview.tex", "\n".join(out) + "\n")


# ------------------------------------------------------------------ I.2 FWM-LERP and E11
def tab_rb_e1():
    rows = []
    for K, runs, r250 in ((4, ("rb-E1-01", "rb-E1-02", "rb-E1-03"), "rb-E1-04"), (16, ("rb-E1-05", "rb-E1-06", "rb-E1-07"), "rb-E1-08")):
        s = smean("E1", f"FWM-LERP | K={K} |")
        rows.append([f"$K{{=}}{K}$"] + seedpts("E1", runs) + [smrow2(s), dci2(cmp("E1", run=r250))])
    write("tab_rb_e1.tex", table("@{}llllll@{}", ["\\FW--LERP", "seed 7", "seed 42", "seed 1337", "mean, 95\\% $t$",
                                                   "250M (seed 1337), 95\\% CI"], rows))
    c = [("\\FW--LERP ($K{=}4$), 250M\\enspace$-$\\enspace suite B \\FW--Weight ($K{=}4$)", one(CMP, section="E1", comparison="E1-04 (FWM-LERP K4, 250M) − suite B FWM-Weight K4")),
         ("\\FW--LERP ($K{=}4$), 250M\\enspace$-$\\enspace suite B \\FW--Prob ($K{=}4$)", one(CMP, section="E1", comparison="E1-04 (FWM-LERP K4, 250M) − suite B FWM-Prob K4")),
         ("\\FW--LERP ($K{=}4$), 2.5B\\enspace$-$\\enspace suite A softmax", cmp("E1b", run="rb-E1b-01")),
         ("\\FW--LERP ($K{=}4$), 2.5B\\enspace$-$\\enspace suite A MinMax--LERP ($K{=}4$)", one(CMP, section="E1b", comparison="E1b − LERP K4 (suite A s1337)")),
         ("\\FW--LERP ($K{=}4$), 2.5B\\enspace$-$\\enspace suite A \\FW--Weight ($K{=}4$)", one(CMP, section="E1b", comparison="E1b − FWM-Weight K4 (suite A s1337)")),
         ("\\FW--LERP ($K{=}4$), 2.5B\\enspace$-$\\enspace suite A \\FW--Prob ($K{=}4$)", one(CMP, section="E1b", comparison="E1b − FWM-Prob K4 (suite A s1337)"))]
    write("tab_rb_e1b.tex", table("@{}lll@{}", ["Comparison (seed 1337)", "NLL", "$\\dNLL$ [95\\% CI]"],
                                  [[a, uns(r.nll_run, 4), dci(r)] for a, r in c]))
    for key, r in (("E1.250.vsW", c[0][1]), ("E1.250.vsP", c[1][1]), ("E1b.vsLERP", c[3][1]), ("E1b.vsW", c[4][1]), ("E1b.vsP", c[5][1])):
        put(key, sgn(r.dnll, 4), "comparisons E1/E1b"); put(key + ".ci", civ(r.ci_lo, r.ci_hi, 4), "")
    put("E1.250.final", sgn(cmp("E1", run="rb-E1-04").dnll, 4), "comparisons rb-E1-04")
    put("E1.250.final.ci", civ(cmp("E1", run="rb-E1-04").ci_lo, cmp("E1", run="rb-E1-04").ci_hi, 4), "")


def downstream_rows(comparisons, labels):
    cfg = [("blimp", "acc_macro", "BLiMP"), ("hellaswag", "acc_norm", "HellaSwag"), ("piqa", "acc_norm", "PIQA"),
           ("arc_easy", "acc_norm", "ARC-Easy"), ("lambada_openai", "acc", "LAMBADA acc"),
           ("lambada_openai", "target_nll_corpus_token_weighted", "LAMBADA NLL"), ("sciq_default", "acc", "SciQ"),
           ("sciq_nosupport", "acc", "SciQ, no support")]
    rows = []
    for c, m, lab in cfg:
        d = 4 if "nll" in m else 2
        rows.append([lab] + [dci2(ds(DSB, cmpn, c, m), val="diff", d=d) for cmpn in comparisons])
    rows.append("MID")
    for corp, lab in (("wikitext103_test", "WikiText-103 test (NLL)"), ("ptb_test", "PTB test (NLL)"), ("c4_validation", "C4 validation (NLL)")):
        rows.append([lab] + [dci2(ds(DSH, cmpn, corp), val="diff", d=4) for cmpn in comparisons])
    return rows


def tab_rb_e1b_down():
    comps = ["rb-E1b-01 − softmax s1337", "rb-E1b-01 − LERP K4 s1337"]
    write("tab_rb_e1b_down.tex", table("@{}lll@{}", ["Configuration", "\\FW--LERP ($K{=}4$)\\enspace$-$\\enspace softmax", "\\FW--LERP ($K{=}4$)\\enspace$-$\\enspace MinMax--LERP ($K{=}4$)"],
                                       downstream_rows(comps, None)))
    for corp, k in (("wikitext103_test", "wt"), ("ptb_test", "ptb"), ("c4_validation", "c4")):
        put(f"E1b.held.{k}", sgn(ds(DSH, comps[0], corp)["diff"], 4), "downstream_heldout E1b - softmax")


def tab_rb_e11():
    pairs = [("\\FW--LERP $K{=}4$", "rb-E11-01", "rb-E1-01", 7), ("\\FW--LERP $K{=}4$", "rb-E11-02", "rb-E1-02", 42),
             ("\\FW--LERP $K{=}16$", "rb-E11-03", "rb-E1-05", 7), ("\\FW--LERP $K{=}16$", "rb-E11-04", "rb-E1-06", 42),
             ("\\FW--Weight $K{=}4$", "rb-E11-05", "rb-E11-06", 7), ("\\FW--Weight $K{=}4$", "rb-E11-07", "rb-E11-08", 42),
             ("\\FW--LERP $K{=}4$, 250M", "rb-E11-09", "rb-E1-04", 1337)]
    rows = []
    for lab, det, full, seed in pairs:
        r = cmp("E11", run=det)
        rows.append([lab, seed, dci2(r)] + [f"{probe(det, m):.{d}f} vs {probe(full, m):.{d}f}" for m, d in
                    (("span", 0), ("pmax_native", 2), ("below_window_entry_frac", 2))] +
                    [f"{probe(det, 'grad_row_zero_sum_residual_rel', None):.2f} vs {probe(full, 'grad_row_zero_sum_residual_rel'):.0e}".replace("e-0", "e-")])
    write("tab_rb_e11.tex", table("@{}lrllllll@{}", ["Operator", "Seed", "detach $-$ full [95\\% CI]", "span", "$p_{\\max}$",
                                                      "clip fraction", "zero-sum residual"], rows))
    zs_d = [probe(d, "grad_row_zero_sum_residual_rel") for _, d, _, _ in pairs]
    zs_f = [probe(f, "grad_row_zero_sum_residual_rel") for _, _, f, _ in pairs]
    put("E11.zs.lo", uns(min(zs_d), 2), "probe grad_row_zero_sum_residual_rel, detached runs min")
    put("E11.zs.hi", uns(max(zs_d), 2), "probe grad_row_zero_sum_residual_rel, detached runs max")
    put("E11.zs.full", f"$10^{{{int(np.ceil(np.log10(max(zs_f))))}}}$", "probe grad_row_zero_sum_residual_rel, full runs: power of ten above the max")
    r = one(CMP, section="E11", comparison="E11-06 − App. C.2 FWM-Weight K4 strict s7 (same configuration)")
    put("E11.06.vsC2", sgn(r.dnll, 4), "comparisons E11-06 vs App C.2 run")
    put("E11.250", sgn(cmp("E11", run="rb-E11-09").dnll, 2), "comparisons rb-E11-09")
    put("E11.250.span", uns(probe("rb-E11-09", "span"), 0), "probe span rb-E11-09")
    m4 = smean("E11", "FWM-LERP | K=4 | none (exact Jacobian) | detach"); m16 = smean("E11", "FWM-LERP | K=16 | none (exact Jacobian) | detach")
    put("E11.K4.mean", sgn(m4.mean_dnll, 3), "seed_means E11 K4 (2 seeds)"); put("E11.K16.mean", sgn(m16.mean_dnll, 3), "seed_means E11 K16 (2 seeds)")


# ------------------------------------------------------------------ I.3 interventions
def tab_rb_e2ab():
    e2 = rd("e2_analysis")
    rows = []
    br = {50: ["rb-E2ab-01", "rb-E2ab-02", "rb-E2ab-03", "rb-E2ab-04", "rb-E2ab-05"],
          125: ["rb-E2ab-06", "rb-E2ab-07", "rb-E2ab-08", "rb-E2ab-09", "rb-E2ab-10"]}
    names = ["W$\\to$W", "W$\\to$P", "W$\\to$W, frozen step", "P$\\to$P", "P$\\to$W"]
    for bp in (50, 125):
        x = e2[(e2.group == "E2a/b") & (e2.branch_point_m == bp)].iloc[0]
        for i, (run, nm) in enumerate(zip(br[bp], names)):
            r = cmp("E2a/b", run=run)
            rows.append([f"{bp}M" if i == 0 else "", nm, uns(r.nll_run, 4), dci2(r), uns(probe(run, "span"), 1),
                         uns(probe(run, "h_native"), 2)])
        rows.append("MID")
    rows = rows[:-1]
    write("tab_rb_e2ab.tex", table("@{}lllllr@{}", ["Branch", "Rule", "NLL, 250M", "$-$ softmax control [95\\% CI]", "span", "$h$"], rows))
    rows = []
    for bp in (50, 125):
        x = e2[(e2.group == "E2a/b") & (e2.branch_point_m == bp)].iloc[0]
        rows.append([f"{bp}M", sgn(x.ww_minus_orig_w, 4), sgn(x.pp_minus_orig_p, 4), uns(x.wp_closes_pct, 1) + "\\%",
                     uns(x.pw_opens_pct, 1) + "\\%", uns(x.freeze_removes_pct, 1) + "\\%", sgn(x.freeze_minus_pp, 4)])
        put(f"E2.{bp}.valid.ww", sgn(x.ww_minus_orig_w, 4), "e2_analysis ww_minus_orig_w")
        put(f"E2.{bp}.valid.pp", sgn(x.pp_minus_orig_p, 4), "e2_analysis pp_minus_orig_p")
        for k in ("ww", "wp", "freeze", "pp", "pw"):
            put(f"E2.{bp}.span.{k}", uns(x[f"span_{k}"], 1), "e2_analysis span")
        put(f"E2.{bp}.freezeleft", sgn(x.freeze_minus_pp, 4), "e2_analysis freeze_minus_pp")
    write("tab_rb_e2chain.tex", table("@{}lllllll@{}", ["Branch", "W$\\to$W $-$ W$^{\\dagger}$", "P$\\to$P $-$ P$^{\\dagger}$",
                                                         "W$\\to$P removes", "P$\\to$W reproduces", "frozen step removes",
                                                         "frozen $-$ P$\\to$P"], rows))
    x = e2[e2.group == "E2c"].iloc[0]
    rows = [["W$\\to$W", uns(x.ww_peak_nll, 3), f"{x.ww_peak_m:.0f}M", uns(x.ww_end_nll, 4)],
            ["W$\\to$P", uns(x.wp_peak_nll, 3), f"{x.wp_peak_m:.0f}M", uns(x.wp_end_nll, 4)],
            ["W$\\to$W, frozen step", uns(x.freeze_peak_nll, 3), f"{x.freeze_peak_m:.0f}M", uns(x.freeze_end_nll, 4)],
            ["original MinMax--Weight (every 125M)", uns(x.orig_peak_nll, 3), f"{x.orig_peak_m:.0f}M", "---"]]
    write("tab_rb_e2c.tex", table("@{}llll@{}", ["Branch from 250M", "highest NLL", "at", "NLL at 625M"], rows))
    put("E2c.ww.peak", uns(x.ww_peak_nll, 2), "e2_analysis E2c ww_peak_nll"); put("E2c.ww.peakm", f"{x.ww_peak_m:.0f}", "")
    put("E2c.orig.peak", uns(x.orig_peak_nll, 2), "e2_analysis orig_peak_nll"); put("E2c.orig.peakm", f"{x.orig_peak_m:.0f}", "")
    put("E2c.freeze.first", uns(x.freeze_first_ckpt_nll, 2), "e2_analysis freeze_first_ckpt_nll")
    put("E2c.freeze.firstm", f"{x.freeze_first_ckpt_m:.0f}", ""); put("E2c.freeze.end", uns(x.freeze_end_nll, 2), "")
    put("E2c.wp.end", uns(x.wp_end_nll, 2), "e2_analysis wp_end_nll")
    r = one(CMP, section="E2c", comparison="W→P − W→W (same branch point)")
    put("E2c.wp.vsww", sgn(r.dnll, 2), "comparisons E2c W->P - W->W")


def tab_rb_e2d():
    e = rd("e2d_endpoints").set_index("seed")
    seeds = (7, 42, 253)
    def same_parent(comp, s):
        run = {"W→P": "rb-E2d-%02d" % (2 + 4 * seeds.index(s)), "P→W": "rb-E2d-%02d" % (4 + 4 * seeds.index(s))}[comp[:3]]
        return cmp("E2d", run=run)
    def cross(comp, s):
        return one(CMP, section="E2d", comparison=comp, seed=str(s) if CMP.seed.dtype == object else s)
    rows = []
    for col, lab in (("ww", "W$\\to$W"), ("wp", "W$\\to$P"), ("pp", "P$\\to$P"), ("pw", "P$\\to$W")):
        rows.append([f"NLL, {lab}"] + [uns(e.loc[s][col], 4) for s in seeds] + [""])
    rows.append("MID")
    for comp, lab, f in (("W→P − W→W (same parent)", "W$\\to$P $-$ W$\\to$W", same_parent), ("P→W − P→P (same parent)", "P$\\to$W $-$ P$\\to$P", same_parent),
                         ("W→P − P→P (different parents)", "W$\\to$P $-$ P$\\to$P", cross), ("P→W − W→W (different parents)", "P$\\to$W $-$ W$\\to$W", cross)):
        rows.append([lab] + [dci2(f(comp, s)) for s in seeds] + [smrow2(smean("E2d", comp))])
    write("tab_rb_e2d.tex", table("@{}lllll@{}", ["Branch or contrast", "seed 7", "seed 42", "seed 253", "mean, 95\\% $t$"], rows))
    for key, comp in (("E2d.WPmPP", "W→P − P→P (different parents)"), ("E2d.PWmWW", "P→W − W→W (different parents)")):
        s = smean("E2d", comp); put(key + ".mean", sgn(s.mean_dnll, 4), "seed_means E2d"); put(key + ".ci", civ(s.t_lo, s.t_hi, 4), "")


# ------------------------------------------------------------------ I.4 EMA window
def tab_rb_e4():
    rows = []
    for sur, runs, r250 in (("Weight-STE", ("rb-E4-01", "rb-E4-02", "rb-E4-03"), "rb-E4-04"), ("Prob-STE", ("rb-E4-05", "rb-E4-06", "rb-E4-07"), "rb-E4-08")):
        s = smean("E4", f"EMA-window Nearest (per layer and head, decay 0.999, zero tail) | K=4 | {sur}")
        rows.append([f"EMA, {sur} $-$ softmax"] + seedpts("E4", runs) + [smrow2(s), dci2(cmp("E4", run=r250))])
    w_p = [one(CMP, section="E4", comparison="E4 EMA Weight − EMA Prob", seed=str(s) if CMP.seed.dtype == object else s) for s in (7, 42, 1337)]
    s = smean("E4", "E4 EMA Weight − EMA Prob")
    rows.append(["EMA, Weight $-$ Prob"] + [sgn(r.dnll, 4) for r in w_p] + [smrow2(s), dci2(one(CMP, section="E4", comparison="E4 EMA Weight − EMA Prob (250M)"))])
    h = rd("e4_historical_gap").set_index("quantity")
    a = h.loc["historical MinMax-Weight − MinMax-Prob, 100M (suite D, RTX 3090)"]
    b = h.loc["historical MinMax-Weight − MinMax-Prob, 250M (suite B, S1)"]
    rows.append("MID")
    rows.append(["per-row MinMax, Weight $-$ Prob$^{\\dagger}$", d4(a.s7), d4(a.s42), d4(a.s1337), d4(a["mean"]), d4(b.s1337)])
    put("E4.hist.100", sgn(a["mean"], 3), "e4_historical_gap"); put("E4.hist.250", sgn(b.s1337, 3), "e4_historical_gap")
    put("E4.WmP.250", sgn(one(CMP, section="E4", comparison="E4 EMA Weight − EMA Prob (250M)").dnll, 3), "comparisons E4 W-P 250M")
    write("tab_rb_e4.tex", table("@{}llllll@{}", ["Contrast", "seed 7", "seed 42", "seed 1337", "mean, 95\\% $t$", "250M (seed 1337), 95\\% CI"], rows))
    rows = []
    for run in ("rb-E4-01", "rb-E4-02", "rb-E4-03", "rb-E4-04", "rb-E4-05", "rb-E4-06", "rb-E4-07", "rb-E4-08"):
        rr = RUNS.loc[run]
        rows.append([rr.surrogate.split(" ")[0], int(rr.seed), f"{rr.final_tokens / 1e6:.0f}M",
                     uns(probe(run, "span"), 1), uns(probe(run, "ema_buffer_mean"), 1), uns(probe(run, "below_window_entry_frac"), 3)])
    write("tab_rb_e4probe.tex", table("@{}lrrrrr@{}", ["Surrogate", "Seed", "Tokens", "span", "EMA width", "clip fraction"], rows))


# ------------------------------------------------------------------ I.5 formats
def tab_rb_e5():
    fm = [("BF16-exp", "BF16-exp", ("rb-E5-01", "rb-E5-02"), "rb-E5-03"), ("FP8-E4M3", "FP8-E4M3 rounded exp", ("rb-E5-04", "rb-E5-05"), "rb-E5-06"),
          ("MXFP8-style", "MXFP8-style", ("rb-E5-07", "rb-E5-08", "rb-E5-09"), "rb-E5-10"),
          ("NVFP4-style", "NVFP4-style", ("rb-E5-11", "rb-E5-12", "rb-E5-13"), "rb-E5-14"),
          ("UINT8", "UINT8 uniform", ("rb-E5-15", "rb-E5-16", "rb-E5-17"), "rb-E5-18")]
    rows = []
    for lab, pref, runs, r250 in fm:
        s = smean("E5", pref)
        per = {int(RUNS.loc[r].seed): sgn(cmp("E5", run=r).dnll, 4) for r in runs}
        if 7 not in per:   # seed 7 = the App. D.5 run (RTX 3090 stack)
            key = {"BF16-exp": "BF16-exp seed 7 (App. D.5 run)", "FP8-E4M3": "FP8-E4M3 exp seed 7 (App. D.5 run)"}[lab]
            per[7] = sgn(one(CMP, section="E5", comparison=key).dnll, 4) + "$^{\\ast}$"
        rows.append([lab, per[7], per[42], per[1337], smrow2(s), dci2(cmp("E5", run=r250))])
    s = smean("E5n", "BF16 full-chain softmax")
    rows.append("MID")
    rows.append(["bf16 chain"] + seedpts("E5n", ("rb-E5n-01", "rb-E5n-02", "rb-E5n-03")) + [smrow2(s), dci2(cmp("E5n", run="rb-E5n-04"))])
    write("tab_rb_e5.tex", table("@{}llllll@{}", ["Rule on $e^z$", "seed 7", "seed 42", "seed 1337", "mean, 95\\% $t$", "250M (seed 1337), 95\\% CI"], rows))
    means = [smean("E5", p).mean_dnll for _, p, _, _ in fm]
    put("E5.mean.lo", sgn(min(means), 3), "seed_means E5 min"); put("E5.mean.hi", sgn(max(means), 3), "seed_means E5 max")
    put("E5n.mean", sgn(s.mean_dnll, 4), "seed_means E5n"); put("E5n.ci", civ(s.t_lo, s.t_hi, 4), "")
    b = rd("e0c_bitwidth").set_index("scheme")
    rows = []
    for key, lab, rule in (("bf16exp", "BF16-exp", "$\\mathrm{bf16}(\\exp(\\mathrm{bf16}(z)))$"),
                           ("fp8e4m3exp", "FP8-E4M3", "E4M3 of $e^z$, round to nearest even"),
                           ("mxfp8", "MXFP8-style", "blocks of 32 keys, scale $2^{e}$ with $e$ the smallest integer such that block max $\\le 448\\cdot2^{e}$; E4M3 elements"),
                           ("nvfp4", "NVFP4-style", "tensor scale $2688/\\max w$ (always 2688), blocks of 16 keys with E4M3 block scale, E2M1 elements"),
                           ("uint8", "UINT8", "$\\mathrm{round}(255\\,e^z)/255$ (rounds $e^z$, not $p$)")):
        x = b.loc[f"E5 format {key}"]
        rows.append([lab, rule, int(x.levels_all_z), "$" + uns(x.underflow_all_z, 4 if key != "bf16exp" else 0) + "$"])
        put(f"E0c.{key}.levels", str(int(x.levels_all_z)), "e0c_bitwidth levels_all_z")
    write("tab_rb_formats.tex", table("@{}llrr@{}".replace("ll", "lp{7.0cm}"), ["Format", "Rule on $e^z$", "non-zero values", "zero for $z\\le$"], rows))


# ------------------------------------------------------------------ I.6 EXAQ
def tab_rb_e3b():
    rows = []
    for lab, pref, runs, r250 in (("statistics detached (d)", "EXAQ 3-bit, E3b-d", ("rb-E3b-01", "rb-E3b-02", "rb-E3b-03"), "rb-E3b-04"),
                                  ("statistics in the graph (g)", "EXAQ 3-bit, E3b-g", ("rb-E3b-05", "rb-E3b-06", "rb-E3b-07"), "rb-E3b-08")):
        rows.append([lab + " $-$ softmax"] + seedpts("E3b", runs) + [smrow2(smean("E3b", pref)), dci2(cmp("E3b", run=r250))])
    dg = [one(CMP, section="E3b", comparison="E3b-d − E3b-g", seed=str(s) if CMP.seed.dtype == object else s) for s in (7, 42, 1337)]
    rows.append(["d $-$ g"] + [sgn(r.dnll, 4) for r in dg] + [smrow2(smean("E3b", "E3b-d − E3b-g")), dci2(one(CMP, section="E3b", comparison="E3b-d − E3b-g (250M)"))])
    gi = [one(CMP, section="E3b", comparison="E3b-g − E3a-i", seed=str(s) if CMP.seed.dtype == object else s) for s in (7, 42, 1337)]
    rows.append(["g $-$ IndexSoftmax (i)"] + [sgn(r.dnll, 4) for r in gi] + [smrow2(smean("E3b", "E3b-g − E3a-i")), dci2(one(CMP, section="E3b", comparison="E3b-g − E3a-i (250M)"))])
    write("tab_rb_e3b.tex", table("@{}llllll@{}", ["Contrast", "seed 7", "seed 42", "seed 1337", "mean, 95\\% $t$", "250M (seed 1337), 95\\% CI"], rows))
    rows = []
    for run in ("rb-E3b-01", "rb-E3b-02", "rb-E3b-03", "rb-E3b-04", "rb-E3b-05", "rb-E3b-06", "rb-E3b-07", "rb-E3b-08"):
        rr = RUNS.loc[run]
        rows.append([rr.attn_variant, int(rr.seed), f"{rr.final_tokens / 1e6:.0f}M", uns(probe(run, "span"), 1),
                     uns(probe(run, "exaq_sigma"), 2), uns(probe(run, "exaq_lo"), 2), uns(probe(run, "exaq_step"), 3),
                     uns(probe(run, "exaq_tail_entry_frac"), 3), f"{probe(run, 'exaq_tail_softmax_mass'):.4f}"])
    write("tab_rb_e3bprobe.tex", table("@{}lrrrrrrp{1.3cm}p{1.5cm}@{}", ["Stats", "Seed", "Tokens", "span", "$\\sigma$", "lower bound", "step",
                                                            "clamped entries", "softmax mass clamped"], rows))
    put("E3b.dg.mean", sgn(smean("E3b", "E3b-d − E3b-g").mean_dnll, 4), "seed_means E3b d-g")
    put("E3b.dg.250", sgn(one(CMP, section="E3b", comparison="E3b-d − E3b-g (250M)").dnll, 4), "comparisons E3b d-g 250M")
    put("E3b.d.250span", uns(probe("rb-E3b-04", "span"), 0), "probe span rb-E3b-04")
    put("E3b.g.250span", uns(probe("rb-E3b-08", "span"), 1), "probe span rb-E3b-08")


# ------------------------------------------------------------------ I.7 IndexSoftmax
def tab_rb_e3a():
    rows = []
    for lab, pref, runs, r250 in (("float normalization (i)", "IndexSoftmax E3a-i lookup", ("rb-E3a-01", "rb-E3a-02", "rb-E3a-03"), "rb-E3a-04"),
                                  ("UINT8 probabilities (ii)", "IndexSoftmax E3a-ii lookup", ("rb-E3a-05", "rb-E3a-06", "rb-E3a-07"), "rb-E3a-08")):
        rows.append([lab + " $-$ softmax"] + seedpts("E3a", runs) + [smrow2(smean("E3a", pref)), dci2(cmp("E3a", run=r250))])
    ii = [one(CMP, section="E3a", comparison="E3a-ii − E3a-i", seed=str(s) if CMP.seed.dtype == object else s) for s in (7, 42, 1337)]
    rows.append(["(ii) $-$ (i)"] + [sgn(r.dnll, 4) for r in ii] + [smrow2(smean("E3a", "E3a-ii − E3a-i")), dci2(one(CMP, section="E3a", comparison="E3a-ii − E3a-i (250M)"))])
    write("tab_rb_e3a.tex", table("@{}llllll@{}", ["Contrast", "seed 7", "seed 42", "seed 1337", "mean, 95\\% $t$", "250M (seed 1337), 95\\% CI"], rows))
    s = smean("E3a", "IndexSoftmax E3a-ii lookup")
    put("E3a.ii.mean", sgn(s.mean_dnll, 4), "seed_means E3a-ii"); put("E3a.ii.ci", civ(s.t_lo, s.t_hi, 4), "")
    put("E3a.ii.250", sgn(cmp("E3a", run="rb-E3a-08").dnll, 4), "comparisons rb-E3a-08")
    put("E3a.i.250", sgn(cmp("E3a", run="rb-E3a-04").dnll, 4), "comparisons rb-E3a-04")
    fw = rd("e3a_2p5b_forwards")
    rows = []
    for _, x in fw.iterrows():
        if "override wrapper" in x.forward:   # self-check of the forward-swap mechanism, same values as the first row
            continue
        nat = "---" if pd.isna(x.d_native_lo) or x.d_native == 0 and "native" in x.forward else f"{sgn(x.d_native, 4)} {civ(x.d_native_lo, x.d_native_hi, 4)}"
        FWD = {"native: IndexSoftmax UINT8 (E3a-ii, as trained)": "native: IndexSoftmax (ii), UINT8, as trained",
               "IndexSoftmax with float normalisation (E3a-i forward)": "IndexSoftmax (i), float normalization"}
        rows.append([tx(FWD.get(x.forward, x.forward)), uns(x.nll, 4), nat, f"{sgn(x.d_softmax, 4)} {civ(x.d_softmax_lo, x.d_softmax_hi, 4)}"])
    for comp, lab in (("E3a-2.5B − FWM-Prob K4 (suite A s1337)", "\\FW--Prob ($K{=}4$)"), ("E3a-2.5B − FWM-Weight K16 (suite A s1337)", "\\FW--Weight ($K{=}16$)"),
                      ("E3a-2.5B − LERP K4 (suite A s1337)", "MinMax--LERP ($K{=}4$)")):
        r = one(CMP, section="E3a-2.5B", comparison=comp)
        rows.append([f"native\\enspace$-$\\enspace suite A {lab}", uns(r.nll_run, 4), "", dci(r)])
    write("tab_rb_e3a25.tex", table("@{}p{4.6cm}lll@{}", ["Forward at evaluation (weights unchanged)", "NLL", "$-$ native", "$-$ suite A softmax (or as stated)"], rows))
    put("E3a25.vsFWMP", sgn(one(CMP, section="E3a-2.5B", comparison="E3a-2.5B − FWM-Prob K4 (suite A s1337)").dnll, 4), "comparisons E3a-2.5B vs FWM-Prob K4")
    comps = ["rb-E3a-2.5B-01 − softmax s1337", "rb-E3a-2.5B-01 − FWM-Prob K4 s1337"]
    write("tab_rb_e3a_down.tex", table("@{}lll@{}", ["Configuration", "IndexSoftmax (UINT8)\\enspace$-$\\enspace softmax", "IndexSoftmax (UINT8)\\enspace$-$\\enspace \\FW--Prob ($K{=}4$)"],
                                       downstream_rows(comps, None)))
    r = ds(DSB, comps[0], "blimp", "acc_macro")
    put("E3a25.blimp", sgn(r["diff"], 2), "downstream_benchmarks E3a-2.5B blimp"); put("E3a25.blimp.ci", civ(r.ci_lo, r.ci_hi, 2), "")
    t2 = TRJ[TRJ.run_id == "rb-E3a-2.5B-01"].sort_values("tokens_m")
    put("E3a25.at1000", sgn(traj("rb-E3a-2.5B-01", 1000.1).dnll, 4), "trajectories E3a-2.5B 1000M")
    put("E3a25.at2000", sgn(traj("rb-E3a-2.5B-01", 2000).dnll, 4), "trajectories E3a-2.5B 2000M")


# ------------------------------------------------------------------ I.8 zeroed mass
def tab_rb_e0b():
    te = rd("e0b_trained_endpoints").set_index("run_id")
    # one operator per row (no "/" inside a cell); a range "a--b" is over the runs at that length
    groups = [("\\FW--LERP $K{=}4$, 100M", ["rb-E1-01", "rb-E1-02", "rb-E1-03"]), ("\\FW--LERP $K{=}16$, 100M", ["rb-E1-05", "rb-E1-06", "rb-E1-07"]),
              ("\\FW--LERP $K{=}4$, 250M", ["rb-E1-04"]), ("\\FW--LERP $K{=}16$, 250M", ["rb-E1-08"]),
              ("\\FW--Weight $K{=}4$, 100M", ["rb-E11-06", "rb-E11-08"]), ("\\FW--Weight $K{=}4$, row maximum detached, 100M", ["rb-E11-05", "rb-E11-07"]),
              ("IndexSoftmax, float normalization, 100M", ["rb-E3a-01", "rb-E3a-02", "rb-E3a-03"]), ("IndexSoftmax, UINT8, 100M", ["rb-E3a-05", "rb-E3a-06", "rb-E3a-07"]),
              ("IndexSoftmax, float normalization, 250M", ["rb-E3a-04"]), ("IndexSoftmax, UINT8, 250M", ["rb-E3a-08"]),
              ("IndexSoftmax, UINT8, 2.5B", ["rb-E3a-2.5B-01"]),
              ("EMA window, Weight-STE, 100M", ["rb-E4-01", "rb-E4-02", "rb-E4-03"]), ("EMA window, Prob-STE, 100M", ["rb-E4-05", "rb-E4-06", "rb-E4-07"]),
              ("FP8-E4M3, 100M", ["rb-E5-04", "rb-E5-05"]), ("NVFP4-style, 100M", ["rb-E5-11", "rb-E5-12", "rb-E5-13"]),
              ("FP8-E4M3, 250M", ["rb-E5-06"]), ("NVFP4-style, 250M", ["rb-E5-14"]),
              ("UINT8 on $e^z$, 100M", ["rb-E5-15", "rb-E5-16", "rb-E5-17"]), ("MXFP8-style, 100M", ["rb-E5-07", "rb-E5-08", "rb-E5-09"]),
              ("UINT8 on $e^z$, 250M", ["rb-E5-18"]), ("MXFP8-style, 250M", ["rb-E5-10"])]
    def f(v):
        return "0" if v == 0 else (f"{v:.4f}" if v >= 1e-4 else f"{v:.0e}".replace("e-0", "e-"))
    def m(runs, col):      # the run values themselves: one value, or the range over the runs (no averaging)
        v = sorted(float(te.loc[r][col]) for r in runs)
        return f(v[0]) if f(v[0]) == f(v[-1]) else f"{f(v[0])}--{f(v[-1])}"
    rows = [[lab, m(runs, "zeroed"), m(runs, "rowsum")] for lab, runs in groups]
    write("tab_rb_e0b.tex", table("@{}lll@{}", ["Operator (range over the runs at that length)", "softmax mass set to zero", "row sum"], rows))
    # Table S47 with two header levels; the all-zero fraction only for 513-1024 keys (0 in every row for <= 512 keys,
    # asserted here; the table note says so)
    BINS = ("1-64", "65-256", "257-512", "513-1024")
    rows = []
    for lab, runs, step in (("IndexSoftmax UINT8, 100M, seed 7", ["rb-E3a-05"], None), ("IndexSoftmax UINT8, 100M, seed 42", ["rb-E3a-06"], None),
                            ("IndexSoftmax UINT8, 100M, seed 1337", ["rb-E3a-07"], None),
                            ("IndexSoftmax UINT8, 250M", ["rb-E3a-08"], None), ("IndexSoftmax UINT8, 2.5B", ["rb-E3a-2.5B-01"], None),
                            ("softmax control, seed 7$^{\\ast}$", ["rb-C0-01"], "cf"),
                            ("softmax control, seed 42$^{\\ast}$", ["rb-C0-02"], "cf"),
                            ("softmax control, seed 1337$^{\\ast}$", ["rb-C0-03"], "cf")):
        cells = [lab]
        for L in BINS:
            pre = "cf_e3a_u8_" if step == "cf" else ""
            z = np.mean([probe(r, f"{pre}len{L}_zeroed_softmax_mass") for r in runs])
            s_ = np.mean([probe(r, f"{pre}len{L}_rowsum") for r in runs])
            a0 = np.mean([probe(r, f"{pre}len{L}_allzero_row_frac") for r in runs])
            cells += [f"{z:.4f}", f"{s_:.4f}"]
            if L == "513-1024":
                cells.append("0" if a0 == 0 else f"{a0:.2e}".replace("e-0", "e-"))
            else:
                assert a0 == 0, (lab, L, a0)
        rows.append(cells)
    head = ["\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.1cm}*{9}{>{\\raggedleft\\arraybackslash}p{0.95cm}}@{}}", "\\toprule",
            " & \\multicolumn{2}{c}{1--64 keys} & \\multicolumn{2}{c}{65--256 keys} & \\multicolumn{2}{c}{257--512 keys} & \\multicolumn{3}{c@{}}{513--1024 keys} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-10}",
            " & mass set to zero & row sum & mass set to zero & row sum & mass set to zero & row sum & mass set to zero & row sum & all-zero rows \\\\",
            "\\midrule"]
    body = head + [" & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
    write("tab_rb_e0b_len.tex", "\n".join(body) + "\n")
    put("E0b.u8.25.long", f"{probe('rb-E3a-2.5B-01', 'len513-1024_zeroed_softmax_mass'):.3f}", "probe E3a-2.5B len513-1024 zeroed")
    az = PRB[(PRB.run_id == "rb-E3a-2.5B-01") & (PRB.metric == "len513-1024_allzero_row_frac")].sort_values("tokens_m")
    pk = az.loc[az.value.idxmax()]
    put("E0b.allzero.max", f"{100 * pk.value:.1f}", "probes_layer_mean E3a-2.5B len513-1024_allzero_row_frac, max over checkpoints (percent)")
    put("E0b.allzero.maxB", f"{pk.tokens_m / 1000:.1f}", "tokens (B) of that checkpoint")
    put("E0b.allzero.end", f"{100 * az.value.iloc[-1]:.1f}", "probes_layer_mean E3a-2.5B len513-1024_allzero_row_frac at the endpoint (percent)")


# ------------------------------------------------------------------ I.9 second seed at 2.5B
def tab_rb_e8():
    comps = ["rb-E8c-01 − rb-E8a-01 (LERP K16 − softmax, seed 7)", "LERP K16 − softmax, seed 1337 (paper)",
             "rb-E8a-01 − softmax s1337 (seed 7 vs seed 1337)"]
    write("tab_rb_e8_down.tex", table("@{}l*{3}{>{\\raggedright\\arraybackslash}p{3.2cm}}@{}", ["Configuration", "LERP ($K{=}16$)\\enspace$-$\\enspace softmax, seed 7",
                                                      "LERP ($K{=}16$)\\enspace$-$\\enspace softmax, seed 1337", "softmax, seed 7\\enspace$-$\\enspace softmax, seed 1337"],
                                       downstream_rows(comps, None)))
    a = cmp("E8a", run="rb-E8a-01")
    put("E8a.vs1337", sgn(a.dnll, 4), "comparisons E8a primary"); put("E8a.vs1337.ci", civ(a.ci_lo, a.ci_hi, 4), "")
    put("E8.ss.c4.ci", civ(ds(DSH, comps[2], "c4_validation").ci_lo, ds(DSH, comps[2], "c4_validation").ci_hi, 4), "")
    r = ds(DSH, comps[0], "wikitext103_test"); put("E8c.held.wt", sgn(r["diff"], 4), "downstream_heldout E8c-E8a wikitext")
    r = ds(DSH, comps[1], "wikitext103_test"); put("E8p.held.wt", sgn(r["diff"], 4), "downstream_heldout paper LERP16 wikitext")
    put("E8.blimp.s7.3", sgn(ds(DSB, comps[0], "blimp", "acc_macro")["diff"], 3), "downstream_benchmarks (verdict precision)")
    put("E8.blimp.s7.3ci", civ(ds(DSB, comps[0], "blimp", "acc_macro").ci_lo, ds(DSB, comps[0], "blimp", "acc_macro").ci_hi, 3), "")
    r = ds(DSB, comps[1], "blimp", "acc_macro")   # seed 1337 at the same precision, for the I.9 BLiMP sentence
    put("E8.blimp.s1337.3", sgn(r["diff"], 3), "downstream_benchmarks LERP K16 − softmax seed 1337 blimp (3 decimals)")
    put("E8.blimp.s1337.3ci", civ(r.ci_lo, r.ci_hi, 3), "")


# ------------------------------------------------------------------ I.10 cost
def tab_rb_cost():
    e0a = rd("e0a_accuracy")
    LAB = {"IndexSoftmax E3a-ii (E3a-2.5B)": "IndexSoftmax, UINT8", "FWM-LERP K4 (E1b)": "\\FW--LERP K4", "LERP K4": "MinMax--LERP K4",
           "LERP K16": "MinMax--LERP K16", "LERP K32": "MinMax--LERP K32", "FWM-Prob K4": "\\FW--Prob K4",
           "FWM-Weight K4": "\\FW--Weight K4", "FWM-Weight K16": "\\FW--Weight K16"}
    rows = [[LAB[x.condition], f"{sgn(x.dnll_vs_softmax, 4)} {civ(x.ci_lo, x.ci_hi, 4)}"] for _, x in e0a.iterrows()]
    write("tab_rb_e0a_acc.tex", table("@{}ll@{}", ["Operator (124M @ 2.5B, seed 1337)", "$\\dNLL$ vs softmax [95\\% CI]"], rows))
    b = rd("e0c_bitwidth")
    g = b[b.scheme.str.startswith("K-interval")]
    rows = [[x.scheme.replace("K-interval grid, K=", ""), int(x.levels), int(x.index_bits)] for _, x in g.iterrows()]
    write("tab_rb_e0c.tex", table("@{}rrr@{}", ["$K$", "levels", "index bits"], rows))
    r = rd("e0d_roofline")
    rows = [[tx(x.implementation), int(x.matmul), int(x.exp_unit), int(x.shared_memory), int(x.bound), f"{x.relative_to_fa4:.2f}$\\times$", tx(x.function)] for _, x in r.iterrows()]
    write("tab_rb_e0d.tex", table("@{}p{4.3cm}rrrrlp{2.6cm}@{}", ["Implementation (per 128$^3$ tile)", "matmul", "exp unit", "shared mem.", "bound", "vs FA4", "function"], rows))
    put("E0d.twopass", f"{r.relative_to_fa4.max():.1f}", "e0d_roofline max relative_to_fa4")


# ------------------------------------------------------------------ I.11 evaluation settings, TF32 off
def tab_rb_e6():
    e6 = rd("e6")
    eps = ["A_FW_P4", "A_FW_W4", "A_FW_W16", "A_MM_W4", "D_FW_W4_s7", "D_MM_W4_s7"]
    lab = {"A_FW_P4": "A \\FW--Prob $K{=}4$", "A_FW_W4": "A \\FW--Weight $K{=}4$", "A_FW_W16": "A \\FW--Weight $K{=}16$",
           "A_MM_W4": "A MinMax--Weight $K{=}4$", "D_FW_W4_s7": "C.2 \\FW--Weight $K{=}4$", "D_MM_W4_s7": "C.2 MinMax--Weight $K{=}4$"}
    sets = ["tf32_on", "batch8", "scores_bf16", "direct", "twopass128", "online128"]
    rows = []
    for ep in eps:
        x = e6[e6.endpoint == ep].set_index("setting")
        cells = [lab[ep], uns(x.loc["ref"].nll_5090, 4)]
        for s_ in sets:
            cells.append("---" if s_ not in x.index else sgn(x.loc[s_].dnll_vs_ref, 4))
        cells.append(f"{abs(x.dnll_3090_minus_5090).max():.4f}")
        rows.append(cells)
    write("tab_rb_e6.tex", "\\setlength{\\tabcolsep}{4pt}" + table("@{}llrrrrrrr@{}", ["Endpoint", "ref NLL", "TF32 on", "batch 8", "bf16 scores", "direct", "two-pass", "single-pass", "3090 $|\\Delta|$"], rows))
    put("E6.maxabs", f"{abs(e6.dnll_vs_ref).max():.4f}", "e6 max |dnll_vs_ref|")
    put("E6.max3090", f"{abs(e6.dnll_3090_minus_5090).max():.4f}", "e6 max |3090-5090|")
    r = cmp("E7", run="rb-E7-01"); put("E7.fwm", sgn(r.dnll, 4), "comparisons rb-E7-01"); put("E7.fwm.ci", civ(r.ci_lo, r.ci_hi, 4), "")
    r = cmp("E7", run="rb-E7-02"); put("E7.mm", sgn(r.dnll, 4), "comparisons rb-E7-02 (mean only)")


# ------------------------------------------------------------------ C.1 hardware rows and compute sentence
def numbers_misc():
    cp = rd("compute_per_experiment")
    rr = RUNS.reset_index()
    for mch in ("A", "B"):
        g = rr[rr.machine == mch]
        put(f"I.machine{mch}.runs", str(len(g)), "runs.csv machine count")
        put(f"I.machine{mch}.gpuh", uns(g.train_gpu_hours.sum(), 1), "runs.csv train_gpu_hours by machine")


def tables_appendix():
    tab_rb_overview(); tab_rb_e1(); tab_rb_e1b_down(); tab_rb_e11(); tab_rb_e2ab(); tab_rb_e2d()
    tab_rb_e4(); tab_rb_e5(); tab_rb_e3b(); tab_rb_e3a(); tab_rb_e0b(); tab_rb_e8(); tab_rb_cost(); tab_rb_e6(); numbers_misc()


# ------------------------------------------------------------------ I.3: step-0 check of the frozen-step parents
def numbers_step0():
    s = pd.read_csv(STEP0)
    def g(section, parent, item):
        r = s[(s.section == section) & (s.parent == parent) & (s["item"] == item)]
        assert len(r) == 1, (section, parent, item, len(r))
        return float(r.value.iloc[0])
    src = "e2c_step0_check.csv"
    for p in ("A250", "B50", "B125"):
        put(f"S0.{p}.diff", sgn(g("step3", p, "diff_b_minus_a"), 3), f"{src} step3 {p} diff_b_minus_a (frozen-step minus MinMax forward)")
        put(f"S0.{p}.ci", civ(g("step3", p, "ci_lo"), g("step3", p, "ci_hi"), 3), f"{src} step3 {p} ci_lo, ci_hi")
    tps = int(float(rd("constants").set_index("name").loc["tokens per step", "value"]))
    nsteps = 5_000_000 // tps                     # optimizer steps in the first 5M-token logging window after the branch
    assert f"step{nsteps:02d}_train_loss_frozen_minus_WtoW" in set(s["item"])
    put("S0.nsteps", str(nsteps), f"optimizer steps in the first 5M tokens after the branch point (5e6 // {tps}, constants.csv)")
    put("S0.tokM", f"{nsteps * tps / 1e6:.0f}", f"{nsteps} steps x {tps} tokens per step (constants.csv), in millions, rounded")
    put("S0.A250.gnratio", f"{g('step1', 'A250', 'step01_grad_norm_frozen') / g('step1', 'A250', 'step01_grad_norm_WtoW'):.0f}",
        f"{src} step1 A250: step01 grad norm frozen / W->W (ratio, rounded)")
    for p in ("A250", "B125"):
        put(f"S0.{p}.dl1", sgn(g("step1", p, "step01_train_loss_frozen_minus_WtoW"), 2), f"{src} step1 {p} step01 train-loss excess")
        put(f"S0.{p}.dlN", sgn(g("step1", p, f"step{nsteps:02d}_train_loss_frozen_minus_WtoW"), 2), f"{src} step1 {p} step{nsteps} train-loss excess")
    for p in ("A250", "B125", "B50"):
        h = [g("step4", p, f"layer{l:02d}_h0") for l in range(12)]
        put(f"S0.{p}.h0", f"{np.mean(h):.1f}", f"{src} step4 {p}: mean over the 12 layers of h0 (from the branch config)")


# ------------------------------------------------------------------ output
def write(name, body):
    (OUT / name).write_text(body)
    print("wrote", name)


def write_numbers():
    lines = ["% generated by ../tables/make_rb_tables.py from the dossier csv/ -- do not edit"]
    for k, v in NUM.items():
        lines.append(f"\\rbnset{{{k}}}{{{v}}}% {SOURCE[k]}")
    write("rb_numbers.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    numbers_main()
    tab_calgrad()
    tables_appendix()
    numbers_step0()
    write_numbers()
