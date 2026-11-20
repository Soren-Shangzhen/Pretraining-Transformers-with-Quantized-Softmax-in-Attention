#!/usr/bin/env python3
"""Render the Stage C result tables as markdown for REPORT.md. Pure formatting, no statistics."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
TAB = f"{BASE}/results/layer3/stageC/tables"
ORDER = ["Exact", "LERP K1", "LERP K2", "LERP K4", "LERP K16", "LERP K32",
         "MinMax-Weight K4", "MinMax-Weight K16", "MinMax-Weight K32", "MinMax-Weight K64",
         "MinMax-Prob K4", "QRM-Weight K2", "QRM-Weight K4", "QRM-Weight K16", "QRM-Prob K4"]
CFGS = ["blimp", "hellaswag", "piqa", "arc_easy", "lambada_openai", "sciq_default", "sciq_nosupport"]
NICE = {"blimp": "BLiMP (subset)", "hellaswag": "HellaSwag", "piqa": "PIQA", "arc_easy": "ARC-Easy",
        "lambada_openai": "LAMBADA", "sciq_default": "SciQ default", "sciq_nosupport": "SciQ no-support"}
PRIMARY = {"blimp": "acc_macro", "hellaswag": "acc_norm", "piqa": "acc_norm", "arc_easy": "acc_norm",
           "lambada_openai": "acc", "sciq_default": "acc", "sciq_nosupport": "acc"}


def md_table(df, cols, headers, fmt):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---" if i == 0 else "---:" for i in range(len(headers))]) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            cells.append(f(v) if f else (f"{v}" if not isinstance(v, float) else f"{v:.4f}"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def sig(r):
    s = ""
    if r.ci_excludes_zero:
        s += "*"
    if r.significant_q05:
        s += "†"
    return s or "—"


def main():
    pts = pd.read_csv(f"{TAB}/point_estimates.csv")
    cmp_ = pd.read_csv(f"{TAB}/paired_comparisons.csv")
    did = pd.read_csv(f"{TAB}/sciq_difference_in_differences.csv")
    bl = pd.read_csv(f"{TAB}/blimp_subtasks.csv")
    meta = json.load(open(f"{TAB}/stats_meta.json"))
    L = []

    # 1. standard metrics, seven configurations x 15 conditions
    L.append("#### Standard metrics, seven configurations × 15 conditions\n")
    piv = {}
    for cfg in CFGS:
        for m in ["acc_macro", "acc", "acc_norm"]:
            sub = pts[(pts.cfg == cfg) & (pts.metric == m)]
            if len(sub):
                piv[f"{NICE[cfg]} {m}"] = sub.set_index("condition").value * 100
    d = pd.DataFrame(piv).reindex(ORDER)
    cols = [c for c in d.columns]
    hdr = ["Condition"] + [c.replace(" acc_macro", " acc (macro)").replace(" acc_norm", " acc_norm")
                           for c in cols]
    L.append("All values are percent. BLiMP is the paradigm macro-average over "
             f'{meta["blimp_n_paradigms"]} paradigms of the limit=200 subset.\n')
    rows = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] + ["---:"] * len(cols)) + "|"]
    for c in ORDER:
        rows.append("| " + c + " | " + " | ".join(f"{d.loc[c, x]:.2f}" for x in cols) + " |")
    L.append("\n".join(rows) + "\n")

    # 2. supplementary continuous metrics
    L.append("#### Supplementary continuous metrics (nats)\n")
    piv = {}
    for cfg in CFGS:
        for m in ["margin_macro", "margin"]:
            sub = pts[(pts.cfg == cfg) & (pts.metric == m)]
            if len(sub):
                piv[f"{NICE[cfg]} margin"] = sub.set_index("condition").value
    for m in ["target_nll_corpus_token_weighted", "target_nll_example_mean",
              "target_nll_token_norm_example_mean"]:
        sub = pts[(pts.cfg == "lambada_openai") & (pts.metric == m)]
        piv["LAMBADA " + {"target_nll_corpus_token_weighted": "NLL corpus/token",
                          "target_nll_example_mean": "NLL per example",
                          "target_nll_token_norm_example_mean": "NLL token-norm mean"}[m]] = \
            sub.set_index("condition").value
    d = pd.DataFrame(piv).reindex(ORDER)
    cols = list(d.columns)
    rows = ["| Condition | " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * len(cols)) + "|"]
    for c in ORDER:
        rows.append("| " + c + " | " + " | ".join(f"{d.loc[c, x]:.4f}" for x in cols) + " |")
    L.append("\n".join(rows) + "\n")

    # 3. paired comparisons
    L.append("#### Paired comparisons, method − Exact\n")
    L.append("`*` the 95% paired bootstrap interval excludes zero (not multiplicity-adjusted); "
             "`†` BH q < 0.05 inside the (configuration × metric) family of 14 comparisons.\n")
    for cfg in CFGS:
        for m, desc, hib in [(x.metric, x.description, x.higher_is_better) for _, x in
                             pts[pts.cfg == cfg].drop_duplicates("metric").iterrows()]:
            sub = cmp_[(cmp_.cfg == cfg) & (cmp_.metric == m)].set_index("method").reindex(ORDER[1:]).reset_index()
            unit = sub.unit.iloc[0]
            star = " — **primary display metric**" if m == PRIMARY[cfg] else ""
            L.append(f'\n**{NICE[cfg]} — {m}** ({desc}; {unit}; '
                     f'{"higher is better" if hib else "positive = worse"}){star}\n')
            L.append(md_table(sub.assign(flag=[sig(r) for _, r in sub.iterrows()]),
                              ["method", "method_value", "diff", "ci_lo", "ci_hi", "p_raw", "q_bh", "flag"],
                              ["Method", "Value", "\u0394 vs Exact", "CI low", "CI high", "p", "q (BH)", ""],
                              {"method_value": lambda v: f"{v:.4f}", "diff": lambda v: f"{v:+.4f}",
                               "ci_lo": lambda v: f"{v:+.4f}", "ci_hi": lambda v: f"{v:+.4f}",
                               "p_raw": lambda v: f"{v:.4f}", "q_bh": lambda v: f"{v:.4f}",
                               "flag": lambda v: v}))

    # 4. SciQ difference in differences
    L.append("\n#### SciQ support difference-in-differences\n")
    for m in ["acc", "acc_norm"]:
        sub = did[did.metric == m].set_index("method").reindex(ORDER[1:]).reset_index()
        e = did[did.metric == m].iloc[0]
        L.append(f'\n**Scored with {m}.** Exact: support {e.exact_support:.2f}%, no-support '
                 f'{e.exact_nosupport:.2f}%, support gain {e.exact_support_gain:+.2f} pp. '
                 f'Negative DiD means the method gains less from the supporting passage than Exact does.\n')
        L.append(md_table(sub.assign(flag=[sig(r) for _, r in sub.iterrows()]),
                          ["method", "method_support", "method_nosupport", "method_support_gain",
                           "did", "ci_lo", "ci_hi", "p_raw", "q_bh", "flag"],
                          ["Method", "support %", "no-support %", "gain pp", "DiD pp", "CI low", "CI high",
                           "p", "q (BH)", ""],
                          {"method_support": lambda v: f"{v:.2f}", "method_nosupport": lambda v: f"{v:.2f}",
                           "method_support_gain": lambda v: f"{v:+.2f}", "did": lambda v: f"{v:+.2f}",
                           "ci_lo": lambda v: f"{v:+.2f}", "ci_hi": lambda v: f"{v:+.2f}",
                           "p_raw": lambda v: f"{v:.4f}", "q_bh": lambda v: f"{v:.4f}",
                           "flag": lambda v: v}))

    # 5. BLiMP paradigms
    L.append("\n#### BLiMP paradigms\n")
    p = bl.pivot_table(index="subtask", columns="condition", values="acc")[ORDER] * 100
    L.append(f"{len(p)} paradigms, 200 pairs each. Full table: `results/layer3/stageC/tables/"
             f"blimp_subtasks.csv`. Summary of each condition's paradigm distribution:\n")
    q = pd.DataFrame(dict(macro=p.mean(), min=p.min(), median=p.median(), max=p.max(),
                          n_below_exact=(p.lt(p["Exact"], axis=0)).sum())).reindex(ORDER)
    rows = ["| Condition | macro % | min | median | max | paradigms below Exact |",
            "|---|---:|---:|---:|---:|---:|"]
    for c in ORDER:
        r = q.loc[c]
        rows.append(f'| {c} | {r.macro:.2f} | {r["min"]:.1f} | {r["median"]:.1f} | {r["max"]:.1f} | '
                    f'{int(r.n_below_exact)} |')
    L.append("\n".join(rows) + "\n")

    out = "\n".join(L)
    open(f"{TAB}/report_section.md", "w").write(out)
    print(out[:3000])
    print(f"\n... wrote {TAB}/report_section.md ({len(out)} chars)")


if __name__ == "__main__":
    main()
