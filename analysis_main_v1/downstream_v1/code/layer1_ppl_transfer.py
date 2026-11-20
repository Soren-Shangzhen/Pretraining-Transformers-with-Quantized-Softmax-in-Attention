#!/usr/bin/env python3
"""Layer 1: zero-shot PPL transfer (§3).

15 runs x 3 held-out corpora x {native, exact} forward. Per-block NLL on non-overlapping 1024-token blocks,
batch 1, bf16 autocast outside the fp32 attention kernel, each run's own frozen code root.

Outputs (results/layer1/):
  per_block/<run>__<corpus>__<mode>.npy   raw per-block NLL
  per_block_rows.csv.gz                   per-example (per-block) rows for recomputation (§7.1)
  ppl_transfer.csv                        condition x corpus x mode summary with paired CI, BH q, equivalence
  noise_floor.csv                         §2.5 descriptive split-half noise scale (Exact run)
  rank_consistency.csv                    Spearman of the per-corpus delta ordering vs FineWebEdu
"""
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_loader import ROOT, block_nll, load, runs, set_numerics, sha256  # noqa: E402
import stats as st  # noqa: E402

DATA = f"{ROOT}/analysis_main_v1/downstream_v1/data/layer1"
OUT = f"{ROOT}/analysis_main_v1/downstream_v1/results/layer1"
PB = f"{OUT}/per_block"
CORPORA = ["wikitext103_test", "ptb_test", "c4_validation"]
MODES = ["native", "exact"]


def corpus_tokens(name):
    return np.memmap(f"{DATA}/{name}.bin", dtype=np.uint16, mode="r")


def clusters_for(name, n_blocks):
    f = f"{DATA}/{name}_blocks.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d.document_id.values[:n_blocks]


def evaluate():
    set_numerics()
    os.makedirs(PB, exist_ok=True)
    R = runs()
    toks = {c: corpus_tokens(c) for c in CORPORA}
    for r in R:
        for mode in MODES:
            need = [c for c in CORPORA if not os.path.exists(f"{PB}/{r['run_name']}__{c}__{mode}.npy")]
            if not need:
                continue
            t0 = time.time()
            ld = load(r, mode)
            for c in need:
                nll = block_nll(ld.model, toks[c])
                np.save(f"{PB}/{r['run_name']}__{c}__{mode}.npy", nll)
                print(f"  {r['cond']:18s} {mode:6s} {c:18s} n={len(nll):4d} nll={nll.mean():.5f}", flush=True)
            del ld
            print(f"  ({r['cond']} {mode} done in {time.time()-t0:.0f}s)", flush=True)


def summarize():
    R = {r["cond"]: r for r in runs()}
    rows, raw_rows = [], []
    for corpus in CORPORA:
        cl_all = None
        for mode in MODES:
            data = {}
            for cond, r in R.items():
                f = f"{PB}/{r['run_name']}__{corpus}__{mode}.npy"
                if os.path.exists(f):
                    data[cond] = np.load(f)
            if "Exact" not in data:
                continue
            base = data["Exact"]
            cl = clusters_for(corpus, len(base))
            cl_all = cl
            kind = "clusters" if cl is not None else "blocks"
            recs = []
            for cond, nll in data.items():
                for i, v in enumerate(nll):
                    raw_rows.append((R[cond]["run_name"], cond, R[cond]["K"], mode, corpus, i,
                                     "" if cl is None else cl[i], float(v)))
                rec = dict(run=R[cond]["run_name"], condition=cond, K=R[cond]["K"], corpus=corpus,
                           forward_mode=mode, n_blocks=len(nll), nll=float(nll.mean()),
                           ppl=float(np.exp(nll.mean())))
                if cond == "Exact":
                    rec.update(delta_vs_exact=0.0, ci_lo=0.0, ci_hi=0.0, p_value=1.0, detectable=False)
                else:
                    p = st.paired(nll, base, kind=kind, clusters=cl)
                    rec.update(delta_vs_exact=p["delta"], ci_lo=p["ci_lo"], ci_hi=p["ci_hi"],
                               p_value=p["p_value"], detectable=p["detectable"])
                recs.append(rec)
            fam = [r for r in recs if r["condition"] != "Exact"]
            qs = st.bh([r["p_value"] for r in fam])
            for r_, q in zip(fam, qs):
                r_["q_value"] = float(q)
            for r_ in recs:
                r_.setdefault("q_value", 1.0)
                r_["equivalence"] = st.equivalence(r_["ci_lo"], r_["ci_hi"])
                r_["within_equivalence_margin"] = r_["equivalence"] == "within pre-declared equivalence margin"
            rows.extend(recs)
        # §2.5 noise floor from the Exact run on this corpus (native forward)
        ex = np.load(f"{PB}/{R['Exact']['run_name']}__{corpus}__native.npy")
        rows_nf.append(dict(corpus=corpus, metric="block NLL", n=len(ex),
                            noise_floor_p90=st.split_half_noise(ex, clusters=cl_all)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/ppl_transfer.csv", index=False)
    pd.DataFrame(rows_nf).to_csv(f"{OUT}/noise_floor.csv", index=False)
    with gzip.open(f"{OUT}/per_block_rows.csv.gz", "wt", newline="") as f:
        f.write("run_id,condition,K,forward_mode,corpus,block_id,document_id,nll\n")
        for t in raw_rows:
            f.write(",".join(str(x) for x in t) + "\n")

    # rank consistency vs FineWebEdu (main-result endpoint table)
    fw = pd.read_csv(f"{ROOT}/analysis_main_v1/A_longhorizon_124m_2p5b/endpoint_fullval/endpoint_table.csv")
    fw = fw.set_index("cond").delta_vs_exact
    rc = []
    for corpus in CORPORA:
        for mode in MODES:
            sub = df[(df.corpus == corpus) & (df.forward_mode == mode) & (df.condition != "Exact")]
            if not len(sub):
                continue
            common = [c for c in sub.condition if c in fw.index]
            rho, p = spearmanr(sub.set_index("condition").loc[common].delta_vs_exact.values, fw.loc[common].values)
            rc.append(dict(corpus=corpus, forward_mode=mode, n_conditions=len(common), spearman_rho=rho,
                           p_value=p))
    pd.DataFrame(rc).to_csv(f"{OUT}/rank_consistency.csv", index=False)
    json.dump(dict(corpora={c: json.load(open(f"{DATA}/{c}_meta.json")) for c in CORPORA},
                   manifest_sha256=sha256(f"{ROOT}/analysis_main_v1/manifests/primary_37_runs.tsv")),
              open(f"{OUT}/layer1_meta.json", "w"), indent=2)
    print(df[df.forward_mode == "native"][["condition", "corpus", "nll", "delta_vs_exact", "ci_lo", "ci_hi",
                                           "detectable", "within_equivalence_margin"]].to_string(index=False))


rows_nf = []

if __name__ == "__main__":
    if "--summarize-only" not in sys.argv:
        evaluate()
    summarize()
