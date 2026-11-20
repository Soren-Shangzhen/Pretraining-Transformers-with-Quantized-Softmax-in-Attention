#!/usr/bin/env python3
"""Stage C statistics: paired comparisons of 14 methods against Exact on seven evaluation configurations.

Everything is computed from the per-item files written by layer3_stageC.py; the model is never re-run.

Direction is always method − Exact. For accuracy and margin a positive value means the method is better; for
NLL a positive value means the method is worse. Accuracy differences are reported in percentage points.

Resampling
  Bootstrap: 4,000 paired resamples, 95% percentile interval, seed 20260920. Method and Exact are always
  resampled with the *same* document draw; the draw is shared by all 14 methods within a configuration so the
  intervals are mutually comparable. Resampling is implemented as multinomial document counts, which is
  identical to drawing an index vector with replacement for every statistic used here.
  BLiMP resamples sentence pairs *within* each of the 67 paradigms and then takes the paradigm macro-average,
  so the interval is conditional on this fixed paradigm set.
  LAMBADA's corpus token-weighted NLL is a ratio of sums: each replicate redraws documents once and recomputes
  both ratios and their difference, never an average of per-document ratios.

  Permutation: two-sided paired label exchange, 20,000 Monte-Carlo draws, seed 20260921. Each draw swaps the
  method and Exact records of a document, keeping the aggregation structure (BLiMP paradigm macro-average,
  LAMBADA ratio of sums, SciQ per-document support gain). p = (1 + #{|T*| >= |T_obs|}) / (1 + B), so p is
  never reported as zero. The exchangeability assumption is over evaluation documents, not training seeds.

  BH: applied separately within each (configuration x metric) family of 14 comparisons; the SciQ
  difference-in-differences forms its own families, one per scoring metric. Raw p and BH q are both reported;
  q < 0.05 is the threshold inside a family. The 95% bootstrap intervals are NOT multiplicity-adjusted.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SC_OUT = f"{BASE}/results/layer3/stageC"
TAB = f"{BASE}/results/layer3/stageC/tables"
N_BOOT, BOOT_SEED = 4000, 20260920
N_PERM, PERM_SEED = 20000, 20260921
CHUNK = 500
EXACT = "Exact"

# display metrics, fixed before any comparison was computed
METRICS = {
    "blimp": [("acc_macro", "paradigm macro-average of standard accuracy", True),
              ("margin_macro", "paradigm macro-average of logP(good)/N_good - logP(bad)/N_bad", True)],
    "hellaswag": [("acc_norm", "standard acc_norm (primary)", True), ("acc", "standard acc", True),
                  ("margin", "token-normalised gold minus best-wrong margin", True)],
    "piqa": [("acc_norm", "standard acc_norm (primary)", True), ("acc", "standard acc", True),
             ("margin", "token-normalised gold minus best-wrong margin", True)],
    "arc_easy": [("acc_norm", "standard acc_norm (primary)", True), ("acc", "standard acc", True),
                 ("margin", "token-normalised gold minus best-wrong margin", True)],
    "sciq_default": [("acc", "standard acc (primary)", True), ("acc_norm", "standard acc_norm", True),
                     ("margin", "token-normalised gold minus best-wrong margin", True)],
    "sciq_nosupport": [("acc", "standard acc (primary)", True), ("acc_norm", "standard acc_norm", True),
                       ("margin", "token-normalised gold minus best-wrong margin", True)],
    "lambada_openai": [("acc", "standard harness acc", True),
                       ("target_nll_corpus_token_weighted", "sum(NLL) / sum(target tokens) (primary)", False),
                       ("target_nll_example_mean", "mean per-example total target NLL", False),
                       ("target_nll_token_norm_example_mean", "mean per-example NLL / target tokens", False)],
}
PCT = {"acc", "acc_norm", "acc_macro"}          # reported in percentage points


# ----------------------------------------------------------------------------- loading

def conditions():
    m = json.load(open(f"{SC_OUT}/manifest_stageC.json"))
    return [c["condition"] for c in m["conditions"]], m


def slug(c):
    return c.replace(" ", "_").replace("-", "")


def load_config(cfg, conds):
    """Per-document values for every condition, aligned on (subtask, doc_id)."""
    per, meta = {}, None
    for c in conds:
        df = pd.read_csv(f"{SC_OUT}/{slug(c)}/{cfg}.csv.gz")
        if cfg == "lambada_openai":
            df = df.sort_values("doc_id").reset_index(drop=True)
            key = df[["subtask", "doc_id", "doc_hash", "n_scored_tokens"]]
            v = pd.DataFrame(dict(acc=df.greedy_correct.astype(float),
                                  nll=df.target_nll.astype(float),
                                  ntok=df.n_scored_tokens.astype(float),
                                  nll_tok=df.token_norm_target_nll.astype(float)))
        else:
            df = df.sort_values(["subtask", "doc_id", "choice_idx"])
            g = df.groupby(["subtask", "doc_id"], sort=False)
            key = g[["doc_hash", "gold_idx"]].first().reset_index()
            v = pd.DataFrame(dict(acc=g.acc_correct.first().astype(float).values,
                                  acc_norm=g.accnorm_correct.first().astype(float).values,
                                  margin=g.margin_token_norm.first().astype(float).values))
        if meta is None:
            meta = key.reset_index(drop=True)
        else:
            assert meta.doc_hash.equals(key.reset_index(drop=True).doc_hash), f"{cfg}/{c}: document mismatch"
            if "gold_idx" in meta:
                assert meta.gold_idx.equals(key.reset_index(drop=True).gold_idx), f"{cfg}/{c}: gold mismatch"
            if "n_scored_tokens" in meta:
                assert meta.n_scored_tokens.equals(key.reset_index(drop=True).n_scored_tokens), \
                    f"{cfg}/{c}: target token count differs between conditions"
        per[c] = v.reset_index(drop=True)
    return meta, per


# ----------------------------------------------------------------------------- weights

def weight_vectors(cfg, meta):
    """Per-document weights w with sum(w * d) == the configuration's aggregation of d."""
    n = len(meta)
    if cfg == "blimp":
        sub = meta.subtask.values
        counts = pd.Series(sub).map(pd.Series(sub).value_counts()).values.astype(float)
        k = meta.subtask.nunique()
        return np.ones(n) / (counts * k), k
    return np.ones(n) / n, None


def point(cfg, metric, v, w, meta):
    if cfg == "lambada_openai":
        if metric == "acc":
            return float(v.acc.mean())
        if metric == "target_nll_corpus_token_weighted":
            return float(v.nll.sum() / v.ntok.sum())
        if metric == "target_nll_example_mean":
            return float(v.nll.mean())
        return float(v.nll_tok.mean())
    col = dict(acc_macro="acc", margin_macro="margin").get(metric, metric)
    return float(np.sum(w * v[col].values))


def diff_vector(cfg, metric, va, vb):
    """Per-document difference (method - Exact) whose weighted sum is the statistic difference."""
    if cfg == "lambada_openai":
        if metric == "acc":
            return va.acc.values - vb.acc.values
        if metric == "target_nll_example_mean" or metric == "target_nll_corpus_token_weighted":
            return va.nll.values - vb.nll.values
        return va.nll_tok.values - vb.nll_tok.values
    col = dict(acc_macro="acc", margin_macro="margin").get(metric, metric)
    return va[col].values - vb[col].values


# ----------------------------------------------------------------------------- resampling

def multinomial_counts(rng, cfg, meta, size):
    """size x n document counts; BLiMP resamples within each paradigm."""
    n = len(meta)
    if cfg != "blimp":
        return rng.multinomial(n, np.full(n, 1.0 / n), size=size).astype(np.float32)
    out = np.zeros((size, n), dtype=np.float32)
    for s, idx in meta.groupby("subtask", sort=False).indices.items():
        k = len(idx)
        out[:, idx] = rng.multinomial(k, np.full(k, 1.0 / k), size=size).astype(np.float32)
    return out


def bootstrap(cfg, metric, meta, w, D, lam=None):
    """Percentile CI of the weighted difference for every method at once. D is (n_methods, n_docs)."""
    rng = np.random.default_rng(BOOT_SEED)
    reps = np.empty((D.shape[0], N_BOOT), dtype=np.float64)
    ratio = cfg == "lambada_openai" and metric == "target_nll_corpus_token_weighted"
    done = 0
    while done < N_BOOT:
        c = min(CHUNK, N_BOOT - done)
        W = multinomial_counts(rng, cfg, meta, c)                     # (c, n)
        if ratio:
            # ratio of sums, recomputed on the redrawn documents: (sum W*NLL_method - sum W*NLL_Exact) / sum W*N
            den = W @ lam["ntok"]                                     # (c,)
            reps[:, done:done + c] = (D @ W.T) / den[None, :]
        else:
            reps[:, done:done + c] = D @ (W * w[None, :]).T
        done += c
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=1)
    return lo, hi, reps.mean(axis=1)


def permutation(cfg, metric, meta, w, D, obs, lam=None):
    """Two-sided paired label-exchange p-values for every method at once."""
    rng = np.random.default_rng(PERM_SEED)
    ratio = cfg == "lambada_openai" and metric == "target_nll_corpus_token_weighted"
    # ratio of sums: the denominator is the same for both conditions, so a label swap only flips the
    # numerator difference; the exchange statistic is sum(eps_i * d_i) / sum(N_i)
    wv = np.full(len(meta), 1.0 / lam["ntok"].sum()) if ratio else w
    Dw = D * wv[None, :]
    ge = np.zeros(D.shape[0], dtype=np.int64)
    a = np.abs(obs)[:, None]
    done = 0
    while done < N_PERM:
        c = min(CHUNK, N_PERM - done)
        E = rng.integers(0, 2, size=(c, len(meta))).astype(np.float32) * 2 - 1
        T = Dw @ E.T                                                  # (n_methods, c)
        ge += (np.abs(T) >= a - 1e-15).sum(axis=1)
        done += c
    return (1.0 + ge) / (1.0 + N_PERM)


def bh(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = o[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        q[i] = prev
    return q


# ----------------------------------------------------------------------------- driver

def analyse_config(cfg, conds):
    meta, per = load_config(cfg, conds)
    w, n_sub = weight_vectors(cfg, meta)
    methods = [c for c in conds if c != EXACT]
    lam = dict(ntok=per[EXACT].ntok.values.astype(np.float64)) if cfg == "lambada_openai" else None
    pts, comps = [], []
    for metric, desc, hib in METRICS[cfg]:
        for c in conds:
            pts.append(dict(cfg=cfg, condition=c, metric=metric, description=desc,
                            higher_is_better=hib, value=point(cfg, metric, per[c], w, meta),
                            n_documents=len(meta), n_subtasks=n_sub))
        D = np.vstack([diff_vector(cfg, metric, per[m], per[EXACT]) for m in methods]).astype(np.float64)
        if cfg == "lambada_openai" and metric == "target_nll_corpus_token_weighted":
            tot = lam["ntok"].sum()
            obs = D.sum(axis=1) / tot
            direct = np.array([per[m].nll.sum() / tot - per[EXACT].nll.sum() / tot for m in methods])
            assert np.abs(obs - direct).max() < 1e-12, "ratio-of-sums mismatch"
        else:
            obs = D @ w
        lo, hi, _ = bootstrap(cfg, metric, meta, w, D, lam)
        p = permutation(cfg, metric, meta, w, D, obs, lam)
        q = bh(p)
        ex = per[EXACT]
        base = dict(acc_macro="acc", margin_macro="margin").get(metric, metric)
        for i, m in enumerate(methods):
            xa = diff_vector(cfg, metric, per[m], per[EXACT])
            if cfg == "lambada_openai":
                col = dict(acc="acc", target_nll_corpus_token_weighted="nll",
                           target_nll_example_mean="nll", target_nll_token_norm_example_mean="nll_tok")[metric]
                e = ex[col].values
                a_ = per[m][col].values
            else:
                e, a_ = ex[base].values, per[m][base].values
            sd_e = float(np.std(e, ddof=1))
            scale = 100.0 if metric in PCT else 1.0
            comps.append(dict(
                cfg=cfg, metric=metric, method=m, higher_is_better=hib,
                unit="percentage points" if metric in PCT else "nats",
                exact_value=point(cfg, metric, ex, w, meta) * scale,
                method_value=point(cfg, metric, per[m], w, meta) * scale,
                diff=obs[i] * scale, ci_lo=lo[i] * scale, ci_hi=hi[i] * scale,
                ci_excludes_zero=bool(lo[i] > 0 or hi[i] < 0),
                p_raw=float(p[i]), q_bh=float(q[i]), significant_q05=bool(q[i] < 0.05),
                sd_paired_diff=float(np.std(xa, ddof=1)), sd_exact=sd_e,
                shrinkage=float(np.std(xa, ddof=1) / sd_e) if sd_e > 0 else float("nan"),
                rho=float(np.corrcoef(a_, e)[0, 1]) if sd_e > 0 else float("nan"),
                n_documents=len(meta), n_boot=N_BOOT, n_perm=N_PERM))
    return pd.DataFrame(pts), pd.DataFrame(comps), meta, per


def sciq_did(conds, per_def, per_nos, meta_def):
    """Difference-in-differences on the support arm, per scoring metric, its own BH family."""
    methods = [c for c in conds if c != EXACT]
    rows, fam = [], {}
    for metric in ["acc", "acc_norm"]:
        gain_ex = per_def[EXACT][metric].values - per_nos[EXACT][metric].values
        D = np.vstack([(per_def[m][metric].values - per_nos[m][metric].values) - gain_ex for m in methods])
        w = np.full(len(meta_def), 1.0 / len(meta_def))
        obs = D @ w
        lo, hi, _ = bootstrap("sciq_did", metric, meta_def, w, D)
        p = permutation("sciq_did", metric, meta_def, w, D, obs)
        q = bh(p)
        fam[metric] = len(methods)
        for i, m in enumerate(methods):
            rows.append(dict(
                metric=metric, method=m, unit="percentage points",
                exact_support=per_def[EXACT][metric].mean() * 100,
                exact_nosupport=per_nos[EXACT][metric].mean() * 100,
                exact_support_gain=gain_ex.mean() * 100,
                method_support=per_def[m][metric].mean() * 100,
                method_nosupport=per_nos[m][metric].mean() * 100,
                method_support_gain=(per_def[m][metric].values - per_nos[m][metric].values).mean() * 100,
                did=obs[i] * 100, ci_lo=lo[i] * 100, ci_hi=hi[i] * 100,
                ci_excludes_zero=bool(lo[i] > 0 or hi[i] < 0),
                p_raw=float(p[i]), q_bh=float(q[i]), significant_q05=bool(q[i] < 0.05),
                n_documents=len(meta_def), n_boot=N_BOOT, n_perm=N_PERM))
    return pd.DataFrame(rows), fam


def blimp_subtasks(conds):
    out = []
    for c in conds:
        df = pd.read_csv(f"{SC_OUT}/{slug(c)}/blimp.csv.gz")
        g = df.groupby(["subtask", "doc_id"], sort=False)[["acc_correct", "margin_token_norm"]].first()
        s = g.groupby("subtask").mean().reset_index()
        s["condition"] = c
        out.append(s)
    d = pd.concat(out, ignore_index=True)
    return d.rename(columns=dict(acc_correct="acc", margin_token_norm="margin"))


def main():
    os.makedirs(TAB, exist_ok=True)
    conds, man = conditions()
    assert conds[0] == EXACT and len(conds) == 15, conds
    all_pts, all_cmp, per_store, meta_store = [], [], {}, {}
    for cfg in METRICS:
        print(f"--- {cfg} ---", flush=True)
        pts, cmp_, meta, per = analyse_config(cfg, conds)
        all_pts.append(pts)
        all_cmp.append(cmp_)
        per_store[cfg], meta_store[cfg] = per, meta
        print(cmp_[["metric", "method", "diff", "ci_lo", "ci_hi", "p_raw", "q_bh"]].round(5).to_string(index=False),
              flush=True)
    pts = pd.concat(all_pts, ignore_index=True)
    cmp_ = pd.concat(all_cmp, ignore_index=True)
    pts.to_csv(f"{TAB}/point_estimates.csv", index=False)
    cmp_.to_csv(f"{TAB}/paired_comparisons.csv", index=False)

    did, fam = sciq_did(conds, per_store["sciq_default"], per_store["sciq_nosupport"],
                        meta_store["sciq_default"])
    did.to_csv(f"{TAB}/sciq_difference_in_differences.csv", index=False)
    print("--- sciq DiD ---")
    print(did[["metric", "method", "method_support_gain", "did", "ci_lo", "ci_hi", "p_raw", "q_bh"]]
          .round(4).to_string(index=False), flush=True)

    bl = blimp_subtasks(conds)
    bl.to_csv(f"{TAB}/blimp_subtasks.csv", index=False)

    json.dump(dict(n_boot=N_BOOT, boot_seed=BOOT_SEED, n_perm=N_PERM, perm_seed=PERM_SEED,
                   conditions=conds, configs=list(METRICS),
                   bh_families={f"{c}:{m}": 14 for c in METRICS for m, _, _ in METRICS[c]},
                   sciq_did_families={f"sciq_did:{k}": v for k, v in fam.items()},
                   n_documents={c: int(len(meta_store[c])) for c in METRICS},
                   blimp_n_paradigms=int(bl.subtask.nunique()),
                   note="95% bootstrap intervals are not multiplicity-adjusted; BH q values are per "
                        "(configuration x metric) family of 14 comparisons. Single training seed per "
                        "condition: all intervals and tests reflect evaluation-sample uncertainty only."),
              open(f"{TAB}/stats_meta.json", "w"), indent=2)
    print("\nwrote", TAB)


if __name__ == "__main__":
    main()
