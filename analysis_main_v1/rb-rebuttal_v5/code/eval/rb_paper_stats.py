#!/usr/bin/env python3
"""The paper's downstream statistics, applied to the runs of Appendix I.  Nothing is re-implemented: the resampling is
the paper's own code.
  held-out corpora (layer 1): stats.paired on per-block NLL, kind "clusters" with the document ids of the paper's
      C4 blocks (all blocks of a document resample together), kind "blocks" (circular moving blocks, L = 16) for
      WikiText-103 and PTB, exactly as layer1_ppl_transfer.summarize chooses; the paper's per-block NLL and C4
      document ids come from its results/layer1/per_block_rows.csv.gz.
  benchmarks (stage C): layer3_stageC_stats weight_vectors / diff_vector / bootstrap / point (4000 resamples, seed
      20260920; BLiMP resampled within each paradigm and macro-averaged; LAMBADA token-weighted NLL as a ratio of sums).
check() recomputes the paper's own seed-1337 rows with these functions and compares them with its tables."""
import csv, gzip, os, sys
import numpy as np
import pandas

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.abspath(os.path.join(HERE, "..", ".."))
AM = os.path.abspath(os.path.join(RB, ".."))
EV = os.path.join(RB, "runs", "_eval")
DS_P = os.path.join(AM, "downstream_v1", "results")
sys.path.insert(0, os.path.join(AM, "downstream_v1", "code"))
import stats as st  # noqa: E402
import layer3_stageC_stats as S3  # noqa: E402

CORPORA = ("wikitext103_test", "ptb_test", "c4_validation")
PRIM = [("blimp", "acc_macro"), ("hellaswag", "acc_norm"), ("piqa", "acc_norm"), ("arc_easy", "acc_norm"),
        ("lambada_openai", "acc"), ("lambada_openai", "target_nll_corpus_token_weighted"), ("sciq_default", "acc"),
        ("sciq_nosupport", "acc")]
_ROWS = None


def _rows():
    global _ROWS
    if _ROWS is None:
        _ROWS = {}
        with gzip.open(os.path.join(DS_P, "layer1", "per_block_rows.csv.gz"), "rt") as f:
            for r in csv.DictReader(f):
                _ROWS.setdefault((r["run_id"], r["forward_mode"], r["corpus"]), []).append(r)
    return _ROWS


def paper_pb(run_id, corpus, mode="native"):
    rs = sorted(_rows()[(run_id, mode, corpus)], key=lambda r: int(r["block_id"]))
    return np.array([float(r["nll"]) for r in rs])


def clusters(corpus):
    """The paper's C4 document ids per block (None for corpora without document structure)."""
    rs = sorted(_rows()[("formal_124m_s1337_exact_2p5b", "native", corpus)], key=lambda r: int(r["block_id"]))
    ids = [r["document_id"] for r in rs]
    return None if all(x == "" for x in ids) else np.array(ids, dtype=object)


def heldout_paired(a, b, corpus):
    """a − b on one held-out corpus with the paper's layer 1 resampling. Returns (delta, lo, hi)."""
    cl = clusters(corpus)
    if cl is not None and len(cl) != len(a):
        raise ValueError(f"{corpus}: {len(a)} blocks vs {len(cl)} cluster ids")
    r = st.paired(a, b, kind="clusters" if cl is not None else "blocks", clusters=cl)
    return r["delta"], r["ci_lo"], r["ci_hi"]


def l1_pb(where, cond, rn, corpus):
    return np.load(os.path.join(EV, "downstream", "layer1", cond, "per_block", f"{rn}__{corpus}__native.npy"))


def stagec_dir(where, cond):
    return os.path.join(DS_P, "layer3", "stageC", cond) if where == "P" else os.path.join(EV, "downstream", "stageC", cond)


def load_one(d, cfg):
    """Per-document values of one condition, built as S3.load_config builds them (same columns, same ordering)."""
    df = pandas.read_csv(os.path.join(d, f"{cfg}.csv.gz"))
    if cfg == "lambada_openai":
        df = df.sort_values("doc_id").reset_index(drop=True)
        key = df[["subtask", "doc_id", "doc_hash", "n_scored_tokens"]]
        v = pandas.DataFrame(dict(acc=df.greedy_correct.astype(float), nll=df.target_nll.astype(float),
                                  ntok=df.n_scored_tokens.astype(float), nll_tok=df.token_norm_target_nll.astype(float)))
    else:
        df = df.sort_values(["subtask", "doc_id", "choice_idx"])
        g = df.groupby(["subtask", "doc_id"], sort=False)
        key = g[["doc_hash", "gold_idx"]].first().reset_index()
        v = pandas.DataFrame(dict(acc=g.acc_correct.first().astype(float).values,
                                  acc_norm=g.accnorm_correct.first().astype(float).values,
                                  margin=g.margin_token_norm.first().astype(float).values))
    return key.reset_index(drop=True), v.reset_index(drop=True)


def compare(cfg, metric, src_a, src_b):
    """a − b with the paper's stage C bootstrap.  Returns value a, value b, diff, lo, hi (pp for accuracies)."""
    ka, va = load_one(stagec_dir(*src_a), cfg)
    kb, vb = load_one(stagec_dir(*src_b), cfg)
    assert ka.doc_hash.equals(kb.doc_hash), f"{cfg}: document mismatch"
    if "gold_idx" in ka:
        assert ka.gold_idx.equals(kb.gold_idx), f"{cfg}: gold mismatch"
    if "n_scored_tokens" in ka:
        assert ka.n_scored_tokens.equals(kb.n_scored_tokens), f"{cfg}: target token count differs"
    w, _ = S3.weight_vectors(cfg, ka)
    lam = dict(ntok=vb.ntok.values.astype(np.float64)) if cfg == "lambada_openai" else None
    D = S3.diff_vector(cfg, metric, va, vb)[None, :].astype(np.float64)
    ratio = cfg == "lambada_openai" and metric == "target_nll_corpus_token_weighted"
    obs = float(D.sum() / lam["ntok"].sum()) if ratio else float((D @ w)[0])
    lo, hi, _ = S3.bootstrap(cfg, metric, ka, w, D, lam)
    sc = 100.0 if metric in S3.PCT else 1.0
    return S3.point(cfg, metric, va, w, ka) * sc, S3.point(cfg, metric, vb, w, kb) * sc, obs * sc, lo[0] * sc, hi[0] * sc


def value(where, cond, cfg, metric):
    """Point estimate of one condition (the paper's aggregation; pp for accuracies)."""
    k, v = load_one(stagec_dir(where, cond), cfg)
    w, _ = S3.weight_vectors(cfg, k)
    return S3.point(cfg, metric, v, w, k) * (100.0 if metric in S3.PCT else 1.0)


def cell(r, k):
    """'diff [lo, hi]' of a compare() / heldout_paired() result."""
    d = r[2:] if len(r) == 5 else r
    return f"{d[0]:+.{k}f} [{d[1]:+.{k}f}, {d[2]:+.{k}f}]"


def check():
    """Recompute the paper's seed-1337 rows for the conditions the summaries use; True if every number matches."""
    ok = True
    pt = {(r["run"], r["corpus"]): r for r in csv.DictReader(open(os.path.join(DS_P, "layer1", "ppl_transfer.csv"))) if r["forward_mode"] == "native"}
    ex = "formal_124m_s1337_exact_2p5b"
    runs = ["formal_124m_s1337_lerp_k16_fullgrad_2p5b", "formal_124m_s1337_lerp_k4_fullgrad_2p5b",
            "formal_124m_s1337_qrm_nearest_k4_tau6_fullgrad_2p5b", "formal_124m_s1337_qrm_nearest_k4_tau6_probste_2p5b",
            "formal_124m_s1337_qrm_nearest_k16_tau6_fullgrad_2p5b"]
    n1 = 0
    for run in runs:
        for c in CORPORA:
            d = heldout_paired(paper_pb(run, c), paper_pb(ex, c), c); p = pt[(run, c)]
            ok &= all(abs(x - float(p[k])) < 1e-12 for x, k in zip(d, ("delta_vs_exact", "ci_lo", "ci_hi"))); n1 += 1
    pc = {(r["cfg"], r["metric"], r["method"]): r for r in csv.DictReader(open(os.path.join(DS_P, "layer3", "stageC", "tables", "paired_comparisons.csv")))}
    n3 = 0
    for meth, cond in (("LERP K16", "LERP_K16"), ("LERP K4", "LERP_K4"), ("QRM-Prob K4", "QRMProb_K4"), ("QRM-Weight K4", "QRMWeight_K4"),
                       ("QRM-Weight K16", "QRMWeight_K16")):
        for cfg, metric in PRIM:
            r = compare(cfg, metric, ("P", cond), ("P", "Exact")); p = pc[(cfg, metric, meth)]
            ok &= all(abs(x - float(p[k])) < 1e-9 for x, k in zip(r[2:], ("diff", "ci_lo", "ci_hi"))); n3 += 1
    # the machines' held-out blocks are the paper's: the suite A softmax re-run on B equals the paper's per-block NLL
    same = all(np.array_equal(l1_pb("B", "Exact_s1337_check", ex, c), paper_pb(ex, c)) for c in CORPORA)
    return ok and same, f"{n1} held-out rows and {n3} benchmark rows of the paper recomputed: identical {ok}; B's re-run blocks = paper blocks: {same}"


if __name__ == "__main__":
    print(check())
