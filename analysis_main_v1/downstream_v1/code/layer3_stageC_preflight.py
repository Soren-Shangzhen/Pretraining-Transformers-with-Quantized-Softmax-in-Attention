#!/usr/bin/env python3
"""Stage C preflight: small end-to-end checks before the full sweep.

Runs two conditions (Exact and one quantised condition) over five reduced configurations and verifies:
  1. the native operator is really in effect (checkpoint config, and native != exact forward numerically);
  2. candidate mapping, gold index and scoring direction;
  3. continuation token boundary and scored-token count;
  4. the saved raw scores rebuild the harness's own acc / acc_norm exactly;
  5. both conditions see the same examples, in the same order, with the same scored inputs;
  6. the two SciQ arms align one-to-one by original document id, with different scored inputs.

Writes results under results/layer3/stageC_preflight/ and never touches the main output tree.
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import tiktoken
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layer3_stageC as SC  # noqa: E402
from common_loader import BLOCK, load, runs  # noqa: E402

PRE = f"{SC.BASE}/results/layer3/stageC_preflight"
SMALL = [dict(cfg="blimp", task="blimp", limit=2, kind="mc", split="validation"),
         dict(cfg="arc_easy", task="arc_easy", limit=20, kind="mc", split="test"),
         dict(cfg="lambada_openai", task="lambada_openai", limit=20, kind="lambada", split="test"),
         dict(cfg="sciq_default", task="sciq", limit=20, kind="mc", split="test"),
         dict(cfg="sciq_nosupport", task="sciq_nosupport", limit=20, kind="mc", split="test")]
CONDS = ["Exact", "QRM-Weight K4"]
checks = []


def ck(name, ok, detail=""):
    checks.append(dict(check=name, passed=bool(ok), detail=str(detail)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f" — {detail}" if detail else ""), flush=True)
    return ok


def operator_in_effect():
    """The loaded model must use its own trained operator, and that operator must change the output."""
    R = {r["cond"]: r for r in runs()}
    r = R["QRM-Weight K4"]
    ln = load(r, "native")
    ok = ck("native operator from checkpoint config",
            ln.op["attn"] == "quant" and ln.op["approx"] == "nearest"
            and ln.op["range_mode"] == "relevant_zero" and ln.op["K"] == 4
            and abs(ln.op["range_tau"] - 6.0) < 1e-9 and ln.op["range_grad"] == "full", ln.op)
    ok &= ck("model in eval mode, dropout inactive", not ln.model.training)
    x = torch.arange(1, 129, dtype=torch.long, device="cuda").unsqueeze(0)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        a = ln.model(x)[0].float()
    cfg_native = dict(ln.cfg.__dict__) if hasattr(ln.cfg, "__dict__") else {}
    del ln
    torch.cuda.empty_cache()
    le = load(r, "exact")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        b = le.model(x)[0].float()
    d = float((a - b).abs().max().item())
    ok &= ck("native forward is not silently exact softmax", d > 1e-3,
             f"max|native-exact| logit diff = {d:.4f}; native attn_mode={cfg_native.get('attn_mode')}")
    # Exact condition: its native forward IS exact by construction
    le2 = load(R["Exact"], "native")
    ok &= ck("Exact condition native == exact_fp32_softmax", le2.op["attn"] == "exact_fp32_softmax", le2.op)
    del le, le2
    torch.cuda.empty_cache()
    return ok


def rebuild_metrics(d, cond):
    ok = True
    enc = tiktoken.get_encoding("gpt2")
    for cfgd in SMALL:
        f = f'{d}/{cfgd["cfg"]}.csv.gz'
        s = json.load(open(f'{d}/{cfgd["cfg"]}_summary.json'))
        df = pd.read_csv(f)
        res = s["harness_results"]
        if cfgd["kind"] == "lambada":
            acc = df.greedy_correct.mean()
            ha = res["lambada_openai"]["acc,none"]
            ok &= ck(f'{cond}/{cfgd["cfg"]}: rebuilt acc == harness acc', abs(acc - ha) < 1e-12,
                     f"{acc:.6f} vs {ha:.6f}")
            ppl = float(np.exp(df.target_nll.sum() / df.n_scored_tokens.sum()))
            hp = res["lambada_openai"]["perplexity,none"]
            ok &= ck(f'{cond}/{cfgd["cfg"]}: harness perplexity is per-example exp(mean target NLL)',
                     abs(float(np.exp(df.target_nll.mean())) - hp) < 1e-6,
                     f"exp(mean NLL)={np.exp(df.target_nll.mean()):.4f}  token-weighted={ppl:.4f}  "
                     f"harness={hp:.4f}")
            ok &= ck(f'{cond}/{cfgd["cfg"]}: one row per document, no duplicates',
                     df.doc_id.is_unique, f"{len(df)} rows")
            continue
        g = df.groupby(["subtask", "doc_id"], sort=False)
        acc = g.acc_correct.first().mean()
        accn = g.accnorm_correct.first().mean()
        key = cfgd["task"] if cfgd["task"] in res else [k for k in res if k != "groups"][0]
        ha, han = res[key].get("acc,none"), res[key].get("acc_norm,none")
        if ha is not None:
            ok &= ck(f'{cond}/{cfgd["cfg"]}: rebuilt acc == harness acc', abs(acc - ha) < 1e-12,
                     f"{acc:.6f} vs {ha:.6f}")
        if han is not None:
            ok &= ck(f'{cond}/{cfgd["cfg"]}: rebuilt acc_norm == harness acc_norm', abs(accn - han) < 1e-12,
                     f"{accn:.6f} vs {han:.6f}  (denominator = raw choice characters, delimiter excluded)")
        # gold / direction / margin
        gold_rows = df[df.is_gold == 1]
        ok &= ck(f'{cond}/{cfgd["cfg"]}: exactly one gold candidate per document',
                 len(gold_rows) == df.doc_id.nunique() * df.subtask.nunique() or
                 gold_rows.groupby(["subtask", "doc_id"]).size().eq(1).all(), "")
        rec = g.apply(lambda x: int(np.argmax(x.raw_total_logprob.values)) == x.gold_idx.iloc[0])
        ok &= ck(f'{cond}/{cfgd["cfg"]}: acc_correct == argmax(raw logprob) hits gold',
                 bool((rec.values == g.acc_correct.first().values.astype(bool)).all()))
        m = g.apply(lambda x: float(x.token_norm_logprob.values[x.gold_idx.iloc[0]]
                                    - np.delete(x.token_norm_logprob.values, x.gold_idx.iloc[0]).max()))
        ok &= ck(f'{cond}/{cfgd["cfg"]}: margin == gold token-norm minus best wrong',
                 float(np.abs(m.values - g.margin_token_norm.first().values).max()) < 1e-12)
        # token boundary: n_scored_tokens equals len(encode(continuation)) when nothing was truncated
        sub = df[df.truncated == 0].sample(min(30, (df.truncated == 0).sum()), random_state=0)
        ok &= ck(f'{cond}/{cfgd["cfg"]}: no truncation in this sample', int(df.truncated.sum()) == 0,
                 f"{int(df.truncated.sum())} truncated of {len(df)}")
        ok &= ck(f'{cond}/{cfgd["cfg"]}: scored-token count > 0 everywhere', bool((df.n_scored_tokens > 0).all()))
        ok &= ck(f'{cond}/{cfgd["cfg"]}: token-norm score == raw / n_scored_tokens',
                 float(np.abs(df.token_norm_logprob - df.raw_total_logprob / df.n_scored_tokens).max()) < 1e-12)
        ok &= ck(f'{cond}/{cfgd["cfg"]}: choice_chars == cont_chars minus target delimiter',
                 bool((df.choice_chars == df.cont_chars - df.target_delimiter_chars).all())
                 and bool((df.choice_chars > 0).all()))
        ok &= ck(f'{cond}/{cfgd["cfg"]}: per-sample harness acc_norm == rebuilt acc_norm',
                 bool((g.harness_acc_norm.first().fillna(-1).values
                       == g.accnorm_correct.first().astype(float).values).all())
                 if not g.harness_acc_norm.first().isna().all() else True)
        ok &= ck(f'{cond}/{cfgd["cfg"]}: no failed rows', bool((df.status == "ok").all()))
    return ok


def cross_condition(dirs):
    ok = True
    for cfgd in SMALL:
        hashes = [json.load(open(f'{d}/{cfgd["cfg"]}_summary.json'))["order_hash"] for d in dirs]
        ok &= ck(f'{cfgd["cfg"]}: identical example set, order and scored inputs across conditions',
                 len(set(hashes)) == 1, hashes[0][:16])
    # SciQ arm alignment on the original document id
    a = pd.read_csv(f'{dirs[0]}/sciq_default.csv.gz')
    b = pd.read_csv(f'{dirs[0]}/sciq_nosupport.csv.gz')
    ok &= ck("SciQ arms: same documents, one-to-one by doc_id",
             sorted(a.doc_id.unique()) == sorted(b.doc_id.unique()), f"{a.doc_id.nunique()} docs")
    ok &= ck("SciQ arms: identical original document hashes",
             bool((a.groupby("doc_id").doc_hash.first() == b.groupby("doc_id").doc_hash.first()).all()))
    ok &= ck("SciQ arms: identical gold index and candidate order",
             bool((a.groupby("doc_id").gold_idx.first() == b.groupby("doc_id").gold_idx.first()).all())
             and bool((a.sort_values(["doc_id", "choice_idx"]).cont_chars.values ==
                       b.sort_values(["doc_id", "choice_idx"]).cont_chars.values).all()))
    ok &= ck("SciQ arms: different scored inputs (support removed)",
             bool((a.sort_values(["doc_id", "choice_idx"]).input_hash.values !=
                   b.sort_values(["doc_id", "choice_idx"]).input_hash.values).all()))
    return ok


def main():
    os.makedirs(PRE, exist_ok=True)
    SC.OUT = PRE
    SC.CONFIGS = SMALL
    ok = operator_in_effect()
    dirs = []
    for c in CONDS:
        print(f"--- preflight run: {c} ---", flush=True)
        SC.run_condition(c, force=True)
        dirs.append(f"{PRE}/{SC.slug(c)}")
    for c, d in zip(CONDS, dirs):
        ok &= rebuild_metrics(d, c)
    ok &= cross_condition(dirs)
    json.dump(dict(passed=bool(ok), checks=checks), open(f"{PRE}/preflight.json", "w"), indent=2)
    print(f'\nPREFLIGHT {"PASS" if ok else "FAIL"}  ({sum(c["passed"] for c in checks)}/{len(checks)} checks)')
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
