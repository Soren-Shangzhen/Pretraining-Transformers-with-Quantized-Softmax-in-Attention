#!/usr/bin/env python3
"""Layer 3: standard benchmarks through lm-eval-harness with a custom adapter (§5.3).

The adapter subclasses the harness LM base class; loglikelihood() loads the run's own frozen operator through
common_loader and scores continuations at batch 1, fp32 attention, TF32 off, context 1024, GPT-2 BPE.
Over-long examples use the harness default left truncation; the truncation rate is recorded per task.

Per-request rows are written for every task so aggregate statistics can be recomputed (§7.1).

Usage:
  layer3_harness.py --tasks lambada_openai --conditions all --mode native
  layer3_harness.py --sanity          # Exact only, LAMBADA + BLiMP subset, checks the expected ranges
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_loader import ROOT, load, runs, set_numerics  # noqa: E402

from lm_eval.api.instance import Instance  # noqa: E402,F401
from lm_eval.api.model import LM  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402
from lm_eval import simple_evaluate  # noqa: E402

OUT = f"{ROOT}/analysis_main_v1/downstream_v1/results/layer3"
BLOCK = 1024
BLIMP_LIMIT = 200          # per paradigm; deterministic prefix, documented deviation for compute budget
TASKS = ["lambada_openai", "blimp", "sciq", "piqa", "arc_easy"]
EXACT_TASKS = ["lambada_openai"]


@register_model("frozenop")
class FrozenOperatorLM(LM):
    """lm-eval adapter: one loaded run, its own training operator, batch 1."""

    def __init__(self, run=None, forward_mode="native", **kwargs):
        super().__init__()
        self.enc = tiktoken.get_encoding("gpt2")
        self.loaded = load(run, forward_mode)
        self.model = self.loaded.model
        self.records = []
        self.truncated = 0
        self.n_requests = 0

    @property
    def eot_token_id(self):
        return self.enc.eot_token

    @property
    def max_length(self):
        return BLOCK

    @property
    def max_gen_toks(self):
        return 64

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string):
        return self.enc.encode_ordinary(string)

    def tok_decode(self, tokens):
        return self.enc.decode(tokens)

    @torch.inference_mode()
    def _score(self, context, continuation):
        ctx = self.tok_encode(context) if context else [self.eot_token_id]
        cont = self.tok_encode(continuation)
        ids = ctx + cont
        truncated = len(ids) > BLOCK + 1
        if truncated:                                   # harness default: left truncation
            ids = ids[-(BLOCK + 1):]
        n_cont = min(len(cont), len(ids) - 1)
        x = torch.tensor(ids[:-1], dtype=torch.long, device="cuda").unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = self.model(x)
        lg = logits.float()[0]
        lp = torch.log_softmax(lg, dim=-1)
        tgt = y[-n_cont:]
        tok_lp = lp[-n_cont:].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        greedy = bool((lg[-n_cont:].argmax(-1) == tgt).all().item())
        return float(tok_lp.sum().item()), greedy, n_cont, truncated

    def loglikelihood(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            context, continuation = req.args
            ll, greedy, n_cont, trunc = self._score(context, continuation)
            self.n_requests += 1
            self.truncated += int(trunc)
            self.records.append(dict(task=req.task_name, doc_id=req.doc_id, idx=req.idx,
                                     n_cont_tokens=n_cont, truncated=int(trunc),
                                     raw_loglik=ll, norm_loglik=ll / max(n_cont, 1), greedy=int(greedy)))
            out.append((ll, greedy))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            (string,) = req.args
            ids = self.tok_encode(string)
            total = 0.0
            for i in range(0, len(ids), BLOCK):
                chunk = ids[i:i + BLOCK]
                if len(chunk) < 2:
                    continue
                ll, _, _, _ = self._score("", self.tok_decode(chunk))
                total += ll
            out.append(total)
        return out

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("layer 3 uses only loglikelihood tasks")


def run_task(r, mode, task, limit=None):
    tag = f'{r["run_name"]}__{task}__{mode}'
    res_path = f"{OUT}/raw/{tag}.json"
    if os.path.exists(res_path):
        return json.load(open(res_path))
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    set_numerics()
    lm = FrozenOperatorLM(run=r, forward_mode=mode)
    t0 = time.time()
    kwargs = dict(model=lm, tasks=[task], batch_size=1, verbosity="ERROR")
    if limit:
        kwargs["limit"] = limit
    res = simple_evaluate(**kwargs)
    elapsed = time.time() - t0
    payload = dict(run=r["run_name"], condition=r["cond"], K=r["K"], forward_mode=mode, task=task,
                   results=res["results"], n_requests=lm.n_requests, truncated=lm.truncated,
                   truncation_rate=lm.truncated / max(lm.n_requests, 1), seconds=round(elapsed, 1),
                   limit=limit)
    json.dump(payload, open(res_path, "w"), indent=2, default=str)
    with gzip.open(f"{OUT}/raw/{tag}_requests.csv.gz", "wt", newline="") as f:
        f.write("task,doc_id,idx,n_cont_tokens,truncated,raw_loglik,norm_loglik,greedy\n")
        for x in lm.records:
            f.write(f'{x["task"]},{x["doc_id"]},{x["idx"]},{x["n_cont_tokens"]},{x["truncated"]},'
                    f'{x["raw_loglik"]:.6f},{x["norm_loglik"]:.6f},{x["greedy"]}\n')
    del lm
    torch.cuda.empty_cache()
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="lambada_openai")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--mode", default="native")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    R = {r["cond"]: r for r in runs()}
    os.makedirs(OUT, exist_ok=True)

    if args.sanity:
        r = R["Exact"]
        lam = run_task(r, "native", "lambada_openai")
        acc = lam["results"]["lambada_openai"].get("acc,none")
        print(f"LAMBADA Exact acc = {acc:.4f} (expected 0.30-0.38), truncation "
              f"{lam['truncation_rate']:.4f}")
        bl = run_task(r, "native", "blimp", limit=BLIMP_LIMIT)
        key = [k for k in bl["results"] if k == "blimp"] or list(bl["results"])[:1]
        bacc = bl["results"][key[0]].get("acc,none")
        print(f"BLiMP Exact acc = {bacc:.4f} (expected 0.75-0.82)")
        ok = (0.28 <= acc <= 0.40) and (0.72 <= bacc <= 0.85)
        print("SANITY", "PASS" if ok else "FAIL")
        json.dump(dict(lambada_acc=acc, blimp_acc=bacc, passed=bool(ok)),
                  open(f"{OUT}/sanity_layer3.json", "w"), indent=2)
        sys.exit(0 if ok else 1)

    conds = list(R) if args.conditions == "all" else args.conditions.split(",")
    for task in args.tasks.split(","):
        limit = BLIMP_LIMIT if task == "blimp" else None
        for cond in conds:
            p = run_task(R[cond], args.mode, task, limit)
            main_metric = list(p["results"].values())[0]
            print(f'  {cond:18s} {args.mode:6s} {task:16s} '
                  f'{ {k: round(v, 4) for k, v in main_metric.items() if isinstance(v, float)} } '
                  f'trunc={p["truncation_rate"]:.4f} ({p["seconds"]:.0f}s)', flush=True)


if __name__ == "__main__":
    main()
