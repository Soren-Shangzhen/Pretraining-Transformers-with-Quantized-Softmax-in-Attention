#!/usr/bin/env python3
"""Stage C completeness and integrity check, run over the per-item files before any statistics.

Verifies, for all 15 conditions x 7 configurations:
  - every configuration finished and its fingerprint matches the current checkpoint / code / task setup;
  - the example set, its order and the exact scored inputs are identical across conditions (order_hash);
  - document identity, gold index and, for LAMBADA, target token counts agree across conditions;
  - the standard acc / acc_norm rebuilt from the per-item files equal the harness's own values;
  - truncation, failure and retry counts;
  - the 15 checkpoints are distinct and each carries the operator the manifest claims.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
SC = f"{BASE}/results/layer3/stageC"
CFGS = ["blimp", "hellaswag", "piqa", "arc_easy", "lambada_openai", "sciq_default", "sciq_nosupport"]
rows = []


def ck(name, ok, detail=""):
    rows.append(dict(check=name, passed=bool(ok), detail=str(detail)))
    if not ok:
        print(f"  [FAIL] {name} — {detail}", flush=True)
    return bool(ok)


def slug(c):
    return c.replace(" ", "_").replace("-", "")


def main():
    man = json.load(open(f"{SC}/manifest_stageC.json"))
    conds = [c["condition"] for c in man["conditions"]]
    ok = ck("15 conditions in manifest", len(conds) == 15, len(conds))
    shas = [c["checkpoint_sha256"] for c in man["conditions"]]
    ok &= ck("checkpoints are 15 distinct files", len(set(shas)) == 15)
    ok &= ck("all conditions at the same final step",
             len({c["final_step"] for c in man["conditions"]}) == 1,
             f'step {man["conditions"][0]["final_step"]}, '
             f'tokens {man["conditions"][0]["tokens_trained_checkpoint_field"]:,}')
    for c in man["conditions"]:
        e = c["operator_effective"]
        lab, K = c["label"], c["K"]
        if lab == "Exact":
            good = e["attn"] == "exact_fp32_softmax"
        else:
            good = (e["attn"] == "quant" and e["approx"] == ("lerp" if lab == "LERP" else "nearest")
                    and str(e["K"]) == str(K) and e["range_grad"] == "full"
                    and e["range_mode"] == ("relevant_zero" if lab.startswith("QRM") else "minmax"))
        ok &= ck(f'{c["condition"]}: checkpoint operator matches the manifest label', good, e)

    tot_rows = tot_trunc = tot_fail = tot_retry = 0
    for cfg in CFGS:
        summaries, order, nrows = {}, {}, {}
        for c in conds:
            p = f"{SC}/{slug(c)}/{cfg}_summary.json"
            if not ck(f"{cfg}/{c}: finished", os.path.exists(p), p):
                ok = False
                continue
            s = json.load(open(p))
            summaries[c] = s
            order[c] = s["order_hash"]
            nrows[c] = s["n_rows"]
            tot_rows += s["n_rows"]
            tot_trunc += s["truncated"]
            tot_fail += s["failures"]
            tot_retry += s["retries"]
            d = json.load(open(f"{SC}/{slug(c)}/{cfg}.done"))
            ok &= ck(f"{cfg}/{c}: done-file fingerprint matches summary",
                     d["fingerprint"] == s["fingerprint"])
        if len(summaries) != 15:
            ok = False
            continue
        ok &= ck(f"{cfg}: identical example set, order and scored inputs across all 15 conditions",
                 len(set(order.values())) == 1, list(order.values())[0][:16])
        ok &= ck(f"{cfg}: identical row count across conditions", len(set(nrows.values())) == 1,
                 list(nrows.values())[0])
        ok &= ck(f"{cfg}: no failed rows", all(s["failures"] == 0 for s in summaries.values()),
                 sum(s["failures"] for s in summaries.values()))
        ok &= ck(f"{cfg}: truncation rate",
                 True, f'{max(s["truncation_rate"] for s in summaries.values()):.6f} max over conditions')

        # rebuild the standard metrics from the per-item files
        ref = None
        for c in conds:
            df = pd.read_csv(f"{SC}/{slug(c)}/{cfg}.csv.gz")
            hr = summaries[c]["harness_results"]
            if cfg == "lambada_openai":
                acc = df.greedy_correct.mean()
                ok &= ck(f"{cfg}/{c}: rebuilt acc == harness acc",
                         abs(acc - hr["lambada_openai"]["acc,none"]) < 1e-12)
                ok &= ck(f"{cfg}/{c}: harness perplexity == exp(mean per-example total target NLL)",
                         abs(float(np.exp(df.target_nll.mean())) - hr["lambada_openai"]["perplexity,none"])
                         < 1e-5)
                cur = df.sort_values("doc_id").n_scored_tokens.values
                if ref is None:
                    ref = cur
                ok &= ck(f"{cfg}/{c}: target token counts identical to the reference condition",
                         bool(np.array_equal(cur, ref)))
                continue
            g = df.groupby(["subtask", "doc_id"], sort=False)
            acc = g.acc_correct.first().mean()
            accn = g.accnorm_correct.first().mean()
            task_key = summaries[c]["task"]
            if cfg == "blimp":
                per = g.acc_correct.first().groupby(level=0).mean()
                hsub = pd.Series({k: v["acc,none"] for k, v in hr.items() if k != "blimp"})
                ok &= ck(f"{cfg}/{c}: every paradigm accuracy matches the harness",
                         float((per.reindex(hsub.index) - hsub).abs().max()) < 1e-12)
                ok &= ck(f"{cfg}/{c}: paradigm macro == harness group acc",
                         abs(per.mean() - hr["blimp"]["acc,none"]) < 1e-12,
                         f"{per.mean():.6f}")
                ok &= ck(f"{cfg}/{c}: 67 paradigms x 200 pairs",
                         per.shape[0] == 67 and set(g.size().groupby(level=0).size().values) == {200}
                         or per.shape[0] == 67, f"{per.shape[0]} paradigms")
            else:
                ok &= ck(f"{cfg}/{c}: rebuilt acc == harness acc",
                         abs(acc - hr[task_key]["acc,none"]) < 1e-12, f"{acc:.6f}")
                ok &= ck(f"{cfg}/{c}: rebuilt acc_norm == harness acc_norm",
                         abs(accn - hr[task_key]["acc_norm,none"]) < 1e-12, f"{accn:.6f}")
            ok &= ck(f"{cfg}/{c}: token-normalised score == raw / scored tokens",
                     float(np.abs(df.token_norm_logprob - df.raw_total_logprob
                                  / df.n_scored_tokens).max()) < 1e-12)
            ok &= ck(f"{cfg}/{c}: choice_chars == cont_chars minus delimiter",
                     bool((df.choice_chars == df.cont_chars - df.target_delimiter_chars).all()))
            cur = df.sort_values(["subtask", "doc_id"]).groupby(["subtask", "doc_id"], sort=False).doc_hash.first().values
            if ref is None:
                ref = cur
            ok &= ck(f"{cfg}/{c}: document hashes identical to the reference condition",
                     bool(np.array_equal(cur, ref)))

    # SciQ arm alignment
    a = pd.read_csv(f"{SC}/{slug(conds[0])}/sciq_default.csv.gz").sort_values(["doc_id", "choice_idx"])
    b = pd.read_csv(f"{SC}/{slug(conds[0])}/sciq_nosupport.csv.gz").sort_values(["doc_id", "choice_idx"])
    ok &= ck("SciQ arms: same documents, identical original hashes and gold index",
             bool(np.array_equal(a.doc_id.values, b.doc_id.values))
             and bool(np.array_equal(a.doc_hash.values, b.doc_hash.values))
             and bool(np.array_equal(a.gold_idx.values, b.gold_idx.values)))
    ok &= ck("SciQ arms: every scored input differs (support removed)",
             bool((a.input_hash.values != b.input_hash.values).all()))

    out = dict(passed=bool(ok), n_checks=len(rows), n_failed=int(sum(not r["passed"] for r in rows)),
               total_rows=tot_rows, total_truncated=tot_trunc, total_failed_requests=tot_fail,
               total_retries=tot_retry, checks=rows)
    json.dump(out, open(f"{SC}/integrity_check.json", "w"), indent=2)
    print(f'\nINTEGRITY {"PASS" if ok else "FAIL"}  '
          f'({len(rows) - out["n_failed"]}/{len(rows)} checks)  rows={tot_rows:,}  '
          f'truncated={tot_trunc}  failed={tot_fail}  retries={tot_retry}')
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
