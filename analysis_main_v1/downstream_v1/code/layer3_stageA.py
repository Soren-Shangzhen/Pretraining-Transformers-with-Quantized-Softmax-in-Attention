#!/usr/bin/env python3
"""Stage A: Exact-only baselines and power screen on nine tasks (§1.7).

No condition-vs-Exact comparison is computed here. For each task we record, from Exact's own per-example
scores: accuracy, length-normalised accuracy, the continuous main metric (margin or target NLL) with mean/SD,
the number of evaluated examples, the chance baseline from the actual option count, single-model SEs, the
truncation rate, and the conservative paired detection floor 2*SD/sqrt(n).

SciQ is run twice: the harness default template (support passage included) and a no-support variant.
BLiMP uses limit=200 per paradigm (~13,400 pairs), recorded as a subset everywhere.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_loader import ROOT, runs, set_numerics  # noqa: E402
import layer3_harness as H  # noqa: E402  (registers the adapter)

from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.tasks import TaskManager  # noqa: E402

OUT = f"{ROOT}/analysis_main_v1/downstream_v1/results/layer3/stageA"
TASKDIR = f"{ROOT}/analysis_main_v1/downstream_v1/code/tasks"
BLIMP_LIMIT = 200
TASKS = ["lambada_openai", "blimp", "sciq", "sciq_nosupport", "piqa", "arc_easy", "openbookqa",
         "hellaswag", "winogrande", "arc_challenge"]


def write_task_yaml():
    """SciQ without the support passage: same dataset and choices, support removed from the input."""
    os.makedirs(TASKDIR, exist_ok=True)
    with open(f"{TASKDIR}/sciq_nosupport.yaml", "w") as f:
        f.write("""task: sciq_nosupport
dataset_path: allenai/sciq
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
doc_to_text: "Question: {{question}}\\nAnswer:"
doc_to_target: 3
doc_to_choice: "{{[distractor1, distractor2, distractor3, correct_answer]}}"
should_decontaminate: true
doc_to_decontamination_query: "{{question}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
""")


def per_example(task, samples):
    """Return (metric_name, per-example continuous metric, acc, acc_norm, n_options, truncation-relevant info)."""
    cont, acc, accn, nopt = [], [], [], []
    for d in samples:
        resps = [r[0] for r in d["filtered_resps"]]           # summed log-likelihood per option
        if task == "lambada_openai":
            ll, greedy = d["filtered_resps"][0][0], d["filtered_resps"][0][1]
            ntok = max(len(d.get("arguments", [{}])), 1)
            cont.append(-ll)                                   # target NLL (continuous main metric)
            acc.append(float(bool(greedy)))
            accn.append(float(bool(greedy)))
            nopt.append(1)
            continue
        gold = d["target"] if isinstance(d["target"], int) else int(d.get("gold", 0))
        lens = [max(len(str(c)), 1) for c in (d.get("doc", {}).get("choices", {}) or {}).get("text", [])] \
            if isinstance((d.get("doc", {}) or {}).get("choices", None), dict) else []
        arr = np.asarray(resps, dtype=float)
        norm = np.asarray([r[0] / max(len(str(a[1])), 1) for r, a in zip(d["filtered_resps"], d["arguments"])],
                          dtype=float) if "arguments" in d else arr
        best_wrong = np.max(np.delete(norm, gold)) if len(norm) > 1 else -np.inf
        cont.append(float(norm[gold] - best_wrong))            # length-normalised margin
        acc.append(float(int(np.argmax(arr) == gold)))
        accn.append(float(int(np.argmax(norm) == gold)))
        nopt.append(len(arr))
    return np.asarray(cont), np.asarray(acc), np.asarray(accn), np.asarray(nopt)


def main():
    set_numerics()
    os.makedirs(OUT, exist_ok=True)
    write_task_yaml()
    tm = TaskManager(include_path=TASKDIR)
    r = {x["cond"]: x for x in runs()}["Exact"]
    rows = []
    for task in TASKS:
        raw = f"{OUT}/{task}_samples.json.gz"
        if os.path.exists(f"{OUT}/{task}_summary.json"):
            rows.append(json.load(open(f"{OUT}/{task}_summary.json")))
            print("  (cached)", task, flush=True)
            continue
        lm = H.FrozenOperatorLM(run=r, forward_mode="native")
        kwargs = dict(model=lm, tasks=[task], batch_size=1, verbosity="ERROR", log_samples=True,
                      task_manager=tm)
        if task == "blimp":
            kwargs["limit"] = BLIMP_LIMIT
        res = simple_evaluate(**kwargs)
        samples = res["samples"]
        allsamp = [s for v in samples.values() for s in v]
        cont, acc, accn, nopt = per_example(task, allsamp)
        n = len(cont)
        p = float(acc.mean())
        chance = float(np.mean(1.0 / nopt)) if task != "lambada_openai" else 0.0
        sd = float(cont.std(ddof=1))
        se_cont = sd / np.sqrt(n)
        summary = dict(
            task=task, subset=("limit=200 per paradigm (~13,400 pairs of 67,000)" if task == "blimp" else "full"),
            n_examples=int(n), n_requests=lm.n_requests, truncation_rate=lm.truncated / max(lm.n_requests, 1),
            metric_name=("target NLL (nats)" if task == "lambada_openai" else
                         "grammatical − ungrammatical log-likelihood margin" if task == "blimp" else
                         "length-normalised margin (gold − best wrong)"),
            acc=p, acc_norm=float(accn.mean()), chance=chance,
            acc_se=float(np.sqrt(p * (1 - p) / n)), acc_above_chance_pp=float((p - chance) * 100),
            metric_mean=float(cont.mean()), metric_sd=sd, metric_se=float(se_cont),
            detect_floor_metric=float(2 * se_cont),
            detect_floor_acc_pp=float(2 * np.sqrt(p * (1 - p) / n) * 100),
            mean_options=float(np.mean(nopt)), seconds=None)
        json.dump(summary, open(f"{OUT}/{task}_summary.json", "w"), indent=2)
        with gzip.open(raw, "wt") as f:
            json.dump([dict(doc_id=s.get("doc_id"), metric=float(c), acc=float(a), acc_norm=float(an))
                       for s, c, a, an in zip(allsamp, cont, acc, accn)], f)
        rows.append(summary)
        print(f'  {task:16s} n={n:6d} acc={p:.4f} (chance {chance:.3f}) metric={cont.mean():+.4f} '
              f'sd={sd:.3f} floor={2*se_cont:.4f} trunc={summary["truncation_rate"]:.4f}', flush=True)
        del lm
    json.dump(rows, open(f"{OUT}/stageA_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
