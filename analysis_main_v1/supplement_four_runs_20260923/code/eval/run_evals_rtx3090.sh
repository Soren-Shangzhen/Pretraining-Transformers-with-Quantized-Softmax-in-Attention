#!/usr/bin/env bash
# Local RTX 3090 evaluations (same evaluator as the historical five-seed controls).
S=<REPO>/analysis_main_v1/supplement_four_runs_20260923
PY=<LOCAL_HOME>/miniconda3/envs/bta/bin/python
cd $S/code/eval
for root in $(cut -f2 eval_jobs_rtx3090.tsv | tail -n +2 | sort -u); do
  SUP_LOCAL=1 $PY sup_eval.py --jobs eval_jobs_rtx3090.tsv --out $S/eval/rtx3090 --root "$root"
done
