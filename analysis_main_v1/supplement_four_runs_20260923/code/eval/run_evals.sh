#!/usr/bin/env bash
# Evaluate every available checkpoint in eval_jobs.tsv, one process per code root (resumable).
S=<CLOUD_DISK>/supplement_four_runs_20260923
PY=/usr/local/miniconda3/envs/py312/bin/python
cd $S/code/eval
for root in $(cut -f2 eval_jobs.tsv | tail -n +2 | sort -u); do
  $PY sup_eval.py --jobs eval_jobs.tsv --out $S/eval --root "$root"
done
