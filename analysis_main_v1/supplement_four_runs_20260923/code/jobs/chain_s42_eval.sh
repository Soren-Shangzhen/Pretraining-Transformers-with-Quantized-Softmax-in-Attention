#!/usr/bin/env bash
# After the seed-42 E1 queue: 3090 evaluation of both runs (5 checkpoints each), then the real-loss zero-sum pass.
S=<REPO>/analysis_main_v1/supplement_four_runs_20260923
PY=<LOCAL_HOME>/miniconda3/envs/bta/bin/python
until grep -q "END sp_s42_lerp_k32_maxonly_100m" $S/logs/e1_local3090_s42_queue.log; do sleep 60; done
grep -q "END sp_s42_lerp_k32_maxonly_100m rc=0" $S/logs/e1_local3090_s42_queue.log || { echo "s42 max_only failed"; exit 1; }
grep -q "END sp_s42_lerp_k32_minonly_100m rc=0" $S/logs/e1_local3090_s42_queue.log || { echo "s42 min_only failed"; exit 1; }
cd $S/code/eval
SUP_LOCAL=1 SUP_SEED=42 $PY sup_eval.py --jobs eval_jobs_rtx3090_s42.tsv --out $S/eval/rtx3090 --root $S/code/e1_lerp_range_grad_single
echo "S42 EVAL DONE $(date -u +%FT%TZ)"
SUP_LOCAL=1 SEED=42 $PY zero_sum_ckpt.py
echo "S42 ZERO SUM DONE $(date -u +%FT%TZ)"
