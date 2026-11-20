#!/usr/bin/env bash
S=<REPO>/analysis_main_v1/supplement_four_runs_20260923
until grep -q "END sp_s7_softmax_fp8e4m3exp_100m" $S/round3/logs/queue.log; do sleep 60; done
grep -q "END sp_s7_softmax_bf16exp_100m rc=0" $S/round3/logs/queue.log && grep -q "END sp_s7_softmax_fp8e4m3exp_100m rc=0" $S/round3/logs/queue.log || { echo "a round-3 run failed"; exit 1; }
cd $S/code/eval && SUP_LOCAL=1 SUP_SEED=7 <LOCAL_HOME>/miniconda3/envs/bta/bin/python sup_eval.py --jobs eval_jobs_rtx3090_r3.tsv --out $S/eval/rtx3090 --root $S/code/r3_lowprec_softmax
echo "R3 EVAL DONE $(date -u +%FT%TZ)"
