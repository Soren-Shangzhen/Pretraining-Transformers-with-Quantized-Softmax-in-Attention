#!/usr/bin/env bash
S=<REPO>/analysis_main_v1/supplement_four_runs_20260923
until grep -q "LOCAL EVAL DONE" $S/logs/local3090_eval.log 2>/dev/null; do sleep 60; done
cd $S/code/eval && SUP_LOCAL=1 <LOCAL_HOME>/miniconda3/envs/bta/bin/python zero_sum_ckpt.py
echo "ZERO SUM DONE $(date -u +%FT%TZ)"
