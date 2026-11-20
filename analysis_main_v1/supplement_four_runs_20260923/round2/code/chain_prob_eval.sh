#!/usr/bin/env bash
R2=<REPO>/analysis_main_v1/supplement_four_runs_20260923/round2
until grep -q DL_DONE $R2/model_only_prob/VERIFY.txt 2>/dev/null; do sleep 30; done
[ "$(grep -c ': OK' $R2/model_only_prob/VERIFY.txt)" = "4" ] || { echo "download verification failed"; cat $R2/model_only_prob/VERIFY.txt; exit 1; }
cd $R2/code && <LOCAL_HOME>/miniconda3/envs/bta/bin/python prob_native_vs_strict.py
