#!/usr/bin/env bash
# Wait for the local E1 queue to finish, then run the local 3090 evaluations.
S=<REPO>/analysis_main_v1/supplement_four_runs_20260923
until grep -q "END sp_s7_lerp_k32_maxonly_100m" $S/logs/e1_local3090_queue.log; do sleep 60; done
grep -q "END sp_s7_lerp_k32_maxonly_100m rc=0" $S/logs/e1_local3090_queue.log || { echo "E1 max_only failed; not evaluating"; exit 1; }
bash $S/code/eval/run_evals_rtx3090.sh
echo "LOCAL EVAL DONE $(date -u +%FT%TZ)"
