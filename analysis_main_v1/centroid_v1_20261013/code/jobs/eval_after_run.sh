#!/usr/bin/env bash
# Evaluate the five checkpoints (20..100M) of one finished centroid_v1 run with its own frozen root
# (full WikiText-103 validation, batch 1, local RTX 3090) -> eval/rtx3090/<run>/<tokens>m_<ckpt>/
set -uo pipefail
run=$1
TR=${KI_REPO:?set KI_REPO to the repository root}
PB=$TR/training_code/phaseB_base
C=$TR/analysis_main_v1/centroid_v1_20261013
PY=<LOCAL_HOME>/miniconda3/envs/bta/bin/python
ROOT=$C/code/centroid_root
JOBS=$C/code/eval/eval_jobs_$run.tsv
printf "label\teval_root\tcheckpoint\ttokens_m\trole\n" > "$JOBS"
for ck in "$PB/centroid_runs_20261013/$run"/ckpt_step*_*m.pt; do
  t=$(basename "$ck" | sed -E 's/.*_([0-9]+)\.[0-9]m\.pt/\1/')
  printf "%s\t%s\t%s\t%s\t%s\n" "$run" "$ROOT" "$ck" "$t" "centroid_v1 run, own frozen root, 3090 evaluator" >> "$JOBS"
done
echo "[$(date -u +%FT%TZ)] EVAL $run ($(($(wc -l < "$JOBS") - 1)) checkpoints)"
cd "$C/code/eval" && SUP_SEED=7 "$PY" cv1_eval.py --jobs "$JOBS" --out "$C/eval/rtx3090" --root "$ROOT"
echo "[$(date -u +%FT%TZ)] EVAL END $run rc=$?"
