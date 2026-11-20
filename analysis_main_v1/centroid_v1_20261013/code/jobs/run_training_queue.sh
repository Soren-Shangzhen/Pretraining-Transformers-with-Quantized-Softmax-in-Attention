#!/usr/bin/env bash
# centroid_v1 training queue: four seed-7 124M@100M WikiText-103 runs on the local RTX 3090, same recipe as the
# seed-7 suite-D controls and the round-3 baselines.  Resumable: a finished run (100M checkpoint present) is
# skipped, an interrupted run resumes from ckpt.pt.  Each finished run is evaluated right away (eval_after_run.sh).
set -uo pipefail
TR=${KI_REPO:?set KI_REPO to the repository root}
PB=$TR/training_code/phaseB_base
C=$TR/analysis_main_v1/centroid_v1_20261013
CODE=$C/code/centroid_root; OUT=$PB/centroid_runs_20261013; DATA=$PB/data/wikitext103
PY=<LOCAL_HOME>/miniconda3/envs/bta/bin/python
mkdir -p "$OUT" "$C/logs"
echo "6c355b28fbe3a6d9c20cd1076f3ebc99bae0d7c731117b4ed501feb5f12cb593  $DATA/train.bin
397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979  $DATA/val.bin" | sha256sum -c - || exit 2
sha256sum "$CODE"/*.py
# frozen-root check against the pre-training record
grep -E '^[0-9a-f]{64}  code/centroid_root/' "$C/manifests/PRETRAINING_RECORD.txt" | (cd "$C" && sha256sum -c -) || exit 3

RUNS=(
  "cv1_s7_fw_centroid_k4_tau6_100m|--approx centroid --cells nearest --K 4 --range_mode relevant_zero --range_tau 6 --range_grad full"
  "cv1_s7_minmax_centroid_k4_100m|--approx centroid --cells nearest --K 4 --range_mode minmax --range_grad full"
  "cv1_s7_fw_centroid2_k4_tau6_100m|--approx centroid2 --cells nearest --K 4 --range_mode relevant_zero --range_tau 6 --range_grad full"
  "cv1_s7_minmax_centroid2_k4_100m|--approx centroid2 --cells nearest --K 4 --range_mode minmax --range_grad full"
)
for spec in "${RUNS[@]}"; do
  run=${spec%%|*}; op=${spec#*|}
  if [ -f "$OUT/$run/ckpt_step000763_100.0m.pt" ]; then echo "SKIP $run"; continue; fi
  resume=(); [ -s "$OUT/$run/ckpt.pt" ] && resume=(--resume)
  echo "[$(date -u +%FT%TZ)] START $run ${resume[*]}"
  ( cd "$CODE" && "$PY" -u train.py --run_name "$run" --out_dir "$OUT" --data_dir "$DATA" \
      --n_layer 12 --n_head 12 --n_embd 768 --block_size 1024 --attn quant $op \
      --micro_batch 1 --batch_tokens 131072 --max_tokens 100000000 --warmup_tokens 10000000 \
      --lr 0.0006 --min_lr_ratio 0.1 --weight_decay 0.1 --beta1 0.9 --beta2 0.95 --grad_clip 1.0 \
      --eval_every_tokens 5000000 --eval_batches 100 --stats_batches 1 \
      --ckpt_every_tokens 20000000 --log_every_steps 0 --device cuda --seed 7 "${resume[@]}" ) \
      >> "$C/logs/train_$run.log" 2>&1
  rc=$?
  echo "[$(date -u +%FT%TZ)] END $run rc=$rc"
  if [ -f "$OUT/$run/ckpt_step000763_100.0m.pt" ]; then
    bash "$C/code/jobs/eval_after_run.sh" "$run" >> "$C/logs/eval_chain.log" 2>&1
  fi
done
echo "[$(date -u +%FT%TZ)] QUEUE DONE"
