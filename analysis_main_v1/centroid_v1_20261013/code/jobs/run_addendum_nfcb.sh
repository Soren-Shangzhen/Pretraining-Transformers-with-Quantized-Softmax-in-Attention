#!/usr/bin/env bash
# centroid_v1 addendum: one seed-7 run, hard Nearest forward + Centroid backward (FW tau 6, K 4); then evaluation.
set -uo pipefail
TR=${KI_REPO:?set KI_REPO to the repository root}
PB=$TR/training_code/phaseB_base
C=$TR/analysis_main_v1/centroid_v1_20261013
CODE=$C/code/centroid_root_v2; OUT=$PB/centroid_runs_20261013; DATA=$PB/data/wikitext103
PY=<LOCAL_HOME>/miniconda3/envs/bta/bin/python
RUN=cv1b_s7_fw_nearestfwd_centroidbwd_k4_tau6_100m
echo "6c355b28fbe3a6d9c20cd1076f3ebc99bae0d7c731117b4ed501feb5f12cb593  $DATA/train.bin
397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979  $DATA/val.bin" | sha256sum -c - || exit 2
grep -E '^[0-9a-f]{64}  code/' "$C/manifests/PRETRAINING_RECORD_addendum.txt" | (cd "$C" && sha256sum -c -) || exit 3
if [ ! -f "$OUT/$RUN/ckpt_step000763_100.0m.pt" ]; then
  resume=(); [ -s "$OUT/$RUN/ckpt.pt" ] && resume=(--resume)
  echo "[$(date -u +%FT%TZ)] START $RUN ${resume[*]}"
  ( cd "$CODE" && "$PY" -u train.py --run_name "$RUN" --out_dir "$OUT" --data_dir "$DATA" \
      --n_layer 12 --n_head 12 --n_embd 768 --block_size 1024 --attn quant \
      --approx nearest_centroid_bwd --cells nearest --K 4 --range_mode relevant_zero --range_tau 6 --range_grad full \
      --micro_batch 1 --batch_tokens 131072 --max_tokens 100000000 --warmup_tokens 10000000 \
      --lr 0.0006 --min_lr_ratio 0.1 --weight_decay 0.1 --beta1 0.9 --beta2 0.95 --grad_clip 1.0 \
      --eval_every_tokens 5000000 --eval_batches 100 --stats_batches 1 \
      --ckpt_every_tokens 20000000 --log_every_steps 0 --device cuda --seed 7 "${resume[@]}" ) \
      >> "$C/logs/train_$RUN.log" 2>&1
  echo "[$(date -u +%FT%TZ)] END $RUN rc=$?"
fi
JOBS=$C/code/eval_v2/eval_jobs_$RUN.tsv
printf "label\teval_root\tcheckpoint\ttokens_m\trole\n" > "$JOBS"
for ck in "$OUT/$RUN"/ckpt_step*_*m.pt; do
  t=$(basename "$ck" | sed -E 's/.*_([0-9]+)\.[0-9]m\.pt/\1/')
  printf "%s\t%s\t%s\t%s\t%s\n" "$RUN" "$CODE" "$ck" "$t" "centroid_v1 addendum run, own frozen root, 3090 evaluator" >> "$JOBS"
done
echo "[$(date -u +%FT%TZ)] EVAL $RUN"
cd "$C/code/eval_v2" && SUP_SEED=7 "$PY" cv1_eval.py --jobs "$JOBS" --out "$C/eval/rtx3090" --root "$CODE"
echo "[$(date -u +%FT%TZ)] ALL DONE rc=$?"
