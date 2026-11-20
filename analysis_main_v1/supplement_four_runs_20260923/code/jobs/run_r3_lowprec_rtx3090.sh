#!/usr/bin/env bash
# Round 3: reduced-precision softmax baselines, seed 7, local RTX 3090, same protocol as the seed-7 Exact control.
set -uo pipefail
PB=<REPO>/training_code/phaseB_base
SUPL=<REPO>/analysis_main_v1/supplement_four_runs_20260923
CODE=$SUPL/code/r3_lowprec_softmax; OUT=$PB/supplement_runs_20260923; DATA=$PB/data/wikitext103
PY=<LOCAL_HOME>/miniconda3/envs/bta/bin/python
echo "6c355b28fbe3a6d9c20cd1076f3ebc99bae0d7c731117b4ed501feb5f12cb593  $DATA/train.bin
397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979  $DATA/val.bin" | sha256sum -c - || exit 2
sha256sum $CODE/*.py
for mode in softmax_bf16exp softmax_fp8e4m3exp; do
  run=sp_s7_${mode}_100m
  if [ -f "$OUT/$run/ckpt_step000763_100.0m.pt" ]; then echo "SKIP $run"; continue; fi
  resume=(); [ -s "$OUT/$run/ckpt.pt" ] && resume=(--resume)
  echo "[$(date -u +%FT%TZ)] START $run ${resume[*]}"
  ( cd "$CODE" && "$PY" -u train.py --run_name "$run" --out_dir "$OUT" --data_dir "$DATA" \
      --n_layer 12 --n_head 12 --n_embd 768 --block_size 1024 --attn "$mode" \
      --micro_batch 1 --batch_tokens 131072 --max_tokens 100000000 --warmup_tokens 10000000 \
      --lr 0.0006 --min_lr_ratio 0.1 --weight_decay 0.1 --beta1 0.9 --beta2 0.95 --grad_clip 1.0 \
      --eval_every_tokens 5000000 --eval_batches 100 --stats_batches 1 \
      --ckpt_every_tokens 20000000 --log_every_steps 0 --device cuda --seed 7 "${resume[@]}" ) \
      > "$SUPL/round3/logs/train_$run.log" 2>&1
  echo "[$(date -u +%FT%TZ)] END $run rc=$?"
done
