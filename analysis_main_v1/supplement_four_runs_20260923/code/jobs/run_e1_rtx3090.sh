#!/usr/bin/env bash
# E1 formal runs on the local RTX 3090 (user instruction 2026-09-23): same GPU and env (bta, torch
# 2.5.1+cu121) as the seed-7 LERP K32 full/detach controls. Sequential; resumes only from ckpt.pt.
set -uo pipefail
PB=<REPO>/training_code/phaseB_base
SUPL=<REPO>/analysis_main_v1/supplement_four_runs_20260923
CODE=$SUPL/code/e1_lerp_range_grad_single
OUT=$PB/supplement_runs_20260923
DATA=$PB/data/wikitext103
PY=<LOCAL_HOME>/miniconda3/envs/bta/bin/python
mkdir -p "$OUT" "$SUPL/logs"
echo "6c355b28fbe3a6d9c20cd1076f3ebc99bae0d7c731117b4ed501feb5f12cb593  $DATA/train.bin
397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979  $DATA/val.bin" | sha256sum -c - || exit 2
sha256sum -c <(grep "  e1_lerp" $SUPL/code/CODE_PROVENANCE.txt | sed "s#  e1_#  $SUPL/code/e1_#") || exit 3
for mode in min_only max_only; do
  run=sp_s7_lerp_k32_${mode/_/}_100m
  if [ -f "$OUT/$run/ckpt_step000763_100.0m.pt" ]; then echo "SKIP $run"; continue; fi
  resume=(); [ -s "$OUT/$run/ckpt.pt" ] && resume=(--resume)
  echo "[$(date -u +%FT%TZ)] START $run ${resume[*]}"
  ( cd "$CODE" && "$PY" -u train.py --run_name "$run" --out_dir "$OUT" --data_dir "$DATA" \
      --n_layer 12 --n_head 12 --n_embd 768 --block_size 1024 \
      --attn quant --K 32 --approx lerp --range_grad "$mode" \
      --micro_batch 1 --batch_tokens 131072 --max_tokens 100000000 --warmup_tokens 10000000 \
      --lr 0.0006 --min_lr_ratio 0.1 --weight_decay 0.1 --beta1 0.9 --beta2 0.95 --grad_clip 1.0 \
      --eval_every_tokens 5000000 --eval_batches 100 --stats_batches 1 \
      --ckpt_every_tokens 20000000 --log_every_steps 0 --device cuda --seed 7 "${resume[@]}" ) \
      > "$SUPL/logs/e1_local_$run.log" 2>&1
  echo "[$(date -u +%FT%TZ)] END $run rc=$?"
done
