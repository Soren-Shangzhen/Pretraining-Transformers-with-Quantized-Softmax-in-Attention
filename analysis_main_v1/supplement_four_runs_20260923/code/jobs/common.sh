# Shared settings for the 2026-09-23 supplement jobs (sourced by every job).
set -euo pipefail
SUP=<CLOUD_DISK>/supplement_four_runs_20260923
PY=/usr/local/miniconda3/envs/py312/bin/python
DATA=<CLOUD_DISK>/data/wikitext103
# Historical protocol shared by every seed-7 control (read from their checkpoint configs).
PROTO=(--data_dir "$DATA" --n_layer 12 --n_head 12 --n_embd 768 --block_size 1024
       --micro_batch 1 --batch_tokens 131072 --max_tokens 100000000 --warmup_tokens 10000000
       --lr 0.0006 --min_lr_ratio 0.1 --weight_decay 0.1 --beta1 0.9 --beta2 0.95 --grad_clip 1.0
       --eval_every_tokens 5000000 --eval_batches 100 --stats_batches 1
       --log_every_steps 0 --device cuda --seed 7)

check_data () {
  echo "6c355b28fbe3a6d9c20cd1076f3ebc99bae0d7c731117b4ed501feb5f12cb593  $DATA/train.bin
397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979  $DATA/val.bin" | sha256sum -c -
}

# formal <code_dir> <entry> <out_dir> <run_name> [args...]: fresh start, or resume only from ckpt.pt
formal () {
  local code="$1" entry="$2" out="$3" run="$4"; shift 4
  local resume=()
  if [ -s "$out/$run/ckpt.pt" ]; then resume=(--resume); echo "RESUME $run from $(readlink -f "$out/$run/ckpt.pt")"; fi
  if [ -f "$out/$run/ckpt_step000763_100.0m.pt" ]; then echo "SKIP $run: endpoint exists"; return 0; fi
  cd "$code"
  sha256sum "$code"/*.py
  "$PY" -u "$entry" --run_name "$run" --out_dir "$out" "${PROTO[@]}" --ckpt_every_tokens 20000000 "$@" "${resume[@]}"
}

# replay <code_dir> <entry> <out_dir> <run_name> <stop_ckpt_basename> [args...]:
# the frozen trainer with the historical schedule (max_tokens 100M), stopped after
# the named checkpoint is written.  --ckpt_every_tokens only sets save points.
replay () {
  local code="$1" entry="$2" out="$3" run="$4" stop="$5"; shift 5
  if [ -f "$out/$run/$stop" ]; then echo "SKIP replay $run: $stop exists"; return 0; fi
  if [ -e "$out/$run" ]; then echo "ERROR: partial replay dir $out/$run"; return 3; fi
  cd "$code"
  sha256sum "$code"/*.py
  "$PY" -u "$entry" --run_name "$run" --out_dir "$out" "${PROTO[@]}" --ckpt_every_tokens 6684672 "$@" &
  local pid=$!
  until [ -f "$out/$run/$stop" ] || ! kill -0 "$pid" 2>/dev/null; do sleep 15; done
  sleep 20   # let os.replace finish and the hard link be made
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  test -f "$out/$run/$stop"
  echo "REPLAY STOPPED after $stop"
}
