#!/usr/bin/env bash
# Old LERP K32 full (frozen phaseB base root; control historically trained on the local RTX 3090)
# to step 51; logged losses are compared with the historical log.
source "$(dirname "$0")/common.sh"; check_data
replay <CLOUD_DISK>/code/phaseB train.py \
  "$SUP/replay" rp_s7_lerp_k32_fullgrad_old ckpt_step000051_6.7m.pt \
  --attn quant --K 32 --approx lerp --range_grad full
