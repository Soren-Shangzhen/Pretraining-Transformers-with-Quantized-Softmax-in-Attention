#!/usr/bin/env bash
# Second replay of the same implementation to step 102 (same-hardware repeatability).
source "$(dirname "$0")/common.sh"; check_data
replay <CLOUD_DISK>/formal_code_probste_5seeds_100m_v1/formal_nearest_probste_v1 train_formal.py \
  "$SUP/replay" rp_b_s7_minmax_nearest_k4_probste_old ckpt_step000102_13.4m.pt \
  --attn quant --K 4 --approx nearest --range_grad full
