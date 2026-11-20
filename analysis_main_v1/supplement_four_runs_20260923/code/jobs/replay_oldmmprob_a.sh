#!/usr/bin/env bash
# Old MinMax-Prob K4 seed 7 (frozen root, historically trained on the cloud) to step 153,
# compared bit-for-bit with the historical 20M checkpoint.
source "$(dirname "$0")/common.sh"; check_data
replay <CLOUD_DISK>/formal_code_probste_5seeds_100m_v1/formal_nearest_probste_v1 train_formal.py \
  "$SUP/replay" rp_a_s7_minmax_nearest_k4_probste_old ckpt_step000153_20.1m.pt \
  --attn quant --K 4 --approx nearest --range_grad full
