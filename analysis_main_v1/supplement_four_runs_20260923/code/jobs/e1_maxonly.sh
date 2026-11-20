#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"; check_data
formal "$SUP/code/e1_lerp_range_grad_single" train.py "$SUP/runs" sp_s7_lerp_k32_maxonly_100m \
  --attn quant --K 32 --approx lerp --range_grad max_only
