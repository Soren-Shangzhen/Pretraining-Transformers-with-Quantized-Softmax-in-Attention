#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"; check_data
formal "$SUP/code/e2_minmax_probste_strict" train_formal.py "$SUP/runs" sp_s7_minmax_nearest_k4_probste_strict_100m \
  --attn quant --K 4 --approx nearest --range_grad full --nearest_surrogate prob
