#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"; check_data
formal "$SUP/code/e2_fwm_probste_strict" train_formal.py "$SUP/runs" sp_s7_fwm_nearest_k4_tau6_weightste_strict_100m \
  --attn quant --K 4 --approx nearest --range_grad full --range_mode relevant_zero --range_tau 6 --nearest_surrogate weight_strict
