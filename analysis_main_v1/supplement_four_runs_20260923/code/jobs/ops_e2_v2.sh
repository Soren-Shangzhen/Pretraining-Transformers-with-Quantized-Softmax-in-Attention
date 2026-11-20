#!/usr/bin/env bash
# E2 operator checks v2 (exact FWM boundary rows added); CPU + GPU, real seed-7 scores.
source "$(dirname "$0")/common.sh"
cd "$SUP/code/checks"
"$PY" ops_check_e2.py --real --out "$SUP/checks/ops_e2_v2.json"
