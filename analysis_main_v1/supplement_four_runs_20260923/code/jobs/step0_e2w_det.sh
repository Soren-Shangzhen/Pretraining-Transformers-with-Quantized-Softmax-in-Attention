#!/usr/bin/env bash
# Step-0 backward of the E2 surrogates vs the historical ones; deterministic algorithms in this
# separate diagnostic process only (never in a training process).
source "$(dirname "$0")/common.sh"
cd "$SUP/code/checks"
CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" step0_check.py --part e2w --deterministic --out "$SUP/checks/step0_e2w_v1_deterministic.json"
