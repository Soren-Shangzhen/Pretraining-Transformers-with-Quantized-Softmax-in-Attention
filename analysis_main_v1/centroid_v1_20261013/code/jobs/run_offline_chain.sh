#!/usr/bin/env bash
# Wait for O1 to finish, then run O2 on every available model (resumable; new-run checkpoints are picked up on rerun).
C=<REPO>/analysis_main_v1/centroid_v1_20261013
PY=<LOCAL_HOME>/miniconda3/envs/bta/bin/python
while pgrep -f "code/analysis/o1_substitutio[n]" > /dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] O2 start"
cd $C && $PY -u code/analysis/o2_rows_grads.py
echo "[$(date -u +%FT%TZ)] O2 end rc=$?"
