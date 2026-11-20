#!/usr/bin/env bash
# Sequential GPU queue: runs every job listed in queue.txt (one per line) that has no .done
# marker; re-reads queue.txt after each job so jobs can be appended while it runs.
set -uo pipefail
J=<REPO>/analysis_main_v1/supplement_four_runs_20260923/code/jobs
L=<CLOUD_DISK>/supplement_four_runs_20260923/logs
S=$J/state; mkdir -p "$S" "$L"
while true; do
  next=""
  while read -r job; do
    [ -z "$job" ] && continue; [[ "$job" == \#* ]] && continue
    [ -e "$S/$job.done" ] || [ -e "$S/$job.failed" ] || { next="$job"; break; }
  done < "$J/queue.txt"
  [ -z "$next" ] && { echo "[$(date -u +%FT%TZ)] queue empty"; break; }
  echo "[$(date -u +%FT%TZ)] START $next"
  bash "$J/$next.sh" > "$L/$next.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch "$S/$next.done"; else echo "rc=$rc" > "$S/$next.failed"; fi
  echo "[$(date -u +%FT%TZ)] END $next rc=$rc"
done
