#!/bin/bash
# All 13.3 code checks of rb-root_v1, one environment.  usage: run_all_checks.sh PYTHON [THIRDPARTY_PATH|--no-thirdparty]
# Exit status 0 only if every check passes.  Logs: ../../logs/checks_<host>_<torch>.log
set -u
PY=$1; TP=${2:---no-thirdparty}
# CPU bitwise gates single-threaded (user decision 2026-11-10); also set in rb_common.py with torch.set_num_threads(1)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
mkdir -p ../../logs
TV=$($PY -c "import torch;print(torch.__version__.split('+')[0])")
LOG=../../logs/checks_$(hostname)_torch$TV.log
: > $LOG
rc=0
# RB_SKIP_CHECKS: space-separated script names to skip
run() { case " ${RB_SKIP_CHECKS:-} " in *" $2 "*) echo "=== $* skipped (RB_SKIP_CHECKS)" | tee -a $LOG; return;; esac
        echo "=== $*" | tee -a $LOG; "$@" >> $LOG 2>&1; r=$?; echo "rc=$r" | tee -a $LOG; [ $r -ne 0 ] && rc=1; }
run $PY smoke_all_modes.py
run $PY check_root_equivalence.py
run $PY check_fwm_ops.py
run $PY check_formats.py
run $PY check_external_ops.py "$TP"
run $PY check_ema.py
run $PY check_train_mechanics.py
if [ "$TP" != "--no-thirdparty" ]; then
  echo "=== test_align (CPU) / test_align_gpu" | tee -a $LOG
  (cd ../e3_ref && $PY -I test_align.py "$TP" && $PY -I test_align_gpu.py "$TP") >> $LOG 2>&1; r=$?; echo "rc=$r" | tee -a $LOG; [ $r -ne 0 ] && rc=1
fi
grep -c "^PASS" $LOG | sed 's/^/PASS lines: /'; grep "^FAIL" $LOG | head -20
echo "overall rc=$rc"; exit $rc
