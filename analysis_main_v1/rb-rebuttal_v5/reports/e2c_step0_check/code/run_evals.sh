#!/bin/bash
# E2c step-0 check: validation NLL of the three parents (native and frozen-step forward) and a wrapper control.
# Read-only on checkpoints/code; writes only under ../eval_out. Local RTX 3090, batch 1, TF32 off, 976 blocks.
set -euo pipefail
T=<REPO>
O=$T/analysis_main_v1/rb-rebuttal_v5/reports/e2c_step0_check
EV=$T/analysis_main_v1/rb-rebuttal_v5/code/eval
RBROOT=$T/analysis_main_v1/rb-rebuttal_v5/code/rb-root_v1
HIST=$T/analysis_main_v1/supplement_four_runs_20260923/round2/task1/roots/native_formal_base_v2
FB=$T/adaptive_k_interval_training_phaseB/files_adaptive_quant_phaseB/rb-upload_staging_20261110/from_private-storage
BR=$T/adaptive_k_interval_training_phaseB/files_adaptive_quant_phaseB/rb-runs_B_20261110
export RB_DATA_FINEWEBEDU_3B=$FB/data/finewebedu_3b/val.bin
export RB_DATA_FINEWEBEDU_3B_SHA256=682c9e4492c051ccb5cbde6e9a61099c17aab9f044a6201b901854e9f3943ec5
export RB_CLOUD=$FB            # provenance: parents are matched against manifests/rb-recorded_sha256.tsv
PA=$FB/formal_runs_scale_v1/formal_124m_s1337_nearest_k4_fullgrad_2p5b/ckpt_step001908_250.1m.pt
PB50=$FB/vast_formal_250m_v1/formal_runs_fbmatch_v1/formal_124m_s1337_nearest_k4_weightste_250m/ckpt_step000382_50.1m.pt
PB125=$FB/vast_formal_250m_v1/formal_runs_fbmatch_v1/formal_124m_s1337_nearest_k4_weightste_250m/ckpt_step000954_125.0m.pt
J=$O/eval_out/jobs; mkdir -p $J
hdr=$'label\teval_root\tcheckpoint\tdataset\ttokens_m\trole\tsurrogate'
job() { printf '%s\n%s\t%s\t%s\tfinewebedu_3b\t%s\tparent\tweight_strict\n' "$hdr" "$1" "$2" "$3" "$4" > "$5"; }
# h0 of each frozen-step branch, read from the branch's own checkpoint config (not recomputed)
h0() { python3 -c "import torch,sys; print(torch.load(sys.argv[1],map_location='cpu',weights_only=False,mmap=True)['config']['resolved_h0'])" "$1"; }
H_A=$(h0 $BR/rb-E2c-03/ckpt_step001947_255.2m.pt)
H_B50=$(h0 $BR/rb-E2ab-03/ckpt_step000421_55.2m.pt)
H_B125=$(h0 $BR/rb-E2ab-08/ckpt_step000993_130.2m.pt)
printf 'parent\tbranch\tresolved_h0\nA250\trb-E2c-03\t%s\nB50\trb-E2ab-03\t%s\nB125\trb-E2ab-08\t%s\n' "$H_A" "$H_B50" "$H_B125" > $O/eval_out/h0_from_branch_configs.tsv
for p in A250:$PA:250.085376 B50:$PB50:50.069504 B125:$PB125:125.042688; do
  IFS=: read lab ck tok <<< "$p"
  job parent_$lab $HIST $ck $tok $J/a_$lab.tsv
  job parent_$lab $RBROOT $ck $tok $J/r_$lab.tsv
done
cd $EV
STAGE=${1:-all}
for lab in A250 B50 B125; do
  echo "== (a) native, historical root: $lab"
  python3 rb_eval.py --jobs $J/a_$lab.tsv --out $O/eval_out/a_native --root $HIST
done
[ "$STAGE" = "a" ] && { echo STAGE_A_DONE; exit 0; }
for lab in A250 B50 B125; do
  echo "== (a') native MinMax through rb-root_v1: $lab"
  python3 $O/code/e2c_forward_override.py --set range_mode=minmax --set tail_policy=zero --set resolved_h0= \
    --set nearest_surrogate=weight_strict --tag __rbroot-minmax --jobs $J/r_$lab.tsv --out $O/eval_out/a_rbroot --root $RBROOT
done
for p in A250:"$H_A" B50:"$H_B50" B125:"$H_B125"; do
  lab=${p%%:*}; h=${p#*:}
  echo "== (b) frozen-step forward: $lab"
  python3 $O/code/e2c_forward_override.py --set range_mode=relevant_zero --set tail_policy=resolved --set range_tau=6.0 \
    --set resolved_h0=$h --set range_grad=full --set nearest_surrogate=weight_strict --tag __frozen-step \
    --jobs $J/r_$lab.tsv --out $O/eval_out/b_frozen --root $RBROOT
done
# (c) wrapper control on the branch's own first checkpoint (255M): native rb_eval vs the wrapper with its own fields
C=$BR/rb-E2c-03/ckpt_step001947_255.2m.pt
printf '%s\nbranch_E2c03\t%s\t%s\tfinewebedu_3b\t255.215616\tbranch\tweight_strict\n' "$hdr" $RBROOT $C > $J/c_branch.tsv
echo "== (c1) rb-E2c-03 255M native"
python3 rb_eval.py --jobs $J/c_branch.tsv --out $O/eval_out/c_control --root $RBROOT
echo "== (c2) rb-E2c-03 255M through the wrapper with its own fields"
python3 $O/code/e2c_forward_override.py --set range_mode=relevant_zero --set tail_policy=resolved --set range_tau=6.0 \
  --set resolved_h0=$H_A --set range_grad=full --set nearest_surrogate=weight_strict --tag __wrapper-ctl \
  --jobs $J/c_branch.tsv --out $O/eval_out/c_control --root $RBROOT
echo ALL_DONE
