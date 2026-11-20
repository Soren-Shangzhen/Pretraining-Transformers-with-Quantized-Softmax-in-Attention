#!/usr/bin/env python3
"""Step 3 and the hand-back: NLL of each parent under the native and the frozen-step forward, the paired difference
with the paper's circular moving-block bootstrap (block 16, 4000 resamples, seed 20260918; downstream_v1/code/stats.py),
the gate on (a) against the recorded NLL, the wrapper control, and the reading of the pre-declared rule.
Writes ../e2c_step0_check.csv (one long table with every number of the page).
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

T = Path(__file__).resolve().parents[5]   # the root of this package
AM = T / "analysis_main_v1"
sys.path.insert(0, str(AM / "downstream_v1/code"))
import stats as st  # noqa: E402

D = Path(__file__).resolve().parents[1]
E = D / "eval_out"
PAR = {"A250": dict(name="suite A MinMax–Weight K=4, 250M (parent of rb-E2c-03)", tok=250.085376, dirtok="0250m",
                    stem="ckpt_step001908_250.1m", rec=("A", "formal_124m_s1337_nearest_k4_fullgrad_2p5b")),
       "B50": dict(name="suite B MinMax–Weight K=4, 50M (parent of rb-E2ab-03)", tok=50.069504, dirtok="0050m",
                   stem="ckpt_step000382_50.1m", rec=("B", "formal_124m_s1337_nearest_k4_weightste_250m")),
       "B125": dict(name="suite B MinMax–Weight K=4, 125M (parent of rb-E2ab-08)", tok=125.042688, dirtok="0125m",
                    stem="ckpt_step000954_125.0m", rec=("B", "formal_124m_s1337_nearest_k4_weightste_250m"))}
SUITE = {"A": AM / "A_longhorizon_124m_2p5b/trajectory/checkpoint_nll.csv", "B": AM / "B_factorial_124m_250m/trajectory/checkpoint_nll.csv"}
GATE = 3e-4


def load(sub, lab, tag=""):
    p = E / sub / f"parent_{lab}" / f"{PAR[lab]['dirtok']}_{PAR[lab]['stem']}{tag}"
    s = json.loads((p / "summary.json").read_text())
    return np.load(p / "per_block_nll.npy"), s, p


rows, out = [], []
def put(section, parent, item, value, unit="", note=""):
    out.append(dict(section=section, parent=parent, item=item, value=value, unit=unit, note=note))


gate_ok = True
for lab, P in PAR.items():
    suite, run = P["rec"]
    c = pd.read_csv(SUITE[suite])
    rec = float(c[(c.run_name == run) & (abs(c.tokens_m - P["tok"]) < 1e-3)].mean_nll.iloc[0])
    a, sa, _ = load("a_native", lab)
    ar, sar, _ = load("a_rbroot", lab, "__rbroot-minmax")
    b, sb, pb = load("b_frozen", lab, "__frozen-step")
    fo = json.loads((pb / "forward_override.json").read_text())
    assert sb["checkpoint_operator"]["tail_policy"] == "resolved" and fo["module_range_mode"] == "relevant_zero"
    assert sa["provenance"]["checkpoint"]["matches_manifest"] is True, sa["provenance"]["checkpoint"]
    d = st.paired(b, a, kind="blocks"); dr = st.paired(b, ar, kind="blocks"); ctl = st.paired(ar, a, kind="blocks")
    g = abs(float(a.mean()) - rec)
    gate_ok &= g <= GATE
    rows.append(dict(parent=lab, description=P["name"], recorded_nll_5090=rec, nll_a_native=float(a.mean()),
                     a_minus_recorded=float(a.mean()) - rec, gate_pass=g <= GATE, nll_a_rbroot=float(ar.mean()),
                     nll_b_frozen=float(b.mean()), diff_b_minus_a=d["delta"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"],
                     frac_blocks_b_worse=1 - d["frac_negative"], diff_b_minus_arb=dr["delta"], arb_minus_a=ctl["delta"],
                     arb_minus_a_ci=f"[{ctl['ci_lo']:+.5f}, {ctl['ci_hi']:+.5f}]", gpu=sa["gpu"], torch=sa["torch"],
                     h0_per_layer_window=json.dumps([round(x, 4) for x in fo["module_range_tau_per_layer"]])))
    for k in ("recorded_nll_5090", "nll_a_native", "a_minus_recorded", "nll_a_rbroot", "nll_b_frozen", "diff_b_minus_a",
              "ci_lo", "ci_hi", "frac_blocks_b_worse", "diff_b_minus_arb", "arb_minus_a"):
        put("step3", lab, k, rows[-1][k], "nats" if "frac" not in k else "fraction")
R = pd.DataFrame(rows)

# wrapper control on the branch's own 255M checkpoint
c1 = np.load(next((E / "c_control/branch_E2c03").glob("*_255.2m/per_block_nll.npy")))
c2 = np.load(next((E / "c_control/branch_E2c03").glob("*__wrapper-ctl/per_block_nll.npy")))
ctl_bitwise = bool(np.array_equal(c1, c2))
rec255 = float(pd.read_csv(AM / "rb-rebuttal_v5/reports/rb-dossier_20261114/csv/trajectories.csv").query("run_id == 'rb-E2c-03'").sort_values("tokens_m").nll.iloc[0])
put("control", "rb-E2c-03 255M", "nll_native_rb_eval", float(c1.mean()), "nats")
put("control", "rb-E2c-03 255M", "nll_through_wrapper", float(c2.mean()), "nats")
put("control", "rb-E2c-03 255M", "bitwise_equal", ctl_bitwise)
put("control", "rb-E2c-03 255M", "recorded_nll_5090", rec255, "nats")

# step 1 and step 4
L = pd.read_csv(E / "step1_logs.csv")
for r in L.itertuples():
    for k in ("train_loss_WtoW", "train_loss_frozen", "train_loss_frozen_minus_WtoW", "grad_norm_WtoW", "grad_norm_frozen"):
        put("step1", r.parent.split(" ")[0], f"step{r.step_after_branch:02d}_{k}", getattr(r, k), "", f"optimizer step {r.step}, tokens {r.tokens_m}M")
W = pd.read_csv(E / "step4_width_probe.csv")
for r in W.itertuples():
    for k in ("h0", "width_mean", "width_p10", "width_p50", "width_p90", "width_cv", "frac_rows_width_gt_2h0", "frac_rows_width_lt_half_h0"):
        put("step4", r.parent, f"layer{r.layer:02d}_{k}", getattr(r, k))


def side(x):
    if x >= 1.0:
        return "≥ +1.0 nat: the jump is mainly caused by the operator switch itself"
    if abs(x) < 0.1:
        return "|difference| < 0.1 nat: the operator switch at step 0 barely changes the function; the jump arises in the training that follows"
    return "in between: both factors contribute"


def band(x):
    return "≥ +1.0" if x >= 1.0 else ("abs < 0.1" if abs(x) < 0.1 else "0.1 to 1.0 (middle)")


for r in R.itertuples():
    put("rule", r.parent, "band", band(r.diff_b_minus_a))
put("rule", "A250", "reading (rule applies to the jumping lineage)", side(R.set_index("parent").loc["A250", "diff_b_minus_a"]))
pd.DataFrame(out).to_csv(D / "e2c_step0_check.csv", index=False)
R.to_csv(E / "step3_table.csv", index=False)
print(R[["parent", "recorded_nll_5090", "nll_a_native", "a_minus_recorded", "gate_pass", "nll_a_rbroot", "nll_b_frozen",
         "diff_b_minus_a", "ci_lo", "ci_hi"]].to_string(index=False))
print("gate", gate_ok, "control bitwise", ctl_bitwise, float(c1.mean()), rec255)
