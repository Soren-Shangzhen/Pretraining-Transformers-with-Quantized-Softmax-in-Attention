"""Short-replay comparison (runs on gpu-host-A, CPU).  Old MinMax-Prob K4 seed 7 replayed twice on the
RTX 5090 with the frozen trainer and the historical schedule: replay A vs the historical pod8
checkpoint at step 153 (parameters, AdamW state, data-RNG state), replay A vs replay B at steps 51
and 102, and the logged train/val loss and grad norm of all three."""
import csv, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sup_common import compare, dump_json, sha256_file  # noqa: E402
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


SUP = KI_CLOUD + "/supplement_four_runs_20260923"
HIST = KI_CLOUD + "/formal_runs_probste_5seeds_100m_v1/formal_124m_s7_minmax_nearest_k4_probste_100m"
A = f"{SUP}/replay/rp_a_s7_minmax_nearest_k4_probste_old"
B = f"{SUP}/replay/rp_b_s7_minmax_nearest_k4_probste_old"


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False, mmap=True)


def cmp_ckpt(pa, pb):
    a, b = load(pa), load(pb)
    rec = {"a": pa, "b": pb, "step": [a["step"], b["step"]], "tokens": [a["tokens_done"], b["tokens_done"]],
           "train_rng_state_equal": a["train_rng_state"] == b["train_rng_state"]}
    pm = {k: compare(a["model"][k], b["model"][k]) for k in a["model"]}
    rec["params_all_bitwise"] = all(v["eq_rawbits"] for v in pm.values())
    rec["params_n_mismatching"] = sum(1 for v in pm.values() if not v["eq_rawbits"])
    rec["params_max_abs"] = max(v["max_abs"] for v in pm.values())
    rec["params_max_rel_l2"] = max(v["rel_l2"] for v in pm.values())
    rec["params_worst"] = max(pm, key=lambda k: pm[k]["rel_l2"])
    sa, sb = a["opt"]["state"], b["opt"]["state"]
    om = {}
    for i in sa:
        for key in ("exp_avg", "exp_avg_sq", "step"):
            om[f"{i}.{key}"] = compare(sa[i][key].float() if key == "step" else sa[i][key],
                                       sb[i][key].float() if key == "step" else sb[i][key])
    rec["opt_all_bitwise"] = all(v["eq_rawbits"] for v in om.values())
    rec["opt_n_mismatching"] = sum(1 for v in om.values() if not v["eq_rawbits"])
    rec["opt_max_rel_l2"] = max(v["rel_l2"] for v in om.values())
    return rec


def logrows(path):
    return {int(r["step"]): r for r in csv.DictReader(open(path))}


def main():
    out = {"hist_20m_sha256": sha256_file(f"{HIST}/ckpt_step000153_20.1m.pt")}
    out["A153_vs_hist153"] = cmp_ckpt(f"{A}/ckpt_step000153_20.1m.pt", f"{HIST}/ckpt_step000153_20.1m.pt")
    for s, name in ((51, "ckpt_step000051_6.7m.pt"), (102, "ckpt_step000102_13.4m.pt")):
        if os.path.exists(f"{B}/{name}"):
            out[f"A{s}_vs_B{s}"] = cmp_ckpt(f"{A}/{name}", f"{B}/{name}")
    h, a = logrows(f"{HIST}/log.csv"), logrows(f"{A}/log.csv")
    b = logrows(f"{B}/log.csv") if os.path.exists(f"{B}/log.csv") else {}
    out["log"] = {s: {"hist": {k: h[s][k] for k in ("train_loss", "val_loss", "grad_norm")},
                      "A": {k: a[s][k] for k in ("train_loss", "val_loss", "grad_norm")},
                      "B": ({k: b[s][k] for k in ("train_loss", "val_loss", "grad_norm")} if s in b else None)}
                  for s in sorted(set(h) & set(a))}
    first = [s for s in sorted(out["log"]) if out["log"][s]["hist"] != out["log"][s]["A"]]
    out["first_logged_step_A_differs_from_hist"] = first[0] if first else None
    dump_json(f"{SUP}/checks/short_replay_v1.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=1, default=str))
    for s, v in out["log"].items():
        print(s, v)


if __name__ == "__main__":
    main()
