#!/usr/bin/env python3
"""Build supplement_four_runs_manifest.tsv: the new supplement runs (4 briefed + 2 authorized on
2026-09-23) and every seed-7 control they are compared with.  Re-runnable: new-run status, step and
checkpoint hashes are filled from gpu-host-A status files as runs finish (see sync step in the report).

Inputs (all small files):
  manifests/control_checkpoints_local.sha256     sha256sum of local control checkpoints
  manifests/control_prob_cloud.sha256            sha256sum of the cloud originals of the Prob controls
  manifests/new_runs_status.json                 (optional) pulled from gpu-host-A after training
"""
import csv, json, os
import torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


TR = KI_REPO + ""
PB = f"{TR}/training_code/phaseB_base"
SN = f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/code_snapshots"
SUPL = f"{TR}/analysis_main_v1/supplement_four_runs_20260923"
SUPC = KI_CLOUD + "/supplement_four_runs_20260923"
CL5 = KI_CLOUD + "/formal_runs_probste_5seeds_100m_v1"
DATA = {"train_sha256": "6c355b28fbe3a6d9c20cd1076f3ebc99bae0d7c731117b4ed501feb5f12cb593",
        "val_sha256": "397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979"}
HW_LOCAL = "local RTX 3090 (local3090); torch 2.5.1+cu121; python 3.11.15 (bta env)"
HW_POD8 = "cloud pod8, GPU model not logged; py312 env (torch version not logged); step-1 loss/val/gnorm identical to gpu-host-A RTX 5090 replay"
HW_POD7 = "gpu-host-A RTX 5090; torch 2.13.0+cu132; python 3.12.11 (py312 env)"
CK = ["ckpt_step000153_20.1m.pt", "ckpt_step000306_40.1m.pt", "ckpt_step000459_60.2m.pt",
      "ckpt_step000612_80.2m.pt", "ckpt_step000763_100.0m.pt"]

CONTROLS = [
    # id, role, label, run_dir, code_root(for eval), train_code_note, hardware, grad_ckpt, eval_dir
    ("C-EXACT", "E1/E2 softmax baseline", "Exact (exact_fp32_softmax)", f"{PB}/runs/p3_s7_exact_100m",
     f"{SN}/local_phaseB_base", "phaseB base; pre-profiling train.py (reconstructed sha ac743f08..., patch-only profiling); quant 9d03d7a6, model dd032a5a", HW_LOCAL, "Exact_s7"),
    ("C-LERP-FULL", "E1 primary control", "MinMax-LERP K32 full", f"{PB}/runs/p3_s7_lerp_k32_fullgrad_100m",
     f"{SN}/local_phaseB_base", "phaseB base; pre-profiling train.py (reconstructed sha ac743f08...); quant 9d03d7a6, model dd032a5a", HW_LOCAL, "LERP_K32_s7"),
    ("C-LERP-DETACH", "E1 control", "MinMax-LERP K32 detach", f"{PB}/runs_20260921/rb_s7_lerp_k32_detach_100m",
     f"{PB}", "phaseB base; train.py 91bf8e10 (profiling off); quant 9d03d7a6, model dd032a5a; --grad_checkpointing (step-0 grads bit-identical with/without)", HW_LOCAL, "LERP-Detach_K32_s7"),
    ("C-MM-W4", "E2 historical Weight control", "MinMax-Nearest-Weight-STE K4", f"{PB}/runs/p3_s7_nearest_k4_fullgrad_100m",
     f"{SN}/local_phaseB_base", "phaseB base; pre-profiling train.py; quant 9d03d7a6 (Weight-STE forward w_L+(w_N-w_L), not strict)", HW_LOCAL, "MinMax-Weight_K4_s7"),
    ("C-MM-P4", "E2 historical Prob control", "MinMax-Nearest-Prob-STE K4 (old native forward)", f"{CL5}/formal_124m_s7_minmax_nearest_k4_probste_100m",
     f"{SN}/formal_code_probste_5seeds_100m_v1/formal_nearest_probste_v1", "formal_nearest_probste_v1: quant 0e20aea0, model dd032a5a, train_formal cd4fe907", HW_POD8, "MinMax-Prob_K4_s7"),
    ("C-FW-W4", "E2 historical Weight control", "FWM-Nearest-Weight-STE K4 tau6", f"{PB}/qrm_runs_v1/qrm_s7_nearest_k4_tau6_fullgrad_100m",
     f"{SN}/local_phaseB_rr_qrm", "phaseB_rr: quant a4024609, model cf76032a, train 63a8001d (Weight-STE forward, not strict on GPU)", HW_LOCAL, "QRM-Weight_K4_s7"),
    ("C-FW-P4", "E2 historical Prob control", "FWM-Nearest-Prob-STE K4 tau6 (old native forward)", f"{CL5}/formal_124m_s7_qrm_nearest_k4_tau6_probste_100m",
     f"{SN}/formal_code_probste_5seeds_100m_v1/formal_qrm_probste_v1", "formal_qrm_probste_v1: quant ce2bd02b, model cf76032a, train_formal 83d72c34", HW_POD8, "QRM-Prob_K4_s7"),
]

NEW = [
    ("E2-MM-P4", "briefed", "MinMax-Nearest strict Prob-STE K4", "sp_s7_minmax_nearest_k4_probste_strict_100m", "e2_minmax_probste_strict", "--nearest_surrogate prob"),
    ("E2-MM-W4S", "authorized 2026-09-23 (reuse premise failed)", "MinMax-Nearest strict Weight-STE K4", "sp_s7_minmax_nearest_k4_weightste_strict_100m", "e2_minmax_probste_strict", "--nearest_surrogate weight_strict"),
    ("E2-FWM-P4", "briefed", "FWM-Nearest strict Prob-STE K4 tau6 zero-tail", "sp_s7_fwm_nearest_k4_tau6_probste_strict_100m", "e2_fwm_probste_strict", "--nearest_surrogate prob --range_mode relevant_zero --range_tau 6"),
    ("E2-FWM-W4S", "authorized 2026-09-23 (reuse premise failed)", "FWM-Nearest strict Weight-STE K4 tau6 zero-tail", "sp_s7_fwm_nearest_k4_tau6_weightste_strict_100m", "e2_fwm_probste_strict", "--nearest_surrogate weight_strict --range_mode relevant_zero --range_tau 6"),
    ("E1-MIN", "briefed", "MinMax-LERP K32 min_only", "sp_s7_lerp_k32_minonly_100m", "e1_lerp_range_grad_single", "--range_grad min_only"),
    ("E1-MAX", "briefed", "MinMax-LERP K32 max_only", "sp_s7_lerp_k32_maxonly_100m", "e1_lerp_range_grad_single", "--range_grad max_only"),
    ("E1-MIN-s42", "seed-42 replication, authorized 2026-09-24 (addendum 35623a7a)", "MinMax-LERP K32 min_only, seed 42", "sp_s42_lerp_k32_minonly_100m", "e1_lerp_range_grad_single", "--range_grad min_only --seed 42"),
    ("E1-MAX-s42", "seed-42 replication, authorized 2026-09-24 (addendum 35623a7a)", "MinMax-LERP K32 max_only, seed 42", "sp_s42_lerp_k32_maxonly_100m", "e1_lerp_range_grad_single", "--range_grad max_only --seed 42"),
]


def read_sha(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            h, p = line.split(None, 1)
            out[p.strip()] = h
    return out


def main():
    loc = read_sha(f"{SUPL}/manifests/control_checkpoints_local.sha256")
    cld = read_sha(f"{SUPL}/manifests/control_prob_cloud.sha256")
    status = {}
    sp = f"{SUPL}/manifests/new_runs_status.json"
    if os.path.exists(sp):
        status = json.load(open(sp))
    cols = ["id", "role", "condition", "run_name", "seed", "model", "dataset", "token_budget", "end_step",
            "end_tokens", "status", "hardware", "train_code", "eval_code_root", "config_json", "data_train_sha256",
            "data_val_sha256", "run_path"] + [f"ckpt_{t}m_sha256" for t in (20, 40, 60, 80, 100)] + ["eval_path", "notes"]
    rows = []
    for cid, role, cond, rd, er, code, hw, ev in CONTROLS:
        cloud = rd.startswith("/cloud")
        localcopy = rd.replace(CL5, f"{PB}/formal_runs_probste_5seeds_100m_v1") if cloud else rd
        ck = torch.load(f"{localcopy}/{CK[-1]}", map_location="cpu", weights_only=False, mmap=True)
        c = dict(ck["config"])
        hashes = []
        for f in CK:
            hl = loc.get(f"{localcopy}/{f}", "")
            hc = cld.get(f"{rd}/{f}", "") if cloud else ""
            hashes.append(hl if not cloud else (hc if hc == hl or not hl else f"MISMATCH local={hl} cloud={hc}"))
        rows.append(dict(id=cid, role=role, condition=cond, run_name=c["run_name"], seed=c["seed"],
                         model="124M GPT-2 (12L/12H/768)", dataset="WikiText-103", token_budget="100M",
                         end_step=ck["step"], end_tokens=ck["tokens_done"], status="historical, frozen",
                         hardware=hw, train_code=code, eval_code_root=er, config_json=json.dumps(c, sort_keys=True),
                         data_train_sha256=DATA["train_sha256"], data_val_sha256=DATA["val_sha256"], run_path=rd,
                         **{f"ckpt_{t}m_sha256": h for t, h in zip((20, 40, 60, 80, 100), hashes)},
                         eval_path=f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1/{ev}",
                         notes=("local copy " + localcopy) if cloud else ""))
    for nid, origin, cond, run, root, args in NEW:
        st = status.get(run, {})
        local = nid.startswith("E1")
        rows.append(dict(id=nid, role=f"new supplement run ({origin})", condition=cond, run_name=run, seed=42 if "s42" in run else 7,
                         model="124M GPT-2 (12L/12H/768)", dataset="WikiText-103", token_budget="100M",
                         end_step=st.get("step", ""), end_tokens=st.get("tokens_done", ""),
                         status=st.get("status", "queued"), hardware=HW_LOCAL if local else HW_POD7,
                         train_code=f"{(SUPL if local else SUPC)}/code/{root} (sha in code/CODE_PROVENANCE.txt); args: {args}",
                         eval_code_root=f"{(SUPL if local else SUPC)}/code/{root}", config_json=json.dumps(st.get("config", {}), sort_keys=True),
                         data_train_sha256=DATA["train_sha256"], data_val_sha256=DATA["val_sha256"],
                         run_path=(f"{PB}/supplement_runs_20260923/{run}" if local else f"{SUPC}/runs/{run}"),
                         **{f"ckpt_{t}m_sha256": st.get("ckpt_sha256", {}).get(f, "") for t, f in zip((20, 40, 60, 80, 100), CK)},
                         eval_path=(f"{SUPL}/eval/rtx3090/{run}" if local else f"{SUPC}/eval/{run}"),
                         notes=st.get("notes", "") or ("moved from gpu-host-A to the local RTX 3090 on user instruction 2026-09-23 (same GPU/env as the E1 controls)" if local else "")))
    out = f"{SUPL}/manifests/supplement_four_runs_manifest.tsv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
