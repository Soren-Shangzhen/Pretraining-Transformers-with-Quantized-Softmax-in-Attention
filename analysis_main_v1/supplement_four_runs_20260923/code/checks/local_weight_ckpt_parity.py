"""Model-level forward parity of the OLD Weight-STE seed-7 K4 checkpoints (local-only
assets) against the strict Weight reference and the new strict Prob implementation.

Runs on the local CPU (fp32, eval, no autocast) with SUP_LOCAL=1.  This is a
diagnostic path, not the training path: the GPU training path is covered by
ops_check_e2.py (real seed-7 scores) and step0_check.py.  CPU and GPU `exp`
differ in the last bit for some FWM grid levels, which changes whether the old
Weight-STE round trip is exact (see ops_check_e2 / diag_fwm_cpu_gpu.py).
"""
import os, sys, time
os.environ["SUP_LOCAL"] = "1"
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sup_common import LocalsCapture, compare, dump_json, env_record, load_root, ROOTS, root_hashes, sha256_file  # noqa
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


TR = KI_REPO + ""
PB = f"{TR}/training_code/phaseB_base"
CK = {
    "minmax": (f"{PB}/runs/p3_s7_nearest_k4_fullgrad_100m", "old_phaseB_base", "new_mm_prob", {}),
    "fwm": (f"{PB}/qrm_runs_v1/qrm_s7_nearest_k4_tau6_fullgrad_100m", "old_phaseB_rr", "new_fw_prob",
            dict(attn_range_mode="relevant_zero", attn_range_tau=6.0)),
}
STEPS = ["ckpt_step000153_20.1m.pt", "ckpt_step000459_60.2m.pt", "ckpt_step000763_100.0m.pt"]


def build(key, extra, sd, surrogate=None, tag=None):
    qa, mm, _ = load_root(key, tag=tag, surrogate=surrogate)
    cfg = mm.GPTConfig(block_size=1024, n_layer=12, n_head=12, n_embd=768, attn_mode="quant", attn_K=4,
                       attn_approx="nearest", attn_range_grad="full", **extra)
    m = mm.GPT(cfg); m.load_state_dict(sd); m.eval()
    return m


def run(m, x, y):
    Ps = []
    with torch.no_grad(), LocalsCapture(["_quant_attention"], ["P"], lambda fn, d: Ps.append(d["P"].detach().clone())):
        logits, loss = m(x, y)
    return Ps, logits, loss


def main(out):
    torch.set_num_threads(8)
    val = np.memmap(f"{PB}/data/wikitext103/val.bin", dtype=np.uint16, mode="r")
    x = torch.from_numpy(val[:1024].astype(np.int64)).view(1, -1)
    y = torch.from_numpy(val[1:1025].astype(np.int64)).view(1, -1)
    rep = {"script": os.path.abspath(__file__), "env": env_record(), "device": "cpu", "precision": "fp32 eval, no autocast",
           "roots": {k: {"path": v, "sha256": root_hashes(v)} for k, v in ROOTS.items()}, "cases": {}}
    for calib, (rd, old_w, new, extra) in CK.items():
        for st in STEPS:
            ck = torch.load(f"{rd}/{st}", map_location="cpu", weights_only=False, mmap=True)
            sd = ck["model"]
            res = {n: run(build(k, extra, sd, s, t), x, y) for n, (k, s, t) in {
                "old_weight": (old_w, None, None),
                "strict_weight_ref": (new, "weight_strict", f"{new}__wstrict"),
                "new_prob": (new, "prob", f"{new}__prob")}.items()}
            case = {"checkpoint": f"{rd}/{st}", "step": ck["step"], "tokens": ck["tokens_done"]}
            for a, b in (("old_weight", "strict_weight_ref"), ("new_prob", "strict_weight_ref")):
                Pc = [compare(p, q) for p, q in zip(res[a][0], res[b][0])]
                case[f"{a}__vs__{b}"] = {
                    "P_layers_eq": [c["eq_rawbits"] for c in Pc],
                    "P_mismatch_per_layer": [c["mismatch_count"] for c in Pc],
                    "P_max_abs_per_layer": [c["max_abs"] for c in Pc],
                    "logits": compare(res[a][1], res[b][1]), "loss": compare(res[a][2], res[b][2]),
                    "loss_values": [float(res[a][2]), float(res[b][2])]}
            rep["cases"][f"{calib}/{st}"] = case
            c = case["old_weight__vs__strict_weight_ref"]
            print(calib, st, "oldW==strict P layers:", sum(c["P_layers_eq"]), "/12  logits max_abs",
                  f'{c["logits"]["max_abs"]:.3e}', "loss", c["loss_values"],
                  "| newProb==strict:", all(case["new_prob__vs__strict_weight_ref"]["P_layers_eq"]), flush=True)
    dump_json(out, rep)


if __name__ == "__main__":
    main(sys.argv[1])
