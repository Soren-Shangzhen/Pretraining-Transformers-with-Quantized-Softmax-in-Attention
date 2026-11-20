#!/usr/bin/env python3
"""(a) Size of the Prob-STE vs Weight-STE forward rounding difference, over the FULL validation split.

Appendix C currently reports only a single-block maximum over the first 8 blocks (1.3e-2 for
MinMax-Prob K4, 5.5e-3 for FW-Prob K4). A single-block maximum bounds the sensitivity of one block;
it is not the size of the confound that enters a mean-NLL comparison. This measures the whole split.

Weight-STE and Prob-STE define the same hard forward P^N mathematically. Prob-STE's training code
evaluates it as P_lerp + (P_hard - P_lerp) in fp32, which differs from P_hard by rounding. So for
every run we compute per-block NLL twice:

    frozen  : the run's own training code root (what every number in the paper uses)
    unified : eval_operator_v1, which evaluates P^N directly

and report the mean, mean-absolute and max block difference, and the difference in MEAN NLL over the
split -- the quantity a paired comparison actually uses. Weight-STE runs are controls: the unified
operator is bit-identical to their frozen code, so they must read ~0 and calibrate the measurement.

Each run is evaluated on the full validation split of its OWN corpus, per the paper's protocol.
Batch 1, bf16 autocast outside the fp32 attention kernel, TF32 off. Every code root runs in its own
subprocess so the modules never mix. Resumable: per-block arrays are cached as .npy.
"""
import csv, json, os, subprocess, sys
import numpy as np

ROOT = KI_CLOUD
AM = f"{ROOT}/analysis_main_v1"
UNIFIED = f"{AM}/code/eval_operator_v1"
OUT = f"{AM}/rounding_audit"
DATA = {"finewebedu_3b": f"{ROOT}/data/finewebedu_3b/val.bin",
        "wikitext103": f"{ROOT}/data/wikitext103/val.bin"}
PY = sys.executable

CHILD = r"""
import sys, json, numpy as np, torch
root, ckpt, val, nb, dst = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
sys.path.insert(0, root)
import model as M
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
c = ck["config"]
kw = dict(block_size=int(c["block_size"]), n_layer=int(c["n_layer"]), n_head=int(c["n_head"]),
          n_embd=int(c["n_embd"]), attn_mode=c.get("attn", "quant"), attn_K=int(c.get("K", 8)),
          attn_approx=c.get("approx", "lerp"), attn_range_grad=c.get("range_grad", "full"))
f = getattr(M.GPTConfig, "__dataclass_fields__", {})
if "attn_range_mode" in f:
    kw["attn_range_mode"] = c.get("range_mode") or "minmax"
    kw["attn_range_tau"] = float(c.get("range_tau") or 6.0)
cfg = M.GPTConfig(**kw); m = M.GPT(cfg)
sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
miss, unexp = m.load_state_dict(sd, strict=False)
assert not miss and not unexp, (miss, unexp)
m.cuda().eval()
d = np.memmap(val, dtype=np.uint16, mode="r")
T = int(cfg.block_size)
out = np.empty(nb, dtype=np.float64)
with torch.inference_mode():
    for i in range(nb):
        s = i * T
        x = torch.from_numpy(np.asarray(d[s:s+T], dtype=np.int64)).unsqueeze(0).cuda()
        y = torch.from_numpy(np.asarray(d[s+1:s+1+T], dtype=np.int64)).unsqueeze(0).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg, _ = m(x)
        out[i] = torch.nn.functional.cross_entropy(
            lg.float().view(-1, lg.size(-1)), y.view(-1)).item()
np.save(dst, out)
print("OK")
"""


def per_block(root, ckpt, val, nb, cache):
    if os.path.exists(cache):
        return np.load(cache)
    r = subprocess.run([PY, "-c", CHILD, root, ckpt, val, str(nb), cache],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2500:], flush=True)
        raise SystemExit(f"child failed: {root} / {ckpt}")
    return np.load(cache)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    runs = list(csv.DictReader(open(f"{AM}/manifests/primary_37_runs.tsv"), delimiter="\t"))
    sel = [r for r in runs if r["backward"] in ("prob_ste", "weight_ste")]
    if only:
        sel = [r for r in sel if r["suite"].startswith(only)]
    sel.sort(key=lambda r: (r["suite"], r["label"], int(r["K"] or 0)))
    os.makedirs(f"{OUT}/per_block", exist_ok=True)
    nblocks = {k: (len(np.memmap(v, dtype=np.uint16, mode="r")) - 1) // 1024 for k, v in DATA.items()}
    print("validation blocks:", nblocks, flush=True)
    rows = []
    for r in sel:
        name = f'{r["suite"].split("_")[0]}_{r["label"]}_K{r["K"]}'
        ckpt = os.path.join(r["run_path"], r["final_checkpoint"])
        val = DATA[r["dataset"]]
        nb = nblocks[r["dataset"]]
        if not os.path.exists(ckpt):
            print(f"  [skip] {name}: missing {ckpt}", flush=True); continue
        frozen = per_block(r["train_code_root"], ckpt, val, nb, f"{OUT}/per_block/{r['run_name']}__frozen.npy")
        unified = per_block(UNIFIED, ckpt, val, nb, f"{OUT}/per_block/{r['run_name']}__unified.npy")
        d = frozen - unified
        rows.append(dict(name=name, run_name=r["run_name"], suite=r["suite"], label=r["label"],
                         K=r["K"], backward=r["backward"], corpus=r["dataset"], n_blocks=int(nb),
                         mean_nll_frozen=float(frozen.mean()), mean_nll_unified=float(unified.mean()),
                         delta_mean_nll=float(frozen.mean() - unified.mean()),
                         block_mean_delta=float(d.mean()),
                         block_mean_abs_delta=float(np.abs(d).mean()),
                         block_max_abs_delta=float(np.abs(d).max()),
                         block_sd_delta=float(d.std(ddof=1)),
                         first8_max_abs_delta=float(np.abs(d[:8]).max())))
        print(f'  {name:26s} {r["backward"]:10s} n={nb:4d}  Dmean {rows[-1]["delta_mean_nll"]:+.3e}  '
              f'mean|d| {rows[-1]["block_mean_abs_delta"]:.3e}  max|d| {rows[-1]["block_max_abs_delta"]:.3e}',
              flush=True)
        json.dump(rows, open(f"{OUT}/rounding_full_split.json", "w"), indent=2)
    if rows:
        with open(f"{OUT}/rounding_full_split.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}/rounding_full_split.{{json,csv}}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
