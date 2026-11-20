#!/usr/bin/env python3
"""Check 1: the new roots in tail_policy=zero must reproduce the frozen roots BIT-IDENTICALLY.

243 WikiText-103 validation blocks, batch 1, bf16 autocast outside the fp32 attention kernel,
TF32 off -- the suite D evaluation protocol. Each root runs in its own subprocess.
"""
import csv, json, subprocess, sys
import numpy as np

PB = (KI_REPO + "/"
      "training_code/phaseB_base")
AM = KI_REPO + "/analysis_main_v1"
MAN = f"{AM}/D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv"
VAL = f"{PB}/data/wikitext103/val.bin"

CHILD = r"""
import sys, json, numpy as np, torch
root, ckpt, val, policy = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, root)
import model as M
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
c = ck["config"]
kw = dict(block_size=int(c["block_size"]), n_layer=int(c["n_layer"]), n_head=int(c["n_head"]),
          n_embd=int(c["n_embd"]), attn_mode=c["attn"], attn_K=int(c["K"]),
          attn_approx=c["approx"], attn_range_grad=c["range_grad"],
          attn_range_mode=c.get("range_mode") or "minmax",
          attn_range_tau=float(c.get("range_tau") or 6.0))
f = getattr(M.GPTConfig, "__dataclass_fields__", {})
if "attn_tail_policy" in f and policy != "-":
    kw["attn_tail_policy"] = policy
cfg = M.GPTConfig(**kw); m = M.GPT(cfg)
sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
miss, unexp = m.load_state_dict(sd, strict=False); assert not miss and not unexp
m.cuda().eval()
d = np.memmap(val, dtype=np.uint16, mode="r"); T = int(cfg.block_size)
nb = (len(d) - 1) // T
out = np.empty(nb, dtype=np.float64)
with torch.inference_mode():
    for i in range(nb):
        s = i * T
        x = torch.from_numpy(np.asarray(d[s:s+T], dtype=np.int64)).unsqueeze(0).cuda()
        y = torch.from_numpy(np.asarray(d[s+1:s+1+T], dtype=np.int64)).unsqueeze(0).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg, _ = m(x)
        out[i] = torch.nn.functional.cross_entropy(lg.float().view(-1, lg.size(-1)), y.view(-1)).item()
np.save(sys.argv[5], out); print(nb)
"""


def run(root, ckpt, policy, dst):
    r = subprocess.run([sys.executable, "-c", CHILD, root, ckpt, VAL, policy, dst],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2000:]); raise SystemExit(f"child failed: {root}")
    return np.load(dst), int(r.stdout.strip().splitlines()[-1])


def main():
    runs = {r["cond"]: r for r in csv.DictReader(open(MAN), delimiter="\t") if r["seed"] == "1337"}
    pairs = [("QRM-Weight K4", "rb_tail_weight"), ("QRM-Prob K4", "rb_tail_prob")]
    ok = True
    for cond, newroot in pairs:
        r = runs[cond]
        ckpt = r["checkpoint_100m"]
        frozen = r["train_code_root"]
        a, nb = run(frozen, ckpt, "-", "/tmp/parity_frozen.npy")
        b, _ = run(f"{PB}/code/{newroot}", ckpt, "zero", "/tmp/parity_new.npy")
        same = np.array_equal(a, b)
        ok &= same
        print(f"  [{'PASS' if same else 'FAIL'}] {cond}: {newroot} tail_policy=zero vs frozen "
              f"-- {nb} blocks, bit-identical={same}, mean {a.mean():.9f} / {b.mean():.9f}, "
              f"max|d|={np.abs(a-b).max():.3e}", flush=True)
    print(f"\nEVAL PARITY {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
