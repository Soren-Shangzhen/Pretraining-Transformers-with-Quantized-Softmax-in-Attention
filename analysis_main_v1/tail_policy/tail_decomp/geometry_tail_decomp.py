#!/usr/bin/env python3
"""§6 geometry probe for the tail decomposition: 8 validation blocks, per layer, endpoint checkpoints.

Native probabilities come from each run's OWN reconstruct with its OWN tail policy -- the shared
geometry_probe_v1.py does not pass tail_policy and would silently report zero-tail geometry for the
resolved runs. Metrics, averaged over rows with >= 2 causally valid keys:
  span                  row max - row min of the scaled scores
  native_h              step of the native grid (tau/K for the FWM family, span/K for MinMax)
  tv_native_exact       0.5 * sum |P_native - P_exact|
  frac_tail_keys        fraction of valid keys with z < -tau
  top_level_mass        exact-softmax mass on keys at the top native level
  native_tail_mass      NEW: native probability mass on keys with z < -tau
  exact_tail_mass       reference: exact-softmax mass on those keys
  levels_used           distinct native levels per row (K+1 = 5 is the budget)
Each condition runs in its own subprocess so code roots never mix.
"""
import csv, json, subprocess, sys
from pathlib import Path

PB = (KI_REPO + "/"
      "training_code/phaseB_base")
AM = KI_REPO + "/analysis_main_v1"
MAN = f"{AM}/D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv"
OUT = Path(__file__).resolve().parent

CHILD = r"""
import sys, json, math, inspect, numpy as np, torch
root, ckpt, kind, policy, val = sys.argv[1:6]
TAU, K, NB = 6.0, 4, 8
H = TAU / K
sys.path.insert(0, root)
import model as M, quant_attention as QA
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True); c = ck["config"]
kw = dict(block_size=int(c["block_size"]), n_layer=int(c["n_layer"]), n_head=int(c["n_head"]),
          n_embd=int(c["n_embd"]), attn_mode=c["attn"], attn_K=int(c.get("K", 8)),
          attn_approx=c.get("approx", "lerp"), attn_range_grad=c.get("range_grad", "detach"),
          attn_range_mode=c.get("range_mode") or "minmax",
          attn_range_tau=float(c.get("range_tau") or 6.0), attn_tail_policy=policy)
f = getattr(M.GPTConfig, "__dataclass_fields__", {})
m = M.GPT(M.GPTConfig(**{k: v for k, v in kw.items() if k in f}))
sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
m.load_state_dict(sd, strict=False); m.cuda().eval()
nl = int(c["n_layer"])
acc = [dict() for _ in range(nl)]
def add(l, k, v):
    acc[l][k] = acc[l].get(k, 0.0) + float(v)
def rz_kwargs():
    sig = inspect.signature(QA._relevant_zero_reconstruct).parameters
    return {"tail_policy": policy} if "tail_policy" in sig else {}
hooks = []
for li, blk in enumerate(m.transformer.h):
    def hook(mod, inp, out, li=li):
        x = inp[0]; B, T, C = x.shape
        q, k, _ = mod.c_attn(x).split(mod.n_embd, dim=2)
        Hh, D = mod.n_head, mod.head_dim
        q = q.view(B, T, Hh, D).transpose(1, 2).float(); k = k.view(B, T, Hh, D).transpose(1, 2).float()
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        valid = causal.view(1, 1, T, T).expand_as(att)
        mx = att.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
        mn = att.masked_fill(~valid, torch.inf).amin(-1, keepdim=True)
        z = att - mx
        span = (mx - mn).squeeze(-1)
        Pex = torch.softmax(att.masked_fill(~valid, -torch.inf), -1)
        if kind == "exact":
            Pn = Pex; lvl = None; hstep = torch.full_like(span, float("nan"))
        elif kind == "minmax":
            w, _, _, lvl = QA._adaptive_interval_reconstruct(att, causal, int(c["K"]), c["approx"],
                                                            return_level_idx=True, range_grad="full")
            Pn = w / w.sum(-1, keepdim=True); hstep = span / K
        else:
            w = QA._relevant_zero_reconstruct(att, causal, K, "nearest", tau=TAU, return_level_idx=True,
                                              range_grad="full", **rz_kwargs())[0]
            Pn = w / w.sum(-1, keepdim=True); hstep = torch.full_like(span, H)
            zs = torch.where(valid, z, torch.zeros_like(z))
            rr = torch.clamp(torch.floor((zs + TAU) / H), max=K - 1)
            tt = (zs - (-TAU + rr * H)) / H
            lvl = (rr + (tt >= 0.5).to(rr.dtype)).clamp(max=K).round().long()
            if policy == "zero":
                lvl = torch.where(z >= -TAU, lvl, torch.full_like(lvl, -10**6))
        rows = valid.sum(-1) >= 2
        tail = valid & (z < -TAU)
        nrow = rows.sum().item()
        add(li, "rows", nrow)
        add(li, "span", span[rows].double().sum())
        add(li, "native_h", hstep[rows].double().nansum())
        add(li, "tv_native_exact", (0.5 * (Pn - Pex).abs().sum(-1))[rows].double().sum())
        add(li, "tail_keys", tail.double().sum()); add(li, "valid_keys", valid.double().sum())
        add(li, "native_tail_mass", (Pn * tail).sum(-1)[rows].double().sum())
        add(li, "exact_tail_mass", (Pex * tail).sum(-1)[rows].double().sum())
        if lvl is not None:
            top = valid & (lvl == K)
            add(li, "top_level_mass", (Pex * top).sum(-1)[rows].double().sum())
            ok = valid & (lvl > -10**6)
            big = torch.where(ok, lvl, torch.full_like(lvl, 10**7))
            srt, _ = torch.sort(big, -1)
            dist = ((srt[..., 1:] != srt[..., :-1]) & (srt[..., 1:] < 10**7)).sum(-1) + (srt[..., 0] < 10**7).long()
            add(li, "levels_used", dist[rows].double().sum())
    hooks.append(blk.attn.register_forward_hook(hook))
d = np.memmap(val, dtype=np.uint16, mode="r")
with torch.inference_mode():
    for i in range(NB):
        x = torch.from_numpy(np.asarray(d[i*1024:(i+1)*1024], dtype=np.int64)).unsqueeze(0).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): m(x)
out = []
for li, a in enumerate(acc):
    r = a["rows"]
    row = dict(layer=li, span=a["span"]/r, native_h=a["native_h"]/r if kind != "exact" else float("nan"),
               tv_native_exact=a["tv_native_exact"]/r, frac_tail_keys=a["tail_keys"]/a["valid_keys"],
               native_tail_mass=a["native_tail_mass"]/r, exact_tail_mass=a["exact_tail_mass"]/r,
               top_level_mass=a.get("top_level_mass", float("nan"))/r if "top_level_mass" in a else float("nan"),
               levels_used=a.get("levels_used", float("nan"))/r if "levels_used" in a else float("nan"))
    out.append(row)
print(json.dumps(out))
"""


def main():
    man = {r["cond"]: r for r in csv.DictReader(open(MAN), delimiter="\t") if r["seed"] == "1337"}
    E = f"{AM}/D_robustness_124m_100m_5seed/eval_native_v1"
    specs = [("Exact", man["Exact"]["train_code_root"], man["Exact"]["checkpoint_100m"], "exact", "zero"),
             ("MinMax-Weight", man["MinMax-Weight K4"]["train_code_root"], man["MinMax-Weight K4"]["checkpoint_100m"], "minmax", "zero"),
             ("MinMax-Prob", man["MinMax-Prob K4"]["train_code_root"], man["MinMax-Prob K4"]["checkpoint_100m"], "minmax", "zero"),
             ("zero-Weight", f"{PB}/code/rb_tail_weight", man["QRM-Weight K4"]["checkpoint_100m"], "fwm", "zero"),
             ("zero-Prob", f"{PB}/code/rb_tail_prob", man["QRM-Prob K4"]["checkpoint_100m"], "fwm", "zero"),
             ("resolved-Weight", f"{PB}/code/rb_tail_weight",
              f"{PB}/runs_20260921/rb_s1337_tail_resolved_weight_100m/ckpt_step000763_100.0m.pt", "fwm", "resolved"),
             ("resolved-Prob", f"{PB}/code/rb_tail_prob",
              f"{PB}/runs_20260921/rb_s1337_tail_resolved_prob_100m/ckpt_step000763_100.0m.pt", "fwm", "resolved")]
    allrows = []
    for name, root, ckpt, kind, pol in specs:
        r = subprocess.run([sys.executable, "-c", CHILD, root, ckpt, kind, pol,
                            f"{PB}/data/wikitext103/val.bin"], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-2500:]); raise SystemExit(f"failed: {name}")
        for row in json.loads(r.stdout.strip().splitlines()[-1]):
            allrows.append(dict(condition=name, **row))
        print(f"  probed {name}", flush=True)
    cols = ["condition", "layer", "span", "native_h", "tv_native_exact", "frac_tail_keys",
            "native_tail_mass", "exact_tail_mass", "top_level_mass", "levels_used"]
    with open(OUT / "tail_decomp_geometry_by_layer.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n"); w.writeheader(); w.writerows(allrows)
    import pandas as pd
    df = pd.DataFrame(allrows)
    lm = df.groupby("condition", sort=False)[cols[2:]].mean()
    lm.to_csv(OUT / "tail_decomp_geometry_layer_mean.csv")
    pd.set_option("display.width", 220)
    print("\nlayer-mean geometry, endpoint, 8 blocks:")
    print(lm.round(4).to_string())


if __name__ == "__main__":
    main()
