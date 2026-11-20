#!/usr/bin/env python3
"""Pre-training measurements, written BEFORE any resolved run exists (§4).

On the suite D seed-1337 Exact final checkpoint, over 8 validation blocks:
  - n_tail  = #{causal-valid keys with z < -tau} per row, by causal-position band
  - the minimum tail mass `floor` would inject: n_tail*e^-tau / (1 + n_tail*e^-tau)
  - the number of distinct grid levels `resolved` spends per row (it is NOT a K+1-level quantiser)
"""
import csv, json, math, sys, datetime as dt
import numpy as np, torch

PB = (KI_REPO + "/"
      "training_code/phaseB_base")
AM = KI_REPO + "/analysis_main_v1"
MAN = f"{AM}/D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv"
TAU, K, NB = 6.0, 4, 8
H = TAU / K

r = {x["cond"]: x for x in csv.DictReader(open(MAN), delimiter="\t") if x["seed"] == "1337"}["Exact"]
sys.path.insert(0, r["train_code_root"])
import model as M
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

torch.backends.cuda.matmul.allow_tf32 = False
ck = torch.load(r["checkpoint_100m"], map_location="cpu", weights_only=False, mmap=True)
c = ck["config"]
kw = dict(block_size=int(c["block_size"]), n_layer=int(c["n_layer"]), n_head=int(c["n_head"]),
          n_embd=int(c["n_embd"]), attn_mode=c["attn"], attn_K=int(c.get("K", 8)),
          attn_approx=c.get("approx", "lerp"), attn_range_grad=c.get("range_grad", "detach"))
# the legacy exact_fp32 root predates some config fields
fields = getattr(M.GPTConfig, "__dataclass_fields__", {})
kw = {k: v for k, v in kw.items() if k in fields}
cfg = M.GPTConfig(**kw)
m = M.GPT(cfg)
sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
m.load_state_dict(sd, strict=False); m.cuda().eval()

rows = []
def hook(mod, inp, out):
    x = inp[0]; B, T, C = x.shape
    q, k, _ = mod.c_attn(x).split(mod.n_embd, dim=2)
    Hd, D = mod.n_head, mod.head_dim
    q = q.view(B, T, Hd, D).transpose(1, 2).float(); k = k.view(B, T, Hd, D).transpose(1, 2).float()
    att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril().view(1, 1, T, T)
    mx = att.masked_fill(~causal, -torch.inf).amax(-1, keepdim=True)
    z = att - mx
    valid = causal.expand_as(z)
    tail = valid & (z < -TAU)
    n_tail = tail.sum(-1).float().flatten().cpu().numpy()
    # levels resolved spends: distinct floor((z+tau)/h) over valid keys, +1 boundary
    ridx = torch.floor((z + TAU) / H).clamp(max=K - 1)
    ridx = torch.where(valid, ridx, torch.full_like(ridx, float("nan")))
    span = (mx - att.masked_fill(~causal, torch.inf).amin(-1, keepdim=True)).float().flatten().cpu().numpy()
    rmin = torch.nan_to_num(ridx, nan=float("inf")).amin(-1).float().flatten().cpu().numpy()
    pos = np.tile(np.arange(T), B * Hd)
    rows.append((n_tail, pos, span, rmin))

hs = [b.attn.register_forward_hook(hook) for b in m.transformer.h]
d = np.memmap(f"{PB}/data/wikitext103/val.bin", dtype=np.uint16, mode="r")
with torch.inference_mode():
    for i in range(NB):
        x = torch.from_numpy(np.asarray(d[i*1024:(i+1)*1024], dtype=np.int64)).unsqueeze(0).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): m(x)
for h in hs: h.remove()

n_tail = np.concatenate([r[0] for r in rows]); pos = np.concatenate([r[1] for r in rows])
span = np.concatenate([r[2] for r in rows]); rmin = np.concatenate([r[3] for r in rows])
e = math.exp(-TAU)
dil = n_tail * e / (1 + n_tail * e)
bands = [(0, 16), (16, 64), (64, 256), (256, 512), (512, 1024)]
out = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
       "source_checkpoint": r["checkpoint_100m"], "code_root": r["train_code_root"],
       "n_blocks": NB, "tau": TAU, "K": K, "h": H, "n_rows": int(len(n_tail)),
       "overall": {"n_tail_mean": float(n_tail.mean()), "n_tail_median": float(np.median(n_tail)),
                   "frac_rows_with_no_tail": float((n_tail == 0).mean()),
                   "floor_tail_mass_mean": float(dil.mean()),
                   "floor_tail_mass_median": float(np.median(dil)),
                   "floor_tail_mass_p99": float(np.percentile(dil, 99)),
                   "span_mean": float(span.mean()),
                   "resolved_levels_mean": float((K - rmin[np.isfinite(rmin)]).mean()),
                   "resolved_levels_p99": float(np.percentile(K - rmin[np.isfinite(rmin)], 99))},
       "by_causal_position": [], "author_predictions": "TO BE SUPPLIED BY THE AUTHOR"}
for lo, hi in bands:
    s = (pos >= lo) & (pos < hi)
    if not s.any(): continue
    out["by_causal_position"].append(
        {"band": f"[{lo},{hi})", "n_rows": int(s.sum()), "n_tail_mean": float(n_tail[s].mean()),
         "frac_no_tail": float((n_tail[s] == 0).mean()),
         "floor_tail_mass_mean": float(dil[s].mean()),
         "resolved_levels_mean": float((K - rmin[s][np.isfinite(rmin[s])]).mean())})
json.dump(out, open("prediction_pretraining.json", "w"), indent=2)
print(json.dumps(out, indent=2))
