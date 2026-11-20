#!/usr/bin/env python3
"""Evaluate every checkpoint of a tail-decomposition run: 243 WikiText-103 blocks, batch 1,
bf16 autocast outside the fp32 attention kernel, TF32 off, through the run's own new code root."""
import json, os, re, sys
import numpy as np, torch
PB=(KI_REPO + "/"
    "training_code/phaseB_base")
AM=KI_REPO + "/analysis_main_v1"
OUTROOT=f"{AM}/D_robustness_124m_100m_5seed/eval_native_v1"
run, root = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{PB}/code/{root}")
import model as M
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
d=np.memmap(f"{PB}/data/wikitext103/val.bin",dtype=np.uint16,mode="r")
CK=re.compile(r"ckpt_step(\d+)_([\d.]+)m\.pt$")
rd=f"{PB}/runs_20260921/{run}"
files=sorted(f for f in os.listdir(rd) if CK.match(f))
for f in files:
    ckp=os.path.join(rd,f); ck=torch.load(ckp,map_location="cpu",weights_only=False,mmap=True)
    c=ck["config"]
    cfg=M.GPTConfig(block_size=int(c["block_size"]),n_layer=int(c["n_layer"]),n_head=int(c["n_head"]),
                    n_embd=int(c["n_embd"]),attn_mode=c["attn"],attn_K=int(c["K"]),
                    attn_approx=c["approx"],attn_range_grad=c["range_grad"],
                    attn_range_mode=c["range_mode"],attn_range_tau=float(c["range_tau"]),
                    attn_tail_policy=c["tail_policy"])
    m=M.GPT(cfg)
    sd={k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k:v for k,v in ck["model"].items()}
    miss,unexp=m.load_state_dict(sd,strict=False); assert not miss and not unexp
    m.cuda().eval()
    T=int(cfg.block_size); nb=(len(d)-1)//T
    nll=np.empty(nb,dtype=np.float64)
    with torch.inference_mode():
        for i in range(nb):
            s=i*T
            x=torch.from_numpy(np.asarray(d[s:s+T],dtype=np.int64)).unsqueeze(0).cuda()
            y=torch.from_numpy(np.asarray(d[s+1:s+1+T],dtype=np.int64)).unsqueeze(0).cuda()
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): lg,_=m(x)
            nll[i]=torch.nn.functional.cross_entropy(lg.float().view(-1,lg.size(-1)),y.view(-1)).item()
    tok=int(ck["tokens_done"]); stem=os.path.splitext(f)[0]
    od=f"{OUTROOT}/{run}/{round(tok/1e6):03d}m_{stem}"
    os.makedirs(od,exist_ok=True)
    np.save(f"{od}/per_block_nll.npy",nll)
    json.dump(dict(run_name=run,cond=f"tail-{c['tail_policy']}-{'Weight' if root=='rb_tail_weight' else 'Prob'} K4",
                   seed=int(c["seed"]),tail_policy=c["tail_policy"],code_root=f"{PB}/code/{root}",
                   step=int(ck["step"]),tokens_done=tok,is_final=int(stem.endswith("100.0m")),
                   n_blocks=int(nb),mean_nll=float(nll.mean()),checkpoint=ckp),
              open(f"{od}/summary.json","w"),indent=2)
    print(f"  {f}: mean_nll={nll.mean():.6f}  n_blocks={nb}",flush=True)
    del ck,m; torch.cuda.empty_cache()
print("EVAL COMPLETE")
