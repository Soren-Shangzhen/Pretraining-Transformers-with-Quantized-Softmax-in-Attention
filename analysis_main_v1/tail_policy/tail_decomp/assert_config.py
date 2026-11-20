#!/usr/bin/env python3
"""§5 pre-training assertion: the new run's config must equal suite D FWM-Weight K4 field for field,
except tail_policy / run_name / out_dir (and data_dir, which differs only by absolute vs relative path)."""
import csv, importlib, sys, torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository

PB=(KI_REPO + "/"
    "training_code/phaseB_base")
AM=KI_REPO + "/analysis_main_v1"
MAN=f"{AM}/D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv"
root, trainer, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
ref=torch.load({x["cond"]:x for x in csv.DictReader(open(MAN),delimiter="\t")
                if x["seed"]=="1337"}["QRM-Weight K4"]["checkpoint_100m"],
               map_location="cpu", weights_only=False, mmap=True)["config"]
sys.path.insert(0,f"{PB}/code/{root}")
mod=importlib.import_module(trainer.replace(".py",""))
sys.argv=[trainer]+argv
new=vars(mod.get_args())
EXEMPT={"tail_policy","run_name","out_dir","data_dir","resume","overwrite_run"}
bad=[]
for k,v in ref.items():
    if k in EXEMPT: continue
    if k not in new: bad.append(f"{k}: MISSING from new args (ref={v!r})"); continue
    if str(new[k])!=str(v): bad.append(f"{k}: new={new[k]!r} vs suiteD={v!r}")
extra=[k for k in new if k not in ref and k not in EXEMPT]
print(f"reference: suite D FWM-Weight K4 seed 1337 ({len(ref)} fields)")
print(f"new args : {root}/{trainer} ({len(new)} fields)")
for k in ("grad_checkpointing","compile","K","approx","range_mode","range_tau","range_grad","seed",
          "max_tokens","warmup_tokens","batch_tokens","micro_batch","lr","min_lr_ratio",
          "weight_decay","beta1","beta2","grad_clip","eval_every_tokens","eval_batches",
          "stats_batches","ckpt_every_tokens","n_layer","n_head","n_embd","block_size"):
    if k in ref: print(f"  {k:22s} suiteD={str(ref[k]):<14s} new={str(new.get(k,'—')):<14s} "
                       f"{'OK' if str(new.get(k))==str(ref[k]) else 'MISMATCH'}")
print(f"  {'tail_policy':22s} suiteD={'(absent)':<14s} new={str(new.get('tail_policy')):<14s} (intended difference)")
if extra: print(f"  new-only fields (allowed): {extra}")
if bad:
    print("\nCONFIG ASSERTION FAILED:"); [print("  -",b) for b in bad]; sys.exit(1)
print("\nCONFIG ASSERTION PASS — field-for-field match except the intended difference")
