#!/usr/bin/env python3
"""E1 auxiliary: zero-sum residual of the REAL loss score-gradient at each checkpoint.

For every LERP K32 checkpoint (full, detach, min_only, max_only; seed 7) the model is built from its
own frozen training root with its own range_grad mode, the loss on fixed validation blocks (the first
8 of the 243, same blocks as the geometry probe) is back-propagated, and dL/ds is captured on the
scaled score tensor `att` of every layer (before calibration).  Per causal row with >= 2 valid keys:

    rho = |sum_j g_ij| / sum_j |g_ij|        (rows with a zero denominator are counted separately)

Evaluation precision (bf16 autocast outside the fp32 attention kernel, TF32 off), model in eval mode
(no dropout exists).  This is the true-loss gradient residual; the random-upstream structural check is
in ops_check_e1.py and is reported separately.  Local RTX 3090, SUP_LOCAL=1.
"""
import csv, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "checks"))
from common import ValBlocks, build_model, import_code  # noqa: E402
from sup_common import LocalsCapture  # noqa: E402
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


TR = KI_REPO + ""
PB = f"{TR}/training_code/phaseB_base"
SUP = f"{TR}/analysis_main_v1/supplement_four_runs_20260923"
VAL = f"{PB}/data/wikitext103/val.bin"
CK = [(20, "ckpt_step000153_20.1m.pt"), (40, "ckpt_step000306_40.1m.pt"), (60, "ckpt_step000459_60.2m.pt"),
      (80, "ckpt_step000612_80.2m.pt"), (100, "ckpt_step000763_100.0m.pt")]
SEED = int(os.environ.get("SEED", "7"))
RUNS = [("lerp_full", f"{PB}/runs/p3_s7_lerp_k32_fullgrad_100m", f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/code_snapshots/local_phaseB_base"),
        ("lerp_detach", f"{PB}/runs_20260921/rb_s7_lerp_k32_detach_100m", PB),
        ("e1_min", f"{PB}/supplement_runs_20260923/sp_s7_lerp_k32_minonly_100m", f"{SUP}/code/e1_lerp_range_grad_single"),
        ("e1_max", f"{PB}/supplement_runs_20260923/sp_s7_lerp_k32_maxonly_100m", f"{SUP}/code/e1_lerp_range_grad_single")]
RUNS = [(k, rd.replace("_s7_", f"_s{SEED}_"), root) for k, rd, root in RUNS]


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    blocks = ValBlocks(VAL, 1024)
    out = []
    for key, rd, root in RUNS:
        mm, qa = import_code(root)
        for t, name in CK:
            ck = torch.load(f"{rd}/{name}", map_location="cpu", weights_only=False, mmap=True)
            assert int(ck["config"]["seed"]) == SEED, (rd, ck["config"]["seed"])
            model, cfg, op = build_model(mm, ck, "cuda")
            T = 1024
            valid = torch.ones(T, T, dtype=torch.bool, device="cuda").tril()
            nvalid = valid.sum(-1)
            per_layer = {li: [] for li in range(cfg.n_layer)}
            for b in range(8):
                x, y = blocks.batch(b, b + 1, "cuda")
                grads = []

                def cap(fn, d):
                    a = d.get("att")
                    if a is not None and a.requires_grad:
                        a.register_hook(lambda g: grads.append(g.detach().float()) and None)
                model.zero_grad(set_to_none=True)
                with LocalsCapture(["_quant_attention"], ["att"], cap), \
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                grads = list(reversed(grads))
                assert len(grads) == cfg.n_layer, len(grads)
                for li, g in enumerate(grads):
                    g = g.masked_fill(~valid, 0.0)
                    num = g.sum(-1).abs()
                    den = g.abs().sum(-1)
                    keep = (nvalid >= 2).view(1, 1, T).expand_as(den) & (den > 0)
                    per_layer[li].append(torch.where(keep, num / den.clamp_min(1e-300), torch.full_like(den, float("nan"))).flatten().cpu())
            for li, chunks in per_layer.items():
                r = torch.cat(chunks)
                zero = int(torch.isnan(r).sum())
                r = r[~torch.isnan(r)].double().numpy()
                out.append(dict(series=key, tokens_m=t, layer=li, range_grad=op["range_grad"], rows=len(r), rows_excluded=zero,
                                rho_median=float(np.median(r)), rho_p90=float(np.percentile(r, 90)),
                                rho_max=float(r.max()), frac_rho_gt_1e_3=float((r > 1e-3).mean())))
            print(key, t, "median rho (layer mean) %.3e" % np.mean([o["rho_median"] for o in out if o["series"] == key and o["tokens_m"] == t]), flush=True)
            del model, ck
            torch.cuda.empty_cache()
    os.makedirs(f"{SUP}/reports", exist_ok=True)
    suffix = "" if SEED == 7 else f"_s{SEED}"
    with open(f"{SUP}/reports/supplement_e1_zero_sum_loss_gradient{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
