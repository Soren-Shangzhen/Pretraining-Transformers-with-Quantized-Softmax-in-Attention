#!/usr/bin/env python3
"""GPU checks for the centroid_v1 root before training (local RTX 3090, bta env).

G1  CUDA fp32 module output vs CPU fp64 evaluation of the same root (Centroid / Centroid-2, MinMax / FW)
G2  bitwise run-to-run determinism of one GPU forward+backward of the 124M model (Centroid rules)
G3  step-0 state with the training recipe (seed 7, 124M, first micro-batch of the seed-7 train stream):
    initial loss, all gradients finite, Q/K gradient norm; throughput and peak memory of 8 micro-steps
    (bf16 autocast, TF32 on, as in train.py) for softmax-exact, FW-Nearest (Weight-STE root) and the new rules
Output: checks/check_gpu_step0.json
"""
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))  # repository root

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
NEW_ROOT = os.path.join(C, "code", "centroid_root")
PB = (KI_REPO + "/training_code/phaseB_base")
SRC_ROOT = f"{PB}/code/phaseB_rr"
DATA = f"{PB}/data/wikitext103"


def import_root(root):
    for m in ("model", "quant_attention"):
        sys.modules.pop(m, None)
    sys.path.insert(0, root)
    try:
        mm = importlib.import_module("model")
        qa = importlib.import_module("quant_attention")
    finally:
        sys.path.remove(root)
    return mm, qa


def first_batch(seed, device):
    data = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(data) - 1024 - 1, 1)
    x = np.stack([data[i:i + 1024] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + 1024] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def main():
    res = {}
    mm, qa = import_root(NEW_ROOT)
    # ---------------- G1 ----------------
    g1 = {}
    rng = np.random.default_rng(7)
    T = 256
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    for rule in ("centroid", "centroid2"):
        for rm in ("minmax", "relevant_zero"):
            for scale in (1.0, 4.0):
                att64 = torch.from_numpy(rng.normal(0, scale, size=(1, 4, T, T)))
                v64 = torch.from_numpy(rng.normal(size=(1, 4, T, 16)))
                w64, *_ = qa._centroid_reconstruct(att64, causal, 4, rule, range_mode=rm, tau=6.0)
                y64 = (w64 / w64.sum(-1, keepdim=True)) @ v64
                att32 = att64.float().cuda()
                w32, *_ = qa._centroid_reconstruct(att32, causal.cuda(), 4, rule, range_mode=rm, tau=6.0)
                y32 = ((w32 / w32.sum(-1, keepdim=True)) @ v64.float().cuda()).double().cpu()
                # cell membership can differ for scores within fp32 rounding of a boundary; report both
                lev64 = qa._centroid_reconstruct(att64, causal, 4, rule, range_mode=rm, tau=6.0)[3]
                lev32 = qa._centroid_reconstruct(att32, causal.cuda(), 4, rule, range_mode=rm, tau=6.0)[3].cpu()
                g1[f"{rule}/{rm}/scale{scale}"] = dict(
                    max_rel_output_err=float((y32 - y64).abs().max() / y64.abs().max()),
                    level_mismatch_fraction=float((lev32 != lev64)[causal.expand_as(lev64)].float().mean()))
    res["G1_gpu_fp32_vs_cpu_fp64"] = g1

    # ---------------- G2 + G3 ----------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    configs = {
        "exact_fp32_softmax": (NEW_ROOT, dict(attn_mode="exact_fp32_softmax")),
        "fw_nearest_weightste_k4 (phaseB_rr)": (SRC_ROOT, dict(attn_mode="quant", attn_K=4, attn_approx="nearest",
                                                               attn_range_grad="full", attn_range_mode="relevant_zero")),
        "fw_centroid_k4": (NEW_ROOT, dict(attn_mode="quant", attn_K=4, attn_approx="centroid", attn_range_grad="full",
                                          attn_range_mode="relevant_zero")),
        "fw_centroid2_k4": (NEW_ROOT, dict(attn_mode="quant", attn_K=4, attn_approx="centroid2",
                                           attn_range_grad="full", attn_range_mode="relevant_zero")),
        "minmax_centroid_k4": (NEW_ROOT, dict(attn_mode="quant", attn_K=4, attn_approx="centroid",
                                              attn_range_grad="full", attn_range_mode="minmax")),
        "minmax_centroid2_k4": (NEW_ROOT, dict(attn_mode="quant", attn_K=4, attn_approx="centroid2",
                                               attn_range_grad="full", attn_range_mode="minmax")),
    }
    g2, g3 = {}, {}
    x, y = first_batch(7, "cuda")
    for name, (root, kw) in configs.items():
        mm, _ = import_root(root)
        torch.manual_seed(7)
        np.random.seed(7)
        cfg = mm.GPTConfig(block_size=1024, n_layer=12, n_head=12, n_embd=768, **kw)
        model = mm.GPT(cfg).cuda()
        model.train()
        ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        grads = []
        for rep in range(2):
            model.zero_grad(set_to_none=True)
            with ac:
                _, loss = model(x, y)
            loss.backward()
            grads.append((float(loss), [p.grad.detach().clone() for p in model.parameters()]))
        det = grads[0][0] == grads[1][0] and all(torch.equal(a, b) for a, b in zip(grads[0][1], grads[1][1]))
        finite = all(torch.isfinite(g).all().item() for g in grads[0][1])
        qkw = model.transformer.h[0].attn.c_attn.weight.grad[:2 * 768]
        g2[name] = bool(det)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(8):
            with ac:
                _, l2 = model(x, y)
            l2.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        g3[name] = dict(step0_loss=grads[0][0], grads_finite=bool(finite), layer0_qk_grad_l2=float(qkw.norm()),
                        tok_per_s=8 * 1024 / dt, peak_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
        print(name, g3[name], "deterministic", det, flush=True)
        del model, grads
        torch.cuda.empty_cache()
    res["G2_bitwise_determinism_two_runs"] = g2
    res["G3_step0_and_throughput"] = g3
    res["env"] = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    out = os.path.join(C, "checks", "check_gpu_step0.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(g1, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
