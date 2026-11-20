"""Diagnostic (not a training run): step-0 gradients of the seed-7 init on the real first update when the FP8 e4m3
exp uses PyTorch's native backward (the gradient is cast to e4m3), versus the fp32-backward variant used for
training.  Training precision (TF32 on, bf16 autocast)."""
import json, os, sys
os.environ["SUP_LOCAL"] = "1"
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from check_r3_lowprec import load  # noqa: E402
from step0_check import first_step_batches, ACCUM  # noqa: E402
from sup_common import sha256_file  # noqa: E402
SUP = os.path.dirname(os.path.dirname(HERE))
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
xs, ys = first_step_batches()
out = {"script_sha256": sha256_file(__file__)}
for tag, root in (("native_fp8_backward", f"{SUP}/code/r3_diag_native_fp8"), ("fp32_backward_used_for_training", f"{SUP}/code/r3_lowprec_softmax")):
    qa, mm = load(root, tag)
    torch.manual_seed(7); np.random.seed(7)
    m = mm.GPT(mm.GPTConfig(attn_mode="softmax_fp8e4m3exp")).to("cuda").train()
    for i in range(ACCUM):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = m(xs[i].cuda(), ys[i].cuda())
        (loss / ACCUM).backward()
    qk = torch.cat([p.grad[:1536].reshape(-1) for n, p in m.named_parameters() if n.endswith("attn.c_attn.weight")])
    out[tag] = {"root_quant_attention_sha256": sha256_file(f"{root}/quant_attention.py"), "qk_grad_l2": float(qk.norm()),
                "qk_grad_frac_exact_zero": float((qk == 0).float().mean()),
                "all_grads_finite": all(bool(torch.isfinite(p.grad).all()) for p in m.parameters())}
    print(tag, out[tag], flush=True)
    del m; torch.cuda.empty_cache()
json.dump(out, open(f"{SUP}/round3/checks/diag_native_fp8.json", "w"), indent=1)
