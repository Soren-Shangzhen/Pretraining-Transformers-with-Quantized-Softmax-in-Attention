#!/usr/bin/env python3
"""Round-3 pre-training checks for the reduced-precision softmax baselines (local RTX 3090, bta env).

1. Operator level (real module path, fp32 attention, TF32 off): for random q/k/v,
   - the round-3 root's exact_fp32_softmax output is bit-identical to the phaseB base root's;
   - softmax_bf16exp / softmax_fp8e4m3exp outputs equal an independent re-computation of their definition
     (w = exp(bf16(z)) resp. e4m3(exp(z)), P = w / sum w, y = P @ v, masked positions zero);
   - their gradients w.r.t. q, k, v are finite and not None; relative deviation from exact reported.
2. Step 0 at training precision (TF32 on, bf16 autocast, eager, no grad checkpointing), seed-7 canonical init
   and the real first update (128 microbatches): loss_acc for exact (both roots), bf16-exp and fp8-exp;
   exact compared with the historical seed-7 log value (printed precision only); finite, non-None param grads.
"""
import importlib.util, json, math, os, sys
os.environ["SUP_LOCAL"] = "1"
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sup_common import compare, env_record, sha256_file, state_sha, tensor_sha  # noqa: E402
from step0_check import first_step_batches, ACCUM  # noqa: E402

SUP = os.path.dirname(os.path.dirname(HERE))
ROOT_NEW = f"{SUP}/code/r3_lowprec_softmax"
ROOT_BASE = f"{SUP}/../D_robustness_124m_100m_5seed/code_snapshots/local_phaseB_base"
MODES = ["exact_fp32_softmax", "softmax_bf16exp", "softmax_fp8e4m3exp"]


def load(root, tag):
    spec = importlib.util.spec_from_file_location(f"qa_{tag}", f"{root}/quant_attention.py")
    qa = importlib.util.module_from_spec(spec); spec.loader.exec_module(qa)
    saved = sys.modules.get("quant_attention"); sys.modules["quant_attention"] = qa
    try:
        spec = importlib.util.spec_from_file_location(f"mm_{tag}", f"{root}/model.py")
        mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
    finally:
        if saved is None:
            sys.modules.pop("quant_attention", None)
        else:
            sys.modules["quant_attention"] = saved
    return qa, mm


def reference(q, k, v, mode):
    D = q.shape[-1]; T = q.shape[-2]
    att = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(D)
    valid = torch.ones(T, T, dtype=torch.bool, device=q.device).tril().view(1, 1, T, T)
    z = att.masked_fill(~valid, -torch.inf) - att.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
    if mode == "softmax_bf16exp":
        w = torch.exp(z.to(torch.bfloat16)).float()
    else:
        w = torch.exp(z).to(torch.float8_e4m3fn).float()
    w = w.masked_fill(~valid, 0.0)
    return (w / w.sum(-1, keepdim=True)) @ v.float()


def main():
    out = {"script": os.path.abspath(__file__), "script_sha256": sha256_file(__file__), "env": env_record(),
           "roots": {"new": {f: sha256_file(f"{ROOT_NEW}/{f}") for f in ("model.py", "quant_attention.py", "train.py")},
                     "base": {f: sha256_file(f"{ROOT_BASE}/{f}") for f in ("model.py", "quant_attention.py")}}}
    qa_n, mm_n = load(ROOT_NEW, "new"); qa_b, mm_b = load(ROOT_BASE, "base")
    # ---------------- 1. operator level
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    op = {}
    for dev in ("cpu", "cuda"):
        g = torch.Generator(device="cpu").manual_seed(3)
        for scale in (1.0, 4.0):
            q, k, v = [(torch.randn(1, 12, 256, 64, generator=g) * scale).to(dev) for _ in range(3)]
            res = {}
            for mode in MODES:
                for tag, mm in (("new", mm_n), ("base", mm_b)):
                    if tag == "base" and mode != "exact_fp32_softmax":
                        continue
                    cfg = mm.GPTConfig(n_layer=1, n_head=12, n_embd=768, attn_mode=mode)
                    mod = mm.QuantCausalSelfAttention(cfg).to(dev)
                    qq, kk, vv = [t.clone().requires_grad_(True) for t in (q, k, v)]
                    fn = mod._exact_fp32_softmax if mode == "exact_fp32_softmax" else (lambda a, b, c, m=mode: mod._softmax_reduced_exp(a, b, c, m.replace("softmax_", "")))
                    y = fn(qq, kk, vv)
                    y.float().pow(2).sum().backward()
                    res[(mode, tag)] = (y.detach(), qq.grad, kk.grad, vv.grad)
            key = f"{dev}/scale{scale}"
            op[key] = {"exact_new_vs_base_fwd": compare(res[("exact_fp32_softmax", "new")][0], res[("exact_fp32_softmax", "base")][0])["eq_rawbits"],
                       "exact_new_vs_base_grads": all(compare(a, b)["eq_rawbits"] for a, b in zip(res[("exact_fp32_softmax", "new")][1:], res[("exact_fp32_softmax", "base")][1:]))}
            for mode in MODES[1:]:
                y, gq, gk, gv = res[(mode, "new")]
                ye = res[("exact_fp32_softmax", "new")][0]
                op[key][mode] = {"fwd_equals_definition": compare(y, reference(q, k, v, mode))["eq_rawbits"],
                                 "rel_dev_vs_exact_output": float((y - ye).norm() / ye.norm()),
                                 "grads_finite_not_none": all(t is not None and bool(torch.isfinite(t).all()) for t in (gq, gk, gv)),
                                 "grad_q_rel_dev_vs_exact": float((gq - res[("exact_fp32_softmax", "new")][1]).norm() / res[("exact_fp32_softmax", "new")][1].norm())}
    out["operator"] = op
    print(json.dumps(op, indent=1), flush=True)
    # ---------------- 2. step 0 at training precision
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(7); np.random.seed(7)
    init = mm_b.GPT(mm_b.GPTConfig(attn_mode="exact_fp32_softmax")).state_dict()
    torch.manual_seed(7); np.random.seed(7)
    init_new = mm_n.GPT(mm_n.GPTConfig(attn_mode="softmax_fp8e4m3exp")).state_dict()
    out["step0"] = {"init_sha_base": state_sha(init), "init_sha_new_root": state_sha(init_new),
                    "init_equal": state_sha(init) == state_sha(init_new)}
    xs, ys = first_step_batches()
    out["step0"]["batch_x_sha256"] = tensor_sha(torch.cat(xs))
    for tag, mm, mode in (("exact_base", mm_b, "exact_fp32_softmax"), ("exact_new", mm_n, "exact_fp32_softmax"),
                          ("bf16exp", mm_n, "softmax_bf16exp"), ("fp8e4m3exp", mm_n, "softmax_fp8e4m3exp")):
        m = mm.GPT(mm.GPTConfig(attn_mode=mode)); m.load_state_dict(init); m.to("cuda").train()
        la = 0.0; losses = []
        for i in range(ACCUM):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = m(xs[i].cuda(), ys[i].cuda())
            (loss / ACCUM).backward()
            la += loss.item() / ACCUM; losses.append(loss.detach())
        gr = [p.grad for p in m.parameters()]
        attn_w = [p.grad for n, p in m.named_parameters() if n.endswith("attn.c_attn.weight")]
        qk_rows = torch.cat([g[:1536].reshape(-1) for g in attn_w])   # Q and K rows of the fused projection
        out.setdefault("step0_qk_grad", {})[tag] = {"qk_grad_l2": float(qk_rows.float().norm()),
                                                    "qk_grad_frac_exact_zero": float((qk_rows == 0).float().mean())}
        out["step0"][tag] = {"loss_acc": la, "loss_acc_5dp": f"{la:.5f}",
                             "grads_finite_not_none": all(g is not None and bool(torch.isfinite(g).all()) for g in gr),
                             "grad_norm_preclip": float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)),
                             "first_loss_sha": tensor_sha(losses[0].reshape(1))}
        print(tag, out["step0"][tag], flush=True)
        del m; torch.cuda.empty_cache()
    out["step0"]["historical_exact_s7_log_step1_train_loss"] = "10.93436 (runs/p3_s7_exact_100m/log.csv, printed to 5 dp)"
    json.dump(out, open(f"{SUP}/round3/checks/check_r3_lowprec.json", "w"), indent=1, default=str)
    print("wrote")


if __name__ == "__main__":
    main()
