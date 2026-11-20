"""GPU smoke: every rb-root_v1 operator mode builds, runs forward+backward on a small GPT, finite."""
import sys, time, torch
from rb_common import import_root, RB_ROOT
mm, qa = import_root(RB_ROOT)
dev = "cuda"
MODES = [
    dict(attn_mode="exact_fp32_softmax"),
    dict(attn_mode="softmax_bf16exp"), dict(attn_mode="softmax_fp8e4m3exp"),
    dict(attn_mode="softmax_mxfp8"), dict(attn_mode="softmax_nvfp4"), dict(attn_mode="softmax_uint8"),
    dict(attn_mode="softmax_bf16chain"),
    dict(attn_mode="indexsoftmax", attn_variant="fp"), dict(attn_mode="indexsoftmax", attn_variant="u8"),
    dict(attn_mode="exaq", attn_variant="detach"), dict(attn_mode="exaq", attn_variant="graph"),
    dict(attn_mode="quant", attn_approx="lerp", attn_K=4, attn_range_mode="relevant_zero", attn_range_grad="full"),
    dict(attn_mode="quant", attn_approx="lerp", attn_K=4, attn_range_mode="relevant_zero", attn_range_grad="detach"),
    dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_mode="relevant_zero", attn_range_grad="full", _sur="weight_strict"),
    dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_mode="minmax", attn_range_grad="full", _sur="prob"),
    dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_mode="minmax", attn_range_grad="full", _sur="weight_strict"),
    dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_mode="relevant_zero", attn_range_grad="full",
         attn_tail_policy="resolved", attn_range_tau_per_layer=(4*1.7, 4*2.3), _sur="weight_strict"),
    dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_mode="ema", attn_range_grad="full", _sur="weight_strict"),
    dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_mode="ema", attn_range_grad="full", _sur="prob"),
]
ok = True
for m in MODES:
    m = dict(m); sur = m.pop("_sur", "prob"); qa.NEAREST_SURROGATE = sur
    torch.manual_seed(0)
    cfg = mm.GPTConfig(block_size=256, n_layer=2, n_head=4, n_embd=128, vocab_size=512, **m)
    model = mm.GPT(cfg).to(dev)
    x = torch.randint(0, 512, (1, 256), device=dev); y = torch.randint(0, 512, (1, 256), device=dev)
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
    fin = torch.isfinite(loss).item() and gn == gn and gn < float("inf")
    ok &= fin
    print(f"{'OK ' if fin else 'BAD'} {m} sur={sur} loss={loss.item():.4f} gnorm={gn:.4f} {time.time()-t0:.2f}s")
print("ALL OK" if ok else "FAILURES"); sys.exit(0 if ok else 1)
