"""Gradient-checkpointing parity test for Phase-B/Phase-C training code.

Purpose
-------
Verify that toggling GPT.grad_checkpointing changes only the memory/compute
strategy, not the mathematical training signal, for the current quantized
attention experiment.

The test uses:
    attn_mode       = quant
    approx          = nearest
    range_grad      = full
    K               = 16 by default
    full 12-layer / 768-dim / 12-head GPT architecture
    shorter sequence length only to make the parity test fast

It compares, on the SAME model parameters and SAME input batch:
    1. forward loss
    2. sampled logits
    3. gradients for EVERY trainable parameter
    4. global gradient cosine similarity
    5. global relative L2 gradient error
    6. worst per-parameter absolute / relative errors

Run from files_adaptive_quant_phaseB:
    python test_grad_checkpoint_parity.py

Optional:
    python test_grad_checkpoint_parity.py --seq_len 128 --K 16 --device cuda

Expected result:
    PASS, with loss/logit differences zero or at numerical roundoff and
    gradient cosine extremely close to 1.

Notes
-----
- This intentionally does NOT run an optimizer step. Gradient checkpointing
  acts before the optimizer; if forward values and all parameter gradients
  match, AdamW will receive the same training signal.
- The first set of gradients is copied to CPU for exact all-parameter
  comparison, requiring roughly 0.5 GB of host RAM for this 124M model.
"""

from __future__ import annotations

import argparse
import gc
import math
import time

import torch
import torch.utils.checkpoint
from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--atol", type=float, default=2e-6)
    p.add_argument("--rel_l2_tol", type=float, default=2e-6)
    p.add_argument("--cos_tol", type=float, default=0.999999)
    return p.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_mem(device):
    if device.type != "cuda":
        return None
    return (
        torch.cuda.memory_allocated(device) / 1024**3,
        torch.cuda.max_memory_allocated(device) / 1024**3,
    )


def make_batch(seq_len, vocab_size, device, seed):
    # Generate on CPU so input generation is completely independent of CUDA RNG.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 100003)
    idx = torch.randint(
        0, vocab_size, (1, seq_len), generator=g, dtype=torch.long
    ).to(device)
    targets = torch.randint(
        0, vocab_size, (1, seq_len), generator=g, dtype=torch.long
    ).to(device)
    return idx, targets


def run_once(model, idx, targets, use_checkpoint, device):
    model.zero_grad(set_to_none=True)
    model.grad_checkpointing = use_checkpoint
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sync(device)
    t0 = time.perf_counter()

    logits, loss = model(idx, targets)
    # Save a deterministic subset of logits before backward.
    # Full logits are not retained because only a forward fingerprint is needed.
    flat = logits.detach().reshape(-1)
    if flat.numel() <= 4096:
        sample = flat.float().cpu().clone()
    else:
        sample_idx = torch.linspace(
            0, flat.numel() - 1, steps=4096, device=flat.device
        ).long()
        sample = flat[sample_idx].float().cpu().clone()

    loss.backward()

    sync(device)
    elapsed = time.perf_counter() - t0
    mem = gpu_mem(device)

    return float(loss.detach()), sample, elapsed, mem


def copy_gradients_to_cpu(model):
    refs = {}
    missing = []
    total_numel = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
            continue
        g = p.grad.detach()
        refs[name] = g.cpu().clone()
        total_numel += g.numel()

    return refs, missing, total_numel


def compare_gradients(model, refs):
    # Global metrics accumulated in float64 to avoid the comparison itself
    # introducing meaningful reduction error.
    dot = 0.0
    ref_sq = 0.0
    cur_sq = 0.0
    diff_sq = 0.0

    worst_abs = (-1.0, None)
    worst_rel_l2 = (-1.0, None)
    exact_params = 0
    compared = 0
    rows = []

    for name, p in model.named_parameters():
        if not p.requires_grad or name not in refs:
            continue
        if p.grad is None:
            raise RuntimeError(f"no-checkpoint run has no gradient for {name}")

        ref = refs[name]
        cur = p.grad.detach().cpu()

        if ref.shape != cur.shape:
            raise RuntimeError(
                f"shape mismatch for {name}: {tuple(ref.shape)} vs {tuple(cur.shape)}"
            )

        d = cur - ref
        max_abs = float(d.abs().max())
        ref_norm = float(torch.linalg.vector_norm(ref.double()))
        cur_norm = float(torch.linalg.vector_norm(cur.double()))
        diff_norm = float(torch.linalg.vector_norm(d.double()))
        rel_l2 = diff_norm / (ref_norm + 1e-30)

        if torch.equal(ref, cur):
            exact_params += 1

        compared += 1
        rows.append((max_abs, rel_l2, name, ref_norm, cur_norm))

        if max_abs > worst_abs[0]:
            worst_abs = (max_abs, name)
        if rel_l2 > worst_rel_l2[0]:
            worst_rel_l2 = (rel_l2, name)

        rd = ref.double().reshape(-1)
        cd = cur.double().reshape(-1)
        dd = d.double().reshape(-1)

        dot += float(torch.dot(rd, cd))
        ref_sq += float(torch.dot(rd, rd))
        cur_sq += float(torch.dot(cd, cd))
        diff_sq += float(torch.dot(dd, dd))

    global_ref_norm = math.sqrt(ref_sq)
    global_cur_norm = math.sqrt(cur_sq)
    global_diff_norm = math.sqrt(diff_sq)
    global_rel_l2 = global_diff_norm / (global_ref_norm + 1e-30)
    global_cos = dot / (
        (global_ref_norm * global_cur_norm) + 1e-300
    )

    rows.sort(reverse=True, key=lambda x: x[0])

    return {
        "compared": compared,
        "exact_params": exact_params,
        "worst_abs": worst_abs,
        "worst_rel_l2": worst_rel_l2,
        "global_ref_norm": global_ref_norm,
        "global_cur_norm": global_cur_norm,
        "global_diff_norm": global_diff_norm,
        "global_rel_l2": global_rel_l2,
        "global_cos": global_cos,
        "top_abs": rows[:10],
    }


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    device = torch.device(args.device)

    print("=" * 108)
    print("Gradient checkpointing parity test")
    print("=" * 108)
    print(f"torch             : {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU               : {torch.cuda.get_device_name(device)}")
    print(f"device            : {device}")
    print(f"seq_len           : {args.seq_len}")
    print(f"K                 : {args.K}")
    print("attention         : quant / nearest / range_grad=full")
    print("architecture      : 12 layers / 12 heads / d_model=768")
    print()

    # Deterministic initialization.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cfg = GPTConfig(
        block_size=args.seq_len,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        bias=False,
        attn_mode="quant",
        attn_K=args.K,
        attn_approx="nearest",
        attn_range_grad="full",
    )

    model = GPT(cfg).to(device)
    idx, targets = make_batch(
        args.seq_len, cfg.vocab_size, device, args.seed
    )

    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params  : {nparams / 1e6:.2f}M")
    print(
        f"CPU grad reference: ~{nparams * 4 / 1024**3:.2f} GiB "
        "(FP32)"
    )

    # ------------------------------------------------------------------
    # Run A: checkpointing ON.
    # ------------------------------------------------------------------
    print("\n[1/3] Forward + backward with gradient checkpointing ON...")
    loss_gc, logits_gc, time_gc, mem_gc = run_once(
        model, idx, targets, True, device
    )
    refs, missing_gc, grad_numel = copy_gradients_to_cpu(model)

    print(f"  loss             = {loss_gc:.10f}")
    print(f"  elapsed          = {time_gc:.3f} s")
    if mem_gc:
        print(
            f"  CUDA allocated   = {mem_gc[0]:.2f} GiB | "
            f"peak={mem_gc[1]:.2f} GiB"
        )
    print(f"  gradient elements= {grad_numel:,}")
    if missing_gc:
        print(f"  WARNING: missing gradients ({len(missing_gc)}):")
        for n in missing_gc[:10]:
            print(f"    {n}")

    # Free GPU gradient tensors before the no-checkpoint run.
    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Run B: checkpointing OFF.
    # ------------------------------------------------------------------
    print("\n[2/3] Forward + backward with gradient checkpointing OFF...")
    loss_no, logits_no, time_no, mem_no = run_once(
        model, idx, targets, False, device
    )

    print(f"  loss             = {loss_no:.10f}")
    print(f"  elapsed          = {time_no:.3f} s")
    if mem_no:
        print(
            f"  CUDA allocated   = {mem_no[0]:.2f} GiB | "
            f"peak={mem_no[1]:.2f} GiB"
        )

    # ------------------------------------------------------------------
    # Compare.
    # ------------------------------------------------------------------
    print("\n[3/3] Comparing forward values and EVERY parameter gradient...")

    loss_abs = abs(loss_gc - loss_no)
    logits_diff = (logits_gc - logits_no).abs()
    logits_max_abs = float(logits_diff.max())
    logits_mean_abs = float(logits_diff.mean())

    metrics = compare_gradients(model, refs)

    print("\nForward parity")
    print(f"  |loss_gc-loss_no|     = {loss_abs:.3e}")
    print(f"  sampled logits maxabs = {logits_max_abs:.3e}")
    print(f"  sampled logits meanabs= {logits_mean_abs:.3e}")

    print("\nGradient parity")
    print(
        f"  compared parameters   = {metrics['compared']}"
    )
    print(
        f"  bitwise-equal params  = "
        f"{metrics['exact_params']}/{metrics['compared']}"
    )
    print(f"  global grad norm (GC) = {metrics['global_ref_norm']:.12e}")
    print(f"  global grad norm (no) = {metrics['global_cur_norm']:.12e}")
    print(f"  global diff norm      = {metrics['global_diff_norm']:.12e}")
    print(f"  global relative L2    = {metrics['global_rel_l2']:.3e}")
    print(f"  global cosine         = {metrics['global_cos']:.12f}")
    print(
        f"  worst max-abs error   = {metrics['worst_abs'][0]:.3e} "
        f"({metrics['worst_abs'][1]})"
    )
    print(
        f"  worst per-param relL2 = {metrics['worst_rel_l2'][0]:.3e} "
        f"({metrics['worst_rel_l2'][1]})"
    )

    print("\nTop parameters by max absolute gradient difference")
    for max_abs, rel_l2, name, ref_norm, cur_norm in metrics["top_abs"]:
        print(
            f"  {name:55s} "
            f"maxabs={max_abs:.3e} relL2={rel_l2:.3e}"
        )

    # We use both absolute forward tolerances and global gradient geometry.
    # Per-parameter worst relative L2 is reported diagnostically but is not
    # a hard failure criterion because tiny-gradient tensors can have unstable
    # relative ratios despite negligible absolute differences.
    forward_ok = (
        loss_abs <= args.atol
        and logits_max_abs <= args.atol
    )
    grad_ok = (
        metrics["global_rel_l2"] <= args.rel_l2_tol
        and metrics["global_cos"] >= args.cos_tol
        and metrics["worst_abs"][0] <= args.atol
    )

    print("\n" + "=" * 108)
    if forward_ok and grad_ok:
        print("PASS")
        print("=" * 108)
        print(
            "Gradient checkpointing ON/OFF gives the same training signal "
            "within numerical tolerance."
        )
        print(
            "It is therefore clean to disable --grad_checkpointing for the "
            "remaining Nearest sweep as a memory/compute implementation choice."
        )
        if time_no > 0:
            print(
                f"This short test timing: no-checkpoint is "
                f"{time_gc / time_no:.2f}x as fast as checkpointed "
                "for forward+backward."
            )
    else:
        print("FAIL")
        print("=" * 108)
        print(f"forward_ok={forward_ok} | grad_ok={grad_ok}")
        print(
            "Do NOT switch the main experimental protocol yet. "
            "Send the full output for diagnosis."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
