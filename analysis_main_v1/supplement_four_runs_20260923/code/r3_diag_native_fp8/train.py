"""Train GPT with fused softmax, exact FP32 softmax, or adaptive K-interval attention.

Research-code invariants:
- quantization uses per-row causal-valid [S_min, S_max];
- Nearest forward uses weight-space LERP surrogate backward;
- LERP forward is weight-space interpolation;
- Exact FP32 Softmax is the matched arithmetic-path scientific control;
- validation RNG is reset at every evaluation for paired comparisons.

Phase B adds one controlled quantized-attention backward ablation:
- --range_grad detach reproduces the historical calibration-detached backward;
- --range_grad full differentiates through per-row adaptive min/max;
- the quantized forward operator is numerically identical between the two.

Instrumentation/checkpointing:
- preserved token-stamped checkpoints plus ckpt.pt -> latest snapshot;
- score-span, row min/max/center, Q/K norm diagnostics for all attention modes;
- interval occupancy for quant modes;
- K+1 nearest reconstruction-level occupancy for Nearest;
- optional diagnostics aggregation over the last N validation batches.
"""

import argparse
import csv
import math
import os
import shutil
import time

import numpy as np
import torch

from model import GPT, GPTConfig


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--overwrite_run", action="store_true",
                    help="delete an existing run directory before starting; "
                         "use this for a deliberate fresh rerun")
    # model (defaults = GPT-2 small scale, 124M)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_embd", type=int, default=768)
    ap.add_argument("--block_size", type=int, default=1024)
    # attention
    ap.add_argument("--attn",
                    choices=["softmax", "exact_fp32_softmax", "quant", "softmax_bf16exp", "softmax_lut2x",
                             "softmax_fp8e4m3exp"],
                    default="softmax")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--approx", choices=["nearest", "lerp"], default="lerp",
                    help="forward reconstruction inside each adaptive interval")
    ap.add_argument("--lut_levels", type=int, default=1024,
                    help="2^x LUT fractional resolution for --attn softmax_lut2x (1024 = 2^-10)")
    ap.add_argument("--range_grad", choices=["detach", "full"], default="detach",
                    help="gradient treatment of adaptive row min/max. 'detach' "
                         "reproduces historical runs; 'full' changes backward "
                         "only and leaves forward values unchanged")
    # training
    ap.add_argument("--max_tokens", type=int, default=500_000_000)
    ap.add_argument("--batch_tokens", type=int, default=131_072,
                    help="tokens per optimizer step via gradient accumulation")
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup_tokens", type=int, default=20_000_000)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--grad_checkpointing", action="store_true")
    # eval / logging / checkpointing
    ap.add_argument("--eval_every_tokens", type=int, default=10_000_000)
    ap.add_argument(
        "--log_every_steps", type=int, default=0,
        help=(
            "lightweight training-only profiling log interval in optimizer "
            "steps; 0 disables it. This does NOT trigger validation. "
            "Use --log_every_steps 1 to print throughput every step."
        ),
    )
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--stats_batches", type=int, default=1,
                    help="collect attention diagnostics on the last N eval "
                         "batches; 1 is cheap and deterministic")
    ap.add_argument("--ckpt_every_tokens", type=int, default=100_000_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


class BinReader:
    def __init__(self, path, block_size, batch, seed):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block = block_size
        self.batch = batch
        self.rng = np.random.default_rng(seed)

    def sample(self, device):
        ix = self.rng.integers(0, len(self.data) - self.block - 1, self.batch)
        x = np.stack([self.data[i:i + self.block] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.block] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if device.startswith("cuda"):
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)


def _raw_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _set_collect_stats(model, enabled):
    raw = _raw_model(model)
    for blk in raw.transformer.h:
        blk.attn.collect_stats = enabled


def _new_stats_accumulator():
    return {
        "row_span_sum": 0.0,
        "row_span_sq_sum": 0.0,
        "row_min_sum": 0.0,
        "row_max_sum": 0.0,
        "row_center_sum": 0.0,
        "row_center_sq_sum": 0.0,
        "row_count": 0,
        "q_norm_sum": 0.0,
        "q_norm_sq_sum": 0.0,
        "k_norm_sum": 0.0,
        "k_norm_sq_sum": 0.0,
        "qk_count": 0,
        "q_proj_weight_norm_sum": 0.0,
        "k_proj_weight_norm_sum": 0.0,
        "proj_count": 0,
        "interval_counts": None,
        "level_counts": None,
    }


def _accumulate_attention_stats(model, acc):
    raw = _raw_model(model)
    for blk in raw.transformer.h:
        a = blk.attn
        if a.last_row_count is None:
            continue
        acc["row_span_sum"] += float(a.last_row_span_sum)
        acc["row_span_sq_sum"] += float(a.last_row_span_sq_sum)
        acc["row_min_sum"] += float(a.last_row_min_sum)
        acc["row_max_sum"] += float(a.last_row_max_sum)
        acc["row_center_sum"] += float(a.last_row_center_sum)
        acc["row_center_sq_sum"] += float(a.last_row_center_sq_sum)
        acc["row_count"] += int(a.last_row_count)
        acc["q_norm_sum"] += float(a.last_q_norm_sum)
        acc["q_norm_sq_sum"] += float(a.last_q_norm_sq_sum)
        acc["k_norm_sum"] += float(a.last_k_norm_sum)
        acc["k_norm_sq_sum"] += float(a.last_k_norm_sq_sum)
        acc["qk_count"] += int(a.last_qk_count)
        if a.last_q_proj_weight_norm is not None:
            acc["q_proj_weight_norm_sum"] += float(a.last_q_proj_weight_norm)
            acc["k_proj_weight_norm_sum"] += float(a.last_k_proj_weight_norm)
            acc["proj_count"] += 1

        if a.last_interval_counts is not None:
            c = a.last_interval_counts.double()
            acc["interval_counts"] = (c.clone() if acc["interval_counts"] is None
                                       else acc["interval_counts"] + c)
        if a.last_level_counts is not None:
            c = a.last_level_counts.double()
            acc["level_counts"] = (c.clone() if acc["level_counts"] is None
                                    else acc["level_counts"] + c)


def _mean_std(sum_v, sq_sum_v, n):
    if n <= 0:
        return None, None
    mean = sum_v / n
    var = max(0.0, sq_sum_v / n - mean * mean)
    return mean, math.sqrt(var)


def _finalize_attention_stats(acc, K=None):
    if acc["row_count"] == 0:
        return None
    span_mean, span_std = _mean_std(acc["row_span_sum"],
                                    acc["row_span_sq_sum"],
                                    acc["row_count"])
    center_mean, center_std = _mean_std(acc["row_center_sum"],
                                        acc["row_center_sq_sum"],
                                        acc["row_count"])
    q_mean, q_std = _mean_std(acc["q_norm_sum"], acc["q_norm_sq_sum"],
                              acc["qk_count"])
    k_mean, k_std = _mean_std(acc["k_norm_sum"], acc["k_norm_sq_sum"],
                              acc["qk_count"])
    out = {
        "row_min_mean": acc["row_min_sum"] / acc["row_count"],
        "row_max_mean": acc["row_max_sum"] / acc["row_count"],
        "row_center_mean": center_mean,
        "row_center_std": center_std,
        "row_span_mean": span_mean,
        "row_span_std": span_std,
        "q_norm_mean": q_mean,
        "q_norm_std": q_std,
        "k_norm_mean": k_mean,
        "k_norm_std": k_std,
        "q_proj_weight_norm_mean": (acc["q_proj_weight_norm_sum"] / acc["proj_count"]
                                    if acc["proj_count"] else None),
        "k_proj_weight_norm_mean": (acc["k_proj_weight_norm_sum"] / acc["proj_count"]
                                    if acc["proj_count"] else None),
        "interval_width_mean": (span_mean / K if K is not None else None),
        "interval_occupancy": None,
        "level_occupancy": None,
    }
    if acc["interval_counts"] is not None:
        c = acc["interval_counts"]
        out["interval_occupancy"] = (c / c.sum()).float()
    if acc["level_counts"] is not None:
        c = acc["level_counts"]
        out["level_occupancy"] = (c / c.sum()).float()
    return out


@torch.no_grad()
def evaluate(model, reader, n_batches, device, autocast_ctx,
             stats_batches=1, quant_K=None):
    if n_batches < 1:
        raise ValueError("eval_batches must be >= 1")
    stats_batches = max(1, min(stats_batches, n_batches))

    model.eval()
    _set_collect_stats(model, False)
    losses = []
    stats_acc = _new_stats_accumulator()
    first_stats_batch = n_batches - stats_batches

    for i in range(n_batches):
        collect = i >= first_stats_batch
        _set_collect_stats(model, collect)
        x, y = reader.sample(device)
        with autocast_ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
        if collect:
            _accumulate_attention_stats(model, stats_acc)

    _set_collect_stats(model, False)
    model.train()
    stats = _finalize_attention_stats(stats_acc, K=quant_K)
    return float(np.mean(losses)), stats


def _checkpoint_filename(run_dir, step, tokens_done):
    return os.path.join(
        run_dir,
        f"ckpt_step{step:06d}_{tokens_done / 1e6:.1f}m.pt"
    )


def save_checkpoint(model, opt, train_reader, args, run_dir,
                    tokens_done, step):
    raw = _raw_model(model)
    payload = {
        "model": raw.state_dict(),
        "opt": opt.state_dict(),
        "tokens_done": tokens_done,
        "step": step,
        "train_rng_state": train_reader.rng.bit_generator.state,
        "config": vars(args),
    }
    snapshot = _checkpoint_filename(run_dir, step, tokens_done)
    tmp = snapshot + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, snapshot)

    # ckpt.pt is a convenient latest pointer.  On Linux use a hard link so it
    # consumes no duplicate model/optimizer storage; fall back to a copy.
    latest = os.path.join(run_dir, "ckpt.pt")
    try:
        if os.path.lexists(latest):
            os.unlink(latest)
        os.link(snapshot, latest)
    except OSError:
        shutil.copy2(snapshot, latest)

    print(f"saved checkpoint: {snapshot}")
    return snapshot


def _write_attention_stats(path, step, tokens_done, stats):
    if stats is None:
        return
    header_needed = not os.path.exists(path)
    cols = [
        "step", "tokens", "row_min_mean", "row_max_mean",
        "row_center_mean", "row_center_std",
        "row_span_mean", "row_span_std", "q_norm_mean", "q_norm_std",
        "k_norm_mean", "k_norm_std", "q_proj_weight_norm_mean",
        "k_proj_weight_norm_mean", "interval_width_mean",
    ]
    vals = [
        step, tokens_done,
        f"{stats['row_min_mean']:.6f}",
        f"{stats['row_max_mean']:.6f}",
        f"{stats['row_center_mean']:.6f}",
        f"{stats['row_center_std']:.6f}",
        f"{stats['row_span_mean']:.6f}",
        f"{stats['row_span_std']:.6f}",
        f"{stats['q_norm_mean']:.6f}",
        f"{stats['q_norm_std']:.6f}",
        f"{stats['k_norm_mean']:.6f}",
        f"{stats['k_norm_std']:.6f}",
        ("" if stats["q_proj_weight_norm_mean"] is None
         else f"{stats['q_proj_weight_norm_mean']:.6f}"),
        ("" if stats["k_proj_weight_norm_mean"] is None
         else f"{stats['k_proj_weight_norm_mean']:.6f}"),
        ("" if stats["interval_width_mean"] is None
         else f"{stats['interval_width_mean']:.6f}"),
    ]
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(cols)
        w.writerow(vals)


def _write_occupancy(path, step, tokens_done, stats):
    if stats is None or stats["interval_occupancy"] is None:
        return
    interval = stats["interval_occupancy"]
    level = stats["level_occupancy"]
    header_needed = not os.path.exists(path)
    header = ["step", "tokens", "row_span_mean", "row_span_std",
              "interval_width_mean"]
    header += [f"bin{i}" for i in range(len(interval))]
    if level is not None:
        header += [f"level{i}" for i in range(len(level))]

    row = [step, tokens_done,
           f"{stats['row_span_mean']:.6f}",
           f"{stats['row_span_std']:.6f}",
           f"{stats['interval_width_mean']:.6f}"]
    row += [f"{v:.6f}" for v in interval.tolist()]
    if level is not None:
        row += [f"{v:.6f}" for v in level.tolist()]

    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(header)
        w.writerow(row)


def main():
    args = get_args()
    if args.resume and args.overwrite_run:
        raise ValueError("--resume and --overwrite_run are mutually exclusive")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir = os.path.join(args.out_dir, args.run_name)
    if args.overwrite_run and os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "log.csv")
    if (not args.resume and not args.overwrite_run and os.path.exists(log_path)):
        raise RuntimeError(
            f"run directory already contains a log: {run_dir}\n"
            "Use --resume to continue it, choose a new --run_name, or pass "
            "--overwrite_run for an intentional fresh rerun."
        )

    cfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd,
                    attn_mode=args.attn, attn_K=args.K,
                    attn_approx=args.approx,
                    attn_range_grad=args.range_grad,
                    attn_lut_levels=args.lut_levels)
    model = GPT(cfg).to(device)
    model.grad_checkpointing = args.grad_checkpointing
    print(f"params (non-emb-pos): {model.num_params()/1e6:.1f}M | attn={args.attn}"
          + (f" K={args.K} approx={args.approx} range_grad={args.range_grad}"
             if args.attn == "quant" else ""))

    opt = model.configure_optimizer(args.weight_decay, args.lr,
                                    (args.beta1, args.beta2), device_type)
    if args.compile:
        model = torch.compile(model)

    use_bf16 = device_type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16 else torch.autocast(device_type=device_type,
                                                    enabled=False))

    assert args.batch_tokens % (args.micro_batch * args.block_size) == 0, \
        "batch_tokens must be a multiple of micro_batch * block_size"
    accum = args.batch_tokens // (args.micro_batch * args.block_size)
    train_reader = BinReader(os.path.join(args.data_dir, "train.bin"),
                             args.block_size, args.micro_batch, args.seed)
    val_reader = BinReader(os.path.join(args.data_dir, "val.bin"),
                           args.block_size, args.micro_batch, args.seed + 1)

    tokens_done = 0
    step = 0
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    if args.resume:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"--resume requested but missing {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_cfg = ck.get("config", {})
        if args.attn == "quant":
            saved_attn = saved_cfg.get("attn", saved_cfg.get("attn_mode", "quant"))
            saved_K = int(saved_cfg.get("K", saved_cfg.get("attn_K", args.K)))
            saved_approx = saved_cfg.get("approx", saved_cfg.get("attn_approx", args.approx))
            saved_range_grad = saved_cfg.get(
                "range_grad", saved_cfg.get("attn_range_grad", "detach")
            )
            mismatch = []
            if saved_attn != args.attn:
                mismatch.append(f"attn {saved_attn!r} != {args.attn!r}")
            if saved_K != args.K:
                mismatch.append(f"K {saved_K} != {args.K}")
            if saved_approx != args.approx:
                mismatch.append(f"approx {saved_approx!r} != {args.approx!r}")
            if saved_range_grad != args.range_grad:
                mismatch.append(
                    f"range_grad {saved_range_grad!r} != {args.range_grad!r}"
                )
            if mismatch:
                raise ValueError(
                    "resume configuration mismatch: " + "; ".join(mismatch) +
                    ". Start a fresh run for a controlled ablation."
                )
        _raw_model(model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        tokens_done, step = ck["tokens_done"], ck["step"]
        if "train_rng_state" in ck:
            train_reader.rng.bit_generator.state = ck["train_rng_state"]
        else:
            print("warning: old checkpoint has no exact train RNG state")
        print(f"resumed at step {step}, {tokens_done:,} tokens")

    occ_path = os.path.join(run_dir, "occupancy.csv")
    stats_path = os.path.join(run_dir, "attention_stats.csv")
    profile_path = os.path.join(run_dir, "profile_steps.csv")
    if args.log_every_steps < 0:
        raise ValueError("--log_every_steps must be >= 0")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step", "tokens", "train_loss", "val_loss", "lr",
                 "grad_norm", "tok_per_s"])

    def lr_at(tokens):
        if tokens < args.warmup_tokens:
            return args.lr * tokens / args.warmup_tokens
        p = min(1.0, (tokens - args.warmup_tokens)
                / max(1, args.max_tokens - args.warmup_tokens))
        mn = args.lr * args.min_lr_ratio
        return mn + 0.5 * (args.lr - mn) * (1 + math.cos(math.pi * p))

    # Keep the historical evaluation schedule so new LERP points align with
    # the already-completed Exact/Nearest 100M runs (step 1, 40, 79, ...).
    next_eval = tokens_done if args.resume else 0
    next_ckpt = tokens_done + args.ckpt_every_tokens
    t0 = time.time()
    tokens_window = 0

    # Lightweight profiling state.  This is deliberately separate from the
    # validation logger above: profile throughput measures only training-step
    # wall time and therefore excludes validation/checkpoint I/O.
    profile_enabled = args.log_every_steps > 0
    profile_time_window = 0.0
    profile_tokens_window = 0
    profile_steps_window = 0

    model.train()

    while tokens_done < args.max_tokens:
        if profile_enabled and device_type == "cuda":
            # Make the per-step timer a true wall-clock GPU measurement.
            # This is profiling-only; formal runs with log_every_steps=0 keep
            # the historical execution path unchanged.
            torch.cuda.synchronize()
        profile_step_t0 = time.perf_counter() if profile_enabled else None

        lr = lr_at(tokens_done)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            x, y = train_reader.sample(device)
            with autocast_ctx:
                _, loss = model(x, y)
            (loss / accum).backward()
            loss_acc += loss.item() / accum

        # Returned norm is the pre-clipping norm; update uses clipped grads.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip).item()
        if not math.isfinite(grad_norm):
            print(f"step {step}: non-finite grad norm, skipping update")
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()

        if profile_enabled and device_type == "cuda":
            torch.cuda.synchronize()

        step += 1
        tokens_done += args.batch_tokens
        tokens_window += args.batch_tokens

        if profile_enabled:
            profile_dt = time.perf_counter() - profile_step_t0
            profile_time_window += profile_dt
            profile_tokens_window += args.batch_tokens
            profile_steps_window += 1

            if step % args.log_every_steps == 0:
                profile_tps = (
                    profile_tokens_window / profile_time_window
                    if profile_time_window > 0 else 0.0
                )
                if device_type == "cuda":
                    peak_alloc_gib = (
                        torch.cuda.max_memory_allocated() / (1024 ** 3)
                    )
                    peak_reserved_gib = (
                        torch.cuda.max_memory_reserved() / (1024 ** 3)
                    )
                    mem_text = (
                        f" | peak_mem {peak_alloc_gib:.1f}GiB "
                        f"reserved {peak_reserved_gib:.1f}GiB"
                    )
                else:
                    peak_alloc_gib = float("nan")
                    peak_reserved_gib = float("nan")
                    mem_text = ""

                print(
                    f"[profile] step {step:6d} | "
                    f"{tokens_done/1e6:8.1f}M tok | "
                    f"train {loss_acc:.4f} | "
                    f"lr {lr:.2e} | "
                    f"gnorm {grad_norm:.2f} | "
                    f"{profile_tps/1e3:.2f}k tok/s | "
                    f"{profile_time_window/profile_steps_window:.2f}s/step"
                    f"{mem_text}",
                    flush=True,
                )

                profile_header_needed = not os.path.exists(profile_path)
                with open(profile_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if profile_header_needed:
                        w.writerow([
                            "step", "tokens", "train_loss", "lr",
                            "grad_norm", "window_steps", "window_time_s",
                            "tok_per_s", "peak_allocated_gib",
                            "peak_reserved_gib",
                        ])
                    w.writerow([
                        step, tokens_done, f"{loss_acc:.6f}", f"{lr:.6e}",
                        f"{grad_norm:.6f}", profile_steps_window,
                        f"{profile_time_window:.6f}", f"{profile_tps:.3f}",
                        f"{peak_alloc_gib:.6f}",
                        f"{peak_reserved_gib:.6f}",
                    ])

                profile_time_window = 0.0
                profile_tokens_window = 0
                profile_steps_window = 0

        if tokens_done >= next_eval:
            dt = time.time() - t0
            tps = tokens_window / dt if dt > 0 else 0.0
            val_reader.rng = np.random.default_rng(args.seed + 1)
            quant_K = args.K if args.attn == "quant" else None
            val_loss, stats = evaluate(
                model, val_reader, args.eval_batches, device, autocast_ctx,
                stats_batches=args.stats_batches, quant_K=quant_K
            )
            print(f"step {step:6d} | {tokens_done/1e6:8.1f}M tok | "
                  f"train {loss_acc:.4f} | val {val_loss:.4f} | "
                  f"lr {lr:.2e} | gnorm {grad_norm:.2f} | {tps/1e3:.1f}k tok/s")
            if stats is not None:
                extra = (f"  stats: center={stats['row_center_mean']:.3f} "
                         f"span={stats['row_span_mean']:.3f} "
                         f"qnorm={stats['q_norm_mean']:.3f} "
                         f"knorm={stats['k_norm_mean']:.3f}")
                if stats["interval_width_mean"] is not None:
                    extra += f" width={stats['interval_width_mean']:.3f}"
                print(extra)

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, tokens_done, f"{loss_acc:.5f}",
                                        f"{val_loss:.5f}", f"{lr:.3e}",
                                        f"{grad_norm:.3f}", f"{tps:.0f}"])
            _write_attention_stats(stats_path, step, tokens_done, stats)
            _write_occupancy(occ_path, step, tokens_done, stats)

            next_eval = tokens_done + args.eval_every_tokens
            t0, tokens_window = time.time(), 0

        if tokens_done >= next_ckpt or tokens_done >= args.max_tokens:
            save_checkpoint(model, opt, train_reader, args, run_dir,
                            tokens_done, step)
            next_ckpt = tokens_done + args.ckpt_every_tokens

    print("done.")


if __name__ == "__main__":
    main()
