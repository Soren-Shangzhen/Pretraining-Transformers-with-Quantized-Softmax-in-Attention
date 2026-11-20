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

rb-root_v1 (2026-11-10, rebuttal runs rb-*):
- operator flags for the new modes (see quant_attention.py docstring);
- --init_from + --branch_override: continue a parent checkpoint (model, optimizer, step, tokens,
  data-RNG state) in a NEW run directory, optionally with a changed operator (E2 branches);
- --stop_at_tokens: stop before the end of the schedule set by --max_tokens (E2c);
- --ckpt_weights_only: intermediate checkpoints hold model weights only; the final checkpoint and
  the tokens in --ckpt_full_tokens keep optimizer and data-RNG state;
- --halt_on_nonfinite: a non-finite loss or gradient norm stops the run (exit code 3, marker file
  NONFINITE.json) instead of skipping the update;
- env.json: stack, GPU, effective TF32 state, code-root SHA-256, argv.

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

import quant_attention
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
                    choices=["softmax", "exact_fp32_softmax", "quant",
                             *quant_attention.SOFTMAX_FAMILY, "indexsoftmax", "exaq"],
                    default="softmax")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--approx", choices=["nearest", "lerp"], default="lerp",
                    help="forward reconstruction inside each adaptive interval")
    ap.add_argument("--nearest_surrogate", choices=["prob", "weight_strict"], default="prob",
                    help="Supplement E2 (2026-09-23): surrogate of the strict hard-Nearest "
                         "forward; both share one hard reconstruction and reduction order")
    ap.add_argument("--range_grad", choices=["detach", "full"], default="detach",
                    help="gradient treatment of adaptive row min/max. 'detach' "
                         "reproduces historical runs; 'full' changes backward "
                         "only and leaves forward values unchanged")
    ap.add_argument(
        "--range_mode",
        choices=["minmax", "relevant_zero", "ema"],
        default="minmax",
        help=(
            "quantization range allocation: original adaptive min/max, "
            "or fixed max-relative zero-tail relevant range"
        ),
    )
    ap.add_argument(
        "--range_tau",
        type=float,
        default=6.0,
        help="window width tau for --range_mode relevant_zero",
    )
    # rb-root_v1 operator options
    ap.add_argument("--tail_policy", choices=["zero", "resolved"], default="zero",
                    help="relevant_zero tail: zero (FWM) or resolved (grid continues, App. D.4)")
    ap.add_argument("--resolved_h0", type=str, default="",
                    help="comma-separated per-layer step h0 (one per layer); sets the per-layer "
                         "window tau_l = K * h0_l (E2 freeze-h)")
    ap.add_argument("--attn_variant", type=str, default="",
                    help="indexsoftmax: fp | u8;  exaq: detach | graph")
    ap.add_argument("--exaq_bits", type=int, default=3)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    # rb-root_v1 run control
    ap.add_argument("--init_from", type=str, default="",
                    help="parent checkpoint to continue in a new run directory")
    ap.add_argument("--branch_override", action="store_true",
                    help="allow the operator of --init_from to differ from the parent's")
    ap.add_argument("--parent_surrogate", choices=["", "weight_hist", "prob_hist"], default="",
                    help="surrogate of a historical Nearest parent (not recorded in old configs; "
                         "identified by the parent code root's quant_attention.py SHA-256)")
    ap.add_argument("--parent_code_root", type=str, default="",
                    help="frozen code root the parent was trained with (its hashes are recorded)")
    ap.add_argument("--stop_at_tokens", type=int, default=0,
                    help="stop once this many tokens are done (0: run to --max_tokens)")
    ap.add_argument("--ckpt_weights_only", action="store_true",
                    help="intermediate checkpoints store model weights only")
    ap.add_argument("--ckpt_full_tokens", type=str, default="",
                    help="comma-separated token milestones whose checkpoints keep optimizer state")
    ap.add_argument("--halt_on_nonfinite", action="store_true")
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
    ap.add_argument(
        "--health_every_steps", type=int, default=0,
        help=(
            "write low-overhead training-health statistics every N optimizer "
            "steps; 0 disables. Adds no explicit CUDA synchronization."
        ),
    )
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--stats_batches", type=int, default=1,
                    help="collect attention diagnostics on the last N eval "
                         "batches; 1 is cheap and deterministic")
    ap.add_argument("--ckpt_every_tokens", type=int, default=100_000_000)
    ap.add_argument(
        "--ckpt_tokens", type=str, default="",
        help=(
            "optional comma-separated absolute token milestones for "
            "checkpoints; when non-empty, overrides periodic checkpointing"
        ),
    )
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


def _finalize_attention_stats(acc, K=None, interval_width_override=None):
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
        "interval_width_mean": (
            float(interval_width_override)
            if interval_width_override is not None
            else (
                span_mean / K
                if K is not None
                else None
            )
        ),
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
             stats_batches=1, quant_K=None,
             quant_interval_width=None):
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
    stats = _finalize_attention_stats(
        stats_acc,
        K=quant_K,
        interval_width_override=quant_interval_width,
    )
    return float(np.mean(losses)), stats


def _checkpoint_filename(run_dir, step, tokens_done):
    return os.path.join(
        run_dir,
        f"ckpt_step{step:06d}_{tokens_done / 1e6:.1f}m.pt"
    )


def save_checkpoint(model, opt, train_reader, args, run_dir,
                    tokens_done, step, weights_only=False):
    raw = _raw_model(model)
    payload = {
        "model": raw.state_dict(),
        "tokens_done": tokens_done,
        "step": step,
        "config": vars(args),
    }
    if not weights_only:
        payload["opt"] = opt.state_dict()
        payload["train_rng_state"] = train_reader.rng.bit_generator.state
    snapshot = _checkpoint_filename(run_dir, step, tokens_done)
    tmp = snapshot + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, snapshot)

    if weights_only:
        print(f"saved checkpoint (weights only): {snapshot}")
        return snapshot

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


# ---------------------------------------------------------------- rb-root_v1 helpers
PROTOCOL_FIELDS = ("n_layer", "n_head", "n_embd", "block_size", "batch_tokens", "micro_batch",
                   "lr", "min_lr_ratio", "warmup_tokens", "max_tokens", "weight_decay", "beta1",
                   "beta2", "grad_clip", "seed")
OPERATOR_FIELDS = ("attn", "K", "approx", "nearest_surrogate", "range_grad", "range_mode",
                   "range_tau", "tail_policy", "resolved_h0", "attn_variant", "exaq_bits",
                   "ema_decay")
# defaults of operator fields that older checkpoints do not record
OPERATOR_DEFAULTS = {"nearest_surrogate": "prob", "range_mode": "minmax", "range_tau": 6.0,
                     "tail_policy": "zero", "resolved_h0": "", "attn_variant": "",
                     "exaq_bits": 3, "ema_decay": 0.999}


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_hashes():
    here = os.path.dirname(os.path.abspath(__file__))
    return {f: _sha256_file(os.path.join(here, f))
            for f in ("model.py", "quant_attention.py", "train_formal.py")}


def _tf32_probe(device):
    """Effective TF32 state of fp32 matmuls (dedicated generator; global RNG untouched)."""
    if not device.startswith("cuda"):
        return {"device": device, "tf32_effective": False}
    g = torch.Generator(device=device)
    g.manual_seed(12345)
    a = torch.randn(256, 256, device=device, generator=g)
    b = torch.randn(256, 256, device=device, generator=g)
    ref = (a.double() @ b.double())
    err = ((a @ b).double() - ref).abs().max().item() / ref.abs().max().item()
    # IEEE fp32 accumulation gives ~1e-7 relative error, TF32 (10-bit mantissa inputs) ~1e-3
    return {"device": device, "max_rel_err_fp32_matmul": err, "tf32_effective": err > 1e-5,
            "allow_tf32_flag": bool(torch.backends.cuda.matmul.allow_tf32),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE")}


def _write_env_record(run_dir, args, device):
    import json, platform, socket, subprocess, sys
    rec = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(), "argv": sys.argv, "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "code_root": os.path.dirname(os.path.abspath(__file__)), "code_sha256": _code_hashes(),
        "tf32": _tf32_probe(device), "config": vars(args),
    }
    try:
        rec["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                           timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        rec["nvidia_smi"] = f"unavailable: {e}"
    name = "env.json" if not os.path.exists(os.path.join(run_dir, "env.json")) else \
        f"env_resume_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json"
    with open(os.path.join(run_dir, name), "w") as f:
        json.dump(rec, f, indent=2, default=str)
    return rec


def _validate_operator_args(args):
    if args.attn == "indexsoftmax" and args.attn_variant not in ("fp", "u8"):
        raise ValueError("--attn indexsoftmax needs --attn_variant fp|u8")
    if args.attn == "exaq" and args.attn_variant not in ("detach", "graph"):
        raise ValueError("--attn exaq needs --attn_variant detach|graph")
    if args.attn not in ("indexsoftmax", "exaq") and args.attn_variant:
        raise ValueError("--attn_variant is only used by indexsoftmax and exaq")
    if args.range_mode == "ema":
        if args.attn != "quant" or args.approx != "nearest" or args.range_grad != "full":
            raise ValueError("--range_mode ema needs --attn quant --approx nearest --range_grad full")
        if args.grad_checkpointing:
            raise ValueError("--range_mode ema updates a buffer in the forward; "
                             "gradient checkpointing would update it twice")
    if args.tail_policy != "zero" and args.range_mode != "relevant_zero":
        raise ValueError("--tail_policy applies to --range_mode relevant_zero only")
    if args.resolved_h0:
        h0 = [float(x) for x in args.resolved_h0.split(",")]
        if len(h0) != args.n_layer or any(not (x > 0) for x in h0):
            raise ValueError("--resolved_h0 needs one positive value per layer")
        if args.range_mode != "relevant_zero" or args.tail_policy != "resolved":
            raise ValueError("--resolved_h0 needs --range_mode relevant_zero --tail_policy resolved")
        return tuple(args.K * x for x in h0)
    return None


def _operator_of(cfg_dict):
    out = {}
    for k in OPERATOR_FIELDS:
        v = cfg_dict.get(k, cfg_dict.get("attn_" + k, OPERATOR_DEFAULTS.get(k)))
        out[k] = v
    if out["attn"] is None:
        out["attn"] = cfg_dict.get("attn_mode", "softmax")
    return out


def main():
    args = get_args()
    if args.resume and args.overwrite_run:
        raise ValueError("--resume and --overwrite_run are mutually exclusive")
    if args.init_from and (args.resume or args.overwrite_run):
        raise ValueError("--init_from starts a new run directory; it excludes --resume/--overwrite_run")
    tau_per_layer = _validate_operator_args(args)
    args.code_sha256 = _code_hashes()

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
                    attn_range_mode=args.range_mode,
                    attn_range_tau=args.range_tau,
                    attn_tail_policy=args.tail_policy,
                    attn_range_tau_per_layer=tau_per_layer,
                    attn_variant=args.attn_variant,
                    attn_exaq_bits=args.exaq_bits,
                    attn_ema_decay=args.ema_decay)
    quant_attention.NEAREST_SURROGATE = args.nearest_surrogate
    model = GPT(cfg).to(device)
    model.grad_checkpointing = args.grad_checkpointing
    print(f"params (non-emb-pos): {model.num_params()/1e6:.1f}M | attn={args.attn}"
          + (f" K={args.K} approx={args.approx} range_grad={args.range_grad}"
             if args.attn == "quant" else ""))

    if args.attn == "quant":
        print(
            f"quant range_mode={args.range_mode} "
            f"range_tau={args.range_tau:g} "
            f"effective_h="
            f"{(args.range_tau / args.K if args.range_mode == 'relevant_zero' else float('nan'))}"
        )

    opt = model.configure_optimizer(args.weight_decay, args.lr,
                                    (args.beta1, args.beta2), device_type)
    env_rec = _write_env_record(run_dir, args, device)
    args.tf32_effective = env_rec["tf32"]["tf32_effective"]
    print(f"tf32_effective={args.tf32_effective} code_sha256={args.code_sha256}")
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
        # rb-root_v1: every operator field must match on resume (older roots checked quant only)
        saved_op, cur_op = _operator_of(saved_cfg), _operator_of(vars(args))
        mismatch = [f"{k} {saved_op[k]!r} != {cur_op[k]!r}" for k in OPERATOR_FIELDS
                    if str(saved_op[k]) != str(cur_op[k])]
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

    if args.init_from:
        # rb-root_v1: continue a parent checkpoint in this new run directory (E2 branches).
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        if "opt" not in ck or "train_rng_state" not in ck:
            raise ValueError("--init_from needs a checkpoint with optimizer and data-RNG state")
        pc = ck.get("config", {})
        bad = []
        for f in PROTOCOL_FIELDS:
            pv, av = pc.get(f), getattr(args, f)
            if pv is None:
                bad.append(f"{f}: parent records no value")
            elif isinstance(av, float) or isinstance(pv, float):
                if not math.isclose(float(pv), float(av), rel_tol=0.0, abs_tol=1e-12):
                    bad.append(f"{f} {pv!r} != {av!r}")
            elif pv != av:
                bad.append(f"{f} {pv!r} != {av!r}")
        pdata = os.path.basename(os.path.normpath(str(pc.get("data_dir", ""))))
        if pdata != os.path.basename(os.path.normpath(args.data_dir)):
            bad.append(f"data_dir {pdata!r} != {os.path.basename(os.path.normpath(args.data_dir))!r}")
        if bad:
            raise ValueError("--init_from protocol mismatch: " + "; ".join(bad))
        parent_op, new_op = _operator_of(pc), _operator_of(vars(args))
        if parent_op["attn"] == "quant" and parent_op["approx"] == "nearest" \
                and "nearest_surrogate" not in pc:
            if not args.parent_surrogate:
                raise ValueError("the parent config does not record its surrogate; pass "
                                 "--parent_surrogate weight_hist|prob_hist (from its code-root hash)")
            parent_op["nearest_surrogate"] = args.parent_surrogate
        if args.parent_code_root:
            args.parent_code_sha256 = {
                f: _sha256_file(os.path.join(args.parent_code_root, f))
                for f in ("model.py", "quant_attention.py")
                if os.path.exists(os.path.join(args.parent_code_root, f))}
        changes = {k: [parent_op[k], new_op[k]] for k in OPERATOR_FIELDS
                   if str(parent_op[k]) != str(new_op[k])}
        if changes and not args.branch_override:
            raise ValueError(f"--init_from changes the operator {changes}; pass --branch_override")
        sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
              for k, v in ck["model"].items()}
        missing, unexpected = _raw_model(model).load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"parent state_dict mismatch missing={missing} unexpected={unexpected}")
        opt.load_state_dict(ck["opt"])
        tokens_done, step = int(ck["tokens_done"]), int(ck["step"])
        train_reader.rng.bit_generator.state = ck["train_rng_state"]
        args.branch_parent = os.path.abspath(args.init_from)
        args.branch_parent_sha256 = _sha256_file(args.init_from)
        args.branch_parent_run = pc.get("run_name")
        args.branch_parent_step = step
        args.branch_parent_tokens = tokens_done
        args.branch_parent_operator = parent_op
        args.branch_operator_changes = changes
        print(f"branched from {args.init_from} at step {step}, {tokens_done:,} tokens; "
              f"operator changes: {changes or 'none'}")
        del ck

    occ_path = os.path.join(run_dir, "occupancy.csv")
    stats_path = os.path.join(run_dir, "attention_stats.csv")
    profile_path = os.path.join(run_dir, "profile_steps.csv")
    health_path = os.path.join(run_dir, "training_health.csv")

    if args.log_every_steps < 0:
        raise ValueError("--log_every_steps must be >= 0")
    if args.health_every_steps < 0:
        raise ValueError("--health_every_steps must be >= 0")
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
    next_eval = tokens_done if (args.resume or args.init_from) else 0

    ckpt_targets = []
    if args.ckpt_tokens.strip():
        try:
            ckpt_targets = sorted({
                int(x.strip())
                for x in args.ckpt_tokens.split(",")
                if x.strip()
            })
        except ValueError as e:
            raise ValueError(
                "--ckpt_tokens must be comma-separated integers"
            ) from e

        if any(x <= 0 for x in ckpt_targets):
            raise ValueError("--ckpt_tokens values must be positive")

    if ckpt_targets:
        remaining_ckpts = [x for x in ckpt_targets if x > tokens_done]
        next_ckpt = (
            remaining_ckpts[0]
            if remaining_ckpts
            else args.max_tokens + args.batch_tokens
        )
    else:
        if args.ckpt_every_tokens <= 0:
            raise ValueError("--ckpt_every_tokens must be positive")
        remaining_ckpts = []
        next_ckpt = tokens_done + args.ckpt_every_tokens

    full_targets = sorted({int(x) for x in args.ckpt_full_tokens.split(",") if x.strip()})
    full_targets = [x for x in full_targets if x > tokens_done]
    end_tokens = args.max_tokens
    if args.stop_at_tokens > 0:
        if args.stop_at_tokens > args.max_tokens:
            raise ValueError("--stop_at_tokens exceeds --max_tokens")
        end_tokens = args.stop_at_tokens

    t0 = time.time()
    tokens_window = 0

    # Lightweight profiling state.  This is deliberately separate from the
    # validation logger above: profile throughput measures only training-step
    # wall time and therefore excludes validation/checkpoint I/O.
    profile_enabled = args.log_every_steps > 0
    profile_time_window = 0.0
    profile_tokens_window = 0
    profile_steps_window = 0

    # Formal-run health logging.
    # loss_acc and grad_norm are already Python scalars in the historical
    # training path, so this adds no new tensor reductions or CUDA syncs.
    health_enabled = args.health_every_steps > 0
    health_t0 = time.time()
    health_steps = 0

    health_loss_count = 0
    health_loss_sum = 0.0
    health_loss_sq_sum = 0.0

    health_grad_count = 0
    health_grad_sum = 0.0
    health_grad_max = 0.0
    health_clip_count = 0

    health_nonfinite_grad = 0
    health_nonfinite_loss = 0

    model.train()

    while tokens_done < end_tokens:
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
        if args.halt_on_nonfinite and not (math.isfinite(loss_acc) and math.isfinite(grad_norm)):
            import json
            rec = {"step": step + 1, "tokens_before_step": tokens_done, "train_loss": repr(loss_acc),
                   "grad_norm": repr(grad_norm), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                          time.gmtime())}
            with open(os.path.join(run_dir, "NONFINITE.json"), "w") as f:
                json.dump(rec, f, indent=2)
            print(f"HALT: non-finite value at step {step + 1}: {rec}", flush=True)
            raise SystemExit(3)
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

        if health_enabled:
            health_steps += 1

            if math.isfinite(loss_acc):
                health_loss_count += 1
                health_loss_sum += loss_acc
                health_loss_sq_sum += loss_acc * loss_acc
            else:
                health_nonfinite_loss += 1

            if math.isfinite(grad_norm):
                health_grad_count += 1
                health_grad_sum += grad_norm
                health_grad_max = max(health_grad_max, grad_norm)
                if grad_norm > args.grad_clip:
                    health_clip_count += 1
            else:
                health_nonfinite_grad += 1

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
            quant_interval_width = (
                args.range_tau / args.K
                if (
                    args.attn == "quant"
                    and args.range_mode == "relevant_zero"
                )
                else None
            )
            val_loss, stats = evaluate(
                model, val_reader, args.eval_batches, device, autocast_ctx,
                stats_batches=args.stats_batches, quant_K=quant_K,
                quant_interval_width=quant_interval_width
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

        if tokens_done >= next_ckpt or tokens_done >= end_tokens:
            full = (not args.ckpt_weights_only) or tokens_done >= end_tokens
            if full_targets and tokens_done >= full_targets[0]:
                full = True
            full_targets = [x for x in full_targets if x > tokens_done]
            save_checkpoint(model, opt, train_reader, args, run_dir,
                            tokens_done, step, weights_only=not full)

            if ckpt_targets:
                remaining_ckpts = [
                    x for x in remaining_ckpts
                    if x > tokens_done
                ]
                next_ckpt = (
                    remaining_ckpts[0]
                    if remaining_ckpts
                    else args.max_tokens + args.batch_tokens
                )
            else:
                next_ckpt = tokens_done + args.ckpt_every_tokens

        if health_enabled and (
            step % args.health_every_steps == 0
            or tokens_done >= end_tokens
        ):
            health_dt = max(1e-12, time.time() - health_t0)
            wall_tps = (
                health_steps * args.batch_tokens / health_dt
                if health_steps > 0 else 0.0
            )

            if health_loss_count:
                loss_mean = health_loss_sum / health_loss_count
                loss_var = max(
                    0.0,
                    health_loss_sq_sum / health_loss_count
                    - loss_mean * loss_mean,
                )
                loss_std = math.sqrt(loss_var)
            else:
                loss_mean = float("nan")
                loss_std = float("nan")

            if health_grad_count:
                grad_mean = health_grad_sum / health_grad_count
                clip_fraction = (
                    health_clip_count / health_grad_count
                )
            else:
                grad_mean = float("nan")
                clip_fraction = float("nan")

            header_needed = not os.path.exists(health_path)

            with open(health_path, "a", newline="") as f:
                w = csv.writer(f)

                if header_needed:
                    w.writerow([
                        "timestamp_utc",
                        "step",
                        "tokens",
                        "window_steps",
                        "window_time_s",
                        "wall_tok_per_s",
                        "train_loss_mean",
                        "train_loss_std",
                        "grad_norm_mean",
                        "grad_norm_max",
                        "clip_fraction",
                        "nonfinite_grad_updates",
                        "nonfinite_loss_steps",
                        "lr",
                    ])

                w.writerow([
                    time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(),
                    ),
                    step,
                    tokens_done,
                    health_steps,
                    f"{health_dt:.6f}",
                    f"{wall_tps:.3f}",
                    f"{loss_mean:.6f}",
                    f"{loss_std:.6f}",
                    f"{grad_mean:.6f}",
                    f"{health_grad_max:.6f}",
                    f"{clip_fraction:.6f}",
                    health_nonfinite_grad,
                    health_nonfinite_loss,
                    f"{lr:.6e}",
                ])

            health_t0 = time.time()
            health_steps = 0

            health_loss_count = 0
            health_loss_sum = 0.0
            health_loss_sq_sum = 0.0

            health_grad_count = 0
            health_grad_sum = 0.0
            health_grad_max = 0.0
            health_clip_count = 0

            health_nonfinite_grad = 0
            health_nonfinite_loss = 0

    print("done.")


if __name__ == "__main__":
    main()
