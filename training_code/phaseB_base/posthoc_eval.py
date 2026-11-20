"""Post-hoc evaluation infrastructure for P0-1 (Train x Eval matrix) and P0-2
(temperature sweep).

Design notes
------------
Deterministic full-coverage protocol (plan v4 section 6.1):
    val.bin is tiled into non-overlapping blocks of `block_size`.  Each block
    needs block_size+1 tokens (inputs and right-shifted targets), so the number
    of blocks is floor((N - block_size - 1) / block_size) + 1.  For
    WikiText-103 (249,750 tokens) with block_size=1024 this is 243 blocks
    covering 248,832 tokens, versus the historical protocol's 100 blocks drawn
    *with replacement* from ~243 possible positions.

    Per-block NLL vectors are always returned and saved so that paired-block
    bootstrap CIs can be recomputed without re-running any forward pass.

Operator override:
    QuantCausalSelfAttention holds only c_attn / c_proj parameters; the
    attention mode, K and approx are plain attributes.  A checkpoint trained
    under any mode therefore loads unchanged into a model configured with any
    other mode, which is what makes the Train x Eval matrix possible.

Temperature injection:
    P0-2 applies alpha directly to the explicit FP32 attention score matrix,
        S_alpha = alpha * (q k^T / sqrt(d)),
    before causal masking / exact softmax / adaptive K-interval reconstruction.
    This avoids dependence on fused-QKV weight layout, projection bias, or BF16
    Q-scaling roundoff.  The training modules are not modified on disk: during
    the temp subcommand only, their exact/quant attention helper methods are
    temporarily rebound to evaluation-only score-scaled versions.

    The same forward pass also records row-wise attention entropy H(alpha), both
    raw and normalized by log(valid row length), so H2b does not require another
    temperature sweep.

Usage
-----
    # sanity: confirm two checkpoints are pairable
    python posthoc_eval.py verify \
        --ckpt-a $OLD/runs/exact_fp32_softmax_100m/ckpt.pt \
        --ckpt-b runs/q32_lerp_fullgrad_100m/ckpt.pt

    # P0-1: full Train x Eval matrix
    python posthoc_eval.py matrix \
        --data-dir data/wikitext103 \
        --manifest manifest.json \
        --out p0_results/matrix

    # P0-2: temperature sweep on the Exact-trained checkpoint
    python posthoc_eval.py temp \
        --data-dir data/wikitext103 \
        --ckpt $OLD/runs/exact_fp32_softmax_100m/ckpt.pt \
        --out p0_results/temp
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

from model import GPT, GPTConfig
from quant_attention import _adaptive_interval_reconstruct


# --------------------------------------------------------------------------
# deterministic full-coverage evaluation set
# --------------------------------------------------------------------------

class FullCoverageEvalSet:
    """Non-overlapping tiling of a .bin token stream.

    Deterministic: no RNG anywhere.  Every model sees exactly the same blocks
    in exactly the same order, which is what makes the per-block NLL vectors
    pairable across models and operators.
    """

    def __init__(self, path, block_size, stride=None, subset=None):
        data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n_tokens = len(data)
        self.block_size = block_size
        stride = stride or block_size

        last_start = self.n_tokens - block_size - 1
        if last_start < 0:
            raise ValueError(
                f"{path} has {self.n_tokens} tokens, too few for block_size={block_size}"
            )
        starts = list(range(0, last_start + 1, stride))

        self.all_starts = starts
        if subset is not None and subset < len(starts):
            # Evenly spaced deterministic subset; keeps coverage representative
            # instead of taking a contiguous prefix.
            idx = np.linspace(0, len(starts) - 1, subset).round().astype(int)
            idx = sorted(set(idx.tolist()))
            starts = [starts[i] for i in idx]
        self.starts = starts

        x = np.stack([data[i:i + block_size] for i in starts]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + block_size] for i in starts]).astype(np.int64)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.starts)

    def describe(self):
        return {
            "n_tokens_in_file": int(self.n_tokens),
            "block_size": int(self.block_size),
            "n_blocks_full_coverage": int(len(self.all_starts)),
            "n_blocks_used": int(len(self.starts)),
            "tokens_covered": int(len(self.starts) * self.block_size),
            "is_subset": len(self.starts) != len(self.all_starts),
        }


# --------------------------------------------------------------------------
# model loading and operator override
# --------------------------------------------------------------------------

CRITICAL_CONFIG_KEYS = [
    "seed", "n_layer", "n_head", "n_embd", "block_size",
    "micro_batch", "batch_tokens", "max_tokens", "warmup_tokens",
    "lr", "min_lr_ratio", "weight_decay", "beta1", "beta2", "grad_clip",
]

# Keys that legitimately differ between an Exact run and a quant run, plus keys
# that argparse fills with defaults which are inert in the other mode.
IGNORED_CONFIG_KEYS = {
    "attn", "attn_mode", "K", "attn_K", "approx", "attn_approx",
    "range_grad", "attn_range_grad", "run_name", "out_dir", "data_dir",
    "resume", "overwrite_run", "compile", "device", "grad_checkpointing",
    "eval_every_tokens", "eval_batches", "stats_batches", "ckpt_every_tokens",
}


def read_checkpoint_config(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("config", {}), ck


def training_operator_of(cfg):
    """Return (attn_mode, K, approx) that this checkpoint was TRAINED with."""
    attn = cfg.get("attn", cfg.get("attn_mode", "softmax"))
    K = int(cfg.get("K", cfg.get("attn_K", 8)))
    approx = cfg.get("approx", cfg.get("attn_approx", "lerp"))
    return attn, K, approx


def load_model(ckpt_path, device, ckpt=None):
    """Load a checkpoint into a GPT.  Attention mode is set separately."""
    if ckpt is None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    gcfg = GPTConfig(
        block_size=int(cfg.get("block_size", 1024)),
        n_layer=int(cfg.get("n_layer", 12)),
        n_head=int(cfg.get("n_head", 12)),
        n_embd=int(cfg.get("n_embd", 768)),
        attn_mode="exact_fp32_softmax",   # placeholder, overridden per eval
        attn_K=8,
        attn_approx="lerp",
        attn_range_grad="full",
    )
    model = GPT(gcfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch loading {ckpt_path}\n"
            f"  missing:    {missing}\n"
            f"  unexpected: {unexpected}\n"
            "The checkpoint was produced by an incompatible model definition."
        )
    model.grad_checkpointing = False
    model.eval()
    return model, cfg


def set_operator(model, attn_mode, K=None, approx=None):
    """Override the attention operator on every block.  No parameters change."""
    for blk in model.transformer.h:
        a = blk.attn
        a.mode = attn_mode
        if K is not None:
            a.K = int(K)
        if approx is not None:
            a.approx = approx
        a.collect_stats = False


class ScoreTemperatureController:
    """Evaluation-only direct score-level temperature injection.

    The model source files and checkpoint parameters are untouched.  We
    temporarily rebind each attention module's explicit exact/quant helper
    methods so that alpha is applied *after* the FP32 QK matmul:

        S_alpha = alpha * (Q K^T / sqrt(d_h)).

    The rebound methods otherwise mirror quant_attention.py and reuse its
    `_adaptive_interval_reconstruct`, so the adaptive grid is recalibrated from
    the scaled scores exactly as intended.

    During a temperature point we also accumulate per-block, per-layer
    attention entropy.  Primary entropy means exclude q=0, matching the geometry
    probe's global/layer convention; q=0 is degenerate and has H=0 anyway.
    """

    def __init__(self, model):
        self.model = model
        self.alpha = 1.0
        self.n_layer = len(model.transformer.h)
        self._original = []
        self._batch_raw_sum = None
        self._batch_norm_sum = None
        self._batch_count = None

        for layer, blk in enumerate(model.transformer.h):
            a = blk.attn
            self._original.append((a, a._exact_fp32_softmax, a._quant_attention))
            a._exact_fp32_softmax = types.MethodType(
                self._make_exact_method(layer), a
            )
            a._quant_attention = types.MethodType(
                self._make_quant_method(layer), a
            )

    def set(self, alpha):
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"temperature alpha must be finite and > 0, got {alpha}")
        self.alpha = alpha

    def start_batch(self, batch_size):
        # CPU float64 accumulators are tiny (B x 12) and avoid keeping graph/GPU
        # state between layers.  Forward runs under no_grad.
        shape = (int(batch_size), self.n_layer)
        self._batch_raw_sum = np.zeros(shape, dtype=np.float64)
        self._batch_norm_sum = np.zeros(shape, dtype=np.float64)
        self._batch_count = np.zeros(shape, dtype=np.int64)

    def finish_batch(self):
        if self._batch_count is None:
            raise RuntimeError("finish_batch called before start_batch")
        c = self._batch_count
        if np.any(c <= 0):
            raise RuntimeError("temperature entropy accumulator has empty layer(s)")
        raw_layer = self._batch_raw_sum / c
        norm_layer = self._batch_norm_sum / c
        # Each layer/head/query contributes equally.  Counts are identical across
        # layers at fixed T, so averaging per-layer means is the same as pooling.
        raw = raw_layer.mean(axis=1)
        norm = norm_layer.mean(axis=1)
        self._batch_raw_sum = None
        self._batch_norm_sum = None
        self._batch_count = None
        return raw, norm, raw_layer, norm_layer

    @torch.no_grad()
    def _record_entropy(self, layer, P):
        # P: (B,H,T,T), causal-invalid locations are exactly zero.
        if self._batch_count is None:
            return
        B, H, T, _ = P.shape
        tiny = torch.finfo(P.dtype).tiny
        logp = torch.where(P > 0, torch.log(P.clamp_min(tiny)), torch.zeros_like(P))
        h_raw = -(P * logp).sum(dim=-1)  # (B,H,T)

        lengths = torch.arange(1, T + 1, device=P.device, dtype=P.dtype)
        logL = torch.log(lengths).view(1, 1, T)
        h_norm = torch.where(
            lengths.view(1, 1, T) > 1,
            h_raw / logL.clamp_min(1e-12),
            torch.zeros_like(h_raw),
        )

        # Match geometry_probe.py primary summaries: exclude q=0.
        if T <= 1:
            return
        raw = h_raw[..., 1:].double().sum(dim=(1, 2)).cpu().numpy()
        norm = h_norm[..., 1:].double().sum(dim=(1, 2)).cpu().numpy()
        count = H * (T - 1)
        self._batch_raw_sum[:, layer] += raw
        self._batch_norm_sum[:, layer] += norm
        self._batch_count[:, layer] += count

    def _make_exact_method(self, layer):
        controller = self

        @torch.no_grad()
        def exact_scaled(module, q, k, v):
            B, H, T, D = q.shape
            with torch.autocast(device_type=q.device.type, enabled=False):
                qf, kf, vf = q.float(), k.float(), v.float()
                att = (qf @ kf.transpose(-2, -1)) / math.sqrt(D)
                att = att * controller.alpha
                causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
                valid = causal.view(1, 1, T, T)

                masked_att = att.masked_fill(~valid, -torch.inf)
                row_max = masked_att.amax(dim=-1, keepdim=True)
                z = masked_att - row_max
                w = torch.exp(z)
                P = w / w.sum(dim=-1, keepdim=True)
                y = P @ vf
                controller._record_entropy(layer, P)

            return y.to(q.dtype)

        return exact_scaled

    def _make_quant_method(self, layer):
        controller = self

        @torch.no_grad()
        def quant_scaled(module, q, k, v):
            B, H, T, D = q.shape
            with torch.autocast(device_type=q.device.type, enabled=False):
                qf, kf, vf = q.float(), k.float(), v.float()
                att = (qf @ kf.transpose(-2, -1)) / math.sqrt(D)
                att = att * controller.alpha
                causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()

                # Forward values do not depend on range_grad; use the module's
                # configured value to mirror the original implementation.
                w, _bins, _row_span = _adaptive_interval_reconstruct(
                    att, causal, module.K, module.approx,
                    return_level_idx=False, range_grad=module.range_grad,
                )
                P = w / w.sum(dim=-1, keepdim=True)
                y = P @ vf
                controller._record_entropy(layer, P)

            return y.to(q.dtype)

        return quant_scaled

    def close(self):
        for a, exact0, quant0 in self._original:
            a._exact_fp32_softmax = exact0
            a._quant_attention = quant0
        self._original.clear()



# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_per_block(model, evalset, device, autocast_ctx, batch=2):
    """Return a float64 vector of per-block mean NLL, one entry per block."""
    out = np.zeros(len(evalset), dtype=np.float64)
    i = 0
    n = len(evalset)
    cur_batch = batch
    while i < n:
        j = min(i + cur_batch, n)
        x = evalset.x[i:j].to(device, non_blocking=True)
        y = evalset.y[i:j].to(device, non_blocking=True)
        try:
            with autocast_ctx:
                logits, _ = model(x)
                # Per-block mean NLL over this block's target tokens.
                ls = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="none",
                ).view(y.shape).mean(dim=-1)
            out[i:j] = ls.double().cpu().numpy()
            i = j
        except torch.cuda.OutOfMemoryError:
            if cur_batch == 1:
                raise
            torch.cuda.empty_cache()
            cur_batch = max(1, cur_batch // 2)
            print(f"  OOM, reducing eval batch to {cur_batch}", file=sys.stderr)
    return out


@torch.no_grad()
def eval_temp_point(model, evalset, device, autocast_ctx, controller, batch=2):
    """Evaluate one alpha and return paired per-block NLL + attention entropy.

    Returns a dict with vectors of shape (n_blocks,) and per-layer matrices of
    shape (n_blocks, n_layer).  All statistics come from the same forward pass.
    """
    n = len(evalset)
    n_layer = len(model.transformer.h)
    nll = np.zeros(n, dtype=np.float64)
    h_raw = np.zeros(n, dtype=np.float64)
    h_norm = np.zeros(n, dtype=np.float64)
    h_raw_layer = np.zeros((n, n_layer), dtype=np.float64)
    h_norm_layer = np.zeros((n, n_layer), dtype=np.float64)

    i = 0
    cur_batch = batch
    while i < n:
        j = min(i + cur_batch, n)
        x = evalset.x[i:j].to(device, non_blocking=True)
        y = evalset.y[i:j].to(device, non_blocking=True)
        try:
            controller.start_batch(j - i)
            with autocast_ctx:
                logits, _ = model(x)
                ls = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    y.view(-1), reduction="none",
                ).view(y.shape).mean(dim=-1)
            br, bn, brl, bnl = controller.finish_batch()
            nll[i:j] = ls.double().cpu().numpy()
            h_raw[i:j] = br
            h_norm[i:j] = bn
            h_raw_layer[i:j] = brl
            h_norm_layer[i:j] = bnl
            i = j
        except torch.cuda.OutOfMemoryError:
            # Discard any partial entropy accumulator from the failed forward.
            controller._batch_raw_sum = None
            controller._batch_norm_sum = None
            controller._batch_count = None
            if cur_batch == 1:
                raise
            torch.cuda.empty_cache()
            cur_batch = max(1, cur_batch // 2)
            print(f"  OOM, reducing eval batch to {cur_batch}", file=sys.stderr)

    return {
        "nll": nll,
        "entropy_raw": h_raw,
        "entropy_norm": h_norm,
        "entropy_raw_by_layer": h_raw_layer,
        "entropy_norm_by_layer": h_norm_layer,
    }


def make_autocast(device_type, dtype):
    if dtype == "fp32" or device_type != "cuda":
        return torch.autocast(device_type=device_type, enabled=False)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but not supported; pass --dtype fp32")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


# --------------------------------------------------------------------------
# paired bootstrap
# --------------------------------------------------------------------------

def paired_bootstrap(fn, n_blocks, n_boot=10000, seed=0, alpha=0.05):
    """Bootstrap over block indices.

    `fn(idx)` must return a scalar computed from the resampled block indices.
    All quantities entering `fn` are evaluated on the SAME resampled blocks,
    which is what makes composite statistics such as A(O) properly paired.
    """
    rng = np.random.default_rng(seed)
    point = float(fn(np.arange(n_blocks)))
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_blocks, n_blocks)
        draws[b] = fn(idx)
    lo = float(np.quantile(draws, alpha / 2))
    hi = float(np.quantile(draws, 1 - alpha / 2))
    return {
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "boot_mean": float(draws.mean()),
        "boot_sd": float(draws.std(ddof=1)),
        "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------------
# operator grid
# --------------------------------------------------------------------------

def operator_key(attn_mode, K=None, approx=None):
    if attn_mode == "quant":
        return f"{approx}_K{K}"
    return "exact"


def parse_operator_key(key):
    if key == "exact":
        return ("exact_fp32_softmax", None, None)
    approx, ks = key.rsplit("_K", 1)
    return ("quant", int(ks), approx)


def default_operator_keys(k_values):
    keys = ["exact"]
    for approx in ("lerp", "nearest"):
        for K in k_values:
            keys.append(f"{approx}_K{K}")
    return keys


# --------------------------------------------------------------------------
# subcommand: verify
# --------------------------------------------------------------------------

def cmd_verify(args):
    cfg_a, ck_a = read_checkpoint_config(args.ckpt_a)
    cfg_b, ck_b = read_checkpoint_config(args.ckpt_b)

    print(f"A: {args.ckpt_a}")
    print(f"   trained operator: {training_operator_of(cfg_a)}")
    print(f"   tokens={ck_a.get('tokens_done')} step={ck_a.get('step')}")
    print(f"B: {args.ckpt_b}")
    print(f"   trained operator: {training_operator_of(cfg_b)}")
    print(f"   tokens={ck_b.get('tokens_done')} step={ck_b.get('step')}")
    print()

    problems = []
    for k in CRITICAL_CONFIG_KEYS:
        va, vb = cfg_a.get(k), cfg_b.get(k)
        if va != vb:
            problems.append(f"  {k}: A={va!r} != B={vb!r}")
    if ck_a.get("tokens_done") != ck_b.get("tokens_done"):
        problems.append(
            f"  tokens_done: A={ck_a.get('tokens_done')} != B={ck_b.get('tokens_done')}"
        )
    if ck_a.get("step") != ck_b.get("step"):
        problems.append(f"  step: A={ck_a.get('step')} != B={ck_b.get('step')}")

    other = []
    keys = set(cfg_a) | set(cfg_b)
    for k in sorted(keys - set(CRITICAL_CONFIG_KEYS) - IGNORED_CONFIG_KEYS):
        if cfg_a.get(k) != cfg_b.get(k):
            other.append(f"  {k}: A={cfg_a.get(k)!r} B={cfg_b.get(k)!r}")

    if problems:
        print("NOT PAIRABLE. Critical differences:")
        print("\n".join(problems))
    else:
        print("PAIRABLE: all critical training config keys and endpoints match.")
    if other:
        print("\nNon-critical differences (informational):")
        print("\n".join(other))
    return 0 if not problems else 1


# --------------------------------------------------------------------------
# subcommand: matrix  (P0-1)
# --------------------------------------------------------------------------

def load_manifest(path):
    """Manifest maps a short model name to a checkpoint path.

    {
      "exact":      "/abs/or/rel/exact_fp32_softmax_100m/ckpt.pt",
      "lerp_K32":   "runs/q32_lerp_fullgrad_100m/ckpt.pt",
      ...
    }

    The key "exact" is required: it identifies theta_E, without which the
    adaptation score A(O) is undefined.
    """
    with open(path) as f:
        man = json.load(f)
    if "exact" not in man:
        raise ValueError(
            "manifest must contain an 'exact' entry pointing at the "
            "Exact-trained checkpoint (theta_E); A(O) is undefined without it."
        )
    for name, p in man.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"manifest entry {name!r}: missing {p}")
    return man


def cmd_matrix(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_ctx = make_autocast(device_type, args.dtype)

    manifest = load_manifest(args.manifest)
    k_values = [int(k) for k in args.k_values.split(",")]
    op_keys = default_operator_keys(k_values)

    evalset = FullCoverageEvalSet(
        os.path.join(args.data_dir, "val.bin"), args.block_size, subset=args.subset
    )
    print(json.dumps(evalset.describe(), indent=2))
    print(f"models: {list(manifest)}")
    print(f"eval operators: {op_keys}")
    print()

    os.makedirs(args.out, exist_ok=True)
    results = {}   # results[model_name][op_key] = per-block vector

    for model_name, ckpt_path in manifest.items():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model, cfg = load_model(ckpt_path, device, ckpt=ckpt)
        trained_op = training_operator_of(cfg)
        results[model_name] = {}
        for op in op_keys:
            attn_mode, K, approx = parse_operator_key(op)
            set_operator(model, attn_mode, K, approx)
            t0 = time.time()
            vec = eval_per_block(model, evalset, device, autocast_ctx, args.batch)
            dt = time.time() - t0
            results[model_name][op] = vec
            print(f"{model_name:>12s} x {op:>12s}  NLL={vec.mean():.6f}  ({dt:.1f}s)")
        del model, ckpt
        if device_type == "cuda":
            torch.cuda.empty_cache()
        print()

    # ---- save raw per-block vectors -------------------------------------
    raw_path = os.path.join(args.out, "per_block_nll.npz")
    flat = {f"{m}|{o}": v for m, d in results.items() for o, v in d.items()}
    np.savez_compressed(raw_path, **flat)
    print(f"saved raw per-block NLL vectors: {raw_path}")

    # ---- adaptation score ------------------------------------------------
    # D_E(O) = L(theta_E, O) - L(theta_E, E)
    # D_O(O) = L(theta_O, O) - L(theta_O, E)
    # A(O)   = D_E(O) - D_O(O)
    n = len(evalset)
    summary = {
        "protocol": evalset.describe(),
        "dtype": args.dtype,
        "manifest": manifest,
        "operator_keys": op_keys,
        "mean_nll": {m: {o: float(v.mean()) for o, v in d.items()}
                     for m, d in results.items()},
        "adaptation": {},
    }

    E = "exact"
    for model_name in manifest:
        if model_name == E:
            continue
        # The operator this model was trained with, as an eval-operator key.
        attn, K, approx = training_operator_of(
            read_checkpoint_config(manifest[model_name])[0]
        )
        if attn != "quant":
            continue
        op = operator_key(attn, K, approx)
        if op not in results[model_name]:
            print(f"warning: {model_name} trained operator {op} not in eval grid")
            continue

        vE_O = results[E][op]
        vE_E = results[E][E]
        vO_O = results[model_name][op]
        vO_E = results[model_name][E]

        def stat_DE(idx):
            return vE_O[idx].mean() - vE_E[idx].mean()

        def stat_DO(idx):
            return vO_O[idx].mean() - vO_E[idx].mean()

        def stat_A(idx):
            return (vE_O[idx].mean() - vE_E[idx].mean()) \
                 - (vO_O[idx].mean() - vO_E[idx].mean())

        entry = {
            "operator": op,
            "D_E": paired_bootstrap(stat_DE, n, args.n_boot, args.boot_seed),
            "D_O": paired_bootstrap(stat_DO, n, args.n_boot, args.boot_seed),
            "A": paired_bootstrap(stat_A, n, args.n_boot, args.boot_seed),
        }
        summary["adaptation"][model_name] = entry

        a = entry["A"]
        print(f"{model_name:>12s}  D_E={entry['D_E']['point']:+.6f}  "
              f"D_O={entry['D_O']['point']:+.6f}  "
              f"A={a['point']:+.6f}  95% CI [{a['ci_lo']:+.6f}, {a['ci_hi']:+.6f}]"
              f"{'  *' if a['ci_lo'] > 0 else ''}")

    out_json = os.path.join(args.out, "matrix_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {out_json}")
    print("\nNote: A(O) > 0 is evidence of reduced operator sensitivity after "
          "operator-aware training, not a causal proof (theta_E != theta_O).")
    return 0


# --------------------------------------------------------------------------
# subcommand: temp  (P0-2)
# --------------------------------------------------------------------------

def cmd_temp(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_ctx = make_autocast(device_type, args.dtype)

    k_values = [int(k) for k in args.k_values.split(",")]
    op_keys = default_operator_keys(k_values)

    coarse_alphas = [round(a, 4) for a in np.arange(
        args.alpha_min, args.alpha_max + 1e-9, args.alpha_step)]

    coarse_set = FullCoverageEvalSet(
        os.path.join(args.data_dir, "val.bin"), args.block_size,
        subset=args.coarse_blocks)
    fine_set = FullCoverageEvalSet(
        os.path.join(args.data_dir, "val.bin"), args.block_size,
        subset=args.fine_blocks)

    print("coarse:", json.dumps(coarse_set.describe()))
    print("fine:  ", json.dumps(fine_set.describe()))
    print(f"alphas (coarse): {coarse_alphas[0]} .. {coarse_alphas[-1]} "
          f"({len(coarse_alphas)} points)")
    print("temperature injection: direct FP32 score scaling S -> alpha*S")
    print("entropy: raw H and H/log(valid_length), primary means exclude q=0")
    print()

    os.makedirs(args.out, exist_ok=True)
    model, cfg = load_model(args.ckpt, device)

    # ------------------------------------------------------------------
    # Implementation sanity before monkey-patching: alpha=1 under the
    # direct-score evaluator must reproduce the original exact attention NLL.
    # ------------------------------------------------------------------
    sanity_set = FullCoverageEvalSet(
        os.path.join(args.data_dir, "val.bin"), args.block_size,
        subset=min(args.sanity_blocks, len(coarse_set.all_starts)),
    )
    set_operator(model, "exact_fp32_softmax", None, None)
    baseline = eval_per_block(model, sanity_set, device, autocast_ctx, args.batch)

    temp = ScoreTemperatureController(model)
    temp.set(1.0)
    patched = eval_temp_point(
        model, sanity_set, device, autocast_ctx, temp, args.batch
    )["nll"]
    sanity_max_abs = float(np.max(np.abs(baseline - patched)))
    sanity_mean_abs = float(np.mean(np.abs(baseline - patched)))
    print(f"alpha=1 exact-forward parity: max|dNLL_block|={sanity_max_abs:.3e}  "
          f"mean={sanity_mean_abs:.3e}  n={len(sanity_set)}")
    if sanity_max_abs > args.sanity_tol:
        temp.close()
        raise RuntimeError(
            "direct score-temperature implementation failed alpha=1 parity: "
            f"max block dNLL={sanity_max_abs:.3e} > --sanity-tol={args.sanity_tol:.3e}"
        )

    out = {
        "checkpoint": args.ckpt,
        "trained_operator": training_operator_of(cfg),
        "dtype": args.dtype,
        "temperature_semantics": "direct_fp32_score_scaling",
        "score_transform": "S_alpha = alpha * (Q K^T / sqrt(d_h)), before masking/reconstruction",
        "entropy_semantics": {
            "raw": "mean row entropy H(P), pooled across layers/heads/query rows",
            "normalized": "H(P)/log(valid_row_length)",
            "primary_excludes_q0": True,
        },
        "alpha1_exact_parity": {
            "n_blocks": len(sanity_set),
            "max_abs_block_nll_diff": sanity_max_abs,
            "mean_abs_block_nll_diff": sanity_mean_abs,
            "tolerance": args.sanity_tol,
            "pass": sanity_max_abs <= args.sanity_tol,
        },
        "coarse_protocol": coarse_set.describe(),
        "fine_protocol": fine_set.describe(),
        "curves": {},
    }

    try:
        for op in op_keys:
            attn_mode, K, approx = parse_operator_key(op)
            set_operator(model, attn_mode, K, approx)

            # ---- stage 1: coarse scan on representative subset ----------
            coarse_nll = []
            coarse_H = []
            coarse_Hn = []
            for a in coarse_alphas:
                temp.set(a)
                r = eval_temp_point(
                    model, coarse_set, device, autocast_ctx, temp, args.batch
                )
                coarse_nll.append(float(r["nll"].mean()))
                coarse_H.append(float(r["entropy_raw"].mean()))
                coarse_Hn.append(float(r["entropy_norm"].mean()))
            best_i = int(np.argmin(coarse_nll))
            a_star_coarse = coarse_alphas[best_i]
            coarse_best_at_boundary = best_i in (0, len(coarse_alphas) - 1)
            if coarse_best_at_boundary:
                print(f"  WARNING {op}: coarse optimum {a_star_coarse:.4f} is at "
                      "the scan boundary; alpha range may be too narrow")

            # ---- stage 2: fine scan near optimum -------------------------
            lo = coarse_alphas[max(0, best_i - 1)]
            hi = coarse_alphas[min(len(coarse_alphas) - 1, best_i + 1)]
            fine_alphas = [round(x, 4) for x in
                           np.arange(lo, hi + 1e-9, args.alpha_step / 5)]
            fine_nll = []
            fine_H = []
            fine_Hn = []
            fine_H_layer = []
            fine_Hn_layer = []
            fine_vecs = {}

            for a in fine_alphas:
                temp.set(a)
                r = eval_temp_point(
                    model, fine_set, device, autocast_ctx, temp, args.batch
                )
                fine_nll.append(float(r["nll"].mean()))
                fine_H.append(float(r["entropy_raw"].mean()))
                fine_Hn.append(float(r["entropy_norm"].mean()))
                fine_H_layer.append(r["entropy_raw_by_layer"].mean(axis=0).tolist())
                fine_Hn_layer.append(r["entropy_norm_by_layer"].mean(axis=0).tolist())

                key = f"alpha_{a}"
                fine_vecs[f"nll|{key}"] = r["nll"]
                fine_vecs[f"entropy_raw|{key}"] = r["entropy_raw"]
                fine_vecs[f"entropy_norm|{key}"] = r["entropy_norm"]
                fine_vecs[f"entropy_raw_by_layer|{key}"] = r["entropy_raw_by_layer"]
                fine_vecs[f"entropy_norm_by_layer|{key}"] = r["entropy_norm_by_layer"]

            best_j = int(np.argmin(fine_nll))
            a_star = fine_alphas[best_j]

            out["curves"][op] = {
                "coarse_alphas": coarse_alphas,
                "coarse_nll": coarse_nll,
                "coarse_entropy_raw": coarse_H,
                "coarse_entropy_norm": coarse_Hn,
                "alpha_star_coarse": a_star_coarse,
                "coarse_best_at_boundary": coarse_best_at_boundary,
                "fine_alphas": fine_alphas,
                "fine_nll": fine_nll,
                "fine_entropy_raw": fine_H,
                "fine_entropy_norm": fine_Hn,
                "fine_entropy_raw_by_layer": fine_H_layer,
                "fine_entropy_norm_by_layer": fine_Hn_layer,
                "alpha_star": a_star,
                "nll_at_alpha_star": fine_nll[best_j],
                "entropy_raw_at_alpha_star": fine_H[best_j],
                "entropy_norm_at_alpha_star": fine_Hn[best_j],
            }

            print(f"{op:>12s}  alpha*={a_star:.4f}  "
                  f"NLL={fine_nll[best_j]:.6f}  H={fine_H[best_j]:.6f}  "
                  f"Hnorm={fine_Hn[best_j]:.6f}  "
                  f"(coarse {a_star_coarse:.2f})")

            np.savez_compressed(
                os.path.join(args.out, f"per_block_{op}.npz"), **fine_vecs
            )
    finally:
        temp.close()

    out_json = os.path.join(args.out, "temperature_summary.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {out_json}")
    print("\nImplementation sanity is alpha=1 parity with the original exact "
          "forward, not a requirement that the validation-optimal alpha* equal 1.")
    print("\nH4a: compare alpha* with the frozen R_central/R_Exact ratios as a "
          "CONSISTENCY CHECK, not a prior magnitude prediction.  A mismatch is "
          "scientifically informative because P0-3 already shows layer-selective "
          "rather than purely global scale adaptation.")
    return 0


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data-dir", default="data/wikitext103")
        p.add_argument("--block-size", type=int, default=1024)
        p.add_argument("--batch", type=int, default=2,
                       help="blocks per forward; halves automatically on OOM")
        p.add_argument("--device", default=None)
        p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                       help="bf16 matches the training autocast setting; "
                            "attention internals are fp32 in either case")
        p.add_argument("--k-values", default="4,8,16,32")

    pv = sub.add_parser("verify", help="check two checkpoints are pairable")
    pv.add_argument("--ckpt-a", required=True)
    pv.add_argument("--ckpt-b", required=True)

    pm = sub.add_parser("matrix", help="P0-1 Train x Eval matrix and A(O)")
    common(pm)
    pm.add_argument("--manifest", required=True)
    pm.add_argument("--out", required=True)
    pm.add_argument("--subset", type=int, default=None,
                    help="use a deterministic evenly-spaced subset of blocks")
    pm.add_argument("--n-boot", type=int, default=10000)
    pm.add_argument("--boot-seed", type=int, default=0)

    pt = sub.add_parser("temp", help="P0-2 post-hoc temperature sweep")
    common(pt)
    pt.add_argument("--ckpt", required=True,
                    help="the Exact-trained checkpoint (theta_E)")
    pt.add_argument("--out", required=True)
    pt.add_argument("--alpha-min", type=float, default=0.30)
    pt.add_argument("--alpha-max", type=float, default=1.60)
    pt.add_argument("--alpha-step", type=float, default=0.05)
    pt.add_argument("--coarse-blocks", type=int, default=60,
                    help="blocks used for the coarse scan")
    pt.add_argument("--fine-blocks", type=int, default=None,
                    help="optional deterministic subset for the fine scan; "
                         "default=None means full coverage. Use only for smoke tests")
    pt.add_argument("--sanity-blocks", type=int, default=8,
                    help="blocks for alpha=1 exact-forward parity check")
    pt.add_argument("--sanity-tol", type=float, default=1e-7,
                    help="max allowed per-block NLL difference for alpha=1 parity")

    args = ap.parse_args()
    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "matrix":
        return cmd_matrix(args)
    if args.cmd == "temp":
        return cmd_temp(args)
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
