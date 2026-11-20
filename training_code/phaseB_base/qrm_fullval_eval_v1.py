#!/usr/bin/env python3

"""
Deterministic matched full-validation evaluator for QRM K4 seed1337 @100M.

Protocol intentionally mirrors frozen posthoc_eval.py:
  - WikiText-103 val.bin
  - uint16 token stream
  - non-overlapping context-1024 blocks
  - 243 full blocks for the canonical val.bin
  - BF16 autocast
  - logits cast to FP32 before cross entropy
  - per-token CE -> per-block mean NLL
  - final NLL = mean of equally-sized block means

This evaluator imports the QRM donor implementation from code/phaseB_rr.
The QRM implementation is a superset of the old MinMax implementation.

Three native endpoints are evaluated:
  1. Exact-trained weights + Exact operator
  2. MinMax-Neareast-K4-trained weights + MinMax Nearest K4
  3. QRM-K4-trained weights + QRM Nearest K4, tau=6

The first two are frozen parity anchors.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
QRM_CODE = ROOT / "code" / "phaseB_rr"

# QRM donor must take precedence over root model.py / quant_attention.py.
sys.path.insert(0, str(QRM_CODE))

from model import GPT, GPTConfig  # noqa: E402
import model as qrm_model_module  # noqa: E402
import quant_attention as qrm_quant_module  # noqa: E402


# ---------------------------------------------------------------------
# Frozen full-coverage validation protocol
# ---------------------------------------------------------------------

class FullCoverageEvalSet:
    def __init__(self, path, block_size):
        data = np.memmap(path, dtype=np.uint16, mode="r")

        self.n_tokens = len(data)
        self.block_size = int(block_size)

        last_start = self.n_tokens - self.block_size - 1

        if last_start < 0:
            raise ValueError(
                f"{path} has {self.n_tokens} tokens; "
                f"too few for block_size={self.block_size}"
            )

        self.starts = list(
            range(
                0,
                last_start + 1,
                self.block_size,
            )
        )

        x = np.stack([
            data[i:i + self.block_size]
            for i in self.starts
        ]).astype(np.int64)

        y = np.stack([
            data[i + 1:i + 1 + self.block_size]
            for i in self.starts
        ]).astype(np.int64)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.starts)

    def describe(self):
        return {
            "n_tokens_in_file": int(self.n_tokens),
            "block_size": int(self.block_size),
            "n_blocks_full_coverage": int(len(self.starts)),
            "n_blocks_used": int(len(self.starts)),
            "tokens_covered": int(
                len(self.starts) * self.block_size
            ),
            "is_subset": False,
        }


def make_autocast(device_type, dtype):
    if dtype == "fp32" or device_type != "cuda":
        return torch.autocast(
            device_type=device_type,
            enabled=False,
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 requested but GPU does not support BF16"
        )

    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def checkpoint_metadata(path):
    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return ck, ck.get("config", {})


def build_model(
    ckpt,
    cfg,
    device,
    *,
    mode,
    K=4,
    approx="nearest",
    range_mode="minmax",
    range_tau=6.0,
):
    gcfg = GPTConfig(
        block_size=int(cfg.get("block_size", 1024)),
        n_layer=int(cfg.get("n_layer", 12)),
        n_head=int(cfg.get("n_head", 12)),
        n_embd=int(cfg.get("n_embd", 768)),

        attn_mode=mode,
        attn_K=int(K),
        attn_approx=approx,
        attn_range_grad="full",

        attn_range_mode=range_mode,
        attn_range_tau=float(range_tau),
    )

    model = GPT(gcfg).to(device)

    missing, unexpected = model.load_state_dict(
        ckpt["model"],
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "state_dict mismatch\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}"
        )

    model.grad_checkpointing = False

    # Explicitly set all runtime operator attributes.
    # This avoids relying on constructor/config defaults.
    for blk in model.transformer.h:
        a = blk.attn

        a.mode = mode
        a.K = int(K)
        a.approx = approx
        a.range_grad = "full"

        if hasattr(a, "range_mode"):
            a.range_mode = range_mode

        if hasattr(a, "range_tau"):
            a.range_tau = float(range_tau)

        a.collect_stats = False

    model.eval()

    return model


# ---------------------------------------------------------------------
# Exact frozen NLL computation
# ---------------------------------------------------------------------

@torch.no_grad()
def eval_per_block(
    model,
    evalset,
    device,
    autocast_ctx,
    batch=2,
):
    out = np.zeros(
        len(evalset),
        dtype=np.float64,
    )

    i = 0
    n = len(evalset)
    cur_batch = int(batch)

    while i < n:
        j = min(i + cur_batch, n)

        x = evalset.x[i:j].to(
            device,
            non_blocking=True,
        )
        y = evalset.y[i:j].to(
            device,
            non_blocking=True,
        )

        try:
            with autocast_ctx:
                logits, _ = model(x)

                # Same numerical convention as frozen posthoc_eval.py:
                # logits -> FP32 before CE.
                ls = torch.nn.functional.cross_entropy(
                    logits.float().view(
                        -1,
                        logits.size(-1),
                    ),
                    y.view(-1),
                    reduction="none",
                ).view(y.shape).mean(dim=-1)

            out[i:j] = (
                ls.double()
                .cpu()
                .numpy()
            )

            i = j

        except torch.cuda.OutOfMemoryError:
            if cur_batch == 1:
                raise

            torch.cuda.empty_cache()

            cur_batch = max(
                1,
                cur_batch // 2,
            )

            print(
                f"OOM: reducing eval batch to {cur_batch}",
                file=sys.stderr,
            )

    return out


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def paired_bootstrap_diff(
    a,
    b,
    *,
    n_boot=10000,
    seed=0,
):
    """
    Difference a - b using matched validation blocks.
    """

    if len(a) != len(b):
        raise ValueError("vector length mismatch")

    d = np.asarray(a) - np.asarray(b)

    rng = np.random.default_rng(seed)

    draws = np.empty(
        n_boot,
        dtype=np.float64,
    )

    n = len(d)

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        draws[i] = d[idx].mean()

    return {
        "point": float(d.mean()),
        "ci_lo": float(
            np.quantile(draws, 0.025)
        ),
        "ci_hi": float(
            np.quantile(draws, 0.975)
        ),
        "boot_mean": float(draws.mean()),
        "boot_sd": float(
            draws.std(ddof=1)
        ),
        "n_boot": int(n_boot),
    }


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-dir",
        required=True,
    )

    ap.add_argument(
        "--exact-ckpt",
        required=True,
    )

    ap.add_argument(
        "--minmax-k4-ckpt",
        required=True,
    )

    ap.add_argument(
        "--qrm-k4-ckpt",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--device",
        default="cuda",
    )

    ap.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--n-boot",
        type=int,
        default=10000,
    )

    args = ap.parse_args()

    device = args.device

    device_type = (
        "cuda"
        if device.startswith("cuda")
        else "cpu"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exact_path = Path(args.exact_ckpt)
    minmax_path = Path(args.minmax_k4_ckpt)
    qrm_path = Path(args.qrm_k4_ckpt)

    # ---------------------------------------------------------------
    # Verify implementation source
    # ---------------------------------------------------------------

    print("=" * 80)
    print("IMPLEMENTATION")
    print("=" * 80)

    print(
        "model.py:",
        Path(qrm_model_module.__file__).resolve(),
    )

    print(
        "quant_attention.py:",
        Path(qrm_quant_module.__file__).resolve(),
    )

    print(
        "model SHA256:",
        sha256(qrm_model_module.__file__),
    )

    print(
        "quant SHA256:",
        sha256(qrm_quant_module.__file__),
    )

    # ---------------------------------------------------------------
    # Load metadata and verify endpoint pairing
    # ---------------------------------------------------------------

    exact_ck, exact_cfg = checkpoint_metadata(
        exact_path
    )

    mm_ck, mm_cfg = checkpoint_metadata(
        minmax_path
    )

    qrm_ck, qrm_cfg = checkpoint_metadata(
        qrm_path
    )

    print()
    print("=" * 80)
    print("CHECKPOINT ENDPOINTS")
    print("=" * 80)

    for name, ck, cfg, path in [
        ("Exact", exact_ck, exact_cfg, exact_path),
        ("MinMax K4", mm_ck, mm_cfg, minmax_path),
        ("QRM K4", qrm_ck, qrm_cfg, qrm_path),
    ]:
        print(
            f"{name:10s}: "
            f"tokens={ck.get('tokens_done')} "
            f"step={ck.get('step')} "
            f"path={path}"
        )

    endpoints = [
        (
            exact_ck.get("tokens_done"),
            exact_ck.get("step"),
        ),
        (
            mm_ck.get("tokens_done"),
            mm_ck.get("step"),
        ),
        (
            qrm_ck.get("tokens_done"),
            qrm_ck.get("step"),
        ),
    ]

    if len(set(endpoints)) != 1:
        raise RuntimeError(
            "CHECKPOINT ENDPOINTS ARE NOT MATCHED: "
            f"{endpoints}"
        )

    critical = [
        "seed",
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "micro_batch",
        "batch_tokens",
        "max_tokens",
        "warmup_tokens",
        "lr",
        "min_lr_ratio",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
    ]

    print()
    print("=" * 80)
    print("MATCHED TRAINING CONFIG")
    print("=" * 80)

    problems = []

    for key in critical:
        vals = [
            exact_cfg.get(key),
            mm_cfg.get(key),
            qrm_cfg.get(key),
        ]

        ok = (
            vals[0]
            == vals[1]
            == vals[2]
        )

        print(
            f"{key:18s}: "
            f"{vals} "
            f"{'PASS' if ok else 'FAIL'}"
        )

        if not ok:
            problems.append(
                (key, vals)
            )

    if problems:
        raise RuntimeError(
            f"critical config mismatch: {problems}"
        )

    print(
        "\nPAIRING CHECK: PASS"
    )

    # ---------------------------------------------------------------
    # Validation set
    # ---------------------------------------------------------------

    evalset = FullCoverageEvalSet(
        Path(args.data_dir) / "val.bin",
        block_size=1024,
    )

    protocol = evalset.describe()

    print()
    print("=" * 80)
    print("FULL VALIDATION PROTOCOL")
    print("=" * 80)

    print(
        json.dumps(
            protocol,
            indent=2,
        )
    )

    if (
        protocol["n_tokens_in_file"]
        != 249750
        or
        protocol["n_blocks_used"]
        != 243
        or
        protocol["tokens_covered"]
        != 248832
    ):
        raise RuntimeError(
            "unexpected WikiText validation protocol: "
            f"{protocol}"
        )

    autocast_ctx = make_autocast(
        device_type,
        args.dtype,
    )

    # ---------------------------------------------------------------
    # Exact
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("1/3 EXACT NATIVE")
    print("=" * 80)

    model = build_model(
        exact_ck,
        exact_cfg,
        device,
        mode="exact_fp32_softmax",
        K=4,
        approx="nearest",
        range_mode="minmax",
        range_tau=6.0,
    )

    t0 = time.time()

    v_exact = eval_per_block(
        model,
        evalset,
        device,
        autocast_ctx,
        args.batch,
    )

    dt = time.time() - t0

    del model

    if device_type == "cuda":
        torch.cuda.empty_cache()

    exact_nll = float(
        v_exact.mean()
    )

    print(
        f"Exact native NLL = "
        f"{exact_nll:.9f} "
        f"({dt:.1f}s)"
    )

    # ---------------------------------------------------------------
    # MinMax K4
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("2/3 MINMAX NEAREST K4 NATIVE")
    print("=" * 80)

    model = build_model(
        mm_ck,
        mm_cfg,
        device,
        mode="quant",
        K=4,
        approx="nearest",
        range_mode="minmax",
        range_tau=6.0,
    )

    t0 = time.time()

    v_mm = eval_per_block(
        model,
        evalset,
        device,
        autocast_ctx,
        args.batch,
    )

    dt = time.time() - t0

    del model

    if device_type == "cuda":
        torch.cuda.empty_cache()

    mm_nll = float(
        v_mm.mean()
    )

    print(
        f"MinMax Nearest K4 native NLL = "
        f"{mm_nll:.9f} "
        f"({dt:.1f}s)"
    )

    # ---------------------------------------------------------------
    # QRM K4
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("3/3 QRM NEAREST K4 TAU6 NATIVE")
    print("=" * 80)

    model = build_model(
        qrm_ck,
        qrm_cfg,
        device,
        mode="quant",
        K=4,
        approx="nearest",
        range_mode="relevant_zero",
        range_tau=6.0,
    )

    # Runtime attribute sanity.
    first = model.transformer.h[0].attn

    print(
        "runtime operator:",
        {
            "mode": first.mode,
            "K": first.K,
            "approx": first.approx,
            "range_grad": first.range_grad,
            "range_mode": getattr(
                first,
                "range_mode",
                None,
            ),
            "range_tau": getattr(
                first,
                "range_tau",
                None,
            ),
        },
    )

    assert first.mode == "quant"
    assert first.K == 4
    assert first.approx == "nearest"
    assert first.range_mode == "relevant_zero"
    assert math.isclose(
        float(first.range_tau),
        6.0,
    )

    t0 = time.time()

    v_qrm = eval_per_block(
        model,
        evalset,
        device,
        autocast_ctx,
        args.batch,
    )

    dt = time.time() - t0

    del model

    if device_type == "cuda":
        torch.cuda.empty_cache()

    qrm_nll = float(
        v_qrm.mean()
    )

    print(
        f"QRM Nearest K4 tau6 native NLL = "
        f"{qrm_nll:.9f} "
        f"({dt:.1f}s)"
    )

    # ---------------------------------------------------------------
    # Frozen parity anchors
    # ---------------------------------------------------------------

    FROZEN_EXACT = 4.1498893769
    FROZEN_MINMAX_K4 = 4.34757853

    exact_err = abs(
        exact_nll - FROZEN_EXACT
    )

    mm_err = abs(
        mm_nll - FROZEN_MINMAX_K4
    )

    print()
    print("=" * 80)
    print("FROZEN EVALUATOR PARITY")
    print("=" * 80)

    print(
        f"Exact:  current={exact_nll:.9f} "
        f"frozen={FROZEN_EXACT:.9f} "
        f"abs_err={exact_err:.3e}"
    )

    print(
        f"MinMax K4: current={mm_nll:.9f} "
        f"frozen={FROZEN_MINMAX_K4:.9f} "
        f"abs_err={mm_err:.3e}"
    )

    PARITY_TOL = 1e-4

    if (
        exact_err > PARITY_TOL
        or
        mm_err > PARITY_TOL
    ):
        raise RuntimeError(
            "FROZEN PARITY FAILED. "
            "Do not interpret QRM result."
        )

    print(
        "FROZEN PARITY: PASS"
    )

    # ---------------------------------------------------------------
    # Paired-block comparisons
    # ---------------------------------------------------------------

    qrm_vs_exact = paired_bootstrap_diff(
        v_qrm,
        v_exact,
        n_boot=args.n_boot,
        seed=0,
    )

    mm_vs_exact = paired_bootstrap_diff(
        v_mm,
        v_exact,
        n_boot=args.n_boot,
        seed=0,
    )

    qrm_vs_mm = paired_bootstrap_diff(
        v_qrm,
        v_mm,
        n_boot=args.n_boot,
        seed=0,
    )

    recovered_fraction = (
        (mm_nll - qrm_nll)
        /
        (mm_nll - exact_nll)
        if mm_nll != exact_nll
        else None
    )

    summary = {
        "protocol": protocol,
        "dtype": args.dtype,
        "eval_batch_requested": args.batch,

        "implementation": {
            "model_py": str(
                Path(
                    qrm_model_module.__file__
                ).resolve()
            ),
            "quant_attention_py": str(
                Path(
                    qrm_quant_module.__file__
                ).resolve()
            ),
            "model_sha256": sha256(
                qrm_model_module.__file__
            ),
            "quant_attention_sha256": sha256(
                qrm_quant_module.__file__
            ),
        },

        "checkpoints": {
            "exact": str(exact_path),
            "minmax_nearest_k4": str(
                minmax_path
            ),
            "qrm_nearest_k4_tau6": str(
                qrm_path
            ),
        },

        "endpoint": {
            "tokens_done": int(
                qrm_ck["tokens_done"]
            ),
            "step": int(
                qrm_ck["step"]
            ),
        },

        "mean_nll": {
            "exact": exact_nll,
            "minmax_nearest_k4": mm_nll,
            "qrm_nearest_k4_tau6": qrm_nll,
        },

        "paired_deltas": {
            "minmax_minus_exact": mm_vs_exact,
            "qrm_minus_exact": qrm_vs_exact,
            "qrm_minus_minmax": qrm_vs_mm,
        },

        "fraction_of_minmax_native_gap_recovered": (
            None
            if recovered_fraction is None
            else float(recovered_fraction)
        ),

        "frozen_parity": {
            "exact_reference": FROZEN_EXACT,
            "minmax_k4_reference": FROZEN_MINMAX_K4,
            "exact_abs_error": exact_err,
            "minmax_abs_error": mm_err,
            "tolerance": PARITY_TOL,
            "pass": True,
        },
    }

    np.savez_compressed(
        out_dir / "per_block_nll.npz",
        exact=v_exact,
        minmax_nearest_k4=v_mm,
        qrm_nearest_k4_tau6=v_qrm,
    )

    with open(
        out_dir / "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("FINAL MATCHED RESULT")
    print("=" * 80)

    print(
        f"Exact                "
        f"{exact_nll:.9f}"
    )

    print(
        f"MinMax Nearest K4    "
        f"{mm_nll:.9f}    "
        f"ΔExact={mm_vs_exact['point']:+.9f} "
        f"95%CI=["
        f"{mm_vs_exact['ci_lo']:+.9f}, "
        f"{mm_vs_exact['ci_hi']:+.9f}]"
    )

    print(
        f"QRM Nearest K4 tau6  "
        f"{qrm_nll:.9f}    "
        f"ΔExact={qrm_vs_exact['point']:+.9f} "
        f"95%CI=["
        f"{qrm_vs_exact['ci_lo']:+.9f}, "
        f"{qrm_vs_exact['ci_hi']:+.9f}]"
    )

    print(
        f"QRM - MinMax         "
        f"{qrm_vs_mm['point']:+.9f} "
        f"95%CI=["
        f"{qrm_vs_mm['ci_lo']:+.9f}, "
        f"{qrm_vs_mm['ci_hi']:+.9f}]"
    )

    if recovered_fraction is not None:
        print(
            "MinMax native gap recovered = "
            f"{100 * recovered_fraction:.2f}%"
        )

    print()
    print(
        "saved:",
        out_dir / "summary.json",
    )

    print(
        "saved:",
        out_dir / "per_block_nll.npz",
    )


if __name__ == "__main__":
    main()
