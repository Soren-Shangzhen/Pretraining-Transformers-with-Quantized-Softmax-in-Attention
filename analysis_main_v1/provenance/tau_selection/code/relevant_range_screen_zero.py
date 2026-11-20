"""
Post-hoc screen for Softmax-Relevant-Range Nearest attention.

Compare, on ONE fixed trained checkpoint:

  1. Exact softmax
  2. Original adaptive min-max Nearest
  3. Fixed-window Nearest:
         z = s - max(s) <= 0
         z_clip = clamp(z, min=-tau)
         h = tau / K
         quantize z_clip on the fixed grid [-tau, 0]

Important:
- No training.
- No checkpoint modification.
- Same deterministic validation blocks for every condition.
- Nearest arithmetic matches the existing training operator:
      w = w_lerp + (w_nearest - w_lerp).detach()
- Scores below -tau receive zero reconstructed weight.

This experiment tests intrinsic operator fidelity before spending compute on
from-scratch training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import types
from contextlib import contextmanager

import numpy as np
import torch

from posthoc_eval import (
    FullCoverageEvalSet,
    load_manifest,
    load_model,
    make_autocast,
    set_operator,
    eval_per_block,
    paired_bootstrap,
)


# ---------------------------------------------------------------------
# Fixed-window reconstruction
# ---------------------------------------------------------------------

def fixed_window_reconstruct(
    att: torch.Tensor,
    causal: torch.Tensor,
    K: int,
    tau: float,
):
    """
    att: (B,H,T,T), fp32 raw scaled QK scores.

    Grid in max-centered score space:
        [-tau, -tau+h, ..., 0]
        h = tau/K

    All scores below -tau saturate to the lowest level.
    """

    if K < 2:
        raise ValueError(f"K must be >=2, got {K}")

    if tau <= 0:
        raise ValueError(f"tau must be >0, got {tau}")

    B, H, T, _ = att.shape

    valid = causal.view(
        1, 1, T, T
    )

    # Only valid causal keys determine max.
    row_max = (
        att.masked_fill(
            ~valid,
            -torch.inf,
        )
        .amax(
            dim=-1,
            keepdim=True,
        )
    )

    # Shift invariance: maximum is always at z=0.
    z = att - row_max

    # Saturate the irrelevant negative tail.
    zc = z.clamp(
        min=-float(tau),
        max=0.0,
    )

    h = float(tau) / float(K)

    # Grid coordinate:
    #   z=-tau -> 0
    #   z=0    -> K
    rel = (
        zc + float(tau)
    ) / h

    # Interval index 0..K-1.
    r_idx = (
        torch.floor(rel)
        .to(torch.long)
        .clamp(0, K - 1)
    )

    r = r_idx.to(
        dtype=att.dtype
    )

    b_lo = (
        -float(tau)
        + r * h
    )

    # Boundary weights can be globally precomputed in a real implementation.
    grid = torch.arange(
        K + 1,
        device=att.device,
        dtype=att.dtype,
    )

    boundary_z = (
        -float(tau)
        + grid * h
    )

    boundary_w = torch.exp(
        boundary_z
    )

    # Broadcast 1-D lookup table into gather-compatible form.
    shape = [1] * r_idx.ndim
    shape[-1] = K + 1

    table = boundary_w.view(
        *shape
    ).expand(
        *r_idx.shape[:-1],
        K + 1,
    )

    e_lo = torch.gather(
        table,
        -1,
        r_idx,
    )

    e_hi = torch.gather(
        table,
        -1,
        (r_idx + 1).clamp_max(K),
    )

    t = (
        (zc - b_lo) / h
    ).clamp(0.0, 1.0)

    # Match existing Nearest arithmetic exactly:
    # hard nearest forward + LERP surrogate backward.
    w_lerp = (
        (1.0 - t) * e_lo
        + t * e_hi
    )

    choose_upper = (
        t >= 0.5
    )

    w_nearest = torch.where(
        choose_upper,
        e_hi,
        e_lo,
    )

    w = (
        w_lerp
        + (w_nearest - w_lerp).detach()
    )

    # Scores outside the softmax-relevant window are treated as
    # negligible rather than being lifted to the lowest positive boundary.
    #
    # This separates two effects:
    #   1) concentrating K resolution into [-tau, 0]
    #   2) imposing an exp(-tau) probability floor.
    #
    # The present ZERO-TAIL variant keeps (1) but removes (2).
    below_window = (
        (z < -float(tau))
        & valid
    )

    w = torch.where(
        below_window,
        torch.zeros_like(w),
        w,
    )

    w = w.masked_fill(
        ~valid,
        0.0,
    )

    return w


@torch.no_grad()
def fixed_window_attention(
    module,
    q,
    k,
    v,
    tau,
):
    B, H, T, D = q.shape

    with torch.autocast(
        device_type=q.device.type,
        enabled=False,
    ):
        qf = q.float()
        kf = k.float()
        vf = v.float()

        att = (
            qf @ kf.transpose(-2, -1)
        ) / math.sqrt(D)

        causal = torch.ones(
            T,
            T,
            dtype=torch.bool,
            device=q.device,
        ).tril()

        w = fixed_window_reconstruct(
            att=att,
            causal=causal,
            K=module.K,
            tau=tau,
        )

        denom = w.sum(
            dim=-1,
            keepdim=True,
        )

        if not torch.isfinite(
            denom
        ).all():
            raise RuntimeError(
                "non-finite denominator"
            )

        if (denom <= 0).any():
            raise RuntimeError(
                "non-positive denominator"
            )

        P = w / denom

        y = P @ vf

    return y.to(q.dtype)


@contextmanager
def install_fixed_window(
    model,
    tau,
):
    originals = []

    try:
        for block in model.transformer.h:
            module = block.attn

            original = (
                module._quant_attention
            )

            originals.append(
                (module, original)
            )

            def patched(
                self,
                q,
                k,
                v,
                _tau=float(tau),
            ):
                return fixed_window_attention(
                    self,
                    q,
                    k,
                    v,
                    _tau,
                )

            module._quant_attention = (
                types.MethodType(
                    patched,
                    module,
                )
            )

        yield

    finally:
        for module, original in originals:
            module._quant_attention = (
                original
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        required=True,
    )

    ap.add_argument(
        "--model",
        default="exact_t02500.1m",
        help=(
            "checkpoint key from manifest; "
            "use the final Exact-trained model"
        ),
    )

    ap.add_argument(
        "--data-dir",
        default="data/wikitext103",
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--k-values",
        default="4,16,32",
    )

    ap.add_argument(
        "--taus",
        default="8,12,16,20,24,32",
    )

    ap.add_argument(
        "--subset",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--block-size",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=1,
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
        "--n-boot",
        type=int,
        default=10000,
    )

    ap.add_argument(
        "--boot-seed",
        type=int,
        default=20260909,
    )

    args = ap.parse_args()

    Ks = [
        int(x)
        for x in args.k_values.split(",")
    ]

    taus = [
        float(x)
        for x in args.taus.split(",")
    ]

    manifest = load_manifest(
        args.manifest
    )

    if args.model not in manifest:
        raise KeyError(
            f"{args.model!r} not found in manifest.\n"
            f"Available keys: {list(manifest)}"
        )

    ckpt_path = manifest[
        args.model
    ]

    device_type = (
        "cuda"
        if args.device.startswith("cuda")
        else "cpu"
    )

    autocast_ctx = make_autocast(
        device_type,
        args.dtype,
    )

    evalset = FullCoverageEvalSet(
        os.path.join(
            args.data_dir,
            "val.bin",
        ),
        args.block_size,
        subset=args.subset,
    )

    os.makedirs(
        args.out,
        exist_ok=True,
    )

    print(
        json.dumps(
            evalset.describe(),
            indent=2,
        )
    )

    print()
    print("checkpoint:", ckpt_path)
    print("K:", Ks)
    print("tau:", taus)

    model, cfg = load_model(
        ckpt_path,
        args.device,
    )

    vectors = {}

    # -------------------------------------------------------------
    # Exact reference
    # -------------------------------------------------------------

    set_operator(
        model,
        "exact_fp32_softmax",
        None,
        None,
    )

    exact = eval_per_block(
        model,
        evalset,
        args.device,
        autocast_ctx,
        args.batch,
    )

    vectors["exact"] = exact

    print()
    print("=" * 100)
    print(
        f"EXACT NLL = "
        f"{exact.mean():.9f}"
    )
    print("=" * 100)

    results = {
        "checkpoint": ckpt_path,
        "model_key": args.model,
        "protocol":
            evalset.describe(),
        "dtype": args.dtype,
        "exact_mean_nll":
            float(exact.mean()),
        "conditions": {},
    }

    n = len(evalset)

    # -------------------------------------------------------------
    # Baseline min-max + fixed-window sweep
    # -------------------------------------------------------------

    for K in Ks:

        print()
        print("#" * 100)
        print(f"K = {K}")
        print("#" * 100)

        # Original adaptive min-max Nearest.
        set_operator(
            model,
            "quant",
            K,
            "nearest",
        )

        mm = eval_per_block(
            model,
            evalset,
            args.device,
            autocast_ctx,
            args.batch,
        )

        mm_key = f"minmax_K{K}"
        vectors[mm_key] = mm

        mm_stat = paired_bootstrap(
            lambda idx, a=mm, b=exact:
                a[idx].mean()
                - b[idx].mean(),
            n,
            args.n_boot,
            args.boot_seed + K,
        )

        results["conditions"][
            mm_key
        ] = {
            "kind": "minmax",
            "K": K,
            "tau": None,
            "mean_nll":
                float(mm.mean()),
            "delta_vs_exact":
                mm_stat,
        }

        print(
            f"minmax      "
            f"NLL={mm.mean():.9f}  "
            f"ΔExact={mm_stat['point']:+.9f}"
        )

        # Fixed relevant-range variants.
        for ti, tau in enumerate(
            taus
        ):
            set_operator(
                model,
                "quant",
                K,
                "nearest",
            )

            with install_fixed_window(
                model,
                tau,
            ):
                vec = eval_per_block(
                    model,
                    evalset,
                    args.device,
                    autocast_ctx,
                    args.batch,
                )

            key = (
                f"fixed_K{K}_"
                f"tau{tau:g}"
            )

            vectors[key] = vec

            stat_exact = paired_bootstrap(
                lambda idx, a=vec, b=exact:
                    a[idx].mean()
                    - b[idx].mean(),
                n,
                args.n_boot,
                args.boot_seed
                + 1000
                + 100 * K
                + ti,
            )

            stat_mm = paired_bootstrap(
                lambda idx, a=vec, b=mm:
                    a[idx].mean()
                    - b[idx].mean(),
                n,
                args.n_boot,
                args.boot_seed
                + 2000
                + 100 * K
                + ti,
            )

            results["conditions"][
                key
            ] = {
                "kind":
                    "fixed_window",
                "K": K,
                "tau":
                    float(tau),
                "interval_width":
                    float(tau / K),
                "tail_weight_per_key":
                    float(
                        math.exp(-tau)
                    ),
                "worst_case_1024_tail_weight":
                    float(
                        1023
                        * math.exp(-tau)
                    ),
                "mean_nll":
                    float(vec.mean()),
                "delta_vs_exact":
                    stat_exact,
                "delta_vs_minmax":
                    stat_mm,
            }

            print(
                f"tau={tau:5.1f}  "
                f"h={tau/K:7.4f}  "
                f"NLL={vec.mean():.9f}  "
                f"ΔExact="
                f"{stat_exact['point']:+.9f}  "
                f"ΔMinMax="
                f"{stat_mm['point']:+.9f}"
            )

    # -------------------------------------------------------------
    # Best fixed window for each K
    # -------------------------------------------------------------

    print()
    print("=" * 100)
    print("BEST FIXED WINDOW PER K")
    print("=" * 100)

    best_by_K = {}

    for K in Ks:
        candidates = []

        for key, d in (
            results[
                "conditions"
            ].items()
        ):
            if (
                d["kind"]
                == "fixed_window"
                and d["K"] == K
            ):
                candidates.append(
                    (d["mean_nll"], key, d)
                )

        candidates.sort(
            key=lambda x: x[0]
        )

        best_nll, key, d = (
            candidates[0]
        )

        best_by_K[str(K)] = {
            "condition": key,
            **d,
        }

        mm = results[
            "conditions"
        ][f"minmax_K{K}"]

        print(
            f"K={K:2d}  "
            f"best tau={d['tau']:5.1f}  "
            f"h={d['interval_width']:.4f}  "
            f"NLL={best_nll:.9f}  "
            f"vs Exact="
            f"{d['delta_vs_exact']['point']:+.6f}  "
            f"gain over MinMax="
            f"{mm['mean_nll']-best_nll:+.6f}"
        )

    results[
        "best_fixed_by_K"
    ] = best_by_K

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    out_json = os.path.join(
        args.out,
        "relevant_range_screen.json",
    )

    with open(
        out_json,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    out_npz = os.path.join(
        args.out,
        "per_block_nll.npz",
    )

    np.savez_compressed(
        out_npz,
        **vectors,
    )

    print()
    print("saved:", out_json)
    print("saved:", out_npz)
    print("DONE")


if __name__ == "__main__":
    main()
