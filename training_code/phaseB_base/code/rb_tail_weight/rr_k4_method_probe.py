"""
Full-validation probe for the 250M Relevant-Zero K4 model.

Questions:
1. On the SAME trained weights, compare:
      Native Relevant-Zero K4 tau=6
      Exact FP32 softmax
      Original Min-Max Nearest K4
2. Under native Relevant-Zero representations, measure for max-relative
   score thresholds z < -t:
      - fraction of causal-valid keys below threshold
      - mean per-row truncated-key fraction
      - Exact-softmax probability mass carried by those keys
3. Report tau=6 statistics by layer.

Primary summaries exclude q=0 because its causal row has length 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


# =====================================================================
# Deterministic full-coverage validation
# =====================================================================

class FullCoverageEvalSet:
    def __init__(self, path, block_size=1024, subset=None):
        data = np.memmap(path, dtype=np.uint16, mode="r")

        self.n_tokens = len(data)
        self.block_size = block_size

        last_start = self.n_tokens - block_size - 1

        starts = list(
            range(
                0,
                last_start + 1,
                block_size,
            )
        )

        self.n_blocks_full = len(starts)

        if subset is not None and subset < len(starts):
            idx = (
                np.linspace(
                    0,
                    len(starts) - 1,
                    subset,
                )
                .round()
                .astype(int)
            )

            idx = sorted(set(idx.tolist()))
            starts = [starts[i] for i in idx]

        self.starts = starts

        x = np.stack(
            [
                data[i:i + block_size]
                for i in starts
            ]
        ).astype(np.int64)

        y = np.stack(
            [
                data[i + 1:i + 1 + block_size]
                for i in starts
            ]
        ).astype(np.int64)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.starts)

    def describe(self):
        return {
            "n_tokens_in_file":
                int(self.n_tokens),
            "block_size":
                int(self.block_size),
            "n_blocks_full_coverage":
                int(self.n_blocks_full),
            "n_blocks_used":
                int(len(self.starts)),
            "tokens_covered":
                int(len(self.starts) * self.block_size),
            "is_subset":
                len(self.starts) != self.n_blocks_full,
        }


# =====================================================================
# Model loading
# =====================================================================

def load_rr_model(path, device):
    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    c = ck.get("config", {})

    cfg = GPTConfig(
        block_size=int(
            c.get("block_size", 1024)
        ),
        vocab_size=int(
            c.get("vocab_size", 50304)
        ),
        n_layer=int(
            c.get("n_layer", 12)
        ),
        n_head=int(
            c.get("n_head", 12)
        ),
        n_embd=int(
            c.get("n_embd", 768)
        ),
        bias=bool(
            c.get("bias", False)
        ),

        attn_mode="quant",

        attn_K=int(
            c.get(
                "K",
                c.get("attn_K", 4),
            )
        ),

        attn_approx=c.get(
            "approx",
            c.get("attn_approx", "nearest"),
        ),

        attn_range_grad=c.get(
            "range_grad",
            c.get("attn_range_grad", "full"),
        ),

        attn_range_mode=c.get(
            "range_mode",
            c.get(
                "attn_range_mode",
                "relevant_zero",
            ),
        ),

        attn_range_tau=float(
            c.get(
                "range_tau",
                c.get("attn_range_tau", 6.0),
            )
        ),
    )

    model = GPT(cfg).to(device)

    missing, unexpected = model.load_state_dict(
        ck["model"],
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "state_dict mismatch\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}"
        )

    model.eval()
    model.grad_checkpointing = False

    return model, c, ck


def set_operator(
    model,
    mode,
    K=4,
    approx="nearest",
    range_mode="relevant_zero",
    tau=6.0,
):
    for block in model.transformer.h:
        a = block.attn

        a.mode = mode
        a.K = int(K)
        a.approx = approx
        a.range_grad = "full"
        a.range_mode = range_mode
        a.range_tau = float(tau)
        a.collect_stats = False


# =====================================================================
# Evaluation
# =====================================================================

def autocast_ctx(device, dtype):
    if (
        dtype == "bf16"
        and device.startswith("cuda")
    ):
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    device_type = (
        "cuda"
        if device.startswith("cuda")
        else "cpu"
    )

    return torch.autocast(
        device_type=device_type,
        enabled=False,
    )


@torch.no_grad()
def eval_blocks(
    model,
    dataset,
    device,
    dtype,
):
    out = np.zeros(
        len(dataset),
        dtype=np.float64,
    )

    for i in range(len(dataset)):
        x = dataset.x[i:i+1].to(device)
        y = dataset.y[i:i+1].to(device)

        with autocast_ctx(device, dtype):
            logits, _ = model(x)

        loss = F.cross_entropy(
            logits.float().view(
                -1,
                logits.size(-1),
            ),
            y.view(-1),
            reduction="mean",
        )

        out[i] = float(loss)

    return out


# =====================================================================
# Tail geometry controller
# =====================================================================

class TailProbe:
    def __init__(
        self,
        model,
        n_blocks,
        thresholds,
        tau_primary=6.0,
    ):
        self.model = model
        self.n_blocks = n_blocks
        self.thresholds = [
            float(x)
            for x in thresholds
        ]
        self.tau_primary = float(
            tau_primary
        )

        self.n_layer = len(
            model.transformer.h
        )

        shape = (
            n_blocks,
            self.n_layer,
        )

        # Common denominators.
        self.valid_key_count = np.zeros(
            shape,
            dtype=np.float64,
        )

        self.row_count = np.zeros(
            shape,
            dtype=np.float64,
        )

        self.raw_span_sum = np.zeros(
            shape,
            dtype=np.float64,
        )

        # Threshold-specific numerators.
        self.tail_key_count = {
            t: np.zeros(
                shape,
                dtype=np.float64,
            )
            for t in self.thresholds
        }

        self.row_tail_frac_sum = {
            t: np.zeros(
                shape,
                dtype=np.float64,
            )
            for t in self.thresholds
        }

        self.exact_tail_mass_sum = {
            t: np.zeros(
                shape,
                dtype=np.float64,
            )
            for t in self.thresholds
        }

        self.current_block = None
        self.handles = []

        for li, block in enumerate(
            model.transformer.h
        ):
            h = block.attn.register_forward_pre_hook(
                self._make_hook(li)
            )

            self.handles.append(h)

    def close(self):
        for h in self.handles:
            h.remove()

        self.handles = []

    def _make_hook(self, layer_idx):
        @torch.no_grad()
        def hook(module, inputs):
            if self.current_block is None:
                raise RuntimeError(
                    "TailProbe.current_block "
                    "was not set"
                )

            x = inputs[0]

            B, T, C = x.shape
            H = module.n_head
            D = module.head_dim

            # Recompute Q/K with exactly the same projection.
            q, k, _ = module.c_attn(x).split(
                module.n_embd,
                dim=2,
            )

            q = (
                q.view(B, T, H, D)
                .transpose(1, 2)
                .float()
            )

            k = (
                k.view(B, T, H, D)
                .transpose(1, 2)
                .float()
            )

            att = (
                q @ k.transpose(-2, -1)
            ) / math.sqrt(D)

            causal = torch.ones(
                T,
                T,
                dtype=torch.bool,
                device=x.device,
            ).tril()

            valid = causal.view(
                1, 1, T, T
            )

            masked = att.masked_fill(
                ~valid,
                -torch.inf,
            )

            row_max = masked.amax(
                dim=-1,
                keepdim=True,
            )

            row_min = (
                att.masked_fill(
                    ~valid,
                    torch.inf,
                )
                .amin(
                    dim=-1,
                    keepdim=True,
                )
            )

            z = masked - row_max

            # Exact-softmax distribution on the SAME scores.
            w_exact = torch.exp(z)

            p_exact = (
                w_exact
                / w_exact.sum(
                    dim=-1,
                    keepdim=True,
                )
            )

            # Primary statistics exclude q=0.
            q_nonzero = (
                torch.arange(
                    T,
                    device=x.device,
                )
                .view(1, 1, T, 1)
                > 0
            )

            metric_valid = (
                valid
                & q_nonzero
            )

            metric_valid = (
                metric_valid.expand(
                    B,
                    H,
                    T,
                    T,
                )
            )

            row_select = (
                torch.arange(
                    T,
                    device=x.device,
                )
                > 0
            )

            # B,H,T row lengths.
            row_lengths = (
                causal.sum(dim=-1)
                .view(1, 1, T)
                .expand(B, H, T)
            )

            spans = (
                row_max.squeeze(-1)
                - row_min.squeeze(-1)
            )

            spans = spans[
                :, :, row_select
            ]

            bi = self.current_block

            self.valid_key_count[
                bi,
                layer_idx,
            ] += float(
                metric_valid.sum()
            )

            n_rows = (
                B * H * (T - 1)
            )

            self.row_count[
                bi,
                layer_idx,
            ] += float(n_rows)

            self.raw_span_sum[
                bi,
                layer_idx,
            ] += float(
                spans.sum()
            )

            for threshold in self.thresholds:
                tail = (
                    metric_valid
                    & (
                        z
                        < -float(threshold)
                    )
                )

                self.tail_key_count[
                    threshold
                ][
                    bi,
                    layer_idx,
                ] += float(
                    tail.sum()
                )

                tail_per_row = tail.sum(
                    dim=-1
                )

                row_frac = (
                    tail_per_row[
                        :, :, row_select
                    ].float()
                    /
                    row_lengths[
                        :, :, row_select
                    ].float()
                )

                self.row_tail_frac_sum[
                    threshold
                ][
                    bi,
                    layer_idx,
                ] += float(
                    row_frac.sum()
                )

                tail_mass = (
                    p_exact
                    * tail
                ).sum(
                    dim=-1
                )

                tail_mass = tail_mass[
                    :, :, row_select
                ]

                self.exact_tail_mass_sum[
                    threshold
                ][
                    bi,
                    layer_idx,
                ] += float(
                    tail_mass.sum()
                )

        return hook


@torch.no_grad()
def eval_native_with_tail_probe(
    model,
    dataset,
    device,
    dtype,
    thresholds,
):
    probe = TailProbe(
        model,
        len(dataset),
        thresholds,
        tau_primary=6.0,
    )

    nll = np.zeros(
        len(dataset),
        dtype=np.float64,
    )

    try:
        for i in range(len(dataset)):
            probe.current_block = i

            x = dataset.x[i:i+1].to(
                device
            )

            y = dataset.y[i:i+1].to(
                device
            )

            with autocast_ctx(
                device,
                dtype,
            ):
                logits, _ = model(x)

            loss = F.cross_entropy(
                logits.float().view(
                    -1,
                    logits.size(-1),
                ),
                y.view(-1),
                reduction="mean",
            )

            nll[i] = float(loss)

    finally:
        probe.close()

    return nll, probe


# =====================================================================
# Bootstrap helpers
# =====================================================================

def bootstrap_difference(
    a,
    b,
    n_boot=10000,
    seed=20260910,
):
    d = a - b

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
            n,
        )

        draws[i] = d[idx].mean()

    lo, hi = np.quantile(
        draws,
        [0.025, 0.975],
    )

    return {
        "point":
            float(d.mean()),
        "ci_lo":
            float(lo),
        "ci_hi":
            float(hi),
        "mean_abs_block":
            float(np.abs(d).mean()),
        "corr":
            float(
                np.corrcoef(a, b)[0, 1]
            ),
    }


def bootstrap_ratio(
    numerator,
    denominator,
    n_boot=10000,
    seed=20260910,
):
    # Inputs are per-block values after summing across layers.
    rng = np.random.default_rng(seed)

    n = len(numerator)

    point = (
        numerator.sum()
        / denominator.sum()
    )

    draws = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for i in range(n_boot):
        idx = rng.integers(
            0,
            n,
            n,
        )

        draws[i] = (
            numerator[idx].sum()
            / denominator[idx].sum()
        )

    lo, hi = np.quantile(
        draws,
        [0.025, 0.975],
    )

    return {
        "point": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--ckpt",
        required=True,
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
        "--subset",
        type=int,
        default=None,
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

    args = ap.parse_args()

    os.makedirs(
        args.out,
        exist_ok=True,
    )

    dataset = FullCoverageEvalSet(
        os.path.join(
            args.data_dir,
            "val.bin",
        ),
        block_size=1024,
        subset=args.subset,
    )

    print(
        json.dumps(
            dataset.describe(),
            indent=2,
        )
    )

    model, cfg, ck = load_rr_model(
        args.ckpt,
        args.device,
    )

    print()
    print(
        "checkpoint:",
        args.ckpt,
    )

    print(
        "tokens_done:",
        ck.get("tokens_done"),
    )

    print(
        "training config:",
        {
            "attn":
                cfg.get("attn"),
            "K":
                cfg.get("K"),
            "approx":
                cfg.get("approx"),
            "range_mode":
                cfg.get("range_mode"),
            "range_tau":
                cfg.get("range_tau"),
        },
    )

    thresholds = [
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
    ]

    # ---------------------------------------------------------
    # Native Relevant-Zero + tail probe
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("NATIVE RELEVANT-ZERO K4 TAU=6")
    print("=" * 100)

    set_operator(
        model,
        mode="quant",
        K=4,
        approx="nearest",
        range_mode="relevant_zero",
        tau=6.0,
    )

    native, probe = (
        eval_native_with_tail_probe(
            model,
            dataset,
            args.device,
            args.dtype,
            thresholds,
        )
    )

    print(
        f"NLL = {native.mean():.9f}"
    )

    # ---------------------------------------------------------
    # Exact
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("EXACT FP32 SOFTMAX")
    print("=" * 100)

    set_operator(
        model,
        mode="exact_fp32_softmax",
        K=4,
        approx="nearest",
        range_mode="relevant_zero",
        tau=6.0,
    )

    exact = eval_blocks(
        model,
        dataset,
        args.device,
        args.dtype,
    )

    print(
        f"NLL = {exact.mean():.9f}"
    )

    # ---------------------------------------------------------
    # Original Min-Max Nearest K4
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("ORIGINAL MIN-MAX NEAREST K4")
    print("=" * 100)

    set_operator(
        model,
        mode="quant",
        K=4,
        approx="nearest",
        range_mode="minmax",
        tau=6.0,
    )

    minmax = eval_blocks(
        model,
        dataset,
        args.device,
        args.dtype,
    )

    print(
        f"NLL = {minmax.mean():.9f}"
    )

    # ---------------------------------------------------------
    # Paired NLL comparisons
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("PAIRED OPERATOR COUNTERFACTUALS")
    print("=" * 100)

    comparisons = {}

    for name, a, b, seed in [
        (
            "native_minus_exact",
            native,
            exact,
            20260910,
        ),
        (
            "minmax_minus_native",
            minmax,
            native,
            20260911,
        ),
        (
            "minmax_minus_exact",
            minmax,
            exact,
            20260912,
        ),
    ]:
        s = bootstrap_difference(
            a,
            b,
            n_boot=args.n_boot,
            seed=seed,
        )

        comparisons[name] = s

        print(
            f"{name:22s} "
            f"Δ={s['point']:+.9f} "
            f"95%CI=["
            f"{s['ci_lo']:+.9f}, "
            f"{s['ci_hi']:+.9f}] "
            f"MAE={s['mean_abs_block']:.6f} "
            f"corr={s['corr']:.6f}"
        )

    # ---------------------------------------------------------
    # Tail statistics
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "GLOBAL SOFTMAX-RELEVANT TAIL STATISTICS "
        "(q>=1)"
    )
    print("=" * 100)

    tail_summary = {}

    # Sum layers within each block first.
    valid_by_block = (
        probe.valid_key_count.sum(
            axis=1
        )
    )

    rows_by_block = (
        probe.row_count.sum(
            axis=1
        )
    )

    print(
        f"{'threshold':>10s} "
        f"{'key frac':>13s} "
        f"{'95% CI':>27s} "
        f"{'mean row frac':>15s} "
        f"{'exact mass':>13s} "
        f"{'95% CI mass':>27s}"
    )

    for ti, threshold in enumerate(
        thresholds
    ):
        tail_keys = (
            probe.tail_key_count[
                threshold
            ].sum(
                axis=1
            )
        )

        row_frac_sum = (
            probe.row_tail_frac_sum[
                threshold
            ].sum(
                axis=1
            )
        )

        mass_sum = (
            probe.exact_tail_mass_sum[
                threshold
            ].sum(
                axis=1
            )
        )

        key_stat = bootstrap_ratio(
            tail_keys,
            valid_by_block,
            n_boot=args.n_boot,
            seed=3000 + ti,
        )

        row_frac_stat = (
            bootstrap_ratio(
                row_frac_sum,
                rows_by_block,
                n_boot=args.n_boot,
                seed=4000 + ti,
            )
        )

        mass_stat = bootstrap_ratio(
            mass_sum,
            rows_by_block,
            n_boot=args.n_boot,
            seed=5000 + ti,
        )

        tail_summary[
            str(int(threshold))
        ] = {
            "tail_key_fraction":
                key_stat,
            "mean_row_tail_fraction":
                row_frac_stat,
            "mean_exact_softmax_tail_mass":
                mass_stat,
        }

        print(
            f"z<-{threshold:<7g} "
            f"{key_stat['point']:13.6f} "
            f"[{key_stat['ci_lo']:.6f},"
            f"{key_stat['ci_hi']:.6f}] "
            f"{row_frac_stat['point']:15.6f} "
            f"{mass_stat['point']:13.8f} "
            f"[{mass_stat['ci_lo']:.8f},"
            f"{mass_stat['ci_hi']:.8f}]"
        )

    # ---------------------------------------------------------
    # Layerwise tau=6
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "LAYERWISE TAU=6 TAIL STATISTICS "
        "(q>=1)"
    )
    print("=" * 100)

    t = 6.0

    layer_summary = {}

    print(
        f"{'layer':>6s} "
        f"{'raw span':>12s} "
        f"{'tail key frac':>15s} "
        f"{'row tail frac':>15s} "
        f"{'exact tail mass':>17s}"
    )

    for li in range(
        probe.n_layer
    ):
        vk = probe.valid_key_count[
            :, li
        ].sum()

        nr = probe.row_count[
            :, li
        ].sum()

        raw_span = (
            probe.raw_span_sum[
                :, li
            ].sum()
            / nr
        )

        key_frac = (
            probe.tail_key_count[t][
                :, li
            ].sum()
            / vk
        )

        row_frac = (
            probe.row_tail_frac_sum[t][
                :, li
            ].sum()
            / nr
        )

        mass = (
            probe.exact_tail_mass_sum[t][
                :, li
            ].sum()
            / nr
        )

        layer_summary[
            str(li)
        ] = {
            "raw_span_mean":
                float(raw_span),
            "tail_key_fraction":
                float(key_frac),
            "mean_row_tail_fraction":
                float(row_frac),
            "mean_exact_softmax_tail_mass":
                float(mass),
        }

        print(
            f"{li:6d} "
            f"{raw_span:12.6f} "
            f"{key_frac:15.6f} "
            f"{row_frac:15.6f} "
            f"{mass:17.9f}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    np.savez_compressed(
        os.path.join(
            args.out,
            "per_block_probe.npz",
        ),
        native=native,
        exact=exact,
        minmax=minmax,
        valid_key_count=
            probe.valid_key_count,
        row_count=
            probe.row_count,
        raw_span_sum=
            probe.raw_span_sum,
        **{
            f"tail_key_count_t{int(t)}":
                probe.tail_key_count[t]
            for t in thresholds
        },
        **{
            f"row_tail_frac_sum_t{int(t)}":
                probe.row_tail_frac_sum[t]
            for t in thresholds
        },
        **{
            f"exact_tail_mass_sum_t{int(t)}":
                probe.exact_tail_mass_sum[t]
            for t in thresholds
        },
    )

    out = {
        "checkpoint":
            args.ckpt,
        "tokens_done":
            ck.get("tokens_done"),
        "protocol":
            dataset.describe(),
        "native_mean_nll":
            float(native.mean()),
        "exact_mean_nll":
            float(exact.mean()),
        "minmax_mean_nll":
            float(minmax.mean()),
        "comparisons":
            comparisons,
        "tail_summary":
            tail_summary,
        "layerwise_tau6":
            layer_summary,
    }

    with open(
        os.path.join(
            args.out,
            "rr_k4_method_probe.json",
        ),
        "w",
    ) as f:
        json.dump(
            out,
            f,
            indent=2,
        )

    print()
    print(
        "saved:",
        os.path.join(
            args.out,
            "rr_k4_method_probe.json",
        ),
    )

    print(
        "saved:",
        os.path.join(
            args.out,
            "per_block_probe.npz",
        ),
    )

    print("DONE")


if __name__ == "__main__":
    main()
