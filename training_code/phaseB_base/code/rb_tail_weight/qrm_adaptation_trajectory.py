"""
QuantRange-Max adaptation trajectory probe.

For every saved checkpoint of a QuantRange-Max run, measure:

1. key0 structural-anchor geometry
   - P(key0 = argmin)
   - key0 lower-bound gain
   - key0 share of raw min-max span

2. fixed-grid / lattice alignment
   For z = score - row_max and active scores z >= -tau:
       h = tau / K
       z = -tau + (r + t) h
       d_grid = min(t, 1-t)

   Report:
       mean d_grid
       exact-softmax-mass-weighted mean d_grid
       fractions within 0.05 / 0.10 / 0.20 of a grid boundary
       reconstruction-level occupancy

3. tail geometry
   - raw score span
   - fraction of keys below -tau
   - exact-softmax probability mass below -tau

Primary statistics exclude q=0.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np
import torch

from rr_k4_method_probe import (
    FullCoverageEvalSet,
    load_rr_model,
    autocast_ctx,
    set_operator,
)


def token_m_from_name(path):
    name = os.path.basename(path)

    m = re.search(
        r"_([0-9]+(?:\.[0-9]+)?)m\.pt$",
        name,
    )

    if m is None:
        return float("inf")

    return float(m.group(1))


class AdaptationProbe:
    def __init__(self, model, K, tau):
        self.model = model
        self.K = int(K)
        self.tau = float(tau)
        self.h = self.tau / self.K

        self.n_layer = len(
            model.transformer.h
        )

        L = self.n_layer

        self.rows = np.zeros(L)
        self.valid_keys = np.zeros(L)

        self.raw_span_sum = np.zeros(L)

        self.key0_min_count = np.zeros(L)
        self.key0_lower_gain_sum = np.zeros(L)
        self.key0_range_share_sum = np.zeros(L)

        self.tail_key_count = np.zeros(L)
        self.tail_mass_sum = np.zeros(L)

        self.active_key_count = np.zeros(L)
        self.active_mass_sum = np.zeros(L)

        self.dgrid_sum = np.zeros(L)
        self.dgrid_mass_sum = np.zeros(L)

        self.edge05_count = np.zeros(L)
        self.edge10_count = np.zeros(L)
        self.edge20_count = np.zeros(L)

        self.level_counts = np.zeros(
            (L, self.K + 1),
            dtype=np.float64,
        )

        self.t_hist = np.zeros(
            (L, 20),
            dtype=np.float64,
        )

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

    def _make_hook(self, li):
        @torch.no_grad()
        def hook(module, inputs):
            x = inputs[0]

            B, T, C = x.shape
            H = module.n_head
            D = module.head_dim

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

            scores = (
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
            ).expand(
                B, H, T, T
            )

            row_select = (
                torch.arange(
                    T,
                    device=x.device,
                )
                > 0
            )

            scores_for_max = scores.masked_fill(
                ~valid,
                -torch.inf,
            )

            scores_for_min = scores.masked_fill(
                ~valid,
                torch.inf,
            )

            row_max = scores_for_max.amax(
                dim=-1,
                keepdim=True,
            )

            row_min = scores_for_min.amin(
                dim=-1,
                keepdim=True,
            )

            span = (
                row_max - row_min
            ).squeeze(-1)

            # -------------------------------------------------
            # key0 anchor
            # -------------------------------------------------

            argmin = scores_for_min.argmin(
                dim=-1
            )

            key0_is_min = (
                argmin == 0
            )

            valid_no_key0 = valid.expand(
                B, H, T, T
            ).clone()

            valid_no_key0[..., 0] = False

            min_no_key0 = (
                scores.masked_fill(
                    ~valid_no_key0,
                    torch.inf,
                )
                .amin(
                    dim=-1
                )
            )

            row_min_flat = row_min.squeeze(-1)

            lower_gain = (
                min_no_key0
                - row_min_flat
            )

            # q=0 has no alternative key and is excluded.
            lower_gain = torch.where(
                torch.isfinite(lower_gain),
                lower_gain,
                torch.zeros_like(lower_gain),
            )

            share = (
                lower_gain
                / span.clamp_min(1e-12)
            )

            # -------------------------------------------------
            # max-relative scores and Exact softmax
            # -------------------------------------------------

            z = (
                scores_for_max
                - row_max
            )

            w_exact = torch.exp(z)

            p_exact = (
                w_exact
                / w_exact.sum(
                    dim=-1,
                    keepdim=True,
                )
            )

            tail = (
                metric_valid
                & (z < -self.tau)
            )

            active = (
                metric_valid
                & (z >= -self.tau)
            )

            # -------------------------------------------------
            # fixed-grid coordinate
            # -------------------------------------------------

            zc = z.clamp(
                min=-self.tau,
                max=0.0,
            )

            rel = (
                zc + self.tau
            ) / self.h

            r_idx = (
                torch.floor(rel)
                .to(torch.long)
                .clamp(0, self.K - 1)
            )

            r = r_idx.to(
                dtype=zc.dtype
            )

            b_lo = (
                -self.tau
                + r * self.h
            )

            t = (
                (zc - b_lo)
                / self.h
            ).clamp(0.0, 1.0)

            dgrid = torch.minimum(
                t,
                1.0 - t,
            )

            choose_upper = (
                t >= 0.5
            )

            level_idx = (
                r_idx
                + choose_upper.to(torch.long)
            ).clamp(
                0,
                self.K,
            )

            # -------------------------------------------------
            # Accumulate
            # -------------------------------------------------

            rows = (
                B * H * (T - 1)
            )

            self.rows[li] += rows

            self.valid_keys[li] += float(
                metric_valid.sum()
            )

            self.raw_span_sum[li] += float(
                span[
                    :, :, row_select
                ].sum()
            )

            self.key0_min_count[li] += float(
                key0_is_min[
                    :, :, row_select
                ].sum()
            )

            self.key0_lower_gain_sum[li] += float(
                lower_gain[
                    :, :, row_select
                ].sum()
            )

            self.key0_range_share_sum[li] += float(
                share[
                    :, :, row_select
                ].sum()
            )

            self.tail_key_count[li] += float(
                tail.sum()
            )

            self.tail_mass_sum[li] += float(
                (
                    p_exact
                    * tail
                )
                .sum(dim=-1)[
                    :, :, row_select
                ]
                .sum()
            )

            self.active_key_count[li] += float(
                active.sum()
            )

            self.active_mass_sum[li] += float(
                (
                    p_exact
                    * active
                ).sum()
            )

            self.dgrid_sum[li] += float(
                dgrid[
                    active
                ].sum()
            )

            self.dgrid_mass_sum[li] += float(
                (
                    p_exact
                    * dgrid
                    * active
                ).sum()
            )

            self.edge05_count[li] += float(
                (
                    active
                    & (dgrid <= 0.05)
                ).sum()
            )

            self.edge10_count[li] += float(
                (
                    active
                    & (dgrid <= 0.10)
                ).sum()
            )

            self.edge20_count[li] += float(
                (
                    active
                    & (dgrid <= 0.20)
                ).sum()
            )

            active_levels = level_idx[
                active
            ]

            if active_levels.numel():
                lc = torch.bincount(
                    active_levels,
                    minlength=self.K + 1,
                )

                self.level_counts[
                    li
                ] += (
                    lc.double()
                    .cpu()
                    .numpy()
                )

            active_t = t[
                active
            ]

            if active_t.numel():
                hist = torch.histc(
                    active_t.float(),
                    bins=20,
                    min=0.0,
                    max=1.0,
                )

                self.t_hist[
                    li
                ] += (
                    hist.double()
                    .cpu()
                    .numpy()
                )

        return hook

    def summary(self):
        rows = self.rows.sum()
        valid = self.valid_keys.sum()
        active = self.active_key_count.sum()

        level_counts = (
            self.level_counts.sum(
                axis=0
            )
        )

        level_occ = (
            level_counts
            / level_counts.sum()
        )

        t_hist = self.t_hist.sum(
            axis=0
        )

        t_hist = (
            t_hist
            / t_hist.sum()
        )

        global_summary = {
            "raw_span_mean":
                float(
                    self.raw_span_sum.sum()
                    / rows
                ),

            "key0_argmin_rate":
                float(
                    self.key0_min_count.sum()
                    / rows
                ),

            "key0_lower_gain_mean":
                float(
                    self.key0_lower_gain_sum.sum()
                    / rows
                ),

            "key0_range_share_mean":
                float(
                    self.key0_range_share_sum.sum()
                    / rows
                ),

            "tail_key_fraction":
                float(
                    self.tail_key_count.sum()
                    / valid
                ),

            "exact_tail_mass":
                float(
                    self.tail_mass_sum.sum()
                    / rows
                ),

            "active_dgrid_mean":
                float(
                    self.dgrid_sum.sum()
                    / active
                ),

            "active_dgrid_mass_weighted":
                float(
                    self.dgrid_mass_sum.sum()
                    / self.active_mass_sum.sum()
                ),

            "frac_dgrid_le_005":
                float(
                    self.edge05_count.sum()
                    / active
                ),

            "frac_dgrid_le_010":
                float(
                    self.edge10_count.sum()
                    / active
                ),

            "frac_dgrid_le_020":
                float(
                    self.edge20_count.sum()
                    / active
                ),

            "level_occupancy":
                level_occ.tolist(),

            "t_hist_20":
                t_hist.tolist(),
        }

        layerwise = []

        for li in range(
            self.n_layer
        ):
            lc = self.level_counts[li]

            if lc.sum() > 0:
                locc = (
                    lc / lc.sum()
                ).tolist()
            else:
                locc = [
                    0.0
                    for _ in range(
                        self.K + 1
                    )
                ]

            layerwise.append({
                "layer": li,

                "raw_span_mean":
                    float(
                        self.raw_span_sum[li]
                        / self.rows[li]
                    ),

                "key0_argmin_rate":
                    float(
                        self.key0_min_count[li]
                        / self.rows[li]
                    ),

                "key0_lower_gain_mean":
                    float(
                        self.key0_lower_gain_sum[li]
                        / self.rows[li]
                    ),

                "key0_range_share_mean":
                    float(
                        self.key0_range_share_sum[li]
                        / self.rows[li]
                    ),

                "tail_key_fraction":
                    float(
                        self.tail_key_count[li]
                        / self.valid_keys[li]
                    ),

                "exact_tail_mass":
                    float(
                        self.tail_mass_sum[li]
                        / self.rows[li]
                    ),

                "active_dgrid_mean":
                    float(
                        self.dgrid_sum[li]
                        / self.active_key_count[li]
                    ),

                "active_dgrid_mass_weighted":
                    float(
                        self.dgrid_mass_sum[li]
                        / self.active_mass_sum[li]
                    ),

                "frac_dgrid_le_005":
                    float(
                        self.edge05_count[li]
                        / self.active_key_count[li]
                    ),

                "frac_dgrid_le_010":
                    float(
                        self.edge10_count[li]
                        / self.active_key_count[li]
                    ),

                "frac_dgrid_le_020":
                    float(
                        self.edge20_count[li]
                        / self.active_key_count[li]
                    ),

                "level_occupancy":
                    locc,
            })

        return global_summary, layerwise


@torch.no_grad()
def run_probe(
    model,
    dataset,
    device,
    dtype,
    K,
    tau,
):
    set_operator(
        model,
        mode="quant",
        K=K,
        approx="nearest",
        range_mode="relevant_zero",
        tau=tau,
    )

    probe = AdaptationProbe(
        model,
        K=K,
        tau=tau,
    )

    try:
        for bi in range(len(dataset)):
            x = dataset.x[
                bi:bi+1
            ].to(device)

            with autocast_ctx(
                device,
                dtype,
            ):
                model(x)

    finally:
        probe.close()

    return probe.summary()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--run-dir",
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
        default=16,
    )

    ap.add_argument(
        "--device",
        default="cuda",
    )

    ap.add_argument(
        "--dtype",
        choices=[
            "bf16",
            "fp32",
        ],
        default="bf16",
    )

    ap.add_argument(
        "--K",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--tau",
        type=float,
        default=6.0,
    )

    ap.add_argument(
        "--only-final",
        action="store_true",
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

    ckpts = sorted(
        glob.glob(
            os.path.join(
                args.run_dir,
                "ckpt_step*_*.pt",
            )
        ),
        key=token_m_from_name,
    )

    if not ckpts:
        raise RuntimeError(
            f"no saved checkpoints in {args.run_dir}"
        )

    if args.only_final:
        ckpts = [
            ckpts[-1]
        ]

    print(
        json.dumps(
            dataset.describe(),
            indent=2,
        )
    )

    print()
    print(
        "uniform-t references:"
    )
    print(
        "  E[d_grid]          = 0.250"
    )
    print(
        "  P(d_grid<=0.05)    = 0.100"
    )
    print(
        "  P(d_grid<=0.10)    = 0.200"
    )
    print(
        "  P(d_grid<=0.20)    = 0.400"
    )

    results = []

    for path in ckpts:
        print()
        print("=" * 110)
        print(path)
        print("=" * 110)

        model, cfg, ck = load_rr_model(
            path,
            args.device,
        )

        tokens = int(
            ck.get(
                "tokens_done",
                round(
                    token_m_from_name(path)
                    * 1e6
                ),
            )
        )

        g, layerwise = run_probe(
            model,
            dataset,
            args.device,
            args.dtype,
            args.K,
            args.tau,
        )

        row = {
            "checkpoint":
                path,
            "tokens":
                tokens,
            "tokens_m":
                tokens / 1e6,
            **g,
            "layerwise":
                layerwise,
        }

        results.append(row)

        print(
            f"tokens={tokens/1e6:7.1f}M"
        )

        print(
            f"raw_span          = "
            f"{g['raw_span_mean']:.4f}"
        )

        print(
            f"key0_argmin_rate  = "
            f"{100*g['key0_argmin_rate']:.3f}%"
        )

        print(
            f"key0_range_share  = "
            f"{100*g['key0_range_share_mean']:.3f}%"
        )

        print(
            f"key0_lower_gain   = "
            f"{g['key0_lower_gain_mean']:.4f}"
        )

        print(
            f"tail_key_fraction = "
            f"{100*g['tail_key_fraction']:.3f}%"
        )

        print(
            f"exact_tail_mass   = "
            f"{100*g['exact_tail_mass']:.4f}%"
        )

        print(
            f"mean d_grid       = "
            f"{g['active_dgrid_mean']:.5f}"
        )

        print(
            f"mass-wtd d_grid   = "
            f"{g['active_dgrid_mass_weighted']:.5f}"
        )

        print(
            f"P(d<=0.05)        = "
            f"{100*g['frac_dgrid_le_005']:.2f}%"
        )

        print(
            f"P(d<=0.10)        = "
            f"{100*g['frac_dgrid_le_010']:.2f}%"
        )

        print(
            f"P(d<=0.20)        = "
            f"{100*g['frac_dgrid_le_020']:.2f}%"
        )

        print(
            "level occupancy    = "
            + " ".join(
                f"L{i}:{100*x:.2f}%"
                for i, x in enumerate(
                    g[
                        "level_occupancy"
                    ]
                )
            )
        )

        del model

        if args.device.startswith(
            "cuda"
        ):
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Save full JSON
    # ---------------------------------------------------------

    json_path = os.path.join(
        args.out,
        "qrm_adaptation_trajectory.json",
    )

    with open(
        json_path,
        "w",
    ) as f:
        json.dump(
            {
                "K":
                    args.K,
                "tau":
                    args.tau,
                "protocol":
                    dataset.describe(),
                "uniform_reference": {
                    "mean_dgrid":
                        0.25,
                    "frac_le_005":
                        0.10,
                    "frac_le_010":
                        0.20,
                    "frac_le_020":
                        0.40,
                },
                "checkpoints":
                    results,
            },
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Compact CSV
    # ---------------------------------------------------------

    csv_path = os.path.join(
        args.out,
        "qrm_adaptation_trajectory.csv",
    )

    fields = [
        "tokens",
        "tokens_m",
        "raw_span_mean",
        "key0_argmin_rate",
        "key0_range_share_mean",
        "key0_lower_gain_mean",
        "tail_key_fraction",
        "exact_tail_mass",
        "active_dgrid_mean",
        "active_dgrid_mass_weighted",
        "frac_dgrid_le_005",
        "frac_dgrid_le_010",
        "frac_dgrid_le_020",
    ]

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        w.writeheader()

        for r in results:
            w.writerow({
                k: r[k]
                for k in fields
            })

    print()
    print("saved:", json_path)
    print("saved:", csv_path)
    print("DONE")


if __name__ == "__main__":
    main()
