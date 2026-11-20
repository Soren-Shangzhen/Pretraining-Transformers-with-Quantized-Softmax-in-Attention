"""P0-3 full-validation geometry probe for adaptive K-interval attention.

This script is designed to run next to ``model.py``, ``quant_attention.py``,
``posthoc_eval.py`` and ``manifest.json`` in the Phase-B training repository.

It implements the frozen v4 P0-3 protocol:

* deterministic full-coverage validation blocks (reuses posthoc_eval.py);
* trained-operator forward for each checkpoint in the manifest;
* score-range tail decomposition per causal query row:
      R_central = Q99(S) - Q01(S)
      T_upper   = max(S) - Q99(S)
      T_lower   = Q01(S) - min(S)
      R_minmax  = max(S) - min(S)
* span_no_key0 as a separate structural-sink diagnostic;
* full layer/head/query-position aggregation;
* actual attention sharpness: raw/normalized entropy, effective support,
  top-1 probability, and top-k mass;
* exact-softmax-vs-trained-operator reconstruction diagnostics:
      0.5 * Var_{P_exact}(delta), Var_j(delta), and exact KL;
* boundary-margin histograms under common Nearest K grids;
* interval occupancy and (for Nearest) K+1 level occupancy by layer;
* per-block/per-layer means saved to NPZ for later paired bootstrap analysis.

Important implementation choice
-------------------------------
The model itself is NOT patched.  A forward-pre-hook on every attention module
observes the layer-normalized input, recomputes Q/K with the same c_attn module,
and computes diagnostics in FP32.  The model then executes its original
attention forward normally, so later-layer representations follow the
checkpoint's TRAINING operator exactly.

Range-distribution quantiles are estimated from high-resolution streaming
histograms (default 16384 bins on [0,512]) while means/stds are accumulated
exactly.  If any value exceeds ``--range-hist-max``, the summary reports an
overflow count and the run should be repeated with a larger maximum before
using tail quantiles in the paper.

Typical use
-----------
Smoke test first::

    python geometry_probe.py \
      --manifest manifest.json \
      --out p0_results/geometry_smoke \
      --models exact,lerp_K4,nearest_K16 \
      --subset 3

Full P0-3::

    python geometry_probe.py \
      --manifest manifest.json \
      --out p0_results/geometry

The output directory contains:
    geometry_summary.json
    per_block_metrics.npz
    <model>/summary.json
    <model>/per_block_metrics.npz

No training is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from posthoc_eval import (
    FullCoverageEvalSet,
    load_manifest,
    load_model,
    make_autocast,
    set_operator,
    training_operator_of,
)
from quant_attention import _adaptive_interval_reconstruct


# -----------------------------------------------------------------------------
# position buckets
# -----------------------------------------------------------------------------


def make_position_buckets(T: int) -> Tuple[List[str], torch.Tensor]:
    """Power-of-two query-position buckets.

    q=0 is kept as its own bucket because its causal row is degenerate.  Primary
    global/layer/head summaries exclude q=0; position-bucket summaries retain it
    explicitly so that this convention is visible rather than hidden.
    """
    specs = [
        (0, 0),
        (1, 3),
        (4, 7),
        (8, 15),
        (16, 31),
        (32, 63),
        (64, 127),
        (128, 255),
        (256, 511),
        (512, 1023),
    ]
    names: List[str] = []
    ids = torch.full((T,), -1, dtype=torch.long)
    b = 0
    for lo, hi in specs:
        if lo >= T:
            continue
        hi2 = min(hi, T - 1)
        if lo == hi2:
            name = f"q{lo}"
        else:
            name = f"q{lo}_{hi2}"
        names.append(name)
        ids[lo : hi2 + 1] = b
        b += 1
    if (ids < 0).any():
        raise RuntimeError("position bucket construction left uncovered queries")
    return names, ids


# -----------------------------------------------------------------------------
# streaming histogram summaries for non-negative range statistics
# -----------------------------------------------------------------------------


QUANTILE_PS = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _hist_quantile(counts: np.ndarray, p: float, hist_max: float) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    if n == 0:
        return float("nan")
    cum = np.cumsum(counts)
    # Quantile of the represented sample.  Return bin center; error is at most
    # half a bin width when there is no overflow.
    target = p * (n - 1)
    idx = int(np.searchsorted(cum, target + 1, side="left"))
    idx = min(max(idx, 0), len(counts) - 1)
    width = hist_max / len(counts)
    return float((idx + 0.5) * width)


def _basic_from_hist(
    hist: np.ndarray,
    total_sum: float,
    total_sumsq: float,
    count: int,
    overflow: int,
    hist_max: float,
) -> dict:
    if count <= 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "overflow": int(overflow),
            **{f"p{int(round(100*p)):02d}": None for p in QUANTILE_PS},
        }
    mean = total_sum / count
    var = max(total_sumsq / count - mean * mean, 0.0)
    out = {
        "count": int(count),
        "mean": float(mean),
        "std": float(math.sqrt(var)),
        "overflow": int(overflow),
    }
    for p in QUANTILE_PS:
        out[f"p{int(round(100*p)):02d}"] = _hist_quantile(hist, p, hist_max)
    return out


class RangeHistogramAccumulator:
    """Exact means + high-resolution approximate quantiles for row ranges.

    Head/global summaries exclude q=0. Position-bucket summaries keep q=0 as a
    separate bucket, making the degenerate row explicit.
    """

    def __init__(
        self,
        metric_names: Sequence[str],
        n_layer: int,
        n_head: int,
        position_names: Sequence[str],
        position_ids_cpu: torch.Tensor,
        device: torch.device,
        hist_max: float = 512.0,
        hist_bins: int = 16384,
    ):
        self.metric_names = list(metric_names)
        self.metric_index = {m: i for i, m in enumerate(self.metric_names)}
        self.n_metric = len(self.metric_names)
        self.n_layer = n_layer
        self.n_head = n_head
        self.position_names = list(position_names)
        self.n_pos = len(position_names)
        self.hist_max = float(hist_max)
        self.hist_bins = int(hist_bins)
        self.device = device
        self.position_ids = position_ids_cpu.to(device)

        M, L, H, R = self.n_metric, n_layer, n_head, hist_bins
        # int64 histograms are exact.  With 5 metrics, 12x12 heads and 4096
        # bins this is modest (~24 MB plus position histograms).
        self.hist_head = torch.zeros(M, L, H, R, dtype=torch.int64, device=device)
        self.sum_head = torch.zeros(M, L, H, dtype=torch.float64, device=device)
        self.sumsq_head = torch.zeros_like(self.sum_head)
        self.count_head = torch.zeros(M, L, H, dtype=torch.int64, device=device)
        self.overflow_head = torch.zeros_like(self.count_head)

        self.hist_pos = torch.zeros(M, self.n_pos, R, dtype=torch.int64, device=device)
        self.sum_pos = torch.zeros(M, self.n_pos, dtype=torch.float64, device=device)
        self.sumsq_pos = torch.zeros_like(self.sum_pos)
        self.count_pos = torch.zeros(M, self.n_pos, dtype=torch.int64, device=device)
        self.overflow_pos = torch.zeros_like(self.count_pos)

        self._head_ids_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def _bin_index(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        overflow = values > self.hist_max
        # Range statistics are theoretically non-negative. Small negative roundoff
        # is clipped; materially negative values are flagged as overflow-like bads.
        bad_negative = values < -1e-5
        overflow = overflow | bad_negative
        v = values.clamp(0.0, self.hist_max)
        idx = torch.floor(v * (self.hist_bins / self.hist_max)).to(torch.long)
        idx.clamp_(0, self.hist_bins - 1)
        return idx, overflow

    @torch.no_grad()
    def update(self, layer: int, metric: str, values: torch.Tensor) -> None:
        """Update from values shaped (B,H,T). NaNs are ignored."""
        mi = self.metric_index[metric]
        B, H, T = values.shape
        if H != self.n_head:
            raise ValueError(f"head mismatch: got {H}, expected {self.n_head}")
        finite = torch.isfinite(values)
        idx, overflow = self._bin_index(torch.nan_to_num(values, nan=0.0))

        # -------- per-head (primary: exclude q=0) ---------------------------
        qmask = torch.arange(T, device=values.device).view(1, 1, T) > 0
        mask = finite & qmask
        v0 = torch.where(mask, values, torch.zeros_like(values)).double()
        self.sum_head[mi, layer] += v0.sum(dim=(0, 2))
        self.sumsq_head[mi, layer] += v0.square().sum(dim=(0, 2))
        self.count_head[mi, layer] += mask.sum(dim=(0, 2))
        self.overflow_head[mi, layer] += (overflow & mask).sum(dim=(0, 2))

        head_ids = torch.arange(H, device=values.device).view(1, H, 1).expand(B, H, T)
        if mask.any():
            combined = head_ids[mask] * self.hist_bins + idx[mask]
            hc = torch.bincount(combined, minlength=H * self.hist_bins)
            self.hist_head[mi, layer] += hc.view(H, self.hist_bins)

        # -------- query-position buckets (q=0 retained explicitly) ----------
        pos_ids = self.position_ids[:T].view(1, 1, T).expand(B, H, T)
        pmask = finite & (pos_ids >= 0)
        vp = torch.where(pmask, values, torch.zeros_like(values)).double()
        # scatter_add across all B/H/T to n_pos bins
        pflat = pos_ids[pmask]
        if pflat.numel():
            s = torch.zeros(self.n_pos, dtype=torch.float64, device=values.device)
            ss = torch.zeros_like(s)
            c = torch.zeros(self.n_pos, dtype=torch.int64, device=values.device)
            ov = torch.zeros_like(c)
            s.scatter_add_(0, pflat, vp[pmask])
            ss.scatter_add_(0, pflat, vp[pmask].square())
            c.scatter_add_(0, pflat, torch.ones_like(pflat, dtype=torch.int64))
            ov.scatter_add_(0, pflat, overflow[pmask].to(torch.int64))
            self.sum_pos[mi] += s
            self.sumsq_pos[mi] += ss
            self.count_pos[mi] += c
            self.overflow_pos[mi] += ov

            combined = pflat * self.hist_bins + idx[pmask]
            pc = torch.bincount(combined, minlength=self.n_pos * self.hist_bins)
            self.hist_pos[mi] += pc.view(self.n_pos, self.hist_bins)

    def summarize(self) -> dict:
        # One transfer at the end of a model, rather than moving row data every layer.
        hh = self.hist_head.cpu().numpy()
        sh = self.sum_head.cpu().numpy()
        ssh = self.sumsq_head.cpu().numpy()
        ch = self.count_head.cpu().numpy()
        oh = self.overflow_head.cpu().numpy()

        hp = self.hist_pos.cpu().numpy()
        sp = self.sum_pos.cpu().numpy()
        ssp = self.sumsq_pos.cpu().numpy()
        cp = self.count_pos.cpu().numpy()
        op = self.overflow_pos.cpu().numpy()

        out = {
            "histogram": {
                "min": 0.0,
                "max": self.hist_max,
                "bins": self.hist_bins,
                "bin_width": self.hist_max / self.hist_bins,
                "quantiles_are_histogram_estimates": True,
                "primary_head_layer_summaries_exclude_q0": True,
            },
            "metrics": {},
        }
        for mi, metric in enumerate(self.metric_names):
            global_hist = hh[mi].sum(axis=(0, 1))
            global_sum = float(sh[mi].sum())
            global_sumsq = float(ssh[mi].sum())
            global_count = int(ch[mi].sum())
            global_overflow = int(oh[mi].sum())

            per_layer = []
            for l in range(self.n_layer):
                per_layer.append(_basic_from_hist(
                    hh[mi, l].sum(axis=0),
                    float(sh[mi, l].sum()),
                    float(ssh[mi, l].sum()),
                    int(ch[mi, l].sum()),
                    int(oh[mi, l].sum()),
                    self.hist_max,
                ))

            per_head = []
            for l in range(self.n_layer):
                row = []
                for h in range(self.n_head):
                    row.append(_basic_from_hist(
                        hh[mi, l, h],
                        float(sh[mi, l, h]),
                        float(ssh[mi, l, h]),
                        int(ch[mi, l, h]),
                        int(oh[mi, l, h]),
                        self.hist_max,
                    ))
                per_head.append(row)

            per_pos = OrderedDict()
            for b, name in enumerate(self.position_names):
                per_pos[name] = _basic_from_hist(
                    hp[mi, b], float(sp[mi, b]), float(ssp[mi, b]),
                    int(cp[mi, b]), int(op[mi, b]), self.hist_max,
                )

            out["metrics"][metric] = {
                "global": _basic_from_hist(
                    global_hist, global_sum, global_sumsq,
                    global_count, global_overflow, self.hist_max,
                ),
                "per_layer": per_layer,
                "per_head": per_head,
                "per_position_bucket": per_pos,
            }
        return out


# -----------------------------------------------------------------------------
# exact-mean accumulator for row metrics that do not need distribution quantiles
# -----------------------------------------------------------------------------


class RowMeanAccumulator:
    def __init__(
        self,
        metric_names: Sequence[str],
        n_layer: int,
        n_head: int,
        position_names: Sequence[str],
        position_ids_cpu: torch.Tensor,
        device: torch.device,
    ):
        self.metric_names = list(metric_names)
        self.metric_index = {m: i for i, m in enumerate(self.metric_names)}
        self.n_metric = len(self.metric_names)
        self.n_layer = n_layer
        self.n_head = n_head
        self.position_names = list(position_names)
        self.n_pos = len(position_names)
        self.position_ids = position_ids_cpu.to(device)
        self.device = device

        M, L, H = self.n_metric, n_layer, n_head
        self.sum_head = torch.zeros(M, L, H, dtype=torch.float64, device=device)
        self.sumsq_head = torch.zeros_like(self.sum_head)
        self.count_head = torch.zeros(M, L, H, dtype=torch.int64, device=device)
        self.sum_pos = torch.zeros(M, self.n_pos, dtype=torch.float64, device=device)
        self.sumsq_pos = torch.zeros_like(self.sum_pos)
        self.count_pos = torch.zeros(M, self.n_pos, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, layer: int, metric: str, values: torch.Tensor) -> None:
        mi = self.metric_index[metric]
        B, H, T = values.shape
        finite = torch.isfinite(values)
        qmask = torch.arange(T, device=values.device).view(1, 1, T) > 0
        mask = finite & qmask
        v0 = torch.where(mask, values, torch.zeros_like(values)).double()
        self.sum_head[mi, layer] += v0.sum(dim=(0, 2))
        self.sumsq_head[mi, layer] += v0.square().sum(dim=(0, 2))
        self.count_head[mi, layer] += mask.sum(dim=(0, 2))

        pos_ids = self.position_ids[:T].view(1, 1, T).expand(B, H, T)
        pmask = finite & (pos_ids >= 0)
        if pmask.any():
            pflat = pos_ids[pmask]
            vals = values[pmask].double()
            s = torch.zeros(self.n_pos, dtype=torch.float64, device=values.device)
            ss = torch.zeros_like(s)
            c = torch.zeros(self.n_pos, dtype=torch.int64, device=values.device)
            s.scatter_add_(0, pflat, vals)
            ss.scatter_add_(0, pflat, vals.square())
            c.scatter_add_(0, pflat, torch.ones_like(pflat, dtype=torch.int64))
            self.sum_pos[mi] += s
            self.sumsq_pos[mi] += ss
            self.count_pos[mi] += c

    @staticmethod
    def _basic(s: float, ss: float, c: int) -> dict:
        if c <= 0:
            return {"count": 0, "mean": None, "std": None}
        mean = s / c
        var = max(ss / c - mean * mean, 0.0)
        return {"count": int(c), "mean": float(mean), "std": float(math.sqrt(var))}

    def summarize(self) -> dict:
        sh = self.sum_head.cpu().numpy()
        ssh = self.sumsq_head.cpu().numpy()
        ch = self.count_head.cpu().numpy()
        sp = self.sum_pos.cpu().numpy()
        ssp = self.sumsq_pos.cpu().numpy()
        cp = self.count_pos.cpu().numpy()
        out = {"primary_head_layer_summaries_exclude_q0": True, "metrics": {}}
        for mi, metric in enumerate(self.metric_names):
            per_layer = []
            for l in range(self.n_layer):
                per_layer.append(self._basic(
                    float(sh[mi, l].sum()), float(ssh[mi, l].sum()), int(ch[mi, l].sum())
                ))
            per_head = []
            for l in range(self.n_layer):
                row = []
                for h in range(self.n_head):
                    row.append(self._basic(
                        float(sh[mi, l, h]), float(ssh[mi, l, h]), int(ch[mi, l, h])
                    ))
                per_head.append(row)
            per_pos = OrderedDict()
            for b, name in enumerate(self.position_names):
                per_pos[name] = self._basic(float(sp[mi, b]), float(ssp[mi, b]), int(cp[mi, b]))
            out["metrics"][metric] = {
                "global": self._basic(float(sh[mi].sum()), float(ssh[mi].sum()), int(ch[mi].sum())),
                "per_layer": per_layer,
                "per_head": per_head,
                "per_position_bucket": per_pos,
            }
        return out


# -----------------------------------------------------------------------------
# boundary-margin and occupancy accumulators
# -----------------------------------------------------------------------------


class MarginAccumulator:
    def __init__(self, k_values: Sequence[int], n_layer: int, bins: int, device: torch.device):
        self.k_values = list(k_values)
        self.k_index = {k: i for i, k in enumerate(self.k_values)}
        self.n_layer = n_layer
        self.bins = int(bins)
        self.hist = torch.zeros(len(k_values), n_layer, bins, dtype=torch.int64, device=device)
        self.sum = torch.zeros(len(k_values), n_layer, dtype=torch.float64, device=device)
        self.sumsq = torch.zeros_like(self.sum)
        self.count = torch.zeros(len(k_values), n_layer, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, layer: int, K: int, margin_values: torch.Tensor) -> None:
        """margin_values is a 1-D vector on valid, non-degenerate score entries."""
        if margin_values.numel() == 0:
            return
        ki = self.k_index[K]
        m = margin_values.clamp(0.0, 0.5)
        idx = torch.floor(m * (self.bins / 0.5)).to(torch.long).clamp_(0, self.bins - 1)
        self.hist[ki, layer] += torch.bincount(idx, minlength=self.bins)
        md = m.double()
        self.sum[ki, layer] += md.sum()
        self.sumsq[ki, layer] += md.square().sum()
        self.count[ki, layer] += md.numel()

    def summarize(self) -> dict:
        hist = self.hist.cpu().numpy()
        s = self.sum.cpu().numpy()
        ss = self.sumsq.cpu().numpy()
        c = self.count.cpu().numpy()
        out = {"bins": self.bins, "range": [0.0, 0.5], "by_K": {}}
        for ki, K in enumerate(self.k_values):
            per_layer = []
            for l in range(self.n_layer):
                per_layer.append(_basic_from_hist(
                    hist[ki, l], float(s[ki, l]), float(ss[ki, l]), int(c[ki, l]), 0, 0.5
                ))
            out["by_K"][str(K)] = {
                "global": _basic_from_hist(
                    hist[ki].sum(axis=0), float(s[ki].sum()), float(ss[ki].sum()),
                    int(c[ki].sum()), 0, 0.5
                ),
                "per_layer": per_layer,
                "hist_counts_per_layer": hist[ki].tolist(),
            }
        return out


class OccupancyAccumulator:
    def __init__(self, n_layer: int, K: int, nearest: bool, device: torch.device):
        self.n_layer = n_layer
        self.K = K
        self.nearest = nearest
        self.interval = torch.zeros(n_layer, K, dtype=torch.int64, device=device)
        self.level = (
            torch.zeros(n_layer, K + 1, dtype=torch.int64, device=device) if nearest else None
        )

    @torch.no_grad()
    def update(self, layer: int, bins: torch.Tensor, level_idx: Optional[torch.Tensor]) -> None:
        vb = bins[bins >= 0]
        if vb.numel():
            self.interval[layer] += torch.bincount(vb, minlength=self.K)
        if self.nearest and level_idx is not None:
            vl = level_idx[level_idx >= 0]
            if vl.numel():
                self.level[layer] += torch.bincount(vl, minlength=self.K + 1)

    @staticmethod
    def _normalize(counts: np.ndarray) -> list:
        s = counts.sum()
        return (counts / s).tolist() if s else [0.0] * len(counts)

    def summarize(self) -> dict:
        ic = self.interval.cpu().numpy()
        out = {
            "interval_counts_per_layer": ic.tolist(),
            "interval_occupancy_per_layer": [self._normalize(x) for x in ic],
            "interval_occupancy_global": self._normalize(ic.sum(axis=0)),
        }
        if self.level is not None:
            lc = self.level.cpu().numpy()
            out.update({
                "level_counts_per_layer": lc.tolist(),
                "level_occupancy_per_layer": [self._normalize(x) for x in lc],
                "level_occupancy_global": self._normalize(lc.sum(axis=0)),
            })
        return out


# -----------------------------------------------------------------------------
# block recorder for later paired bootstrap
# -----------------------------------------------------------------------------


class BlockRecorder:
    def __init__(self, n_blocks: int, n_layer: int):
        self.n_blocks = n_blocks
        self.n_layer = n_layer
        self.arrays: Dict[str, np.ndarray] = {}

    def record(self, block: int, layer: int, metric: str, value: float) -> None:
        if metric not in self.arrays:
            self.arrays[metric] = np.full((self.n_blocks, self.n_layer), np.nan, dtype=np.float32)
        self.arrays[metric][block, layer] = np.float32(value)

    def save(self, path: str, prefix: Optional[str] = None) -> None:
        if prefix is None:
            payload = self.arrays
        else:
            payload = {f"{prefix}|{k}": v for k, v in self.arrays.items()}
        np.savez_compressed(path, **payload)


# -----------------------------------------------------------------------------
# geometry math
# -----------------------------------------------------------------------------


@torch.no_grad()
def row_quantiles_causal(att: torch.Tensor, ps: Sequence[float]) -> List[torch.Tensor]:
    """Exact linear-interpolation row quantiles over causal-valid keys.

    att: (B,H,T,T), raw scaled QK scores without -inf masking.
    Returns one tensor (B,H,T) per p.

    A single sort is shared by Q01/Q99.  Invalid future positions are filled with
    +inf, so the first L=q+1 sorted entries are exactly the valid causal scores.
    """
    B, H, T, _ = att.shape
    causal = torch.ones(T, T, dtype=torch.bool, device=att.device).tril()
    valid = causal.view(1, 1, T, T)
    sorted_vals = torch.sort(att.masked_fill(~valid, torch.inf), dim=-1).values
    L = torch.arange(1, T + 1, device=att.device, dtype=att.dtype)
    out = []
    for p in ps:
        rank = float(p) * (L - 1.0)
        lo = torch.floor(rank).to(torch.long)
        hi = torch.ceil(rank).to(torch.long)
        frac = (rank - lo.to(rank.dtype)).view(1, 1, T)
        ilo = lo.view(1, 1, T, 1).expand(B, H, T, 1)
        ihi = hi.view(1, 1, T, 1).expand(B, H, T, 1)
        vlo = torch.gather(sorted_vals, -1, ilo).squeeze(-1)
        vhi = torch.gather(sorted_vals, -1, ihi).squeeze(-1)
        out.append(vlo + frac * (vhi - vlo))
    return out


@torch.no_grad()
def exact_probabilities(att: torch.Tensor, causal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = causal.view(1, 1, att.size(-2), att.size(-1))
    masked = att.masked_fill(~valid, -torch.inf)
    row_max = masked.amax(dim=-1, keepdim=True)
    z = masked - row_max
    w = torch.exp(z)
    P = w / w.sum(dim=-1, keepdim=True)
    return z, P


@torch.no_grad()
def row_entropy_metrics(P: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return row metrics (B,H,T)."""
    B, H, T, _ = P.shape
    eps = torch.finfo(P.dtype).tiny
    logp = torch.where(P > 0, torch.log(P.clamp_min(eps)), torch.zeros_like(P))
    entropy_raw = -(P * logp).sum(dim=-1)
    lengths = torch.arange(1, T + 1, device=P.device, dtype=P.dtype).view(1, 1, T)
    logL = torch.log(lengths)
    entropy_norm = torch.where(lengths > 1, entropy_raw / logL.clamp_min(1e-12),
                               torch.zeros_like(entropy_raw))
    eff_support = torch.exp(entropy_raw)

    kmax = min(64, T)
    top = torch.topk(P, k=kmax, dim=-1, largest=True, sorted=True).values
    csum = top.cumsum(dim=-1)
    def mass(k: int) -> torch.Tensor:
        kk = min(k, kmax)
        return csum[..., kk - 1]

    return {
        "attn_entropy_raw": entropy_raw,
        "attn_entropy_norm": entropy_norm,
        "eff_support": eff_support,
        "top1_prob": mass(1),
        "topk_mass_4": mass(4),
        "topk_mass_16": mass(16),
        "topk_mass_64": mass(64),
    }


@torch.no_grad()
def reconstruction_error_metrics(
    z: torch.Tensor,
    P_exact: torch.Tensor,
    w_hat: torch.Tensor,
    causal: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Row-wise exact and leading-order approximation metrics."""
    B, H, T, _ = z.shape
    valid = causal.view(1, 1, T, T)
    # w_hat is positive on valid locations for the stable full-gradient runs.
    tiny = torch.finfo(w_hat.dtype).tiny
    logw = torch.log(w_hat.clamp_min(tiny))
    z_valid = torch.where(valid, z, torch.zeros_like(z))
    delta = torch.where(valid, logw - z_valid, torch.zeros_like(z))

    mu_p = (P_exact * delta).sum(dim=-1)
    var_p = (P_exact * delta.square()).sum(dim=-1) - mu_p.square()
    var_p.clamp_min_(0.0)

    lengths = torch.arange(1, T + 1, device=z.device, dtype=z.dtype).view(1, 1, T)
    n = lengths
    sum_d = delta.sum(dim=-1)
    sum_d2 = delta.square().sum(dim=-1)
    mean_d = sum_d / n
    var_j = sum_d2 / n - mean_d.square()
    var_j.clamp_min_(0.0)

    P_hat = w_hat / w_hat.sum(dim=-1, keepdim=True)
    # Exact KL, useful as a direct check on the leading-order 0.5 Var_P(delta).
    logP_exact = torch.where(P_exact > 0, torch.log(P_exact.clamp_min(tiny)),
                             torch.zeros_like(P_exact))
    logP_hat = torch.where(P_hat > 0, torch.log(P_hat.clamp_min(tiny)),
                           torch.zeros_like(P_hat))
    kl = (P_exact * (logP_exact - logP_hat)).sum(dim=-1)
    kl.clamp_min_(0.0)

    return {
        "half_varP_delta": 0.5 * var_p,
        "varj_delta": var_j,
        "kl_exact_to_operator": kl,
    }


# -----------------------------------------------------------------------------
# per-model analyzer
# -----------------------------------------------------------------------------


RANGE_METRICS = [
    "R_central",
    "T_upper",
    "T_lower",
    "R_minmax",
    "span_no_key0",
]

ROW_METRICS = [
    "score_center_minmax",
    "q_norm",
    "k_norm",
    "attn_entropy_raw",
    "attn_entropy_norm",
    "eff_support",
    "top1_prob",
    "topk_mass_4",
    "topk_mass_16",
    "topk_mass_64",
    "half_varP_delta",
    "varj_delta",
    "kl_exact_to_operator",
]


@dataclass
class ModelInfo:
    name: str
    ckpt_path: str
    attn_mode: str
    K: Optional[int]
    approx: Optional[str]


class GeometryAnalyzer:
    def __init__(
        self,
        model,
        info: ModelInfo,
        n_blocks: int,
        block_size: int,
        device: torch.device,
        range_hist_max: float,
        range_hist_bins: int,
        margin_ks: Sequence[int],
        margin_bins: int,
    ):
        self.model = model
        self.info = info
        self.device = device
        self.n_layer = len(model.transformer.h)
        self.n_head = model.cfg.n_head
        self.block_size = block_size
        self.position_names, position_ids = make_position_buckets(block_size)
        self.position_ids_cpu = position_ids

        self.range_acc = RangeHistogramAccumulator(
            RANGE_METRICS, self.n_layer, self.n_head,
            self.position_names, position_ids, device,
            hist_max=range_hist_max, hist_bins=range_hist_bins,
        )
        self.row_acc = RowMeanAccumulator(
            ROW_METRICS, self.n_layer, self.n_head,
            self.position_names, position_ids, device,
        )
        self.margin_acc = MarginAccumulator(margin_ks, self.n_layer, margin_bins, device)
        self.margin_ks = list(margin_ks)
        self.block_rec = BlockRecorder(n_blocks, self.n_layer)
        self.current_block = -1

        if info.attn_mode == "quant":
            assert info.K is not None and info.approx is not None
            self.occupancy = OccupancyAccumulator(
                self.n_layer, info.K, info.approx == "nearest", device
            )
        else:
            self.occupancy = None

        self.projection_norms = []
        for block in model.transformer.h:
            a = block.attn
            qw, kw, _ = a.c_attn.weight.split(a.n_embd, dim=0)
            self.projection_norms.append({
                "q_proj_weight_norm": float(torch.linalg.vector_norm(qw).detach().double().cpu()),
                "k_proj_weight_norm": float(torch.linalg.vector_norm(kw).detach().double().cpu()),
            })

        self.hooks = []
        for l, block in enumerate(model.transformer.h):
            self.hooks.append(block.attn.register_forward_pre_hook(self._make_hook(l)))

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _make_hook(self, layer: int):
        def hook(module, inputs):
            self.process_layer(layer, module, inputs[0])
        return hook

    @staticmethod
    def _block_mean(values: torch.Tensor) -> float:
        # Primary block means exclude q=0, matching global/layer/head summaries.
        if values.size(-1) <= 1:
            return float("nan")
        v = values[..., 1:]
        v = v[torch.isfinite(v)]
        return float(v.mean().double().cpu()) if v.numel() else float("nan")

    @torch.no_grad()
    def process_layer(self, layer: int, module, x: torch.Tensor) -> None:
        B, T, C = x.shape
        if T != self.block_size:
            # The code supports smaller smoke block sizes, but position IDs and
            # causal geometry are constructed from the actual T below.
            pass

        # Match the module's own QKV projection: this hook executes inside the
        # same autocast context as the forward that follows it.
        q, k, _v = module.c_attn(x).split(module.n_embd, dim=2)
        q = q.view(B, T, module.n_head, module.head_dim).transpose(1, 2)
        k = k.view(B, T, module.n_head, module.head_dim).transpose(1, 2)

        # Everything diagnostic from here is explicit FP32, matching the exact
        # and quantized attention paths in quant_attention.py.
        with torch.autocast(device_type=x.device.type, enabled=False):
            qf, kf = q.float(), k.float()
            att = (qf @ kf.transpose(-2, -1)) / math.sqrt(module.head_dim)
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            valid = causal.view(1, 1, T, T)

            row_max = att.masked_fill(~valid, -torch.inf).amax(dim=-1)
            row_min = att.masked_fill(~valid, torch.inf).amin(dim=-1)
            R_minmax = row_max - row_min
            center = 0.5 * (row_max + row_min)

            q01, q99 = row_quantiles_causal(att, (0.01, 0.99))
            R_central = q99 - q01
            T_upper = row_max - q99
            T_lower = q01 - row_min

            # Structural sink diagnostic: remove key position 0. q=0 then has no
            # valid key and is stored as NaN/ignored outside its explicit bucket.
            valid_no0 = valid.clone()
            valid_no0[..., 0] = False
            max_no0 = att.masked_fill(~valid_no0, -torch.inf).amax(dim=-1)
            min_no0 = att.masked_fill(~valid_no0, torch.inf).amin(dim=-1)
            span_no_key0 = max_no0 - min_no0
            qpos = torch.arange(T, device=x.device).view(1, 1, T)
            span_no_key0 = torch.where(
                qpos > 0, span_no_key0, torch.full_like(span_no_key0, float("nan"))
            )

            ranges = {
                "R_central": R_central,
                "T_upper": T_upper,
                "T_lower": T_lower,
                "R_minmax": R_minmax,
                "span_no_key0": span_no_key0,
            }
            for name, vals in ranges.items():
                self.range_acc.update(layer, name, vals)
                self.block_rec.record(self.current_block, layer, name, self._block_mean(vals))

            # Q/K norms and center are cheap but useful controls when interpreting
            # learned geometry.
            q_norm = torch.linalg.vector_norm(qf, dim=-1)
            k_norm = torch.linalg.vector_norm(kf, dim=-1)
            for name, vals in {
                "score_center_minmax": center,
                "q_norm": q_norm,
                "k_norm": k_norm,
            }.items():
                self.row_acc.update(layer, name, vals)
                self.block_rec.record(self.current_block, layer, name, self._block_mean(vals))

            # Exact reference distribution from this checkpoint's own score shape.
            z, P_exact = exact_probabilities(att, causal)

            # Actual trained-operator distribution used to propagate to later layers.
            bins = None
            level_idx = None
            if self.info.attn_mode == "quant":
                if self.info.approx == "nearest":
                    w_hat, bins, _row_span, level_idx = _adaptive_interval_reconstruct(
                        att, causal, int(self.info.K), self.info.approx,
                        return_level_idx=True, range_grad="full"
                    )
                else:
                    w_hat, bins, _row_span = _adaptive_interval_reconstruct(
                        att, causal, int(self.info.K), self.info.approx,
                        return_level_idx=False, range_grad="full"
                    )
                P_actual = w_hat / w_hat.sum(dim=-1, keepdim=True)
                err = reconstruction_error_metrics(z, P_exact, w_hat, causal)
            else:
                P_actual = P_exact
                zero = torch.zeros(B, module.n_head, T, dtype=att.dtype, device=att.device)
                err = {
                    "half_varP_delta": zero,
                    "varj_delta": zero,
                    "kl_exact_to_operator": zero,
                }

            sharp = row_entropy_metrics(P_actual)
            for name, vals in {**sharp, **err}.items():
                self.row_acc.update(layer, name, vals)
                self.block_rec.record(self.current_block, layer, name, self._block_mean(vals))

            if self.occupancy is not None and bins is not None:
                self.occupancy.update(layer, bins, level_idx)

            # Boundary margin under common Nearest grids.  It depends only on the
            # score geometry and K, not on whether the checkpoint trained with
            # LERP or Nearest.  Exclude q=0 / any degenerate row.
            span4 = R_minmax.unsqueeze(-1)
            row_min4 = row_min.unsqueeze(-1)
            for Kmargin in self.margin_ks:
                h = span4 / float(Kmargin)
                nondeg = span4 > 1e-12
                h_safe = torch.where(nondeg, h, torch.ones_like(h))
                rel = (att - row_min4) / h_safe
                frac = rel - torch.floor(rel)
                margin = (frac - 0.5).abs()
                # Exact upper boundary has frac=0 rather than t=1, but both give
                # margin=0.5.  Restrict to causal-valid, nondegenerate rows.
                full_valid = valid.expand(B, module.n_head, T, T)
                mmask = full_valid & nondeg.expand(B, module.n_head, T, T)
                mv = margin[mmask]
                self.margin_acc.update(layer, Kmargin, mv)

    def summarize(self) -> dict:
        rs = self.range_acc.summarize()
        row = self.row_acc.summarize()
        # Convenience derived h summaries from range means.  Histograms remain on
        # R; divide any desired R quantile by K downstream for a quant model.
        own_K = self.info.K if self.info.attn_mode == "quant" else None
        if own_K:
            for rname in ("R_central", "R_minmax"):
                g = rs["metrics"][rname]["global"]
                if g["mean"] is not None:
                    g[f"h_mean_at_K{own_K}"] = g["mean"] / own_K

        warnings = []
        for metric, body in rs["metrics"].items():
            ov = body["global"]["overflow"]
            if ov:
                warnings.append(
                    f"{metric}: {ov} primary rows exceeded histogram range; rerun with larger --range-hist-max before using quantiles"
                )

        return {
            "model": self.info.name,
            "checkpoint": self.info.ckpt_path,
            "trained_operator": {
                "attn_mode": self.info.attn_mode,
                "K": self.info.K,
                "approx": self.info.approx,
            },
            "position_buckets": self.position_names,
            "range_statistics": rs,
            "row_metrics": row,
            "boundary_margin": self.margin_acc.summarize(),
            "occupancy": self.occupancy.summarize() if self.occupancy else None,
            "projection_weight_norms_per_layer": self.projection_norms,
            "warnings": warnings,
        }


# -----------------------------------------------------------------------------
# model forward without LM head
# -----------------------------------------------------------------------------


@torch.no_grad()
def forward_backbone(model, idx: torch.Tensor) -> None:
    """Run exactly the transformer stack, skipping ln_f/lm_head we do not need."""
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    x = model.transformer.wte(idx) + model.transformer.wpe(pos)
    for block in model.transformer.h:
        x = block(x)
    # Deliberately no final LN or vocabulary projection.


def _safe_json_dump(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _sanitize_for_json(obj):
    """Replace NaN/inf recursively because allow_nan=False is intentional."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-dir", default="data/wikitext103")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None,
                    help="comma-separated manifest keys; default = all")
    ap.add_argument("--subset", type=int, default=None,
                    help="deterministic evenly-spaced validation-block subset for smoke tests")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--range-hist-max", type=float, default=512.0)
    ap.add_argument("--range-hist-bins", type=int, default=16384)
    ap.add_argument("--margin-ks", default="16,32",
                    help="common K grids for boundary-margin comparison")
    ap.add_argument("--margin-bins", type=int, default=100)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    device_type = "cuda" if device_str.startswith("cuda") else "cpu"
    autocast_ctx = make_autocast(device_type, args.dtype)

    manifest = load_manifest(args.manifest)
    if args.models:
        wanted = [x.strip() for x in args.models.split(",") if x.strip()]
        missing = [x for x in wanted if x not in manifest]
        if missing:
            raise KeyError(f"--models keys absent from manifest: {missing}")
        manifest = OrderedDict((k, manifest[k]) for k in wanted)

    evalset = FullCoverageEvalSet(
        os.path.join(args.data_dir, "val.bin"), args.block_size, subset=args.subset
    )
    margin_ks = parse_int_list(args.margin_ks)
    if not margin_ks:
        raise ValueError("--margin-ks must contain at least one K")

    os.makedirs(args.out, exist_ok=True)
    protocol = evalset.describe()
    protocol.update({
        "dtype": args.dtype,
        "device": device_str,
        "range_hist_max": args.range_hist_max,
        "range_hist_bins": args.range_hist_bins,
        "margin_ks": margin_ks,
        "margin_bins": args.margin_bins,
        "global_geometry_excludes_degenerate_q0": True,
        "position_bucket_summary_retains_q0": True,
    })
    print(json.dumps(protocol, indent=2))
    print("models:", list(manifest))
    print()

    combined_summary = {
        "protocol": protocol,
        "manifest": dict(manifest),
        "models": {},
    }
    combined_block_payload = {}

    for model_name, ckpt_path in manifest.items():
        print("=" * 80)
        print(f"MODEL {model_name}: {ckpt_path}")
        print("=" * 80)
        t_model = time.time()

        model, cfg = load_model(ckpt_path, device_str)
        attn_mode, K, approx = training_operator_of(cfg)
        if attn_mode == "quant":
            set_operator(model, "quant", K, approx)
            info = ModelInfo(model_name, ckpt_path, "quant", int(K), str(approx))
        else:
            set_operator(model, "exact_fp32_softmax", None, None)
            info = ModelInfo(model_name, ckpt_path, "exact_fp32_softmax", None, None)
        print("trained operator:", (info.attn_mode, info.K, info.approx))

        analyzer = GeometryAnalyzer(
            model, info, len(evalset), args.block_size, device,
            args.range_hist_max, args.range_hist_bins,
            margin_ks, args.margin_bins,
        )

        try:
            for bi in range(len(evalset)):
                analyzer.current_block = bi
                x = evalset.x[bi : bi + 1].to(device, non_blocking=True)
                with autocast_ctx:
                    forward_backbone(model, x)
                if (bi + 1) % args.progress_every == 0 or bi == 0 or bi + 1 == len(evalset):
                    elapsed = time.time() - t_model
                    rate = (bi + 1) / max(elapsed, 1e-9)
                    eta = (len(evalset) - bi - 1) / max(rate, 1e-9)
                    print(f"  block {bi+1:4d}/{len(evalset)}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
        finally:
            analyzer.close()

        summary = analyzer.summarize()
        model_out = os.path.join(args.out, model_name)
        os.makedirs(model_out, exist_ok=True)
        _safe_json_dump(_sanitize_for_json(summary), os.path.join(model_out, "summary.json"))
        analyzer.block_rec.save(os.path.join(model_out, "per_block_metrics.npz"))

        for metric, arr in analyzer.block_rec.arrays.items():
            combined_block_payload[f"{model_name}|{metric}"] = arr
        combined_summary["models"][model_name] = summary

        # Short terminal summary: the values most likely to catch a broken run.
        rg = summary["range_statistics"]["metrics"]
        rm = summary["row_metrics"]["metrics"]
        print(f"  R_central mean={rg['R_central']['global']['mean']:.6f}  "
              f"R_minmax mean={rg['R_minmax']['global']['mean']:.6f}")
        print(f"  entropy_norm mean={rm['attn_entropy_norm']['global']['mean']:.6f}  "
              f"eff_support mean={rm['eff_support']['global']['mean']:.3f}")
        if info.attn_mode == "quant":
            print(f"  0.5 VarP(delta)={rm['half_varP_delta']['global']['mean']:.6e}  "
                  f"KL(exact||op)={rm['kl_exact_to_operator']['global']['mean']:.6e}")
        if summary["warnings"]:
            print("  WARNINGS:")
            for w in summary["warnings"]:
                print("   -", w)
        print(f"  finished {model_name} in {(time.time()-t_model)/60:.1f} min\n")

        del analyzer, model
        if device_type == "cuda":
            torch.cuda.empty_cache()

    np.savez_compressed(os.path.join(args.out, "per_block_metrics.npz"), **combined_block_payload)
    _safe_json_dump(_sanitize_for_json(combined_summary), os.path.join(args.out, "geometry_summary.json"))
    print("saved:", os.path.join(args.out, "geometry_summary.json"))
    print("saved:", os.path.join(args.out, "per_block_metrics.npz"))
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
