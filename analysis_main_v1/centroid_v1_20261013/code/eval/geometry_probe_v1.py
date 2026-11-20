"""Layerwise attention-score geometry probe (analysis_main_v1; centroid_v1 copy: adds the centroid rules).

Hooks every attention module, recomputes fp32 scores exactly as the training operator does
(S = QK^T / sqrt(d_h) from the autocast c_attn output), and measures geometry plus the
native operator's distortion relative to exact softmax on the SAME scores.

Row set: causal query rows i >= 1 (row 0 has one key and zero span).
z = S - row_max  (z <= 0).

Per-layer metrics (all "mean" are over rows unless noted):
  row_max_mean, row_min_mean, center_mean, span_mean, span_p50, span_p90, span_p99
  q_norm_mean, k_norm_mean                 per-token per-head activation norms
  width_native_mean                        MinMax/LERP: span/K ; QRM: tau/K ; Exact: nan
  z_p0001, z_p001, z_p01, z_p05, z_p50     quantiles of valid z (pooled subsample of entries)
  frac_entries_below_tau6                  fraction of valid entries with z < -6
  exact_mass_below_tau6_mean               exact-softmax mass on keys with z < -6
  relevant_keys_tau6_mean                  number of keys with z >= -6
  exact_entropy_mean, exact_pmax_mean      exact-softmax row entropy (nats) / max probability
  tv_native_exact_mean                     0.5 * sum_j |P_native - P_exact|
  keys_top_bucket_mean                     keys mapped to the top reconstruction bucket
  exact_mass_top_bucket_mean               exact-softmax mass inside the top bucket
  eff_buckets_count                        exp(entropy) of bucket occupancy by entry count
  eff_buckets_mass                         exp(entropy) of bucket occupancy weighted by exact mass
  frac_entries_bottom_bucket               entries in bucket 0 (lowest level / interval)
Bucket = Nearest reconstruction level in [0,K] (tail entries of QRM excluded), or LERP interval in [0,K-1].
Exact runs have no buckets; those columns are nan.
"""
import math

import numpy as np
import torch

QUANTS = [0.0001, 0.001, 0.01, 0.05, 0.5]
Z_SUBSAMPLE_PER_CALL = 200_000


class LayerAcc:
    def __init__(self, n_buckets):
        self.sums = {}
        self.rows = 0
        self.tokens = 0
        self.entries = 0
        self.spans = []
        self.z = []
        self.hist_count = np.zeros(max(n_buckets, 1), dtype=np.float64)
        self.hist_mass = np.zeros(max(n_buckets, 1), dtype=np.float64)

    def add(self, key, value):
        self.sums[key] = self.sums.get(key, 0.0) + float(value)


def _entropy_eff(h):
    s = h.sum()
    if s <= 0:
        return float("nan")
    p = h[h > 0] / s
    return float(math.exp(-(p * np.log(p)).sum()))


class GeometryProbe:
    def __init__(self, model, qa_mod, op, n_layer, seed=0):
        self.model = model
        self.qa = qa_mod
        self.op = op
        self.quant = op["attn"] == "quant"
        K = op["K"]
        if not self.quant:
            self.n_buckets = 0
        elif op["approx"] == "nearest" or (op["approx"] in ("centroid", "centroid2")
                                           and op.get("cells", "nearest") == "nearest"):
            self.n_buckets = K + 1
        else:
            self.n_buckets = K
        self.acc = [LayerAcc(self.n_buckets) for _ in range(n_layer)]
        self.handles = []
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def __enter__(self):
        for li, blk in enumerate(self.model.transformer.h):
            self.handles.append(blk.attn.register_forward_pre_hook(self._make_hook(li)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, li):
        def hook(mod, inputs):
            self._measure(li, mod, inputs[0])
        return hook

    @torch.no_grad()
    def _measure(self, li, mod, x):
        acc = self.acc[li]
        B, T, C = x.shape
        H, D = mod.n_head, mod.head_dim
        q, k, _ = mod.c_attn(x).split(mod.n_embd, dim=2)   # runs under the caller's autocast, like training
        with torch.autocast(device_type="cuda", enabled=False):
            qf = q.float().view(B, T, H, D).transpose(1, 2)
            kf = k.float().view(B, T, H, D).transpose(1, 2)
            att = (qf @ kf.transpose(-2, -1)) / math.sqrt(D)
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            valid = causal.view(1, 1, T, T).expand(B, H, T, T)
            rowsel = torch.arange(T, device=x.device) >= 1          # rows i >= 1
            rowsel4 = rowsel.view(1, 1, T, 1)
            vrow = valid & rowsel4

            neg_inf = torch.finfo(att.dtype).min
            row_max = att.masked_fill(~valid, -torch.inf).amax(dim=-1)
            row_min = att.masked_fill(~valid, torch.inf).amin(dim=-1)
            span = (row_max - row_min)[..., 1:]
            center = (0.5 * (row_max + row_min))[..., 1:]
            n_rows = span.numel()
            acc.rows += n_rows
            acc.add("row_max", row_max[..., 1:].double().sum())
            acc.add("row_min", row_min[..., 1:].double().sum())
            acc.add("center", center.double().sum())
            acc.add("span", span.double().sum())
            acc.spans.append(span.flatten().cpu())

            qn = torch.linalg.vector_norm(qf, dim=-1)
            kn = torch.linalg.vector_norm(kf, dim=-1)
            acc.tokens += qn.numel()
            acc.add("q_norm", qn.double().sum())
            acc.add("k_norm", kn.double().sum())

            z = att - row_max.unsqueeze(-1)
            zv = z[vrow]
            acc.entries += zv.numel()
            idx = torch.randint(0, zv.numel(), (min(Z_SUBSAMPLE_PER_CALL, zv.numel()),), generator=self.gen)
            acc.z.append(zv[idx.to(zv.device)].cpu())
            acc.add("below_tau6_entries", (zv < -6.0).double().sum())
            del zv

            P_exact = torch.softmax(att.masked_fill(~valid, neg_inf), dim=-1).masked_fill(~valid, 0.0)
            rs = (slice(None), slice(None), slice(1, None))
            acc.add("exact_mass_below_tau6", (P_exact * (z < -6.0)).sum(-1)[rs].double().sum())
            acc.add("relevant_keys_tau6", ((z >= -6.0) & valid).sum(-1)[rs].double().sum())
            logp = torch.log(P_exact.clamp_min(1e-30))
            acc.add("exact_entropy", (-(P_exact * logp).sum(-1))[rs].double().sum())
            acc.add("exact_pmax", P_exact.amax(-1)[rs].double().sum())
            del logp

            if not self.quant:
                return

            K, tau = self.op["K"], self.op["range_tau"]
            if self.op["approx"] in ("centroid", "centroid2"):
                # centroid_v1: same calibration and cells as the existing operator, cell-mean representative
                w, bins, _, level, tail = self.qa._centroid_reconstruct(
                    att, causal, K, self.op["approx"], range_mode=self.op["range_mode"], tau=tau,
                    cells=self.op.get("cells", "nearest"), range_grad="full")
                acc.add("width_native", (tau / K * n_rows) if self.op["range_mode"] == "relevant_zero"
                        else float((span / K).double().sum()))
            elif self.op["range_mode"] == "relevant_zero":
                w, bins, _, level, tail = self.qa._relevant_zero_reconstruct(
                    att, causal, K, self.op["approx"], tau=tau, return_level_idx=True, range_grad="full")
                acc.add("width_native", tau / K * n_rows)
            elif self.op["range_mode"] == "window_tau_floor":
                # FW's grid exactly (W = tau on every row), tail floored rather than zeroed, so the
                # step is tau/K and there is no excluded tail.
                w, bins, _, level, _ = self.qa._window_tau_floor_reconstruct(
                    att, causal, K, self.op["approx"], tau=tau, return_level_idx=True, range_grad="full")
                tail = None
                acc.add("width_native", tau / K * n_rows)
            elif self.op["range_mode"] == "window_floor":
                # Span clipped at tau with a saturating tail: the step is min(span, tau)/K, and there is
                # no zero tail. Without this branch the mode would silently fall through to the MinMax
                # reconstruct below and report MinMax geometry for a different operator.
                w, bins, _, level, tail = self.qa._window_floor_reconstruct(
                    att, causal, K, self.op["approx"], tau=tau, return_level_idx=True, range_grad="full")
                tail = None
                acc.add("width_native", (span.clamp(max=tau) / K).double().sum())
            else:
                w, bins, _, level = self.qa._adaptive_interval_reconstruct(
                    att, causal, K, self.op["approx"], return_level_idx=True, range_grad="full")
                tail = None
                acc.add("width_native", (span / K).double().sum())
            P_nat = w / w.sum(dim=-1, keepdim=True)
            acc.add("tv_native_exact", (0.5 * (P_nat - P_exact).abs().sum(-1))[rs].double().sum())
            del w, P_nat

            bucket = level if (self.op["approx"] == "nearest" or (self.op["approx"] in ("centroid", "centroid2")
                                and self.op.get("cells", "nearest") == "nearest")) else bins
            top = self.n_buckets - 1
            inb = vrow & (bucket >= 0)
            if tail is not None:
                inb = inb & ~tail
            is_top = inb & (bucket == top)
            acc.add("keys_top_bucket", is_top.sum(-1)[rs].double().sum())
            acc.add("exact_mass_top_bucket", (P_exact * is_top).sum(-1)[rs].double().sum())
            acc.add("bottom_bucket_entries", (inb & (bucket == 0)).double().sum())
            acc.add("bucketed_entries", inb.double().sum())
            b = bucket[inb]
            acc.hist_count += torch.bincount(b, minlength=self.n_buckets).double().cpu().numpy()
            acc.hist_mass += torch.bincount(b, weights=P_exact[inb].double(),
                                            minlength=self.n_buckets).cpu().numpy()

    def results(self):
        out = []
        for li, a in enumerate(self.acc):
            r = {"layer": li}
            s = a.sums
            r["row_max_mean"] = s["row_max"] / a.rows
            r["row_min_mean"] = s["row_min"] / a.rows
            r["center_mean"] = s["center"] / a.rows
            r["span_mean"] = s["span"] / a.rows
            spans = torch.cat(a.spans).double()
            for qv, name in ((0.5, "span_p50"), (0.9, "span_p90"), (0.99, "span_p99")):
                r[name] = float(torch.quantile(spans, qv))
            r["q_norm_mean"] = s["q_norm"] / a.tokens
            r["k_norm_mean"] = s["k_norm"] / a.tokens
            r["width_native_mean"] = s["width_native"] / a.rows if "width_native" in s else float("nan")
            zs = torch.cat(a.z).double()
            for qv in QUANTS:
                r[f"z_p{str(qv).replace('0.', '')}"] = float(torch.quantile(zs, qv))
            r["frac_entries_below_tau6"] = s["below_tau6_entries"] / a.entries
            r["exact_mass_below_tau6_mean"] = s["exact_mass_below_tau6"] / a.rows
            r["relevant_keys_tau6_mean"] = s["relevant_keys_tau6"] / a.rows
            r["exact_entropy_mean"] = s["exact_entropy"] / a.rows
            r["exact_pmax_mean"] = s["exact_pmax"] / a.rows
            if self.quant:
                r["tv_native_exact_mean"] = s["tv_native_exact"] / a.rows
                r["keys_top_bucket_mean"] = s["keys_top_bucket"] / a.rows
                r["exact_mass_top_bucket_mean"] = s["exact_mass_top_bucket"] / a.rows
                r["eff_buckets_count"] = _entropy_eff(a.hist_count)
                r["eff_buckets_mass"] = _entropy_eff(a.hist_mass)
                r["frac_entries_bottom_bucket"] = s["bottom_bucket_entries"] / max(s["bucketed_entries"], 1)
                r["n_buckets"] = self.n_buckets
            else:
                for kname in ("tv_native_exact_mean", "keys_top_bucket_mean", "exact_mass_top_bucket_mean",
                              "eff_buckets_count", "eff_buckets_mass", "frac_entries_bottom_bucket"):
                    r[kname] = float("nan")
                r["n_buckets"] = 0
            r["rows"] = a.rows
            out.append(r)
        return out
