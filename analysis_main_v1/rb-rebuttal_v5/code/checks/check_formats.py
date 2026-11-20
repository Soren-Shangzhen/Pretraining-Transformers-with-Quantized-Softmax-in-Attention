#!/usr/bin/env python3
"""13.3 checks for the E5 formats (MXFP8-style, NVFP4-style, UINT8; FP8-E4M3 and BF16-exp kept).

1. Forward bit for bit equal to an independently computed quantised value, on CPU and GPU.
   The reference is written separately in numpy: E4M3 and E2M1 by enumerating every code and
   rounding to the nearest value with ties to the even code; the MX scale exponent by integer logic
   on frexp (no log2); same fp32 operation order as the formula of the plan.
2. Row gradient zero-sum residual (row maximum on the autograd path).
3. Number of nonzero levels in (0, 1] and underflow threshold in z, compared with the E0c estimates.
"""
import math, sys
import numpy as np
import torch
from rb_common import RB_ROOT, import_root, write_json, env_info

mm, qa = import_root(RB_ROOT)
RES = {"env": env_info(), "checks": []}
OK = True


def record(name, passed, **info):
    global OK
    OK &= bool(passed)
    RES["checks"].append(dict(name=name, passed=bool(passed), **info))
    print(f"{'PASS' if passed else 'FAIL'}  {name}  " + " ".join(f"{k}={v}" for k, v in info.items()), flush=True)


# ------------------------------------------------------------- independent numpy reference
def e4m3_table():
    vals, codes = [], []
    for c in range(128):                       # nonnegative codes
        e, m = (c >> 3) & 0xF, c & 0x7
        if e == 15 and m == 7:
            continue                           # NaN
        v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
        vals.append(v); codes.append(c)
    order = np.argsort(vals)
    return np.array(vals)[order], np.array(codes)[order]


E4M3_V, E4M3_C = e4m3_table()
E2M1_V = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]); E2M1_C = np.arange(8)


def round_table(x, vals, codes):
    """nearest representable value, ties to the even code, saturating at the largest value"""
    x = np.asarray(x, dtype=np.float64)
    x = np.minimum(x, vals[-1])
    i = np.clip(np.searchsorted(vals, x), 1, len(vals) - 1)
    lo, hi = vals[i - 1], vals[i]
    dlo, dhi = x - lo, hi - x
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (codes[i] % 2 == 0))
    out = np.where(pick_hi, hi, lo)
    out = np.where(x <= vals[0], vals[0], out)
    return out.astype(np.float32)


def ref_mxfp8(w):
    w = np.asarray(w, dtype=np.float32)
    T = w.shape[-1]
    wb = w.reshape(*w.shape[:-1], T // 32, 32)
    out = np.zeros_like(wb)
    it = np.ndindex(*wb.shape[:-1])
    for idx in it:
        blk = wb[idx]
        bmax = float(blk.max())
        if bmax <= 0:
            continue
        mant, E = math.frexp(bmax)              # bmax = mant * 2^E, mant in [0.5, 1)
        # smallest e with bmax <= 448 * 2^e = 7 * 2^(6+e):  mant * 2^E <= 7 * 2^(6+e)
        e = E - 9 if mant <= 0.875 else E - 8
        assert bmax <= 448.0 * 2.0 ** e and bmax > 448.0 * 2.0 ** (e - 1)
        e = max(-127, min(127, e))
        X = np.float32(2.0 ** e)
        out[idx] = round_table(blk / X, E4M3_V, E4M3_C) * X
    return out.reshape(w.shape)


def ref_nvfp4(w, valid):
    w = np.asarray(w, dtype=np.float32)
    T = w.shape[-1]
    tmax = np.float32(np.where(valid, w, 0).max())
    s_enc = np.float32(np.float32(6.0 * 448.0) / tmax)
    wb = w.reshape(*w.shape[:-1], T // 16, 16)
    bmax = wb.max(axis=-1, keepdims=True)
    S_b = round_table(np.float32(np.float32(bmax / np.float32(6.0)) * s_enc), E4M3_V, E4M3_C)
    ok = S_b > 0
    S_safe = np.where(ok, S_b, np.float32(1.0)).astype(np.float32)
    q = round_table(np.float32(np.float32(wb * s_enc) / S_safe), E2M1_V, E2M1_C)
    out = np.float32(np.float32(q * S_safe) / s_enc)
    return np.where(ok, out, np.float32(0)).astype(np.float32).reshape(w.shape)


def ref_uint8(w):
    w = np.asarray(w, dtype=np.float32)
    return (np.round(w * np.float32(255.0)) / np.float32(255.0)).astype(np.float32)


def ref_fp8(w):
    return round_table(np.asarray(w, dtype=np.float32), E4M3_V, E4M3_C)


REF = {"softmax_mxfp8": lambda w, v: ref_mxfp8(w), "softmax_nvfp4": ref_nvfp4,
       "softmax_uint8": lambda w, v: ref_uint8(w), "softmax_fp8e4m3exp": lambda w, v: ref_fp8(w)}
QFN = {"softmax_mxfp8": lambda w, v: qa._mxfp8_quantize(w), "softmax_nvfp4": qa._nvfp4_quantize,
       "softmax_uint8": lambda w, v: qa._uint8_quantize(w),
       "softmax_fp8e4m3exp": lambda w, v: w.to(torch.float8_e4m3fn).float()}


def rand_scores(B, H, T, scale, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(B, H, T, T, generator=g) * scale).to(device)


def forward_checks(device):
    T = 256
    causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    valid = causal.view(1, 1, T, T)
    for scale in (0.3, 2.0, 6.0, 20.0):
        att = rand_scores(1, 3, T, scale, device, int(scale * 10))
        z = att.masked_fill(~valid, -torch.inf)
        z = z - z.amax(-1, keepdim=True)
        w = torch.exp(z)
        for kind in ("softmax_fp8e4m3exp", "softmax_mxfp8", "softmax_nvfp4", "softmax_uint8"):
            wq = QFN[kind](w, valid)
            wref = torch.from_numpy(REF[kind](w.cpu().numpy(), valid.expand_as(w).cpu().numpy())).to(device)
            same_w = torch.equal(wq, wref)
            P = qa.softmax_family_probs(att, causal, kind)
            wr = wref.masked_fill(~valid, 0.0)
            Pref = wr / wr.sum(-1, keepdim=True)
            record(f"[{device}] {kind} scale{scale}: quantised weights == independent reference (bitwise)", same_w,
                   max_abs=f"{(wq - wref).abs().max().item():.1e}")
            record(f"[{device}] {kind} scale{scale}: training probabilities == reference normalisation (bitwise)",
                   torch.equal(P, Pref))
    # block edge cases: all-masked blocks, exact powers of two and ties
    w = torch.zeros(1, 1, 1, 64, device=device)
    w[..., 0] = 1.0
    w[..., 1] = 0.25 / 6.0                       # NVFP4 tie 0.25 -> 0
    w[..., 2] = 0.75 / 6.0                       # NVFP4 tie 0.75 -> 1.0
    w[..., 3] = 2.5 / 6.0
    w[..., 33] = 2.0 ** -20                      # MXFP8 block with a tiny max
    w[..., 34] = 2.0 ** -21
    v = torch.ones(1, 1, 1, 64, dtype=torch.bool, device=device)
    for kind in ("softmax_mxfp8", "softmax_nvfp4", "softmax_uint8", "softmax_fp8e4m3exp"):
        wq = QFN[kind](w, v)
        wref = torch.from_numpy(REF[kind](w.cpu().numpy(), v.cpu().numpy())).to(device)
        record(f"[{device}] {kind}: edge cases (ties, zero blocks, tiny block max) == reference (bitwise)",
               torch.equal(wq, wref), values=wq[0, 0, 0, :4].tolist() + wq[0, 0, 0, 33:35].tolist())


def zero_sum_checks(device):
    T = 128
    causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    for kind in qa.SOFTMAX_FAMILY:
        worst = 0.0
        for seed in range(5):
            att = rand_scores(1, 2, T, 3.0, device, 100 + seed).requires_grad_(True)
            P = qa.softmax_family_probs(att, causal, kind)
            up = torch.randn_like(P)
            (P * up).sum().backward()
            g = att.grad.masked_fill(~causal.view(1, 1, T, T), 0.0)
            resid = (g.sum(-1).abs() / g.abs().sum(-1).clamp_min(1e-30))[..., 1:]
            worst = max(worst, resid.max().item())
            if kind != "softmax_bf16chain":
                nonvalid = att.grad.masked_fill(causal.view(1, 1, T, T), 0.0).abs().max().item()
                worst = max(worst, nonvalid)
        # fp32 accumulation: |sum| / sum|.| at the 1e-6 level; bf16 casts in the graph: 1e-2 level
        tol = 5e-2 if kind in ("softmax_bf16chain", "softmax_bf16exp") else 1e-5
        record(f"[{device}] {kind}: row-gradient zero-sum residual |sum g| / sum |g|", worst < tol,
               worst=f"{worst:.2e}", tol=tol)


def level_checks():
    """Levels in (0,1] and underflow threshold for a row whose maximum sits in the same block."""
    out = {}
    z = torch.linspace(-20.0, 0.0, 2_000_001, dtype=torch.float64).float()
    for kind in ("softmax_fp8e4m3exp", "softmax_mxfp8", "softmax_nvfp4", "softmax_uint8", "softmax_bf16exp"):
        w = torch.exp(z)
        if kind == "softmax_bf16exp":
            wq = torch.exp(z.to(torch.bfloat16)).float()
        else:
            # one block per value: [w, 1, 0, 0, ...] so that the block maximum is 1 (= the row maximum)
            blk = 32 if kind == "softmax_mxfp8" else 16
            W = torch.zeros(z.numel(), blk)
            W[:, 0] = w
            W[:, 1] = 1.0
            wq = QFN[kind](W.view(1, 1, z.numel(), blk), torch.ones(1, 1, z.numel(), blk, dtype=torch.bool))[0, 0, :, 0]
        nz = wq > 0
        levels = torch.unique(wq[nz]).numel()
        zu = z[~nz].max().item() if (~nz).any() else float("-inf")
        out[kind] = dict(nonzero_levels_in_0_1=levels, underflow_below_z=round(zu, 4))
    exp = {"softmax_fp8e4m3exp": (56, -6.93), "softmax_mxfp8": (120, -12.48), "softmax_nvfp4": (7, -3.18),
           "softmax_uint8": (255, -6.23)}
    for kind, (lev, zu) in exp.items():
        got = out[kind]
        record(f"E0c {kind}: levels and underflow threshold match the estimate", got["nonzero_levels_in_0_1"] == lev
               and abs(got["underflow_below_z"] - zu) < 0.02, got=got, expected=dict(levels=lev, underflow_z=zu))
    RES["E0c_levels"] = out
    print("E0c table:", out)


def exp2_exact():
    for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        e = torch.arange(-150, 151, dtype=torch.float64, device=dev)
        got = torch.exp2(e).cpu()
        exact = torch.tensor([math.ldexp(1.0, int(i)) for i in range(-150, 151)], dtype=torch.float64)
        record(f"[{dev}] torch.exp2 returns exact powers of two for integer exponents -150..150",
               torch.equal(got, exact))


def main():
    exp2_exact()
    forward_checks("cpu")
    if torch.cuda.is_available():
        forward_checks("cuda")
    zero_sum_checks("cpu")
    if torch.cuda.is_available():
        zero_sum_checks("cuda")
    level_checks()
    RES["all_pass"] = OK
    print("written", write_json(f"13_3_formats_torch{torch.__version__.split('+')[0]}.json", RES))
    print("ALL PASS" if OK else "SOME FAILED")
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
