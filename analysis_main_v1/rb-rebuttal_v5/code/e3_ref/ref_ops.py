"""Reference forward passes for the two external baselines (E3a, E3b).

No file of the IntAttention release is included.
IndexSoftmax: our implementation follows the PyTorch implementation released with IntAttention
(Zhong et al., 2026).
EXAQ: the original EXAQ source code is no longer publicly available; our implementation follows the
EXAQ paper (Shkolnik et al., 2024) and the reimplementation released in the IntAttention artifact
repository.
Forward only. The backward rule (strict-forward STE around these values)
belongs in the training code root:

    P = sg(P_hard) + (P_surrogate - sg(P_surrogate))

All functions take fp32 scores of shape (..., S) and a boolean mask of the
same shape (True = valid key).
"""
import math

import torch

# ---------------------------------------------------------------- IndexSoftmax

IDX_THR = 6.6       # clipping threshold in nats
IDX_BINS = 31       # 5-bit index, 32 table entries

# round(255 * exp(-i * 6.6 / 31)), first entry 255, last entry 0
IDX_TABLE = (255, 206, 167, 135, 109, 88, 71, 57, 46, 38, 30, 25, 20, 16, 13,
             10, 8, 7, 6, 4, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0)


def index_softmax_table_from_formula(device=None):
    """Regenerates the table from its formula; used only to check IDX_TABLE."""
    i = torch.arange(IDX_BINS + 1, device=device)
    t = (torch.exp(i * -IDX_THR / IDX_BINS) * 255).round().to(torch.int32)
    t[0] = 255
    t[-1] = 0
    return t


def index_softmax_lookup(scores, mask, thr=IDX_THR):
    """Table values e_j in {0..255} as fp32; masked keys get 0.

    The index is trunc((d * 31) / thr) with d = min(rowmax - s, thr).
    Keep this operation order: (d * 31) / thr, not d * (31 / thr).
    """
    s = scores.masked_fill(~mask, float("-inf"))
    d = s.amax(dim=-1, keepdim=True) - s
    d = torch.clamp(d, max=thr)
    idx = ((d * IDX_BINS) / thr).to(torch.int64)          # truncation
    table = torch.tensor(IDX_TABLE, dtype=torch.float32, device=scores.device)
    return table[idx].masked_fill(~mask, 0.0)


def index_softmax_fp(scores, mask, thr=IDX_THR):
    """E3a-i: lookup + floating-point normalisation."""
    e = index_softmax_lookup(scores, mask, thr)
    return e / e.sum(dim=-1, keepdim=True)


def index_softmax_u8_codes(scores, mask, thr=IDX_THR):
    """Integer codes round(255 * e / sum(e)) in {0..255}, as fp32."""
    e = index_softmax_lookup(scores, mask, thr)
    return ((e * 255.0) / e.sum(dim=-1, keepdim=True)).round()


def index_softmax_u8(scores, mask, thr=IDX_THR):
    """E3a-ii: lookup + UINT8 probabilities, P = code / 255 (rows need not sum to 1)."""
    return index_softmax_u8_codes(scores, mask, thr) / 255.0


# ------------------------------------------------------------------------ EXAQ

# linear rule C* = a * sigma + b for the clipping point, per bit width
EXAQ_COEF = {2: (-1.66, -1.85), 3: (-1.75, -2.06), 4: (-1.83, -2.27)}


def exaq_window(scores, mask, bits=3):
    """Returns (x, lo, delta): shifted scores and the tensor-level window.

    sigma (population std) and the minimum are taken over every valid entry of
    the whole tensor and recomputed on every call. lo and delta are Python
    floats, as in the reproduction this is aligned to.
    """
    s = scores.masked_fill(~mask, float("-inf"))
    x = s - s.amax(dim=-1, keepdim=True)
    xv = x[mask]
    sigma = xv.float().std(unbiased=False).item()
    a, b = EXAQ_COEF[bits]
    c = a * sigma + b
    if not math.isfinite(c) or c >= 0.0:
        c = -1e-6
    lo = max(c, xv.min().item())
    delta = (0.0 - lo) / float(2 ** bits - 1)
    return x, lo, delta


@torch.no_grad()
def exaq_softmax(scores, mask, bits=3):
    """E3b hard forward: clamp to [lo, 0], round to 2**bits levels, exact softmax."""
    x, lo, delta = exaq_window(scores, mask, bits)
    x = x.clone()
    xv = x[mask].clamp_(min=lo, max=0.0)
    if math.isfinite(delta) and delta > 0.0:
        q = torch.round((xv - lo) / delta).clamp_(0, 2 ** bits - 1)
        xv = lo + q * delta
    x[mask] = xv
    return torch.softmax(x, dim=-1)
