#!/usr/bin/env python3
"""Degenerate-row audit of the frozen operator implementations (paper Appendix A).

Imports `_adaptive_interval_reconstruct` (MinMax path) and `_relevant_zero_reconstruct` (fixed-window
path) from every frozen quant_attention.py used by the formal suites and the legacy detach record, builds
the surrogate forwards exactly as the training code does (Weight-STE: P = w / sum w with w = w_lerp +
sg(w_hard - w_lerp); Prob-STE: P = P_lerp + sg(P_hard - P_lerp); LERP: P = w_lerp / sum), and checks on
four edge cases: (1) a single valid key, (2) several identical valid scores, (3) tied maxima and minima with
a positive span, (4) a very small positive span (1e-6, above the code's eps=1e-12). For each: forward P
finite, row sums = 1, loss finite, dL/ds finite. Rows are the causal rows of one (B=1,H=1,T) score matrix.

Run:  python paper/check_degenerate_rows.py
"""
import importlib.util
import sys
from pathlib import Path

import torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


TRAIN = Path(KI_REPO)
ROOTS = {
    "9d03d7a6 base (Exact/LERP/MinMax-Weight)": TRAIN / "training_code/phaseB_base/quant_attention.py",
    "a402460d FW-Weight": TRAIN / "training_code/phaseB_base/code/phaseB_rr/quant_attention.py",
    "0e20aea0 MinMax-Prob": TRAIN / "analysis_main_v1/D_robustness_124m_100m_5seed/code_snapshots/formal_code_probste_5seeds_100m_v1/formal_nearest_probste_v1/quant_attention.py",
    "ce2bd02b FW-Prob": TRAIN / "analysis_main_v1/D_robustness_124m_100m_5seed/code_snapshots/formal_code_probste_5seeds_100m_v1/formal_qrm_probste_v1/quant_attention.py",
    "3da199f8 legacy detach": TRAIN / "training_code/legacy_exact_fp32/quant_attention.py",
}


def load(path):
    spec = importlib.util.spec_from_file_location("qa_" + path.parent.name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def cases(dtype):
    T = 4
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    base = torch.tensor([[0.0, -9.0, -9.0, -9.0], [2.0, -1.0, -9.0, -9.0], [1.0, 1.0, 1.0, -9.0], [3.0, 0.5, -2.0, 3.0]], dtype=dtype)
    # row 0: single valid key; row 2: three identical valid scores; row 3: tied maxima (3.0, 3.0) with positive span
    tiny = base.clone(); tiny[1] = torch.tensor([1.0, 1.0 + 1e-6, -9.0, -9.0], dtype=dtype)  # row 1: span 1e-6
    return causal, {"single-key / equal-scores / tied-max": base, "tiny span 1e-6 (row 1)": tiny}


def forwards(mod, att, causal, K, family, kw_full):
    """Return P for one family using the frozen functions, with autograd on att."""
    rz = getattr(mod, "_relevant_zero_reconstruct", None)
    ar = mod._adaptive_interval_reconstruct
    if family == "LERP":
        w = ar(att, causal, K, "lerp", **kw_full)[0]; return w / w.sum(-1, keepdim=True)
    if family == "MinMax-Weight":
        w = ar(att, causal, K, "nearest", **kw_full)[0]; return w / w.sum(-1, keepdim=True)
    if family == "MinMax-Prob":
        with torch.no_grad():
            wh = ar(att, causal, K, "nearest", **kw_full)[0]; Ph = wh / wh.sum(-1, keepdim=True)
        wl = ar(att, causal, K, "lerp", **kw_full)[0]; Pl = wl / wl.sum(-1, keepdim=True)
        return Pl + (Ph - Pl).detach()
    if family == "FW-Weight":
        w = rz(att, causal, K, "nearest", range_grad="full")[0]; return w / w.sum(-1, keepdim=True)
    if family == "FW-Prob":
        with torch.no_grad():
            wh = rz(att, causal, K, "nearest", range_grad="full")[0]; Ph = wh / wh.sum(-1, keepdim=True)
        wl = rz(att, causal, K, "lerp", range_grad="full")[0]; Pl = wl / wl.sum(-1, keepdim=True)
        return Pl + (Ph - Pl).detach()
    raise ValueError(family)


def main():
    torch.manual_seed(0)
    bad = 0
    for name, path in ROOTS.items():
        mod = load(path)
        has_fw = hasattr(mod, "_relevant_zero_reconstruct")
        has_grad_flag = "range_grad" in mod._adaptive_interval_reconstruct.__code__.co_varnames
        kw_full = {"range_grad": "full"} if has_grad_flag else {}
        fams = ["LERP", "MinMax-Weight", "MinMax-Prob"] + (["FW-Weight", "FW-Prob"] if has_fw else [])
        print(f"== {name}  ({path.relative_to(TRAIN)})  calibration grad: {'full' if has_grad_flag else 'detach (legacy, hard-wired)'}")
        for dtype in (torch.float32, torch.float64):
            causal, cs = cases(dtype)
            for cname, base in cs.items():
                for fam in fams:
                    for K in (4, 16):
                        att = base.clone().view(1, 1, 4, 4).requires_grad_(True)
                        P = forwards(mod, att, causal, K, fam, kw_full)
                        g = torch.randn_like(P)
                        loss = (P * g).sum()
                        loss.backward()
                        rs = P.sum(-1).view(-1)
                        ok = bool(torch.isfinite(P).all() and torch.isfinite(loss) and torch.isfinite(att.grad).all()
                                  and torch.allclose(rs, torch.ones_like(rs), atol=1e-6))
                        bad += not ok
                        flag = "OK " if ok else "BAD"
                        r0 = P[0, 0, 0, :1].tolist(); r2 = P[0, 0, 2, :3].tolist()
                        print(f"  {flag} {str(dtype)[6:]:8s} K={K:<3d} {fam:14s} {cname:34s} P[row0]={r0[0]:.3f} P[row2]={[round(x,3) for x in r2]} "
                              f"rowsum_max_err={float((rs-1).abs().max()):.1e} |dL/ds|max={float(att.grad.abs().max()):.3g}")
    print("RESULT:", "PASS" if bad == 0 else f"FAIL ({bad} cases)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
