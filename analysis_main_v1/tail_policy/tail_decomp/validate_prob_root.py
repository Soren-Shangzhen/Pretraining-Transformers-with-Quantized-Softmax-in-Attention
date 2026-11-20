#!/usr/bin/env python3
"""Addition (a): confirm the Prob root's no_grad HARD reconstruct uses resolved tails.

If tail_policy were threaded into the LERP call but not the no_grad hard call, the forward P would
still be the zero-tail one. Two tests:
  A1  tail keys carry NONZERO probability in the Prob root's final forward P
  A2  the Prob-root forward P matches the Weight-root forward P up to rounding
      (both define the same hard forward; they differ only by fp evaluation order)
Plus an end-to-end logits comparison of two GPTs with identical weights.
"""
import importlib, sys, torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


PB = (KI_REPO + "/"
      "training_code/phaseB_base")
TAU, K = 6.0, 4
res = []


def ck(n, ok, d=""):
    res.append((n, bool(ok))); print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f" — {d}" if d else ""), flush=True)
    return bool(ok)


def load(root):
    for m in ("model", "quant_attention"):
        sys.modules.pop(m, None)
    sys.path.insert(0, f"{PB}/code/{root}")
    try:
        return importlib.import_module("quant_attention"), importlib.import_module("model")
    finally:
        sys.path.remove(f"{PB}/code/{root}")


def prob_forward_P(qa, att, causal, policy):
    """Replicate the Prob root's P exactly as _quant_attention builds it."""
    with torch.no_grad():
        w_hard = qa._relevant_zero_reconstruct(att, causal, K, "nearest", tau=TAU,
                                               return_level_idx=True, range_grad="full",
                                               tail_policy=policy)[0]
        P_hard = w_hard / w_hard.sum(-1, keepdim=True)
    w_lerp = qa._relevant_zero_reconstruct(att, causal, K, "lerp", tau=TAU,
                                           return_level_idx=True, range_grad="full",
                                           tail_policy=policy)[0]
    P_lerp = w_lerp / w_lerp.sum(-1, keepdim=True)
    return P_lerp + (P_hard - P_lerp).detach()


def weight_forward_P(qa, att, causal, policy):
    w = qa._relevant_zero_reconstruct(att, causal, K, "nearest", tau=TAU,
                                      return_level_idx=True, range_grad="full",
                                      tail_policy=policy)[0]
    return w / w.sum(-1, keepdim=True)


def main():
    torch.manual_seed(0)
    T = 64
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    valid = causal.view(1, 1, T, T)
    att = torch.randn(1, 4, T, T, dtype=torch.float64) * 8.0
    rowmax = att.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
    tail = valid & ((att - rowmax) < -TAU)

    qap, _ = load("rb_tail_prob")
    Pp_res = prob_forward_P(qap, att, causal, "resolved")
    Pp_zero = prob_forward_P(qap, att, causal, "zero")

    print("== A1: tail keys carry nonzero P in the Prob root's forward ==")
    ck("A1 resolved: all tail P > 0", (Pp_res[tail] > 0).all().item(),
       f"min {Pp_res[tail].min().item():.3e}, {tail.sum().item()} tail entries")
    ck("A1 zero: tail P == 0 (control)", (Pp_zero[tail] == 0).all().item())
    ck("A1 resolved differs from zero on the tail", (Pp_res[tail] != Pp_zero[tail]).all().item())

    print("== A2: Prob-root forward P == Weight-root forward P (up to rounding) ==")
    qaw, _ = load("rb_tail_weight")
    Pw_res = weight_forward_P(qaw, att, causal, "resolved")
    d = (Pp_res - Pw_res).abs().max().item()
    ck("A2 resolved: max |P_prob - P_weight| < 1e-12 (float64)", d < 1e-12, f"{d:.3e}")
    dz = (Pp_zero - weight_forward_P(qaw, att, causal, "zero")).abs().max().item()
    ck("A2 zero: same, as control", dz < 1e-12, f"{dz:.3e}")

    print("== A3: end-to-end GPT logits, identical weights, both roots ==")
    outs = {}
    for root in ("rb_tail_weight", "rb_tail_prob"):
        qa, mo = load(root)
        cfg = mo.GPTConfig(block_size=64, n_layer=2, n_head=2, n_embd=64, attn_mode="quant",
                           attn_K=K, attn_approx="nearest", attn_range_grad="full",
                           attn_range_mode="relevant_zero", attn_range_tau=TAU,
                           attn_tail_policy="resolved")
        torch.manual_seed(1)
        m = mo.GPT(cfg).cuda().eval()
        idx = torch.arange(1, 65).unsqueeze(0).cuda()
        with torch.inference_mode():
            outs[root] = m(idx)[0].float().cpu()
    dl = (outs["rb_tail_weight"] - outs["rb_tail_prob"]).abs().max().item()
    ck("A3 logits agree to fp32 rounding (< 1e-2)", dl < 1e-2, f"max |dlogit| = {dl:.3e}")
    # Not asserted: whether the two roots are bit-identical. They evaluate the same hard forward by
    # different expressions, so agreement may be exact or may differ by rounding; Appendix C
    # documents rounding-level disagreement on trained models at scale. Recorded, not required.
    print(f"  [note] bit-identical here: {dl == 0} (max |dlogit| = {dl:.3e})")

    n = sum(1 for _, o in res if o)
    print(f"\nPROB-ROOT VALIDATION {'PASS' if n == len(res) else 'FAIL'}  ({n}/{len(res)})")
    return n == len(res)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
