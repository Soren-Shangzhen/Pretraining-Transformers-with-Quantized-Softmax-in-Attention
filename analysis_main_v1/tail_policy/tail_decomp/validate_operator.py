#!/usr/bin/env python3
"""Pre-training validation for tail_policy=resolved (checks 2, 2b, 2b-prob, 3, 4, 5, 6).

Check 1 (243-block evaluation parity against the frozen roots) is a separate script because it
needs real checkpoints: validate_eval_parity.py.
"""
import importlib, itertools, math, sys
import torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


PB = (KI_REPO + "/"
      "training_code/phaseB_base")
TAU, K = 6.0, 4
H = TAU / K
results = []


def ck(name, ok, detail=""):
    results.append((name, bool(ok), str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return bool(ok)


def load(root):
    for m in ("model", "quant_attention"):
        sys.modules.pop(m, None)
    sys.path.insert(0, f"{PB}/code/{root}")
    try:
        qa = importlib.import_module("quant_attention")
        mo = importlib.import_module("model")
    finally:
        sys.path.remove(f"{PB}/code/{root}")
    return qa, mo


def rec(qa, att, causal, policy, approx="nearest", grad="full"):
    return qa._relevant_zero_reconstruct(att, causal, K, approx, tau=TAU,
                                         return_level_idx=True, range_grad=grad,
                                         tail_policy=policy)


def main():
    qa, mo = load("rb_tail_weight")
    torch.manual_seed(0)
    T = 96
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    valid = causal.view(1, 1, T, T)
    att = torch.randn(1, 4, T, T, dtype=torch.float64) * 8.0
    att[:, :2] *= 0.15                                  # low-dispersion heads -> no-tail rows
    rowmax = att.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
    z = att - rowmax
    tail = valid & (z < -TAU)
    inw = valid & (z >= -TAU)

    print("== check 2: in-window weights bit-identical (resolved vs zero) ==")
    for approx in ("nearest", "lerp"):
        wz = rec(qa, att, causal, "zero", approx)[0]
        wr = rec(qa, att, causal, "resolved", approx)[0]
        ck(f"check2 {approx}: in-window w bit-identical",
           torch.equal(wz[inw], wr[inw]), f"{inw.sum().item()} entries")
        ck(f"check2 {approx}: tail differs", (wr[tail] != wz[tail]).all().item())

    print("== check 2b: tail uses raw z, not clamped zc ==")
    a2 = torch.full((1, 1, 2, 2), 0.0, dtype=torch.float64)
    a2[0, 0, 1, 0] = -20.0; a2[0, 0, 1, 1] = 0.0        # row 1: one key at z=-20, max at 0
    c2 = torch.tril(torch.ones(2, 2, dtype=torch.bool))
    w2 = rec(qa, a2, c2, "resolved", "nearest")[0]
    got, want, floorv = w2[0, 0, 1, 0].item(), math.exp(-19.5), math.exp(-TAU)
    ck("check2b z=-20 -> w^N = e^-19.5 (not e^-6)", abs(got - want) < 1e-12 * max(1, want),
       f"got {got:.6e}  e^-19.5={want:.6e}  e^-6={floorv:.6e}")
    # sweep: w must track z, not sit at the floor
    zs = [-8.0, -12.0, -20.0, -30.0]
    ws = []
    for zv in zs:
        a3 = torch.zeros((1, 1, 2, 2), dtype=torch.float64); a3[0, 0, 1, 0] = zv
        ws.append(rec(qa, a3, c2, "resolved", "nearest")[0][0, 0, 1, 0].item())
    ck("check2b w tracks z over a sweep (strictly decreasing, none at e^-6)",
       all(ws[i] > ws[i + 1] for i in range(len(ws) - 1)) and all(abs(v - floorv) > 1e-12 for v in ws),
       " ".join(f"z={a:g}:{b:.3e}" for a, b in zip(zs, ws)))
    # and confirm `floor` really does sit at e^-tau, i.e. the two policies are distinguishable
    wf = rec(qa, a2, c2, "floor", "nearest")[0][0, 0, 1, 0].item()
    ck("check2b floor policy gives exactly e^-6", abs(wf - floorv) < 1e-15, f"{wf:.6e}")

    print("== check 3: closed-form gradient vs autograd (float64) ==")
    for approx, tag in (("nearest", "Weight-STE"),):
        a = att.clone().requires_grad_(True)
        w = rec(qa, a, causal, "resolved", approx)[0]
        P = w / w.sum(-1, keepdim=True)
        g = torch.randn_like(P)
        (P * g).sum().backward()
        ck(f"check3 {tag}: grads finite", torch.isfinite(a.grad).all().item())
        rs = a.grad.masked_fill(~valid, 0.0).sum(-1).abs().max().item()
        ck(f"check3 {tag}: per-row gradient sum ~ 0 (zero-sum identity)", rs < 1e-9, f"max {rs:.3e}")

    print("== check 4: degenerate rows ==")
    cases = {"single key": torch.zeros(1, 1, 1, 1, dtype=torch.float64),
             "all equal": torch.full((1, 1, 8, 8), 3.0, dtype=torch.float64),
             "tied maxima": None, "span 1e-6": None, "span 1e3": None}
    t8 = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    x = torch.zeros(1, 1, 8, 8, dtype=torch.float64); x[..., 0] = x[..., 1] = 5.0
    cases["tied maxima"] = x
    y = torch.zeros(1, 1, 8, 8, dtype=torch.float64); y[..., 0] = 1e-6
    cases["span 1e-6"] = y
    zz = torch.zeros(1, 1, 8, 8, dtype=torch.float64); zz[..., 0] = -1e3
    cases["span 1e3"] = zz
    for name, a in cases.items():
        cz = torch.tril(torch.ones(a.shape[-1], a.shape[-1], dtype=torch.bool))
        aa = a.clone().requires_grad_(True)
        w = rec(qa, aa, cz, "resolved", "nearest")[0]
        P = w / w.sum(-1, keepdim=True)
        P.sum().backward()
        v = cz.view(1, 1, *cz.shape)
        ck(f"check4 {name}: finite fwd/bwd, row sums 1",
           torch.isfinite(w).all().item() and torch.isfinite(aa.grad).all().item()
           and torch.allclose(P.masked_fill(~v, 0.0).sum(-1),
                              torch.ones(1, 1, a.shape[-2], dtype=torch.float64), atol=1e-12))

    print("== check 5: full causal attention fwd/bwd, NaN safety + masked positions ==")
    cfg = mo.GPTConfig(block_size=64, n_layer=2, n_head=2, n_embd=64, attn_mode="quant",
                       attn_K=K, attn_approx="nearest", attn_range_grad="full",
                       attn_range_mode="relevant_zero", attn_range_tau=TAU,
                       attn_tail_policy="resolved")
    torch.manual_seed(1); m = mo.GPT(cfg).cuda()
    idx = torch.randint(0, 50257, (2, 64)).cuda()
    logits, _ = m(idx)
    logits.float().square().mean().backward()
    gs = [p.grad for p in m.parameters() if p.grad is not None]
    ck("check5 all parameter gradients finite", all(torch.isfinite(g).all().item() for g in gs),
       f"{len(gs)} tensors")
    # masked positions must carry no weight / probability
    Tn = 32
    cz = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool))
    a = torch.randn(1, 2, Tn, Tn, dtype=torch.float64) * 8.0
    w = rec(qa, a, cz, "resolved", "nearest")[0]
    P = w / w.sum(-1, keepdim=True)
    inv = ~cz.view(1, 1, Tn, Tn).expand_as(w)
    ck("check5 causally masked positions have w = 0", (w[inv] == 0).all().item())
    ck("check5 causally masked positions have P = 0", (P[inv] == 0).all().item())

    print("== check 6: no-tail rows -> resolved gradients bit-identical to zero ==")
    for approx in ("nearest", "lerp"):
        ga = {}
        for pol in ("zero", "resolved"):
            a = att.clone().requires_grad_(True)
            w = rec(qa, a, causal, pol, approx)[0]
            (w.sum()).backward()
            ga[pol] = a.grad.clone()
        rows_no_tail = ~(tail.any(-1))                       # (1,H,T) rows with no tail key
        sel = rows_no_tail.unsqueeze(-1).expand_as(ga["zero"]) & valid
        ck(f"check6 {approx}: grads bit-identical on no-tail rows",
           torch.equal(ga["zero"][sel], ga["resolved"][sel]),
           f"{rows_no_tail.sum().item()} rows, {sel.sum().item()} entries")
        has = tail.any(-1)
        ck(f"check6 {approx}: grads DO differ on rows with tails",
           not torch.equal(ga["zero"][has.unsqueeze(-1).expand_as(ga['zero']) & valid],
                           ga["resolved"][has.unsqueeze(-1).expand_as(ga['zero']) & valid]))

    n_ok = sum(1 for _, o, _ in results if o)
    print(f"\nOPERATOR VALIDATION {'PASS' if n_ok == len(results) else 'FAIL'}  ({n_ok}/{len(results)})")
    return n_ok == len(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
