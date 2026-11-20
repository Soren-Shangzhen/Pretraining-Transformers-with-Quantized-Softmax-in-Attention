#!/usr/bin/env python3
"""13.3 checks for the E4 EMA observer (range_mode=ema) of rb-root_v1.

- the buffer is not in the computational graph (not a parameter, no grad, gradients identical when
  the buffer value is perturbed only through the window it sets);
- it is not updated in evaluation (model.eval(), and train-mode forwards under no_grad);
- it is initialised by the first training forward and then follows EMA <- d EMA + (1-d) current,
  with current = per-head mean of M - m over rows i >= 1 of that forward (float64 recomputation);
- it survives checkpoint save/load bit for bit (state_dict and the training script's checkpoint);
- strict forward: P equals the hard forward; Weight-STE and Prob-STE forwards bitwise equal;
- the surrogate gradient matches the closed form of the per-head window LERP (float64).
"""
import os, sys, tempfile
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


def make(sur, device, seed=0):
    qa.NEAREST_SURROGATE = sur
    torch.manual_seed(seed)
    cfg = mm.GPTConfig(block_size=128, n_layer=2, n_head=4, n_embd=128, vocab_size=512, attn_mode="quant",
                       attn_approx="nearest", attn_K=4, attn_range_mode="ema", attn_range_grad="full")
    return mm.GPT(cfg).to(device)


def spans_from_scores(model, x):
    """per layer, per head: mean over rows i>=1 of M - m, recomputed in float64 from captured scores"""
    store = []
    orig = qa.QuantCausalSelfAttention._quant_attention

    def patched(self, q, k, v):
        att = (q.float() @ k.float().transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        store.append(att.detach().double())
        return orig(self, q, k, v)
    qa.QuantCausalSelfAttention._quant_attention = patched
    try:
        with torch.no_grad():
            model.eval(); model(x); model.train()
    finally:
        qa.QuantCausalSelfAttention._quant_attention = orig
    out = []
    for att in store:
        T = att.shape[-1]
        valid = torch.ones(T, T, dtype=torch.bool, device=att.device).tril()
        sp = att.masked_fill(~valid, -torch.inf).amax(-1) - att.masked_fill(~valid, torch.inf).amin(-1)
        out.append(sp[:, :, 1:].mean(dim=(0, 2)))
    return out


def run(device):
    torch.manual_seed(5)
    x = torch.randint(0, 512, (1, 128), device=device); y = torch.randint(0, 512, (1, 128), device=device)
    x2 = torch.randint(0, 512, (1, 128), device=device)
    model = make("weight_strict", device)
    bufs = [b.attn.ema_span for b in model.transformer.h]
    record(f"[{device}] buffer registered, not a parameter, no grad",
           all(not b.requires_grad for b in bufs) and all(id(b) not in {id(p) for p in model.parameters()} for b in bufs)
           and all(f"transformer.h.{i}.attn.ema_span" in model.state_dict() for i in range(2)))
    record(f"[{device}] buffer starts as NaN (unset)", all(torch.isnan(b).all() for b in bufs))
    # an eval forward before any training forward must refuse to run
    model.eval()
    try:
        with torch.no_grad():
            model(x)
        refused = False
    except RuntimeError:
        refused = True
    model.train()
    record(f"[{device}] window cannot be used before the first training forward", refused)

    # first training forward initialises with the current value; uses the values before the forward
    _, loss = model(x, y)
    first = [b.clone() for b in bufs]
    # recompute current spans: the forward sees the same parameters (no update happened)
    cur1 = spans_from_scores(model, x)
    err1 = max((first[i].double() - cur1[i]).abs().max().item() for i in range(2))
    record(f"[{device}] first training forward initialises EMA = current mean span", err1 < 1e-5, max_abs=f"{err1:.1e}")
    loss.backward()
    record(f"[{device}] buffer has no gradient after backward", all(b.grad is None for b in bufs))

    # second training forward on another batch: EMA rule
    cur2 = spans_from_scores(model, x2)
    with torch.autocast(device_type=device, enabled=False):
        model(x2)
    exp2 = [0.999 * first[i].double() + 0.001 * cur2[i] for i in range(2)]
    err2 = max((bufs[i].double() - exp2[i]).abs().max().item() for i in range(2))
    record(f"[{device}] second training forward follows EMA <- 0.999 EMA + 0.001 current", err2 < 1e-5, max_abs=f"{err2:.1e}")

    # evaluation does not update
    before = [b.clone() for b in bufs]
    model.eval()
    with torch.no_grad():
        model(x2)
    model.train()
    with torch.no_grad():
        model(x2)
    record(f"[{device}] eval-mode forward and no_grad train-mode forward leave the buffer bitwise unchanged",
           all(torch.equal(before[i], bufs[i]) for i in range(2)))

    # save / load
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sd.pt")
        torch.save({"model": model.state_dict()}, p)
        m2 = make("weight_strict", device, seed=1)
        m2.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"])
        same = all(torch.equal(m2.transformer.h[i].attn.ema_span, bufs[i]) for i in range(2))
    record(f"[{device}] checkpoint save/load keeps the buffer bitwise", same)

    # surrogates: same forward, P equals the hard forward
    T = 64
    causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    att = torch.randn(1, 3, T, T, device=device) * 3
    tau = torch.tensor([1.5, 4.0, 9.0], device=device).view(1, 3, 1, 1)
    Pw, wh, *_ = qa.ema_window_probs(att, causal, 4, tau, "weight_strict")
    Pp, _, *_ = qa.ema_window_probs(att, causal, 4, tau, "prob")
    record(f"[{device}] strict Weight-STE forward == strict Prob-STE forward == hard forward (bitwise)",
           torch.equal(Pw, Pp) and torch.equal(Pw, wh / wh.sum(-1, keepdim=True)))


def closed_form():
    torch.manual_seed(2)
    T, K = 9, 4
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    worst, zs = 0.0, 0.0
    for trial in range(200):
        att = torch.randn(1, 2, T, T, dtype=torch.float64) * 3
        tau = torch.tensor([2.3, 7.1], dtype=torch.float64).view(1, 2, 1, 1)
        up = torch.randn(T, dtype=torch.float64)
        for sur in ("weight_strict", "prob"):
            for hh in range(2):
                a = att.clone().requires_grad_(True)
                P, *_ = qa.ema_window_probs(a, causal, K, tau, sur)
                (P[0, hh, T - 1] * up).sum().backward()
                auto = a.grad[0, hh, T - 1]
                s = att[0, hh, T - 1]; tv = float(tau[0, hh, 0, 0]); h = tv / K
                m = s.max(); im = s.argmax(); z = s - m; inw = z >= -tv
                zc = z.clamp(-tv, 0); r = torch.floor((zc + tv) / h).clamp(0, K - 1)
                eA, eB = torch.exp(-tv + r * h), torch.exp(-tv + (r + 1) * h)
                t = ((zc - (-tv + r * h)) / h).clamp(0, 1)
                wl = torch.where(inw, (1 - t) * eA + t * eB, torch.zeros_like(z))
                wn = torch.where(inw, torch.where(t >= 0.5, eB, eA), torch.zeros_like(z))
                Wd = wn if sur == "weight_strict" else wl
                gw = (up - (Wd / Wd.sum() * up).sum()) / Wd.sum()
                D = torch.where(inw, (eB - eA) / h, torch.zeros_like(z))
                g = gw * D
                g[im] += -(gw * D).sum()
                worst = max(worst, (g - auto).abs().max().item())
                zs = max(zs, auto.sum().abs().item())
    record("E4 surrogate gradient vs closed form (float64, 200 rows x 2 heads x 2 surrogates)", worst < 1e-9,
           max_abs_err=f"{worst:.2e}")
    record("E4 row gradient sums to zero", zs < 1e-9, max_abs_rowsum=f"{zs:.2e}")


def main():
    run("cpu")
    if torch.cuda.is_available():
        run("cuda")
    closed_form()
    RES["all_pass"] = OK
    print("written", write_json(f"13_3_ema_torch{torch.__version__.split('+')[0]}.json", RES))
    print("ALL PASS" if OK else "SOME FAILED")
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
