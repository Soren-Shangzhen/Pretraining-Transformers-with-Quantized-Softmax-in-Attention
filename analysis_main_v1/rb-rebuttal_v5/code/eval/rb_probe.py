#!/usr/bin/env python3
"""Attention probes for the rb-root_v1 runs (user decision 2026-11-12, evaluation list part 2; plan E0b, E3a/E3b,
E4, E11).  First 8 validation blocks of the run's corpus, batch 1, bf16 autocast outside the fp32 operator,
TF32 off, rows i >= 1 (the paper's geometry-probe convention); evaluation mode, so the E4 buffer stays frozen.

The probe does not re-implement any operator.  It patches the four mode methods of the run's own
QuantCausalSelfAttention and calls the original method a second time:
  probabilities: method(q, k, V = identity) returns P @ I = P, the operator's exact P (fp32 matmul with an
                 identity, TF32 off, is exact); a self-check compares (P @ V) with the method's normal output.
  score gradient (--grad): method(att * sqrt(T), identity, V) recomputes att' = (att*32) @ I / 32 == att
                 bitwise for T = 1024, so the backward through the operator is the training backward and
                 att.grad is dL/dS; the loss is checked to equal the normal forward's loss bitwise.
Per layer (means over rows i >= 1 of the 8 blocks):
  span (M - m), per head as well; h (native step: MinMax span/K, FWM tau/K incl. per-layer tau, EMA buffer/K);
  TV(P, softmax); softmax mass on keys the operator sets to 0 (E0b); row sum of P; fraction of all-zero rows;
  p_max of P and of softmax; entries below the window's lower bound and their softmax mass (quantised modes);
  the same zero-mass / row-sum / all-zero / p_max by row length 1-64, 65-256, 257-512, 513-1024 (E3a);
  E3b: sigma, lo, step of the tensor-level window, fraction of clamped-tail entries, softmax and native mass on them;
  E4: buffer value per head next to the measured span per head;
  --cf-e3a (softmax runs): the E3a-i / E3a-ii forwards applied to the run's own scores (same metrics by row length);
  --grad: per layer ||dL/dS||_F, mean |sum_j dL/dS_ij| and its ratio to sum_j |dL/dS_ij| (row zero-sum residual),
          and the global parameter-gradient norm, per block.
Jobs TSV: label, eval_root, checkpoint, dataset, tokens_m, role, surrogate[, probes] (probes: grad, cf_e3a).
Output: <out>/<label>/<tokens>m_<ckpt stem>/probe.json (skipped if present).
"""
import argparse, csv, datetime as dt, json, math, os, socket, sys, time
from pathlib import Path
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rb_eval import COMMON, build_rb_model, val_spec  # noqa: E402,F401
from common import ValBlocks, import_code, sha256  # noqa: E402
from rb_provenance import provenance  # noqa: E402

GROUPS = [(1, 64), (65, 256), (257, 512), (513, 1024)]          # row length = i + 1 valid keys
METHODS = ("_exact_fp32_softmax", "_softmax_family", "_external_operator", "_quant_attention")


class State:
    mode = "off"           # off | probs | grad
    inner = False
    store = []
    selfcheck = None


S = State()


def patch(qa):
    cls = qa.QuantCausalSelfAttention
    for name in METHODS:
        orig = getattr(cls, name)

        def patched(self, q, k, v, _orig=orig):
            if S.inner or S.mode == "off":
                return _orig(self, q, k, v)
            B, H, T, D = q.shape
            with torch.autocast(device_type="cuda", enabled=False):
                att = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(D)
            S.inner = True
            try:
                if S.mode == "probs":
                    eye = torch.eye(T, device=q.device, dtype=torch.float32).expand(B, H, T, T)
                    P = _orig(self, q.float(), k.float(), eye)
                    y = _orig(self, q, k, v)
                    if S.selfcheck is None:            # once per checkpoint: P @ V equals the operator output
                        with torch.autocast(device_type="cuda", enabled=False):
                            S.selfcheck = bool(torch.equal((P @ v.float()).to(q.dtype), y))
                    S.store.append((self, att, P))
                    return y
                assert T == 1024, "the identity trick is exact for T = 1024 (sqrt(T) = 32)"
                att.retain_grad()
                S.store.append((self, att, None))
                k2 = torch.eye(T, device=q.device, dtype=torch.float32).expand(B, H, T, T)
                return _orig(self, att * math.sqrt(T), k2, v).to(q.dtype)
            finally:
                S.inner = False
        setattr(cls, name, patched)


class Acc:
    def __init__(self):
        self.s = {}

    def add(self, key, val, n):
        a = self.s.setdefault(key, [0.0, 0])
        a[0] += float(val); a[1] += int(n)

    def mean(self):
        return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in self.s.items()}


@torch.no_grad()
def layer_metrics(qa, mod, att, P, acc, head_acc, cf):
    T = att.shape[-1]
    causal = torch.ones(T, T, dtype=torch.bool, device=att.device).tril()
    valid = causal.view(1, 1, T, T)
    neg = att.masked_fill(~valid, -torch.inf)
    Mx = neg.amax(-1)
    mn = att.masked_fill(~valid, torch.inf).amin(-1)
    span = (Mx - mn)[..., 1:]                                   # (B,H,T-1)
    Pe = torch.softmax(neg, -1).masked_fill(~valid, 0.0)
    z = att - Mx.unsqueeze(-1)
    n = span.numel()
    r = lambda t: t[..., 1:]
    acc.add("span", span.double().sum(), n)
    acc.add("nonfinite_P", (~torch.isfinite(P) & valid).sum(), 1)
    acc.add("tv_native_exact", r(0.5 * (P - Pe).abs().sum(-1)).double().sum(), n)
    zero = valid & (P == 0)
    zmass, rows, pmn, pme = (Pe * zero).sum(-1), (P * valid).sum(-1), P.amax(-1), Pe.amax(-1)
    for key, t in (("zeroed_softmax_mass", zmass), ("rowsum_native", rows), ("pmax_native", pmn), ("pmax_exact", pme)):
        acc.add(key, r(t).double().sum(), n)
    acc.add("allzero_row_frac", r(rows == 0).double().sum(), n)
    L = torch.arange(1, T + 1, device=att.device)

    def by_length(prefix, Pn):
        zm = (Pe * (valid & (Pn == 0))).sum(-1); rs = (Pn * valid).sum(-1); pm = Pn.amax(-1)
        for lo, hi in GROUPS:
            sel = (L >= max(lo, 2)) & (L <= hi)
            m = int(sel.sum()) * zm.shape[0] * zm.shape[1]
            g = f"{prefix}len{lo}-{hi}_"
            acc.add(g + "zeroed_softmax_mass", zm[..., sel].double().sum(), m)
            acc.add(g + "rowsum", rs[..., sel].double().sum(), m)
            acc.add(g + "allzero_row_frac", (rs[..., sel] == 0).double().sum(), m)
            acc.add(g + "pmax", pm[..., sel].double().sum(), m)
    by_length("", P)
    if cf:                                                       # E3a forwards on a softmax run's own scores
        by_length("cf_e3a_u8_", qa._index_softmax_hard(att, valid, "u8"))
        by_length("cf_e3a_fp_", qa._index_softmax_hard(att, valid, "fp"))
    span_h = span.mean(dim=(0, 2))                               # (H,)
    for h in range(span_h.numel()):
        head_acc.add(f"span_head{h:02d}", span_h[h], 1)
    if mod.mode == "quant":
        K = mod.K
        if mod.range_mode == "minmax":
            hrow, lb = span / K, None
        elif mod.range_mode == "relevant_zero":
            hrow, lb = torch.full_like(span, mod.range_tau / K), torch.full_like(Mx, -mod.range_tau)
        else:                                                    # ema: per-head buffer (frozen at evaluation)
            e = mod.ema_span.view(1, -1, 1).to(span.dtype)
            hrow, lb = (e / K).expand_as(span), (-mod.ema_span.view(1, -1, 1)).expand_as(Mx)
            for h in range(mod.ema_span.numel()):
                head_acc.add(f"ema_buffer_head{h:02d}", mod.ema_span[h], 1)
        acc.add("h_native", hrow.double().sum(), n)
        if lb is not None:
            below = valid & (z < lb.unsqueeze(-1))
            acc.add("below_window_entry_frac", r(below.sum(-1)).double().sum(), int(r(valid.expand_as(z).sum(-1)).sum()))
            acc.add("below_window_softmax_mass", r((Pe * below).sum(-1)).double().sum(), n)
    if mod.mode == "exaq":
        vf = valid.expand_as(att)
        x, lo, delta = qa._exaq_window(att, vf, mod.exaq_bits)
        sigma = x[vf].float().std(unbiased=False).item()
        acc.add("exaq_sigma", sigma, 1); acc.add("exaq_lo", lo, 1); acc.add("exaq_step", delta, 1)
        tail = valid & (z < lo)
        acc.add("exaq_tail_entry_frac", r(tail.sum(-1)).double().sum(), int(r(vf.sum(-1)).sum()))
        acc.add("exaq_tail_softmax_mass", r((Pe * tail).sum(-1)).double().sum(), n)
        acc.add("exaq_tail_native_mass", r((P * tail).sum(-1)).double().sum(), n)


def grad_metrics(model, x, y, acc_layers):
    S.mode = "off"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss_ref = model(x, y)
    S.mode, S.store = "grad", []
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    same = bool(torch.equal(loss.detach(), loss_ref))
    loss.backward()
    pn = math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters() if p.grad is not None))
    T = x.shape[1]
    valid = torch.ones(T, T, dtype=torch.bool, device=x.device).tril().view(1, 1, T, T)
    for li, (_, att, _) in enumerate(S.store):
        g = att.grad.masked_fill(~valid, 0.0)[..., 1:, :]       # rows i >= 1
        rs, ab = g.sum(-1), g.abs().sum(-1)
        n = rs.numel()
        a = acc_layers[li]
        a.add("score_grad_fro", float(torch.linalg.vector_norm(g.double())), 1)
        a.add("row_grad_sum_abs", rs.abs().double().sum(), n)
        a.add("row_grad_abs_sum", ab.double().sum(), n)
        nz = ab > 0
        a.add("row_zero_sum_residual_rel", (rs.abs()[nz] / ab[nz]).double().sum(), int(nz.sum()))
    S.store = []
    model.zero_grad(set_to_none=True)
    S.mode = "off"
    return float(loss.detach()), pn, same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--root", required=True); ap.add_argument("--blocks", type=int, default=8)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mm, qa = import_code(a.root)
    patch(qa)
    hashes = {f: sha256(os.path.join(a.root, f)) for f in ("model.py", "quant_attention.py")}
    jobs = [j for j in csv.DictReader(open(a.jobs), delimiter="\t") if j["eval_root"] == a.root]
    cache = {}
    for j in jobs:
        ckp = j["checkpoint"]
        out = Path(a.out) / j["label"] / f"{int(round(float(j['tokens_m']))):04d}m_{Path(ckp).stem}"
        if (out / "probe.json").exists():
            continue
        flags = set((j.get("probes") or "").split(",")) - {""}
        spec = val_spec(j["dataset"])
        if j["dataset"] not in cache:
            cache[j["dataset"]] = (ValBlocks(spec["path"], 1024), sha256(spec["path"]))
        blocks, vsha = cache[j["dataset"]]
        if spec["sha256"] and vsha != spec["sha256"]:
            raise SystemExit(f"validation file hash mismatch for {j['dataset']}")
        t0 = time.time()
        ck = torch.load(ckp, map_location="cpu", weights_only=False, mmap=True)
        model, cfg, op = build_rb_model(mm, ck, "cuda")
        nl = int(cfg.n_layer)
        acc = [Acc() for _ in range(nl)]; head = [Acc() for _ in range(nl)]; gacc = [Acc() for _ in range(nl)]
        S.selfcheck, losses, pnorms, same = None, [], [], []
        for b in range(a.blocks):
            x, y = blocks.batch(b, b + 1, "cuda")
            S.mode, S.store = "probs", []
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(x)
            S.mode = "off"
            for li, (mod, att, P) in enumerate(S.store):
                layer_metrics(qa, mod, att, P, acc[li], head[li], "cf_e3a" in flags)
            S.store = []
            if "grad" in flags:
                l, pn, eq = grad_metrics(model, x, y, gacc)
                losses.append(l); pnorms.append(pn); same.append(eq)
        layers = []
        for li in range(nl):
            d = dict(layer=li, **acc[li].mean(), **head[li].mean())
            if "grad" in flags:
                d.update({f"grad_{k}": v for k, v in gacc[li].mean().items()})
            layers.append(d)
        keys = [k for k in layers[0] if k != "layer"]
        res = dict(label=j["label"], role=j.get("role"), checkpoint=ckp, step=int(ck["step"]),
                   tokens_done=int(ck["tokens_done"]), checkpoint_operator=op, probes=sorted(flags),
                   eval_root=a.root, eval_code_sha256=hashes, dataset=j["dataset"], val_sha256=vsha,
                   probe_blocks=a.blocks, rows="i>=1", tf32=False, autocast="bfloat16",
                   selfcheck_P_times_V_equals_output=S.selfcheck,
                   layer_mean={k: float(sum(l[k] for l in layers) / nl) for k in keys
                               if all(isinstance(l.get(k), float) for l in layers)},
                   layers=layers, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                   host=socket.gethostname(), seconds=round(time.time() - t0, 1),
                   finished=dt.datetime.now().isoformat(timespec="seconds"),
                   provenance=provenance(ckp, a.root, [os.path.abspath(__file__), os.path.join(HERE, "rb_eval.py"),
                                                       os.path.join(COMMON, "common.py"), os.path.join(HERE, "rb_provenance.py")]))
        if "grad" in flags:
            res["grad"] = dict(loss_per_block=losses, param_grad_norm_per_block=pnorms,
                               param_grad_norm_mean=sum(pnorms) / len(pnorms),
                               patched_loss_equals_normal_loss=all(same))
        tmp = out.with_name(out.name + f".tmp_{os.getpid()}")
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "probe.json").write_text(json.dumps(res, indent=1, default=float))
        os.rename(tmp, out)
        lm = res["layer_mean"]
        print(f"done {j['label']} {j['tokens_m']}M span={lm.get('span', float('nan')):.3f} "
              f"tv={lm.get('tv_native_exact', float('nan')):.4g} selfcheck={S.selfcheck}"
              + (f" grad_same={all(same)} gnorm={res['grad']['param_grad_norm_mean']:.3f}" if "grad" in flags else "")
              + f" {res['seconds']}s", flush=True)
        del model, ck
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
