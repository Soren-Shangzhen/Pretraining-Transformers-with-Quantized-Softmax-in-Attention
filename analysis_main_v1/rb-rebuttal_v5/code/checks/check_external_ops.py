#!/usr/bin/env python3
"""13.3 checks for IndexSoftmax (E3a) and EXAQ (E3b) inside rb-root_v1.

1. The training hard forward equals ref_ops (our reference forward) and the IntAttention PyTorch
   simulation, loaded read-only from the path given on the command line (never copied), on random
   causal rows and on real attention scores of every layer of a trained model, CPU and GPU.
   E3a-i: probabilities bitwise.  E3a-ii: integer codes bitwise; the value is code/255 correctly
   rounded on every device.  EXAQ: bitwise.
2. The training module returns exactly the hard forward (strict expression).
3. Surrogate gradient vs an independent derivative (float64): E3a and E3b-d against the closed form,
   E3b-g against central finite differences (entries whose grid index or clamp state would change
   under the perturbation are skipped and counted).
4. E3b-d and E3b-g forwards are bitwise identical; row gradients sum to zero (row maximum in graph).

usage: python check_external_ops.py /path/to/IntAttention-main/ArtifactEvaluation/pysimulation
"""
import importlib.util, os, sys
import torch
from rb_common import RB, RB_ROOT, LOCAL_CKPT, import_root, write_json, env_info, capture_scores

mm, qa = import_root(RB_ROOT)
RES = {"env": env_info(), "checks": []}
OK = True


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = load(os.path.join(RB, "e3_ref", "ref_ops.py"), "ref_ops")
TP = sys.argv[1]
if TP == "--no-thirdparty":
    # cloud machines: the third-party simulation stays on the local machine (user rule); the training
    # forward is compared with ref_ops, whose equality with the simulation was shown locally
    theirs_fp = theirs_exaq = None
else:
    theirs_fp = load(os.path.join(TP, "IndexSoftmaxOnly.py"), "tp_idx_fp")
    theirs_exaq = load(os.path.join(TP, "EXAQ.py"), "tp_exaq")
RES["thirdparty_path"] = TP


def record(name, passed, **info):
    global OK
    OK &= bool(passed)
    RES["checks"].append(dict(name=name, passed=bool(passed), **info))
    print(f"{'PASS' if passed else 'FAIL'}  {name}  " + " ".join(f"{k}={v}" for k, v in info.items()), flush=True)


def compare_hard(att, tag):
    """att: (B,H,T,T) fp32 scores on some device."""
    B, H, T, _ = att.shape
    causal = torch.ones(T, T, dtype=torch.bool, device=att.device).tril()
    valid = causal.view(1, 1, T, T)
    m = valid.expand(B, H, T, T)
    mine_fp = qa._index_softmax_hard(att, valid, "fp")
    ok_ref = torch.equal(mine_fp, ref.index_softmax_fp(att, m))
    ok_tp = None if theirs_fp is None else torch.equal(mine_fp, theirs_fp.idx_softmax_fp(att.masked_fill(~m, float("-inf")), 5, 6.6))
    mine_u8 = qa._index_softmax_hard(att, valid, "u8")
    codes = ref.index_softmax_u8_codes(att, m)
    ok_codes = torch.equal((mine_u8 * 255.0).round(), codes)
    ok_val = torch.equal(mine_u8, codes / torch.full_like(codes, 255.0))
    mine_ex, lo, delta = qa._exaq_hard(att, m, 3)
    ok_ex_ref = torch.equal(mine_ex, ref.exaq_softmax(att, m, 3))
    ok_ex_tp = None if theirs_exaq is None else torch.equal(mine_ex, theirs_exaq.exaq().exaq_softmax(att.clone(), -1, m, bitwidth=3))
    record(f"{tag} E3a-i training == ref_ops (bitwise)", ok_ref)
    if ok_tp is not None:
        record(f"{tag} E3a-i training == IntAttention idx_softmax_fp (bitwise)", ok_tp)
    record(f"{tag} E3a-ii codes == ref_ops codes (bitwise); value == code/255", ok_codes and ok_val,
           codes=ok_codes, value=ok_val)
    record(f"{tag} EXAQ 3-bit training == ref_ops (bitwise)", ok_ex_ref, lo=f"{lo:.4f}", delta=f"{delta:.4f}")
    if ok_ex_tp is not None:
        record(f"{tag} EXAQ 3-bit training == IntAttention-repo reproduction (bitwise)", ok_ex_tp)
    # module path: strict expression returns the hard forward
    for mode, variant, hard in (("indexsoftmax", "fp", mine_fp), ("indexsoftmax", "u8", mine_u8),
                                ("exaq", "detach", mine_ex), ("exaq", "graph", mine_ex)):
        P = qa.external_operator_probs(att, causal, mode, variant)
        record(f"{tag} {mode}-{variant} training probabilities == hard forward (bitwise)", torch.equal(P, hard))
    # zero rows (E3a-ii) and table usage, for the record
    rs = mine_u8.sum(-1)
    return dict(u8_rows_all_zero=float((rs == 0).float().mean()), u8_rowsum_mean=float(rs.mean()))


def random_and_real(device):
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    for n in (64, 256, 1024):
        for scale in (0.3, 3.0, 8.0):
            compare_hard(torch.randn(1, 4, n, n, device=device) * scale, f"[{device}] random n={n} scale={scale}")
    if device == "cuda":
        stats = {}
        for name in ("softmax_s7_100m", "fwm_weight_k4_s7_100m"):
            atts = capture_scores(LOCAL_CKPT[name], n_blocks=1)
            for li, att in enumerate(atts):
                stats[f"{name}_L{li}"] = compare_hard(att, f"[cuda] real scores {name} layer {li}")
                if li in (0, 11):
                    compare_hard(att.cpu(), f"[cpu] real scores {name} layer {li}")
        RES["real_score_u8_stats"] = stats


# ----------------------------------------------------------------- gradients (float64)
def cf_lerp_window(s, gw, K, tau):
    m = s.max(); im = s.argmax(); h = tau / K
    z = s - m; inwin = z >= -tau
    zc = z.clamp(-tau, 0)
    r = torch.floor((zc + tau) / h).clamp(0, K - 1)
    eA, eB = torch.exp(-tau + r * h), torch.exp(-tau + (r + 1) * h)
    t = ((zc - (-tau + r * h)) / h).clamp(0, 1)
    w = torch.where(inwin, (1 - t) * eA + t * eB, torch.zeros_like(z))
    D = torch.where(inwin, (eB - eA) / h, torch.zeros_like(z))
    g = gw * D
    g[im] += -(gw * D).sum()
    return w, g


def cf_exaq_detach(s, gw, lo, delta, n_lev=7):
    m = s.max(); im = s.argmax(); x = s - m
    inwin = x > lo
    xc = x.clamp(lo, 0)
    k = torch.floor((xc - lo) / delta).clamp(0, n_lev - 1)
    g0 = lo + k * delta
    t = ((xc - g0) / delta).clamp(0, 1)
    w = (1 - t) * torch.exp(g0) + t * torch.exp(g0 + delta)
    D = torch.where(inwin, (torch.exp(g0 + delta) - torch.exp(g0)) / delta, torch.zeros_like(x))
    g = gw * D
    g[im] += -(gw * D).sum()
    return w, g


def gradient_checks():
    torch.manual_seed(1)
    T = 10
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    row = T - 1
    worst = {"E3a_fp": 0.0, "E3a_u8": 0.0, "E3b_detach": 0.0}
    zs = {"E3a": 0.0, "E3b_detach": 0.0, "E3b_graph": 0.0}
    for trial in range(200):
        att = torch.randn(1, 1, T, T, dtype=torch.float64) * 3
        up = torch.randn(T, dtype=torch.float64)
        s = att[0, 0, row]
        for variant in ("fp", "u8"):
            a = att.clone().requires_grad_(True)
            P = qa.external_operator_probs(a, causal, "indexsoftmax", variant)
            (P[0, 0, row] * up).sum().backward()
            wl, _ = cf_lerp_window(s, torch.zeros(T, dtype=torch.float64), 31, 6.6)
            gw = (up - (wl / wl.sum() * up).sum()) / wl.sum()
            _, g = cf_lerp_window(s, gw, 31, 6.6)
            worst[f"E3a_{variant}"] = max(worst[f"E3a_{variant}"], (g - a.grad[0, 0, row]).abs().max().item())
            zs["E3a"] = max(zs["E3a"], a.grad[0, 0, row].sum().abs().item())
        a = att.clone().requires_grad_(True)
        P = qa.external_operator_probs(a, causal, "exaq", "detach")
        (P[0, 0, row] * up).sum().backward()
        _, lo, delta = qa._exaq_hard(att, causal.view(1, 1, T, T).expand(1, 1, T, T), 3)
        wl, _ = cf_exaq_detach(s, torch.zeros(T, dtype=torch.float64), lo, delta)
        gw = (up - (wl / wl.sum() * up).sum()) / wl.sum()
        _, g = cf_exaq_detach(s, gw, lo, delta)
        worst["E3b_detach"] = max(worst["E3b_detach"], (g - a.grad[0, 0, row]).abs().max().item())
        zs["E3b_detach"] = max(zs["E3b_detach"], a.grad[0, 0, row].sum().abs().item())
    for k, v in worst.items():
        record(f"surrogate gradient vs closed form (float64, 200 random rows): {k}", v < 1e-9, max_abs_err=f"{v:.2e}")

    # E3b-g: autograd vs central finite differences of the surrogate, whole-tensor loss
    T = 8
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    valid = causal.view(1, 1, T, T)
    vf = valid.expand(1, 2, T, T)
    worst_fd, checked, skipped = 0.0, 0, 0
    eps = 1e-6
    for trial in range(30):
        att = torch.randn(1, 2, T, T, dtype=torch.float64) * 2.5
        up = torch.randn(1, 2, T, T, dtype=torch.float64)
        _, lo, delta = qa._exaq_hard(att, vf, 3)

        def f(a):
            return (qa._exaq_surrogate(a, valid, vf, 3, "graph", lo, delta) * up).sum()

        def state(a):
            x = a - a.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
            xv = x[vf]
            c = -1.75 * xv.std(unbiased=False) - 2.06
            lo_t = torch.maximum(c, xv.min()); d = -lo_t / 7
            xc = torch.minimum(torch.maximum(x, lo_t), torch.zeros_like(x))
            return torch.floor((xc - lo_t) / d).clamp(0, 6), x > lo_t, c > xv.min()
        a = att.clone().requires_grad_(True)
        f(a).backward()
        zs["E3b_graph"] = max(zs["E3b_graph"], a.grad.masked_fill(~valid, 0).sum(-1).abs().max().item())
        base = state(att)
        for hh in range(2):
            for i in range(T):
                for j in range(i + 1):
                    ap, am = att.clone(), att.clone()
                    ap[0, hh, i, j] += eps; am[0, hh, i, j] -= eps
                    sp, sm = state(ap), state(am)
                    if not all(torch.equal(x, y) and torch.equal(x, z) for x, y, z in zip(base, sp, sm)):
                        skipped += 1
                        continue
                    fd = (f(ap) - f(am)).item() / (2 * eps)
                    worst_fd = max(worst_fd, abs(fd - a.grad[0, hh, i, j].item()) / max(1.0, abs(fd)))
                    checked += 1
    record("E3b-g surrogate gradient vs central finite differences (float64)", worst_fd < 1e-6,
           max_rel_err=f"{worst_fd:.2e}", entries_checked=checked, entries_skipped_near_kinks=skipped)
    for k, v in zs.items():
        record(f"row gradients sum to zero: {k}", v < 1e-9, max_abs_rowsum=f"{v:.2e}")


def main():
    random_and_real("cpu")
    if torch.cuda.is_available():
        random_and_real("cuda")
    gradient_checks()
    RES["all_pass"] = OK
    print("written", write_json(f"13_3_external_ops_torch{torch.__version__.split('+')[0]}.json", RES))
    print("ALL PASS" if OK else "SOME FAILED")
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
