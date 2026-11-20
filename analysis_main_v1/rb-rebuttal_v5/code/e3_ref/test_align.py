"""Local alignment tests against the IntAttention artifact (not vendored).

usage: python -I test_align.py /path/to/IntAttention-main/ArtifactEvaluation/pysimulation
The third-party files are loaded from the path given on the command line and
are never copied.
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = load(os.path.join(HERE, "ref_ops.py"), "ref_ops")
tp = sys.argv[1]
theirs_fp = load(os.path.join(tp, "IndexSoftmaxOnly.py"), "tp_idx_fp")
theirs_int = load(os.path.join(tp, "IntAttention.py"), "tp_idx_int")
theirs_exaq = load(os.path.join(tp, "EXAQ.py"), "tp_exaq")

torch.manual_seed(0)
ok = True


def causal(b, h, n):
    return torch.ones(n, n, dtype=torch.bool).tril().expand(b, h, n, n)


def report(name, same, extra=""):
    global ok
    ok &= bool(same)
    print(f"{'PASS' if same else 'FAIL'}  {name}  {extra}")


# 0. table
t = theirs_fp.load_table(5, 6.6, "cpu")
report("table constants == formula == their load_table",
       t.tolist() == list(ref.IDX_TABLE) == ref.index_softmax_table_from_formula().tolist(),
       f"distinct nonzero = {len(set(t.tolist()) - {0})}")

# 1. E3a-i vs idx_softmax_fp, fp32 scores, bitwise
for n in (64, 256, 1024):
    m = causal(1, 4, n)
    for scale in (0.02, 0.3, 1.0, 3.0, 8.0, 25.0):
        s = torch.randn(1, 4, n, n) * scale
        mine = ref.index_softmax_fp(s, m)
        th = theirs_fp.idx_softmax_fp(s.masked_fill(~m, float("-inf")), 5, 6.6)
        same = torch.equal(mine, th)
        if not same:
            report(f"E3a-i bitwise n={n} scale={scale}", False,
                   f"max abs diff {(mine - th).abs().max().item():.3e}")
report("E3a-i == idx_softmax_fp bitwise on random causal rows (18 settings)", ok)

# 1b. scores placed exactly on and next to every bin boundary
k = torch.arange(0, 40, dtype=torch.float32)
base = k * torch.tensor(6.6, dtype=torch.float32) / 31
cands = torch.cat([base, torch.nextafter(base, torch.tensor(float("inf"))),
                   torch.nextafter(base, torch.tensor(float("-inf")))])
s = -cands.abs().reshape(1, 1, 1, -1).clone()
s[..., 0] = 0.0
m = torch.ones_like(s, dtype=torch.bool)
report("E3a-i == idx_softmax_fp bitwise on bin boundaries",
       torch.equal(ref.index_softmax_fp(s, m), theirs_fp.idx_softmax_fp(s, 5, 6.6)))

# 2. E3a-ii codes vs the integer path idx_softmax, integer scores, power-of-two scale
all_same = True
for nbits in (6, 10, 12):
    sc = torch.tensor(2.0 ** -nbits, dtype=torch.float32)
    for n in (64, 512, 1024):
        m = causal(1, 4, n)
        for spread in (0.5, 3.0, 12.0):
            ints = torch.round(torch.randn(1, 4, n, n) * spread / sc).to(torch.int32)
            th = theirs_int.idx_softmax(ints, torch.tensor(1.0), torch.tensor(1.0), sc, m, 5, 6.6)
            zero_thr = int((torch.tensor(6.6) / sc).to(torch.int32))
            mine = ref.index_softmax_u8_codes(ints.to(torch.float32) * sc, m, thr=zero_thr * float(sc))
            all_same &= torch.equal(mine.to(torch.int32), th)
report("E3a-ii codes == idx_softmax codes on integer scores (27 settings)", all_same)

# 3. EXAQ vs the reproduction in the artifact, bitwise
all_same = True
worst = 0.0
for bits in (2, 3, 4):
    for n in (64, 256, 1024):
        m = causal(1, 4, n)
        for scale in (0.02, 1.0, 3.0, 8.0):
            s = torch.randn(1, 4, n, n) * scale
            th = theirs_exaq.exaq().exaq_softmax(s.clone(), -1, m, bitwidth=bits)
            mine = ref.exaq_softmax(s, m, bits)
            all_same &= torch.equal(mine, th)
            worst = max(worst, (mine - th).abs().max().item())
report("EXAQ == artifact reproduction bitwise (36 settings)", all_same, f"max abs diff {worst:.1e}")

# 4. strict-forward expression returns the hard value bit for bit
s = (torch.randn(1, 4, 256, 256) * 3).requires_grad_()
m = causal(1, 4, 256)
p_hard = ref.index_softmax_u8(s.detach(), m)
p_sur = torch.softmax(s.masked_fill(~m, float("-inf")), -1)
p = p_hard + (p_sur - p_sur.detach())
report("strict expression forward == hard value bitwise", torch.equal(p.detach(), p_hard))

print("\nALL PASS" if ok and all_same else "\nSOME FAILED")
