#!/usr/bin/env python3
"""Local only: reference outputs of the IntAttention PyTorch simulation for the 5090 comparison
(user decision 2026-11-10, item 4).  Their files are loaded read-only from the path on the command line;
the output file holds tensors (inputs and their outputs), no code.

Contents (torch.save dict):
  cases[name] = {scores fp32 (B,H,T,T) | int scores + scale, mask: causal,
                 e3a_i_cpu: idx_softmax_fp on CPU, e3a_i_gpu_diff: (flat index, value) where their RTX 3090
                 output differs from their CPU output, exaq_cpu / exaq_gpu_diff likewise,
                 e3a_ii_codes (integer cases only): their integer idx_softmax codes}
usage: python -I make_e3_reference.py /path/to/IntAttention-main/ArtifactEvaluation/pysimulation OUT.pt
"""
import importlib.util, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "checks"))
import rb_common  # noqa: E402,F401  (single-threaded CPU: references comparable on any machine)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TP, OUT = sys.argv[1], sys.argv[2]
fp = load(os.path.join(TP, "IndexSoftmaxOnly.py"), "tp_fp")
it = load(os.path.join(TP, "IntAttention.py"), "tp_int")
ex = load(os.path.join(TP, "EXAQ.py"), "tp_exaq")


def causal(b, h, n, dev="cpu"):
    return torch.ones(n, n, dtype=torch.bool, device=dev).tril().expand(b, h, n, n)


def diff(gpu, cpu):
    g = gpu.cpu().reshape(-1); c = cpu.reshape(-1)
    idx = torch.nonzero(g.view(torch.int32) != c.view(torch.int32)).squeeze(1)
    return idx, g[idx]


def theirs(s):
    m = causal(s.shape[0], s.shape[1], s.shape[-1])
    out = {"e3a_i_cpu": fp.idx_softmax_fp(s.masked_fill(~m, float("-inf")), 5, 6.6),
           "exaq_cpu": ex.exaq().exaq_softmax(s.clone(), -1, m, bitwidth=3)}
    sg, mg = s.cuda(), m.cuda()
    out["e3a_i_gpu_diff"] = diff(fp.idx_softmax_fp(sg.masked_fill(~mg, float("-inf")), 5, 6.6), out["e3a_i_cpu"])
    out["exaq_gpu_diff"] = diff(ex.exaq().exaq_softmax(sg.clone(), -1, mg, bitwidth=3), out["exaq_cpu"])
    return out


cases = {}
g = torch.Generator().manual_seed(20261110)
for n, H in ((64, 4), (256, 4), (1024, 2)):
    for scale in (0.3, 3.0, 8.0):
        s = torch.randn(1, H, n, n, generator=g) * scale
        cases[f"random_n{n}_s{scale}"] = dict(scores=s, **theirs(s))
k = torch.arange(0, 40, dtype=torch.float32)
base = k * torch.tensor(6.6, dtype=torch.float32) / 31
cands = torch.cat([base, torch.nextafter(base, torch.tensor(float("inf"))), torch.nextafter(base, torch.tensor(float("-inf")))])
s = -cands.abs().reshape(1, 1, 1, -1).clone(); s[..., 0] = 0.0
s = s.expand(1, 1, s.shape[-1], s.shape[-1]).clone()           # every row: boundary values (causal mask applies)
cases["bin_boundaries"] = dict(scores=s, **theirs(s))
from rb_common import LOCAL_CKPT, capture_scores  # noqa: E402
for name in ("softmax_s7_100m", "fwm_weight_k4_s7_100m"):
    atts = capture_scores(LOCAL_CKPT[name], n_blocks=1)
    for li in (0, 11):
        s = atts[li][:, :4].cpu().contiguous()
        cases[f"real_{name}_L{li}_h0-3"] = dict(scores=s, **theirs(s))
for nbits in (6, 10, 12):
    sc = torch.tensor(2.0 ** -nbits, dtype=torch.float32)
    for n in (64, 512):
        ints = torch.round(torch.randn(1, 2, n, n, generator=g) * 3.0 / sc).to(torch.int32)
        m = causal(1, 2, n)
        codes = it.idx_softmax(ints, torch.tensor(1.0), torch.tensor(1.0), sc, m, 5, 6.6)
        zero_thr = int((torch.tensor(6.6) / sc).to(torch.int32))
        cases[f"int_nbits{nbits}_n{n}"] = dict(int_scores=ints, scale=sc, thr=zero_thr * float(sc),
                                               e3a_ii_codes=codes.to(torch.uint8))
meta = dict(source="IntAttention-main ArtifactEvaluation/pysimulation (loaded read-only, not copied)",
            torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
torch.save(dict(meta=meta, cases=cases), OUT)
for k_, v in cases.items():
    if "e3a_i_gpu_diff" in v:
        print(f"{k_:40s} their GPU != CPU elements: E3a-i {v['e3a_i_gpu_diff'][0].numel()}, EXAQ {v['exaq_gpu_diff'][0].numel()}")
print("written", OUT, f"{os.path.getsize(OUT) / 1e6:.0f} MB")
