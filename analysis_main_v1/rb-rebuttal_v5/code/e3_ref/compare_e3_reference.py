#!/usr/bin/env python3
"""On a 5090 (no third-party code on the machine): compare the rb-root_v1 training forwards of E3a/E3b with
the reference outputs generated locally from the IntAttention simulation (make_e3_reference.py).

CPU: training op on the stored inputs == their CPU output, bitwise.
GPU: training op on CUDA == their RTX 3090 output (their CPU output with the recorded GPU differences
     applied), bitwise; if not, the number of differing elements and the largest difference are reported.
E3a-ii: codes round(255 e / sum e) of the training op == their integer-path codes on integer scores.
usage: compare_e3_reference.py REF.pt [--out JSON]
"""
import argparse, json, os, socket, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "checks"))
from rb_common import RB_ROOT, import_root  # noqa: E402

mm, qa = import_root(RB_ROOT)


def apply_diff(cpu, d):
    flat = cpu.clone().reshape(-1)
    flat[d[0]] = d[1]
    return flat.reshape(cpu.shape)


def cmp(a, b):
    a, b = a.cpu(), b.cpu()
    same = torch.equal(a.view(torch.int32), b.view(torch.int32)) if a.dtype == torch.float32 else torch.equal(a, b)
    nd = int((a != b).sum()) if not same else 0
    md = float((a.double() - b.double()).abs().max()) if not same else 0.0
    return same, nd, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    R = torch.load(a.ref, map_location="cpu", weights_only=False)
    res, ok = [], True
    for name, c in R["cases"].items():
        if "scores" in c:
            s = c["scores"]
            B, H, T, _ = s.shape
            for dev in ("cpu", "cuda"):
                sd = s.to(dev)
                valid = torch.ones(T, T, dtype=torch.bool, device=dev).tril().view(1, 1, T, T)
                fp_ = qa._index_softmax_hard(sd, valid, "fp")
                ex_, _, _ = qa._exaq_hard(sd, valid.expand(B, H, T, T), 3)
                ref_fp = c["e3a_i_cpu"] if dev == "cpu" else apply_diff(c["e3a_i_cpu"], c["e3a_i_gpu_diff"])
                ref_ex = c["exaq_cpu"] if dev == "cpu" else apply_diff(c["exaq_cpu"], c["exaq_gpu_diff"])
                for op, mine, ref in (("E3a-i", fp_, ref_fp), ("EXAQ", ex_, ref_ex)):
                    same, nd, md = cmp(mine, ref)
                    ok &= same
                    res.append(dict(case=name, op=op, device=dev, bitwise=same, n_diff=nd, max_abs_diff=md))
                    print(f"{'PASS' if same else 'FAIL'}  {name:40s} {op:6s} {dev:4s} n_diff={nd} max={md:.1e}", flush=True)
        else:
            for dev in ("cpu", "cuda"):
                sc = c["scale"].to(dev)
                s = c["int_scores"].to(dev).to(torch.float32) * sc
                T = s.shape[-1]
                valid = torch.ones(T, T, dtype=torch.bool, device=dev).tril().view(1, 1, T, T)
                codes = (qa._index_softmax_hard(s, valid, "u8", thr=c["thr"]) * 255.0).round().to(torch.uint8)
                same, nd, md = cmp(codes, c["e3a_ii_codes"])
                ok &= same
                res.append(dict(case=name, op="E3a-ii codes", device=dev, bitwise=same, n_diff=nd, max_abs_diff=md))
                print(f"{'PASS' if same else 'FAIL'}  {name:40s} E3a-ii codes {dev:4s} n_diff={nd}", flush=True)
    out = a.out or os.path.join(HERE, "..", "..", "checks", f"13_3_e3_reference_{socket.gethostname()}.json")
    json.dump(dict(reference_meta=R["meta"], torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                   all_bitwise=ok, results=res), open(out, "w"), indent=1)
    print("written", out, "| ALL BITWISE" if ok else "| DIFFERENCES (see file)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
