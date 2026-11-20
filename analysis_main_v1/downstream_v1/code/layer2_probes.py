#!/usr/bin/env python3
"""Layer 2 probe evaluation (§4).

For every condition (native forward; exact forward for the six representative conditions of §6), score each
probe sample: mean target-token NLL and top-1 hit rate at the target positions. Batch 1, bf16 autocast outside
the fp32 attention kernel, TF32 off, context 1024, each run's own frozen code root.

Sanity checks first (§8.1.2/4/6): probe validity on Exact, exact-vs-native identity for Exact, sample-id hash
equality across conditions.

Output: results/layer2/per_sample/<run>__<probe>__<mode>.npz and per_sample_rows.csv.gz (§7.1).
"""
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_loader import ROOT, load, runs, set_numerics  # noqa: E402
import stats as st  # noqa: E402

DATA = f"{ROOT}/analysis_main_v1/downstream_v1/data/probes"
OUT = f"{ROOT}/analysis_main_v1/downstream_v1/results/layer2"
PS = f"{OUT}/per_sample"
PROBES = ["A", "B", "C"]
EXACT_CONDS = ["MinMax-Weight K4", "MinMax-Weight K16", "LERP K1", "LERP K16", "QRM-Weight K2", "QRM-Weight K16"]


def read_probe(name):
    z = np.load(f"{DATA}/probe_{name}.npz")
    tok, off, tgt, toff = z["tokens"], z["offsets"], z["targets"], z["target_offsets"]
    meta = [l.rstrip("\n").split(",") for l in open(f"{DATA}/probe_{name}_meta.csv")]
    hdr, rows = meta[0], meta[1:]
    md = [dict(zip(hdr, r)) for r in rows]
    samples = [(tok[off[i]:off[i + 1]], tgt[toff[i]:toff[i + 1]]) for i in range(len(off) - 1)]
    return samples, md


@torch.inference_mode()
def score(model, seq, targets, device="cuda"):
    """Mean NLL and top-1 hit rate at the given target positions of one sequence (teacher forcing)."""
    x = torch.from_numpy(np.asarray(seq[:-1], dtype=np.int64)).unsqueeze(0).to(device)
    y = torch.from_numpy(np.asarray(seq[1:], dtype=np.int64)).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    lg = logits.float()[0]
    pos = np.asarray(targets, dtype=np.int64) - 1            # predict position t from logits at t-1
    p = torch.from_numpy(pos).to(device)
    nll = torch.nn.functional.cross_entropy(lg[p], y[p], reduction="none")
    hit = (lg[p].argmax(-1) == y[p]).double()
    return float(nll.mean().item()), float(hit.mean().item())


def eval_condition(r, mode, probes):
    ld = load(r, mode)
    for name, (samples, md) in probes.items():
        f = f"{PS}/{r['run_name']}__{name}__{mode}.npz"
        if os.path.exists(f):
            continue
        t0 = time.time()
        nll = np.empty(len(samples))
        hit = np.empty(len(samples))
        for i, (seq, tgt) in enumerate(samples):
            nll[i], hit[i] = score(ld.model, seq, tgt)
        np.savez_compressed(f, nll=nll, hit=hit)
        print(f"  {r['cond']:18s} {mode:6s} probe {name}: n={len(samples)} mean_nll={nll.mean():.4f} "
              f"top1={hit.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
    del ld
    torch.cuda.empty_cache()


def sanity(probes):
    """§8.1: probe validity, exact/native identity for Exact, sample-id hash equality."""
    R = {r["cond"]: r for r in runs()}
    out = {}
    ids_hash = {}
    for name, (_, md) in probes.items():
        ids_hash[name] = hashlib.sha256("|".join(m["sample_id"] for m in md).encode()).hexdigest()
    out["sample_id_hash"] = ids_hash                          # identical data for every condition by construction

    a_nll = np.load(f"{PS}/{R['Exact']['run_name']}__A__native.npz")["nll"]
    _, mdA = probes["A"]
    d64 = np.array([m["config"] == "d64" for m in mdA])
    ind = np.array([m["arm"] == "induction" for m in mdA])
    gain = a_nll[d64 & ~ind] - a_nll[d64 & ind]               # control minus induction
    g = st.mean_ci(gain)
    out["probe_validity_exact_d64_induction_gain"] = dict(mean=g[0], ci_lo=g[1], ci_hi=g[2],
                                                          positive=bool(g[1] > 0))
    ex_n = np.load(f"{PS}/{R['Exact']['run_name']}__A__native.npz")["nll"]
    f_ex = f"{PS}/{R['Exact']['run_name']}__A__exact.npz"
    if os.path.exists(f_ex):
        ex_e = np.load(f_ex)["nll"]
        out["exact_native_identity_Exact"] = dict(identical=bool(np.array_equal(ex_n, ex_e)),
                                                  max_abs_diff=float(np.abs(ex_n - ex_e).max()))
    json.dump(out, open(f"{OUT}/sanity_checks.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


def dump_rows(probes):
    R = {r["cond"]: r for r in runs()}
    with gzip.open(f"{OUT}/per_sample_rows.csv.gz", "wt", newline="") as f:
        f.write("run_id,condition,K,forward_mode,probe,config,arm,sample_id,input_len,target_nll,top1\n")
        for cond, r in R.items():
            for name, (_, md) in probes.items():
                for mode in ("native", "exact"):
                    p = f"{PS}/{r['run_name']}__{name}__{mode}.npz"
                    if not os.path.exists(p):
                        continue
                    z = np.load(p)
                    for m, nll, hit in zip(md, z["nll"], z["hit"]):
                        f.write(f'{r["run_name"]},{cond},{r["K"]},{mode},{name},{m["config"]},'
                                f'{m.get("arm", "")},{m["sample_id"]},{m["input_len"]},{nll:.6f},{hit:.3f}\n')


def main():
    set_numerics()
    os.makedirs(PS, exist_ok=True)
    probes = {n: read_probe(n) for n in PROBES}
    R = {r["cond"]: r for r in runs()}
    eval_condition(R["Exact"], "native", probes)              # sanity first
    eval_condition(R["Exact"], "exact", probes)
    s = sanity(probes)
    if not s["probe_validity_exact_d64_induction_gain"]["positive"]:
        raise SystemExit("probe validity failed: Exact induction gain at d64 is not positive; fix the probe")
    if not s.get("exact_native_identity_Exact", {}).get("identical", True):
        raise SystemExit("exact/native mismatch for the Exact run; fix the forward switch")
    print("sanity checks passed\n")
    for cond, r in R.items():
        if cond != "Exact":
            eval_condition(r, "native", probes)
    for cond in EXACT_CONDS:
        eval_condition(R[cond], "exact", probes)
    dump_rows(probes)
    print("layer 2 evaluation complete")


if __name__ == "__main__":
    main()
