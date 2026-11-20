#!/usr/bin/env python3
"""O1: frozen-model substitution on the suite-D Exact endpoints (124M@100M WikiText-103, 5 seeds).

All 12 attention layers are replaced at inference by rule x calibration x K; full validation split (243 blocks of
1024), batch 1, bf16 autocast outside the fp32 operator, TF32 off (same evaluator protocol as the paper).  The
baseline is the same model through the same analysis path with exact softmax; it is also compared with the stored
suite-D evaluation of that endpoint.  Resumable: results/o1_substitution/s<seed>/<config>.npy.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
import ops_ref  # noqa: E402
from common import ValBlocks, per_block_nll, sha256  # noqa: E402
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))  # repository root

C = ops_ref.C
TR = KI_REPO
VAL = f"{TR}/training_code/phaseB_base/data/wikitext103/val.bin"
MANIFEST = f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/manifests/five_seed_wikitext_130.tsv"
HIST = f"{TR}/analysis_main_v1/D_robustness_124m_100m_5seed/eval_native_v1"
OUT = os.path.join(C, "results", "o1_substitution")
SEEDS = (7, 42, 253, 1337, 2026)
RULES = ("nearest", "lerp", "centroid", "centroid2", "centroid0", "centroid_int")
CALS = ("minmax", "fw")
KS = (4, 8, 16, 32)


def configs():
    yield "exact", None
    for K in KS:
        for cal in CALS:
            for rule in RULES:
                yield f"{rule}|{cal}|K{K}", (rule, cal, K)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha256(VAL) == "397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979"
    blocks = ValBlocks(VAL, 1024)
    assert blocks.n_blocks == 243
    rows = {int(r["seed"]): r for r in csv.DictReader(open(MANIFEST), delimiter="\t") if r["cond"] == "Exact"}
    only = [int(s) for s in sys.argv[1:]] or SEEDS
    for seed in only:
        ck = rows[seed]["checkpoint_100m"]
        model, cfg = ops_ref.load_gpt(ck)
        od = os.path.join(OUT, f"s{seed}")
        os.makedirs(od, exist_ok=True)
        meta_p = os.path.join(od, "meta.json")
        if not os.path.exists(meta_p):
            json.dump(dict(seed=seed, checkpoint=ck, checkpoint_sha256=sha256(ck),
                           ops_ref_sha256=sha256(ops_ref.__file__),
                           root_quant_sha256=sha256(os.path.join(ops_ref.ROOT, "quant_attention.py")),
                           torch=torch.__version__, gpu=torch.cuda.get_device_name(0)), open(meta_p, "w"), indent=1)
        for name, spec in configs():
            out = os.path.join(od, name.replace("|", "__") + ".npy")
            if os.path.exists(out):
                continue
            if spec is None:
                ops_ref.set_operator(model, None)
            else:
                rule, cal, K = spec
                ops_ref.set_operator(model, lambda att, causal, r=rule, c=cal, k=K:
                                     ops_ref.operator_P(att, causal, r, c, k))
            t0 = time.time()
            nll = per_block_nll(model, blocks, 0, blocks.n_blocks, "cuda", 1)
            np.save(out + ".tmp.npy", nll)
            os.replace(out + ".tmp.npy", out)
            print(f"s{seed} {name} {nll.mean():.6f} {time.time() - t0:.1f}s", flush=True)
        # agreement of the analysis-path exact baseline with the stored suite-D evaluation
        hist = sorted(p for p in os.listdir(f"{HIST}/Exact_s{seed}") if p.startswith("100m_"))
        if hist:
            ref = np.load(f"{HIST}/Exact_s{seed}/{hist[0]}/per_block_nll.npy")
            mine = np.load(os.path.join(od, "exact.npy"))
            json.dump(dict(stored_mean=float(ref.mean()), analysis_path_mean=float(mine.mean()),
                           max_abs_block_diff=float(np.abs(ref - mine).max()),
                           bitwise_equal=bool(np.array_equal(ref, mine))),
                      open(os.path.join(od, "baseline_vs_stored.json"), "w"), indent=1)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
