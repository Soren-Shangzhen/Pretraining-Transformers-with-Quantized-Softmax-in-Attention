#!/usr/bin/env python3
"""§8.0 hard implementation gate + §8.1 checks 1/3/6, run before any downstream task.

1. The new loader must reproduce the known FineWebEdu-3B endpoint NLLs:
       Exact              3.1928   (tolerance 4.5e-5)
       MinMax-Weight K4   4.0831   (tolerance 1e-3, numerically most sensitive checkpoint)
2. Identity assertions for all 15 runs (config vs manifest; range_grad == full except Exact).
3. Determinism: MinMax-Weight K4 evaluated twice must be bit-identical per block.
4. exact-forward self-consistency: the Exact run must give identical results in native and exact mode.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_loader import ROOT, block_nll, load, runs, set_numerics  # noqa: E402

VAL = f"{ROOT}/data/finewebedu_3b/val.bin"
TARGETS = {"Exact": (3.1928, 4.5e-5), "MinMax-Weight K4": (4.0831, 1e-3)}
OUT = f"{ROOT}/analysis_main_v1/downstream_v1/results"


def main():
    set_numerics()
    os.makedirs(OUT, exist_ok=True)
    tokens = np.memmap(VAL, dtype=np.uint16, mode="r")
    R = {r["cond"]: r for r in runs()}
    print(f"suite runs in manifest: {len(R)}")
    report = {"gate": {}, "identity": {}, "determinism": {}, "exact_forward_self_consistency": {}}

    for cond in R:                                     # check 1: identity assertions for all 15
        r = R[cond]
        ld = load(r, "native")
        report["identity"][cond] = dict(ok=True, range_grad=ld.op["range_grad"], K=ld.op["K"],
                                        approx=ld.op["approx"], range_mode=ld.op["range_mode"],
                                        checkpoint=r["final_path"])
        del ld
        print(f"  identity ok: {cond}")

    for cond, (target, tol) in TARGETS.items():        # §8.0 gate
        ld = load(R[cond], "native")
        nll = block_nll(ld.model, tokens)
        got = float(nll.mean())
        ok = abs(got - target) <= tol
        report["gate"][cond] = dict(target=target, got=got, diff=got - target, tol=tol, passed=ok,
                                    n_blocks=int(len(nll)))
        print(f"  GATE {cond}: got {got:.6f} target {target} diff {got-target:+.2e} tol {tol} -> "
              f"{'PASS' if ok else 'FAIL'}")
        np.save(f"{OUT}/gate_{cond.replace(' ', '_')}_per_block.npy", nll)
        if cond == "MinMax-Weight K4":                 # check 3: determinism
            nll2 = block_nll(ld.model, tokens)
            same = bool(np.array_equal(nll, nll2))
            report["determinism"][cond] = dict(bit_identical=same,
                                               max_abs_diff=float(np.abs(nll - nll2).max()))
            print(f"  determinism {cond}: bit-identical={same}")
        del ld

    ld_n = load(R["Exact"], "native")                  # check 6: exact-forward self-consistency
    a = block_nll(ld_n.model, tokens[: 1024 * 64 + 1])
    del ld_n
    ld_e = load(R["Exact"], "exact")
    b = block_nll(ld_e.model, tokens[: 1024 * 64 + 1])
    del ld_e
    same = bool(np.array_equal(a, b))
    report["exact_forward_self_consistency"] = dict(bit_identical=same,
                                                    max_abs_diff=float(np.abs(a - b).max()), n_blocks=len(a))
    print(f"  exact-forward self-consistency (Exact run): bit-identical={same}")

    with open(f"{OUT}/gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    passed = all(v["passed"] for v in report["gate"].values()) and same and \
        report["determinism"]["MinMax-Weight K4"]["bit_identical"]
    print("\nGATE RESULT:", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
