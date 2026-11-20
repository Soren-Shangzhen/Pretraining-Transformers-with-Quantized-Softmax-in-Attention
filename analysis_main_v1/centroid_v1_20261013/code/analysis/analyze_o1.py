#!/usr/bin/env python3
"""O1 summary: Delta NLL (substituted - exact through the same path) per rule x calibration x K on the five suite-D
Exact endpoints; mean and SD over seeds, per-seed values, and paired seed-level differences between rules
(Centroid - Nearest, Centroid-2 - Centroid, Centroid0 - Centroid, Centroid_int - Centroid, LERP - Centroid).
Non-finite NLL (Centroid-2 overflow) is reported as such.  Output: results/o1_substitution/summary.{json,csv}"""
import csv
import json
import os

import numpy as np
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))  # repository root

C = (KI_REPO + "/analysis_main_v1/centroid_v1_20261013")
O = os.path.join(C, "results", "o1_substitution")
SEEDS = (7, 42, 253, 1337, 2026)
RULES = ("nearest", "lerp", "centroid", "centroid2", "centroid0", "centroid_int")
CALS = ("minmax", "fw")
KS = (4, 8, 16, 32)
PAIRS = [("centroid", "nearest"), ("centroid2", "centroid"), ("centroid0", "centroid"), ("centroid_int", "centroid"),
         ("lerp", "centroid")]


def load(seed, name):
    p = os.path.join(O, f"s{seed}", name.replace("|", "__") + ".npy")
    return np.load(p) if os.path.exists(p) else None


def main():
    rows, out = [], {"delta": {}, "pairs": {}, "baseline_vs_stored": {}, "seeds_complete": []}
    for s in SEEDS:
        bp = os.path.join(O, f"s{s}", "baseline_vs_stored.json")
        if os.path.exists(bp):
            out["baseline_vs_stored"][str(s)] = json.load(open(bp))
    for cal in CALS:
        for K in KS:
            for rule in RULES:
                vals = {}
                for s in SEEDS:
                    e, x = load(s, "exact"), load(s, f"{rule}|{cal}|K{K}")
                    if e is not None and x is not None:
                        vals[s] = float(x.mean() - e.mean())
                if not vals:
                    continue
                v = np.array(list(vals.values()))
                fin = np.isfinite(v)
                rec = dict(cal=cal, K=K, rule=rule, n_seeds=len(v), n_nonfinite=int((~fin).sum()),
                           mean=float(v[fin].mean()) if fin.any() else float("nan"),
                           sd=float(v[fin].std(ddof=1)) if fin.sum() > 1 else float("nan"),
                           per_seed=vals)
                out["delta"][f"{rule}|{cal}|K{K}"] = rec
                rows.append({k: rec[k] for k in ("cal", "K", "rule", "n_seeds", "n_nonfinite", "mean", "sd")})
            for a, b in PAIRS:
                da, db = out["delta"].get(f"{a}|{cal}|K{K}"), out["delta"].get(f"{b}|{cal}|K{K}")
                if not da or not db:
                    continue
                common = [s for s in da["per_seed"] if s in db["per_seed"]]
                d = np.array([da["per_seed"][s] - db["per_seed"][s] for s in common])
                fin = np.isfinite(d)
                out["pairs"][f"{a}-{b}|{cal}|K{K}"] = dict(
                    n=int(fin.sum()), mean=float(d[fin].mean()) if fin.any() else float("nan"),
                    min=float(d[fin].min()) if fin.any() else float("nan"),
                    max=float(d[fin].max()) if fin.any() else float("nan"),
                    all_same_sign=bool(fin.any() and (np.all(d[fin] < 0) or np.all(d[fin] > 0))))
    out["seeds_complete"] = [s for s in SEEDS if all(load(s, f"{r}|{c}|K{k}") is not None
                                                      for r in RULES for c in CALS for k in KS)]
    json.dump(out, open(os.path.join(O, "summary.json"), "w"), indent=1)
    with open(os.path.join(O, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['cal']:6s} K{r['K']:<3d} {r['rule']:13s} n={r['n_seeds']} nonfinite={r['n_nonfinite']} "
              f"{r['mean']:+.4f} sd {r['sd']:.4f}")
    print("complete seeds", out["seeds_complete"])


if __name__ == "__main__":
    main()
