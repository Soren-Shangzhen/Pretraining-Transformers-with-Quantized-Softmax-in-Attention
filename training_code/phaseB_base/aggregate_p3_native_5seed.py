#!/usr/bin/env python3

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEEDS = [1337, 42, 7, 253, 2026]
KS = [4, 8, 16, 32, 64]

SOURCES = {
    1337: Path("k64_results/posthoc_11x11/matrix/matrix_summary.json"),
    42:   Path("p3_results_v9/seed42/matrix/matrix_summary.json"),
    7:    Path("p3_results_v9/seed7/matrix/matrix_summary.json"),
    253:  Path("p3_results_v9/seed253/matrix/matrix_summary.json"),
    2026: Path("p3_results_v9/seed2026/matrix/matrix_summary.json"),
}

OUT = Path("p3_results_v9/five_seed_native")
OUT.mkdir(parents=True, exist_ok=True)


def mean_sd(xs):
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1))


def condition(rule, k):
    return f"{rule}_K{k}"


# ----------------------------------------------------------------------
# Load source matrices
# ----------------------------------------------------------------------

matrices = {}

for seed in SEEDS:
    p = SOURCES[seed]

    if not p.is_file():
        raise FileNotFoundError(
            f"seed {seed}: missing source matrix: {p}"
        )

    with p.open() as f:
        matrices[seed] = json.load(f)

    if "mean_nll" not in matrices[seed]:
        raise RuntimeError(
            f"seed {seed}: mean_nll missing from {p}"
        )

    print(f"seed {seed}: {p}")


# ----------------------------------------------------------------------
# Per-seed native endpoints
# ----------------------------------------------------------------------

by_seed = []

for seed in SEEDS:
    ms = matrices[seed]["mean_nll"]

    exact = float(ms["exact"]["exact"])

    by_seed.append({
        "seed": seed,
        "rule": "exact",
        "K": "",
        "condition": "exact",
        "native_nll": exact,
        "delta_nll_vs_exact": 0.0,
        "ppl_ratio_vs_exact": 1.0,
        "source": str(SOURCES[seed]),
    })

    for rule in ["lerp", "nearest"]:
        for k in KS:
            c = condition(rule, k)

            native = float(ms[c][c])
            delta = native - exact

            by_seed.append({
                "seed": seed,
                "rule": rule,
                "K": k,
                "condition": c,
                "native_nll": native,
                "delta_nll_vs_exact": delta,
                "ppl_ratio_vs_exact": math.exp(delta),
                "source": str(SOURCES[seed]),
            })


with (OUT / "native_by_seed.csv").open("w", newline="") as f:
    fields = [
        "seed",
        "rule",
        "K",
        "condition",
        "native_nll",
        "delta_nll_vs_exact",
        "ppl_ratio_vs_exact",
        "source",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(by_seed)


# ----------------------------------------------------------------------
# Aggregate across TRAINING SEEDS
# ----------------------------------------------------------------------

summary = []

# Exact
exact_vals = [
    r["native_nll"]
    for r in by_seed
    if r["rule"] == "exact"
]

m, sd = mean_sd(exact_vals)

summary.append({
    "rule": "exact",
    "K": "",
    "n_seeds": len(exact_vals),
    "native_nll_mean": m,
    "native_nll_sd": sd,
    "delta_nll_mean": 0.0,
    "delta_nll_sd": 0.0,
    "ppl_ratio_mean": 1.0,
    "ppl_ratio_sd": 0.0,
})


for rule in ["lerp", "nearest"]:
    for k in KS:
        rows = [
            r for r in by_seed
            if r["rule"] == rule and r["K"] == k
        ]

        nll = [r["native_nll"] for r in rows]
        delta = [r["delta_nll_vs_exact"] for r in rows]
        ppl = [r["ppl_ratio_vs_exact"] for r in rows]

        nll_m, nll_sd = mean_sd(nll)
        d_m, d_sd = mean_sd(delta)
        p_m, p_sd = mean_sd(ppl)

        summary.append({
            "rule": rule,
            "K": k,
            "n_seeds": len(rows),
            "native_nll_mean": nll_m,
            "native_nll_sd": nll_sd,
            "delta_nll_mean": d_m,
            "delta_nll_sd": d_sd,
            "ppl_ratio_mean": p_m,
            "ppl_ratio_sd": p_sd,
        })


with (OUT / "native_summary.csv").open("w", newline="") as f:
    fields = list(summary[0].keys())
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary)


# ----------------------------------------------------------------------
# Matched Nearest - LERP gaps
# ----------------------------------------------------------------------

gap_rows = []

for seed in SEEDS:
    ms = matrices[seed]["mean_nll"]

    for k in KS:
        l = float(ms[f"lerp_K{k}"][f"lerp_K{k}"])
        n = float(ms[f"nearest_K{k}"][f"nearest_K{k}"])

        gap_rows.append({
            "seed": seed,
            "K": k,
            "nearest_minus_lerp": n - l,
        })


with (OUT / "matched_gaps_by_seed.csv").open("w", newline="") as f:
    fields = ["seed", "K", "nearest_minus_lerp"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(gap_rows)


gap_summary = []

for k in KS:
    vals = [
        r["nearest_minus_lerp"]
        for r in gap_rows
        if r["K"] == k
    ]

    m, sd = mean_sd(vals)

    gap_summary.append({
        "K": k,
        "n_seeds": len(vals),
        "mean": m,
        "sd": sd,
        "positive_seeds": sum(x > 0 for x in vals),
    })


with (OUT / "matched_gaps_summary.csv").open("w", newline="") as f:
    fields = list(gap_summary[0].keys())
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(gap_summary)


# ----------------------------------------------------------------------
# Ordering audit
# ----------------------------------------------------------------------

audit = {
    "seeds": SEEDS,
    "tests": {},
}

nearest_monotone = {}
lerp_to_k32_monotone = {}
lerp_k64_better_k32 = {}

for seed in SEEDS:
    ms = matrices[seed]["mean_nll"]

    nearest = [
        float(ms[f"nearest_K{k}"][f"nearest_K{k}"])
        for k in KS
    ]

    # NLL should strictly decrease as K gets finer.
    nearest_monotone[str(seed)] = all(
        nearest[i] > nearest[i + 1]
        for i in range(len(nearest) - 1)
    )

    lerp_4_32 = [
        float(ms[f"lerp_K{k}"][f"lerp_K{k}"])
        for k in [4, 8, 16, 32]
    ]

    lerp_to_k32_monotone[str(seed)] = all(
        lerp_4_32[i] > lerp_4_32[i + 1]
        for i in range(len(lerp_4_32) - 1)
    )

    l32 = float(ms["lerp_K32"]["lerp_K32"])
    l64 = float(ms["lerp_K64"]["lerp_K64"])

    lerp_k64_better_k32[str(seed)] = (l64 < l32)


audit["tests"]["nearest_strict_K4_to_K64"] = {
    "per_seed": nearest_monotone,
    "count": sum(nearest_monotone.values()),
    "total": len(SEEDS),
}

audit["tests"]["lerp_strict_K4_to_K32"] = {
    "per_seed": lerp_to_k32_monotone,
    "count": sum(lerp_to_k32_monotone.values()),
    "total": len(SEEDS),
}

audit["tests"]["lerp_K64_better_than_K32"] = {
    "per_seed": lerp_k64_better_k32,
    "count": sum(lerp_k64_better_k32.values()),
    "total": len(SEEDS),
}

audit["tests"]["lerp_better_than_nearest_matched"] = {
    "positive_comparisons": sum(
        r["nearest_minus_lerp"] > 0
        for r in gap_rows
    ),
    "total_comparisons": len(gap_rows),
}

with (OUT / "ordering_audit.json").open("w") as f:
    json.dump(audit, f, indent=2)


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

provenance = {
    "description": (
        "Five-training-seed native-quality aggregate. "
        "Seed1337 is the legacy discovery/formal run and is read from "
        "k64_results/posthoc_11x11; seeds 42,7,253,2026 use the later "
        "P3 v9 directory layout. No validation blocks are treated as "
        "independent training replicates; aggregate SD is across seeds."
    ),
    "seeds": SEEDS,
    "matrix_sources": {
        str(seed): str(SOURCES[seed])
        for seed in SEEDS
    },
}

with (OUT / "PROVENANCE.json").open("w") as f:
    json.dump(provenance, f, indent=2)


# ----------------------------------------------------------------------
# Checksums
# ----------------------------------------------------------------------

files = [
    "native_by_seed.csv",
    "native_summary.csv",
    "matched_gaps_by_seed.csv",
    "matched_gaps_summary.csv",
    "ordering_audit.json",
    "PROVENANCE.json",
]

with (OUT / "SHA256SUMS").open("w") as f:
    for name in files:
        data = (OUT / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        f.write(f"{digest}  {name}\n")


# ----------------------------------------------------------------------
# Human-readable output
# ----------------------------------------------------------------------

print()
print("=" * 78)
print("FIVE-SEED NATIVE QUALITY SUMMARY")
print("=" * 78)

for r in summary:
    if r["rule"] == "exact":
        print(
            f"Exact       "
            f"{r['native_nll_mean']:.6f} ± "
            f"{r['native_nll_sd']:.6f}"
        )
    else:
        print(
            f"{r['rule']:7s} K{int(r['K']):2d}  "
            f"NLL={r['native_nll_mean']:.6f} ± "
            f"{r['native_nll_sd']:.6f}  "
            f"Δ={r['delta_nll_mean']:+.6f} ± "
            f"{r['delta_nll_sd']:.6f}"
        )

print()
print("Matched Nearest - LERP:")
for r in gap_summary:
    print(
        f"K{r['K']:2d}: "
        f"{r['mean']:+.6f} ± {r['sd']:.6f}  "
        f"positive={r['positive_seeds']}/{r['n_seeds']}"
    )

print()
print(
    "LERP < Nearest matched comparisons: "
    f"{audit['tests']['lerp_better_than_nearest_matched']['positive_comparisons']}"
    f"/{audit['tests']['lerp_better_than_nearest_matched']['total_comparisons']}"
)

print(
    "Nearest strict K4→K64: "
    f"{audit['tests']['nearest_strict_K4_to_K64']['count']}/"
    f"{audit['tests']['nearest_strict_K4_to_K64']['total']}"
)

print(
    "LERP strict K4→K32: "
    f"{audit['tests']['lerp_strict_K4_to_K32']['count']}/"
    f"{audit['tests']['lerp_strict_K4_to_K32']['total']}"
)

print(
    "LERP K64 better than K32: "
    f"{audit['tests']['lerp_K64_better_than_K32']['count']}/"
    f"{audit['tests']['lerp_K64_better_than_K32']['total']}"
)

print()
print("saved:", OUT)
