"""Shared statistics for all three downstream layers (§2 of the instructions).

Every comparison is paired on identical examples in identical order: differences are formed per example and the
bootstrap runs on the differences. 4000 resamples, 95% percentile intervals.

Bootstrap kind by data structure (§2.2):
  examples : paired bootstrap over independent examples (probes, multiple choice)
  blocks   : circular moving-block bootstrap, block length 16 (continuous token streams)
  clusters : document-cluster bootstrap (C4: all blocks of a document resample together)
"""
from __future__ import annotations

import numpy as np

N_BOOT = 4000
BLOCK_LEN = 16
SEED = 20260918
EQUIV_NATS = 0.01          # §2.6 pre-declared practical equivalence margin for NLL-type metrics


def _idx_examples(n, rng, n_boot):
    return rng.integers(0, n, size=(n_boot, n))


def _idx_blocks(n, rng, n_boot, L=BLOCK_LEN):
    starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n / L))))
    return ((starts[:, :, None] + np.arange(L)) % n).reshape(n_boot, -1)[:, :n]


def _idx_clusters(clusters, rng, n_boot):
    """clusters: array of cluster ids per element. Resample clusters with replacement, keep all their elements."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        out.append(np.concatenate([members[p] for p in pick]))
    return out


def boot_indices(n, kind="examples", clusters=None, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    if kind == "examples":
        return _idx_examples(n, rng, n_boot)
    if kind == "blocks":
        return _idx_blocks(n, rng, n_boot)
    if kind == "clusters":
        if clusters is None:
            raise ValueError("clusters required")
        return _idx_clusters(np.asarray(clusters), rng, n_boot)
    raise ValueError(kind)


def paired(a, b, kind="examples", clusters=None, n_boot=N_BOOT, seed=SEED):
    """Paired difference a - b with a bootstrap CI. a, b are per-example (or per-block) scores, same order."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"unpaired shapes {a.shape} vs {b.shape}")
    d = a - b
    idx = boot_indices(len(d), kind, clusters, n_boot, seed)
    means = np.array([d[i].mean() for i in idx]) if isinstance(idx, list) else d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(n=int(len(d)), delta=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                p_value=boot_p(means), detectable=bool(lo > 0 or hi < 0),
                frac_negative=float((d < 0).mean()))


def mean_ci(x, kind="examples", clusters=None, n_boot=N_BOOT, seed=SEED):
    x = np.asarray(x, dtype=np.float64)
    idx = boot_indices(len(x), kind, clusters, n_boot, seed)
    means = np.array([x[i].mean() for i in idx]) if isinstance(idx, list) else x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def boot_p(boot_means):
    """Two-sided bootstrap p-value for H0: mean difference = 0 (used only for BH correction, §2.7)."""
    b = np.asarray(boot_means)
    p = 2.0 * min((b <= 0).mean(), (b >= 0).mean())
    return float(min(max(p, 1.0 / (len(b) + 1)), 1.0))


def bh(pvals):
    """Benjamini-Hochberg q-values for one confirmatory family (§2.7)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def equivalence(ci_lo, ci_hi, margin=EQUIV_NATS):
    """§2.6 decision rule for NLL-type metrics."""
    if ci_lo > 0 or ci_hi < 0:
        return "detectable difference"
    if ci_lo >= -margin and ci_hi <= margin:
        return "within pre-declared equivalence margin"
    return "no detectable difference (not equivalence)"


def split_half_noise(scores, n_rep=20, q=0.90, seed=SEED, clusters=None):
    """§2.5 descriptive noise scale: |half A mean - half B mean| over random halves, 90th percentile of n_rep."""
    x = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    diffs = []
    if clusters is not None:
        cl = np.asarray(clusters)
        uniq = np.unique(cl)
        for _ in range(n_rep):
            perm = rng.permutation(uniq)
            half = set(perm[: len(perm) // 2].tolist())
            m = np.array([c in half for c in cl])
            diffs.append(abs(x[m].mean() - x[~m].mean()))
    else:
        n = len(x)
        for _ in range(n_rep):
            perm = rng.permutation(n)
            a, b = perm[: n // 2], perm[n // 2:]
            diffs.append(abs(x[a].mean() - x[b].mean()))
    return float(np.percentile(diffs, q * 100))
