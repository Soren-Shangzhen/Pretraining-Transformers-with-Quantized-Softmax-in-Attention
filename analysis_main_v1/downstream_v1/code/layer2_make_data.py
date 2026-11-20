#!/usr/bin/env python3
"""Layer 2 probe generation (§4.2). Fixed seed, generated once, shared by every condition.

Probe A  induction : random sequence with an inserted bigram (A,B); A recurs later, predict B.
                     Matched no-induction control (same sequence, earlier (A,B) replaced by fillers).
                     Distances {64,128,256,512,896} are a matched family: same token content, distance varies.
Probe B  key-value : "k1:v1 k2:v2 ... kn:vn" then a query for k_i, predict v_i.
                     n in {16,32,64}; queried key near (last quarter) or far (first quarter).
Probe C  prefix copy: random segment S, "S <sep> S", score the second copy per token.

Token pool (§4.2 B): alphabetic tiktoken tokens of moderate frequency, unique decoded string, no byte-level or
control tokens. Saved with the data.

Output: data/probes/{token_pool.json, probe_A.npz, probe_B.npz, probe_C.npz, probes_meta.json}
Each npz: tokens (int32 flat), offsets, targets (int32 flat target positions), target_offsets, plus a meta CSV.
"""
import hashlib
import json
import os
import re

import numpy as np
import tiktoken
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


OUT = KI_REPO + "/analysis_main_v1/downstream_v1/data/probes"
SEED = 20260920
BLOCK = 1024
DISTANCES = [64, 128, 256, 512, 896]
N_A = 400
N_KV = 400
N_COPY = 300
enc = tiktoken.get_encoding("gpt2")


def build_pool():
    """Alphabetic, unique, moderate-frequency tokens. GPT-2 ids are roughly frequency ordered, so ids in
    [1000, 20000] avoid both the byte-level/punctuation head and the rare tail."""
    pat = re.compile(r"^ ?[A-Za-z]{3,10}$")
    seen, pool = set(), []
    for tid in range(1000, 20000):
        try:
            s = enc.decode([tid])
        except Exception:                                   # noqa: BLE001
            continue
        if not pat.match(s):
            continue
        key = s.strip().lower()
        if key in seen:
            continue
        if enc.encode_ordinary(s) != [tid]:                 # must re-tokenize to itself
            continue
        seen.add(key)
        pool.append(tid)
    return pool


def save(name, seqs, targets, meta_rows, extra):
    os.makedirs(OUT, exist_ok=True)
    tok = np.concatenate([np.asarray(s, dtype=np.int32) for s in seqs])
    off = np.cumsum([0] + [len(s) for s in seqs]).astype(np.int64)
    tgt = np.concatenate([np.asarray(t, dtype=np.int32) for t in targets])
    toff = np.cumsum([0] + [len(t) for t in targets]).astype(np.int64)
    np.savez_compressed(f"{OUT}/{name}.npz", tokens=tok, offsets=off, targets=tgt, target_offsets=toff)
    hdr = list(meta_rows[0].keys())
    with open(f"{OUT}/{name}_meta.csv", "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in meta_rows:
            f.write(",".join(str(r[h]) for h in hdr) + "\n")
    h = hashlib.sha256(tok.tobytes() + tgt.tobytes()).hexdigest()
    print(f"{name}: {len(seqs)} samples, {len(tok):,} tokens, sha {h[:16]}")
    return dict(name=name, n_samples=len(seqs), n_tokens=int(len(tok)), sha256=h, seed=SEED, **extra)


def probe_A(pool, rng):
    """Each sample id is a matched family: identical (A,B) and filler content, evaluated at every distance,
    with and without the earlier induction occurrence."""
    seqs, targets, meta = [], [], []
    for i in range(N_A):
        a, b = rng.choice(pool, size=2, replace=False)
        filler_pool = [t for t in pool if t != a and t != b]
        base = rng.choice(filler_pool, size=BLOCK, replace=True).astype(np.int32)
        q = BLOCK - 2                                       # trigger A here, target B at BLOCK-1
        for d in DISTANCES:
            for induction in (True, False):
                s = base.copy()
                s[q] = a
                s[q + 1] = b
                p = q - d
                if induction:
                    s[p] = a
                    s[p + 1] = b
                else:                                       # matched control: fillers, never A or B
                    s[p], s[p + 1] = rng.choice(filler_pool, size=2, replace=False)
                seqs.append(s.tolist())
                targets.append([BLOCK - 1])
                meta.append(dict(sample_id=f"A{i:04d}_d{d}_{'ind' if induction else 'ctl'}", family=f"A{i:04d}",
                                 probe="A", config=f"d{d}", distance=d,
                                 arm="induction" if induction else "control",
                                 target_token=int(b), input_len=BLOCK))
    return save("probe_A", seqs, targets, meta, dict(distances=DISTANCES, families=N_A))


def probe_B(pool, rng):
    colon = enc.encode_ordinary(":")
    query_head = enc.encode_ordinary("\nQ:")
    arrow = enc.encode_ordinary(" ->")
    seqs, targets, meta = [], [], []
    for n in (16, 32, 64):
        for where in ("near", "far"):
            for i in range(N_KV):
                ids = rng.choice(pool, size=2 * n, replace=False)
                keys, vals = ids[:n].tolist(), ids[n:].tolist()
                lo, hi = (int(0.75 * n), n) if where == "near" else (0, max(int(0.25 * n), 1))
                qi = int(rng.integers(lo, hi))
                seq = []
                for k, v in zip(keys, vals):
                    seq += [k] + colon + [v]
                seq += query_head + [keys[qi]] + arrow + [vals[qi]]
                if len(seq) > BLOCK:
                    seq = seq[-BLOCK:]
                seqs.append(seq)
                targets.append([len(seq) - 1])
                meta.append(dict(sample_id=f"B_n{n}_{where}_{i:04d}", family=f"B_n{n}_{where}_{i:04d}",
                                 probe="B", config=f"n{n}_{where}", n_pairs=n, position=where,
                                 queried_index=qi, target_token=int(vals[qi]), input_len=len(seq)))
    return save("probe_B", seqs, targets, meta, dict(n_pairs=[16, 32, 64], positions=["near", "far"]))


def probe_C(pool, rng):
    sep = enc.encode_ordinary("\n")                          # separator uses existing tokenizer tokens only
    seqs, targets, meta = [], [], []
    for L in (32, 64, 128):
        for i in range(N_COPY):
            S = rng.choice(pool, size=L, replace=True).astype(np.int32).tolist()
            seq = S + sep + S
            start = len(S) + len(sep)
            seqs.append(seq)
            targets.append(list(range(start, len(seq))))     # every token of the second copy
            meta.append(dict(sample_id=f"C_L{L}_{i:04d}", family=f"C_L{L}_{i:04d}", probe="C", config=f"L{L}",
                             seg_len=L, target_token=-1, input_len=len(seq)))
    return save("probe_C", seqs, targets, meta, dict(segment_lengths=[32, 64, 128]))


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    pool = build_pool()
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(n=len(pool), ids=pool, criteria="^ ?[A-Za-z]{3,10}$, unique lowercase form, id in "
                                                   "[1000,20000), re-tokenizes to itself, no byte/control tokens",
                   sha256=hashlib.sha256(np.asarray(pool, dtype=np.int32).tobytes()).hexdigest()),
              open(f"{OUT}/token_pool.json", "w"))
    print(f"token pool: {len(pool)} tokens")
    metas = [probe_A(pool, rng), probe_B(pool, rng), probe_C(pool, rng)]
    json.dump(dict(seed=SEED, tokenizer="tiktoken:gpt2", pool_size=len(pool), probes=metas),
              open(f"{OUT}/probes_meta.json", "w"), indent=2)
