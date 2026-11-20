#!/usr/bin/env python3
"""Layer 1 data preparation: three held-out corpora tokenized exactly like the training data.

GPT-2 BPE via tiktoken, `encode_ordinary`, EOT appended after every non-empty document, uint16
(identical to `prepare_data_wikitext_reconstruction.py` / `prepare_finewebedu_gpt2.py`).

  wikitext103_test : Salesforce/wikitext / wikitext-103-raw-v1, split test, full
  ptb_test         : ptb_text_only, split test, full
  c4_validation    : allenai/c4 en validation, fixed-seed document sample to >= 300 blocks of 1024 tokens,
                     with the block -> document-id map saved for cluster bootstrap (§2.2)

Writes data/layer1/<corpus>.bin, <corpus>_meta.json, and c4_validation_blocks.csv.
"""
import hashlib
import json
import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


OUT = KI_REPO + "/analysis_main_v1/downstream_v1/data/layer1"
BLOCK = 1024
C4_MIN_BLOCKS = 300
C4_SEED = 20260918
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def write(name, token_lists, doc_ids=None):
    os.makedirs(OUT, exist_ok=True)
    toks, block_doc = [], []
    for did, t in zip(doc_ids or range(len(token_lists)), token_lists):
        toks.extend(t)
        block_doc.extend([did] * len(t))
    arr = np.asarray(toks, dtype=np.uint16)
    path = f"{OUT}/{name}.bin"
    arr.tofile(path)
    n_blocks = (len(arr) - 1) // BLOCK
    meta = dict(corpus=name, tokens=int(len(arr)), n_blocks=int(n_blocks), block=BLOCK,
                tokenizer="tiktoken:gpt2 encode_ordinary + EOT per document", dtype="uint16",
                sha256=hashlib.sha256(arr.tobytes()).hexdigest(), documents=len(token_lists))
    if doc_ids is not None:
        rows = []
        for b in range(n_blocks):
            seg = block_doc[b * BLOCK:(b + 1) * BLOCK]
            vals, counts = np.unique(np.asarray(seg), return_counts=True)
            rows.append((b, str(vals[counts.argmax()])))     # majority document of the block
        with open(f"{OUT}/{name}_blocks.csv", "w") as f:
            f.write("block,document_id\n")
            for b, d in rows:
                f.write(f"{b},{d}\n")
        meta["block_document_map"] = f"{name}_blocks.csv"
    json.dump(meta, open(f"{OUT}/{name}_meta.json", "w"), indent=2)
    print(f"{name}: {len(arr):,} tokens, {n_blocks} blocks, {len(token_lists)} docs, sha {meta['sha256'][:16]}")
    return meta


def docs_to_tokens(texts):
    out = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        out.append(enc.encode_ordinary(t) + [EOT])
    return out


def wikitext():
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "".join(ds["text"])                       # preserve original order/format, single stream
    ids = enc.encode_ordinary(text) + [EOT]
    return write("wikitext103_test", [ids])


def ptb():
    """PTB test split. datasets>=5 refuses script-based repos, so try parquet mirrors, then the canonical text file."""
    src = None
    for name, kwargs in [("ptb-text-only/ptb_text_only", dict(name="penn_treebank", split="test")),
                         ("EleutherAI/ptb_text_only", dict(split="test")),
                         ("mattymchen/ptb-text-only", dict(split="test"))]:
        try:
            ds = load_dataset(name, **kwargs)
            text = "\n".join(s.strip() for s in ds["sentence"])
            src = f"huggingface:{name}"
            break
        except Exception as e:                        # noqa: BLE001
            print(f"  ptb via {name} {kwargs}: {type(e).__name__}: {str(e)[:100]}")
            text = None
    if text is None:
        import urllib.request
        url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
        with urllib.request.urlopen(url, timeout=120) as r:
            raw = r.read()
        text = raw.decode("utf-8").strip()
        src = f"url:{url} sha256:{hashlib.sha256(raw).hexdigest()}"
        print(f"  ptb via {src}")
    ids = enc.encode_ordinary(text) + [EOT]
    m = write("ptb_test", [ids])
    m["source"] = src
    json.dump(m, open(f"{OUT}/ptb_test_meta.json", "w"), indent=2)
    return m


def c4():
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    rng = np.random.default_rng(C4_SEED)
    keep_prob = 0.25                                  # fixed-seed thinning of the stream
    toks, ids, total = [], [], 0
    for i, rec in enumerate(ds):
        if rng.random() > keep_prob:
            continue
        t = enc.encode_ordinary(rec["text"].strip()) + [EOT]
        toks.append(t)
        ids.append(f"c4val_{i:07d}")
        total += len(t)
        if total >= (C4_MIN_BLOCKS + 2) * BLOCK:
            break
    return write("c4_validation", toks, ids)


if __name__ == "__main__":
    which = sys.argv[1:] or ["wikitext", "ptb", "c4"]
    metas = {}
    for w in which:
        print(f"== {w}")
        metas[w] = dict(wikitext=wikitext, ptb=ptb, c4=c4)[w]()
    json.dump(metas, open(f"{OUT}/corpora_meta.json", "w"), indent=2)
