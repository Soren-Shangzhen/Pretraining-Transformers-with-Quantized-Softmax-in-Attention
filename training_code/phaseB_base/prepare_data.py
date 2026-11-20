"""Tokenize a dataset into train.bin / val.bin (uint16 memmap, GPT-2 BPE).

Examples:
    python prepare_data.py --dataset wikitext103 --out_dir data/wikitext103
    python prepare_data.py --dataset fineweb-edu --num_tokens 2000000000 \
        --out_dir data/fineweb2B

wikitext103 : ~117M train tokens. Fine for debugging and short runs; too small
              for a full 124M-model comparison (heavy repetition past 1 epoch).
fineweb-edu : streams sample-10BT; use --num_tokens to cap (2B recommended
              for the 124M comparison).

Requires: pip install datasets tiktoken numpy tqdm
"""

import argparse
import os

import numpy as np
import tiktoken
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def write_bin(path, token_iter, expected=None):
    buf = []
    total = 0
    arr = None
    with open(path, "wb") as f:
        pbar = tqdm(total=expected, unit="tok", desc=os.path.basename(path))
        for toks in token_iter:
            buf.extend(toks)
            if len(buf) > 8_000_000:
                np.asarray(buf, dtype=np.uint16).tofile(f)
                total += len(buf)
                pbar.update(len(buf))
                buf = []
            if expected is not None and total >= expected:
                break
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)
            total += len(buf)
            pbar.update(len(buf))
        pbar.close()
    print(f"{path}: {total:,} tokens")
    return total


def tokenize_docs(dataset, key="text"):
    for ex in dataset:
        text = ex[key]
        if not text:
            continue
        yield enc.encode_ordinary(text) + [EOT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["wikitext103", "fineweb-edu"],
                    required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_tokens", type=int, default=None,
                    help="cap on train tokens (fineweb-edu)")
    ap.add_argument("--val_tokens", type=int, default=5_000_000)
    args = ap.parse_args()

    from datasets import load_dataset
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        write_bin(os.path.join(args.out_dir, "train.bin"),
                  tokenize_docs(ds["train"]))
        write_bin(os.path.join(args.out_dir, "val.bin"),
                  tokenize_docs(ds["validation"]))
    else:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        it = tokenize_docs(ds)
        write_bin(os.path.join(args.out_dir, "val.bin"), it,
                  expected=args.val_tokens)
        write_bin(os.path.join(args.out_dir, "train.bin"), it,
                  expected=args.num_tokens)


if __name__ == "__main__":
    main()
