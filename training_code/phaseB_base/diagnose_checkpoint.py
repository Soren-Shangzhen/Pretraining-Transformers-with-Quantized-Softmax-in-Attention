"""Run attention diagnostics on an existing checkpoint without retraining.

Examples:
  python diagnose_checkpoint.py \
    --ckpt runs/exact_fp32_softmax_100m/ckpt.pt \
    --data_dir data/wikitext103 --batches 4 --device cuda

  python diagnose_checkpoint.py \
    --ckpt runs/q4_nearest_100m/ckpt.pt \
    --data_dir data/wikitext103 --batches 1 --device cuda

Older checkpoints from previous code revisions are supported because diagnostics
add no model parameters; only the saved model/config are required.
"""

import argparse
import json
import os

import numpy as np
import torch

from model import GPT, GPTConfig
from train import BinReader, evaluate


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--batches", type=int, default=1,
                    help="number of fixed validation batches to evaluate and diagnose")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_json", default=None)
    return ap.parse_args()


def main():
    args = get_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    c = ck.get("config", {})
    attn_mode = c.get("attn", c.get("attn_mode", "softmax"))
    K = int(c.get("K", c.get("attn_K", 8)))
    approx = c.get("approx", c.get("attn_approx", "lerp"))
    range_grad = c.get("range_grad", c.get("attn_range_grad", "detach"))

    cfg = GPTConfig(
        block_size=int(c.get("block_size", 1024)),
        n_layer=int(c.get("n_layer", 12)),
        n_head=int(c.get("n_head", 12)),
        n_embd=int(c.get("n_embd", 768)),
        attn_mode=attn_mode,
        attn_K=K,
        attn_approx=approx,
        attn_range_grad=range_grad,
    )
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.grad_checkpointing = False

    seed = int(c.get("seed", 1337))
    micro_batch = int(c.get("micro_batch", 1))
    val_reader = BinReader(os.path.join(args.data_dir, "val.bin"),
                           cfg.block_size, micro_batch, seed + 1)
    # Match training evaluation sampling.
    val_reader.rng = np.random.default_rng(seed + 1)

    use_bf16 = device_type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16 else torch.autocast(device_type=device_type,
                                                    enabled=False))
    quant_K = K if attn_mode == "quant" else None
    val_loss, stats = evaluate(model, val_reader, args.batches, device,
                               autocast_ctx, stats_batches=args.batches,
                               quant_K=quant_K)

    result = {
        "checkpoint": args.ckpt,
        "tokens_done": int(ck.get("tokens_done", -1)),
        "step": int(ck.get("step", -1)),
        "attn": attn_mode,
        "K": K if attn_mode == "quant" else None,
        "approx": approx if attn_mode == "quant" else None,
        "range_grad": range_grad if attn_mode == "quant" else None,
        "val_loss": float(val_loss),
    }
    if stats is not None:
        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                result[key] = [float(x) for x in value.tolist()]
            elif value is not None:
                result[key] = float(value)
            else:
                result[key] = None

    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
