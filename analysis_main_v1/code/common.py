"""Shared helpers: build a GPT from a checkpoint using a given code root, and read validation blocks."""
import hashlib
import importlib
import os
import sys

import numpy as np
import torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


DATA = {
    "finewebedu_3b": KI_CLOUD + "/data/finewebedu_3b/val.bin",
    "wikitext103": KI_CLOUD + "/data/wikitext103/val.bin",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_code(code_root):
    """Import model/quant_attention from code_root, evicting any previously imported copies."""
    for m in ("model", "quant_attention"):
        sys.modules.pop(m, None)
    sys.path.insert(0, code_root)
    try:
        model_mod = importlib.import_module("model")
        qa_mod = importlib.import_module("quant_attention")
    finally:
        sys.path.remove(code_root)
    return model_mod, qa_mod


def operator_from_config(c):
    return dict(
        attn=c.get("attn", c.get("attn_mode", "softmax")),
        K=int(c.get("K", c.get("attn_K", 8))),
        approx=c.get("approx", c.get("attn_approx", "lerp")),
        range_grad=c.get("range_grad", c.get("attn_range_grad", "detach")),
        range_mode=c.get("range_mode", c.get("attn_range_mode")) or "minmax",
        range_tau=float(c.get("range_tau", c.get("attn_range_tau")) or 6.0),
    )


def build_model(model_mod, ck, device):
    c = ck["config"]
    op = operator_from_config(c)
    kwargs = dict(block_size=int(c["block_size"]), n_layer=int(c["n_layer"]), n_head=int(c["n_head"]),
                  n_embd=int(c["n_embd"]), attn_mode=op["attn"], attn_K=op["K"], attn_approx=op["approx"],
                  attn_range_grad=op["range_grad"])
    fields = getattr(model_mod.GPTConfig, "__dataclass_fields__", {})
    if "attn_range_grad" not in fields:          # legacy exact_fp32 code (pre Phase B): calibration always detached
        if op["attn"] == "quant" and op["range_grad"] != "detach":
            raise RuntimeError("legacy code root (detached calibration only) cannot build range_grad="
                               f"{op['range_grad']}")
        kwargs.pop("attn_range_grad")
    if "attn_range_mode" in fields:
        kwargs["attn_range_mode"] = op["range_mode"]
        kwargs["attn_range_tau"] = op["range_tau"]
    elif op["range_mode"] != "minmax":
        raise RuntimeError("code root does not support relevant_zero but checkpoint requires it")
    cfg = model_mod.GPTConfig(**kwargs)
    model = model_mod.GPT(cfg)
    sd = ck["model"]
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    model.grad_checkpointing = False
    return model, cfg, op


class ValBlocks:
    """Non-overlapping blocks: x = data[i*T : i*T+T], y = data[i*T+1 : i*T+T+1]. Same as eval_fbmatch_native.py."""

    def __init__(self, path, T):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.T = T
        self.n_blocks = (len(self.data) - 1) // T

    def batch(self, start, end, device):
        T = self.T
        xs = np.stack([np.asarray(self.data[i * T:i * T + T], dtype=np.int64) for i in range(start, end)])
        ys = np.stack([np.asarray(self.data[i * T + 1:i * T + T + 1], dtype=np.int64) for i in range(start, end)])
        return torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)


@torch.inference_mode()
def per_block_nll(model, blocks, start, end, device, batch_size):
    out = []
    for s in range(start, end, batch_size):
        e = min(end, s + batch_size)
        x, y = blocks.batch(s, e, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        ce = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                               reduction="none").view(e - s, -1).mean(dim=1)
        out.append(ce.double().cpu())
        del logits, ce
    return torch.cat(out).numpy()
