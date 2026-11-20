"""Load a 124M @ 2.5B run's final checkpoint with its own frozen training operator.

Two forward modes (§0.3 of the instructions):
  native : the run's own training operator (attn/K/approx/range_mode/tau/range_grad from the checkpoint config)
  exact  : identical weights, forward replaced by the explicit fp32 exact softmax of the same frozen code root

Numerics are fixed to the main-result protocol: batch 1, bf16 autocast outside the fp32 attention kernel,
TF32 off, context 1024, GPT-2 BPE.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


ROOT = KI_CLOUD
MANIFEST = f"{ROOT}/analysis_main_v1/manifests/primary_37_runs.tsv"
SUITE = "A_longhorizon_124m_2p5b"
BLOCK = 1024


def set_numerics():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            read += len(chunk)
            if limit and read >= limit:
                break
    return h.hexdigest()


def runs(manifest=MANIFEST, suite=SUITE):
    out = []
    for r in csv.DictReader(open(manifest), delimiter="\t"):
        if r["suite"] == suite:
            r["cond"] = r["label"] if r["label"] == "Exact" else f'{r["label"]} K{r["K"]}'
            r["final_path"] = os.path.join(r["run_path"], r["final_checkpoint"])
            out.append(r)
    return out


def import_code(code_root):
    for m in ("model", "quant_attention"):
        sys.modules.pop(m, None)
    sys.path.insert(0, code_root)
    try:
        model_mod = importlib.import_module("model")
        qa_mod = importlib.import_module("quant_attention")
    finally:
        sys.path.remove(code_root)
    return model_mod, qa_mod


def operator(c):
    return dict(attn=c.get("attn", c.get("attn_mode", "softmax")), K=int(c.get("K", 8)),
                approx=c.get("approx", "lerp"), range_grad=c.get("range_grad", "detach"),
                range_mode=c.get("range_mode") or "minmax", range_tau=float(c.get("range_tau") or 6.0))


@dataclass
class Loaded:
    model: object
    cfg: object
    op: dict
    run: dict
    forward_mode: str
    ckpt_sha256: str


def assert_identity(run, c, op):
    """§0.4 / §8.1: condition identity comes from the manifest and the checkpoint config, never the directory name."""
    errs = []
    if os.path.basename(os.path.normpath(c["data_dir"])) != run["dataset"]:
        errs.append(f'dataset {c["data_dir"]} != {run["dataset"]}')
    lab = run["label"]
    if lab == "Exact":
        if op["attn"] != "exact_fp32_softmax":
            errs.append(f'attn {op["attn"]}')
    else:
        want_approx = "lerp" if lab == "LERP" else "nearest"
        want_mode = "relevant_zero" if lab.startswith("QRM") else "minmax"
        if op["attn"] != "quant":
            errs.append(f'attn {op["attn"]}')
        if op["approx"] != want_approx:
            errs.append(f'approx {op["approx"]}')
        if str(op["K"]) != run["K"]:
            errs.append(f'K {op["K"]} != {run["K"]}')
        if op["range_mode"] != want_mode:
            errs.append(f'range_mode {op["range_mode"]}')
        if want_mode == "relevant_zero" and abs(op["range_tau"] - 6.0) > 1e-9:
            errs.append(f'range_tau {op["range_tau"]}')
        if op["range_grad"] != "full":
            errs.append(f'range_grad {op["range_grad"]}')
    if errs:
        raise RuntimeError(f'{run["run_name"]}: identity assertion failed: ' + "; ".join(errs))


def load(run, forward_mode="native", device="cuda", ckpt_hash=False):
    set_numerics()
    model_mod, _ = import_code(run["train_code_root"])
    ck = torch.load(run["final_path"], map_location="cpu", weights_only=False, mmap=True)
    c = ck["config"]
    op = operator(c)
    assert_identity(run, c, op)
    if int(ck["step"]) != 19074:
        raise RuntimeError(f'{run["run_name"]}: final step {ck["step"]} != 19074')
    kwargs = dict(block_size=int(c["block_size"]), n_layer=int(c["n_layer"]), n_head=int(c["n_head"]),
                  n_embd=int(c["n_embd"]),
                  attn_mode="exact_fp32_softmax" if forward_mode == "exact" else op["attn"],
                  attn_K=op["K"], attn_approx=op["approx"], attn_range_grad=op["range_grad"])
    fields = getattr(model_mod.GPTConfig, "__dataclass_fields__", {})
    if "attn_range_mode" in fields:
        kwargs["attn_range_mode"] = op["range_mode"]
        kwargs["attn_range_tau"] = op["range_tau"]
    elif op["range_mode"] != "minmax":
        raise RuntimeError("code root lacks relevant_zero support")
    cfg = model_mod.GPTConfig(**kwargs)
    model = model_mod.GPT(cfg)
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    model.grad_checkpointing = False
    h = sha256(run["final_path"]) if ckpt_hash else ""
    del ck
    return Loaded(model=model, cfg=cfg, op=op, run=run, forward_mode=forward_mode, ckpt_sha256=h)


@torch.inference_mode()
def block_nll(model, tokens, device="cuda", block=BLOCK, progress=None):
    """Per-block mean NLL over non-overlapping blocks of a uint16 token stream (batch 1, bf16 autocast)."""
    n_blocks = (len(tokens) - 1) // block
    out = np.empty(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        s = i * block
        x = torch.from_numpy(np.asarray(tokens[s:s + block], dtype=np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(np.asarray(tokens[s + 1:s + block + 1], dtype=np.int64)).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        out[i] = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)).item()
        if progress and (i + 1) % progress == 0:
            print(f"    block {i+1}/{n_blocks}", flush=True)
    return out


@torch.inference_mode()
def sequence_target_nll(model, ctx_ids, target_ids, device="cuda"):
    """Mean NLL of target_ids given ctx_ids (teacher forcing, single sequence, batch 1).

    Returns (mean_nll, per_target_nll, top1_hits) with per-target values in nats.
    """
    ids = list(ctx_ids) + list(target_ids)
    if len(ids) > BLOCK + 1:
        ids = ids[-(BLOCK + 1):]
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    lg = logits.float()[0]
    nll = torch.nn.functional.cross_entropy(lg, y, reduction="none")
    k = len(target_ids)
    per = nll[-k:].double().cpu().numpy()
    hits = (lg[-k:].argmax(-1) == y[-k:]).double().cpu().numpy()
    return float(per.mean()), per, hits


@torch.inference_mode()
def choice_loglik(model, ctx_ids, cont_ids, device="cuda"):
    """Total and per-token log-likelihood of a continuation given a context (lm-eval style)."""
    ids = list(ctx_ids) + list(cont_ids)
    truncated = len(ids) > BLOCK + 1
    if truncated:
        ids = ids[-(BLOCK + 1):]
    n_cont = min(len(cont_ids), len(ids) - 1)
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    lg = logits.float()[0]
    lp = torch.log_softmax(lg, dim=-1)
    tgt = y[-n_cont:]
    tok_lp = lp[-n_cont:].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    greedy = bool((lg[-n_cont:].argmax(-1) == tgt).all().item())
    return float(tok_lp.sum().item()), n_cont, greedy, truncated


def write_manifest(path, extra):
    """EVAL_MANIFEST.json (§7.2)."""
    man = dict(
        manifest_tsv=MANIFEST, manifest_sha256=sha256(MANIFEST), suite=SUITE,
        tokenizer="tiktoken:gpt2 (encode_ordinary, EOT between documents)",
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        dtype="bf16 autocast outside fp32 attention kernel", tf32=False, batch_size=1, context=BLOCK,
        **extra)
    with open(path, "w") as f:
        json.dump(man, f, indent=2)
    return man
