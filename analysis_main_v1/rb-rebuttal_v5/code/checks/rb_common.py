"""Shared helpers for the rb-rebuttal_v5 checks: import a code root in isolation, build models."""
import hashlib, importlib, json, os, sys

# CPU bitwise gates run single-threaded (user decision 2026-11-10; multithreaded CPU training is not reproducible
# run to run, see checks/13_3_cpu_thread_determinism_*.json).  The environment variables are inherited by the
# training subprocesses that the checks launch; torch.set_num_threads covers this process.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch  # noqa: E402
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.abspath(os.path.join(HERE, ".."))
RB_ROOT = os.path.join(RB, "rb-root_v1")
# The root of this package; set RB_TRAIN_ROOT to use another copy.
TRAIN = os.environ.get("RB_TRAIN_ROOT", os.path.abspath(os.path.join(RB, "..", "..", "..")))
SUP = f"{TRAIN}/analysis_main_v1/supplement_four_runs_20260923/code"
ROOTS = {
    "rb": RB_ROOT,
    "e2_fwm": f"{SUP}/e2_fwm_probste_strict",
    "e2_minmax": f"{SUP}/e2_minmax_probste_strict",
    "r3": f"{SUP}/r3_lowprec_softmax",
    "phaseB_base": f"{TRAIN}/analysis_main_v1/D_robustness_124m_100m_5seed/code_snapshots/local_phaseB_base",
    "rb_tail_weight": f"{TRAIN}/training_code/phaseB_base/code/rb_tail_weight",
}
CHECK_OUT = os.path.join(RB, "..", "checks")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_root(root):
    for m in ("model", "quant_attention"):
        sys.modules.pop(m, None)
    sys.path.insert(0, root)
    try:
        mm = importlib.import_module("model")
        qa = importlib.import_module("quant_attention")
    finally:
        sys.path.remove(root)
    return mm, qa


def write_json(name, obj):
    """checks/<name>_<host>.json (one file per machine: the checks directory may be shared)"""
    import socket
    os.makedirs(CHECK_OUT, exist_ok=True)
    stem, ext = os.path.splitext(name)
    p = os.path.abspath(os.path.join(CHECK_OUT, f"{stem}_{socket.gethostname()}{ext}"))
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return p


def env_info():
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "rb_root_sha256": {f: sha256(os.path.join(RB_ROOT, f))
                               for f in ("model.py", "quant_attention.py", "train_formal.py")}}


PHASEB = f"{TRAIN}/training_code/phaseB_base"   # tokenized data and checkpoints are not part of this package
WIKI_VAL = os.environ.get("RB_DATA_WIKITEXT103", f"{PHASEB}/data/wikitext103/val.bin")
WIKI_VAL_SHA = "397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979"
LOCAL_CKPT = {   # local suite-D seed-7 endpoints (RTX 3090 stack); RB_CKPT_<NAME> overrides on a cloud machine
    "softmax_s7_100m": f"{PHASEB}/runs/p3_s7_exact_100m/ckpt_step000763_100.0m.pt",
    "fwm_weight_k4_s7_100m": f"{PHASEB}/qrm_runs_v1/qrm_s7_nearest_k4_tau6_fullgrad_100m/ckpt_step000763_100.0m.pt",
    "minmax_weight_k4_s7_40m": f"{PHASEB}/runs/p3_s7_nearest_k4_fullgrad_100m/ckpt_step000306_40.1m.pt",
}
LOCAL_CKPT = {k: os.environ.get(f"RB_CKPT_{k.upper()}", v) for k, v in LOCAL_CKPT.items()}


@torch.no_grad()
def capture_scores(ckpt_path, n_blocks=1, device="cuda"):
    """fp32 attention scores of every layer for the first n validation blocks, computed exactly as
    in the training module (bf16 autocast outside attention, fp32 QK^T / sqrt(d))."""
    import numpy as np
    mm, qa = import_root(RB_ROOT)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    c = ck["config"]
    cfg = mm.GPTConfig(block_size=int(c["block_size"]), n_layer=int(c["n_layer"]), n_head=int(c["n_head"]),
                       n_embd=int(c["n_embd"]), attn_mode="exact_fp32_softmax")
    model = mm.GPT(cfg)
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    store = []
    orig = qa.QuantCausalSelfAttention._exact_fp32_softmax

    def patched(self, q, k, v):
        import math as _m
        with torch.autocast(device_type=q.device.type, enabled=False):
            att = (q.float() @ k.float().transpose(-2, -1)) / _m.sqrt(q.shape[-1])
        store.append(att.clone())
        return orig(self, q, k, v)
    qa.QuantCausalSelfAttention._exact_fp32_softmax = patched
    try:
        data = np.memmap(WIKI_VAL, dtype=np.uint16, mode="r")
        for b in range(n_blocks):
            x = torch.from_numpy(np.asarray(data[b * 1024:(b + 1) * 1024], dtype=np.int64))[None].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(x)
    finally:
        qa.QuantCausalSelfAttention._exact_fp32_softmax = orig
    return store
