"""Shared helpers for the 2026-09-23 supplement checks (E1 / E2).

Nothing here re-implements an attention operator.  Every operator value is
produced by the frozen or new code roots themselves, loaded side by side under
unique module names so that roots never share a `quant_attention` module.
"""

import hashlib
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


CLOUD = KI_CLOUD
SUP = f"{CLOUD}/supplement_four_runs_20260923"

ROOTS = {
    # historical frozen roots (read-only)
    "old_phaseB_base": f"{CLOUD}/code/phaseB",            # LERP full/detach, MinMax-Weight
    "old_phaseB_rr": f"{CLOUD}/code/phaseB_rr",           # FWM(QRM)-Weight
    "old_mm_prob": f"{CLOUD}/formal_code_probste_5seeds_100m_v1/formal_nearest_probste_v1",
    "old_fw_prob": f"{CLOUD}/formal_code_probste_5seeds_100m_v1/formal_qrm_probste_v1",
    # new roots
    "new_e1": f"{SUP}/code/e1_lerp_range_grad_single",
    "new_mm_prob": f"{SUP}/code/e2_minmax_probste_strict",
    "new_fw_prob": f"{SUP}/code/e2_fwm_probste_strict",
}

if os.environ.get("SUP_LOCAL") == "1":
    # local mirror of the same code roots (identical SHA-256; see manifest)
    _TR = KI_REPO + ""
    _SN = f"{_TR}/analysis_main_v1/D_robustness_124m_100m_5seed/code_snapshots"
    _SC = f"{_TR}/analysis_main_v1/supplement_four_runs_20260923/code"
    ROOTS = {
        "old_phaseB_base": f"{_SN}/local_phaseB_base",
        "old_phaseB_rr": f"{_SN}/local_phaseB_rr_qrm",
        "old_mm_prob": f"{_SN}/formal_code_probste_5seeds_100m_v1/formal_nearest_probste_v1",
        "old_fw_prob": f"{_SN}/formal_code_probste_5seeds_100m_v1/formal_qrm_probste_v1",
        "new_e1": f"{_SC}/e1_lerp_range_grad_single",
        "new_mm_prob": f"{_SC}/e2_minmax_probste_strict",
        "new_fw_prob": f"{_SC}/e2_fwm_probste_strict",
    }

_LOADED = {}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def root_hashes(root):
    return {fn: sha256_file(os.path.join(root, fn))
            for fn in sorted(os.listdir(root)) if fn.endswith(".py")}


def load_root(key, tag=None, surrogate=None):
    """Load quant_attention.py + model.py of one root under unique names.

    `surrogate` sets the module-level NEAREST_SURROGATE switch of the new E2
    roots ("prob" | "weight_strict"); it must be None for other roots.
    """
    tag = tag or key
    if tag in _LOADED:
        return _LOADED[tag]
    root = ROOTS[key]
    qname, mname = f"qa__{tag}", f"model__{tag}"
    spec = importlib.util.spec_from_file_location(qname, os.path.join(root, "quant_attention.py"))
    qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa)
    if surrogate is not None:
        assert hasattr(qa, "NEAREST_SURROGATE"), key
        qa.NEAREST_SURROGATE = surrogate
    saved = sys.modules.get("quant_attention")
    sys.modules["quant_attention"] = qa
    try:
        spec = importlib.util.spec_from_file_location(mname, os.path.join(root, "model.py"))
        mm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mm)
    finally:
        if saved is None:
            sys.modules.pop("quant_attention", None)
        else:
            sys.modules["quant_attention"] = saved
    assert mm.QuantCausalSelfAttention is qa.QuantCausalSelfAttention
    _LOADED[tag] = (qa, mm, root)
    return _LOADED[tag]


def rng_snapshot():
    st = {"py": random.getstate(), "np": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def rng_restore(st):
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def tensor_sha(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool
                          else t.numpy().tobytes()).hexdigest()


def state_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().contiguous().cpu()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _int_view(t):
    return {torch.float32: torch.int32, torch.float64: torch.int64,
            torch.bfloat16: torch.int16, torch.float16: torch.int16}[t.dtype]


def compare(a, b, mask=None):
    """Element-wise comparison record.

    eq_numeric : torch.equal (does not distinguish +0/-0)
    eq_rawbits : identical IEEE bit patterns
    NaN/Inf are never accepted as equal: finite_a/finite_b must both be True.
    `mask` (bool, broadcastable) restricts the comparison to valid positions.
    """
    if a is None or b is None:
        return {"a_is_none": a is None, "b_is_none": b is None,
                "eq_numeric": a is None and b is None, "eq_rawbits": a is None and b is None}
    rec = {"shape_a": list(a.shape), "shape_b": list(b.shape),
           "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}
    if a.shape != b.shape or a.dtype != b.dtype:
        rec.update(eq_numeric=False, eq_rawbits=False, note="shape/dtype mismatch")
        return rec
    a = a.detach()
    b = b.detach()
    if mask is not None:
        mask = mask.expand_as(a)
        a = a[mask]
        b = b[mask]
    fa = bool(torch.isfinite(a).all())
    fb = bool(torch.isfinite(b).all())
    eqn = bool(torch.equal(a, b))
    eqb = bool(torch.equal(a.view(_int_view(a)), b.view(_int_view(b))))
    diff = (a.double() - b.double())
    diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    mism = int((a.view(_int_view(a)) != b.view(_int_view(b))).sum())
    nb = float(b.double().norm())
    rec.update(
        n=int(a.numel()), finite_a=fa, finite_b=fb,
        eq_numeric=eqn and fa and fb, eq_rawbits=eqb and fa and fb,
        mismatch_count=mism, max_abs=float(diff.abs().max()) if diff.numel() else 0.0,
        rel_l2=(float(diff.norm()) / nb) if nb > 0 else (0.0 if float(diff.norm()) == 0 else float("inf")),
    )
    return rec


class LocalsCapture:
    """Capture locals of named functions at return, via sys.setprofile.

    Read-only: it never alters a value.  Used to obtain the actual `P` and
    `att` tensors computed inside the frozen `_quant_attention` code.
    """

    def __init__(self, func_names, var_names, on_capture):
        self.func_names = set(func_names)
        self.var_names = var_names
        self.on_capture = on_capture
        self._prev = None

    def _prof(self, frame, event, arg):
        if event == "return" and frame.f_code.co_name in self.func_names:
            loc = frame.f_locals
            self.on_capture(frame.f_code.co_name, {v: loc.get(v) for v in self.var_names})

    def __enter__(self):
        self._prev = sys.getprofile()
        sys.setprofile(self._prof)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self._prev)
        return False


def env_record():
    rec = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if torch.cuda.is_available():
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["driver_capability"] = list(torch.cuda.get_device_capability(0))
    return rec


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
