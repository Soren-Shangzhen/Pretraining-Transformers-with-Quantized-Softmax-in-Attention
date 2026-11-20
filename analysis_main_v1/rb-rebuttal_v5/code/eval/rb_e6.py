#!/usr/bin/env python3
"""E6, evaluation-setting sensitivity (plan E6; user decision 2026-11-12: run it now, phase 5, on the local RTX 3090
and on machine B).  Six existing Nearest endpoints, each evaluated on its full validation split under:

  numerical settings     ref (TF32 off, batch 1, fp32 scores, native forward of the run's frozen root)
                         tf32_on | batch8 | scores_bf16 (QK^T as a bf16 matmul, then the root's own operator)
                         | 3090 vs 5090 (the same script on both machines)
  same function,         direct      : the paper's strict copy of the root (forward = w_N exactly, Table S6)
  other implementation   twopass128  : full-row max (and min) first, then the row sum of w_N accumulated over
                                       128-column blocks in order
  other function (FWM)   online128   : keys in 128-blocks, grid anchored at the running max m_b, earlier sums
                                       rescaled by exact exp(m_{b-1} - m_b); P_j = T(m_b(j) - s_j) exp(m_b(j) - M) / l
  self-checks            direct_mine : the same patched path as twopass/online with one ordinary row sum; must equal
                                       `direct` bitwise.   online1024 (FWM): one block of 1024 keys; must equal
                                       direct_mine bitwise (validates the copied FWM lookup arithmetic).
The variants replace only the root's _quant_attention for this process; nothing in the roots is edited.  The FWM
lookup in online/direct_mine copies the arithmetic of the root's _relevant_zero_reconstruct line for line
(strict_formal_qrm_v1/quant_attention.py lines 225-337); MinMax weights come from the strict root's own
_adaptive_interval_reconstruct.
Output: <out>/<endpoint>/<setting>/{per_block_nll.npy, summary.json}; skipped if present.
usage: rb_e6.py --endpoint NAME[,NAME] [--settings s1,s2,...] --out DIR
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.environ.get("RB_TRAIN_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))   # the root of this package
sys.path.insert(0, f"{TRAIN}/analysis_main_v1/code")
from common import ValBlocks, build_model, import_code, per_block_nll, sha256  # noqa: E402
sys.path.insert(0, HERE)
from rb_provenance import provenance  # noqa: E402

ROOTS = f"{TRAIN}/analysis_main_v1/supplement_four_runs_20260923/round2/task1/roots"
LOCAL_ST = "<REPO>/training_code/phaseB_base/rb-upload_staging_20261110"
CLOUD = os.environ.get("RB_CLOUD") if os.path.exists(os.environ.get("RB_CLOUD", "/nonexistent")) else f"{LOCAL_ST}/from_private-storage"
LEP = os.environ.get("RB_LOCAL_ENDPOINTS", f"{LOCAL_ST}/endpoints_from_local")
VAL = {"finewebedu_3b": (os.environ.get("RB_DATA_FINEWEBEDU_3B", f"{LOCAL_ST}/from_private-storage/data/finewebedu_3b/val.bin"),
                         "682c9e4492c051ccb5cbde6e9a61099c17aab9f044a6201b901854e9f3943ec5"),
       "wikitext103": (os.environ.get("RB_DATA_WIKITEXT103", "<REPO>/"
                                      "adaptive_k_interval_training_phaseB/files_adaptive_quant_phaseB/data/wikitext103/val.bin"),
                       "397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979")}
A = "finewebedu_3b"
ENDPOINTS = {   # checkpoint, dataset, native root, strict root, kind, recorded native NLL (GPU)
    "A_MM_W4": (f"{CLOUD}/formal_runs_scale_v1/formal_124m_s1337_nearest_k4_fullgrad_2p5b/ckpt_step019074_2500.1m.pt",
                A, "formal_base_v2", "minmax", 4.083115883049418, "RTX 5090"),
    "A_FW_W4": (f"{CLOUD}/vast_formal_extension_v1/formal_runs_extension_v1/formal_124m_s1337_qrm_nearest_k4_tau6_fullgrad_2p5b/"
                "ckpt_step019074_2500.1m.pt", A, "formal_qrm_v1", "fwm", 3.235497679622447, "RTX 5090"),
    "A_FW_P4": (f"{CLOUD}/formal_runs_probste_2p5b_v1/formal_124m_s1337_qrm_nearest_k4_tau6_probste_2p5b/ckpt_step019074_2500.1m.pt",
                A, "formal_qrm_probste_v1", "fwm", 3.2120564816916577, "RTX 5090"),
    "A_FW_W16": (f"{CLOUD}/formal_runs_qrm_k16_2p5b_v1/formal_124m_s1337_qrm_nearest_k16_tau6_fullgrad_2p5b/ckpt_step019074_2500.1m.pt",
                 A, "formal_qrm_v1", "fwm", 3.196657452182692, "RTX 5090"),
    "D_MM_W4_s7": (f"{LEP}/MM_W4_s7_100m.model_only.pt", "wikitext103", "local_phaseB_base", "minmax", 4.361313138969641, "RTX 3090"),
    "D_FW_W4_s7": (f"{LEP}/FW_W4_s7_100m.model_only.pt", "wikitext103", "local_phaseB_rr_qrm", "fwm", 4.169096055835363, "RTX 3090"),
}
BASE_SETTINGS = ["ref", "tf32_on", "batch8", "scores_bf16", "direct", "direct_mine", "twopass128"]
FWM_SETTINGS = ["online1024", "online128"]


def fwm_hard(z, K, tau):
    """w_N of the FWM lookup for z = s - anchor (copied from _relevant_zero_reconstruct, nearest branch)."""
    tau = float(tau)
    h = tau / float(K)
    zc = z.clamp(min=-tau, max=0.0)
    rel = (zc + tau) / h
    r_idx = torch.floor(rel).to(torch.long).clamp(0, K - 1)
    r = r_idx.to(z.dtype)
    b_lo = -tau + r * h
    grid = torch.arange(K + 1, device=z.device, dtype=z.dtype)
    boundary_w = torch.exp(-tau + grid * h)
    e_lo = boundary_w[r_idx]
    e_hi = boundary_w[(r_idx + 1).clamp_max(K)]
    t = ((zc - b_lo) / h).clamp(0.0, 1.0)
    w = torch.where(t >= 0.5, e_hi, e_lo)
    return torch.where(z < -tau, torch.zeros_like(w), w)


def online_probs(att, valid, K, tau, blk):
    B, H, T, _ = att.shape
    neg = att.masked_fill(~valid, -torch.inf)
    anchor = torch.empty_like(att)
    m = torch.full((B, H, T, 1), -torch.inf, device=att.device, dtype=att.dtype)
    l = torch.zeros((B, H, T, 1), device=att.device, dtype=att.dtype)
    wraw = torch.zeros_like(att)
    for b0 in range(0, T, blk):
        sl = slice(b0, min(T, b0 + blk))
        m_new = torch.maximum(m, neg[..., sl].amax(-1, keepdim=True))
        anchor[..., sl] = m_new
        wb = fwm_hard(att[..., sl] - m_new, K, tau).masked_fill(~valid[..., sl], 0.0)
        wraw[..., sl] = wb
        scale = torch.exp(m - m_new)                                  # exact rescaling of what is accumulated
        l = torch.where(torch.isfinite(m), l * scale, l) + wb.sum(-1, keepdim=True)
        m = m_new
    w = wraw * torch.exp(anchor - m)                                  # m is the final row maximum M
    return w / l


def patch(qa, strict_qa, kind, setting):
    cls = qa.QuantCausalSelfAttention
    orig = cls._quant_attention

    def method(self, q, k, v):
        B, H, T, D = q.shape
        with torch.autocast(device_type="cuda", enabled=False):
            if setting == "scores_bf16":
                att = (q.bfloat16() @ k.bfloat16().transpose(-2, -1)).float() / math.sqrt(D)
                assert T == 1024
                eye = torch.eye(T, device=q.device, dtype=torch.float32).expand(B, H, T, T)
                return orig(self, att * math.sqrt(T), eye, v).to(q.dtype)   # root's own operator on bf16 scores
            qf, kf, vf = q.float(), k.float(), v.float()
            att = (qf @ kf.transpose(-2, -1)) / math.sqrt(D)
            causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            valid = causal.view(1, 1, T, T)
            tau = float(getattr(self, "range_tau", 6.0))
            if setting.startswith("online"):
                P = online_probs(att, valid, self.K, tau, int(setting[len("online"):]))
            else:
                if kind == "fwm":
                    w = fwm_hard(att - att.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True), self.K, tau)
                    w = w.masked_fill(~valid, 0.0)
                else:
                    w = strict_qa._adaptive_interval_reconstruct(att, causal, self.K, "nearest", range_grad="full")[0]
                if setting == "twopass128":
                    denom = torch.zeros_like(w[..., :1])
                    for b0 in range(0, T, 128):
                        denom = denom + w[..., b0:b0 + 128].sum(-1, keepdim=True)
                else:
                    denom = w.sum(dim=-1, keepdim=True)
                P = w / denom
            y = P @ vf
        return y.to(q.dtype)
    cls._quant_attention = method


def evaluate(name, setting, out):
    ck_path, ds, root, kind, rec, rec_gpu = ENDPOINTS[name]
    od = os.path.join(out, name, setting)
    if os.path.exists(os.path.join(od, "summary.json")):
        return
    vp, vsha = VAL[ds]
    if sha256(vp) != vsha:
        raise SystemExit(f"{vp}: validation SHA-256 differs")
    blocks = ValBlocks(vp, 1024)
    tf32 = setting == "tf32_on"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    eroot = f"{ROOTS}/{'strict' if setting == 'direct' else 'native'}_{root}"
    mm, qa = import_code(eroot)
    strict_qa = None
    if setting not in ("ref", "tf32_on", "batch8", "direct"):
        if kind == "minmax":
            _, strict_qa = import_code(f"{ROOTS}/strict_{root}")
            mm, qa = import_code(eroot)                  # re-import the native root (import_code evicts modules)
        patch(qa, strict_qa, kind, setting)
    t0 = time.time()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False, mmap=True)
    model, cfg, op = build_model(mm, ck, "cuda")
    nll = per_block_nll(model, blocks, 0, blocks.n_blocks, "cuda", 8 if setting == "batch8" else 1)
    os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "per_block_nll.npy"), nll)
    json.dump(dict(endpoint=name, setting=setting, checkpoint=ck_path, eval_root=eroot,
                   eval_code_sha256={f: sha256(os.path.join(eroot, f)) for f in ("model.py", "quant_attention.py")},
                   dataset=ds, val_sha256=vsha, n_blocks=int(blocks.n_blocks), mean_nll=float(nll.mean()),
                   nonfinite=int((~np.isfinite(nll)).sum()), recorded_native=rec, recorded_gpu=rec_gpu,
                   tf32=tf32, batch_size=8 if setting == "batch8" else 1, operator=op, torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(0), host=socket.gethostname(), seconds=round(time.time() - t0, 1),
                   provenance=provenance(ck_path, eroot, [os.path.abspath(__file__), os.path.join(HERE, "rb_provenance.py"),
                                                          f"{TRAIN}/analysis_main_v1/code/common.py"],
                                         extra_roots=[f"{ROOTS}/strict_{root}"] if strict_qa is not None else [])),
              open(os.path.join(od, "summary.json"), "w"), indent=1)
    print(f"done {name} {setting} nll={nll.mean():.9f} ({time.time() - t0:.0f}s)", flush=True)
    del model, ck
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--settings", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for name in a.endpoint.split(","):
        sets = a.settings.split(",") if a.settings else BASE_SETTINGS + (FWM_SETTINGS if ENDPOINTS[name][3] == "fwm" else [])
        for s in sets:
            evaluate(name, s, a.out)


if __name__ == "__main__":
    main()
