#!/usr/bin/env python3
"""Step 4: per-layer spread of the native MinMax interval width on each parent checkpoint.

First 8 validation blocks of FineWebEdu-3B, batch 1, bf16 autocast outside the fp32 operator, TF32 off, rows i >= 1
(the paper's geometry-probe convention).  The parent is built through rb-root_v1 with its native MinMax forward (the
same construction as the (a') control of e2c_forward_override.py); the operator method is wrapped (as in
code/eval/rb_probe.py) only to read the fp32 scores att = q k^T / sqrt(d) it receives, and is then called unchanged.
Per row i: native width h_i = (max_j s_ij - min_j s_ij) / K over the causal-valid keys j <= i.  Per layer, pooled over
the 8 blocks, 12 heads and 1023 rows: mean, 10/50/90 percentiles and coefficient of variation (population SD / mean);
next to it the branch's h0 for that layer (read from the branch checkpoint config, not recomputed).
Output: ../eval_out/step4_width_probe.csv
"""
import math, os, sys
from pathlib import Path
import pandas as pd
import torch

T = Path("<REPO>")
RB = T / "analysis_main_v1/rb-rebuttal_v5"
sys.path.insert(0, str(RB / "code/eval"))
import rb_eval  # noqa: E402
from common import ValBlocks, import_code  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "eval_out"
FB = T / "adaptive_k_interval_training_phaseB/files_adaptive_quant_phaseB/rb-upload_staging_20261110/from_private-storage"
PARENTS = {"A250": FB / "formal_runs_scale_v1/formal_124m_s1337_nearest_k4_fullgrad_2p5b/ckpt_step001908_250.1m.pt",
           "B50": FB / "vast_formal_250m_v1/formal_runs_fbmatch_v1/formal_124m_s1337_nearest_k4_weightste_250m/ckpt_step000382_50.1m.pt",
           "B125": FB / "vast_formal_250m_v1/formal_runs_fbmatch_v1/formal_124m_s1337_nearest_k4_weightste_250m/ckpt_step000954_125.0m.pt"}
H0 = pd.read_csv(OUT / "h0_from_branch_configs.tsv", sep="\t").set_index("parent")
VAL = FB / "data/finewebedu_3b/val.bin"
DEFAULTS = {"range_mode": "minmax", "range_tau": 6.0, "tail_policy": "zero", "resolved_h0": "", "attn_variant": "",
            "exaq_bits": 3, "ema_decay": 0.999}

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
mm, qa = import_code(str(RB / "code/rb-root_v1"))
qa.NEAREST_SURROGATE = "weight_strict"
STORE = []
orig = qa.QuantCausalSelfAttention._quant_attention


def wrapped(self, q, k, v):
    with torch.autocast(device_type="cuda", enabled=False):
        att = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    Tn = att.shape[-1]
    valid = torch.ones(Tn, Tn, dtype=torch.bool, device=att.device).tril()
    span = att.masked_fill(~valid, -torch.inf).amax(-1) - att.masked_fill(~valid, torch.inf).amin(-1)
    STORE.append((self.layer_idx, (span[..., 1:] / self.K).flatten().float()))
    return orig(self, q, k, v)


qa.QuantCausalSelfAttention._quant_attention = wrapped
blocks = ValBlocks(str(VAL), 1024)
rows = []
for name, path in PARENTS.items():
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    c = dict(ck["config"])
    for k, v in DEFAULTS.items():
        if c.get(k) is None:
            c[k] = v
    model, cfg, op = rb_eval.build_rb_model(mm, {**ck, "config": c}, "cuda")
    assert all(b.attn.range_mode == "minmax" for b in model.transformer.h)
    STORE.clear()
    with torch.inference_mode():
        for i in range(8):
            x, _ = blocks.batch(i, i + 1, "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(x)
    h0 = [float(v) for v in H0.loc[name, "resolved_h0"].split(",")]
    for layer in range(cfg.n_layer):
        w = torch.cat([s for l, s in STORE if l == layer]).double()
        q = torch.quantile(w, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=w.device)).tolist()
        mean, sd = w.mean().item(), w.std(unbiased=False).item()
        rows.append(dict(parent=name, branch=H0.loc[name, "branch"], layer=layer, h0=h0[layer], n_rows=int(w.numel()),
                         width_mean=mean, width_p10=q[0], width_p50=q[1], width_p90=q[2], width_cv=sd / mean,
                         frac_rows_width_gt_2h0=(w > 2 * h0[layer]).double().mean().item(),
                         frac_rows_width_lt_half_h0=(w < 0.5 * h0[layer]).double().mean().item()))
    print(name, "done", flush=True)
    del model, ck
    torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(OUT / "step4_width_probe.csv", index=False)
print(pd.DataFrame(rows).groupby("parent")[["h0", "width_mean", "width_p10", "width_p50", "width_p90", "width_cv"]].mean())
