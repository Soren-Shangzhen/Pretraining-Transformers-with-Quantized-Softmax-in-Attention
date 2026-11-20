#!/usr/bin/env python3
"""Step 1: the first training records after the branch point, side by side (read from the runs' own logs; nothing is
recomputed).  training_health.csv has one row per optimizer step (--health_every_steps 1): the mean training loss of
the step's 128 micro-batches, computed in the forward of that step, i.e. the first row uses the parent's weights under
the branch's forward.  The two branches of a pair restore the same checkpoint, data position and seed, so the same step
sees the same batch.  log.csv gives the 100-batch validation loss logged every 5M tokens.
Output: ../eval_out/step1_logs.csv (first 40 steps, about 5.2M tokens) and step1_val_log.csv.
"""
from pathlib import Path
import pandas as pd

T = Path("<REPO>")
BR = T / "adaptive_k_interval_training_phaseB/files_adaptive_quant_phaseB/rb-runs_B_20261110"
OUT = Path(__file__).resolve().parents[1] / "eval_out"
PAIRS = [("A250 (2.5B lineage)", "rb-E2c-01", "rb-E2c-03"), ("B50", "rb-E2ab-01", "rb-E2ab-03"), ("B125", "rb-E2ab-06", "rb-E2ab-08")]
N = 40

rows, vals = [], []
for parent, ww, fz in PAIRS:
    h = {r: pd.read_csv(BR / r / "training_health.csv") for r in (ww, fz)}
    for i in range(N):
        a, b = h[ww].iloc[i], h[fz].iloc[i]
        assert a.step == b.step and a.tokens == b.tokens, (parent, i)
        rows.append(dict(parent=parent, step=int(a.step), step_after_branch=i + 1, tokens_m=round(a.tokens / 1e6, 3),
                         train_loss_WtoW=a.train_loss_mean, train_loss_frozen=b.train_loss_mean,
                         train_loss_frozen_minus_WtoW=b.train_loss_mean - a.train_loss_mean,
                         grad_norm_WtoW=a.grad_norm_mean, grad_norm_frozen=b.grad_norm_mean,
                         clip_fraction_WtoW=a.clip_fraction, clip_fraction_frozen=b.clip_fraction,
                         nonfinite_frozen=int(b.nonfinite_grad_updates + b.nonfinite_loss_steps),
                         run_WtoW=ww, run_frozen=fz))
    lg = {r: pd.read_csv(BR / r / "log.csv") for r in (ww, fz)}
    for i in range(4):
        a, b = lg[ww].iloc[i], lg[fz].iloc[i]
        vals.append(dict(parent=parent, step=int(a.step), tokens_m=round(a.tokens / 1e6, 3), val_loss_WtoW=a.val_loss,
                         val_loss_frozen=b.val_loss, train_loss_WtoW=a.train_loss, train_loss_frozen=b.train_loss))
pd.DataFrame(rows).to_csv(OUT / "step1_logs.csv", index=False)
pd.DataFrame(vals).to_csv(OUT / "step1_val_log.csv", index=False)
print(pd.DataFrame(rows).query("parent.str.startswith('A')", engine="python")[["step", "tokens_m", "train_loss_WtoW", "train_loss_frozen", "grad_norm_WtoW", "grad_norm_frozen"]].head(12).to_string(index=False))
print(pd.DataFrame(vals).to_string(index=False))
