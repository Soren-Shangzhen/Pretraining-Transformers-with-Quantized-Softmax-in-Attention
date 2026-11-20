#!/usr/bin/env python3
"""Gradient of torch.clamp exactly at its bounds (user rule 2026-11-10 for E11).

The FWM detach runs (E11) keep the argmax key's own-score term gamma*Delta only if clamp passes a gradient
of 1 at its bounds (torch 2.5.1, 2.12.0 and, by the user's CPU test, 2.13.0; torch 2.14.1 passes 0).  E11 is
queued on a machine only if this prints ok=true for CPU and every visible GPU.  Exit code 0 iff ok.
"""
import json, sys, torch

res = {"torch": torch.__version__, "devices": {}}
devs = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
for d in devs:
    g = []
    for lo, hi, x in ((-6.0, 0.0, 0.0), (-6.0, 0.0, -6.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.0)):
        for dt in (torch.float32, torch.float64):
            t = torch.tensor([x], dtype=dt, device=d, requires_grad=True)
            t.clamp(min=lo, max=hi).sum().backward()
            g.append(float(t.grad[0]))
    res["devices"][d] = g
res["ok"] = all(v == 1.0 for g in res["devices"].values() for v in g)
print(json.dumps(res))
sys.exit(0 if res["ok"] else 1)
