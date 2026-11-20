#!/usr/bin/env python3
"""Standardized evaluation of the pre-Phase-B detached-calibration runs (development record).

Same protocol as eval_worker_5seed.py: WikiText-103 full validation (243 blocks), batch 1, bf16 autocast outside the
fp32 attention kernel, the run's own frozen training code, geometry probe on 8 blocks.
Matched full-gradient seed-1337 counterparts are read from the five-seed evaluation (identical protocol).
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ValBlocks, build_model, import_code, per_block_nll, sha256  # noqa: E402
from geometry_probe_v1 import GeometryProbe  # noqa: E402
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


TRAIN = Path(KI_REPO)
OUT = TRAIN / "analysis_main_v1/E_detach_history"
SNAP = TRAIN / "analysis_main_v1/D_robustness_124m_100m_5seed/code_snapshots"
LEG = TRAIN / "training_code/legacy_exact_fp32/runs"
PB = TRAIN / "training_code/phaseB_base/runs"
VAL = TRAIN / "training_code/phaseB_base/data/wikitext103/val.bin"
VAL_SHA = "397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979"


def jobs():
    J = []
    for r, lab, K in [("q2_nearest_100m", "Nearest", 2), ("q4_nearest_100m", "Nearest", 4), ("q8_nearest_100m", "Nearest", 8),
                      ("q16_nearest_100m", "Nearest", 16)]:
        J.append((r, lab, K, "detach", 1337, LEG / r / "ckpt.pt", SNAP / "legacy_exact_fp32"))
    for r, K in [("q4_lerp_100m", 4), ("q32_lerp_100m", 32)]:
        for f in sorted((LEG / r).glob("ckpt_step*.pt")):
            J.append((r, "LERP", K, "detach", 1337, f, SNAP / "legacy_exact_fp32"))
    for f in sorted((PB / "q32_lerp_detachproj_100m").glob("ckpt_step*.pt")):
        J.append(("q32_lerp_detachproj_100m", "LERP", 32, "detach_proj", 1337, f, SNAP / "local_phaseB_base"))
    for s in (1337, 1338, 1339):
        suf = "" if s == 1337 else f"_s{s}"
        J.append((f"exact_fp32_softmax_5m{suf}", "Exact", 0, "n/a", s, LEG / f"exact_fp32_softmax_5m{suf}" / "ckpt.pt",
                  SNAP / "legacy_exact_fp32"))
    for s in (1338, 1339):
        for K in (2, 4, 8):
            r = f"q{K}_nearest_5m_s{s}"
            J.append((r, "Nearest", K, "detach", s, LEG / r / "ckpt.pt", SNAP / "legacy_exact_fp32"))
    return J


def main():
    if sha256(VAL) != VAL_SHA:
        raise SystemExit("val hash mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    blocks = ValBlocks(str(VAL), 1024)
    rows = []
    for run, lab, K, rg, seed, ck_path, code in jobs():
        out = OUT / "eval_native_v1" / run / ck_path.stem
        if (out / "summary.json").exists():
            rows.append(json.loads((out / "summary.json").read_text()))
            continue
        mm, qa = import_code(str(code))
        # geometry probe needs the Phase-B reconstruct signature; its forward values are identical to legacy code
        _, qa_geom = import_code(str(SNAP / "local_phaseB_base"))
        ck = torch.load(ck_path, map_location="cpu", weights_only=False, mmap=True)
        c = ck["config"]
        model, cfg, op = build_model(mm, ck, "cuda")
        assert int(c["seed"]) == seed and os.path.basename(os.path.normpath(c["data_dir"])) == "wikitext103"
        if lab != "Exact":
            assert int(op["K"]) == K and op["approx"] == ("lerp" if lab == "LERP" else "nearest")
            assert op["range_grad"] == rg, (op["range_grad"], rg)
        nll = per_block_nll(model, blocks, 0, blocks.n_blocks, "cuda", 1)
        with GeometryProbe(model, qa_geom, op, int(cfg.n_layer)) as probe, torch.inference_mode():
            for i in range(8):
                x, _ = blocks.batch(i, i + 1, "cuda")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(x)
            geom = probe.results()
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "per_block_nll.npy", nll)
        with open(out / "geometry_layers.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(geom[0].keys()))
            w.writeheader()
            w.writerows(geom)
        s = dict(run=run, label=lab, K=K, range_grad=rg, seed=seed, checkpoint=str(ck_path), step=int(ck["step"]),
                 tokens_done=int(ck["tokens_done"]), max_tokens=int(c["max_tokens"]), mean_nll=float(nll.mean()),
                 n_blocks=int(blocks.n_blocks), batch_size=1, eval_code=str(code),
                 eval_code_sha256=sha256(code / "quant_attention.py"), gpu=torch.cuda.get_device_name(0))
        (out / "summary.json").write_text(json.dumps(s, indent=2))
        rows.append(s)
        print(f"{run:32s} {lab:8s} K={K:<3d} {rg:12s} seed={seed} step={s['step']:4d} nll={s['mean_nll']:.5f}", flush=True)
        del model, ck
        torch.cuda.empty_cache()
    with open(OUT / "detach_history_endpoints.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
