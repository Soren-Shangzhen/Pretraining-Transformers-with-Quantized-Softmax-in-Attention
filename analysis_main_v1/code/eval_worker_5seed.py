#!/usr/bin/env python3
"""Local (RTX 3090) native evaluation of the WikiText-103 five-seed suite.

Same protocol as eval_worker.py (primary suites): full validation split (243 non-overlapping 1024-token blocks),
batch 1, bf16 autocast outside the fp32 attention kernel, forward from the run's own frozen training code snapshot,
layerwise geometry probe on the first --geom-blocks validation blocks. Jobs come only from
D_robustness_124m_100m_5seed/manifests/five_seed_checkpoints.tsv (gate-verified).

Output: D_robustness_124m_100m_5seed/eval_native_v1/<run_name>/<ckpt_stem>/{per_block_nll.npy, summary.json,
geometry_layers.csv}. Resumable: a job is done when summary.json exists.
"""
import argparse
import csv
import datetime as dt
import json
import os
import socket
import sys
import time
import traceback
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


SUITE = Path(KI_REPO + "/analysis_main_v1/D_robustness_124m_100m_5seed")
VAL = (KI_REPO + "/adaptive_k_interval_training_phaseB/"
       "files_adaptive_quant_phaseB/data/wikitext103/val.bin")
VAL_SHA = "397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979"
_CODE = {}


def log(msg):
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def code_for(root):
    if root not in _CODE:
        mm, qa = import_code(root)
        _CODE.clear()
        _CODE[root] = (mm, qa, {f: sha256(os.path.join(root, f)) for f in ("model.py", "quant_attention.py")})
    return _CODE[root]


def run_job(job, run, args, blocks):
    stem = Path(job["checkpoint_path"]).stem
    out = SUITE / "eval_native_v1" / job["run_name"] / f"{int(job['tokens_m']):03d}m_{stem}"
    if (out / "summary.json").exists():
        return None
    t0 = time.time()
    mm, qa, hashes = code_for(run["train_code_root"])
    ck = torch.load(job["checkpoint_path"], map_location="cpu", weights_only=False, mmap=True)
    model, cfg, op = build_model(mm, ck, "cuda")
    c = ck["config"]
    errs = []
    if int(ck["step"]) != int(job["step"]):
        errs.append(f"step {ck['step']} != {job['step']}")
    if os.path.basename(os.path.normpath(c["data_dir"])) != "wikitext103":
        errs.append(f"data_dir {c['data_dir']}")
    if int(c["seed"]) != int(job["seed"]):
        errs.append(f"seed {c['seed']} != {job['seed']}")
    if job["K"] and int(op["K"]) != int(job["K"]):
        errs.append(f"K {op['K']} != {job['K']}")
    if errs:
        raise RuntimeError("CONFIG MISMATCH: " + "; ".join(errs))
    t_load = time.time() - t0

    t1 = time.time()
    nll = per_block_nll(model, blocks, 0, blocks.n_blocks, "cuda", 1)
    t_nll = time.time() - t1
    t2 = time.time()
    geom = []
    if args.geom_blocks:
        with GeometryProbe(model, qa, op, int(cfg.n_layer)) as probe, torch.inference_mode():
            for i in range(args.geom_blocks):
                x, _ = blocks.batch(i, i + 1, "cuda")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(x)
            geom = probe.results()
    t_geom = time.time() - t2

    tmp = out.with_name(out.name + f".tmp_{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "per_block_nll.npy", nll)
    if geom:
        with open(tmp / "geometry_layers.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(geom[0].keys()))
            w.writeheader()
            w.writerows(geom)
    summary = dict(
        job_id=job["job_id"], phase=job["phase"], run_name=job["run_name"], label=run["label"], cond=run["cond"],
        calibration=run["calibration"], reconstruction=run["reconstruction"], backward=run["backward"],
        K=run["K"], seed=int(run["seed"]), range_tau=run["range_tau"], range_grad=run["range_grad"],
        dataset="wikitext103", source_origin=run["source_origin"], checkpoint=job["checkpoint_path"],
        ckpt_index=int(job["ckpt_index"]), is_final=int(job["is_final"]), step=int(ck["step"]),
        tokens_done=int(ck["tokens_done"]), checkpoint_operator=op, val_path=VAL, val_sha256=VAL_SHA,
        n_blocks=int(blocks.n_blocks), block_size=int(cfg.block_size), batch_size=1, autocast="bfloat16",
        mean_nll=float(nll.mean()), geom_blocks=args.geom_blocks, eval_code=run["train_code_root"],
        eval_code_sha256=hashes, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), host=socket.gethostname(),
        seconds=dict(load=round(t_load, 1), nll=round(t_nll, 1), geometry=round(t_geom, 1)),
        finished=dt.datetime.now().isoformat(timespec="seconds"))
    (tmp / "summary.json").write_text(json.dumps(summary, indent=2))
    os.rename(tmp, out)
    del model, ck
    torch.cuda.empty_cache()
    return f"nll={summary['mean_nll']:.6f} load={t_load:.0f}s nll={t_nll:.0f}s geom={t_geom:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["endpoint", "trajectory", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--geom-blocks", type=int, default=8)
    ap.add_argument("--runs-tsv", default="five_seed_wikitext_130.tsv")
    ap.add_argument("--jobs-tsv", default="five_seed_checkpoints_130.tsv")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(VAL) != VAL_SHA:
        raise SystemExit("validation file hash mismatch")
    man = SUITE / "manifests"
    runs = {}
    for r in csv.DictReader(open(man / args.runs_tsv), delimiter="\t"):
        runs[f"{r['cond'].replace(' ', '_')}_s{r['seed']}"] = r
    jobs = [j for j in csv.DictReader(open(man / args.jobs_tsv), delimiter="\t")
            if args.phase == "all" or j["phase"] == args.phase]
    blocks = ValBlocks(VAL, 1024)
    log(f"pid={os.getpid()} runs={args.runs_tsv} jobs={len(jobs)} val_blocks={blocks.n_blocks}")
    done = 0
    for j in jobs:
        try:
            msg = run_job(j, runs[j["run_name"]], args, blocks)
        except Exception:
            log(f"{j['job_id']} FAILED {j['run_name']} {j['checkpoint_path']}\n{traceback.format_exc()}")
            with open(SUITE / "eval_native_v1" / "FAILED_JOBS.txt", "a") as fh:
                fh.write(f"{j['job_id']}\t{j['run_name']}\t{j['checkpoint_path']}\n")
            torch.cuda.empty_cache()
            continue
        if msg is None:
            continue
        done += 1
        log(f"{j['job_id']} {j['phase']:10s} {j['run_name']} {j['tokens_m']}M: {msg}")
        if args.limit and done >= args.limit:
            break
    log(f"finished evaluated={done}")


if __name__ == "__main__":
    main()
