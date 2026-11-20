#!/usr/bin/env python3
"""Unified native evaluation worker for the 35 primary runs (analysis_main_v1).

For every checkpoint job in manifests/eval_jobs.tsv:
  1. load the checkpoint (mmap, model weights only), assert its config matches the manifest,
  2. evaluate per-block NLL on the FULL validation split of the run's own dataset
     (non-overlapping 1024-token blocks, batch 1, bf16 autocast outside the fp32 attention kernel,
      forward computed by the run's OWN frozen training code root; identical protocol to
      eval_fbmatch_native.py). Parity check showed Prob-STE frozen code and batch size both perturb
      hard-Nearest block NLL at the 1e-3..1e-2 level, so neither is substituted.
  3. run the layerwise geometry probe on the first --geom-blocks validation blocks.

Outputs (atomic, one directory per checkpoint):
  eval_native_v1/<suite>/<run_name>/<ckpt_stem>/{per_block_nll.npy, summary.json, geometry_layers.csv}

Several workers may run concurrently on one host; jobs are claimed with O_EXCL lock files.
A job is done when summary.json exists.
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

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DATA, ValBlocks, build_model, import_code, per_block_nll, sha256  # noqa: E402
from geometry_probe_v1 import GeometryProbe  # noqa: E402
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


ROOT = KI_REPO + "/analysis_main_v1"
LOCKS = "<CLOUD_HOME>/analysis_main_v1_locks"

EXPECT_APPROX = {"Exact": None, "LERP": "lerp", "MinMax-Weight": "nearest", "MinMax-Prob": "nearest",
                 "QRM-Weight": "nearest", "QRM-Prob": "nearest"}


def log(msg):
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def claim(job_id):
    os.makedirs(LOCKS, exist_ok=True)
    path = os.path.join(LOCKS, f"{job_id}.lock")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(open(path).read().strip() or 0)
            os.kill(pid, 0)
            return None
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            os.remove(path)          # stale lock from a dead worker
            return claim(job_id)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return path


def check_config(job, run, op, ck, cfg_raw):
    errs = []
    lab = run["label"]
    if lab == "Exact":
        if op["attn"] != "exact_fp32_softmax":
            errs.append(f"attn={op['attn']}")
    else:
        if op["attn"] != "quant":
            errs.append(f"attn={op['attn']}")
        if op["approx"] != EXPECT_APPROX[lab]:
            errs.append(f"approx={op['approx']}")
        if str(op["K"]) != run["K"]:
            errs.append(f"K={op['K']} manifest={run['K']}")
        if op["range_grad"] != "full":
            errs.append(f"range_grad={op['range_grad']}")
        want_mode = "relevant_zero" if lab.startswith("QRM") else "minmax"
        if op["range_mode"] != want_mode:
            errs.append(f"range_mode={op['range_mode']}")
        if want_mode == "relevant_zero" and abs(op["range_tau"] - 6.0) > 1e-9:
            errs.append(f"range_tau={op['range_tau']}")
    if os.path.basename(os.path.normpath(cfg_raw["data_dir"])) != run["dataset"]:
        errs.append(f"data_dir={cfg_raw['data_dir']} manifest={run['dataset']}")
    if int(ck["step"]) != int(job["step"]):
        errs.append(f"step={ck['step']} manifest={job['step']}")
    if errs:
        raise RuntimeError("CONFIG MISMATCH: " + "; ".join(errs))


_CODE_CACHE = {}


def code_for(root):
    if root not in _CODE_CACHE:
        mm, qa = import_code(root)
        _CODE_CACHE.clear()
        _CODE_CACHE[root] = (mm, qa, {f: sha256(os.path.join(root, f)) for f in ("model.py", "quant_attention.py")})
    return _CODE_CACHE[root]


def run_job(job, run, args, device):
    stem = os.path.splitext(os.path.basename(job["checkpoint_path"]))[0]
    out = os.path.join(ROOT, "eval_native_v1", job["suite"], job["run_name"], stem)
    if os.path.exists(os.path.join(out, "summary.json")):
        return "skip"
    t0 = time.time()
    model_mod, qa_mod, code_hashes = code_for(run["train_code_root"])
    ck = torch.load(job["checkpoint_path"], map_location="cpu", weights_only=False, mmap=True)
    cfg_raw = ck["config"]
    model, cfg, op = build_model(model_mod, ck, device)
    check_config(job, run, op, ck, cfg_raw)
    t_load = time.time() - t0

    blocks = ValBlocks(DATA[run["dataset"]], int(cfg.block_size))
    bs = args.batch_1b if run["model_scale"] == "1B" else args.batch_124m
    t1 = time.time()
    nll = per_block_nll(model, blocks, 0, blocks.n_blocks, device, bs)
    t_eval = time.time() - t1

    t2 = time.time()
    geom = []
    if args.geom_blocks > 0:
        with GeometryProbe(model, qa_mod, op, int(cfg.n_layer)) as probe, torch.inference_mode():
            for i in range(args.geom_blocks):
                x, _ = blocks.batch(i, i + 1, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(x)
            geom = probe.results()
    t_geom = time.time() - t2

    tmp = out + f".tmp_{socket.gethostname()}_{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(tmp, "per_block_nll.npy"), nll)
    if geom:
        with open(os.path.join(tmp, "geometry_layers.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(geom[0].keys()))
            w.writeheader()
            w.writerows(geom)
    summary = dict(
        job_id=job["job_id"], phase=job["phase"], suite=job["suite"], run_name=job["run_name"],
        label=run["label"], calibration=run["calibration"], reconstruction=run["reconstruction"],
        backward=run["backward"], K=run["K"], range_tau=run["range_tau"], range_grad=run["range_grad"],
        dataset=run["dataset"], seed=run["seed"], model_scale=run["model_scale"],
        checkpoint=job["checkpoint_path"], ckpt_index=int(job["ckpt_index"]), is_final=int(job["is_final"]),
        step=int(ck["step"]), tokens_done=int(ck["tokens_done"]),
        checkpoint_operator=op, val_path=DATA[run["dataset"]], n_blocks=int(blocks.n_blocks),
        block_size=int(cfg.block_size), eval_tokens=int(blocks.n_blocks * cfg.block_size),
        batch_size=bs, autocast="bfloat16", mean_nll=float(nll.mean()),
        geom_blocks=args.geom_blocks, eval_code=run["train_code_root"], eval_code_sha256=code_hashes,
        eval_scripts=HERE, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), host=socket.gethostname(),
        seconds=dict(load=round(t_load, 1), nll=round(t_eval, 1), geometry=round(t_geom, 1)),
        finished=dt.datetime.now().isoformat(timespec="seconds"),
    )
    with open(os.path.join(tmp, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if os.path.exists(out):
        raise RuntimeError(f"output dir appeared concurrently: {out}")
    os.rename(tmp, out)
    del model, ck
    torch.cuda.empty_cache()
    return f"nll={summary['mean_nll']:.6f} load={t_load:.0f}s nll={t_eval:.0f}s geom={t_geom:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="w0")
    ap.add_argument("--phase", choices=["endpoint", "trajectory", "all"], default="all")
    ap.add_argument("--suites", default="B_factorial_124m_250m,A_longhorizon_124m_2p5b,C_scale_1b_100m")
    ap.add_argument("--scales", default="124M,1B")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-124m", type=int, default=1)
    ap.add_argument("--batch-1b", type=int, default=1)
    ap.add_argument("--geom-blocks", type=int, default=8)
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    man = os.path.join(ROOT, "manifests")
    runs = {r["run_name"]: r for r in csv.DictReader(open(f"{man}/primary_37_runs.tsv"), delimiter="\t")}
    jobs = list(csv.DictReader(open(f"{man}/eval_jobs.tsv"), delimiter="\t"))
    suites = set(args.suites.split(","))
    scales = set(args.scales.split(","))
    jobs = [j for j in jobs if j["suite"] in suites and j["model_scale"] in scales
            and (args.phase == "all" or j["phase"] == args.phase)]
    if args.reverse:
        jobs = jobs[::-1]
    log(f"worker={args.worker} pid={os.getpid()} candidate_jobs={len(jobs)}")

    done = 0
    for job in jobs:
        stem = os.path.splitext(os.path.basename(job["checkpoint_path"]))[0]
        if os.path.exists(os.path.join(ROOT, "eval_native_v1", job["suite"], job["run_name"], stem, "summary.json")):
            continue
        lock = claim(job["job_id"])
        if lock is None:
            continue
        try:
            msg = run_job(job, runs[job["run_name"]], args, device)
            log(f"{job['job_id']} {job['phase']:10s} {job['run_name']} {stem}: {msg}")
            done += msg != "skip"
        except Exception:
            log(f"{job['job_id']} FAILED {job['run_name']} {stem}\n{traceback.format_exc()}")
            with open(os.path.join(ROOT, "eval_native_v1", "FAILED_JOBS.txt"), "a") as fh:
                fh.write(f"{job['job_id']}\t{job['run_name']}\t{stem}\t{args.worker}\n")
            torch.cuda.empty_cache()
        finally:
            os.remove(lock)
        if args.limit and done >= args.limit:
            break
    log(f"worker={args.worker} finished, evaluated={done}")


if __name__ == "__main__":
    main()
