#!/usr/bin/env python3
"""Supplement evaluation on gpu-host-A (RTX 5090), historical protocol of eval_worker_5seed.py:
full WikiText-103 validation split (243 non-overlapping 1024-token blocks), batch 1, bf16 autocast
outside the fp32 attention kernel, TF32 off, forward from an explicitly named code root, layerwise
geometry probe on the first 8 blocks.  Jobs come from a TSV (columns: label, eval_root, checkpoint,
tokens_m, role).  Output: <out>/<label>/<tokens_m>m_<ckpt stem>/{per_block_nll.npy, summary.json,
geometry_layers.csv}.  Resumable: a job is done when summary.json exists.  One process per code root
(common.import_code evicts previously imported model/quant_attention modules).
"""
import argparse, csv, datetime as dt, json, os, socket, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ValBlocks, build_model, import_code, per_block_nll, sha256  # noqa: E402
from geometry_probe_v1 import GeometryProbe  # noqa: E402
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


VAL = KI_CLOUD + "/data/wikitext103/val.bin"
if os.environ.get("SUP_LOCAL") == "1":   # local RTX 3090 (identical file, hash checked below)
    VAL = (KI_REPO + "/adaptive_k_interval_training_phaseB/"
           "files_adaptive_quant_phaseB/data/wikitext103/val.bin")
VAL_SHA = "397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", required=True, help="evaluate only jobs whose eval_root equals this path")
    ap.add_argument("--geom-blocks", type=int, default=8)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(VAL) != VAL_SHA:
        raise SystemExit("validation file hash mismatch")
    blocks = ValBlocks(VAL, 1024)
    assert blocks.n_blocks == 243, blocks.n_blocks
    mm, qa = import_code(args.root)
    hashes = {f: sha256(os.path.join(args.root, f)) for f in ("model.py", "quant_attention.py")}
    jobs = [j for j in csv.DictReader(open(args.jobs), delimiter="\t") if j["eval_root"] == args.root]
    for j in jobs:
        ckp = j["checkpoint"]
        if not os.path.exists(ckp):
            print("MISSING", ckp, flush=True); continue
        out = Path(args.out) / j["label"] / f"{int(j['tokens_m']):03d}m_{Path(ckp).stem}"
        if (out / "summary.json").exists():
            continue
        t0 = time.time()
        ck = torch.load(ckp, map_location="cpu", weights_only=False, mmap=True)
        surrogate = ck["config"].get("nearest_surrogate")
        if hasattr(qa, "NEAREST_SURROGATE"):
            qa.NEAREST_SURROGATE = surrogate or "prob"
        elif surrogate not in (None, "prob"):
            raise RuntimeError(f"root {args.root} cannot evaluate nearest_surrogate={surrogate}")
        model, cfg, op = build_model(mm, ck, "cuda")
        c = ck["config"]
        assert int(c["seed"]) == int(os.environ.get("SUP_SEED", "7")) and os.path.basename(os.path.normpath(c["data_dir"])) == "wikitext103", c
        nll = per_block_nll(model, blocks, 0, blocks.n_blocks, "cuda", 1)
        geom = []
        if args.geom_blocks:
            with GeometryProbe(model, qa, op, int(cfg.n_layer)) as probe, torch.inference_mode():
                for i in range(args.geom_blocks):
                    x, _ = blocks.batch(i, i + 1, "cuda")
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model(x)
                geom = probe.results()
        tmp = out.with_name(out.name + f".tmp_{os.getpid()}")
        tmp.mkdir(parents=True, exist_ok=True)
        np.save(tmp / "per_block_nll.npy", nll)
        if geom:
            with open(tmp / "geometry_layers.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(geom[0].keys())); w.writeheader(); w.writerows(geom)
        summary = dict(label=j["label"], role=j["role"], checkpoint=ckp,
                       nearest_surrogate_used=getattr(qa, "NEAREST_SURROGATE", None), step=int(ck["step"]),
                       tokens_done=int(ck["tokens_done"]), checkpoint_operator=op, config=c,
                       eval_root=args.root, eval_code_sha256=hashes, val_path=VAL, val_sha256=VAL_SHA,
                       n_blocks=int(blocks.n_blocks), block_size=1024, batch_size=1, autocast="bfloat16",
                       tf32=False, mean_nll=float(nll.mean()), geom_blocks=args.geom_blocks,
                       torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                       host=socket.gethostname(), seconds=round(time.time() - t0, 1),
                       finished=dt.datetime.now().isoformat(timespec="seconds"))
        (tmp / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        os.rename(tmp, out)
        print(f"done {j['label']} {j['tokens_m']}M nll={summary['mean_nll']:.6f} {summary['seconds']}s", flush=True)
        del model, ck
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
