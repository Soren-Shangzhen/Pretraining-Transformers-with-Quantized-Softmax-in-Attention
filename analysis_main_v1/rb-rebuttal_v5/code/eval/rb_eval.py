#!/usr/bin/env python3
"""Evaluation of rb-* runs and their paired endpoints (paper protocol, Section 4 / Appendix C.3).

Full validation split of the run's corpus (WikiText-103: 243 blocks, FineWebEdu-3B: 976 blocks of
1024 tokens), batch 1, bf16 autocast outside the fp32 attention operator, TF32 off, forward from an
explicitly named frozen code root.  The per-block functions are the paper's
(analysis_main_v1/code/common.py: ValBlocks, per_block_nll, import_code, build_model).

rb-root_v1 checkpoints carry operator fields that older roots do not know (tail policy, per-layer
step, external operators, EMA buffer); they are built with every field of their config.  Historical
checkpoints are built exactly as in the paper (common.build_model).

Jobs TSV columns: label, eval_root, checkpoint, dataset, tokens_m, role[, surrogate]
  surrogate (optional) overrides NEAREST_SURROGATE of the root (the forward is the same for both
  strict surrogates; historical roots ignore it).
Output: <out>/<label>/<tokens_m>m_<ckpt stem>/{per_block_nll.npy, summary.json}; resumable.

Environment: RB_DATA_<DATASET> overrides the validation path, e.g. RB_DATA_WIKITEXT103=/path/val.bin.
"""
import argparse, csv, datetime as dt, json, os, socket, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "..", "..", "code"))     # analysis_main_v1/code
sys.path.insert(0, COMMON)
from common import ValBlocks, build_model, import_code, per_block_nll, sha256  # noqa: E402
sys.path.insert(0, HERE)
from rb_provenance import provenance  # noqa: E402

VAL = {
    "wikitext103": dict(path="<CLOUD_DISK>/data/wikitext103/val.bin",
                        sha256="397ae25de9c593190ddc226fe15577337038a549046a90eaa785a1fc6fc7e979", n_blocks=243),
    "finewebedu_3b": dict(path="<CLOUD_DISK>/data/finewebedu_3b/val.bin", sha256=None, n_blocks=976),
}


def val_spec(dataset):
    spec = dict(VAL[dataset])
    env = os.environ.get(f"RB_DATA_{dataset.upper()}")
    if env:
        spec["path"] = env
    sha_env = os.environ.get(f"RB_DATA_{dataset.upper()}_SHA256")
    if sha_env:
        spec["sha256"] = sha_env
    return spec


def build_rb_model(mm, ck, device):
    """Build a model of an rb-root_v1 checkpoint from every operator field of its config."""
    c = ck["config"]
    h0 = c.get("resolved_h0") or ""
    tau_pl = tuple(int(c["K"]) * float(x) for x in h0.split(",")) if h0 else None
    cfg = mm.GPTConfig(block_size=int(c["block_size"]), n_layer=int(c["n_layer"]), n_head=int(c["n_head"]),
                       n_embd=int(c["n_embd"]), attn_mode=c["attn"], attn_K=int(c["K"]), attn_approx=c["approx"],
                       attn_range_grad=c["range_grad"], attn_range_mode=c["range_mode"],
                       attn_range_tau=float(c["range_tau"]), attn_tail_policy=c.get("tail_policy", "zero"),
                       attn_range_tau_per_layer=tau_pl, attn_variant=c.get("attn_variant", ""),
                       attn_exaq_bits=int(c.get("exaq_bits", 3)), attn_ema_decay=float(c.get("ema_decay", 0.999)))
    model = mm.GPT(cfg)
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    model.grad_checkpointing = False
    op = {k: c.get(k) for k in ("attn", "K", "approx", "range_grad", "range_mode", "range_tau", "tail_policy",
                                "resolved_h0", "attn_variant", "exaq_bits", "ema_decay", "nearest_surrogate")}
    return model, cfg, op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", required=True, help="evaluate only jobs whose eval_root equals this path")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--tf32", choices=["off", "on"], default="off")
    ap.add_argument("--tag", default="", help="suffix of the output directory (evaluation-setting variants)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == "on"
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on"
    mm, qa = import_code(args.root)
    is_rb = "attn_tail_policy" in getattr(mm.GPTConfig, "__dataclass_fields__", {})
    hashes = {f: sha256(os.path.join(args.root, f)) for f in ("model.py", "quant_attention.py")}
    jobs = [j for j in csv.DictReader(open(args.jobs), delimiter="\t") if j["eval_root"] == args.root]
    blocks_cache = {}
    for j in jobs:
        ckp = j["checkpoint"]
        if not os.path.exists(ckp):
            print("MISSING", ckp, flush=True)
            continue
        out = Path(args.out) / j["label"] / f"{int(round(float(j['tokens_m']))):04d}m_{Path(ckp).stem}{args.tag}"
        if (out / "summary.json").exists():
            continue
        spec = val_spec(j["dataset"])
        if j["dataset"] not in blocks_cache:
            vsha = sha256(spec["path"])
            if spec["sha256"] and vsha != spec["sha256"]:
                raise SystemExit(f"validation file hash mismatch for {j['dataset']}: {vsha}")
            b = ValBlocks(spec["path"], 1024)
            assert b.n_blocks == spec["n_blocks"], (b.n_blocks, spec)
            blocks_cache[j["dataset"]] = (b, vsha)
        blocks, vsha = blocks_cache[j["dataset"]]
        t0 = time.time()
        ck = torch.load(ckp, map_location="cpu", weights_only=False, mmap=True)
        c = ck["config"]
        sur = (j.get("surrogate") or "").strip() or c.get("nearest_surrogate") or "prob"
        if sur not in ("prob", "weight_strict"):
            sur = "prob"          # historical labels (weight_hist/prob_hist): the forward is unaffected
        if hasattr(qa, "NEAREST_SURROGATE"):
            qa.NEAREST_SURROGATE = sur
        if is_rb and "tail_policy" in c:
            model, cfg, op = build_rb_model(mm, ck, "cuda")
        else:
            model, cfg, op = build_model(mm, ck, "cuda")
        ds = os.path.basename(os.path.normpath(str(c.get("data_dir", ""))))
        if ds and ds != j["dataset"]:
            raise SystemExit(f"{ckp}: trained on {ds}, job says {j['dataset']}")
        nll = per_block_nll(model, blocks, 0, blocks.n_blocks, "cuda", args.batch_size)
        tmp = out.with_name(out.name + f".tmp_{os.getpid()}")
        tmp.mkdir(parents=True, exist_ok=True)
        np.save(tmp / "per_block_nll.npy", nll)
        summary = dict(label=j["label"], role=j.get("role"), checkpoint=ckp, step=int(ck["step"]),
                       tokens_done=int(ck["tokens_done"]), checkpoint_operator=op, nearest_surrogate_used=sur,
                       config=c, eval_root=args.root, eval_code_sha256=hashes, dataset=j["dataset"],
                       val_path=spec["path"], val_sha256=vsha, n_blocks=int(blocks.n_blocks), block_size=1024,
                       batch_size=args.batch_size, autocast="bfloat16", tf32=args.tf32 == "on",
                       mean_nll=float(nll.mean()), torch=torch.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0), host=socket.gethostname(),
                       seconds=round(time.time() - t0, 1), finished=dt.datetime.now().isoformat(timespec="seconds"),
                       provenance=provenance(ckp, args.root, [os.path.abspath(__file__), os.path.join(COMMON, "common.py"),
                                                              os.path.join(HERE, "rb_provenance.py")]))
        (tmp / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        os.rename(tmp, out)
        print(f"done {j['label']} {j['tokens_m']}M nll={summary['mean_nll']:.6f} {summary['seconds']}s", flush=True)
        del model, ck
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
