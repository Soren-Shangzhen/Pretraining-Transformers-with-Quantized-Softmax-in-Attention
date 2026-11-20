#!/usr/bin/env python3
"""Downstream evaluation of machine A's runs on machine B (user decision 2026-11-12; plan section 3 and E8a/E8c):
the three held-out corpora (paper layer 1) and the 7 benchmark configurations (paper layer 3, stage C).

The paper's code is imported unchanged from analysis_main_v1/downstream_v1/code; three things are replaced:
  runs()   the conditions evaluated here: rb-E8a-01, rb-E1b-01, and the suite A seed-1337 softmax endpoint
           (Exact_s1337_check), re-run on B as a same-machine check against the paper's S1 results;
  load()   rb-root_v1 checkpoints are built from every operator field of their config (rb_eval.build_rb_model;
           the paper's loader passes only the historical fields and its label checks have no FWM-LERP entry);
           historical checkpoints still go through the paper's loader and its identity checks;
  output   $RB_RUNS/_eval/downstream/{stageC, layer1, tasks} instead of the paper's results tree.
Stage C keeps the paper's configs, seeds, scorer, acc_norm rule and .done fingerprints; layer 1 keeps its corpora
(SHA-256 checked against the transfer record) and block protocol, native and exact forward.

  --what prefetch   (online, once): fill the HF and tiktoken caches for the 7 tasks, then check that every cached
                    dataset revision equals the one the paper used (stage C log); writes downstream/PREFETCH_OK
                    or PREFETCH_FAIL.json.  Later runs use HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1.
  --what layer1 --cond COND      three corpora x native/exact, per-block NLL (.npy) + summary.json
  --what stagec --cond COND      7 configurations (per-item csv.gz, summary, .done)
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rb_eval import build_rb_model  # noqa: E402
TRAIN = os.environ.get("RB_TRAIN_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))   # the root of this package
DSCODE = f"{TRAIN}/analysis_main_v1/downstream_v1/code"
sys.path.insert(0, DSCODE)
import common_loader as cl  # noqa: E402
from rb_provenance import provenance  # noqa: E402
PROV_SCRIPTS = [os.path.abspath(__file__), os.path.join(HERE, "rb_eval.py"), os.path.join(HERE, "rb_provenance.py"),
                f"{DSCODE}/common_loader.py", f"{DSCODE}/layer3_stageC.py"]

RBROOT = os.path.abspath(os.path.join(HERE, "..", "rb-root_v1"))
CLOUD = os.environ.get("RB_CLOUD", "<CLOUD_DISK>")
AO = os.environ.get("RB_A_EVAL", "<POD_WORKSPACE>/A_eval_model_only")
OUT = os.path.join(os.environ.get("RB_RUNS", "<POD_WORKSPACE>/runs"), "_eval", "downstream")
LAYER1 = f"{TRAIN}/analysis_main_v1/downstream_v1/data/layer1"
CORPORA = {"wikitext103_test": "345b5f4436ea6d8536d6a1853f7dd515772101e0b05c1dce415235d41339d4de",
           "ptb_test": "f225b4b5cba14dcdd5d1d1b737f7630d7e4b4a39e10e2a9c483535aa71613612",
           "c4_validation": "f251c3dddf2dafe60c91a5d9e0d133c7bbd9c79d26e7fb749a55d0a7b20c8c72"}
REVISIONS = {"nyu-mll___blimp": "877fba0801ffb7cbd8c39c1ff314a46f053f6036",       # downstream_v1/logs/stageC.log
             "Rowan___hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
             "baber___piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
             "allenai___ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
             "EleutherAI___lambada_openai": "900124bf3b8235c6daf21033af9948b3f07346c4",
             "allenai___sciq": "2c94ad3e1aafab77146f384e23536f97a4849815"}
base = dict(suite="A_longhorizon_124m_2p5b", dataset="finewebedu_3b", model_scale="124m", token_horizon_m="2500",
            K="", calibration="", reconstruction="", backward="", range_tau="", range_grad="")
RUNS = {
    "rb-E8a-01": dict(base, run_name="rb-E8a-01", label="rb-E8a-01", seed="7", train_code_root=RBROOT,
                      final_path=f"{AO}/rb-E8a-01/ckpt_step019074_2500.1m.model_only.pt"),
    "rb-E1b-01": dict(base, run_name="rb-E1b-01", label="rb-E1b-01", seed="1337", train_code_root=RBROOT,
                      final_path=f"{AO}/rb-E1b-01/ckpt_step019074_2500.1m.model_only.pt"),
    # machine A (2026-11-12): its own 2.5B endpoints, evaluated on A when they end
    "rb-E8c-01": dict(base, run_name="rb-E8c-01", label="rb-E8c-01", seed="7", train_code_root=RBROOT,
                      final_path=f"{os.environ.get('RB_RUNS', '<POD_WORKSPACE>/runs')}/rb-E8c-01/ckpt_step019074_2500.1m.pt"),
    "rb-E3a-2.5B-01": dict(base, run_name="rb-E3a-2.5B-01", label="rb-E3a-2.5B-01", seed="1337", train_code_root=RBROOT,
                           final_path=f"{os.environ.get('RB_RUNS', '<POD_WORKSPACE>/runs')}/rb-E3a-2.5B-01/ckpt_step019074_2500.1m.pt"),
    "Exact_s1337_check": dict(base, run_name="formal_124m_s1337_exact_2p5b", label="Exact", seed="1337",
                              train_code_root=f"{CLOUD}/formal_code_factorial250_v1/formal_base_v2",
                              final_path=f"{CLOUD}/formal_runs_scale_v1/formal_124m_s1337_exact_2p5b/"
                                         "ckpt_step019074_2500.1m.pt"),
}


def runs(*_a, **_k):
    return [dict(r, cond=c) for c, r in RUNS.items()]


def load(run, forward_mode="native", device="cuda", ckpt_hash=False):
    if os.path.realpath(run["train_code_root"]) != os.path.realpath(RBROOT):
        return cl.load(run, forward_mode, device, ckpt_hash)            # paper loader + identity checks
    cl.set_numerics()
    mm, qa = cl.import_code(RBROOT)
    ck = torch.load(run["final_path"], map_location="cpu", weights_only=False, mmap=True)
    c = dict(ck["config"])
    if os.path.basename(os.path.normpath(c["data_dir"])) != run["dataset"]:
        raise RuntimeError(f'{run["run_name"]}: trained on {c["data_dir"]}, expected {run["dataset"]}')
    if int(ck["step"]) != 19074:
        raise RuntimeError(f'{run["run_name"]}: final step {ck["step"]} != 19074')
    if forward_mode == "exact":
        c["attn"] = "exact_fp32_softmax"
    model, cfg, op = build_rb_model(mm, dict(ck, config=c), device)
    if hasattr(qa, "NEAREST_SURROGATE"):
        qa.NEAREST_SURROGATE = c.get("nearest_surrogate") if c.get("nearest_surrogate") in ("prob", "weight_strict") else "prob"
    h = cl.sha256(run["final_path"]) if ckpt_hash else ""
    return cl.Loaded(model=model, cfg=cfg, op=op, run=run, forward_mode=forward_mode, ckpt_sha256=h)


def stage_c_module():
    import layer3_stageC as s3
    s3.runs, s3.load = runs, load
    s3.OUT, s3.TASKDIR = f"{OUT}/stageC", f"{OUT}/tasks"
    os.makedirs(s3.OUT, exist_ok=True); os.makedirs(s3.TASKDIR, exist_ok=True)
    return s3


def prefetch():
    s3 = stage_c_module()
    import tiktoken
    from lm_eval.tasks import TaskManager, get_task_dict
    s3.write_task_yaml()
    tiktoken.get_encoding("gpt2")
    tm = TaskManager(include_path=s3.TASKDIR)
    for c in s3.CONFIGS:
        get_task_dict([c["task"]], task_manager=tm)
        print("cached", c["task"], flush=True)
    root = os.environ.get("HF_DATASETS_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "datasets")   # vast sets HF_HOME
    found, bad = {}, {}
    for ds, rev in REVISIONS.items():
        revs = sorted({os.path.basename(p) for p in glob.glob(f"{root}/{ds}/*/*/*") if len(os.path.basename(p)) == 40})
        found[ds] = revs
        if revs != [rev]:
            bad[ds] = dict(expected=rev, cached=revs)
    res = dict(found=found, mismatches=bad, cache=root, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if bad:
        json.dump(res, open(f"{OUT}/PREFETCH_FAIL.json", "w"), indent=1)
        print("PREFETCH FAIL", json.dumps(bad)); sys.exit(1)
    json.dump(res, open(f"{OUT}/PREFETCH_OK", "w"), indent=1)
    print("PREFETCH OK: every cached revision equals the paper's")


def layer1(cond):
    run = dict(RUNS[cond], cond=cond)
    d = f"{OUT}/layer1/{cond}"
    os.makedirs(f"{d}/per_block", exist_ok=True)
    toks = {}
    for name, sha in CORPORA.items():
        p = f"{LAYER1}/{name}.bin"
        if cl.sha256(p) != sha:
            raise SystemExit(f"{p}: SHA-256 differs from the transfer record")
        toks[name] = np.memmap(p, dtype=np.uint16, mode="r")
    summ = dict(condition=cond, run_name=run["run_name"], checkpoint=run["final_path"],
                checkpoint_sha256=cl.sha256(run["final_path"]), train_code_root=run["train_code_root"],
                corpora_sha256=CORPORA, block=cl.BLOCK, batch_size=1, autocast="bfloat16", tf32=False,
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0), results={})
    for mode in ("native", "exact"):
        t0 = time.time()
        ld = load(run, mode)
        for name, t in toks.items():
            nll = cl.block_nll(ld.model, t)
            np.save(f"{d}/per_block/{run['run_name']}__{name}__{mode}.npy", nll)
            summ["results"][f"{name}__{mode}"] = dict(n_blocks=int(len(nll)), nll=float(nll.mean()),
                                                     ppl=float(np.exp(nll.mean())),
                                                     nonfinite=int((~np.isfinite(nll)).sum()))
            print(f"  {cond} {mode:6s} {name:18s} n={len(nll)} nll={nll.mean():.5f}", flush=True)
        summ[f"seconds_{mode}"] = round(time.time() - t0, 1)
        del ld
        torch.cuda.empty_cache()
    summ["provenance"] = provenance(run["final_path"], run["train_code_root"], PROV_SCRIPTS)
    json.dump(summ, open(f"{d}/summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["prefetch", "layer1", "stagec"], required=True)
    ap.add_argument("--cond", choices=sorted(RUNS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what == "prefetch":
        prefetch()
    elif a.what == "layer1":
        layer1(a.cond)
    else:
        s3 = stage_c_module()
        d = f"{s3.OUT}/{s3.slug(a.cond)}"
        os.makedirs(d, exist_ok=True)
        run = RUNS[a.cond]
        json.dump(provenance(run["final_path"], run["train_code_root"], PROV_SCRIPTS), open(f"{d}/provenance.json", "w"), indent=1)
        s3.run_condition(a.cond)


if __name__ == "__main__":
    main()
