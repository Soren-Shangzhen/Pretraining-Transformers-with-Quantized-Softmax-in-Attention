#!/usr/bin/env python3
"""rb-root_v1 reproduces its base roots for every mode the bases already had.

Part A (module level, same seed): parameters at initialisation, forward loss and every parameter
gradient, rb-root_v1 vs the base root, on CPU (fp32, deterministic: must be bitwise) and on the GPU
(bf16 autocast outside fp32 attention: forward must be bitwise; gradients are reported together
with the same-root repeat difference, because index/gather backward uses atomics on the GPU).

Part B (training loop, CPU): the base root's own training script and rb-root_v1/train_formal.py
train a tiny GPT for 3 optimizer steps on a synthetic token file; logged losses and the final
weights must be bitwise identical.

usage: python check_root_equivalence.py [--skip-gpu]
"""
import argparse, csv, os, shutil, subprocess, sys, tempfile
import numpy as np
import torch
from rb_common import ROOTS, import_root, write_json, env_info

PAIRS = [  # (name, base root key, GPTConfig kwargs, surrogate)
    ("exact_fp32_softmax", "e2_fwm", dict(attn_mode="exact_fp32_softmax"), "prob"),
    ("exact_fp32_softmax", "r3", dict(attn_mode="exact_fp32_softmax"), "prob"),
    ("bf16exp", "r3", dict(attn_mode="softmax_bf16exp"), "prob"),
    ("fp8e4m3exp", "r3", dict(attn_mode="softmax_fp8e4m3exp"), "prob"),
    ("minmax_lerp_k4_full", "e2_fwm", dict(attn_mode="quant", attn_approx="lerp", attn_K=4, attn_range_grad="full"), "prob"),
    ("minmax_lerp_k4_full", "phaseB_base", dict(attn_mode="quant", attn_approx="lerp", attn_K=4, attn_range_grad="full"), "prob"),
    ("minmax_nearest_k4_prob_strict", "e2_minmax", dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_grad="full"), "prob"),
    ("minmax_nearest_k4_weight_strict", "e2_minmax", dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_grad="full"), "weight_strict"),
    ("fwm_nearest_k4_prob_strict", "e2_fwm", dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_grad="full", attn_range_mode="relevant_zero"), "prob"),
    ("fwm_nearest_k4_weight_strict", "e2_fwm", dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_grad="full", attn_range_mode="relevant_zero"), "weight_strict"),
    ("fwm_nearest_k4_weight_strict_detach", "e2_fwm", dict(attn_mode="quant", attn_approx="nearest", attn_K=4, attn_range_grad="detach", attn_range_mode="relevant_zero"), "weight_strict"),
    ("fwm_lerp_k4_full", "e2_fwm", dict(attn_mode="quant", attn_approx="lerp", attn_K=4, attn_range_grad="full", attn_range_mode="relevant_zero"), "prob"),
    ("fwm_lerp_k16_detach", "e2_fwm", dict(attn_mode="quant", attn_approx="lerp", attn_K=16, attn_range_grad="detach", attn_range_mode="relevant_zero"), "prob"),
]


def build(root_key, kw, sur, device, seed=1234):
    mm, qa = import_root(ROOTS[root_key])
    fields = mm.GPTConfig.__dataclass_fields__
    kw = {k: v for k, v in kw.items() if k in fields or not k.startswith("attn_range")}
    qa.NEAREST_SURROGATE = sur
    torch.manual_seed(seed)
    cfg = mm.GPTConfig(block_size=128, n_layer=2, n_head=4, n_embd=128, vocab_size=1024, **kw)
    return mm.GPT(cfg).to(device), qa


def run_once(root_key, kw, sur, device, x, y):
    model, qa = build(root_key, kw, sur, device)
    qa.NEAREST_SURROGATE = sur
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    if device == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
    else:
        _, loss = model(x, y)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    return params, loss.detach(), grads


def maxdiff(a, b):
    return max((a[k].double() - b[k].double()).abs().max().item() for k in a)


# GPU gradient rule (user decision 2026-11-10, second version; CPU bitwise equality stays the gate):
# for every parameter tensor  max|g_rb - g_base| <= max(1e-4 * max|g_base|, 3 * repeat_noise),
# repeat_noise = largest max|g_i - g_j| over 6 repeats of the SAME root (rb-root_v1) on the same inputs.
# Reason: GPU index/gather backward uses atomic adds, so identical code gives different low-order bits;
# the first relative-only rule (1e-4) failed on torch 2.5.1 for a tensor with max|g| ~ 0.02 whose
# same-root repeats differ by the same 3.78e-6.
GPU_REL_TOL = 1e-4
GPU_NOISE_FACTOR = 3.0
GPU_REPEATS = 6


def gpu_tensor_check(g_rb, g_base, repeats):
    """per tensor: (passes, worst ratio diff / allowed)"""
    worst, ok = 0.0, True
    for k in g_rb:
        d = (g_rb[k].double() - g_base[k].double()).abs().max().item()
        noise = max((repeats[i][k].double() - repeats[j][k].double()).abs().max().item()
                    for i in range(len(repeats)) for j in range(i + 1, len(repeats)))
        allowed = max(GPU_REL_TOL * g_base[k].double().abs().max().item(), GPU_NOISE_FACTOR * noise)
        r = d / allowed if allowed > 0 else (0.0 if d == 0 else float("inf"))
        worst = max(worst, r)
        ok &= d <= allowed
    return ok, worst


def part_a(device):
    torch.manual_seed(99)
    x = torch.randint(0, 1024, (1, 128), device=device)
    y = torch.randint(0, 1024, (1, 128), device=device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True     # training numerics
    out = []
    for name, base, kw, sur in PAIRS:
        pb, lb, gb = run_once(base, kw, sur, device, x, y)
        pr, lr, gr = run_once("rb", kw, sur, device, x, y)
        reps = [gr] + [run_once("rb", kw, sur, device, x, y)[2] for _ in range(GPU_REPEATS - 1)] if device == "cuda" else None
        pr2, lr2, gr2 = run_once("rb", kw, sur, device, x, y)
        init_same = all(torch.equal(pb[k], pr[k]) for k in pb) and pb.keys() == pr.keys()
        rec = dict(mode=name, base=base, surrogate=sur, device=device, init_bitwise=init_same,
                   loss_base=lb.item(), loss_rb=lr.item(), loss_bitwise=torch.equal(lb, lr),
                   grad_bitwise=all(torch.equal(gb[k], gr[k]) for k in gb),
                   grad_maxdiff_vs_base=maxdiff(gb, gr), grad_maxdiff_rb_repeat=maxdiff(gr, gr2))
        if device == "cpu":
            rec["pass"] = rec["init_bitwise"] and rec["loss_bitwise"] and rec["grad_bitwise"]
        else:
            # forward bitwise; gradients per tensor within max(1e-4 max|g|, 3 x same-root repeat noise)
            g_ok, worst = gpu_tensor_check(gr, gb, reps)
            rec["grad_worst_ratio_to_allowed"] = worst
            rec["grad_rule"] = f"max|d| <= max({GPU_REL_TOL} max|g|, {GPU_NOISE_FACTOR} x noise of {GPU_REPEATS} repeats)"
            rec["pass"] = rec["init_bitwise"] and rec["loss_bitwise"] and g_ok
        out.append(rec)
        print(f"{'PASS' if rec['pass'] else 'FAIL'} A/{device} {name:40s} vs {base:12s} init={init_same} "
              f"loss_bit={rec['loss_bitwise']} grad_bit={rec['grad_bitwise']} "
              f"dgrad={rec['grad_maxdiff_vs_base']:.2e} repeat={rec['grad_maxdiff_rb_repeat']:.2e}"
              + (f" worst diff/allowed={rec['grad_worst_ratio_to_allowed']:.2f}" if device == "cuda" else ""), flush=True)
    return out


TRAIN_PAIRS = [  # (name, base root key, base script, args)
    ("exact_fp32_softmax", "r3", "train.py", ["--attn", "exact_fp32_softmax"]),
    ("bf16exp", "r3", "train.py", ["--attn", "softmax_bf16exp"]),
    ("fp8e4m3exp", "r3", "train.py", ["--attn", "softmax_fp8e4m3exp"]),
    ("minmax_lerp_k4_full", "r3", "train.py", ["--attn", "quant", "--approx", "lerp", "--K", "4", "--range_grad", "full"]),
    ("minmax_nearest_k4_prob_strict", "e2_minmax", "train_formal.py",
     ["--attn", "quant", "--approx", "nearest", "--K", "4", "--range_grad", "full", "--nearest_surrogate", "prob"]),
    ("minmax_nearest_k4_weight_strict", "e2_minmax", "train_formal.py",
     ["--attn", "quant", "--approx", "nearest", "--K", "4", "--range_grad", "full", "--nearest_surrogate", "weight_strict"]),
    ("fwm_nearest_k4_weight_strict", "e2_fwm", "train_formal.py",
     ["--attn", "quant", "--approx", "nearest", "--K", "4", "--range_grad", "full", "--range_mode", "relevant_zero",
      "--range_tau", "6", "--nearest_surrogate", "weight_strict"]),
    ("fwm_nearest_k4_weight_strict_detach", "e2_fwm", "train_formal.py",
     ["--attn", "quant", "--approx", "nearest", "--K", "4", "--range_grad", "detach", "--range_mode", "relevant_zero",
      "--range_tau", "6", "--nearest_surrogate", "weight_strict"]),
    ("fwm_lerp_k4_full", "e2_fwm", "train_formal.py",
     ["--attn", "quant", "--approx", "lerp", "--K", "4", "--range_grad", "full", "--range_mode", "relevant_zero",
      "--range_tau", "6"]),
]
COMMON = ["--n_layer", "2", "--n_head", "2", "--n_embd", "64", "--block_size", "64", "--batch_tokens", "256",
          "--micro_batch", "1", "--max_tokens", "768", "--warmup_tokens", "256", "--eval_every_tokens", "256",
          "--eval_batches", "2", "--ckpt_every_tokens", "768", "--seed", "7", "--device", "cpu"]


def part_b(py, tmp):
    rng = np.random.default_rng(0)
    data = os.path.join(tmp, "data")
    os.makedirs(data, exist_ok=True)
    rng.integers(0, 50000, 200_000, dtype=np.uint16).tofile(os.path.join(data, "train.bin"))
    rng.integers(0, 50000, 20_000, dtype=np.uint16).tofile(os.path.join(data, "val.bin"))
    out = []
    for name, base, script, args in TRAIN_PAIRS:
        res = {}
        for tag, root, scr in (("base", ROOTS[base], script), ("rb", ROOTS["rb"], "train_formal.py")):
            od = os.path.join(tmp, f"runs_{tag}")
            cmd = [py, scr, "--run_name", name, "--data_dir", data, "--out_dir", od] + COMMON + args
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{tag} {name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
            rd = os.path.join(od, name)
            log = list(csv.DictReader(open(os.path.join(rd, "log.csv"))))
            ck = torch.load(os.path.join(rd, "ckpt.pt"), map_location="cpu", weights_only=False)
            res[tag] = (log, ck)
        lb, cb = res["base"]
        lr, cr = res["rb"]
        loss_same = [(a["train_loss"], a["val_loss"]) for a in lb] == [(a["train_loss"], a["val_loss"]) for a in lr]
        w_same = cb["model"].keys() == cr["model"].keys() and all(torch.equal(cb["model"][k], cr["model"][k]) for k in cb["model"])
        rng_same = cb.get("train_rng_state") == cr.get("train_rng_state")
        opt_same = all(torch.equal(cb["opt"]["state"][i][f], cr["opt"]["state"][i][f])
                       for i in cb["opt"]["state"] for f in cb["opt"]["state"][i])
        rec = dict(mode=name, base=base, base_script=script, steps=int(cr["step"]), logged_losses_identical=loss_same,
                   final_weights_bitwise=w_same, optimizer_state_bitwise=opt_same, data_rng_identical=rng_same,
                   log_rows=[(a["step"], a["train_loss"], a["val_loss"]) for a in lr])
        rec["pass"] = loss_same and w_same and opt_same and rng_same
        out.append(rec)
        print(f"{'PASS' if rec['pass'] else 'FAIL'} B/cpu {name:40s} vs {base}/{script}: losses={loss_same} "
              f"weights={w_same} opt={opt_same} rng={rng_same} rows={rec['log_rows']}", flush=True)
        shutil.rmtree(os.path.join(tmp, "runs_base")); shutil.rmtree(os.path.join(tmp, "runs_rb"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true")
    a = ap.parse_args()
    res = {"env": env_info(), "roots": ROOTS}
    res["part_a_cpu"] = part_a("cpu")
    if not a.skip_gpu and torch.cuda.is_available():
        res["part_a_gpu"] = part_a("cuda")
    with tempfile.TemporaryDirectory(dir=os.environ.get("RB_TMP")) as tmp:
        res["part_b_cpu_training"] = part_b(sys.executable, tmp)
    allp = all(r["pass"] for k in ("part_a_cpu", "part_a_gpu", "part_b_cpu_training") for r in res.get(k, []))
    res["all_pass"] = allp
    tag = torch.__version__.split("+")[0]
    print("written", write_json(f"13_3_root_equivalence_torch{tag}.json", res))
    print("ALL PASS" if allp else "SOME FAILED")
    sys.exit(0 if allp else 1)


if __name__ == "__main__":
    main()
