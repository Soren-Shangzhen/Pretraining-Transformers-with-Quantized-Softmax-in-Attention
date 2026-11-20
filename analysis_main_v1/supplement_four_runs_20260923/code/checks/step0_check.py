"""Step-0 (before any optimizer update) model-level checks on the real training path.

Training path reproduced from the frozen train.py: seed -> torch.manual_seed /
np.random.seed -> GPT(cfg) on CPU -> .to(cuda); TF32 on (matmul + cudnn); bf16
autocast; eager (no compile); grad_checkpointing off; micro_batch 1; 128
microbatches accumulated as (loss / 128).backward(); no GradScaler; no dropout.
The canonical initial state is built once, hashed and saved; every
implementation loads it and the RNG state is restored before every forward.

E2 : per calibration, forward identity of old Weight (frozen root), strict Weight
     reference (new root, verification switch), new strict Prob, old Prob
     (recorded separately): per-layer P, logits, loss, for every microbatch of
     the first optimizer step.  Auxiliary: accumulated parameter gradients of new
     Prob vs old Prob.
E1 : four-mode forward identity (new root) and old frozen full/detach forward;
     backward regression new full vs old full and new detach vs old detach
     (per-microbatch per-layer dL/dscores, accumulated parameter gradients),
     3 old + 3 new repeats when the old implementation is not bitwise
     self-reproducible.  Auxiliary: min_only/max_only gradients finite / non-None;
     grad_checkpointing on vs off for old detach (the seed-7 detach control used it).

Usage: python step0_check.py --part e2|e1|ckpt --out <json>
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sup_common import (CLOUD, SUP, LocalsCapture, compare, dump_json, env_record,  # noqa: E402
                        load_root, rng_restore, rng_snapshot, root_hashes, ROOTS,
                        sha256_file, state_sha, tensor_sha)
import os as _os
KI_REPO = _os.environ.get("KI_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../..")))
KI_CLOUD = _os.environ.get("KI_CLOUD_DISK", "<CLOUD_DISK>")  # runs/checkpoints/tokenized data are not part of this repository


SEED = 7
ACCUM = 131072 // 1024
DATA = f"{CLOUD}/data/wikitext103"
WORK = "<CLOUD_HOME>/sup_step0"
if os.environ.get("SUP_LOCAL") == "1":   # local RTX 3090 workstation (same files, same hashes)
    DATA = (KI_REPO + "/adaptive_k_interval_training_phaseB/"
            "files_adaptive_quant_phaseB/data/wikitext103")
    WORK = "<LOCAL_TMP>/sup_step0"


def training_precision():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.is_bf16_supported()
    return lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def first_step_batches(seed=SEED, n=ACCUM, block=1024):
    """Identical sampling to train.py BinReader(train.bin, 1024, 1, seed)."""
    data = np.memmap(f"{DATA}/train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        ix = rng.integers(0, len(data) - block - 1, 1)
        x = np.stack([data[i:i + block] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + block] for i in ix]).astype(np.int64)
        xs.append(torch.from_numpy(x))
        ys.append(torch.from_numpy(y))
    return xs, ys


def make_cfg(mm, attn, K=None, approx=None, range_grad="full", range_mode=None, tau=None):
    kw = dict(block_size=1024, n_layer=12, n_head=12, n_embd=768, attn_mode=attn)
    if attn == "quant":
        kw.update(attn_K=K, attn_approx=approx, attn_range_grad=range_grad)
        if range_mode is not None:
            kw.update(attn_range_mode=range_mode, attn_range_tau=tau)
    return mm.GPTConfig(**kw)


def canonical_init(record):
    """Build the seed-7 initial state through two different roots and check they agree."""
    os.makedirs(WORK, exist_ok=True)
    states = {}
    for key, cfgkw in (("old_phaseB_base", dict(attn="quant", K=32, approx="lerp")),
                       ("new_e1", dict(attn="quant", K=32, approx="lerp")),
                       ("old_phaseB_rr", dict(attn="quant", K=4, approx="nearest", range_mode="relevant_zero", tau=6.0)),
                       ("old_mm_prob", dict(attn="quant", K=4, approx="nearest"))):
        _, mm, _ = load_root(key)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = mm.GPT(make_cfg(mm, **cfgkw))
        states[key] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    hashes = {k: state_sha(v) for k, v in states.items()}
    ref = states["old_phaseB_base"]
    path = f"{WORK}/canonical_init_s{SEED}.pt"
    torch.save(ref, path)
    record["canonical_init"] = {"hash_by_root": hashes, "all_equal": len(set(hashes.values())) == 1,
                                "path": path, "file_sha256": sha256_file(path),
                                "note": ("no historical step-0 snapshot exists (first checkpoint is step 153); "
                                         "the state is rebuilt with the frozen code and seed 7 on CPU exactly as "
                                         "train.py does, which cannot prove every historical environment state")}
    return ref


def build(key, cfgkw, init_sd, surrogate=None, tag=None):
    qa, mm, _ = load_root(key, tag=tag, surrogate=surrogate)
    model = mm.GPT(make_cfg(mm, **cfgkw))
    model.load_state_dict(init_sd)
    model.to("cuda").train()
    model.grad_checkpointing = False
    return model


class Probe:
    """Collect per-layer P (and optionally att for score-gradient hooks)."""

    def __init__(self, want_P=True, want_sgrad=False):
        self.P, self.sgrad, self.want_P, self.want_sgrad = [], [], want_P, want_sgrad

    def __call__(self, fn, d):
        if self.want_P and d.get("P") is not None:
            self.P.append(d["P"].detach().clone())
        if self.want_sgrad and d.get("att") is not None and d["att"].requires_grad:
            store = self.sgrad
            d["att"].register_hook(lambda g, store=store: store.append(g.detach().clone()) and None)


def fwd(model, x, y, amp, rng, probe=None, backward=False):
    rng_restore(rng)
    ctx = LocalsCapture(["_quant_attention"], ["P", "att"], probe) if probe else None
    if ctx:
        ctx.__enter__()
    try:
        with amp():
            logits, loss = model(x, y)
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
    if backward:
        (loss / ACCUM).backward()
    return logits.detach(), loss.detach()


def grads_of(model):
    return {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in model.named_parameters()}


def cmp_grads(ga, gb):
    out = {"keys_equal": sorted(ga) == sorted(gb)}
    per = {}
    for n in ga:
        per[n] = compare(ga[n], gb.get(n))
    out["all_eq_rawbits"] = all(v.get("eq_rawbits") for v in per.values())
    out["n_params"] = len(per)
    out["n_mismatching_params"] = sum(1 for v in per.values() if not v.get("eq_rawbits"))
    worst = max(per.items(), key=lambda kv: kv[1].get("max_abs", 0.0))
    out["worst_param"] = worst[0]
    out["worst"] = worst[1]
    out["max_rel_l2"] = max(v.get("rel_l2", 0.0) for v in per.values())
    out["per_param"] = {n: {k: v.get(k) for k in ("eq_rawbits", "mismatch_count", "max_abs", "rel_l2")}
                        for n, v in per.items()}
    return out


# ----------------------------------------------------------------------------- E2

def part_e2(rec, init_sd, xs, ys, amp):
    rng0 = rng_snapshot()
    out = {}
    for calib, extra, old_w, old_p, new in (
            ("minmax", {}, "old_phaseB_base", "old_mm_prob", "new_mm_prob"),
            ("fwm", dict(range_mode="relevant_zero", tau=6.0), "old_phaseB_rr", "old_fw_prob", "new_fw_prob")):
        cfgkw = dict(attn="quant", K=4, approx="nearest", **extra)
        impls = {
            "new_prob": build(new, cfgkw, init_sd, surrogate="prob", tag=f"{new}__prob"),
            "strict_weight_ref": build(new, cfgkw, init_sd, surrogate="weight_strict", tag=f"{new}__wstrict"),
            "old_weight": build(old_w, cfgkw, init_sd),
            "old_prob": build(old_p, cfgkw, init_sd),
        }
        agg = {k: {"P_all_eq": True, "P_mism_total": 0, "P_max_abs": 0.0, "logits_all_eq": True,
                   "logits_max_abs": 0.0, "loss_all_eq": True, "loss_max_abs": 0.0}
               for k in ("strict_weight_ref", "old_weight", "old_prob", "new_prob_repeat")}
        loss_acc = {k: 0.0 for k in impls}
        first = {}
        for mb in range(ACCUM):
            x, y = xs[mb].cuda(), ys[mb].cuda()
            res = {}
            for name, model in impls.items():
                pr = Probe()
                lg, ls = fwd(model, x, y, amp, rng0, pr, backward=(name in ("new_prob", "old_prob")))
                res[name] = (pr.P, lg, ls)
                loss_acc[name] += ls.item() / ACCUM
            pr = Probe()
            lg, ls = fwd(impls["new_prob"], x, y, amp, rng0, pr, backward=False)
            res["new_prob_repeat"] = (pr.P, lg, ls)
            ref = res["new_prob"]
            for other in agg:
                Pc = [compare(a, b) for a, b in zip(res[other][0], ref[0])]
                lc, sc = compare(res[other][1], ref[1]), compare(res[other][2], ref[2])
                a = agg[other]
                a["P_all_eq"] &= all(c["eq_rawbits"] for c in Pc) and len(Pc) == 12
                a["P_mism_total"] += sum(c["mismatch_count"] for c in Pc)
                a["P_max_abs"] = max(a["P_max_abs"], max(c["max_abs"] for c in Pc))
                a["logits_all_eq"] &= lc["eq_rawbits"]
                a["logits_max_abs"] = max(a["logits_max_abs"], lc["max_abs"])
                a["loss_all_eq"] &= sc["eq_rawbits"]
                a["loss_max_abs"] = max(a["loss_max_abs"], sc["max_abs"])
                if mb == 0:
                    first[other] = {"P_per_layer": Pc, "logits": lc, "loss": sc}
            if mb == 0:
                first["hashes"] = {"new_prob_logits_sha256": tensor_sha(ref[1]), "new_prob_loss": float(ref[2]),
                                   "new_prob_P_layer0_sha256": tensor_sha(ref[0][0])}
            del res
        g_new, g_old = grads_of(impls["new_prob"]), grads_of(impls["old_prob"])
        out[calib] = {"comparisons_vs_new_prob_all_microbatches": agg, "first_microbatch": first,
                      "loss_acc_train_py_style": loss_acc,
                      "aux_param_grads_new_prob_vs_old_prob": {k: v for k, v in cmp_grads(g_new, g_old).items()
                                                               if k != "per_param"}}
        print("E2", calib, {k: (v["P_all_eq"], v["logits_all_eq"], v["loss_all_eq"]) for k, v in agg.items()},
              flush=True)
        del impls
        torch.cuda.empty_cache()
    rec["e2_forward_identity"] = out


# ----------------------------------------------------------------------------- E1

def full_step(models, xs, ys, amp, rng0, sgrad=True, want_P=False):
    """Run all 128 microbatches for every model in `models` (dict), interleaved per
    microbatch; return per-layer score-gradient comparisons against model pairs and
    per-model loss_acc."""
    names = list(models)
    per_layer = {}
    loss_acc = {n: 0.0 for n in names}
    fwd_eq = {n: True for n in names}
    for mb in range(ACCUM):
        x, y = xs[mb].cuda(), ys[mb].cuda()
        cap = {}
        for n in names:
            pr = Probe(want_P=want_P, want_sgrad=sgrad)
            lg, ls = fwd(models[n], x, y, amp, rng0, pr, backward=True)
            loss_acc[n] += ls.item() / ACCUM
            # dL/ds hooks fire in reverse layer order
            cap[n] = (pr.P, lg, ls, list(reversed(pr.sgrad)))
        yield mb, cap, loss_acc


def summarize_sgrad(pairs, cap, store):
    for a, b in pairs:
        key = f"{a}__vs__{b}"
        s = store.setdefault(key, {"layers": [dict(max_abs=0.0, max_rel_l2=0.0, mismatch=0, eq=True) for _ in range(12)],
                                   "finite": True})
        for li, (ga, gb) in enumerate(zip(cap[a][3], cap[b][3])):
            c = compare(ga, gb)
            L = s["layers"][li]
            L["max_abs"] = max(L["max_abs"], c["max_abs"])
            L["max_rel_l2"] = max(L["max_rel_l2"], c["rel_l2"])
            L["mismatch"] += c["mismatch_count"]
            L["eq"] &= c["eq_rawbits"]
            s["finite"] &= c["finite_a"] and c["finite_b"]


def part_e1(rec, init_sd, xs, ys, amp):
    rng0 = rng_snapshot()
    cfg = lambda mode: dict(attn="quant", K=32, approx="lerp", range_grad=mode)  # noqa: E731
    out = {"forward_identity": {}, "backward_regression": {}}

    # ---- forward identity of the four modes + old frozen full/detach, first-step microbatches
    models = {f"new_{m}": build("new_e1", cfg(m), init_sd) for m in ("full", "detach", "min_only", "max_only")}
    models.update({f"old_{m}": build("old_phaseB_base", cfg(m), init_sd) for m in ("full", "detach")})
    fi = {n: {"P_all_eq": True, "logits_all_eq": True, "loss_all_eq": True} for n in models if n != "new_full"}
    sg = {}
    grads_finite = {}
    for mb, cap, loss_acc in full_step(models, xs, ys, amp, rng0, sgrad=True, want_P=True):
        ref = cap["new_full"]
        for n in fi:
            fi[n]["P_all_eq"] &= all(compare(a, b)["eq_rawbits"] for a, b in zip(cap[n][0], ref[0]))
            fi[n]["logits_all_eq"] &= compare(cap[n][1], ref[1])["eq_rawbits"]
            fi[n]["loss_all_eq"] &= compare(cap[n][2], ref[2])["eq_rawbits"]
        summarize_sgrad([("new_full", "old_full"), ("new_detach", "old_detach"),
                         ("new_min_only", "new_full"), ("new_max_only", "new_full"),
                         ("new_min_only", "new_detach"), ("new_max_only", "new_detach")], cap, sg)
        del cap
    for n, m in models.items():
        g = grads_of(m)
        grads_finite[n] = {"none_params": [k for k, v in g.items() if v is None],
                           "all_finite": all(v is not None and bool(torch.isfinite(v).all()) for v in g.values())}
    out["forward_identity"] = {"vs_new_full_all_128_microbatches": fi, "loss_acc": loss_acc}
    out["single_pass_score_grad"] = sg
    out["param_grads_single_pass"] = {
        "new_full_vs_old_full": {k: v for k, v in cmp_grads(grads_of(models["new_full"]), grads_of(models["old_full"])).items() if k != "per_param"},
        "new_detach_vs_old_detach": {k: v for k, v in cmp_grads(grads_of(models["new_detach"]), grads_of(models["old_detach"])).items() if k != "per_param"},
        "new_min_only_vs_new_full": {k: v for k, v in cmp_grads(grads_of(models["new_min_only"]), grads_of(models["new_full"])).items() if k != "per_param"},
        "new_max_only_vs_new_full": {k: v for k, v in cmp_grads(grads_of(models["new_max_only"]), grads_of(models["new_full"])).items() if k != "per_param"},
        "finite_and_not_none": grads_finite,
    }
    print("E1 forward identity:", fi, flush=True)
    del models
    torch.cuda.empty_cache()

    # ---- backward regression with repeats (3 old + 3 new per mode)
    for mode in ("full", "detach"):
        models = {}
        for r in range(3):
            models[f"old{r}"] = build("old_phaseB_base", cfg(mode), init_sd)
            models[f"new{r}"] = build("new_e1", cfg(mode), init_sd)
        sg = {}
        pairs = [("old0", "old1"), ("old0", "old2"), ("old1", "old2"),
                 ("new0", "old0"), ("new1", "old1"), ("new2", "old2")]
        for mb, cap, loss_acc in full_step(models, xs, ys, amp, rng0, sgrad=True):
            summarize_sgrad(pairs, cap, sg)
            del cap
        G = {n: grads_of(m) for n, m in models.items()}
        pg = {f"{a}__vs__{b}": cmp_grads(G[a], G[b]) for a, b in pairs}
        oo = [p for p in pg if p.startswith("old") and "__vs__old" in p and p.split("__vs__")[0].startswith("old")]
        on = [p for p in pg if p.startswith("new")]
        old_bitwise = all(pg[p]["all_eq_rawbits"] for p in oo) and all(
            all(L["eq"] for L in sg[p]["layers"]) for p in oo)
        new_bitwise = all(pg[p]["all_eq_rawbits"] for p in on) and all(
            all(L["eq"] for L in sg[p]["layers"]) for p in on)
        # descriptive reference scale: max over old-old pairs, per parameter / per layer
        status_detail = {}
        exceed = []
        for n in G["old0"]:
            ref_abs = max(pg[p]["per_param"][n]["max_abs"] for p in oo)
            ref_rel = max(pg[p]["per_param"][n]["rel_l2"] for p in oo)
            on_abs = max(pg[p]["per_param"][n]["max_abs"] for p in on)
            on_rel = max(pg[p]["per_param"][n]["rel_l2"] for p in on)
            if on_abs > ref_abs or on_rel > ref_rel:
                exceed.append({"param": n, "old_old_max_abs": ref_abs, "old_new_max_abs": on_abs,
                               "old_old_rel_l2": ref_rel, "old_new_rel_l2": on_rel})
        for li in range(12):
            ref_abs = max(sg[p]["layers"][li]["max_abs"] for p in oo)
            on_abs = max(sg[p]["layers"][li]["max_abs"] for p in on)
            ref_rel = max(sg[p]["layers"][li]["max_rel_l2"] for p in oo)
            on_rel = max(sg[p]["layers"][li]["max_rel_l2"] for p in on)
            status_detail[f"layer{li}"] = {"old_old_max_abs": ref_abs, "old_new_max_abs": on_abs,
                                           "old_old_max_rel_l2": ref_rel, "old_new_max_rel_l2": on_rel}
            if on_abs > ref_abs or on_rel > ref_rel:
                exceed.append({"score_grad_layer": li, **status_detail[f"layer{li}"]})
        finite = all(sg[p]["finite"] for p in sg) and all(
            v is not None and bool(torch.isfinite(v).all()) for g in G.values() for v in g.values())
        if old_bitwise:
            status = "pass_bitwise" if new_bitwise else "fail_old_bitwise_new_differs"
        elif not exceed:
            status = "pass_with_observed_nondeterminism"
        else:
            status = "inconclusive_gpu_nondeterminism"
        out["backward_regression"][mode] = {
            "status": status, "old_self_bitwise": old_bitwise, "old_new_bitwise": new_bitwise,
            "finite": finite, "n_exceeding_old_old_scale": len(exceed), "exceeding": exceed[:40],
            "score_grad_layers": status_detail,
            "param_grad_pairs": {p: {k: v for k, v in pg[p].items() if k != "per_param"} for p in pg},
            "loss_acc": loss_acc,
        }
        print("E1 backward", mode, status, "exceed:", len(exceed), flush=True)
        del models, G
        torch.cuda.empty_cache()

    # ---- grad_checkpointing on vs off for the old detach implementation (the seed-7 detach
    #      control was trained with --grad_checkpointing; the full control without)
    models = {"ckpt_off": build("old_phaseB_base", cfg("detach"), init_sd),
              "ckpt_on": build("old_phaseB_base", cfg("detach"), init_sd),
              "ckpt_off_b": build("old_phaseB_base", cfg("detach"), init_sd)}
    models["ckpt_on"].grad_checkpointing = True
    la = None
    for mb, cap, loss_acc in full_step(models, xs, ys, amp, rng0, sgrad=False):
        la = loss_acc
        del cap
    G = {n: grads_of(m) for n, m in models.items()}
    out["grad_checkpointing_old_detach"] = {
        "off_vs_off": {k: v for k, v in cmp_grads(G["ckpt_off"], G["ckpt_off_b"]).items() if k != "per_param"},
        "on_vs_off": {k: v for k, v in cmp_grads(G["ckpt_on"], G["ckpt_off"]).items() if k != "per_param"},
        "loss_acc": la}
    del models, G
    torch.cuda.empty_cache()
    rec["e1"] = out


def part_e2w(rec, init_sd, xs, ys, amp):
    """Step-0 backward of the trained E2 surrogates against the historical ones (same
    initial forward at step 0): strict Weight vs old Weight, new Prob vs old Prob.
    Per-microbatch per-layer dL/dscores and accumulated parameter gradients."""
    rng0 = rng_snapshot()
    out = {}
    for calib, extra, old_w, old_p, new in (
            ("minmax", {}, "old_phaseB_base", "old_mm_prob", "new_mm_prob"),
            ("fwm", dict(range_mode="relevant_zero", tau=6.0), "old_phaseB_rr", "old_fw_prob", "new_fw_prob")):
        cfgkw = dict(attn="quant", K=4, approx="nearest", **extra)
        models = {
            "strict_weight": build(new, cfgkw, init_sd, surrogate="weight_strict", tag=f"{new}__wstrict"),
            "old_weight": build(old_w, cfgkw, init_sd),
            "new_prob": build(new, cfgkw, init_sd, surrogate="prob", tag=f"{new}__prob"),
            "old_prob": build(old_p, cfgkw, init_sd),
            "strict_weight_repeat": build(new, cfgkw, init_sd, surrogate="weight_strict", tag=f"{new}__wstrict"),
        }
        sg = {}
        pairs = [("strict_weight", "old_weight"), ("new_prob", "old_prob"),
                 ("strict_weight", "strict_weight_repeat"), ("new_prob", "strict_weight")]
        la = None
        for mb, cap, loss_acc in full_step(models, xs, ys, amp, rng0, sgrad=True):
            summarize_sgrad(pairs, cap, sg)
            la = loss_acc
            del cap
        G = {n: grads_of(m) for n, m in models.items()}
        out[calib] = {
            "loss_acc": la,
            "score_grad": {p: {"all_layers_bitwise": all(L["eq"] for L in v["layers"]),
                               "max_rel_l2": max(L["max_rel_l2"] for L in v["layers"]), "finite": v["finite"]}
                           for p, v in sg.items()},
            "param_grads": {f"{a}__vs__{b}": {k: v for k, v in cmp_grads(G[a], G[b]).items() if k != "per_param"}
                            for a, b in pairs},
        }
        print("E2w", calib, {p: v["all_layers_bitwise"] for p, v in out[calib]["score_grad"].items()},
              {p: v["all_eq_rawbits"] for p, v in out[calib]["param_grads"].items()}, flush=True)
        del models, G
        torch.cuda.empty_cache()
    rec["e2_step0_backward"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["e2", "e1", "e2w"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic", action="store_true",
                    help="diagnostic only (separate process): torch.use_deterministic_algorithms(True)")
    args = ap.parse_args()
    det_note = None
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        det_note = "torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8 (diagnostic only)"
    amp = training_precision()
    rec = {"script": os.path.abspath(__file__), "script_sha256": sha256_file(__file__),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "env": env_record(),
           "definition": "step 0 = the initial forward/backward before any optimizer update",
           "run_mode": {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                        "tf32_cudnn": torch.backends.cudnn.allow_tf32, "autocast": "bf16 (cuda)",
                        "compile": False, "grad_checkpointing": False, "micro_batch": 1,
                        "accum_microbatches": ACCUM, "loss_reduction": "mean CE, (loss/128).backward()",
                        "grad_scaler": None, "dropout": None,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "rng": "python/numpy/torch-cpu/torch-cuda states snapshotted once and restored before every forward"},
           "roots": {k: {"path": v, "sha256": root_hashes(v)} for k, v in ROOTS.items()},
           "diagnostic_determinism": det_note}
    init_sd = canonical_init(rec)
    xs, ys = first_step_batches()
    rec["first_step_batch"] = {"seed": SEED, "microbatches": ACCUM,
                               "x_sha256": tensor_sha(torch.cat(xs)), "y_sha256": tensor_sha(torch.cat(ys)),
                               "train_bin_sha256": sha256_file(f"{DATA}/train.bin")}
    if args.part == "e2":
        part_e2(rec, init_sd, xs, ys, amp)
    elif args.part == "e2w":
        part_e2w(rec, init_sd, xs, ys, amp)
    else:
        part_e1(rec, init_sd, xs, ys, amp)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump_json(args.out, rec)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
